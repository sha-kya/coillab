"""Independent closed-form references used to validate CoilLab.

These come from the public literature and are deliberately *different*
methods from the ones in :mod:`coillab.inductance`, so agreement is a real
check rather than a tautology.

* Wheeler (1928), single-layer and multilayer air-core coils.
* Grover, *Inductance Calculations*: square loop of round wire.
* Mohan, del Mar Hershenson, Boyd and Lee (1999), "Simple accurate
  expressions for planar spiral inductances", IEEE JSSC 34(10):
  modified-Wheeler and current-sheet expressions.
* Brute-force Neumann double integral over discretised filaments.
"""

from __future__ import annotations

import numpy as np

from .constants import MU0

INCH = 0.0254


def wheeler_single_layer(radius: float, length: float, n_turns: int) -> float:
    """Wheeler's single-layer solenoid formula, ~1% for length > 0.8 radius.

    ``L[uH] = r^2 N^2 / (9 r + 10 l)`` with ``r``, ``l`` in inches.
    """
    r, ln = radius / INCH, length / INCH
    return 1e-6 * r**2 * n_turns**2 / (9.0 * r + 10.0 * ln)


def wheeler_multilayer(mean_radius: float, length: float, depth: float, n_turns: int) -> float:
    """Wheeler's multilayer formula, ~1% when the three denominator terms are similar.

    ``L[uH] = 0.8 r^2 N^2 / (6 r + 9 l + 10 c)`` with mean radius ``r``,
    winding length ``l`` and radial depth ``c``, all in inches.
    """
    r, ln, c = mean_radius / INCH, length / INCH, depth / INCH
    return 1e-6 * 0.8 * r**2 * n_turns**2 / (6.0 * r + 9.0 * ln + 10.0 * c)


def grover_square_loop(side: float, wire_radius: float, internal: float = 0.25) -> float:
    """Grover's square loop of round wire (side ``s``, wire radius ``a``).

    ``L = (2 mu0 s / pi) [ln(s/a) + a/s - 0.774 + Y]``.
    """
    s, a = side, wire_radius
    return (2.0 * MU0 * s / np.pi) * (np.log(s / a) + a / s - 0.774 + internal)


def mohan_modified_wheeler(n_turns: float, d_out: float, d_in: float) -> float:
    """Mohan et al. modified-Wheeler expression for a square planar spiral."""
    k1, k2 = 2.34, 2.75
    d_avg = 0.5 * (d_out + d_in)
    rho = (d_out - d_in) / (d_out + d_in)
    return k1 * MU0 * n_turns**2 * d_avg / (1.0 + k2 * rho)


def mohan_current_sheet(n_turns: float, d_out: float, d_in: float) -> float:
    """Mohan et al. current-sheet expression for a square planar spiral."""
    c1, c2, c3, c4 = 1.27, 2.07, 0.18, 0.13
    d_avg = 0.5 * (d_out + d_in)
    rho = (d_out - d_in) / (d_out + d_in)
    return 0.5 * MU0 * n_turns**2 * d_avg * c1 * (np.log(c2 / rho) + c3 * rho + c4 * rho**2)


def neumann_mutual(path1: np.ndarray, path2: np.ndarray) -> float:
    """Brute-force Neumann integral between two closed polylines (N x 3 arrays).

    ``M = mu0/(4 pi) sum_i sum_j dl_i . dl_j / |r_i - r_j|`` using segment
    midpoints. Accurate when segments are short compared with the separation.
    """
    p1 = np.vstack([path1, path1[:1]])
    p2 = np.vstack([path2, path2[:1]])
    dl1 = np.diff(p1, axis=0)
    dl2 = np.diff(p2, axis=0)
    m1 = 0.5 * (p1[1:] + p1[:-1])
    m2 = 0.5 * (p2[1:] + p2[:-1])
    dist = np.linalg.norm(m1[:, None, :] - m2[None, :, :], axis=-1)
    dots = dl1 @ dl2.T
    return float(MU0 / (4.0 * np.pi) * np.sum(dots / dist))


def circle_path(radius: float, z: float = 0.0, n: int = 2000) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t), np.full(n, z)])
