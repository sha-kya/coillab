"""Inductance of multi-turn windings.

Two exact building blocks are used, so no field solver is needed:

* **Circular turns** (coaxial loops): Maxwell's elliptic-integral formula for the
  mutual inductance between two coaxial circular filaments, plus the classical
  self-inductance of a thin circular loop of round wire.
* **Rectangular (Manhattan) turns**: the Greenhouse method. Every turn is split
  into straight, axis-aligned segments. Perpendicular segments have zero mutual
  inductance; parallel segments use the exact formula for two parallel
  filaments, and each segment's self-inductance uses the exact straight-wire
  (round) or Ruehli (rectangular bar) formula.

Total inductance is the sum of every entry of the turn (or segment) inductance
matrix. All lengths are in metres; results are in henries.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ellipe, ellipk

from .constants import MU0

# ---------------------------------------------------------------------------
# Circular loops
# ---------------------------------------------------------------------------


def loop_self_inductance(radius, wire_radius, internal: float = 0.25):
    """Self-inductance of a thin circular loop of round wire.

    ``L = mu0 R [ln(8R/a) - 2 + Y]`` with ``Y = 0.25`` for uniform (low-frequency)
    current and ``Y = 0`` when the current is confined to the surface (high
    frequency). Valid for ``a << R``.
    """
    radius = np.asarray(radius, dtype=float)
    return MU0 * radius * (np.log(8.0 * radius / wire_radius) - 2.0 + internal)


def mutual_coaxial_loops(r1, r2, dz):
    """Mutual inductance between two coaxial circular filaments (Maxwell).

    ``r1``, ``r2`` are the loop radii and ``dz`` the axial separation. Inputs
    broadcast. Coincident filaments (r1 == r2, dz == 0) return ``inf``.
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    dz = np.asarray(dz, dtype=float)
    m = 4.0 * r1 * r2 / ((r1 + r2) ** 2 + dz**2)  # m = k^2
    m = np.clip(m, 0.0, 1.0 - 1e-16)
    k = np.sqrt(m)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = MU0 * np.sqrt(r1 * r2) * ((2.0 / k - k) * ellipk(m) - (2.0 / k) * ellipe(m))
    return np.where(k > 0, out, 0.0)


def circular_winding_inductance(radii, zs, wire_radius, internal: float = 0.25) -> float:
    """Total inductance of series-connected coaxial circular turns.

    ``radii`` and ``zs`` give each turn's centreline radius and axial position.
    """
    radii = np.asarray(radii, dtype=float)
    zs = np.asarray(zs, dtype=float)
    r1, r2 = np.meshgrid(radii, radii, indexing="ij")
    z1, z2 = np.meshgrid(zs, zs, indexing="ij")
    mat = mutual_coaxial_loops(r1, r2, z1 - z2)
    np.fill_diagonal(mat, loop_self_inductance(radii, wire_radius, internal))
    return float(mat.sum())


# ---------------------------------------------------------------------------
# Straight segments (Greenhouse method)
# ---------------------------------------------------------------------------


def segment_self_round(length, wire_radius, internal: float = 0.25):
    """Exact self-inductance of a straight round wire of finite length.

    Partial inductance (Rosa/Grover) with internal term ``Y``.
    """
    length = np.asarray(length, dtype=float)
    q = length / wire_radius
    return (MU0 * length / (2.0 * np.pi)) * (
        np.arcsinh(q) - np.sqrt(1.0 + 1.0 / q**2) + 1.0 / q + internal
    )


def segment_self_rect(length, width, thickness):
    """Self-inductance of a straight rectangular bar (Ruehli / Greenhouse)."""
    length = np.asarray(length, dtype=float)
    wt = width + thickness
    return (MU0 * length / (2.0 * np.pi)) * (
        np.log(2.0 * length / wt) + 0.50049 + wt / (3.0 * length)
    )


def _f(u, d):
    return u * np.arcsinh(u / d) - np.sqrt(u**2 + d**2)


def parallel_filament_mutual(l1, l2, offset, distance):
    """Mutual inductance of two parallel straight filaments (same direction).

    Filament 1 spans ``[0, l1]`` and filament 2 spans ``[offset, offset + l2]``
    along their common direction; ``distance`` is their perpendicular
    separation. Inputs broadcast.
    """
    d = np.maximum(np.asarray(distance, dtype=float), 1e-9)
    s = np.asarray(offset, dtype=float)
    return (MU0 / (4.0 * np.pi)) * (
        _f(s + l2, d) - _f(s + l2 - l1, d) - _f(s, d) + _f(s - l1, d)
    )


def square_turn_segments(side: float, z: float, x0: float = 0.0, y0: float = 0.0):
    """Four counter-clockwise segments of a square turn centred at (x0, y0, z).

    Returns a list of ``(axis, start_along, length, sign, perp1, perp2)`` where
    ``axis`` is 0 (x) or 1 (y), ``start_along`` the lower coordinate along that
    axis, ``sign`` the current direction (+1/-1) and ``perp1, perp2`` the two
    transverse coordinates (the other in-plane coordinate, then z).
    """
    h = side / 2.0
    return [
        (0, x0 - h, side, +1.0, y0 - h, z),  # bottom, +x
        (1, y0 - h, side, +1.0, x0 + h, z),  # right, +y
        (0, x0 - h, side, -1.0, y0 + h, z),  # top, -x
        (1, y0 - h, side, -1.0, x0 - h, z),  # left, -y
    ]


def segments_inductance(segments, self_fn) -> float:
    """Total inductance of a set of axis-aligned segments in series.

    ``self_fn(length)`` returns the self-inductance of one segment.
    """
    seg = np.array(segments, dtype=float)
    total = 0.0
    for axis in (0.0, 1.0, 2.0):
        sel = seg[seg[:, 0] == axis]
        if len(sel) == 0:
            continue
        a, ln, sg, p1, p2 = sel[:, 1], sel[:, 2], sel[:, 3], sel[:, 4], sel[:, 5]
        A1, A2 = np.meshgrid(a, a, indexing="ij")
        L1, L2 = np.meshgrid(ln, ln, indexing="ij")
        S1, S2 = np.meshgrid(sg, sg, indexing="ij")
        dist = np.hypot(p1[:, None] - p1[None, :], p2[:, None] - p2[None, :])
        mat = S1 * S2 * parallel_filament_mutual(L1, L2, A2 - A1, dist)
        np.fill_diagonal(mat, self_fn(ln))
        total += mat.sum()
    return float(total)
