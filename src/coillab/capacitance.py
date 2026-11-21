"""Self-capacitance of windings by the energy method.

Each turn ``i`` is given a potential ``v_i`` (as a fraction of the terminal
voltage) equal to the conductor-length fraction at the turn's midpoint, i.e. a
uniform voltage drop along the winding, valid well below self-resonance. Every
pair of neighbouring turns contributes a capacitance ``C_ij``, and the stored
energy fixes the equivalent terminal capacitance:

    C_eq = sum_ij C_ij (v_i - v_j)^2

Only nearest neighbours are included (adjacent turns in a layer, and facing
turns in adjacent layers). This is a first-order model; for accurate SRF
predictions calibrate against one measured SRF with ``capacitance_scale``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ellipk

from .constants import EPS0


def parallel_wires_c_per_length(pitch: float, wire_radius: float, eps_r: float = 1.0) -> float:
    """Capacitance per metre between two parallel round wires.

    ``C' = pi eps / acosh(p / 2a)`` with centre spacing ``p`` and radius ``a``.
    """
    if pitch <= 2.0 * wire_radius:
        raise ValueError("wire pitch must exceed the conductor diameter")
    return np.pi * EPS0 * eps_r / np.arccosh(pitch / (2.0 * wire_radius))


def coplanar_strips_c_per_length(width: float, gap: float, eps_eff: float = 1.0) -> float:
    """Capacitance per metre between two coplanar strips (conformal mapping).

    Two strips of width ``w`` separated by ``gap`` on a substrate with
    effective permittivity ``eps_eff``: ``C' = eps0 eps_eff K(k') / K(k)``
    with ``k = s / (s + 2w)``.
    """
    k = gap / (gap + 2.0 * width)
    kp = np.sqrt(1.0 - k**2)
    return EPS0 * eps_eff * ellipk(kp**2) / ellipk(k**2)


def parallel_plate(area: float, separation: float, eps_r: float = 1.0) -> float:
    return EPS0 * eps_r * area / separation


def voltage_fractions(turn_lengths) -> np.ndarray:
    """Potential of each turn's midpoint as a fraction of terminal voltage."""
    lengths = np.asarray(turn_lengths, dtype=float)
    cum = np.cumsum(lengths) - 0.5 * lengths
    return cum / lengths.sum()


def energy_method(pairs, potentials) -> float:
    """Equivalent capacitance from neighbour pairs ``(i, j, C_ij)``."""
    v = np.asarray(potentials, dtype=float)
    return float(sum(c * (v[i] - v[j]) ** 2 for i, j, c in pairs))
