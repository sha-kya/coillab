"""Conductor resistance including skin effect.

Proximity effect between neighbouring turns is *not* modelled in v0.1; it can
raise AC resistance well above these values in densely wound coils at high
frequency.
"""

from __future__ import annotations

import numpy as np
from scipy.special import jv

from .constants import MU0


def skin_depth(freq, resistivity):
    """Skin depth in metres for a non-magnetic conductor."""
    freq = np.asarray(freq, dtype=float)
    with np.errstate(divide="ignore"):
        return np.sqrt(resistivity / (np.pi * freq * MU0))


def round_wire_ac_factor(freq, wire_radius, resistivity):
    """Exact R_ac / R_dc for an isolated round wire (Bessel-function solution).

    Uses ``Z / R_dc = (ka/2) J0(ka) / J1(ka)`` with ``k = (1 - j)/delta``, and
    the asymptotic form ``a/(2 delta) + 1/4`` once ``a/delta > 40``.
    """
    freq = np.atleast_1d(np.asarray(freq, dtype=float))
    out = np.ones_like(freq)
    pos = freq > 0
    x = wire_radius / skin_depth(freq[pos], resistivity)
    exact = x <= 40.0
    ka = (1.0 - 1.0j) * x[exact]
    vals = np.empty_like(x)
    vals[exact] = np.real(0.5 * ka * jv(0, ka) / jv(1, ka))
    vals[~exact] = x[~exact] / 2.0 + 0.25
    out[pos] = vals
    return out if out.size > 1 else float(out[0])


def trace_ac_factor(freq, width, thickness, resistivity):
    """Approximate R_ac / R_dc for a flat PCB trace.

    ``R_ac = rho l / (w delta (1 - exp(-t/delta)))`` (Yue and Wong), which tends
    to the DC value when ``delta >> t``.
    """
    freq = np.atleast_1d(np.asarray(freq, dtype=float))
    out = np.ones_like(freq)
    pos = freq > 0
    d = skin_depth(freq[pos], resistivity)
    out[pos] = thickness / (d * (1.0 - np.exp(-thickness / d)))
    return out if out.size > 1 else float(out[0])
