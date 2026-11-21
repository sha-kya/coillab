"""Coil geometries and their equivalent-circuit analysis.

Three coil families are supported in v0.1:

* :class:`CircularCoil` -- round wire wound on a circular former, as a
  single-layer solenoid or a multilayer bundle.
* :class:`SquareCoil` -- round wire wound on a square frame (a square loop
  antenna), single- or multilayer.
* :class:`PCBSquareSpiral` -- a square planar spiral on one or more copper
  layers, connected in series.

Each coil produces a list of turns in winding order. From those, the coil
computes its DC/AC resistance, inductance, self-capacitance, summed turn
area, self-resonant frequency and a simple field-equivalent noise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from . import capacitance as cap
from . import inductance as ind
from . import resistance as res
from .constants import CU_DENSITY, K_BOLTZMANN, OZ_COPPER_THICKNESS, copper_resistivity

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """One closed turn of the winding (centreline)."""

    size: float  # radius (circular) or side length (square), m
    z: float  # axial position of the turn's plane, m
    length: float  # conductor length of the turn, m
    area: float  # enclosed area, m^2
    layer: int
    index: int  # position within its layer


@dataclass
class Amplifier:
    """Simple preamplifier noise model, referred to its input.

    Defaults are typical of a good low-noise JFET input stage.
    """

    en: float = 1.0e-9  # input voltage noise density, V/sqrt(Hz)
    in_: float = 10.0e-15  # input current noise density, A/sqrt(Hz)
    c_in: float = 10.0e-12  # input capacitance, F


@dataclass
class CoilReport:
    """Results of :meth:`Coil.analyze` at one frequency."""

    coil: str
    freq: float  # Hz
    n_turns: int
    wire_length: float  # m
    copper_mass: float  # kg
    r_dc: float  # ohm
    r_ac: float  # ohm, at freq (skin effect only)
    inductance: float  # H
    c_self: float  # F
    c_total: float  # F, self + amplifier input + extra load
    srf: float  # Hz, with c_total
    area_sum: float  # m^2
    sensitivity: float  # V/T, open-circuit EMF per tesla at freq
    emf: float  # V, open-circuit EMF for `field_amplitude`
    field_amplitude: float  # T
    noise_emf: float  # V/sqrt(Hz), EMF-referred total noise
    noise_field: float  # T/sqrt(Hz), field-equivalent noise
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        rows = [
            ("Coil", self.coil),
            ("Frequency", eng(self.freq, "Hz")),
            ("Turns", str(self.n_turns)),
            ("Wire length", eng(self.wire_length, "m")),
            ("Copper mass", eng(self.copper_mass * 1e3, "g")),
            ("R (DC)", eng(self.r_dc, "Ohm")),
            ("R (AC, skin)", eng(self.r_ac, "Ohm")),
            ("Inductance", eng(self.inductance, "H")),
            ("Self-capacitance", eng(self.c_self, "F")),
            ("Total capacitance", eng(self.c_total, "F")),
            ("Self-resonance", eng(self.srf, "Hz")),
            ("Summed turn area", f"{self.area_sum:.4g} m^2"),
            ("Sensitivity", eng(self.sensitivity, "V/T")),
            (f"EMF @ {eng(self.field_amplitude, 'T')}", eng(self.emf, "V")),
            ("EMF noise", eng(self.noise_emf, "V/rtHz")),
            ("Field noise", eng(self.noise_field, "T/rtHz")),
        ]
        width = max(len(k) for k, _ in rows)
        lines = [f"{k:<{width}}  {v}" for k, v in rows]
        lines += [f"warning: {w}" for w in self.warnings]
        return "\n".join(lines)


_PREFIXES = [(1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "u"),
             (1e-9, "n"), (1e-12, "p"), (1e-15, "f"), (1e-18, "a")]


def eng(value: float, unit: str = "", digits: int = 4) -> str:
    """Format a number with an SI prefix, e.g. ``eng(2.2e-6, 'H') -> '2.2 uH'``."""
    if value == 0 or not np.isfinite(value):
        return f"{value} {unit}".strip()
    for scale, prefix in _PREFIXES:
        if abs(value) >= scale:
            return f"{value / scale:.{digits}g} {prefix}{unit}".strip()
    scale, prefix = _PREFIXES[-1]
    return f"{value / scale:.{digits}g} {prefix}{unit}".strip()


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Coil:
    """Base class. Subclasses implement the geometry-specific pieces."""

    name = "coil"

    #: multiply the modelled self-capacitance (calibrate against a measured SRF)
    capacitance_scale: float = 1.0

    # -- geometry hooks -------------------------------------------------
    def turns(self) -> List[Turn]:
        raise NotImplementedError

    def _capacitance_pairs(self, turns: List[Turn]) -> List[Tuple[int, int, float]]:
        raise NotImplementedError

    def _conductor_area(self) -> float:
        raise NotImplementedError

    def _ac_factor(self, freq, resistivity):
        raise NotImplementedError

    def inductance(self) -> float:
        raise NotImplementedError

    # -- shared quantities ----------------------------------------------
    def turn_count(self) -> int:
        return len(self.turns())

    def wire_length(self) -> float:
        return float(sum(t.length for t in self.turns()))

    def area_sum(self) -> float:
        """Summed area of all turns (A_sigma), m^2."""
        return float(sum(t.area for t in self.turns()))

    def copper_mass(self) -> float:
        return self.wire_length() * self._conductor_area() * CU_DENSITY

    def resistance_dc(self, temp_c: float = 20.0) -> float:
        return copper_resistivity(temp_c) * self.wire_length() / self._conductor_area()

    def resistance_ac(self, freq, temp_c: float = 20.0):
        rho = copper_resistivity(temp_c)
        return self.resistance_dc(temp_c) * self._ac_factor(freq, rho)

    def self_capacitance(self) -> float:
        t = self.turns()
        v = cap.voltage_fractions([x.length for x in t])
        return self.capacitance_scale * cap.energy_method(self._capacitance_pairs(t), v)

    def srf(self, c_extra: float = 0.0) -> float:
        """Self-resonant frequency with optional extra shunt capacitance."""
        c = self.self_capacitance() + c_extra
        return 1.0 / (2.0 * np.pi * np.sqrt(self.inductance() * c))

    # -- frequency-domain model ------------------------------------------
    def spectrum(
        self,
        freqs,
        amplifier: Optional[Amplifier] = None,
        c_load: float = 0.0,
        temp_c: float = 20.0,
    ) -> dict:
        """EMF-referred noise and field-equivalent noise versus frequency.

        Model: the induced EMF ``e`` and the coil's thermal noise sit in series
        with ``R_ac + jwL``; the node is shunted by ``C = C_self + C_in +
        c_load``. The amplifier's voltage noise adds at the node and its current
        noise flows into the node impedance. Everything is referred back to the
        EMF through ``H = 1 / (1 + (R + jwL) jwC)``. No damping network is
        included in v0.1, so the curves show the undamped resonance.
        """
        amp = amplifier or Amplifier()
        f = np.atleast_1d(np.asarray(freqs, dtype=float))
        w = 2.0 * np.pi * f
        r = np.atleast_1d(self.resistance_ac(f, temp_c))
        L = self.inductance()
        c = self.self_capacitance() + amp.c_in + c_load
        zs = r + 1j * w * L
        h = 1.0 / (1.0 + zs * 1j * w * c)
        z_node = zs * h  # (R + jwL) || 1/(jwC)
        temp_k = temp_c + 273.15
        thermal = 4.0 * K_BOLTZMANN * temp_k * r
        amp_node = amp.en**2 + (amp.in_ * np.abs(z_node)) ** 2
        n_e = np.sqrt(thermal + amp_node / np.abs(h) ** 2)
        a_sum = self.area_sum()
        n_b = n_e / (w * a_sum)
        return {
            "freq": f,
            "r_ac": r,
            "transfer": h,
            "noise_emf": n_e,
            "noise_field": n_b,
            "noise_thermal": np.sqrt(thermal),
        }

    def analyze(
        self,
        freq: float = 1e3,
        amplifier: Optional[Amplifier] = None,
        c_load: float = 0.0,
        field_amplitude: float = 1e-9,
        temp_c: float = 20.0,
    ) -> CoilReport:
        """Compute the full equivalent-circuit report at one frequency.

        ``field_amplitude`` is the peak of a uniform sinusoidal field normal to
        the turns; ``emf`` is the corresponding peak open-circuit EMF.
        """
        amp = amplifier or Amplifier()
        spec = self.spectrum(freq, amp, c_load, temp_c)
        c_self = self.self_capacitance()
        c_total = c_self + amp.c_in + c_load
        L = self.inductance()
        srf = 1.0 / (2.0 * np.pi * np.sqrt(L * c_total))
        a_sum = self.area_sum()
        sens = 2.0 * np.pi * freq * a_sum
        warnings = []
        if freq > 0.3 * srf:
            warnings.append(
                f"frequency is {freq / srf:.2f} x SRF; the open-circuit EMF model is "
                "not valid near resonance"
            )
        return CoilReport(
            coil=self.describe(),
            freq=float(freq),
            n_turns=self.turn_count(),
            wire_length=self.wire_length(),
            copper_mass=self.copper_mass(),
            r_dc=self.resistance_dc(temp_c),
            r_ac=float(spec["r_ac"][0]),
            inductance=L,
            c_self=c_self,
            c_total=c_total,
            srf=float(srf),
            area_sum=a_sum,
            sensitivity=sens,
            emf=sens * field_amplitude,
            field_amplitude=field_amplitude,
            noise_emf=float(spec["noise_emf"][0]),
            noise_field=float(spec["noise_field"][0]),
            warnings=warnings,
        )

    def describe(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Wound round-wire coils
# ---------------------------------------------------------------------------


@dataclass
class _WireWinding:
    n_turns: int
    wire_diameter: float  # bare copper diameter, m
    insulation: float = 20e-6  # insulation thickness per side, m
    turns_per_layer: Optional[int] = None  # None -> single layer
    eps_r: float = 2.0  # effective permittivity between turns (enamel + air)
    internal: float = 0.25  # internal-inductance term (0.25 low f, 0 high f)
    capacitance_scale: float = 1.0

    def __post_init__(self):
        if self.n_turns < 1:
            raise ValueError("n_turns must be at least 1")
        if self.wire_diameter <= 0:
            raise ValueError("wire_diameter must be positive")
        if self.turns_per_layer is None:
            self.turns_per_layer = self.n_turns
        if self.turns_per_layer < 1:
            raise ValueError("turns_per_layer must be at least 1")

    @property
    def wire_radius(self) -> float:
        return self.wire_diameter / 2.0

    @property
    def pitch(self) -> float:
        """Centre-to-centre spacing of close-wound insulated wire."""
        return self.wire_diameter + 2.0 * self.insulation

    @property
    def n_layers(self) -> int:
        return -(-self.n_turns // self.turns_per_layer)

    def _grid(self):
        """(layer, index) for each turn in winding order (layers alternate)."""
        out = []
        tpl = self.turns_per_layer
        for k in range(self.n_turns):
            layer, pos = divmod(k, tpl)
            idx = pos if layer % 2 == 0 else tpl - 1 - pos
            out.append((layer, idx))
        return out

    def _conductor_area(self) -> float:
        return np.pi * self.wire_radius**2

    def _ac_factor(self, freq, resistivity):
        return res.round_wire_ac_factor(freq, self.wire_radius, resistivity)

    def _capacitance_pairs(self, turns):
        cpl = cap.parallel_wires_c_per_length(self.pitch, self.wire_radius, self.eps_r)
        where = {(t.layer, t.index): i for i, t in enumerate(turns)}
        pairs = []
        for i, t in enumerate(turns):
            for nb in ((t.layer, t.index + 1), (t.layer + 1, t.index)):
                j = where.get(nb)
                if j is not None:
                    length = 0.5 * (t.length + turns[j].length)
                    pairs.append((i, j, cpl * length))
        return pairs


@dataclass
class CircularCoil(_WireWinding, Coil):
    """Round wire on a circular former.

    ``radius`` is the centreline radius of the innermost layer. Turns in a
    layer are stacked axially at the wire pitch; further layers are wound
    outward, also at the wire pitch.

    Example: ``CircularCoil(radius=0.15, n_turns=80, wire_diameter=0.5e-3,
    turns_per_layer=10)``.
    """

    radius: float = 0.1
    name = "circular"

    def __post_init__(self):
        _WireWinding.__post_init__(self)
        if self.radius <= self.wire_radius:
            raise ValueError("radius must exceed the wire radius")

    def turns(self) -> List[Turn]:
        p = self.pitch
        out = []
        for layer, idx in self._grid():
            r = self.radius + layer * p
            out.append(Turn(size=r, z=idx * p, length=2 * np.pi * r, area=np.pi * r**2,
                            layer=layer, index=idx))
        return out

    def inductance(self) -> float:
        t = self.turns()
        return ind.circular_winding_inductance(
            [x.size for x in t], [x.z for x in t], self.wire_radius, self.internal
        )

    def describe(self) -> str:
        return (f"circular, r={self.radius * 1e3:.4g} mm, N={self.n_turns}, "
                f"{self.n_layers} layer(s), wire {self.wire_diameter * 1e3:.4g} mm")


@dataclass
class SquareCoil(_WireWinding, Coil):
    """Round wire on a square frame (square loop antenna).

    ``side`` is the centreline side length of the innermost layer.
    """

    side: float = 0.2
    name = "square"

    def __post_init__(self):
        _WireWinding.__post_init__(self)
        if self.side <= self.wire_diameter:
            raise ValueError("side must exceed the wire diameter")

    def turns(self) -> List[Turn]:
        p = self.pitch
        out = []
        for layer, idx in self._grid():
            s = self.side + 2 * layer * p
            out.append(Turn(size=s, z=idx * p, length=4 * s, area=s**2, layer=layer, index=idx))
        return out

    def inductance(self) -> float:
        segs = []
        for t in self.turns():
            segs += ind.square_turn_segments(t.size, t.z)
        a, y = self.wire_radius, self.internal
        return ind.segments_inductance(segs, lambda ln: ind.segment_self_round(ln, a, y))

    def describe(self) -> str:
        return (f"square, side={self.side * 1e3:.4g} mm, N={self.n_turns}, "
                f"{self.n_layers} layer(s), wire {self.wire_diameter * 1e3:.4g} mm")


# ---------------------------------------------------------------------------
# PCB spiral
# ---------------------------------------------------------------------------


@dataclass
class PCBSquareSpiral(Coil):
    """Square planar spiral, optionally on several series-connected layers.

    Each turn is modelled as a closed square at its centreline (the short jog
    between turns is ignored, the standard approximation). Odd layers spiral
    inward and even layers outward, so layers connect in series through vias.

    Parameters
    ----------
    outer_side : outer edge-to-edge side of the spiral, m
    turns_per_layer : turns on each layer
    trace_width, spacing : trace width and edge-to-edge gap, m
    n_layers : copper layers used
    copper_thickness : trace thickness, m (1 oz = 35 um)
    layer_spacing : dielectric thickness between adjacent layers, m
    eps_r : substrate relative permittivity (FR-4 ~ 4.4)
    """

    outer_side: float = 0.1
    turns_per_layer: int = 10
    trace_width: float = 0.3e-3
    spacing: float = 0.3e-3
    n_layers: int = 1
    copper_thickness: float = OZ_COPPER_THICKNESS
    layer_spacing: float = 1.5e-3
    eps_r: float = 4.4
    capacitance_scale: float = 1.0
    name = "pcb-spiral"

    def __post_init__(self):
        if self.turns_per_layer < 1 or self.n_layers < 1:
            raise ValueError("turns_per_layer and n_layers must be at least 1")
        if self.inner_side <= 0:
            raise ValueError(
                "spiral does not fit: reduce turns, trace width or spacing, "
                f"or enlarge outer_side (inner opening = {self.inner_side * 1e3:.3g} mm)"
            )

    @property
    def inner_side(self) -> float:
        """Inner edge-to-edge opening (Mohan's d_in)."""
        n = self.turns_per_layer
        return self.outer_side - 2 * n * self.trace_width - 2 * (n - 1) * self.spacing

    def _side(self, k: int) -> float:
        return self.outer_side - self.trace_width - 2 * k * (self.trace_width + self.spacing)

    def turns(self) -> List[Turn]:
        n = self.turns_per_layer
        out = []
        for layer in range(self.n_layers):
            order = range(n) if layer % 2 == 0 else range(n - 1, -1, -1)
            for k in order:
                s = self._side(k)
                out.append(Turn(size=s, z=layer * self.layer_spacing, length=4 * s, area=s**2,
                                layer=layer, index=k))
        return out

    def _conductor_area(self) -> float:
        return self.trace_width * self.copper_thickness

    def _ac_factor(self, freq, resistivity):
        return res.trace_ac_factor(freq, self.trace_width, self.copper_thickness, resistivity)

    def inductance(self) -> float:
        segs = []
        for t in self.turns():
            segs += ind.square_turn_segments(t.size, t.z)
        w, th = self.trace_width, self.copper_thickness
        return ind.segments_inductance(segs, lambda ln: ind.segment_self_rect(ln, w, th))

    def _capacitance_pairs(self, turns):
        eps_eff = 0.5 * (self.eps_r + 1.0)
        cpl = cap.coplanar_strips_c_per_length(self.trace_width, self.spacing, eps_eff)
        where = {(t.layer, t.index): i for i, t in enumerate(turns)}
        pairs = []
        for i, t in enumerate(turns):
            j = where.get((t.layer, t.index + 1))
            if j is not None:
                pairs.append((i, j, cpl * 0.5 * (t.length + turns[j].length)))
            j = where.get((t.layer + 1, t.index))
            if j is not None:
                c = cap.parallel_plate(self.trace_width * t.length, self.layer_spacing, self.eps_r)
                pairs.append((i, j, c))
        return pairs

    def describe(self) -> str:
        return (f"pcb spiral, {self.outer_side * 1e3:.4g} mm, {self.turns_per_layer} turns x "
                f"{self.n_layers} layer(s), w={self.trace_width * 1e3:.3g} mm, "
                f"s={self.spacing * 1e3:.3g} mm")
