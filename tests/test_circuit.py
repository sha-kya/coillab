"""Resistance, capacitance, area, SRF and noise."""

import json
import math

import numpy as np
import pytest

from coillab import Amplifier, CircularCoil, PCBSquareSpiral, SquareCoil
from coillab import capacitance as cap
from coillab import resistance as res
from coillab.cli import main
from coillab.constants import K_BOLTZMANN, copper_resistivity

RHO = copper_resistivity(20.0)


# -- resistance -------------------------------------------------------------

def test_dc_resistance_of_known_wire():
    coil = CircularCoil(radius=0.1, n_turns=10, wire_diameter=1e-3)
    length = 10 * 2 * math.pi * 0.1
    assert coil.resistance_dc() == pytest.approx(RHO * length / (math.pi * 0.5e-3**2))


def test_skin_effect_limits_round_wire():
    a = 0.5e-3
    assert res.round_wire_ac_factor(1.0, a, RHO) == pytest.approx(1.0, abs=1e-9)
    f = 100e6
    x = a / res.skin_depth(f, RHO)
    assert res.round_wire_ac_factor(f, a, RHO) == pytest.approx(x / 2 + 0.25, rel=1e-3)
    # continuity across the switch to the asymptote at a/delta = 40
    f40 = RHO / (math.pi * 4e-7 * math.pi * (a / 40) ** 2)
    lo, hi = res.round_wire_ac_factor([f40 * 0.999, f40 * 1.001], a, RHO)
    assert lo == pytest.approx(hi, rel=2e-3)


def test_trace_ac_factor_tends_to_one():
    assert res.trace_ac_factor(10.0, 0.3e-3, 35e-6, RHO) == pytest.approx(1.0, rel=1e-3)
    assert res.trace_ac_factor(1e8, 0.3e-3, 35e-6, RHO) > 3.0


# -- capacitance ------------------------------------------------------------

def test_two_turn_energy_method():
    # two equal turns sit at potentials 0.25 and 0.75 -> C_eq = C_tt / 4
    v = cap.voltage_fractions([1.0, 1.0])
    assert v == pytest.approx([0.25, 0.75])
    assert cap.energy_method([(0, 1, 8e-12)], v) == pytest.approx(2e-12)


def test_parallel_wire_capacitance_decreases_with_pitch():
    c1 = cap.parallel_wires_c_per_length(1.1e-3, 0.5e-3)
    c2 = cap.parallel_wires_c_per_length(2e-3, 0.5e-3)
    assert c1 > c2 > 0
    with pytest.raises(ValueError):
        cap.parallel_wires_c_per_length(0.9e-3, 0.5e-3)


def test_coplanar_strip_capacitance_symmetry_point():
    # k = s/(s+2w) = 1/sqrt(2) gives K(k') = K(k), so C' = eps0 * eps_eff
    gap = 1.0
    width = gap * (math.sqrt(2) - 1) / 2
    assert cap.coplanar_strips_c_per_length(width, gap, 1.0) == pytest.approx(8.8541878128e-12, rel=1e-9)


def test_capacitance_scale_calibration():
    base = SquareCoil(side=0.3, n_turns=20, wire_diameter=0.4e-3)
    scaled = SquareCoil(side=0.3, n_turns=20, wire_diameter=0.4e-3, capacitance_scale=2.0)
    assert scaled.self_capacitance() == pytest.approx(2 * base.self_capacitance())


def test_srf_falls_with_more_turns():
    srfs = [PCBSquareSpiral(outer_side=0.1, turns_per_layer=n, trace_width=0.3e-3, spacing=0.3e-3).srf()
            for n in (5, 10, 20)]
    assert srfs[0] > srfs[1] > srfs[2]


# -- area, EMF and noise --------------------------------------------------------

def test_emf_of_draft_mockup():
    """80 turns at 150 mm (one radius), 6.1 kHz, 1 nT peak -> about 216.7 uV."""
    coil = CircularCoil(radius=0.15, n_turns=80, wire_diameter=0.05e-3, insulation=1e-6)
    assert all(abs(t.size - 0.15) < 1e-12 for t in coil.turns())  # single layer: same radius
    report = coil.analyze(freq=6100, field_amplitude=1e-9)
    assert report.area_sum == pytest.approx(80 * math.pi * 0.15**2)
    assert report.emf == pytest.approx(216.8e-6, rel=1e-3)


def test_pcb_summed_area():
    coil = PCBSquareSpiral(outer_side=0.05, turns_per_layer=3, trace_width=1e-3, spacing=1e-3)
    sides = [0.05 - 1e-3 - 2 * k * 2e-3 for k in range(3)]
    assert coil.area_sum() == pytest.approx(sum(s**2 for s in sides))


def test_pcb_that_does_not_fit_raises():
    with pytest.raises(ValueError):
        PCBSquareSpiral(outer_side=0.01, turns_per_layer=50, trace_width=0.3e-3, spacing=0.3e-3)


def test_noise_is_thermal_limited_with_quiet_amplifier():
    coil = SquareCoil(side=0.3, n_turns=30, wire_diameter=0.3e-3)
    amp = Amplifier(en=1e-15, in_=1e-18, c_in=0.0)
    spec = coil.spectrum([100.0], amp)
    r = coil.resistance_ac(100.0)
    assert spec["noise_emf"][0] == pytest.approx(math.sqrt(4 * K_BOLTZMANN * 293.15 * r), rel=1e-3)
    assert spec["noise_field"][0] == pytest.approx(spec["noise_emf"][0] / (2 * math.pi * 100 * coil.area_sum()))


def test_more_area_lowers_field_noise():
    small = CircularCoil(radius=0.05, n_turns=50, wire_diameter=0.3e-3, turns_per_layer=10)
    large = CircularCoil(radius=0.20, n_turns=50, wire_diameter=0.3e-3, turns_per_layer=10)
    assert large.analyze(1e3).noise_field < small.analyze(1e3).noise_field


def test_warning_near_resonance():
    coil = PCBSquareSpiral(outer_side=0.1, turns_per_layer=20, trace_width=0.3e-3, spacing=0.3e-3)
    assert coil.analyze(freq=0.9 * coil.srf(10e-12)).warnings
    assert not coil.analyze(freq=1e3).warnings


# -- CLI ----------------------------------------------------------------------

def test_cli_text_and_json(capsys):
    assert main(["circular", "--radius", "150", "--turns", "80", "--wire", "0.5", "--turns-per-layer", "10"]) == 0
    assert "Inductance" in capsys.readouterr().out
    assert main(["pcb", "--outer-side", "100", "--turns", "20", "--width", "0.3", "--spacing", "0.3",
                 "--layers", "2", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["n_turns"] == 40 and data["inductance"] > 0


def test_cli_reports_bad_geometry(capsys):
    assert main(["pcb", "--outer-side", "10", "--turns", "50", "--width", "0.3", "--spacing", "0.3"]) == 2
    assert "error" in capsys.readouterr().err


def test_spectrum_shapes():
    coil = SquareCoil(side=0.3, n_turns=30, wire_diameter=0.3e-3)
    f = np.logspace(1, 5, 50)
    spec = coil.spectrum(f)
    assert spec["noise_field"].shape == f.shape
    assert np.all(spec["noise_field"] > 0)
