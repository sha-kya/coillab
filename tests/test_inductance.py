"""Inductance models checked against independent public references."""

import numpy as np
import pytest

from coillab import CircularCoil, PCBSquareSpiral, SquareCoil
from coillab import inductance as ind
from coillab import references as ref
from coillab.constants import MU0


def test_maxwell_mutual_matches_brute_force_neumann():
    exact = ind.mutual_coaxial_loops(0.10, 0.12, 0.03)
    numeric = ref.neumann_mutual(ref.circle_path(0.10, 0.0, 3000), ref.circle_path(0.12, 0.03, 3000))
    assert exact == pytest.approx(numeric, rel=1e-5)


def test_mutual_is_symmetric_and_decays():
    assert ind.mutual_coaxial_loops(0.1, 0.2, 0.05) == pytest.approx(ind.mutual_coaxial_loops(0.2, 0.1, -0.05))
    near = ind.mutual_coaxial_loops(0.1, 0.1, 0.01)
    far = ind.mutual_coaxial_loops(0.1, 0.1, 1.0)
    assert near > far > 0
    # far-field dipole limit: M -> mu0 pi r1^2 r2^2 / (2 d^3)
    d = 5.0
    dipole = MU0 * np.pi * 0.1**2 * 0.1**2 / (2 * d**3)
    assert ind.mutual_coaxial_loops(0.1, 0.1, d) == pytest.approx(dipole, rel=1e-3)


def test_loop_self_inductance_vs_surface_filament():
    radius, a = 0.1, 1e-3
    formula = ind.loop_self_inductance(radius, a, internal=0.0)
    numeric = ref.neumann_mutual(ref.circle_path(radius, 0, 6000), ref.circle_path(radius - a, 0, 6000))
    assert formula == pytest.approx(numeric, rel=0.01)


@pytest.mark.parametrize("side,a", [(0.1, 0.5e-3), (0.5, 1e-3), (1.0, 0.25e-3)])
def test_square_loop_vs_grover(side, a):
    coil = SquareCoil(side=side, n_turns=1, wire_diameter=2 * a)
    assert coil.inductance() == pytest.approx(ref.grover_square_loop(side, a), rel=1e-3)


@pytest.mark.parametrize("radius,n,wire", [(0.02, 100, 0.5e-3), (0.05, 60, 1e-3)])
def test_single_layer_solenoid_vs_wheeler(radius, n, wire):
    coil = CircularCoil(radius=radius, n_turns=n, wire_diameter=wire)
    wheeler = ref.wheeler_single_layer(radius, n * coil.pitch, n)
    assert coil.inductance() == pytest.approx(wheeler, rel=0.02)


def test_multilayer_vs_wheeler():
    pitch = 1.04e-3
    coil = CircularCoil(radius=0.03 - 8 * pitch, n_turns=19 * 17, wire_diameter=1e-3, turns_per_layer=19)
    wheeler = ref.wheeler_multilayer(0.03, 19 * coil.pitch, 17 * coil.pitch, 323)
    assert coil.inductance() == pytest.approx(wheeler, rel=0.02)


def test_thin_bundle_vs_maxwell_gmd():
    """A thin multilayer bundle behaves like a ring with the winding's GMD."""
    coil = CircularCoil(radius=0.15, n_turns=80, wire_diameter=0.5e-3, turns_per_layer=10)
    p = coil.pitch
    b, c = 10 * p, 8 * p
    r_mean = 0.15 + 3.5 * p
    gmd = 0.2235 * (b + c)
    maxwell = MU0 * 80**2 * r_mean * (np.log(8 * r_mean / gmd) - 2)
    assert coil.inductance() == pytest.approx(maxwell, rel=0.01)


@pytest.mark.parametrize(
    "d_out,n,w,s",
    [(0.1, 20, 0.3e-3, 0.3e-3), (0.05, 10, 0.5e-3, 0.25e-3), (0.2, 30, 1e-3, 0.5e-3), (0.02, 8, 0.2e-3, 0.2e-3)],
)
def test_pcb_spiral_vs_mohan(d_out, n, w, s):
    coil = PCBSquareSpiral(outer_side=d_out, turns_per_layer=n, trace_width=w, spacing=s)
    d_in = coil.inner_side
    L = coil.inductance()
    assert L == pytest.approx(ref.mohan_current_sheet(n, d_out, d_in), rel=0.03)
    assert L == pytest.approx(ref.mohan_modified_wheeler(n, d_out, d_in), rel=0.04)


def test_inductance_scales_roughly_as_n_squared():
    l10 = CircularCoil(radius=0.1, n_turns=10, wire_diameter=0.5e-3, turns_per_layer=5).inductance()
    l20 = CircularCoil(radius=0.1, n_turns=20, wire_diameter=0.5e-3, turns_per_layer=5).inductance()
    assert 3.0 < l20 / l10 < 4.2


def test_two_layer_pcb_more_than_doubles_inductance():
    one = PCBSquareSpiral(outer_side=0.05, turns_per_layer=10, trace_width=0.3e-3, spacing=0.3e-3)
    two = PCBSquareSpiral(outer_side=0.05, turns_per_layer=10, trace_width=0.3e-3, spacing=0.3e-3,
                          n_layers=2, layer_spacing=0.2e-3)
    # tightly coupled layers in series approach 4x
    assert 3.0 < two.inductance() / one.inductance() < 4.0
