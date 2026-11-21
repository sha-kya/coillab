"""Command-line interface: ``coillab <circular|square|pcb> [options]``.

Lengths are given in millimetres on the command line for convenience.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .coils import Amplifier, CircularCoil, PCBSquareSpiral, SquareCoil

MM = 1e-3


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--freq", type=float, default=1e3, help="analysis frequency, Hz (default 1000)")
    p.add_argument("--field", type=float, default=1e-9, help="peak field for the EMF line, T (default 1 nT)")
    p.add_argument("--temp", type=float, default=20.0, help="temperature, C")
    p.add_argument("--en", type=float, default=1e-9, help="amplifier voltage noise, V/rtHz")
    p.add_argument("--in", dest="in_", type=float, default=10e-15, help="amplifier current noise, A/rtHz")
    p.add_argument("--cin", type=float, default=10e-12, help="amplifier input capacitance, F")
    p.add_argument("--json", action="store_true", help="print the report as JSON")


def _wire(p: argparse.ArgumentParser) -> None:
    p.add_argument("--turns", type=int, required=True, help="number of turns")
    p.add_argument("--wire", type=float, required=True, help="bare wire diameter, mm")
    p.add_argument("--insulation", type=float, default=0.02, help="insulation per side, mm")
    p.add_argument("--turns-per-layer", type=int, default=None, help="turns per layer (default: single layer)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coillab", description="Analyse an inductive magnetic sensor coil.")
    parser.add_argument("--version", action="version", version=f"coillab {__version__}")
    sub = parser.add_subparsers(dest="kind", required=True)

    c = sub.add_parser("circular", help="round wire on a circular former")
    c.add_argument("--radius", type=float, required=True, help="innermost centreline radius, mm")
    _wire(c)
    _common(c)

    s = sub.add_parser("square", help="round wire on a square frame")
    s.add_argument("--side", type=float, required=True, help="innermost centreline side, mm")
    _wire(s)
    _common(s)

    b = sub.add_parser("pcb", help="square planar PCB spiral")
    b.add_argument("--outer-side", type=float, required=True, help="outer side, mm")
    b.add_argument("--turns", type=int, required=True, help="turns per layer")
    b.add_argument("--width", type=float, required=True, help="trace width, mm")
    b.add_argument("--spacing", type=float, required=True, help="trace gap, mm")
    b.add_argument("--layers", type=int, default=1, help="copper layers in series")
    b.add_argument("--copper", type=float, default=0.035, help="copper thickness, mm")
    b.add_argument("--layer-spacing", type=float, default=1.5, help="dielectric between layers, mm")
    b.add_argument("--eps-r", type=float, default=4.4, help="substrate permittivity")
    _common(b)
    return parser


def make_coil(args):
    if args.kind == "circular":
        return CircularCoil(radius=args.radius * MM, n_turns=args.turns, wire_diameter=args.wire * MM,
                            insulation=args.insulation * MM, turns_per_layer=args.turns_per_layer)
    if args.kind == "square":
        return SquareCoil(side=args.side * MM, n_turns=args.turns, wire_diameter=args.wire * MM,
                          insulation=args.insulation * MM, turns_per_layer=args.turns_per_layer)
    return PCBSquareSpiral(outer_side=args.outer_side * MM, turns_per_layer=args.turns,
                           trace_width=args.width * MM, spacing=args.spacing * MM, n_layers=args.layers,
                           copper_thickness=args.copper * MM, layer_spacing=args.layer_spacing * MM,
                           eps_r=args.eps_r)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        coil = make_coil(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    amp = Amplifier(en=args.en, in_=args.in_, c_in=args.cin)
    report = coil.analyze(freq=args.freq, amplifier=amp, field_amplitude=args.field, temp_c=args.temp)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
