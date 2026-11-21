"""The turns trade-off on a fixed board size.

More turns add summed area, which lowers field-equivalent noise at low
frequency, but they also raise inductance and capacitance, pulling the
self-resonance down. Writes ``docs/turns_tradeoff.png``.

Run from the repository root:  python examples/turns_tradeoff.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from coillab import PCBSquareSpiral, eng

DOCS = Path(__file__).resolve().parents[1] / "docs"
TURNS = [5, 10, 20, 40]
COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]  # one hue, light -> dark = more turns


def main() -> None:
    import matplotlib.pyplot as plt

    f = np.logspace(1, 7, 600)
    ink, quiet, grid = "#0b0b0b", "#52514e", "#e4e3df"
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    print(f"{'turns':>5} {'L':>10} {'C_self':>10} {'SRF':>10} {'nB @1kHz':>14}")
    for n, color in zip(TURNS, COLORS):
        coil = PCBSquareSpiral(outer_side=0.1, turns_per_layer=n, trace_width=0.4e-3,
                               spacing=0.3e-3, n_layers=2, layer_spacing=1.5e-3)
        spec = coil.spectrum(f)
        nb = spec["noise_field"] * 1e15  # fT/rtHz
        rep = coil.analyze(1e3)
        print(f"{n:>5} {eng(rep.inductance, 'H'):>10} {eng(rep.c_self, 'F'):>10} {eng(rep.srf, 'Hz'):>10} "
              f"{rep.noise_field * 1e15:>10.1f} fT/rtHz")
        ax.loglog(f, nb, color=color, lw=2, label=f"{n}")
        ax.plot(rep.srf, np.interp(rep.srf, f, nb), "v", color=color, ms=8, mec="#fcfcfb", mew=1.5)
    leg = ax.legend(title="Turns per layer", loc="lower left", frameon=False, fontsize=8.5, title_fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(ink)
    leg.get_title().set_color(quiet)
    ax.set_xlabel("Frequency (Hz)", color=quiet, fontsize=9)
    ax.set_ylabel("Field-equivalent noise (fT/√Hz)", color=quiet, fontsize=9)
    ax.tick_params(colors=quiet, which="both", length=0, labelsize=8.5)
    ax.grid(True, which="major", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("More turns cut low-frequency noise but pull self-resonance (▼) down", loc="left",
                 fontsize=11, color=ink, fontweight="bold", pad=22)
    ax.text(0, 1.015, "100 mm two-layer PCB spiral, undamped, ideal 1 nV/√Hz amplifier",
            transform=ax.transAxes, fontsize=8.5, color=quiet, va="bottom")
    fig.tight_layout()
    DOCS.mkdir(exist_ok=True)
    fig.savefig(DOCS / "turns_tradeoff.png", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
