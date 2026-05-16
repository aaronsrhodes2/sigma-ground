"""Visualise the per-toggle ablation: which toggle moved which body, by how much.

Reads rolling_shootout_toggle_iteration.json (8 predictors x 4 windows x 25
bodies x 4 samples). For each (predictor, body) pair, computes the
end-of-horizon error mean across the 4 windows, then takes its ratio to the
finedt baseline.

Produces:

  fig_toggle_heatmap.png
    Body x toggle matrix. Cell colour = log-ratio of error vs finedt baseline.
    Red = the toggle MADE THINGS WORSE. Blue = the toggle improved things.
    White = no effect at the 0.01% level.

    Most cells are white -- the finedt baseline is already so accurate
    that most BORROWED-pending-derivation textbook layers are inert at the
    3y horizon. The story is the few non-white cells: notably Enceladus
    under j4_zonal, which is the only meaningful regression and drove the
    decision to disable j4_zonal in the canonical jpl_de440 predictor.

  fig_toggle_movers_only.png
    Filtered version: only bodies where some toggle moved the error by >1%.
    Easier to read; loses the "most cells are inert" message.

Usage:
    python scripts/plot_toggle_ablation.py
        [--results PATH]    default: rolling_shootout_toggle_iteration.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


_FIXTURES = (
    Path(__file__).parent.parent
    / "sigma_ground" / "field" / "interface" / "fixtures"
)
_PLOTS_DIR = _FIXTURES / "rolling_shootout_plots"
_PLOTS_DIR.mkdir(exist_ok=True)

_BODY_ORDER = [
    "Mercury", "Venus", "Earth", "Moon", "Mars",
    "Phobos", "Deimos", "Jupiter", "Io", "Europa",
    "Ganymede", "Callisto", "Saturn", "Enceladus", "Titan",
    "Uranus", "Miranda", "Ariel", "Umbriel", "Titania",
    "Oberon", "Neptune", "Triton", "Pluto", "Charon",
]

# Order columns left to right: the toggles get progressively more "stuff" enabled.
_TOGGLE_ORDER = [
    ("finedt_gr2pn",         "+2PN"),
    ("finedt_j3",            "+J3"),
    ("finedt_j4",            "+J4"),
    ("finedt_tidal",         "+tidal"),
    ("finedt_tidal_j4",      "+tidal+J4"),
    ("finedt_tidal_j4_j3",   "+tidal+J4+J3"),
    ("finedt_all",           "all"),
]


def load_mean_end_errors(results_path: Path, horizon_yr: float = 3.0,
                          tolerance_yr: float = 0.3
                          ) -> tuple[dict, dict]:
    """Returns (mean_err[predictor][body], stdev_err[predictor][body])."""
    with results_path.open() as f:
        data = json.load(f)
    samples = data["samples"]
    by = defaultdict(lambda: defaultdict(list))
    for s in samples:
        t_yr = (s["sample_jd"] - s["window_start_jd"]) / 365.25
        if abs(t_yr - horizon_yr) < tolerance_yr and s["error_au"] is not None:
            by[s["predictor"]][s["body"]].append(s["error_au"])
    mean = {p: {b: float(np.mean(v)) for b, v in d.items()}
            for p, d in by.items()}
    std  = {p: {b: float(np.std(v))  for b, v in d.items()}
            for p, d in by.items()}
    return mean, std


def plot_toggle_heatmap(mean: dict, output: Path) -> None:
    """Body x toggle heatmap. Cell colour = log10(ratio to finedt baseline)."""
    base = mean.get("finedt", {})
    bodies = [b for b in _BODY_ORDER if b in base]
    toggle_keys = [k for k, _ in _TOGGLE_ORDER if k in mean]
    toggle_labels = [label for k, label in _TOGGLE_ORDER if k in mean]

    matrix = np.zeros((len(bodies), len(toggle_keys)))
    for i, body in enumerate(bodies):
        b0 = base.get(body, 0.0)
        for j, pred in enumerate(toggle_keys):
            val = mean[pred].get(body, 0.0)
            if b0 > 0 and val > 0:
                matrix[i, j] = np.log10(val / b0)
            else:
                matrix[i, j] = 0.0

    # Symmetric colour scale so worse=red, better=blue, no-change=white.
    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(8, len(toggle_keys) * 1.5),
                                     max(8, len(bodies) * 0.32)))
    im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(range(len(toggle_keys)))
    ax.set_xticklabels(toggle_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(bodies)))
    ax.set_yticklabels(bodies, fontsize=9)
    ax.set_xlabel("Toggle (added to finedt baseline)")
    ax.set_title("Per-body error change from each toggle, 3y horizon "
                  "(mean over 4 windows)\n"
                  "Red = toggle WORSENED prediction. Blue = improved. White = inert.",
                  fontsize=11, pad=10)

    # Annotate cells with the ratio (only where |delta| > 0.1%)
    for i in range(len(bodies)):
        for j in range(len(toggle_keys)):
            ratio = 10 ** matrix[i, j]
            if abs(ratio - 1.0) > 0.001:
                colour = "white" if abs(matrix[i, j]) > 0.02 else "black"
                ax.text(j, i, f"{ratio:.3f}x", ha="center", va="center",
                         fontsize=7, color=colour)

    cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("log10(error_with_toggle / error_baseline)", fontsize=9)

    plt.tight_layout()
    plt.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output.name}")


def plot_toggle_movers_only(mean: dict, output: Path,
                              threshold_pct: float = 1.0) -> None:
    """Show only bodies where some toggle moves the error by >threshold_pct."""
    base = mean.get("finedt", {})
    bodies_all = [b for b in _BODY_ORDER if b in base]
    toggle_keys = [k for k, _ in _TOGGLE_ORDER if k in mean]
    toggle_labels = [label for k, label in _TOGGLE_ORDER if k in mean]

    # Filter to bodies with at least one toggle exceeding threshold
    bodies = []
    for body in bodies_all:
        b0 = base.get(body, 0.0)
        if b0 == 0:
            continue
        ratios = [mean[p].get(body, 0.0) / b0 for p in toggle_keys]
        max_dev = max(abs(r - 1.0) for r in ratios) * 100
        if max_dev > threshold_pct:
            bodies.append(body)

    if not bodies:
        print(f"  no bodies exceed {threshold_pct}% on any toggle -- skip movers plot")
        return

    matrix = np.zeros((len(bodies), len(toggle_keys)))
    for i, body in enumerate(bodies):
        b0 = base.get(body, 0.0)
        for j, pred in enumerate(toggle_keys):
            val = mean[pred].get(body, 0.0)
            matrix[i, j] = (val / b0 - 1.0) * 100  # % delta

    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.5)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(8, len(toggle_keys) * 1.5),
                                     max(3, len(bodies) * 0.7)))
    im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(range(len(toggle_keys)))
    ax.set_xticklabels(toggle_labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(bodies)))
    ax.set_yticklabels(bodies, fontsize=10)
    ax.set_xlabel("Toggle (added to finedt baseline)")
    ax.set_title(f"Bodies where at least one toggle moved error by "
                  f">{threshold_pct:.1f}% (3y horizon)\n"
                  "Red = worse, blue = better. The Enceladus J4 cell is the "
                  "diagnostic that drove jpl_de440's j4_zonal=False decision.",
                  fontsize=11, pad=10)

    for i in range(len(bodies)):
        for j in range(len(toggle_keys)):
            val = matrix[i, j]
            if abs(val) > 0.1:
                colour = "white" if abs(val) > vmax * 0.5 else "black"
                ax.text(j, i, f"{val:+.2f}%", ha="center", va="center",
                         fontsize=10, color=colour, weight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("% delta vs finedt baseline", fontsize=9)
    plt.tight_layout()
    plt.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=_FIXTURES / "rolling_shootout_toggle_iteration.json",
                        help="toggle-iteration results JSON")
    parser.add_argument("--horizon-yr", type=float, default=3.0,
                        help="end-of-horizon time for error snapshot")
    args = parser.parse_args()

    print(f"Loading {args.results.name}")
    mean, _ = load_mean_end_errors(args.results, horizon_yr=args.horizon_yr)

    print(f"Predictors present: {sorted(mean.keys())}")
    print(f"Bodies present:     {len(mean['finedt'])}")

    print("Plotting full heatmap (all 25 bodies)...")
    plot_toggle_heatmap(mean, _PLOTS_DIR / "fig_toggle_heatmap.png")

    print("Plotting movers-only (bodies with >1% delta on some toggle)...")
    plot_toggle_movers_only(mean, _PLOTS_DIR / "fig_toggle_movers_only.png",
                              threshold_pct=1.0)

    print(f"\nDone. Plots in {_PLOTS_DIR}/")


if __name__ == "__main__":
    main()
