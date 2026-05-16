"""Two cleaner per-body visualizations of the rolling-window shootout.

Replaces the busy per_body_curves.png with two focused views:

  fig_per_body_magnitude.png
    25-panel grid, one body per panel. Y = log10(|Δr|/AU), x = months from
    window start. For each predictor: MEDIAN curve (bold) + IQR shaded band
    across the 16 windows. Cleaner than the all-windows-faint version.

  fig_per_body_rtn.png
    Same grid, but for each body shows RTN components (dR, dT, dN) as three
    colored lines for ONE predictor (over_physics_finedt by default).
    Recovers the 3D structure of the position error vector. R-dominance
    means wrong orbit shape, T-dominance means wrong phase, N-dominance
    means wrong orbital plane.

Usage:
    python scripts/plot_per_body_grid.py
        [--results PATH]  default: rolling_shootout_after_dt_fix.json
        [--predictor NAME]  for RTN plot, default over_physics_finedt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sigma_ground.field.interface.rolling_analysis import (
    load_results, _load_predicted_km_into_df, add_rtn_components,
)


_FIXTURES = (
    Path(__file__).parent.parent
    / "sigma_ground" / "field" / "interface" / "fixtures"
)
_PLOTS_DIR = _FIXTURES / "rolling_shootout_plots"
_PLOTS_DIR.mkdir(exist_ok=True)

# Body order: by orbital regime (inner/outer/dwarf), grouped meaningfully.
_BODY_ORDER = [
    "Mercury", "Venus", "Earth", "Moon", "Mars",
    "Phobos", "Deimos", "Jupiter", "Io", "Europa",
    "Ganymede", "Callisto", "Saturn", "Enceladus", "Titan",
    "Uranus", "Miranda", "Ariel", "Umbriel", "Titania",
    "Oberon", "Neptune", "Triton", "Pluto", "Charon",
]

# Predictor colors -- consistent across both plots
_PREDICTOR_COLORS = {
    "pure_newton":         "#999999",   # gray (baseline)
    "kepler":              "#88aacc",   # pale blue (2-body fit)
    "standard":            "#ee8844",   # orange (Newton + 1PN GR)
    "over_physics_no_j2":  "#bb44bb",   # purple (ablation: no J2)
    "over_physics":        "#cc3333",   # red (dt=1d, has the bug)
    "over_physics_finedt": "#3366cc",   # blue (the dt-fix winner)
}


def plot_per_body_magnitude(df: pd.DataFrame, output: Path) -> None:
    """25-panel grid: |Δr| vs time per body, median + IQR band per predictor."""
    df = df.dropna(subset=["error_au"]).copy()
    df = df[df["t_yr_bin"] >= 0]   # drop the t=0 sample (always 0)

    bodies = [b for b in _BODY_ORDER if b in df["body"].values]
    predictors = [
        p for p in _PREDICTOR_COLORS
        if p in df["predictor"].values
    ]

    n_cols = 5
    n_rows = (len(bodies) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 2.8 * n_rows),
                              sharex=True)
    axes = axes.flatten()

    for ax_idx, body in enumerate(bodies):
        ax = axes[ax_idx]
        body_df = df[df["body"] == body]

        for pred in predictors:
            sub = body_df[body_df["predictor"] == pred]
            if sub.empty:
                continue
            # Group by time bin, compute median + IQR (25th/75th %ile)
            grouped = sub.groupby("t_yr_bin")["error_au"]
            t_yrs = sorted(grouped.groups.keys())
            t_months = [t * 12 for t in t_yrs]
            medians = [grouped.get_group(t).median() for t in t_yrs]
            q25     = [grouped.get_group(t).quantile(0.25) for t in t_yrs]
            q75     = [grouped.get_group(t).quantile(0.75) for t in t_yrs]
            color = _PREDICTOR_COLORS.get(pred, "#666666")
            ax.fill_between(t_months, q25, q75, color=color, alpha=0.18,
                             linewidth=0)
            ax.plot(t_months, medians, color=color, linewidth=1.5,
                     label=(pred if ax_idx == 0 else None))

        ax.set_yscale("log")
        ax.set_title(body, fontsize=10, pad=2)
        ax.tick_params(labelsize=8)
        ax.grid(True, which="both", alpha=0.25)
        if ax_idx % n_cols == 0:
            ax.set_ylabel("|Δr| (AU)", fontsize=9)
        if ax_idx >= len(bodies) - n_cols:
            ax.set_xlabel("months from window start", fontsize=9)

    for k in range(len(bodies), len(axes)):
        axes[k].axis("off")

    fig.suptitle("Per-body prediction error (median + 25-75% band across 16 windows)\n"
                 "5y horizon, monthly sampling -- each panel is one body",
                 fontsize=12, y=0.995)
    fig.legend(loc="lower right", bbox_to_anchor=(0.99, 0.01),
                fontsize=9, framealpha=0.95, ncol=3)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(output, dpi=130)
    plt.close(fig)
    print(f"saved {output.name}")


def plot_per_body_rtn(df_rtn: pd.DataFrame, predictor: str,
                       output: Path) -> None:
    """25-panel grid: RTN components vs time for one chosen predictor.

    For each body, three lines:
      |dR| = radial   (shape error / wrong distance)
      |dT| = along-track (phase / timing error)
      |dN| = normal   (out-of-plane / inclination error)

    Plotting the absolute values on log scale because the signs flip across
    the orbit (along-track error can be +/-); we care about magnitude.
    """
    df_rtn = df_rtn[df_rtn["predictor"] == predictor].dropna(
        subset=["dR_au", "dT_au", "dN_au"]).copy()
    df_rtn = df_rtn[df_rtn["t_yr_bin"] >= 0]

    bodies = [b for b in _BODY_ORDER if b in df_rtn["body"].values]

    n_cols = 5
    n_rows = (len(bodies) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 2.8 * n_rows),
                              sharex=True)
    axes = axes.flatten()

    rtn_colors = {"R": "#dd3344", "T": "#33aa55", "N": "#3366cc"}
    rtn_labels = {"R": "|dR|  radial (shape)",
                  "T": "|dT|  along-track (phase)",
                  "N": "|dN|  normal (plane)"}

    for ax_idx, body in enumerate(bodies):
        ax = axes[ax_idx]
        body_df = df_rtn[df_rtn["body"] == body]

        for axis_name, col_name in (("R", "dR_au"), ("T", "dT_au"), ("N", "dN_au")):
            grouped = body_df.groupby("t_yr_bin")[col_name]
            t_yrs = sorted(grouped.groups.keys())
            t_months = [t * 12 for t in t_yrs]
            medians = [grouped.get_group(t).abs().median() for t in t_yrs]
            color = rtn_colors[axis_name]
            ax.plot(t_months, medians, color=color, linewidth=1.6,
                     label=(rtn_labels[axis_name] if ax_idx == 0 else None))

        ax.set_yscale("log")
        ax.set_title(body, fontsize=10, pad=2)
        ax.tick_params(labelsize=8)
        ax.grid(True, which="both", alpha=0.25)
        if ax_idx % n_cols == 0:
            ax.set_ylabel("component (AU)", fontsize=9)
        if ax_idx >= len(bodies) - n_cols:
            ax.set_xlabel("months from window start", fontsize=9)

    for k in range(len(bodies), len(axes)):
        axes[k].axis("off")

    fig.suptitle(f"Per-body RTN decomposition  --  predictor: {predictor}\n"
                 "R = radial (shape), T = along-track (phase), "
                 "N = orbital-plane normal",
                 fontsize=12, y=0.995)
    fig.legend(loc="lower right", bbox_to_anchor=(0.99, 0.01),
                fontsize=9, framealpha=0.95, ncol=3)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(output, dpi=130)
    plt.close(fig)
    print(f"saved {output.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=_FIXTURES / "rolling_shootout_after_dt_fix.json",
                        help="results JSON to plot from")
    parser.add_argument("--predictor", default="over_physics_finedt",
                        help="which predictor to use for the RTN plot")
    args = parser.parse_args()

    print(f"Loading {args.results.name}")
    _, df = load_results(args.results)

    print("Plotting per-body magnitude grid...")
    plot_per_body_magnitude(
        df,
        _PLOTS_DIR / "fig_per_body_magnitude.png",
    )

    print("Computing RTN decomposition...")
    df_rtn_src = _load_predicted_km_into_df(args.results)
    df_rtn     = add_rtn_components(df_rtn_src)

    print(f"Plotting per-body RTN grid for {args.predictor}...")
    plot_per_body_rtn(
        df_rtn, args.predictor,
        _PLOTS_DIR / "fig_per_body_rtn.png",
    )

    print(f"\nDone. Plots in {_PLOTS_DIR}/")


if __name__ == "__main__":
    main()
