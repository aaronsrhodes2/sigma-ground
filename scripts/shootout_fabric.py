"""The forecast "fabric": every forecast's error, laid out by target date, and a test for stripes.

Input: a rolling shoot-out file (``samples``: predictor, body, window_start_jd, sample_jd, error_au). Each window is one forecast issued at its start date
and scored at later sample dates (lead = sample - start). Plotting every forecast against its target date gives your picture: lines that start near zero
at each issue date and rise with lead time.

The diagnostic that tells a real event from a model deficiency (the Captain's ripple idea, 2026-10-11):
  * error that grows with LEAD TIME and repeats for every issue date is a model deficiency (a diagonal in the fabric);
  * error that sits at one TARGET DATE in forecasts issued at DIFFERENT times is an event or a data anomaly at that date (a vertical stripe).
For each body and lead k >= 1, z = ln(error / median error at that lead over all issue dates). A target date's stripe score is the median z over
the forecasts that reach it (at least 3). A date is flagged when its score exceeds 3 robust sigma (1.4826 MAD) of all that body's stripe scores.

Honest limits: this uses the shoot-out's DE440 "truth", which is itself a fit, so a stripe here means "our integration disagrees with DE440 at that date
in every forecast", not yet a statement about the sky. With real measurements and old JPL ephemerides (DE405/421/430) the same code applies.

usage: python scripts/shootout_fabric.py <rolling_shootout.json> --label v2physics_r1 [--predictor de440_stack] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

AU_KM = 149597870.7
J2000 = 2451545.0
PLANETS = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Moon"]


def year_of(jd: float) -> float:
    return 2000.0 + (jd - J2000) / 365.25


def load(path: Path, predictor: str):
    d = json.loads(path.read_text())
    rows = [s for s in d["samples"] if s["predictor"] == predictor and s.get("error_au") is not None]
    if not rows:
        raise SystemExit(f"no samples for predictor {predictor!r}; have {sorted({s['predictor'] for s in d['samples']})}")
    return d["metadata"], rows


def build(rows, bodies):
    by_body = defaultdict(dict)       # body -> {(origin_jd, lead_k): err_km}
    for s in rows:
        if s["body"] not in bodies:
            continue
        k = int(round((s["sample_jd"] - s["window_start_jd"]) / 365.25))
        by_body[s["body"]][(s["window_start_jd"], k)] = s["error_au"] * AU_KM
    return by_body


def stripes(table):
    """Return {target_year: (score, n)} for one body."""
    leads = defaultdict(list)
    for (o, k), e in table.items():
        if k >= 1 and e > 0:
            leads[k].append(e)
    base = {k: float(np.median(v)) for k, v in leads.items()}
    per_target = defaultdict(list)
    for (o, k), e in table.items():
        if k >= 1 and e > 0 and k in base:
            per_target[int(round(year_of(o + k * 365.25)))].append(math.log(e / base[k]))
    return {t: (float(np.median(z)), len(z)) for t, z in per_target.items() if len(z) >= 3}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shootout", type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--predictor", default="de440_stack")
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--bodies", nargs="+", default=PLANETS)
    a = ap.parse_args()
    meta, rows = load(a.shootout, a.predictor)
    tables = build(rows, set(a.bodies))
    bodies = [b for b in a.bodies if b in tables]
    result = {"label": a.label, "predictor": a.predictor, "source": str(a.shootout), "bodies": {}}
    score_rows = {}
    for b in bodies:
        st = stripes(tables[b])
        vals = np.array([v[0] for v in st.values()])
        med = float(np.median(vals))
        sig = 1.4826 * float(np.median(np.abs(vals - med))) or 1e-9
        flagged = {t: v for t, v in st.items() if (v[0] - med) / sig > 3.0}
        result["bodies"][b] = {"stripe_scores": {str(t): {"score": v[0], "forecasts": v[1]} for t, v in sorted(st.items())},
                               "median_score": med, "robust_sigma": sig,
                               "flagged_target_years": {str(t): {"score": v[0], "sigmas": (v[0] - med) / sig, "forecasts": v[1]} for t, v in sorted(flagged.items())}}
        score_rows[b] = st
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"FABRIC_{a.label}.json").write_text(json.dumps(result, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(bodies)
    cols = 3
    rws = math.ceil(n / cols)
    fig, axes = plt.subplots(rws, cols, figsize=(15, 3.1 * rws), squeeze=False)
    origins = sorted({o for t in tables.values() for (o, k) in t})
    cmap = plt.get_cmap("viridis")
    for ax, b in zip(axes.ravel(), bodies):
        for i, o in enumerate(origins):
            pts = sorted((k, e) for (oo, k), e in tables[b].items() if oo == o)
            xs = [year_of(o + k * 365.25) for k, _ in pts]
            ys = [max(e, 1e-3) for _, e in pts]
            ax.plot(xs, ys, color=cmap(i / max(1, len(origins) - 1)), lw=1.3, marker="o", ms=2)
        ax.set_yscale("log")
        ax.set_title(b, fontsize=10)
        ax.set_xlabel("target date")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))
        ax.set_ylabel("error (km)")
        for t in result["bodies"][b]["flagged_target_years"]:
            ax.axvline(float(t), color="red", lw=0.8, alpha=0.6)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Forecast fabric, {a.predictor} vs DE440 ({a.label}): one line per issue date; red = stripe flagged (>3 robust sigma)", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out / f"FABRIC_{a.label}_lines.png", dpi=110)
    plt.close(fig)
    # stripe heat map: body x target year
    years = sorted({t for st in score_rows.values() for t in st})
    grid = np.full((len(bodies), len(years)), np.nan)
    for i, b in enumerate(bodies):
        for t, v in score_rows[b].items():
            grid[i, years.index(t)] = v[0]
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(bodies) + 2.2))
    im = ax.imshow(grid, aspect="auto", cmap="coolwarm", vmin=-1.5, vmax=1.5)
    ax.set_yticks(range(len(bodies)))
    ax.set_yticklabels(bodies)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=90, fontsize=8)
    ax.set_xlabel("target year (score: median ln(error / typical error at that lead) over the forecasts reaching it; red = worse than typical)")
    fig.colorbar(im, ax=ax, label="stripe score")
    ax.set_title(f"Stripe scores by target date ({a.predictor}, {a.label})")
    fig.tight_layout()
    fig.savefig(a.out / f"FABRIC_{a.label}_stripes.png", dpi=110)
    plt.close(fig)
    for b in bodies:
        f = result["bodies"][b]["flagged_target_years"]
        print(f"{b:8s} median {result['bodies'][b]['median_score']:+.2f} sigma {result['bodies'][b]['robust_sigma']:.2f}  flagged: "
              + (", ".join(f"{t} ({v['sigmas']:.1f}s, n={v['forecasts']})" for t, v in f.items()) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
