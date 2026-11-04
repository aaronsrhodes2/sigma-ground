"""Whole-sky report (stage 2, phase 2): the sky scoreboard by class and lead, the frozen-start residual growth, and the counts; misc/MPC_SKY.md and a JSON for the portal.

    python scripts/mpc_sky_report.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parent.parent
SKY = Path("D:/datasets/mpc_sky")
LEAD_BINS = [(-1e9, 0.0, "fit (lead <= 0)"), (0.0, 30.0, "0-30 d"), (30.0, 90.0, "30-90 d"), (90.0, 180.0, "90-180 d"), (180.0, 365.0, "180-365 d"), (365.0, 1e9, "> 1 yr")]


def main() -> int:
    sb_f = SKY / "sky_scoreboard.parquet"
    res_files = sorted((SKY / "residuals").glob("*.parquet"))
    if not sb_f.exists() or not res_files:
        print("no whole-sky rows yet"); return 0
    sb = pq.read_table(sb_f).to_pandas()
    res = pd.concat([pq.read_table(f, columns=["night", "packed", "orbit_class", "total_arcsec", "dra_cosdec_arcsec", "ddec_arcsec", "lead_days"]).to_pandas() for f in res_files])
    nights = sorted(sb["night"].unique())
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "nights": nights, "n_bodies": int(sb["packed"].nunique()), "n_obs": int(len(res)),
           "n_forward": int((res["lead_days"] > 0).sum()), "by_class": {}, "by_lead": {}}
    lines = ["# The whole sky against our frozen predictions", "", f"Generated {out['generated_utc']}. Nights: {nights[0]} .. {nights[-1]} ({len(nights)}). Bodies {out['n_bodies']:,}; observations {out['n_obs']:,}, of which {out['n_forward']:,} are forward predictions (observed after the start was frozen).", "",
             "Every prediction runs from the MPC elements of the day the body was first seen, never refreshed; a residual at lead <= 0 is the fit's own residual (the baseline), a residual at lead > 0 is ours ahead of the sky.", "",
             "| class | bodies | observations | median arcsec (all) | median arcsec (forward) | mean dRA cos dec | mean dDec |", "|---|---|---|---|---|---|---|"]
    for k, g in res.groupby("orbit_class"):
        f = g[g["lead_days"] > 0]
        row = {"bodies": int(g["packed"].nunique()), "n_obs": int(len(g)), "median_arcsec": float(g["total_arcsec"].median()), "median_forward_arcsec": float(f["total_arcsec"].median()) if len(f) else None,
               "mean_dra": float(g["dra_cosdec_arcsec"].mean()), "mean_ddec": float(g["ddec_arcsec"].mean())}
        out["by_class"][k] = row
        fwd = "-" if row["median_forward_arcsec"] is None else f"{row['median_forward_arcsec']:.3f}"
        lines.append(f"| {k} | {row['bodies']:,} | {row['n_obs']:,} | {row['median_arcsec']:.3f} | {fwd} | {row['mean_dra']:+.3f} | {row['mean_ddec']:+.3f} |")
    lines += ["", "## Residual against lead (all classes)", "", "| lead | observations | median arcsec | 90th percentile |", "|---|---|---|---|"]
    for lo, hi, name in LEAD_BINS:
        g = res[(res["lead_days"] > lo) & (res["lead_days"] <= hi)]
        if len(g) == 0:
            continue
        row = {"n": int(len(g)), "median_arcsec": float(g["total_arcsec"].median()), "p90_arcsec": float(np.percentile(g["total_arcsec"], 90))}
        out["by_lead"][name] = row
        lines.append(f"| {name} | {row['n']:,} | {row['median_arcsec']:.3f} | {row['p90_arcsec']:.3f} |")
    lines += ["", f"Source files: `D:/datasets/mpc_sky/residuals/<night>.parquet`, `sky_scoreboard.parquet`; sealed forecasts in `seals/` with their hashes in `fixtures/mpc_live/sky_seal_log.jsonl`.", ""]
    (_ROOT / "misc" / "MPC_SKY.md").write_text("\n".join(lines), encoding="utf-8")
    (_ROOT / "sigma_ground/field/interface/fixtures/mpc_live/sky_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    latest = sb.sort_values("night").groupby("packed").tail(1)                       # each body's newest night: the compact per-body fixture Mentat answers from (committed with the day)
    cols = ["packed", "number", "designation", "orbit_class", "night", "n_obs", "n_forward", "median_arcsec", "median_forward_arcsec", "mean_along_arcsec", "mean_cross_arcsec", "lead_days_max", "h_mag", "frozen_on"]
    cols = [c for c in cols if c in latest.columns]
    rows = json.loads(latest[cols].to_json(orient="records"))
    (_ROOT / "sigma_ground/field/interface/fixtures/mpc_live/sky_scoreboard_latest.json").write_text(json.dumps({"generated_utc": out["generated_utc"], "nights": nights, "rows": rows}, separators=(",", ":")), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
