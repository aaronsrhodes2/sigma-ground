"""The MPC live scoreboard as a per-body table (residual against lead, ours and JPL) and as the JSON the portal will draw (misc/CANDIDATE_LEDGER.md C27).

Reads D:/datasets/mpc_live/scoreboard.jsonl (one line per body per window, written by scripts/mpc_live_score.py), prints a Markdown table, and writes
sigma_ground/field/interface/fixtures/mpc_live/scoreboard_portal.json: per body its start epoch and source and the series of windows (lead in days, dates, counts, the two medians and robust
sigmas, the median of ours minus JPL). The portal itself (SkippyTel) is not touched here.

usage: python scripts/mpc_live_report.py [--scoreboard FILE] [--out FILE.json] [--md FILE.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
FIX = _ROOT / "sigma_ground/field/interface/fixtures"
DEFAULT_SCOREBOARD = Path("D:/datasets/mpc_live/scoreboard.jsonl")


def load_rows(path: Path) -> list[dict]:
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            rows.append(json.loads(ln))
    return rows


def build(rows: list[dict]) -> dict:
    """Per-body series from the scoreboard rows; a body's windows are sorted by lead, rows without a score are kept as gaps (n_obs 0)."""
    bodies: dict[int, dict] = {}
    for r in rows:
        b = bodies.setdefault(r["body"], {"body": r["body"], "name": r["name"], "start_epoch_jd_tdb": r.get("start_epoch_jd_tdb"), "start_source": r.get("start_source"), "series": [], "empty_runs": []})
        if r.get("n_scored"):
            b["series"].append({"date": r["date"], "lead_days": r["lead_days"], "lead_days_first": r.get("lead_days_first"), "window_utc": r.get("window_utc"), "n_scored": r["n_scored"],
                                "n_dropped_outliers": r.get("n_dropped_outliers", 0), "n_no_jpl": r.get("n_no_jpl", 0), "backfill": bool(r.get("backfill")),
                                "ours_median_arcsec": r["ours"]["median_arcsec"], "ours_sigma_arcsec": r["ours"]["robust_sigma_arcsec"],
                                "jpl_median_arcsec": r["jpl"]["median_arcsec"], "jpl_sigma_arcsec": r["jpl"]["robust_sigma_arcsec"],
                                "ours_minus_jpl_median_arcsec": r["ours_minus_jpl"]["median_arcsec"], "ours_minus_jpl_dra": r["ours_minus_jpl"]["median_dra_cosdec"], "ours_minus_jpl_ddec": r["ours_minus_jpl"]["median_ddec"],
                                "stations": len(r.get("stations", {}))})
        else:
            b["empty_runs"].append({"date": r["date"], "lead_days": r.get("lead_days"), "note": r.get("note", "")})
    for b in bodies.values():
        b["series"].sort(key=lambda s: s["lead_days"])
        b["n_scored_total"] = sum(s["n_scored"] for s in b["series"])
        b["last_date"] = max([s["date"] for s in b["series"]] + [e["date"] for e in b["empty_runs"]], default=None)
    out = sorted(bodies.values(), key=lambda b: (b["start_epoch_jd_tdb"] or 0, b["body"]))
    return {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "ledger": "misc/CANDIDATE_LEDGER.md C27",
            "units": {"residuals": "arcsec, total = sqrt((dRA cos Dec)^2 + dDec^2), O minus C", "lead": "days from the body's fixed start epoch to the newest observation in the window"},
            "arms": {"ours": "sigma-ground C24 engine from the fixed start state (scripts/eros_sky_residuals.py pipeline)", "jpl": "JPL Horizons topocentric astrometric RA/Dec at the observation instants"},
            "bodies": out}


def markdown(report: dict) -> str:
    lines = ["| body | start (JD TDB) | window (UTC) | lead d | n | ours median | ours sigma | JPL median | JPL sigma | ours-JPL median | dropped |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in report["bodies"]:
        if not b["series"]:
            note = b["empty_runs"][-1]["note"] if b["empty_runs"] else ""
            lines.append(f"| {b['body']} {b['name']} | {b['start_epoch_jd_tdb']} | - | - | 0 | - | - | - | - | - | {note} |")
            continue
        for s in b["series"]:
            w = s["window_utc"] or ["", ""]
            lines.append(f"| {b['body']} {b['name']} | {b['start_epoch_jd_tdb']} | {w[0][:10]} .. {w[1][:10]} | {s['lead_days']:.0f} | {s['n_scored']} | {s['ours_median_arcsec']:.3f} | {s['ours_sigma_arcsec']:.3f} "
                         f"| {s['jpl_median_arcsec']:.3f} | {s['jpl_sigma_arcsec']:.3f} | {s['ours_minus_jpl_median_arcsec']:.4f} | {s['n_dropped_outliers']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scoreboard", type=Path, default=DEFAULT_SCOREBOARD)
    ap.add_argument("--out", type=Path, default=FIX / "mpc_live" / "scoreboard_portal.json")
    ap.add_argument("--md", type=Path, default=_ROOT / "misc" / "MPC_LIVE_SCOREBOARD.md")
    a = ap.parse_args()
    if not a.scoreboard.exists():
        print(f"no scoreboard at {a.scoreboard}"); return 1
    rep = build(load_rows(a.scoreboard))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    md = markdown(rep)
    a.md.write_text(f"# MPC live scoreboard (C27)\n\nGenerated {rep['generated_utc']} from `{a.scoreboard}` by `scripts/mpc_live_report.py`; residuals in arcsec (O minus C), lead in days since each body's fixed start epoch.\n\n" + md, encoding="utf-8")
    print(md)
    print(f"wrote {a.out} and {a.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
