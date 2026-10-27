"""Paired comparison of two predictors on the SAME hindcast starts: the stored `restated_zon` cells (fixtures/prediction_error_table.json, `per_start`) against another predictor's per-start files from
scripts/run_hindcast_variants.py. Same code, step, fixture snapshot and DE440 answer key; only the predictor's configuration differs, so any difference is that configuration.

For every body and lead: median heliocentric position error of each over the starts that have the lead, their ratio, and in how many starts the new one is closer to DE440.
Written to <out>/<label>.md and .json.

usage: python scripts/hindcast_paired_report.py NEW_DIR [--new restated_hdr] [--control-table PATH | --control-dir DIR [--control restated_zon]] [--out DIR] [--label HINDCAST_HDR_PAIRED]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONTROL = _ROOT / "sigma_ground/field/interface/fixtures/prediction_error_table.json"
LEADS = (1, 2, 3, 5, 10, 15, 20)


def load_new(folder: Path, predictor: str) -> dict[str, dict[str, dict[str, float]]]:
    per: dict[str, dict] = {}
    for f in sorted(folder.glob(f"{predictor}_*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        if doc["metadata"]["predictor"] != predictor:
            raise SystemExit(f"{f.name} was made by predictor {doc['metadata']['predictor']}, not {predictor}")
        per.update(doc["per_start"])
    return per


def compare(control: dict, new: dict) -> list[dict]:
    starts = sorted(set(control) & set(new))
    rows = []
    bodies = sorted({b for s in starts for b in new[s]})
    for b in bodies:
        for lead in LEADS:
            pairs = [(control[s][b][str(lead)], new[s][b][str(lead)]) for s in starts if b in control[s] and b in new[s] and str(lead) in control[s][b] and str(lead) in new[s][b]]
            if not pairs:
                continue
            mc, mn = statistics.median(x for x, _ in pairs), statistics.median(y for _, y in pairs)
            rows.append({"body": b, "lead": lead, "control_km": mc, "new_km": mn, "ratio": mn / mc if mc else None, "starts_better": sum(y < x for x, y in pairs), "starts": len(pairs)})
    return rows


def fmt(x: float) -> str:
    return f"{x:.3g}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("new_dir", type=Path)
    ap.add_argument("--new", default="restated_hdr")
    ap.add_argument("--control-table", type=Path, default=DEFAULT_CONTROL)
    ap.add_argument("--control-dir", type=Path, default=None, help="per-start files of the control predictor (same step as NEW_DIR), instead of the stored table")
    ap.add_argument("--control", default="restated_zon", help="predictor name of the files in --control-dir")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--label", default="HINDCAST_PAIRED")
    a = ap.parse_args()
    new = load_new(a.new_dir, a.new)
    if a.control_dir:
        control = {"metadata": {"predictor": a.control}, "per_start": load_new(a.control_dir, a.control)}
    else:
        control = json.loads(a.control_table.read_text(encoding="utf-8"))
    dts = {json.loads(f.read_text(encoding="utf-8"))["metadata"]["dt_days"] for f in a.new_dir.glob(f"{a.new}_*.json")}
    step = ", ".join(f"{d:g}" for d in sorted(dts))
    starts = sorted(set(control["per_start"]) & set(new))
    rows = compare(control["per_start"], new)
    lines = [f"# Hindcast, `{a.new}` against `{control['metadata']['predictor']}` ({len(starts)} starts: {', '.join(starts)})", "",
             f"Same code, step ({step} d), fixture snapshot (j2010) and DE440 answer key; only the predictor's configuration differs. Median heliocentric position error over the starts that reach the lead, km. "
             "Ratio below 1 means the new configuration is closer to DE440.", "",
             "| body | lead | control | new | ratio | starts better |", "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["body"], r["lead"])):
        lines.append(f"| {r['body']} | +{r['lead']} yr | {fmt(r['control_km'])} | {fmt(r['new_km'])} | {r['ratio']:.3f} | {r['starts_better']}/{r['starts']} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"{a.label}.md").write_text(text + "\n", encoding="utf-8")
        (a.out / f"{a.label}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
