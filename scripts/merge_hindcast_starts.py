"""Fold the per-start hindcast files of one predictor (scripts/run_hindcast_variants.py writes <outdir>/<predictor>_<year>.json, one start each, the same shape as the served
error table) into the served table, fixtures/prediction_error_table.json: per body and lead the median and maximum over the starts and the number of starts, with the
metadata (predictor, step, truth kernel, fingerprint) carried over and checked to agree across the files. The previous served table is kept beside it under its own name,
never overwritten in place (a stored table is a record).

usage: python scripts/merge_hindcast_starts.py D:/datasets/sigma_runs/core_restated_math_tide --predictor restated_math_tide [--out fixtures/prediction_error_table.json]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
OUT = _ROOT / "sigma_ground/field/interface/fixtures/prediction_error_table.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--predictor", required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    files = sorted(a.folder.glob(f"{a.predictor}_*.json"))
    if not files:
        print("no per-start files found", file=sys.stderr)
        return 1
    docs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    metas = docs[0]["metadata"]
    for k in ("predictor", "dt_days", "truth", "model_fingerprint", "targets_end"):
        vals = {json.dumps(d["metadata"].get(k)) for d in docs}
        if len(vals) != 1:
            print(f"the files disagree on {k}: {vals}", file=sys.stderr)
            return 1
    cells: dict = {}
    per_start = {}
    for d in docs:
        for y, bodies in d["per_start"].items():
            per_start[y] = bodies
            for b, leads in bodies.items():
                for k, e in leads.items():
                    cells.setdefault(b, {}).setdefault(str(k), []).append(float(e))
    table = {b: {k: {"median_km": statistics.median(v), "max_km": max(v), "n_starts": len(v)} for k, v in sorted(c.items(), key=lambda kv: int(kv[0]))} for b, c in cells.items()}
    if a.out.exists():
        old = json.loads(a.out.read_text(encoding="utf-8"))
        keep = a.out.with_name(f"prediction_error_table_{old['metadata'].get('predictor', 'previous')}_{old['metadata'].get('dt_days', 'x')}d.json")
        if not keep.exists():
            shutil.copy2(a.out, keep)
            print("kept the previous table as", keep.name)
    doc = {"metadata": {**metas, "starts": sorted(int(y) for y in per_start), "merged_from": [f.name for f in files], "merged_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "generator": "scripts/make_prediction_error_table.py per start (scripts/run_hindcast_variants.py), folded by scripts/merge_hindcast_starts.py"},
           "table": table, "per_start": per_start}
    a.out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print("wrote", a.out, "predictor", metas["predictor"], "truth", metas.get("truth"), "starts", doc["metadata"]["starts"])
    for b in ("Mercury", "Venus", "Earth", "Moon", "Mars"):
        print(f"  {b:8s} +5 yr {table[b]['5']['median_km']:.4f} km over {table[b]['5']['n_starts']} starts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
