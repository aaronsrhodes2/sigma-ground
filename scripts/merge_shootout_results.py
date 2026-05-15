"""Merge multiple rolling_shootout result JSONs into one.

The rolling_shootout.py CLI overwrites the results file each run. When we
iterate predictor-by-predictor (e.g. running over_physics_finedt on its own
to avoid rerunning the 4-hour full pass), we want to keep the baseline
samples and add the new ones to the same analysis.

Usage:
    python scripts/merge_shootout_results.py \
        sigma_ground/field/interface/fixtures/rolling_shootout_baseline_5pred.json \
        sigma_ground/field/interface/fixtures/rolling_shootout_results.json \
        --out sigma_ground/field/interface/fixtures/rolling_shootout_merged.json

Note: metadata fields (predictors list, n_windows, etc.) are merged by union.
If two source files report different window_start_keys or horizons, this
script flags it and exits — diff-able fingerprints require apples-to-apples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+",
                        help="result JSON files to merge")
    parser.add_argument("--out", type=Path, required=True,
                        help="output merged JSON path")
    args = parser.parse_args()

    metadata_check_keys = ("window_start_keys", "prediction_horizon_yr",
                           "sample_interval_yr")
    all_samples: list[dict] = []
    all_predictors: set[str] = set()
    merged_meta: dict = {}

    for path in args.inputs:
        data = json.loads(path.read_text())
        meta = data["metadata"]
        if not merged_meta:
            merged_meta = {k: meta[k] for k in metadata_check_keys}
        else:
            for k in metadata_check_keys:
                if meta[k] != merged_meta[k]:
                    raise SystemExit(
                        f"Cannot merge: {path.name} has {k}={meta[k]!r}, "
                        f"first file had {merged_meta[k]!r}")
        all_predictors.update(meta.get("predictors", []))
        all_samples.extend(data["samples"])
        print(f"  {path.name}: {len(data['samples']):,} samples, "
              f"predictors={meta.get('predictors')}")

    merged_meta["predictors"] = sorted(all_predictors)
    merged_meta["n_windows"] = len(merged_meta["window_start_keys"])
    merged_meta["merged_from"] = [str(p) for p in args.inputs]

    payload = {"metadata": merged_meta, "samples": all_samples}
    args.out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"\nMerged {len(all_samples):,} samples across "
          f"{len(all_predictors)} predictors -> {args.out}")


if __name__ == "__main__":
    main()
