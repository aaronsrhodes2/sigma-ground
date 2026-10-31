"""Build fixtures/layer_effect_table.json from the C23b layer-ablation runs: for each of the ten bodies and each of eight physics layers, the median error (km) of the forecast's own model
(`restated_zon`) with that one layer removed, at leads 1 to 5 years, against the model with every layer on. Medians over the four hindcast starts (2000, 2005, 2010, 2015), reduced model, step 0.0125 d,
truth = JPL DE440 (so this is "how much does this layer change the agreement with DE440", the size of the layer, not an absolute accuracy).

usage: python scripts/make_layer_effect_table.py [--runs D:/datasets/sigma_runs] [--check]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
OUT = _ROOT / "sigma_ground/field/interface/fixtures/layer_effect_table.json"
LAYERS = ("eih", "zonals", "tides", "srp", "asteroids", "ring", "kbo", "c22")
STARTS = (2000, 2005, 2010, 2015)
LEADS = (1, 2, 3, 4, 5)


def medians(directory: Path, prefix: str) -> dict:
    vals: dict[str, dict[int, list[float]]] = {}
    for y in STARTS:
        d = json.loads((directory / f"{prefix}_{y}.json").read_text(encoding="utf-8"))
        for body, leads in d["per_start"][str(y)].items():
            for lead in LEADS:
                v = leads.get(str(lead))
                if v is not None:
                    vals.setdefault(body, {}).setdefault(lead, []).append(float(v))
    return {b: {str(k): {"km": statistics.median(v), "n_starts": len(v)} for k, v in sorted(ls.items())} for b, ls in vals.items()}


def build(runs: Path) -> dict:
    control = medians(runs / "zon_fine2", "restated_zon")
    layers = {k: medians(runs / f"abl_{k}", f"restated_zon_no_{k}") for k in LAYERS}
    meta = json.loads((runs / "zon_fine2" / "restated_zon_2000.json").read_text(encoding="utf-8"))["metadata"]
    return {"what": __doc__.split("\n\n")[0].replace("\n", " "), "predictor": "restated_zon", "dt_days": meta["dt_days"], "starts": list(STARTS), "leads_years": list(LEADS),
            "model_fingerprint": meta["model_fingerprint"], "truth": "JPL DE440 (de440s.bsp)", "run_ledger": "misc/CANDIDATE_LEDGER.md C23b",
            "control_km": control, "without_layer_km": layers}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("D:/datasets/sigma_runs"))
    ap.add_argument("--check", action="store_true", help="rebuild and compare with the stored fixture instead of writing")
    a = ap.parse_args()
    table = build(a.runs)
    if a.check:
        same = json.loads(OUT.read_text(encoding="utf-8")) == json.loads(json.dumps(table))
        print("fixture matches the runs" if same else "fixture DIFFERS from the runs")
        return 0 if same else 1
    OUT.write_text(json.dumps(table, indent=1), encoding="utf-8")
    print("wrote", OUT, "bodies:", len(table["control_km"]), "layers:", len(table["without_layer_km"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
