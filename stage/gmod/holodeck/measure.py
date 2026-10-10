"""Measure every model named in catalog_src.py inside the engine and save the raw answers.

    python stage/gmod/holodeck/measure.py            # dedicated server, ~30 s
Writes stage/gmod/holodeck/data/measured.json and prints a table of what each model really is (metres)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import catalog_src  # noqa: E402
import run_tool  # noqa: E402

HU_M = 0.01905
OUT = HERE / "data" / "measured.json"


def model_paths() -> list[str]:
    seen, out = set(), []
    for k in catalog_src.KINDS:
        for v in k["variants"]:
            p = v if isinstance(v, str) else v["model"]
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def main() -> int:
    paths = model_paths()
    res = run_tool.run_tool("measure_models", {"models": paths}, timeout=240)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"hu_m": HU_M, "models": res["models"]}, indent=1), encoding="utf-8")
    by = {m["model"]: m for m in res["models"]}
    print(f"{len(paths)} models asked, wrote {OUT}")
    for k in catalog_src.KINDS:
        print(f"\n{k['kind']}  (real width {k['size_m'][0]}-{k['size_m'][1]} m)")
        for v in k["variants"]:
            p = v if isinstance(v, str) else v["model"]
            m = by.get(p, {})
            short = p.replace("models/", "")
            if m.get("missing"):
                print(f"   MISSING      {short}")
            elif m.get("no_physics"):
                print(f"   NO PHYSICS   {short}")
            else:
                mn, mx = m["mins"], m["maxs"]
                dx, dy, dz = ((mx[i] - mn[i]) * HU_M for i in range(3))
                f = m.get("facing") or {}
                print(f"   {dx:5.2f} x {dy:5.2f} x {dz:5.2f} m  {m['mass']:7.1f} kg  {m.get('material', '?'):12s} back={f.get('back', '-'):>2s} "
                      f"conf={f.get('confidence', 0):.2f} skins={m['skins']:2d} phys={m['phys_objects']}  {short}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
