"""Write src/gametest/resources/records.json: one matter record per material the client test draws, from sigma-ground."""
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent))
import matter  # noqa: E402

PITCH, V = 0.05, 0.05 ** 3
rec = {
    "gold": matter.record("gold", 19300 * V, PITCH),
    "oak": matter.record("wood_oak", 700 * V, PITCH),
    "glass": matter.record("glass", 2500 * V, PITCH),
    "water": matter.record("water", 1000 * V, PITCH),
    "hot_tungsten": matter.record("tungsten", 19300 * V, PITCH, 2400.0, t=1.25, v=(0.0, -2.0, 0.0)),
    "warm_tungsten": matter.record("tungsten", 19300 * V, PITCH, 1500.0),
}
out = here / "src" / "gametest" / "resources" / "records.json"
out.write_text(json.dumps({k: matter.to_wire(v) for k, v in rec.items()}, indent=1), encoding="utf-8")
for k, v in rec.items():
    print(f"{k:14s} rgba={v['rgba']} light={v['light']} opacity_physical={v['opacity_physical']:.3f}")
