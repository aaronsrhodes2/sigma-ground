"""Write src/gametest/resources/slosh_frames.json: real theaters (the same Theater, in matter mode: slosh at chosen sim times, and the melt
scene's hot plate) as full matter frames (record palette + runs) shifted to y = 200, z <= 0, for the headless client test."""
import json
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent))
import theater  # noqa: E402

TIMES = [float(t) for t in (sys.argv[1:] or ["0.0", "3.0"])]
ORIGIN = (0, -60, 100)
DY, DZ = 200 - ORIGIN[1], -ORIGIN[2]


class Capture:
    def __init__(self):
        self.frames = []

    def send_matter(self, records, runs):
        self.frames.append((list(records), list(runs)))
        return (0, 0)


def full_frame(world, looks):
    cap = Capture()
    theater.Theater(world, looks, ORIGIN, lambda c: None, cap, matter=True).draw()
    records, runs = cap.frames[0]
    return records, [[x, y + DY, z + DZ, n, i] for x, y, z, n, i in runs if records[i]]       # air needs no block here


def main():
    looks, _ = theater.shared_looks()
    out = []
    for scene, times in (("slosh", TIMES), ("melt", [1.0])):
        _, world = theater.build(scene)
        for t in times:
            while world.t < t - 1e-9:
                world.step(0.04 if scene == "slosh" else 0.05)
            records, runs = full_frame(world, looks)
            xs = [r[0] for r in runs] + [r[0] + r[3] - 1 for r in runs]
            ys, zs = [r[1] for r in runs], [r[2] for r in runs]
            out.append({"scene": scene, "t": float(world.t), "records": records, "runs": runs,
                        "bbox": [min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)]})
            print(f"{scene} t={world.t:.2f} s: {sum(r[3] for r in runs)} blocks, {len(records)} records", flush=True)
    (here / "src" / "gametest" / "resources" / "slosh_frames.json").write_text(json.dumps(out), encoding="utf-8")


main()
