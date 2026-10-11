"""Photograph the transporter effect in the real game (a visible window, about one to two minutes per scenario).

    python stage/gmod/holodeck/transporter_shots.py success    # a boombox arriving, pictures at real progress 0 / 25 / 50 / packing / sending / arrival / done
    python stage/gmod/holodeck/transporter_shots.py failure    # a cannon whose download stalls (then times out) and a fire hydrant whose download breaks
    python stage/gmod/holodeck/transporter_shots.py avatar     # Skippy leaving and arriving
    python stage/gmod/holodeck/transporter_shots.py perf       # milliseconds per frame, beams on / plain boxes / nothing

The bridge here is the real one (real inventory, real packing); only the DOWNLOAD is replaced: bytes come from the local asset cache (the
boombox) or are made up (the cannon, the hydrant, which never get decoded) and are delivered slowly, like a slow link, so that the
pipeline spends long enough in each stage to be photographed. The progress counters, /progress, the chunks, the game's heartbeat and the
effect are all the production code. Pictures: stage/gmod/holodeck/data/shots/transporter_<scenario>_*.jpg (gitignored) and a contact sheet.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments"))
sys.path.insert(0, str(HERE.parents[1]))
import run as exp_run  # noqa: E402
import run_tool  # noqa: E402
import skippy_bridge  # noqa: E402

CACHE = HERE.parents[1] / "skippyverse" / "cache" / "assets"
STOP = threading.Event()


class SlowFetch:
    """A fetch(url, *, max_bytes) that stands in for the download: a Poly Haven /files listing and the files it names, delivered at a
    throttled rate while the bridge's own progress counter is told every slice (what bounded_fetch does for a real socket)."""

    def __init__(self, plan: dict[str, dict], slice_bytes: int = 12_000, slice_s: float = 0.4):
        self.plan, self.slice_bytes, self.slice_s = plan, slice_bytes, slice_s
        self.delivered: dict[str, int] = {}

    def _files(self, pid: str):
        cached = CACHE / f"found.poly-haven.{pid.replace('_', '-')}"
        if cached.is_dir() and (cached / f"{pid}_1k.gltf").exists():
            return (cached / f"{pid}_1k.gltf").read_bytes(), (cached / f"{pid}.bin").read_bytes()
        return b'{"asset":{"version":"2.0"}}' + b" " * 5000, os.urandom(400_000)        # made up: never decoded (these downloads end in failure)

    def __call__(self, url, *, max_bytes):
        from urllib.parse import urlsplit
        path = urlsplit(url).path
        if url.startswith("https://api.polyhaven.com/files/"):
            pid = path.rsplit("/", 1)[-1]
            gl, bn = self._files(pid)
            doc = {"gltf": {"1k": {"gltf": {"url": f"https://dl.example/{pid}_1k.gltf", "size": len(gl), "include": {
                f"{pid}.bin": {"url": f"https://dl.example/{pid}.bin", "size": len(bn)}}}}}}
            return json.dumps(doc).encode()
        pid = path.rsplit("/", 1)[-1].replace("_1k.gltf", "").replace(".bin", "")
        gl, bn = self._files(pid)
        data = gl if url.endswith(".gltf") else bn
        card = f"found.poly-haven.{pid.replace('_', '-')}"
        prog = skippy_bridge.MESH_PROGRESS.get(skippy_bridge.mesh_key(card))
        plan = self.plan.get(pid, {"mode": "ok"})
        sent = 0
        while sent < len(data):
            if STOP.is_set():
                raise TimeoutError("the run ended")
            n = min(self.slice_bytes, len(data) - sent)
            sent += n
            total = self.delivered.get(pid, 0) + n
            self.delivered[pid] = total
            if prog is not None:
                prog.fetching += n
            if plan["mode"] == "error" and total >= plan["after"]:
                raise ConnectionResetError("connection reset by peer (made to happen for the photograph)")
            if plan["mode"] == "stall" and total >= plan["after"]:
                while not STOP.is_set():                                 # the link has gone quiet: nothing more arrives
                    time.sleep(0.2)
                raise TimeoutError("the run ended")
            time.sleep(self.slice_s)
        return data


def tailscale_ip() -> str:
    from skippy_e2e import tailscale_ip as t
    return t()


def main(argv=None) -> int:
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=["success", "failure", "avatar", "perf"])
    ap.add_argument("--out", default=str(HERE / "data" / "shots"))
    ap.add_argument("--map", default="gm_flatgrass")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--slice-bytes", type=int, default=12_000)
    ap.add_argument("--slice-s", type=float, default=0.4)
    a = ap.parse_args(argv)
    os.environ.setdefault("GMOD_EXTRA_ARGS", "+maxplayers 4")
    targs: dict = {"scenario": a.scenario}
    srv = None
    tmp = None
    if a.scenario != "avatar":
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="tp-assets-"))
        skippy_bridge.TOKEN = secrets.token_hex(8)
        skippy_bridge.warm = lambda: None
        skippy_bridge.MESH_CACHE_DIR = tmp                                    # empty: every model is "downloaded"
        plan = {}
        if a.scenario == "failure":
            plan = {"fire_hydrant": {"mode": "error", "after": 160_000}, "cannon_01": {"mode": "stall", "after": 110_000}}
            targs.update(timeout=18, stall=6)
        if a.scenario == "perf":
            plan = {"boombox": {"mode": "stall", "after": 150_000}}              # never finishes: the beams stay up while the frames are timed
            targs.update(timeout=120)
        if a.scenario == "success":
            targs.update(chunk=500, pump=0.4, timeout=120)
        skippy_bridge.FOUND_FETCHER = SlowFetch(plan, a.slice_bytes, a.slice_s)
        skippy_bridge.found_index()
        host = os.environ.get("SKIPPY_BRIDGE_HOST") or tailscale_ip()
        srv = skippy_bridge.serve(host, 8777)
        targs.update(bridge=f"http://{host}:8777", token=skippy_bridge.TOKEN)
    else:
        subprocess.run([exp_run._bash(), str(HERE.parent / "install_skippy_body.sh").replace("\\", "/")], stdin=subprocess.DEVNULL)
    for old in exp_run.DATA.glob("shot_tp_*.jpg"):
        old.unlink()
    try:
        res = run_tool.run_tool("transporter_shots", targs, client=True, visible=True, map_name=a.map, timeout=300)
    finally:
        STOP.set()
        if srv is not None:
            srv.shutdown()
            srv.server_close()
    print(json.dumps(res, indent=1))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for f in res.get("frames", []):
        src = exp_run.DATA / f"shot_tp_{f['name']}.jpg"
        if not src.exists():
            print("missing picture:", f["name"])
            continue
        im = Image.open(src).convert("RGB")
        im = im.resize((a.width, int(im.height * a.width / im.width)))
        dest = out / f"transporter_{a.scenario}_{f['name']}.jpg"
        im.save(dest, quality=88)
        print(dest, f"stage={f['stage']} progress={f['progress']:.3f} stalled={f['stalled']:.1f} cam={[round(v) for v in f.get('cam', [])]}")
        tiles.append(im)
    if tiles:
        cols = 3
        w = 560
        sm = [t.resize((w, int(t.height * w / t.width))) for t in tiles]
        rows = (len(sm) + cols - 1) // cols
        sheet = Image.new("RGB", (w * cols, sm[0].height * rows), (20, 20, 20))
        for i, im in enumerate(sm):
            sheet.paste(im, ((i % cols) * w, (i // cols) * sm[0].height))
        sheet.save(out / f"transporter_{a.scenario}_sheet.jpg", quality=88)
        print(out / f"transporter_{a.scenario}_sheet.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
