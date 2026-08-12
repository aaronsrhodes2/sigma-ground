"""measure_compile_time.py -- R4.7 CSG leaf-count shader compile-time sweep.

Closes the gap misc/MPDR_CAPABILITY_ENVELOPE.md's own S7 flagged and never
built: real compile-time-vs-leaf-count data, plus an A/B/C component
isolation pass (does the SDF/compose/matsel axis or the material-chain axis
actually drive the measured worse-than-linear blowup?) instead of estimating
from character counts alone.

Reuses render_regression.py's _CDPBrowser/_start_server/_free_port/Infra
directly (the tools/measure_faceting.py precedent), and gen_shader.node.js
(the R4 SVO-investigation instrument, unmodified) to dump real generated
shader text with zero browser cost. compile_probe.html (this dir's sibling
to svo_bisect.html) does the actual isolated compile+link timing.

Deliberate departure from render_regression.py's OWN usage pattern: this
tool spins up a FRESH _CDPBrowser per (n, variant, rep) rather than reusing
one across a run -- a wedged ANGLE compile poisons the whole GPU process,
not just its tab (svo_bisect.html's own measured lesson, session 32).
A timed-out probe is recorded as a first-class "timeout" result, never
silently dropped.

Run:
    python tools/measure_compile_time.py --sweep
    python tools/measure_compile_time.py --sweep --leaves 11,22,44,88 --reps 5
    python tools/measure_compile_time.py --report      # re-print the last campaign
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics
import subprocess
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

import render_regression as rr                                # noqa: E402

_WEB = _ROOT / "sigma_ground" / "radiance" / "web"
_DATA = _WEB / "data"
_SCRATCH = _WEB / "_compile_scratch"          # servable by serve.py's static root; gitignored
_BROWSER_REPORT = _ROOT / "misc" / "BROWSER_REPORT.json"
_REPORT_PATH = _ROOT / "misc" / "COMPILE_TIME_REPORT.json"
_GEN_SHADER_NODE = _ROOT / "tools" / "gen_shader.node.js"
_GEN_SCENE_PY = _ROOT / "tools" / "gen_leaf_stress_scene.py"

DEFAULT_LEAVES = [4, 8, 11, 16, 22, 32, 44, 64, 88, 128, 176]
DEFAULT_VARIANTS = ["full", "stub_matchains", "stub_sdf"]
# stub variants only run at these checkpoints -- bounding total wall-clock
# (a full ladder x 3 variants x reps x fresh-Chrome-per-point is genuinely
# slow); disclosed here and printed at run time, never silently narrowed.
STUB_CHECKPOINTS = {11, 22, 44, 88}

MATCHAIN_FNS = {
    "matColor": "return vec3(0.72);",
    "metalness": "return 0.0;",
    "emissivityOf": "return vec3(0.5);",
    "matTempK": "return 293.0;",
    "bodyTempK": "return -1.0;",
    "iorOf": "return -1.0;",
    "opacityOf": "return 1.0;",
}


# ── brace-balanced function extraction (colorfn-style chains have no nested
# braces, but mapD's decls can -- e.g. bvhCull's `if(...){...} else {...}` --
# so a naive "first }" regex is wrong for mapD specifically; this handles
# both correctly) ────────────────────────────────────────────────────────
def _balanced_end(src: str, open_idx: int) -> int:
    depth = 0
    i = open_idx
    n = len(src)
    while i < n:
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced braces from index %d" % open_idx)


def _function_span(src: str, name: str):
    m = re.search(re.escape(name) + r"\s*\([^)]*\)\s*\{", src)
    if not m:
        raise ValueError(f"function {name!r} not found in shader text")
    body_open = m.end() - 1
    body_end = _balanced_end(src, body_open)
    return m.start(), body_open, body_end


def _stub_function(src: str, name: str, stub_return: str) -> str:
    sig_start, body_open, body_end = _function_span(src, name)
    sig = src[sig_start:body_open]
    return src[:sig_start] + sig + "{ " + stub_return + " }" + src[body_end:]


def make_variant(pt_src: str, variant: str) -> str:
    if variant == "full":
        return pt_src
    if variant == "stub_matchains":
        out = pt_src
        for fn, ret in MATCHAIN_FNS.items():
            out = _stub_function(out, fn, ret)
        return out
    if variant == "stub_sdf":
        return _stub_function(pt_src, "mapD", "mat=-1; return 1e9;")
    if variant == "stub_both":
        out = make_variant(pt_src, "stub_matchains")
        return _stub_function(out, "mapD", "mat=-1; return 1e9;")
    raise ValueError(f"unknown variant {variant!r}")


# ── scene/shader generation ─────────────────────────────────────────────
def _gen_scenes(leaves):
    subprocess.run([sys.executable, str(_GEN_SCENE_PY), "--n",
                     ",".join(str(n) for n in leaves)],
                    check=True, cwd=str(_ROOT))


def _dump_shader(scene_path: pathlib.Path, out_prefix: pathlib.Path) -> pathlib.Path:
    r = subprocess.run(["node", str(_GEN_SHADER_NODE), str(scene_path), str(out_prefix), "svo=0"],
                        cwd=str(_ROOT), capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise rr.Infra(f"gen_shader.node.js failed for {scene_path}: {r.stderr[:300]}")
    return pathlib.Path(str(out_prefix) + "_pt.glsl")


# ── browser-side probe ──────────────────────────────────────────────────
def _read_browser_report() -> dict:
    try:
        d = json.loads(_BROWSER_REPORT.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _run_probe(port: int, cdp: "rr._CDPBrowser", glsl_path: pathlib.Path,
               probe_id: str, timeout: float) -> dict:
    slug = f"compileprobe_{probe_id}"
    before = _read_browser_report().get(slug)
    rel = glsl_path.relative_to(_WEB).as_posix()
    url = f"http://127.0.0.1:{port}/compile_probe.html?f=/{rel}&id={probe_id}"
    cdp.navigate(url)
    t0 = time.time()
    while time.time() - t0 < timeout:
        cur = _read_browser_report().get(slug)
        if cur is not None and cur != before:
            return cur
        time.sleep(0.3)
    return {"ok": False, "timeout": True, "compile_ms": None, "link_ms": None, "total_ms": None}


# ── sweep driver ─────────────────────────────────────────────────────────
def run_sweep(leaves, variants, reps, timeout, port_arg):
    _SCRATCH.mkdir(parents=True, exist_ok=True)
    _gen_scenes(leaves)
    points = []
    for n in leaves:
        scene_path = _DATA / f"mpdr_leafsweep_n{n}.json"
        pt_path = _dump_shader(scene_path, _SCRATCH / f"n{n}")
        pt_src = pt_path.read_text(encoding="utf-8")
        active = [v for v in variants if v == "full" or n in STUB_CHECKPOINTS]
        for variant in active:
            variant_src = make_variant(pt_src, variant)
            variant_path = _SCRATCH / f"n{n}_{variant}.glsl"
            variant_path.write_text(variant_src, encoding="utf-8")
            samples, any_timeout = [], False
            for rep in range(reps):
                port = port_arg or rr._free_port()
                proc = rr._start_server(port)
                cdp = rr._CDPBrowser()
                try:
                    result = _run_probe(port, cdp, variant_path, f"n{n}_{variant}_{rep}", timeout)
                finally:
                    cdp.close()
                    proc.terminate()
                if result.get("timeout") or not result.get("ok"):
                    any_timeout = True
                    print(f"  n={n:>4} {variant:<15} rep={rep}: "
                          f"{'TIMEOUT' if result.get('timeout') else 'FAIL'} "
                          f"{str(result.get('log',''))[:100]}")
                    continue
                samples.append(result)
                print(f"  n={n:>4} {variant:<15} rep={rep}: "
                      f"compile={result['compile_ms']}ms link={result['link_ms']}ms")
            point = {"n": n, "variant": variant, "src_len": len(variant_src), "timeout": any_timeout}
            if samples:
                for key in ("compile_ms", "link_ms", "total_ms"):
                    vals = [s[key] for s in samples if s.get(key) is not None]
                    if vals:
                        point[key] = {"median": statistics.median(vals), "min": min(vals),
                                      "max": max(vals), "reps": len(vals)}
                point["khr_parallel_ext"] = samples[0].get("khr_parallel_ext")
                point["khr_max_threads"] = samples[0].get("khr_max_threads")
            points.append(point)
    return points


def _fit_power_law(points, variant, metric="total_ms"):
    xs, ys = [], []
    for p in points:
        if p["variant"] != variant or metric not in p:
            continue
        v = p[metric]["median"]
        if v and v > 0 and p["n"] > 0:
            xs.append(math.log(p["n"]))
            ys.append(math.log(v))
    if len(xs) < 2:
        return None
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    yhat = [slope * x + intercept for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot
    return {"exponent": round(slope, 3), "r2": round(r2, 4)}


def save_campaign(points, sweep_meta):
    hist = []
    if _REPORT_PATH.exists():
        try:
            hist = json.loads(_REPORT_PATH.read_text(encoding="utf-8"))
            if not isinstance(hist, list):
                hist = []
        except Exception:
            hist = []
    fits = {}
    for variant in DEFAULT_VARIANTS + ["stub_both"]:
        f = _fit_power_law(points, variant)
        if f:
            fits[variant] = f
    campaign = {"date": None, "env": {"platform": sys.platform},
                "sweep": sweep_meta, "points": points, "fits": fits}
    hist.append(campaign)
    hist = hist[-20:]
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(json.dumps(hist, indent=1), encoding="utf-8")
    return campaign


def _print_campaign(campaign):
    print(f"\n{'n':>5} {'variant':<16} {'compile_ms':>12} {'link_ms':>10} "
          f"{'total_ms':>10} {'src_len':>8}")
    for p in campaign["points"]:
        cm = p.get("compile_ms", {}).get("median") if p.get("compile_ms") else None
        lm = p.get("link_ms", {}).get("median") if p.get("link_ms") else None
        tm = p.get("total_ms", {}).get("median") if p.get("total_ms") else None
        tag = "TIMEOUT" if p.get("timeout") and cm is None else ""
        print(f"{p['n']:>5} {p['variant']:<16} {str(cm or tag):>12} {str(lm or ''):>10} "
              f"{str(tm or ''):>10} {p['src_len']:>8}")
    if campaign.get("fits"):
        print("\nfits (ms ~= a * n^exponent, on total_ms):")
        for variant, f in campaign["fits"].items():
            print(f"  {variant:<16} exponent={f['exponent']:.3f}  r2={f['r2']:.4f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", action="store_true", help="run a full measurement campaign")
    ap.add_argument("--report", action="store_true", help="re-print the last saved campaign")
    ap.add_argument("--leaves", default=",".join(str(n) for n in DEFAULT_LEAVES))
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args()

    if args.report:
        if not _REPORT_PATH.exists():
            print("no campaigns recorded yet -- run --sweep first")
            return
        hist = json.loads(_REPORT_PATH.read_text(encoding="utf-8"))
        if not hist:
            print("no campaigns recorded yet -- run --sweep first")
            return
        _print_campaign(hist[-1])
        return

    if not args.sweep:
        ap.error("pass --sweep to run a measurement campaign, or --report to print the last one")

    leaves = [int(x) for x in args.leaves.split(",") if x.strip()]
    variants = [v for v in args.variants.split(",") if v.strip()]
    print(f"sweep: leaves={leaves}")
    print(f"       variants={variants}, reps={args.reps}, timeout={args.timeout:.0f}s")
    print(f"       stub variants run only at n={sorted(STUB_CHECKPOINTS)} "
          "(bounds total wall-clock, disclosed not silent)")
    t0 = time.time()
    points = run_sweep(leaves, variants, args.reps, args.timeout, args.port)
    campaign = save_campaign(points, {"leaves": leaves, "variants": variants,
                                       "reps": args.reps, "timeout_s": args.timeout})
    print(f"\ncampaign complete in {time.time()-t0:.1f}s, written to {_REPORT_PATH}")
    _print_campaign(campaign)


if __name__ == "__main__":
    main()
