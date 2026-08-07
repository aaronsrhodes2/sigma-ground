"""R0 — the pixel-regression net for the Radiance web viewer.

Extends the house ?probe=1 idiom (browser_gauntlet.py pattern: spawn serve.py,
drive the REAL default browser via webbrowser.open, wait per scene): capture
mode (?probe=1&capture=<id>) renders one PINNED deterministic exposure and
POSTs the framebuffer to serve.py's /capture sink, which canonicalizes it
(top-down rows, alpha stripped) into misc/render_captures/<id>.png + <id>.json.
Playwright/puppeteer were REJECTED: absent from the zero-dependency repo, and
a headless GPU context is a different renderer fingerprint than the pipeline
the Captain actually looks at.

    python tools/render_regression.py --capture [--only id,..] [--force] [--reason txt]
    python tools/render_regression.py --check   [--only id,..] [--allow-foreign-env]
    python tools/render_regression.py --selftest [--id opacity_cup]
    python tools/render_regression.py --noise --reps 5 [--only id,..]
    python tools/render_regression.py --derive-cam <slug> <body_label> [--t sec]

--capture renders every baseline THREE times back-to-back and only marks
deterministic:true when all three SHA-256s match (measured, never assumed —
and a 2-render proof measurably mislabeled 3 of the first 11 baselines, each
demoted after a later 1-LSB flicker). It REFUSES
to baseline a scene whose probe selfcheck failed or that reported console
errors — a baseline of a broken render institutionalizes the breakage.

--check: Tier 1 (authoritative) = exact sha256 vs manifest, applied when the
environment fingerprint (GPU renderer string + browser UA) matches AND the
entry measured deterministic:true. Tier 2 (advisory, foreign env or
non-deterministic entry) = statistical gates whose thresholds + justification
live in the MANIFEST, not here. A changed data/<slug>.json is reported as
"input changed", a distinct verdict from "renderer regressed" (scene_sha256).

Exit codes: 0 = pass · 1 = pixel regression / missing baseline / input
changed · 2 = infrastructure (server death, timeout, capture refusal) —
NEVER conflated with a regression.

PERTURBATION DRILL (the acceptance procedure for the net itself, run after
any change to the net): edit ONE shader constant in viewer.js — e.g. the 1.7
tone-map gain (`return e/(1.0+max(e.r,max(e.g,e.b)))*1.7;`) — then --check
must FAIL loudly naming the scenes, with diff artifacts under
misc/render_regression/; `git restore` viewer.js; --check must pass again.

Ergonomics (accepted, the gauntlet's): webbrowser.open accumulates tabs —
close them afterwards; keep the browser window VISIBLE (a minimized window
throttles requestAnimationFrame and the run times out with exit 2, never a
false FAIL). Note: capture tabs also POST ?probe=1 reports, so
misc/BROWSER_REPORT.json gets touched — the gauntlet resets it per run.
Baselines are PER-(GPU, driver, browser/ANGLE) artifacts: a driver or browser
update legitimately breaks exact hashes — eyeball the diffs, recapture with
--force + a dated reason.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from sigma_ground.radiance import png_io                       # noqa: E402

_DATA = _ROOT / "sigma_ground" / "radiance" / "web" / "data"
_BASE = _ROOT / "sigma_ground" / "radiance" / "web" / "baselines"
_SINK = _ROOT / "misc" / "render_captures"
_FAIL = _ROOT / "misc" / "render_regression"
_MANIFEST = _BASE / "manifest.json"

_PX = (640, 360)
_NMAX_WATER = 24        # viewer.js NMAX — an animated scene's accN plateaus here

# Default Tier-2 tolerances, written INTO the manifest at --capture (--check
# reads them back from there; they are data, not code). Mechanistic
# justification: pinned seeds make the Monte-Carlo sample set IDENTICAL across
# environments, so residuals are (i) float-rounding / mad-contraction /
# transcendental differences — 1-2 LSBs post-gamma, bounded by mean+p99 — and
# (ii) discrete sphere-trace hit/miss flips at silhouettes — an O(edge-length)
# pixel set, bounded by changed-fraction. Per-pixel max-diff is deliberately
# NOT a gate: one legitimate silhouette flip is full-scale.
_TOL = {"changed_frac": 0.002, "mean_abs": 0.10, "p99_abs": 3,
        "justification": ("gates >= 3x (actually 6-10x) the worst pairwise "
                          "noise MEASURED by --noise on the capture machine "
                          "(worst: changed>4 = 0.0304%, mean 0.0101, p99 0); "
                          "tightened from the blueprint's a-priori 0.01/2.0/12 "
                          "after a review perturbation drill showed a real "
                          "all-|d|<=4 renderer change slipping those gates on "
                          "every Tier-2 entry; re-derive with --noise whenever "
                          "an entry measures non-deterministic")}

# ── the baseline set (blueprint §5): every renderer subsystem once ─────────
# cam = [az, el, radius, tx, ty, tz] — EXPLICIT floats (locateBody is never
# called in capture mode: its framing heuristics will evolve in R3+, and a
# committed camera must be frozen). Values are the scene's own defaults except
# solar_earth, minted ONCE by `--derive-cam solar_system_theater Earth`
# (target = Earth's frame-0 pose, radius = 3.2 x radius_m, sun-facing az/el).
# t = absolute sim seconds (None = leave simTime at 0).
BASELINES = {
    "solar_wide": dict(
        slug="solar_system_theater", frames=64, t=None,
        cam=[0.6, 0.5, 329115315540.0, 0.0, 0.0, 0.0],
        reason="far-scale anchor: DE440 scale, R1-lite rebase, 30 bodies "
               "(9 Voxel + 21 Sphere), Sun NEE — gates R3/R4/R5 far-field"),
    "solar_earth": dict(
        slug="solar_system_theater", frames=64, t=None,
        cam=[-2.947777057667831, -0.40217876364702315, 20387200.0,
             -26531002415.56548, 132064399546.2592, 57268703374.95859],
        reason="near-field precision inside far scale (the regime R1-lite "
               "exists for); the most sensitive R4 brick-grid tripwire"),
    "table_stack": dict(
        slug="scene_doughnut_on_plate_on_1700s_table", frames=64, t=None,
        cam=[0.7, 0.16, 1.3678032407861598, 0.0, 0.0, 0.5606233650179518],
        reason="48 csg_leaves (repo maximum): deep CSG chains + the bvhCull "
               "thin-leaf history"),
    "clock_mid": dict(
        slug="clock", frames=64, t=30.0005,
        cam=[0.15, 0.12, 0.5592, 0.149, -0.06854271422008813, 0.0],
        reason="involute-gear GLSL (x9, heaviest analytic SDF), Clipped "
               "wrappers, trajectory interpolation + quaternion nlerp"),
    "iron_cube_irt_mid": dict(
        slug="hot_iron_cube_on_a_copper_slab_irt", frames=64, t=60.0,
        cam=[0.7, 0.16, 0.24, 0.0, 0.0, 0.05500000000000001],
        reason="the ONLY per-cell field scene (uFldA/B/U 3-D texture path) + "
               ">700 K incandescence + NEE; animated mid-scrub representative"),
    "water_shutter": dict(
        slug="water", frames=24, t=None,          # 24 == NMAX plateau first-arrival
        cam=[0.6, 0.1, 2.4, 0.0, 0.02, 0.0],
        reason="waveH/waveGrad + deterministic motion blur under the virtual "
               "clock + the bounded-shutter integration branch itself"),
    "flagship_glass": dict(
        slug="flagship_glass", frames=64, t=None,
        cam=[0.6, 0.28, 0.2856, 0.0, 0.0, 0.06],
        reason="dielectric Fresnel/Snell/TIR incl. the rnd() roulette draw "
               "(10 leaves, pt_env key)"),
    "opacity_cup": dict(
        slug="opacity_demo_coffee_cup", frames=64, t=None,
        cam=[0.8, 0.45, 0.19205272713502405, 0.0, 0.0, 0.0475],  # az/el = viewer defaults (scene ships none)
        reason="stochastic alpha gate — a distinct rnd() consumer"),
    "vox_chair": dict(
        slug="vox_chair", frames=64, t=0.0,
        cam=[0.35, 0.22, 4.0, 0.0, 1.25, 0.0],
        reason="ShapeNet real-mesh Voxel leaf (int8 SDF 3-D texture) + a "
               "32-frame trajectory, at t=0"),
    "ring_sph_mid": dict(
        slug="phase_e_ring_sph", frames=64, t=9286.57923384884,
        cam=[0.5, 0.55, 32703.475171255293, 0.0, 0.0, 0.0],
        reason="32-body pose-array stress (MAXB), drop-family representative"),
    "plate_push_mid": dict(
        slug="plate_pushed_off_the_1700s_table_v30", frames=64, t=0.50155,
        cam=[0.5, 0.25, 2.2, 0.44999999999999996, 0.0, 0.5712859650179519],
        reason="39 leaves incl. Torus/Clipped/Ellipsoid x24 (shapes no other "
               "pick covers), 122 frames, follow-up-prompt flagship family"),
}


class Infra(RuntimeError):
    """Infrastructure failure (server/timeout/refusal) — exit 2, never 1."""


def _f(x: float) -> str:
    return format(x, ".17g")                       # round-trips through parseFloat


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _scene_sha(slug: str) -> str:
    return hashlib.sha256((_DATA / f"{slug}.json").read_bytes()).hexdigest()


def _has_water(slug: str) -> bool:
    try:
        o = json.loads((_DATA / f"{slug}.json").read_text(encoding="utf-8"))
        sc = o.get("scene") if o.get("kind") == "trajectory" else o
        return any((l.get("shape") or {}).get("type") == "Water"
                   for l in (sc.get("csg_leaves") or []))
    except Exception:
        return False


# ── server lifecycle (browser_gauntlet pattern; EPHEMERAL port by default —
# 8765 is the interactive port and stale servers shadow it) ─────────────────
def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_server(port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "sigma_ground.radiance.web.serve", str(port)],
        cwd=str(_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < 15.0:
        if proc.poll() is not None:
            raise Infra(f"serve.py exited immediately (code {proc.returncode})")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2):
                return proc
        except Exception:
            time.sleep(0.25)
    proc.terminate()
    raise Infra("serve.py did not come up within 15 s")


def _capture_once(port: int, bid: str, spec: dict, frames: int | None = None,
                  timeout: float = 120.0, proc: subprocess.Popen | None = None
                  ) -> tuple:
    """Open one capture tab, wait for the sink files → (meta, png_bytes)."""
    frames = spec["frames"] if frames is None else frames
    if _has_water(spec["slug"]) and frames > _NMAX_WATER:
        raise Infra(f"{bid}: frames={frames} > NMAX={_NMAX_WATER} for a Water "
                    f"scene — accN plateaus and the capture would never fire")
    jpath, ppath = _SINK / f"{bid}.json", _SINK / f"{bid}.png"
    _SINK.mkdir(parents=True, exist_ok=True)
    for p in (jpath, ppath):
        if p.exists():
            p.unlink()
    url = (f"http://127.0.0.1:{port}/?scene={spec['slug']}&probe=1"
           f"&capture={bid}&px={_PX[0]}x{_PX[1]}&frames={frames}"
           f"&cam={','.join(_f(v) for v in spec['cam'])}")
    if spec.get("t") is not None:
        url += f"&t={_f(spec['t'])}"
    if spec.get("extra"):                          # R4.4: e.g. "svo=0" A/B evidence
        url += "&" + spec["extra"]
    webbrowser.open(url)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise Infra(f"{bid}: serve.py died mid-capture "
                        f"(code {proc.returncode})")
        if jpath.exists():
            try:
                meta = json.loads(jpath.read_text(encoding="utf-8"))
            except Exception:
                time.sleep(0.2)                    # mid-write — retry
                continue
            if meta.get("error"):
                raise Infra(f"{bid}: viewer refused capture — {meta['error']}")
            if meta.get("sha256_rgb"):
                return meta, ppath.read_bytes()
        time.sleep(0.4)
    raise Infra(f"{bid}: no capture report in {timeout:.0f} s (browser window "
                f"minimized? scene failed to load? see misc/BROWSER_REPORT.json)")


def _capture_retry(port, bid, spec, frames=None, timeout=120.0, proc=None):
    """_capture_once with ONE retry on a no-report timeout (a cold shader
    compile or tab-open hiccup), never on an explicit viewer refusal."""
    try:
        return _capture_once(port, bid, spec, frames=frames,
                             timeout=timeout, proc=proc)
    except Infra as e:
        if "no capture report" not in str(e):
            raise
        print(f"  {bid}: timed out once - retrying with a fresh tab ...",
              flush=True)
        return _capture_once(port, bid, spec, frames=frames,
                             timeout=timeout, proc=proc)


def _healthy(bid: str, meta: dict) -> None:
    """Refuse to trust a capture whose in-page checks failed."""
    if meta.get("selfcheck_pass") is False:
        raise Infra(f"{bid}: in-page SDF selfcheck FAILED — refusing to "
                    f"baseline/compare a broken render")
    errs = meta.get("console_errors") or []
    if errs:
        raise Infra(f"{bid}: console errors during render — {errs[:3]}")
    if [meta.get("w"), meta.get("h")] != list(_PX):
        raise Infra(f"{bid}: captured {meta.get('w')}x{meta.get('h')}, "
                    f"expected {_PX[0]}x{_PX[1]} — the px pin failed")


# ── comparison (blueprint §4) ───────────────────────────────────────────────
def _metrics(got: bytes, exp: bytes) -> dict:
    diffs = [abs(a - b) for a, b in zip(got, exp)]
    n = len(diffs)
    mean_abs = sum(diffs) / n
    pmax = [max(diffs[i], diffs[i+1], diffs[i+2]) for i in range(0, n, 3)]
    changed = sum(1 for v in pmax if v > 4) / len(pmax)
    p99 = sorted(pmax)[int(0.99 * (len(pmax) - 1))]
    return {"changed_frac": changed, "mean_abs": mean_abs,
            "p99_abs": p99, "max_abs": max(pmax)}


def _write_fail_artifacts(bid: str, w: int, h: int, got: bytes, exp: bytes) -> None:
    _FAIL.mkdir(parents=True, exist_ok=True)
    png_io.write_rgb_png(str(_FAIL / f"{bid}_got.png"), w, h, got)
    diff = bytes(min(255, abs(a - b) * 8) for a, b in zip(got, exp))  # x8 amplified
    png_io.write_rgb_png(str(_FAIL / f"{bid}_diff.png"), w, h, diff)


def _fingerprint(meta: dict) -> tuple:
    return (meta.get("renderer") or "?", meta.get("ua") or "?")


def _load_manifest() -> dict:
    if not _MANIFEST.exists():
        raise Infra(f"no baselines manifest at {_MANIFEST} — run --capture first")
    return json.loads(_MANIFEST.read_text(encoding="utf-8"))


def _select(only: str | None) -> dict:
    if not only:
        return dict(BASELINES)
    ids = [s.strip() for s in only.split(",") if s.strip()]
    bad = [i for i in ids if i not in BASELINES]
    if bad:
        raise SystemExit(f"unknown baseline id(s): {bad} — "
                         f"known: {list(BASELINES)}")
    return {i: BASELINES[i] for i in ids}


# ── --capture ───────────────────────────────────────────────────────────────
def cmd_capture(args) -> int:
    sel = _select(args.only)
    manifest = (json.loads(_MANIFEST.read_text(encoding="utf-8"))
                if _MANIFEST.exists() else None)
    if manifest and not args.force:
        clash = [i for i in sel if i in manifest.get("baselines", {})]
        if clash:
            raise SystemExit(f"baselines already exist for {clash} — pass "
                             f"--force (appends a dated recapture reason)")
    print("Keep the browser window VISIBLE - a minimized window throttles "
          "rAF and the run times out. Do not touch the capture tabs.\nClose the accumulated tabs afterwards.")
    port = args.port or _free_port()
    proc = _start_server(port)
    entries, env = {}, None
    if manifest is None:
        manifest = {"format": 1, "px": list(_PX), "default_frames": 64,
                    "env": None, "tolerances": dict(_TOL),
                    "recaptures": [], "baselines": {}}
    elif args.force:
        manifest.setdefault("recaptures", []).append(
            {"date": _now(), "reason": args.reason or "recapture",
             "renderer": None})           # filled once the first capture lands
    try:
        for bid, spec in sel.items():
            # THREE renders, not two: the 2-render proof measurably lied —
            # water_shutter, clock_mid and ring_sph_mid each matched twice at
            # capture, then flickered a single |d|=1 pixel on a later check
            # and had to be demoted. Three matches is still no guarantee
            # (Tier-1 FAIL hints + --noise remain the backstop), just a
            # cheaper-than-demotion first filter.
            print(f"  {bid}: render 1/3 ...", flush=True)
            meta1, png1 = _capture_retry(port, bid, spec,
                                        timeout=args.timeout, proc=proc)
            _healthy(bid, meta1)
            det = True
            for rep in (2, 3):
                print(f"  {bid}: render {rep}/3 (determinism proof) ...",
                      flush=True)
                meta2, _png2 = _capture_retry(port, bid, spec,
                                             timeout=args.timeout, proc=proc)
                _healthy(bid, meta2)
                if meta1["sha256_rgb"] != meta2["sha256_rgb"]:
                    det = False
                    break
            if not det:
                print(f"  {bid}: WARNING - two back-to-back renders differ; "
                      f"marked deterministic:false (Tier-2 only; run --noise "
                      f"and re-derive thresholds before R3 gates on it)")
            _BASE.mkdir(parents=True, exist_ok=True)
            (_BASE / f"{bid}.png").write_bytes(png1)       # run 1 is the baseline
            entries[bid] = {
                "slug": spec["slug"], "cam": spec["cam"], "t": spec["t"],
                "frames": spec["frames"], "sha256_rgb": meta1["sha256_rgb"],
                "png": f"{bid}.png", "deterministic": det,
                "scene_sha256": _scene_sha(spec["slug"]),
                "reason": spec["reason"]}
            if env is None:
                env = {"renderer": meta1.get("renderer"),
                       "ua": meta1.get("ua"), "os": sys.platform,
                       "captured_at": _now(),
                       "note": ("baselines are PER-(GPU, driver, browser/"
                                "ANGLE) artifacts; a driver or browser update"
                                " legitimately breaks exact hashes")}
                if args.force and manifest.get("recaptures"):
                    manifest["recaptures"][-1]["renderer"] = env["renderer"]
            elif env["renderer"] != meta1.get("renderer"):
                print(f"  {bid}: WARNING - renderer string differs from "
                      f"earlier captures ({meta1.get('renderer')!r} vs "
                      f"{env['renderer']!r}) - dual-GPU selection?")
            # FLUSH per scene: a mid-run infrastructure failure (one slow
            # capture timing out) must not discard the captures that finished
            manifest["env"] = env
            manifest["baselines"].update(entries)
            _MANIFEST.write_text(json.dumps(manifest, indent=1),
                                 encoding="utf-8")
            print(f"  {bid}: {'deterministic' if det else 'NON-deterministic'}"
                  f"  sha256={meta1['sha256_rgb'][:16]}...  "
                  f"{len(png1)/1024:.0f} KB")
    finally:
        proc.terminate()
    ndet = sum(1 for e in entries.values() if e["deterministic"])
    print(f"\n{len(entries)} baseline(s) written to {_BASE}  "
          f"({ndet}/{len(entries)} deterministic)")
    return 0


def _tol_for(entry: dict, manifest: dict) -> dict:
    """Per-baseline tolerance, falling back to the shared manifest default.

    Some scenes have a genuinely higher noise floor than others (e.g. a
    deep-CSG scene has more silhouette edges, hence more discrete hit/miss
    pixel flips per render, than a simple one) — a single global tolerance
    either starves the noisy scenes of headroom (false FAILs on unchanged
    code) or gives up discrimination power everywhere to cover them. An
    entry MAY carry its own "tolerances" dict (same shape as the manifest
    default); measure it with --noise and record the same >=3x-observed
    justification style the global default uses.
    """
    base = dict(manifest.get("tolerances", _TOL))
    base.update(entry.get("tolerances") or {})
    return base


# ── --check ─────────────────────────────────────────────────────────────────
def cmd_check(args) -> int:
    manifest = _load_manifest()
    baselines = manifest.get("baselines", {})
    sel = _select(args.only)
    missing = [i for i in sel if i not in baselines]
    rows, any_fail, any_infra = [], False, False
    if missing:
        for bid in missing:
            rows.append((bid, "-", "MISSING", "", "", "",
                         "no committed baseline — run --capture"))
        any_fail = True
    todo = {i: s for i, s in sel.items() if i in baselines}
    print("Keep the browser window VISIBLE - a minimized window throttles "
          "rAF and the run times out. Do not touch the capture tabs.\nClose the accumulated tabs afterwards.")
    port = args.port or _free_port()
    proc = _start_server(port)
    env_banner_shown = False
    try:
        for bid, spec in todo.items():
            entry = baselines[bid]
            espec = {"slug": entry["slug"], "cam": entry["cam"],
                     "t": entry["t"], "frames": entry["frames"]}
            try:
                if _scene_sha(entry["slug"]) != entry.get("scene_sha256"):
                    rows.append((bid, "-", "INPUT-CHANGED", "", "", "",
                                 "scene JSON regenerated — recapture to "
                                 "re-baseline (NOT a renderer regression)"))
                    any_fail = True
                    continue
                meta, _png = _capture_retry(port, bid, espec,
                                           timeout=args.timeout, proc=proc)
                _healthy(bid, meta)
            except Infra as e:
                rows.append((bid, "-", "INFRA", "", "", "", str(e)))
                any_infra = True
                continue
            w, h, got = png_io.read_rgb_png(str(_SINK / f"{bid}.png"))
            try:
                _ew, _eh, exp = png_io.read_rgb_png(str(_BASE / entry["png"]))
            except (FileNotFoundError, ValueError) as e:
                # A manifest entry whose PNG is missing/corrupt is an
                # integrity failure of the BASELINE SET, not a renderer
                # regression — verdict it honestly per-entry and keep
                # checking the others (one lost file must never abort or
                # obscure the rest of the net).
                rows.append((bid, "-", "BASELINE-MISSING", "", "", "",
                             f"{entry['png']}: {e} — baseline file lost or "
                             f"corrupt; restore from git or --capture to "
                             f"re-baseline"))
                any_fail = True
                continue
            env_match = (_fingerprint(meta)
                         == (manifest["env"].get("renderer") or "?",
                             manifest["env"].get("ua") or "?"))
            tier1 = env_match and entry.get("deterministic")
            if not env_match and not env_banner_shown:
                env_banner_shown = True
                print(f"  ADVISORY — baselines captured on "
                      f"{manifest['env'].get('renderer')!r} / "
                      f"{manifest['env'].get('ua')!r}; this machine reports "
                      f"{meta.get('renderer')!r} / {meta.get('ua')!r}. "
                      f"Recapture on this machine for the authoritative "
                      f"exact gate.")
            if tier1 and hashlib.sha256(got).hexdigest() == entry["sha256_rgb"]:
                rows.append((bid, "exact", "PASS", "0.000%", "0.00", "0", ""))
                continue
            tol = _tol_for(entry, manifest)
            m = _metrics(got, exp)
            fmt = (f"{m['changed_frac']*100:.3f}%", f"{m['mean_abs']:.2f}",
                   f"{m['p99_abs']:d}")
            if tier1:                              # exact gate, hash differed
                _write_fail_artifacts(bid, w, h, got, exp)
                within_t2 = (m["changed_frac"] <= tol["changed_frac"]
                             and m["mean_abs"] <= tol["mean_abs"]
                             and m["p99_abs"] <= tol["p99_abs"])
                hint = (" [metrics are WITHIN Tier-2 noise bounds - the "
                        "entry's 2-render determinism proof likely got "
                        "lucky; --noise it and demote deterministic:false]"
                        if within_t2 else "")
                rows.append((bid, "exact", "FAIL", *fmt,
                             f"hash mismatch (max|d|={m['max_abs']}); "
                             f"artifacts in misc/render_regression/" + hint))
                any_fail = True
                continue
            ok = (m["changed_frac"] <= tol["changed_frac"]
                  and m["mean_abs"] <= tol["mean_abs"]
                  and m["p99_abs"] <= tol["p99_abs"])
            note = ("ADVISORY (foreign env)" if not env_match
                    else "ADVISORY (entry non-deterministic)")
            if ok:
                rows.append((bid, "advisory", "PASS", *fmt, note))
            else:
                _write_fail_artifacts(bid, w, h, got, exp)
                if args.allow_foreign_env and not env_match:
                    rows.append((bid, "advisory", "WARN", *fmt,
                                 note + " — failure downgraded by "
                                 "--allow-foreign-env"))
                else:
                    rows.append((bid, "advisory", "FAIL", *fmt,
                                 note + "; artifacts in "
                                 "misc/render_regression/"))
                    any_fail = True
    finally:
        proc.terminate()
    print(f"\n{'id':18s} {'tier':9s} {'result':14s} {'changed%':>9s} "
          f"{'mean|d|':>8s} {'p99|d|':>7s}  note")
    for r in rows:
        print(f"{r[0]:18s} {r[1]:9s} {r[2]:14s} {r[3]:>9s} {r[4]:>8s} "
              f"{r[5]:>7s}  {r[6]}")
    _FAIL.mkdir(parents=True, exist_ok=True)
    (_FAIL / "report.json").write_text(json.dumps(
        {"date": _now(), "rows": [list(r) for r in rows]}, indent=1),
        encoding="utf-8")
    npass = sum(1 for r in rows if r[2] in ("PASS", "WARN"))
    print(f"\n{npass}/{len(rows)} baselines hold")
    if any_fail:
        return 1
    if any_infra:
        return 2
    return 0


# ── --selftest : the comparator must DISCRIMINATE, not just accept ──────────
def cmd_selftest(args) -> int:
    """Tier-aware negative + positive controls.

    deterministic:true entry  -> exact-hash controls: a 48-frame render MUST
    mismatch the 64-frame baseline hash; a 64-frame render MUST match it.

    deterministic:false entry -> the operative gate is Tier-2, so the
    controls exercise the STATISTICAL comparator: a camera nudged by +0.05
    rad az (a real image change) MUST trip the gates; a plain recapture
    (carrying only the measured per-page-load GPU noise) MUST pass them.
    """
    manifest = _load_manifest()
    bid = args.id or "opacity_cup"
    entry = manifest["baselines"].get(bid)
    if not entry:
        raise Infra(f"--selftest needs a committed baseline for {bid!r}")
    tol = _tol_for(entry, manifest)
    spec = {"slug": entry["slug"], "cam": entry["cam"], "t": entry["t"],
            "frames": entry["frames"]}
    _bw, _bh, base_rgb = png_io.read_rgb_png(str(_BASE / entry["png"]))
    port = args.port or _free_port()
    proc = _start_server(port)
    try:
        if entry.get("deterministic"):
            neg_frames = max(1, entry["frames"] - 16)      # 64 -> 48
            print(f"negative control: {bid} at frames={neg_frames} "
                  f"(baseline is {entry['frames']}) - MUST mismatch ...")
            meta_n, _ = _capture_once(port, bid, spec, frames=neg_frames,
                                      timeout=args.timeout, proc=proc)
            _healthy(bid, meta_n)
            if _fingerprint(meta_n)[0] != (manifest["env"].get("renderer") or "?"):
                raise Infra("environment differs from the baselines - "
                            "recapture before running --selftest")
            neg_ok = meta_n["sha256_rgb"] != entry["sha256_rgb"]
            print("  negative: " + ("hash mismatch (good)" if neg_ok else
                                    "MATCHED - the comparator cannot discriminate!"))
            print(f"positive control: {bid} at frames={entry['frames']} - "
                  f"MUST match ...")
            meta_p, _ = _capture_once(port, bid, spec, timeout=args.timeout,
                                      proc=proc)
            _healthy(bid, meta_p)
            pos_ok = meta_p["sha256_rgb"] == entry["sha256_rgb"]
            print("  positive: " + ("exact match (good)" if pos_ok else
                                    "MISMATCH - determinism broke since capture!"))
        else:
            def gates(m):
                return (m["changed_frac"] <= tol["changed_frac"]
                        and m["mean_abs"] <= tol["mean_abs"]
                        and m["p99_abs"] <= tol["p99_abs"])
            nudged = dict(spec)
            nudged["cam"] = [spec["cam"][0] + 0.05] + list(spec["cam"][1:])
            print(f"negative control: {bid} with az nudged +0.05 rad "
                  f"(a REAL image change) - Tier-2 gates MUST trip ...")
            meta_n, _ = _capture_once(port, bid, nudged,
                                      timeout=args.timeout, proc=proc)
            _healthy(bid, meta_n)
            if _fingerprint(meta_n)[0] != (manifest["env"].get("renderer") or "?"):
                raise Infra("environment differs from the baselines - "
                            "recapture before running --selftest")
            _w, _h, got = png_io.read_rgb_png(str(_SINK / f"{bid}.png"))
            m_n = _metrics(got, base_rgb)
            neg_ok = not gates(m_n)
            print(f"  negative: changed={m_n['changed_frac']*100:.3f}% "
                  f"mean={m_n['mean_abs']:.3f} p99={m_n['p99_abs']} -> "
                  + ("gates tripped (good)" if neg_ok else
                     "PASSED - the statistical gate cannot discriminate!"))
            print(f"positive control: {bid} recaptured as-committed - "
                  f"Tier-2 gates MUST pass ...")
            meta_p, _ = _capture_once(port, bid, spec, timeout=args.timeout,
                                      proc=proc)
            _healthy(bid, meta_p)
            _w, _h, got = png_io.read_rgb_png(str(_SINK / f"{bid}.png"))
            m_p = _metrics(got, base_rgb)
            pos_ok = gates(m_p)
            print(f"  positive: changed={m_p['changed_frac']*100:.3f}% "
                  f"mean={m_p['mean_abs']:.3f} p99={m_p['p99_abs']} -> "
                  + ("gates pass (good)" if pos_ok else
                     "FAILED - noise exceeds the committed tolerances!"))
    finally:
        proc.terminate()
    if neg_ok and pos_ok:
        print("selftest PASSED - the gate discriminates and still accepts")
        return 0
    print("selftest FAILED")
    return 1


# ── --noise : measure the actual pairwise-noise distribution ────────────────
def cmd_noise(args) -> int:
    sel = _select(args.only)
    port = args.port or _free_port()
    proc = _start_server(port)
    worst = {"changed_frac": 0.0, "mean_abs": 0.0, "p99_abs": 0}
    try:
        for bid, spec in sel.items():
            caps = []
            try:
                for rep in range(args.reps):
                    meta, _ = _capture_retry(port, bid, spec,
                                             timeout=args.timeout, proc=proc)
                    _healthy(bid, meta)
                    _w, _h, rgb = png_io.read_rgb_png(str(_SINK / f"{bid}.png"))
                    caps.append((meta["sha256_rgb"], rgb))
            except Infra as e:
                # one stuck scene must not abort the measurement campaign
                print(f"{bid}: INFRA after {len(caps)} rep(s) - {e}")
                if len(caps) < 2:
                    continue
            hashes = {c[0] for c in caps}
            print(f"{bid}: {args.reps} reps -> {len(hashes)} distinct hash(es)")
            for i in range(len(caps)):
                for j in range(i + 1, len(caps)):
                    if caps[i][0] == caps[j][0]:
                        continue
                    m = _metrics(caps[i][1], caps[j][1])
                    print(f"  pair {i}-{j}: changed={m['changed_frac']*100:.4f}% "
                          f"mean={m['mean_abs']:.3f} p99={m['p99_abs']} "
                          f"max={m['max_abs']}")
                    for k in worst:
                        worst[k] = max(worst[k], m[k])
    finally:
        proc.terminate()
    print(f"\nworst pairwise: {worst}")
    print("threshold rule: any non-deterministic entry must gate at >= 3x "
          "the worst observed pairwise metric (edit the manifest tolerances "
          "+ justification).")
    return 0


# ── --derive-cam : replicate locateBody's arithmetic OFFLINE to mint floats ─
def cmd_derive_cam(args) -> int:
    o = json.loads((_DATA / f"{args.slug}.json").read_text(encoding="utf-8"))
    sc = o.get("scene") if o.get("kind") == "trajectory" else o
    tr = o.get("trajectory") if o.get("kind") == "trajectory" else o.get("trajectory")
    bodies = sc.get("bodies") or []
    labels = [b.get("label") for b in bodies]
    if args.body not in labels:
        raise SystemExit(f"no body labeled {args.body!r} — labels: {labels}")
    i = labels.index(args.body)

    def pose_at(t):                                # posesAt, positions only
        F = tr["frames"]
        if t is None or t <= F[0]["t_sim"]:
            return [b["pos"] for b in F[0]["bodies"]]
        if t >= F[-1]["t_sim"]:
            return [b["pos"] for b in F[-1]["bodies"]]
        lo, hi = 0, len(F) - 1
        while hi - lo > 1:
            m = (lo + hi) >> 1
            if F[m]["t_sim"] <= t:
                lo = m
            else:
                hi = m
        a, b = F[lo], F[hi]
        u = (t - a["t_sim"]) / max(1e-9, b["t_sim"] - a["t_sim"])
        return [[pa["pos"][k] + (pb["pos"][k] - pa["pos"][k]) * u
                 for k in range(3)]
                for pa, pb in zip(a["bodies"], b["bodies"])]

    poses = pose_at(args.t)
    tgt = poses[i]
    # the EXACT _bodyRadius fallback chain: radius_m || csg-leaf bounds radius || 1.0
    rM = bodies[i].get("radius_m")
    if not rM:
        leaf = next((l for l in (sc.get("csg_leaves") or [])
                     if l.get("body") == i), {})
        rM = (leaf.get("bounds") or {}).get("radius")
    rM = rM or 1.0
    radius = max(rM * 3.2, 1e-6)
    up = (sc.get("camera") or {}).get("up") or [0, 0, 1]

    def norm(v):
        L = math.hypot(*v) or 1
        return [x / L for x in v]

    def cross(a, b):
        return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

    def dot(a, b):
        return sum(x*y for x, y in zip(a, b))

    az, el = 0.8, 0.45                             # viewer defaults if no sun
    sun = labels.index("Sun") if "Sun" in labels else -1
    if sun >= 0 and sun != i:
        U2 = norm(up)
        H = [1, 0, 0] if abs(U2[0]) < 0.9 else [0, 0, 1]
        A = norm(cross(U2, H))
        B = cross(U2, A)
        d = norm([s - t for s, t in zip(poses[sun], tgt)])
        se = max(-1.0, min(1.0, dot(d, U2)))
        el = math.asin(se)
        if abs(math.cos(el)) > 1e-6:
            az = math.atan2(dot(d, B), dot(d, A))
    cam = [az, el, radius] + list(tgt)
    print(f"cam=[{', '.join(_f(v) for v in cam)}]")
    print(f"# {args.slug} · {args.body} · t={args.t} · radius = 3.2 x {rM} m")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--derive-cam", nargs=2, metavar=("SLUG", "BODY"),
                    dest="derive_cam")
    ap.add_argument("--only", help="comma-separated baseline ids")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing baselines (logs a dated reason)")
    ap.add_argument("--reason", help="recapture reason for --force")
    ap.add_argument("--allow-foreign-env", action="store_true",
                    help="downgrade advisory failures to warnings (exit 0)")
    ap.add_argument("--port", type=int, default=0,
                    help="server port (default: OS-assigned ephemeral — "
                         "NEVER default to 8765, stale servers shadow it)")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="per-scene capture timeout, seconds")
    ap.add_argument("--reps", type=int, default=5, help="--noise repetitions")
    ap.add_argument("--id", help="--selftest baseline id (default opacity_cup)")
    ap.add_argument("--t", type=float, default=None,
                    help="--derive-cam: sim time (default: first frame)")
    args = ap.parse_args(argv)
    try:
        if args.capture:
            return cmd_capture(args)
        if args.check:
            return cmd_check(args)
        if args.selftest:
            return cmd_selftest(args)
        if args.noise:
            return cmd_noise(args)
        if args.derive_cam:
            args.slug, args.body = args.derive_cam
            return cmd_derive_cam(args)
    except Infra as e:
        print(f"INFRASTRUCTURE: {e}", file=sys.stderr)
        return 2
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
