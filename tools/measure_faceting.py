"""R4.4 faceting checkpoint — MEASURE the Lambert-sphere seam ratio.

The pre-existing defect (R3 review, session 31): the voxel raymarch renders
~1.6x/px brighter than a true Lambert sphere (calcN finite differences over
the 50^3 trilinear field quantize into per-cell facets that alias against
the Lambert term), producing a x1.72 one-time brightness STEP at the 24 px
raymarch<->impostor band edge (the impostor's closed-form disc radiance is
the correct side). Blueprint Sec 7 makes re-measuring this a MANDATORY
checkpoint at R4.4 BEFORE the re-baseline session.

Method (the R3 review's own capture-ladder idiom, made a tool): frame the
body sun-facing (the --derive-cam az/el arithmetic verbatim) at distances
where its apparent diameter is 23 px (impostor band, the CORRECT closed
form by R3's review) and 25 px (raymarch band). A Lambert sphere's
disc-MEAN radiance is distance-independent at fixed phase, so crossing the
band edge should be continuous: the measured step IS the faceting bias.

    ratio = mean_linear_disc(25 px raymarch) / mean_linear_disc(23 px impostor)

Run (spawns serve.py + capture tabs, the render_regression machinery):

    python tools/measure_faceting.py                 # t48: svo=0 (before) + svo=1 (after)
    python tools/measure_faceting.py --tier 192      # t192 scene (svo=1 only: mechanisms 1+2+3)

Captured PNGs + a JSON report land in misc/render_regression/ as
r44_faceting_* (the R4.4 evidence trail). Linearization: captures are
display-pass output (pow 1/2.2), so pixels are linearized with ^2.2 before
averaging. Disc pixels = those within 0.42 x apparent-diameter of the
screen centre (the camera targets the body), comfortably inside the
silhouette so MC edge aliasing never enters the mean.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

import render_regression as rr                                 # noqa: E402
from sigma_ground.radiance import png_io                       # noqa: E402

_EVID = _ROOT / "misc" / "render_regression"


def _sun_facing_cam(slug: str, body: str, radius: float):
    """--derive-cam's arithmetic verbatim (frame-0 pose, sun-facing az/el),
    with an explicit orbit radius instead of the 3.2x framing default."""
    o = json.loads((rr._DATA / f"{slug}.json").read_text(encoding="utf-8"))
    sc = o.get("scene") if o.get("kind") == "trajectory" else o
    tr = o.get("trajectory")
    labels = [b.get("label") for b in (sc.get("bodies") or [])]
    i = labels.index(body)
    poses = [b["pos"] for b in tr["frames"][0]["bodies"]]
    tgt = poses[i]
    up = (sc.get("camera") or {}).get("up") or [0, 0, 1]

    def norm(v):
        L = math.hypot(*v) or 1
        return [x / L for x in v]

    def cross(a, b):
        return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

    def dot(a, b):
        return sum(x*y for x, y in zip(a, b))

    az, el = 0.8, 0.45
    if "Sun" in labels and labels.index("Sun") != i:
        U2 = norm(up)
        H = [1, 0, 0] if abs(U2[0]) < 0.9 else [0, 0, 1]
        A = norm(cross(U2, H))
        B = cross(U2, A)
        d = norm([s - t for s, t in zip(poses[labels.index("Sun")], tgt)])
        se = max(-1.0, min(1.0, dot(d, U2)))
        el = math.asin(se)
        if abs(math.cos(el)) > 1e-6:
            az = math.atan2(dot(d, B), dot(d, A))
    fov = (sc.get("camera") or {}).get("fov_deg", 45.0)
    r_m = (sc.get("bodies") or [])[i].get("radius_m")
    return [az, el, radius] + list(tgt), fov, r_m


def _disc_mean_linear(png_path: pathlib.Path, disc_px: float):
    w, h, rgb = png_io.read_rgb_png(str(png_path))
    cx, cy = w / 2.0, h / 2.0
    rad = 0.42 * disc_px                       # inside the silhouette
    tot = [0.0, 0.0, 0.0]
    n = 0
    for y in range(int(cy - rad) - 1, int(cy + rad) + 2):
        for x in range(int(cx - rad) - 1, int(cx + rad) + 2):
            if (x - cx) ** 2 + (y - cy) ** 2 > rad * rad:
                continue
            o = (y * w + x) * 3
            for c in range(3):
                tot[c] += (rgb[o + c] / 255.0) ** 2.2      # undo display gamma
            n += 1
    return [t / max(n, 1) for t in tot], n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tier", type=int, default=48, choices=(48, 192))
    ap.add_argument("--body", default="Earth")
    ap.add_argument("--svo", default=None,
                    help="comma list of svo flags to measure (default: 0,1 "
                         "at tier 48; 1 at other tiers)")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args(argv)

    slug = ("solar_system_theater" if args.tier == 48
            else f"solar_system_theater_t{args.tier}")
    svos = ([s.strip() for s in args.svo.split(",")] if args.svo
            else (["0", "1"] if args.tier == 48 else ["1"]))

    # distances for the two ladder rungs around the 24 px band edge
    cam0, fov, r_m = _sun_facing_cam(slug, args.body, 1.0)
    tan_half = math.tan(math.radians(fov) / 2.0)
    px_per_rad = rr._PX[1] / (2.0 * tan_half)
    rungs = {"imp23": 23.0, "ray25": 25.0}
    print(f"{slug} / {args.body}: r={r_m:,.0f} m, sun-facing az/el = "
          f"{cam0[0]:.4f}/{cam0[1]:.4f}, pxPerRad={px_per_rad:.1f}")

    _EVID.mkdir(parents=True, exist_ok=True)
    port = args.port or rr._free_port()
    proc = rr._start_server(port)
    out = {"slug": slug, "body": args.body, "tier": args.tier,
           "frames": args.frames, "rungs": {}, "ratios": {}}
    try:
        for sv in svos:
            for rung, px in rungs.items():
                d = 2.0 * r_m * px_per_rad / px
                cam = [cam0[0], cam0[1], d] + cam0[3:]
                bid = f"r44_faceting_t{args.tier}_svo{sv}_{rung}"
                spec = {"slug": slug, "cam": cam, "t": 0.0,
                        "frames": args.frames, "extra": f"svo={sv}"}
                print(f"  {bid}: apparent {px:.1f} px at d={d:.4e} m ...",
                      flush=True)
                meta, _png = rr._capture_retry(port, bid, spec,
                                               timeout=args.timeout, proc=proc)
                rr._healthy(bid, meta)
                src = rr._SINK / f"{bid}.png"
                shutil.copy2(src, _EVID / f"{bid}.png")
                mean, n = _disc_mean_linear(src, px)
                out["rungs"][bid] = {"px": px, "dist_m": d, "cam": cam,
                                     "mean_linear_rgb": mean, "disc_pixels": n,
                                     "sha256": meta.get("sha256_rgb")}
                print(f"    disc mean (linear) = "
                      f"[{mean[0]:.5f}, {mean[1]:.5f}, {mean[2]:.5f}]  "
                      f"({n} px)")
            a = out["rungs"][f"r44_faceting_t{args.tier}_svo{sv}_ray25"]
            b = out["rungs"][f"r44_faceting_t{args.tier}_svo{sv}_imp23"]
            num = sum(a["mean_linear_rgb"]) / 3.0
            den = sum(b["mean_linear_rgb"]) / 3.0
            ratio = num / den if den > 0 else float("inf")
            out["ratios"][f"svo{sv}"] = ratio
            print(f"  seam ratio (raymarch/impostor), svo={sv}: x{ratio:.3f}")
    finally:
        proc.terminate()
    rp = _EVID / f"r44_faceting_t{args.tier}_report.json"
    rp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nreport -> {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
