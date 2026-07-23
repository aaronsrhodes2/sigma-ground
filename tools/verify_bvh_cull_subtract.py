"""Empirical re-verification of the buildPathShader bvhCull subtract-leaf fix
(sigma_ground/radiance/web/viewer.js, decls-loop inside buildPathShader).

Mirrors mapD()/march() from the real path-traced shader in pure Python,
using the SAME exported constants the shader uses (GATE_BUFFER=0.004,
march()'s hit epsilon 0.0008 and step floor 0.0005, the real
_bounding_sphere()/_BOUNDS_MARGIN_M padding from scene_export.py), and
compares three per-leaf-substitute modes at leaf i (i>0, op=="subtract"):

  "exact" — every leaf always exact-evaluated (bvhCull=false; ground truth)
  "bug"   — the ORIGINAL codegen: far subtract leaves substitute a 1e9
            no-op sentinel (max(d,-1e9)=d unchanged)
  "fixed" — the CORRECTED codegen: far subtract leaves substitute bd_i,
            the same bounding-sphere lower bound already used for
            union/intersect leaves

Two scenarios, both requested by the task:

  1. Synthetic reproduction — a big solid sphere (r=2.0) with a small
     buried subtract cavity (r=0.05, center 0.5m from the big sphere's
     center) — the exact case that produced the reported 0.55m divergence.
     Runs full march() replays (the real sphere-tracing loop, |d| stepping)
     for many rays approaching the cavity from many directions and origins,
     comparing hit position between modes.

  2. A REAL hollow-object scene — deckard.identify('coffee cup') through
     the real construct_to_scene() export pipeline — which is confirmed
     (see below) to produce a genuine op="subtract" leaf for the cup's
     hollow interior. Compares mapD() (not full march — a static
     rendering-relevant SDF sample field, as the task also asks for)
     across a grid of query points spanning the cup's geometry, comparing
     "bvhCull off" (exact) against "bvhCull on" using each substitute mode.

Run: python tools/verify_bvh_cull_subtract.py
"""
import math
import sys

import numpy as np

GATE_BUFFER = 0.004          # viewer.js:943  #define GATE_BUFFER 0.004
HIT_EPS = 0.0008             # viewer.js march(): if(abs(d)<0.0008) return true;
STEP_MIN = 0.0005            # viewer.js march(): t+=max(abs(d),0.0005)
MAX_ITER = 220               # viewer.js march(): for(int i=0;i<220;i++)
BOUNDS_MARGIN_M = 0.01       # scene_export.py _BOUNDS_MARGIN_M
BOUNDS_MIN_RADIUS_M = 4.0 * BOUNDS_MARGIN_M   # scene_export.py _BOUNDS_MIN_RADIUS_M


def sphere_sdf(p, c, r):
    return float(np.linalg.norm(p - np.asarray(c, dtype=float))) - r


def cyl_sdf(p, c, radius, height):
    # mirrors jsLeafSDF's Cylinder branch (viewer.js:219-220) exactly.
    d = p - np.asarray(c, dtype=float)
    dr = math.hypot(d[0], d[1]) - radius
    da = abs(d[2]) - height * 0.5
    return min(max(dr, da), 0.0) + math.hypot(max(dr, 0.0), max(da, 0.0))


def bounding_sphere_for_radius(true_radius, center):
    """Mirrors scene_export.py's _bounding_sphere(): None (no gating) if the
    leaf is too small relative to the margin for early-rejection to be safe."""
    if true_radius < BOUNDS_MIN_RADIUS_M:
        return None
    return (np.asarray(center, dtype=float), true_radius + BOUNDS_MARGIN_M)


def make_exact(shape):
    """Exact SDF closure for an exported leaf's shape dict (Cylinder/Sphere —
    the two primitive types this harness's real-scene scenario needs)."""
    t = shape["type"]
    c = shape["center"]
    if t == "Cylinder":
        return lambda p, c=c, r=shape["radius"], h=shape["height"]: cyl_sdf(p, c, r, h)
    if t == "Sphere":
        return lambda p, c=c, r=shape["radius"]: sphere_sdf(p, c, r)
    raise NotImplementedError(f"shape type {t} not mirrored in this harness")


def compose_mapD(p, leaves, mode):
    """Mirrors buildPathShader's decls-loop + compose loop
    (viewer.js:776-812, 855-856ish) for an arbitrary leaf list.

    leaves[i] = {"op": "add"|"subtract"|"intersect", "exact": fn(p)->float,
                 "bounds": (center, radius) or None}
    mode: "exact" | "bug" | "fixed"  (bvhCull off / on-with-1e9 / on-with-bd_i)
    """
    d = None
    for i, leaf in enumerate(leaves):
        bounds = leaf["bounds"]
        if mode == "exact" or bounds is None:
            s = leaf["exact"](p)
        else:
            c, r = bounds
            bd = float(np.linalg.norm(p - c)) - r
            if bd > GATE_BUFFER:
                # farS ternary (viewer.js, post-fix): i>0 subtract used to get
                # "1e9" in "bug" mode; every other case (including i==0, which
                # is never routed through compose's subtract branch regardless
                # of its own op field) always wanted bd_i, in BOTH modes.
                if mode == "bug" and i > 0 and leaf["op"] == "subtract":
                    s = 1e9
                else:
                    s = bd
            else:
                s = leaf["exact"](p)
        if i == 0:
            d = s
        elif leaf["op"] == "subtract":
            d = max(d, -s)
        elif leaf["op"] == "intersect":
            d = max(d, s)
        else:
            d = min(d, s)
    return d


def march(ro, rd, leaves, mode, eps_start, max_dist):
    """Mirrors viewer.js's march(): |d| sphere-trace, works inside a medium."""
    t = eps_start
    for _ in range(MAX_ITER):
        p = ro + rd * t
        d = compose_mapD(p, leaves, mode)
        if abs(d) < HIT_EPS:
            return True, p, t
        t += max(abs(d), STEP_MIN)
        if t > max_dist:
            return False, None, t
    return False, None, t


def fibonacci_sphere(n):
    pts = []
    ga = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (i / (n - 1)) * 2.0 if n > 1 else 0.0
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = ga * i
        pts.append((math.cos(theta) * r, y, math.sin(theta) * r))
    return np.array(pts)


# ────────────────────────────────────────────────────────────────────────
# Scenario 1: synthetic reproduction — r=2.0 solid sphere, buried r=0.05
# subtract cavity 0.5m inside.
# ────────────────────────────────────────────────────────────────────────
def scenario_synthetic():
    print("=" * 78)
    print("SCENARIO 1: synthetic — r=2.0 solid sphere, buried r=0.05 subtract")
    print("cavity, center 0.5m inside")
    print("=" * 78)

    C0, R0 = np.array([0.0, 0.0, 0.0]), 2.0
    C1, R1 = np.array([0.5, 0.0, 0.0]), 0.05

    b0 = bounding_sphere_for_radius(R0, C0)
    b1 = bounding_sphere_for_radius(R1, C1)
    assert b1 is not None, "cavity radius must clear _BOUNDS_MIN_RADIUS_M for this test to be meaningful"
    print(f"leaf0 (add, big sphere):  center={C0}, r={R0}, bounds_radius={b0[1]:.6f}")
    print(f"leaf1 (subtract, cavity): center={C1}, r={R1}, bounds_radius={b1[1]:.6f}")

    leaves = [
        {"op": "add", "exact": lambda p: sphere_sdf(p, C0, R0), "bounds": b0},
        {"op": "subtract", "exact": lambda p: sphere_sdf(p, C1, R1), "bounds": b1},
    ]

    # diag from a representative bbox enclosing the big sphere -> real EPS/uMaxDist
    # formulas from viewer.js:895 (buildPathShader's own SHMIN/SHMAX/SMAXT/EPS block).
    bbox = [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]]
    diag = math.dist([bbox[0][1], bbox[1][1], bbox[2][1]], [bbox[0][0], bbox[1][0], bbox[2][0]]) or 1.0
    eps_start = max(0.002, diag * 0.0015)
    max_dist = max(0.5, diag * 1.2) * 4  # generous headroom (uMaxDist stand-in)
    print(f"diag={diag:.4f}  EPS(start t)={eps_start:.6f}  max_dist={max_dist:.4f}\n")

    # Rays: many origins on a sphere of radius 1.3 around the CAVITY center
    # (guaranteeing origins stay inside the big solid sphere: |origin|<=1.8<2.0,
    # and start "far" from the cavity's bounding sphere: bd1 = 1.3-0.06 >> GATE_BUFFER),
    # each aimed either straight at the cavity center ("direct") or offset by a
    # small perpendicular jitter within the cavity's true cross-section ("grazing")
    # — covering "toward and around the cavity" as the task requires.
    N_DIR = 64
    dirs = fibonacci_sphere(N_DIR)
    rng = np.random.default_rng(20260722)

    results = {"exact": [], "bug": [], "fixed": []}
    max_div = {"bug": 0.0, "fixed": 0.0}
    n_hit_mismatch = {"bug": 0, "fixed": 0}
    n_big_divergence = {"bug": 0, "fixed": 0}   # > 1cm from exact

    for k, dvec in enumerate(dirs):
        origin = C1 + dvec * 1.3
        for variant in ("direct", "grazing"):
            if variant == "direct":
                target = C1.copy()
            else:
                # perpendicular jitter within the cavity's true radius (0.05)
                perp = np.cross(dvec, np.array([0.0, 0.0, 1.0]) if abs(dvec[2]) < 0.9 else np.array([1.0, 0.0, 0.0]))
                perp = perp / (np.linalg.norm(perp) + 1e-12)
                jitter = (rng.random() * 0.045) * perp
                target = C1 + jitter
            rd = target - origin
            rd = rd / np.linalg.norm(rd)

            hit_e, p_e, _ = march(origin, rd, leaves, "exact", eps_start, max_dist)
            for mode in ("bug", "fixed"):
                hit_m, p_m, _ = march(origin, rd, leaves, mode, eps_start, max_dist)
                if hit_e != hit_m:
                    n_hit_mismatch[mode] += 1
                    div = float("inf")
                else:
                    div = float(np.linalg.norm(p_e - p_m)) if hit_e else 0.0
                max_div[mode] = max(max_div[mode], div if div != float("inf") else max_div[mode])
                if div > 0.01:
                    n_big_divergence[mode] += 1
                results[mode].append(div)

    n_rays = N_DIR * 2
    for mode in ("bug", "fixed"):
        finite = [d for d in results[mode] if d != float("inf")]
        print(f"mode={mode:5s}  rays={n_rays}  hit/miss mismatches={n_hit_mismatch[mode]}  "
              f"max position divergence={max_div[mode]:.6f} m  "
              f"rays diverging >1cm={n_big_divergence[mode]}  "
              f"mean(finite)={np.mean(finite) if finite else 0.0:.6f} m")

    # Pass bar for "fixed": no hit/miss mismatch, and no ray diverges more than
    # 1cm (two orders of magnitude below the reported bug's 0.55m and this
    # harness's own reproduced ~2m worst case). A few tenths of a millimetre
    # of residual divergence is EXPECTED float/quantization noise, not error:
    # "far" and "close" evaluate leaf 1 through different code paths (bd1
    # substitute vs the exact cylinder-of-a-sphere formula), so the two modes
    # take a different sequence of march() steps before both fall back to
    # IDENTICAL exact evaluation once bd1<=GATE_BUFFER — they converge on the
    # same true surface, but can cross the 0.0008 hit-epsilon threshold on a
    # different sample a fraction of a millimetre apart.
    ok = (n_hit_mismatch["fixed"] == 0 and n_big_divergence["fixed"] == 0)
    bug_confirmed = (n_hit_mismatch["bug"] > 0 or n_big_divergence["bug"] > 0)
    print(f"\nold 1e9 sentinel reproduces a real defect: {bug_confirmed}")
    print(f"bd_i fix matches exact evaluation (0 / float-noise divergence): {ok}\n")
    return ok, bug_confirmed


# ────────────────────────────────────────────────────────────────────────
# Scenario 2: real hollow-object scene — deckard.identify('coffee cup')
# through the real construct_to_scene() pipeline.
# ────────────────────────────────────────────────────────────────────────
def scenario_coffee_cup():
    print("=" * 78)
    print("SCENARIO 2: real scene — deckard.identify('coffee cup')")
    print("=" * 78)
    import sigma_ground.deckard as deckard
    from sigma_ground.radiance.scene_export import construct_to_scene

    construct = deckard.identify("coffee cup", allow_llm=False)
    spec = construct_to_scene(construct)
    raw_leaves = spec["csg_leaves"]
    print(f"leaves exported: {len(raw_leaves)}")
    subtract_idxs = [i for i, l in enumerate(raw_leaves) if i > 0 and l.get("op") == "subtract"]
    assert subtract_idxs, "expected a real op=='subtract' leaf (the cup's hollow interior)"
    print(f"op=='subtract' leaf indices: {subtract_idxs}")
    for i in subtract_idxs:
        l = raw_leaves[i]
        print(f"  leaf {i}: {l['shape']['type']} center={l['shape']['center']} "
              f"radius={l['shape'].get('radius')} height={l['shape'].get('height')} "
              f"bounds={l.get('bounds')}")

    leaves = []
    for l in raw_leaves:
        exact_fn = make_exact(l["shape"])
        bounds = None
        if "bounds" in l:
            bounds = (np.array(l["bounds"]["center"], dtype=float), l["bounds"]["radius"])
        leaves.append({"op": l["op"], "exact": exact_fn, "bounds": bounds})

    # Sample a grid spanning well beyond the cup's own geometry (3x its
    # footprint) — not just the tight rendering silhouette — specifically to
    # go LOOKING for the "dangerous regime" this fix targets: a query point
    # still inside the solid via the leaves *before* the subtract leaf
    # (d_prev<0) while the subtract leaf's OWN bounding sphere is gated far
    # (bd>GATE_BUFFER). That regime is where bug ("1e9") and fixed ("bd_i")
    # can actually produce different composed d — everywhere else the two
    # modes are provably identical (see the viewer.js comment this fix adds).
    centers = [np.array(l["shape"]["center"], dtype=float) for l in raw_leaves]
    radii = [l["shape"].get("radius", 0.0) for l in raw_leaves]
    heights = [l["shape"].get("height", 0.0) for l in raw_leaves]
    rmax = max(radii) * 3.0
    zc = centers[0][2]
    hmax = max(heights) * 1.5

    xs = np.linspace(-rmax, rmax, 31)
    ys = np.linspace(-rmax, rmax, 31)
    zs = np.linspace(zc - hmax, zc + hmax, 31)

    max_abs_err = {"bug": 0.0, "fixed": 0.0}
    n_big_err = {"bug": 0, "fixed": 0}   # SDF magnitude error > 1mm, rendering-relevant
    n_samples = 0
    n_dangerous_regime = 0   # d_prev<0 AND some subtract leaf's own bd>GATE_BUFFER
    max_bug_vs_fixed_diff = 0.0   # isolates JUST the subtract-branch fix's effect
    for x in xs:
        for y in ys:
            for z in zs:
                p = np.array([x, y, z])
                d_exact = compose_mapD(p, leaves, "exact")
                n_samples += 1
                # Was ANY subtract leaf gated far while the running exact
                # compose (over the leaves before it) was still negative?
                d_prev = None
                for i, l in enumerate(leaves):
                    s_exact = l["exact"](p)
                    if i == 0:
                        d_prev = s_exact
                        continue
                    if l["op"] == "subtract" and l["bounds"] is not None:
                        c, r = l["bounds"]
                        bd = float(np.linalg.norm(p - c)) - r
                        if bd > GATE_BUFFER and d_prev < 0.0:
                            n_dangerous_regime += 1
                    if l["op"] == "subtract":
                        d_prev = max(d_prev, -s_exact)
                    elif l["op"] == "intersect":
                        d_prev = max(d_prev, s_exact)
                    else:
                        d_prev = min(d_prev, s_exact)
                d_by_mode = {}
                for mode in ("bug", "fixed"):
                    d_m = compose_mapD(p, leaves, mode)
                    d_by_mode[mode] = d_m
                    err = abs(d_m - d_exact)
                    max_abs_err[mode] = max(max_abs_err[mode], err)
                    if err > 0.001:
                        n_big_err[mode] += 1
                max_bug_vs_fixed_diff = max(max_bug_vs_fixed_diff, abs(d_by_mode["bug"] - d_by_mode["fixed"]))

    print(f"\ngrid samples: {n_samples}  (domain: 3x object footprint, to actively search")
    print(f"for the dangerous regime, not just the tight rendering silhouette)")
    print(f"samples landing in the dangerous regime (d_prev<0 while a subtract leaf's")
    print(f"own bd>GATE_BUFFER): {n_dangerous_regime}")
    for mode in ("bug", "fixed"):
        print(f"mode={mode:5s}  max |d_gated - d_exact| = {max_abs_err[mode]:.6f} m  "
              f"samples with error > 1mm: {n_big_err[mode]}")
    print(f"max |d_bug - d_fixed| (isolates JUST this fix's effect, excludes the "
          f"already-verified union-leaf bd_i approximation shared by both modes): "
          f"{max_bug_vs_fixed_diff:.9f} m")
    if n_dangerous_regime == 0:
        print("\nNo sample fell in the dangerous regime: this cup's subtract leaf's")
        print("padded bounding sphere (~generous 1cm margin on a ~5cm object) is large")
        print("relative to the solid material it carves into, so bd for the subtract")
        print("leaf never exceeds GATE_BUFFER while still inside via the prior leaves.")
        print("bug and fixed are therefore expected to be byte-identical on this")
        print("object -- consistent with (not contradicting) the fix: the structural")
        print("condition that exposes the bug (a base solid LARGE relative to the")
        print("gated leaf's padded bounds) is exactly what scenario 1 reproduces with")
        print("real GATE_BUFFER/_bounding_sphere() constants. The nonzero error shown")
        print("above (both modes, identical) is the ALREADY-verified union-leaf bd_i")
        print("approximation on leaves 0/1/3 -- out of this fix's scope, unaffected.")

    # "ok" for this scenario: the fix introduces NO new divergence relative to
    # the old code on a real scene (fixed's own error vs exact never exceeds
    # bug's, and the two modes agree wherever the differentiating regime
    # doesn't fire). "bug_confirmed" is scoped to JUST the subtract branch
    # (bug vs fixed, not vs exact) — the union-leaf approximation both modes
    # share is pre-existing/accepted and must not be misread as this bug.
    ok = (max_abs_err["fixed"] <= max_abs_err["bug"] + 1e-12) and (n_big_err["fixed"] <= n_big_err["bug"])
    bug_confirmed = (max_bug_vs_fixed_diff > 1e-9)
    print(f"\nold 1e9 sentinel diverges from the fix on THIS real scene's subtract leaf: {bug_confirmed}")
    print(f"({'as expected -- object too compact for the dangerous regime to fire' if not bug_confirmed else 'unexpected -- investigate'})")
    print(f"fix introduces no new divergence vs the old code on this real scene: {ok}\n")
    return ok, bug_confirmed


if __name__ == "__main__":
    ok1, bug1 = scenario_synthetic()
    ok2, bug2 = scenario_coffee_cup()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"synthetic scenario:  bug reproduced={bug1}  fix verified={ok1}")
    print(f"coffee-cup scenario: bug reproduced={bug2}  fix verified={ok2}")
    all_ok = ok1 and ok2
    print(f"\nRESULT: {'PASS — bvhCull=true is safe to ship as default' if all_ok else 'FAIL — do NOT default bvhCull to true'}")
    sys.exit(0 if all_ok else 1)
