/* test_splats.node.js — headless checks for splats.js (R3 band classifier,
 * hysteresis, flux closed form, pack determinism).
 *
 * Run:  node sigma_ground/radiance/web/test_splats.node.js   (exit 0 = pass)
 * Or via pytest: test_render_regression.py-style shell-out (node-gated).
 *
 * The apparent-size expectations are RECOMPUTED here from the committed
 * baselines/manifest.json cameras and the scene JSON's frame-0 positions —
 * never hardcoded blindly — so a re-derived camera or regenerated scene
 * shifts the expectation with the data instead of silently invalidating it.
 */
"use strict";
const path = require("path");
const fs = require("fs");
const S = require(path.join(__dirname, "splats.js"));

let fails = 0, checks = 0;
function ok(name, cond, detail) {
  checks++;
  if (!cond) { fails++; console.log(`FAIL  ${name}  ${detail || ""}`); }
}
function near(a, b, tol) { return Math.abs(a - b) <= tol; }

// ── 1. classifier truth tables (both directions, sticky zones, jumps) ─────
{
  const f = S.bandFor;
  // plain zones
  ok("bandFor: 30px → ray", f(30, 1) === 0);
  ok("bandFor: 10px → impostor", f(10, 0) === 1);
  ok("bandFor: 1px → splat", f(1, 1) === 2);
  // sticky-raymarch [20,24)
  ok("bandFor: 22px from ray stays ray", f(22, 0) === 0);
  ok("bandFor: 22px from impostor stays impostor", f(22, 1) === 1);
  ok("bandFor: 22px from splat → impostor", f(22, 2) === 1);
  // boundaries exact
  ok("bandFor: 24.0 enters ray", f(24.0, 1) === 0);
  ok("bandFor: 20.0 holds ray", f(20.0, 0) === 0);
  ok("bandFor: 19.999 exits ray", f(19.999, 0) === 1);
  // sticky-splat [2.4,3)
  ok("bandFor: 2.7px from splat stays splat", f(2.7, 2) === 2);
  ok("bandFor: 2.7px from impostor stays impostor", f(2.7, 1) === 1);
  ok("bandFor: 2.7px from ray → impostor", f(2.7, 0) === 1);
  ok("bandFor: 3.0 enters impostor", f(3.0, 2) === 1);
  ok("bandFor: 2.399 enters splat", f(2.399, 1) === 2);
  // multi-band single-frame jumps (a stalled dt can cross both boundaries)
  ok("bandFor: 30px from splat jumps straight to ray", f(30, 2) === 0);
  ok("bandFor: 1px from ray jumps straight to splat", f(1, 0) === 2);

  const g = S.bandForEmitter;
  ok("emitter: 10px → ray (never impostor)", g(10, 2) === 0);
  ok("emitter: 3.0 → ray", g(3.0, 2) === 0);
  ok("emitter: 1px → splat", g(1, 0) === 2);
  ok("emitter: 2.7 from splat stays splat", g(2.7, 2) === 2);
  ok("emitter: 2.7 from ray stays ray", g(2.7, 0) === 0);
  ok("emitter: never returns impostor",
     [0.1, 2.4, 2.7, 3.0, 10, 100].every(p => [0, 2].includes(g(p, 0)) && [0, 2].includes(g(p, 2))));
}

// ── 2. init rule: sticky zones resolve to the safer band, deterministically
{
  ok("init: non-emitter 2.7px (no history) → impostor", S.bandFor(2.7, 1) === 1);
  ok("init: emitter 2.7px (no history) → ray", S.bandForEmitter(2.7, 0) === 0);
  ok("init: non-emitter 22px (no history) → impostor", S.bandFor(22, 1) === 1);
}

// ── 3. apparent sizes recomputed from committed manifest + scene JSON ─────
{
  const web = __dirname;
  const man = JSON.parse(fs.readFileSync(path.join(web, "baselines", "manifest.json"), "utf8"));
  const o = JSON.parse(fs.readFileSync(path.join(web, "data", "solar_system_theater.json"), "utf8"));
  const sc = o.scene, tr = o.trajectory;
  const labels = sc.bodies.map(b => b.label);
  const pos = tr.frames[0].bodies.map(b => b.pos);
  const tanHalf = Math.tan((sc.camera.fov_deg || 40) * Math.PI / 360);
  const H = man.px[1];                              // capture backing height
  function eyeOf(cam) {                             // independent eyePos rewrite
    const up = [0, 0, 1];
    const Hax = Math.abs(up[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1];
    let A = [up[1] * Hax[2] - up[2] * Hax[1], up[2] * Hax[0] - up[0] * Hax[2],
             up[0] * Hax[1] - up[1] * Hax[0]];
    const La = Math.hypot(...A); A = A.map(x => x / La);
    const B = [up[1] * A[2] - up[2] * A[1], up[2] * A[0] - up[0] * A[2],
               up[0] * A[1] - up[1] * A[0]];
    const [az, el, R, tx, ty, tz] = cam;
    const ce = Math.cos(el), se = Math.sin(el);
    return [0, 1, 2].map(i =>
      [tx, ty, tz][i] + (A[i] * Math.cos(az) * ce + B[i] * Math.sin(az) * ce + up[i] * se) * R);
  }
  function apx(label, eye) {
    const i = labels.indexOf(label);
    const d = Math.hypot(pos[i][0] - eye[0], pos[i][1] - eye[1], pos[i][2] - eye[2]);
    return S.apparentPx(sc.bodies[i].radius_m, d, tanHalf, H);
  }
  const eyeW = eyeOf(man.baselines.solar_wide.cam);
  const eyeE = eyeOf(man.baselines.solar_earth.cam);
  const pSun = apx("Sun", eyeW), pEarth = apx("Earth", eyeE), pMoon = apx("Moon", eyeE);
  ok("Sun @ solar_wide ≈ 1.836 px", near(pSun, 1.836, 0.01), `got ${pSun}`);
  ok("Earth @ solar_earth ≈ 271.6 px", near(pEarth, 271.6, 0.2), `got ${pEarth}`);
  ok("Moon @ solar_earth ≈ 3.99 px", near(pMoon, 3.99, 0.05), `got ${pMoon}`);
  // band consequences at those pinned cameras (memoryless init)
  ok("Sun @ solar_wide is a splat (emitter, <2.4px)", S.bandForEmitter(pSun, 0) === 2);
  ok("Earth @ solar_earth stays raymarched", S.bandFor(pEarth, 1) === 0);
  ok("Moon @ solar_earth becomes an impostor", S.bandFor(pMoon, 1) === 1);
}

// ── 4. lambertPhase vs numerical integration of the PT cone estimator ─────
// The PT's NEE contribution at a diffuse surface point is alb·Le·ndl·G·neeExp
// with ndl = max(0, n·ωl). Disc-averaging ndl over the camera-visible
// hemisphere with projected-area weight (n·c) must equal lambertPhase(α).
{
  function numPhase(a, N) {
    const s = [Math.sin(a), 0, Math.cos(a)], c = [0, 0, 1];
    let tot = 0, wsum = 0;
    for (let i = 0; i < N; i++) {
      const ct = 1 - 2 * (i + 0.5) / N, st = Math.sqrt(Math.max(0, 1 - ct * ct));
      for (let j = 0; j < 2 * N; j++) {
        const ph = 2 * Math.PI * (j + 0.5) / (2 * N);
        const n = [st * Math.cos(ph), st * Math.sin(ph), ct];
        const muC = n[0] * c[0] + n[1] * c[1] + n[2] * c[2];
        if (muC <= 0) continue;
        tot += muC * Math.max(0, n[0] * s[0] + n[1] * s[1] + n[2] * s[2]);
        wsum += muC;
      }
    }
    return tot / wsum;
  }
  for (const deg of [0, 30, 60, 90, 120, 150]) {
    const a = deg * Math.PI / 180;
    const cf = S.lambertPhase(a), nu = numPhase(a, 200);
    ok(`lambertPhase(${deg}°) matches numerical disc average`, near(cf, nu, 1e-3),
       `closed=${cf} num=${nu}`);
  }
  ok("lambertPhase(0) = 2/3", near(S.lambertPhase(0), 2 / 3, 1e-12));
}

// ── 5. flux-continuity identity at the splat↔impostor seam (P = 3) ────────
// total sprite flux = n²·(disc radiance · min(1,(π/4)P²/n²)) must equal
// disc radiance · (π/4)P² — i.e. the min() clamp must NOT engage in-band.
{
  for (const P of [0.5, 1.0, 2.4, 2.9, 3.0]) {
    const n = S.spriteSize(P);
    const spread = Math.min(1, (Math.PI / 4) * P * P / (n * n));
    ok(`flux conserved at P=${P}: n²·spread == (π/4)P²`,
       near(n * n * spread, (Math.PI / 4) * P * P, 1e-12),
       `n=${n} n²·spread=${n * n * spread} target=${(Math.PI / 4) * P * P}`);
  }
  // inverse-square dimming: halving P quarters the total flux
  const o = (P) => {
    const d = 2 * 1e5 * 434.6 / P;                 // distance giving apparent P
    return S.splatFluxJS({ pos: [d, 0, 0], radius_m: 1e5,
      albedo: [0, 0, 0], emisSelf: [1, 1, 1], tempK: 5778,
      emitters: [], neeExp: 1, pxPerRad: 434.6 });
  };
  const a = o(2.0), b = o(1.0);
  const fluxA = a.n * a.n * a.rgb[0], fluxB = b.n * b.n * b.rgb[0];
  ok("splat flux dims by inverse square (P 2→1 quarters flux)",
     near(fluxA / fluxB, 4.0, 1e-6), `ratio=${fluxA / fluxB}`);
}

// ── 6. incandescence mirror sanity (ties splatFluxJS to the shared chunk) ─
{
  ok("incandescence below Draper point is zero",
     S.incandescenceJS(699, [1, 1, 1]).every(v => v === 0));
  const sun = S.incandescenceJS(5778, [0.5, 0.5, 0.5]);
  ok("5778 K glow: all channels lit, tone-capped ≤1.7, Wien peak in green",
     sun.every(v => v > 0 && v <= 1.7) && sun[1] >= sun[0] && sun[1] >= sun[2],
     JSON.stringify(sun));
  const ember = S.incandescenceJS(1000, [0.5, 0.5, 0.5]);
  ok("1000 K ember glow is red-dominant (r>g>b)",
     ember[0] > ember[1] && ember[1] > ember[2], JSON.stringify(ember));
}

// ── 7. pack determinism: two runs from the same state → byte-identical ────
{
  const mk = () => {
    const sp = new S.SGSplats();
    return sp.frame({
      eye: [2e11, 1e11, 5e10],
      poses: [[0, 0, 0], [1.5e11, 0, 0], [1.5e11, 3.8e8, 0], [7.8e11, 1e10, 0]],
      radius: [6.96e8, 6.371e6, 1.7374e6, 6.9911e7],
      emitterBody: new Set([0]),
      emitters: [{ body: 0, center: [0, 0, 0], radius: 8.0079e8 }],
      tanHalf: Math.tan(22.5 * Math.PI / 180), h: 360,
    });
  };
  const A = mk(), B = mk();
  ok("pack: bands byte-identical across two fresh runs",
     Buffer.from(A.bands.buffer).equals(Buffer.from(B.bands.buffer)));
  ok("pack: dyn byte-identical", Buffer.from(A.dyn.buffer).equals(Buffer.from(B.dyn.buffer)));
  ok("pack: vis byte-identical", Buffer.from(A.vis.buffer).equals(Buffer.from(B.vis.buffer)));
  ok("pack: counts sum to nb", A.counts[0] + A.counts[1] + A.counts[2] === 4);
}

// ── 8. occlusion cull: positive and negative cases ────────────────────────
{
  // eye at origin; a splat body at x=10 AU behind a big band-0 body at x=1 AU
  const sp = new S.SGSplats();
  const out = sp.frame({
    eye: [0, 0, 0],
    poses: [[1.5e11, 0, 0], [1.5e12, 0, 0], [1.5e12, 1e12, 0]],
    radius: [7e9, 1e5, 1e5],                       // big near body; two tiny far ones
    emitterBody: new Set(), emitters: [],
    tanHalf: 0.41421356, h: 360,
  });
  ok("occlusion setup: near body is band 0", out.bands[0] === 0,
     `bands=${Array.from(out.bands)}`);
  ok("occlusion: hidden splat culled (bit0=0)", (out.vis[1] & 1) === 0);
  ok("occlusion: visible splat drawn (bit0=1)", (out.vis[2] & 1) === 1);
  // emitter eclipse: splat–occluder–emitter collinear → emitter bit cleared
  const sp2 = new S.SGSplats();
  const out2 = sp2.frame({
    eye: [0, 1e12, 0],
    poses: [[0, 0, 0], [2e11, 0, 0], [4e11, 0, 0], [4e11, 5e11, 0]],
    radius: [6.96e8, 7e7, 1e5, 1e5],               // sun, eclipser, eclipsed splat, clear splat
    emitterBody: new Set([0]),
    emitters: [{ body: 0, center: [0, 0, 0], radius: 6.96e8 }],
    tanHalf: 0.41421356, h: 360,
  });
  ok("eclipse: shadowed splat loses the emitter bit", (out2.vis[2] & 2) === 0,
     `vis=${Array.from(out2.vis)} bands=${Array.from(out2.bands)}`);
  ok("eclipse: clear splat keeps the emitter bit", (out2.vis[3] & 2) === 2);
}

// ── 9. reset memorylessness + transition counting ─────────────────────────
{
  const sp = new S.SGSplats();
  const envAt = (d) => ({
    eye: [0, 0, 0], poses: [[d, 0, 0]], radius: [1e6],
    emitterBody: new Set(), emitters: [],
    tanHalf: 0.41421356237309503, h: 360,
  });
  const dFor = (P) => 2 * 1e6 * (360 / (2 * 0.41421356237309503)) / P;
  // approach: 1 px → 2.7 px (sticky-splat holds) → 10 px → 22 px (sticky-imp) → 30 px
  sp.frame(envAt(dFor(1.0)));
  ok("approach: starts splat", sp._bands[0] === 2);
  sp.frame(envAt(dFor(2.7)));
  ok("approach: 2.7px sticky keeps splat", sp._bands[0] === 2);
  sp.frame(envAt(dFor(10)));
  ok("approach: 10px → impostor", sp._bands[0] === 1);
  sp.frame(envAt(dFor(22)));
  ok("approach: 22px sticky keeps impostor", sp._bands[0] === 1);
  sp.frame(envAt(dFor(30)));
  ok("approach: 30px → raymarch", sp._bands[0] === 0);
  ok("approach: exactly two transitions counted", sp.transitions === 2
     && sp.transitionsByBody[0] === 2, `t=${sp.transitions}`);
  // retreat through the same points lands differently INSIDE sticky zones …
  sp.frame(envAt(dFor(22)));
  ok("retreat: 22px sticky keeps raymarch (path-dependent, by design)",
     sp._bands[0] === 0);
  // … but reset() forgets: 22 px with no history is an impostor
  sp.reset();
  sp.frame(envAt(dFor(22)));
  ok("reset: 22px memoryless → impostor (the init rule)", sp._bands[0] === 1);
  ok("reset: transition counter cleared", sp.transitions === 0);
}

// ── 10. bench harness sanity (deterministic, no RNG) ──────────────────────
{
  const sp = new S.SGSplats();
  const r = sp.bench(1000, 5);
  ok("bench returns timing stats", r.n === 1000 && r.reps === 5
     && r.median_ms >= 0 && r.p95_ms >= r.median_ms - 1e-9);
  // same synthetic env twice → identical band census (no hidden RNG)
  const a = new S.SGSplats(), b = new S.SGSplats();
  a.bench(500, 1); b.bench(500, 1);
  const fa = a.frame; // (bands live inside; compare via a fresh frame call)
  ok("bench is deterministic (same census both runs)",
     JSON.stringify(Array.from(a._bands)) === JSON.stringify(Array.from(b._bands)));
}

// ── 11. R4.3 lodCapFor: SVO per-body LOD cap (pure, hysteretic) ───────────
{
  const f = S.lodCapFor, L = 5;
  // no-history nominal: cap = ceil(L + log2(pxCell)), clamped
  ok("lodCap: 1px cell → full depth", f(1.0, L, -1) === L);
  ok("lodCap: 0.51px cell → full depth (fine bias)", f(0.51, L, -1) === L);
  ok("lodCap: 0.49px cell → one level up", f(0.49, L, -1) === L - 1);
  ok("lodCap: 4px cell clamps at L", f(4.0, L, -1) === L);
  ok("lodCap: vanishing cell clamps at 0", f(1e-9, L, -1) === 0);
  ok("lodCap: undefined prev = init", f(0.49, L, undefined) === L - 1);
  // effective footprint law: cell at the chosen cap always covers (0.5,1] px
  for (const px of [0.9, 0.45, 0.2, 0.11, 0.06, 0.031]) {
    const c = f(px, L, -1);
    const eff = px * Math.pow(2, L - c);
    ok(`lodCap: footprint (0.5,1] at px=${px}`,
       c === 0 || (eff > 0.5 && eff <= 1.0 + 1e-12), `cap=${c} eff=${eff}`);
  }
  // monotone non-decreasing in pxCell (no history)
  let last = -1, mono = true;
  for (let px = 0.01; px <= 2.0; px *= 1.07) {
    const c = f(px, L, -1);
    if (c < last) mono = false;
    last = c;
  }
  ok("lodCap: monotone in pxCell", mono);
  // hysteresis: sticky window is capF ∈ [prev-1.5, prev+0.5]
  // prev=L-1 nominal window is capF ∈ (L-2, L-1]; widened both ways:
  ok("lodCap: sticky above (capF=prev+0.4 holds)",
     f(Math.pow(2, -0.6), L, L - 1) === L - 1);      // capF = L-0.6
  ok("lodCap: releases above (capF=prev+0.6 → finer)",
     f(Math.pow(2, -0.4), L, L - 1) === L);          // capF = L-0.4 → ceil = L
  ok("lodCap: sticky below (capF=prev-1.4 holds)",
     f(Math.pow(2, -2.4), L, L - 1) === L - 1);      // capF = L-2.4
  ok("lodCap: releases below (capF=prev-1.6 → coarser)",
     f(Math.pow(2, -2.6), L, L - 1) === L - 2);      // capF = L-2.6 → ceil = L-2
  // slow drift can never oscillate: on a monotone px sweep the cap sequence
  // is monotone (each boundary crossed at most once per direction, no
  // flicker) — the property the ±0.5-level dead zone exists to guarantee
  {
    let prev = -1, upOK = true, downOK = true, lastCap = -1;
    for (let s = 0; s <= 200; s++) {                 // ascend, +1% steps
      prev = f(0.2 * Math.pow(1.01, s), L, prev);
      if (lastCap >= 0 && prev < lastCap) upOK = false;
      lastCap = prev;
    }
    for (let s = 200; s >= 0; s--) {                 // descend the same path
      prev = f(0.2 * Math.pow(1.01, s), L, prev);
      if (prev > lastCap) downOK = false;
      lastCap = prev;
    }
    ok("lodCap: monotone caps on monotone sweep (no oscillation)",
       upOK && downOK);
  }
  // deterministic: same inputs, same output (pure function, no state)
  ok("lodCap: pure", f(0.37, 7, 3) === f(0.37, 7, 3));
}

console.log(`splats.js node checks: ${checks - fails}/${checks} passed`);
process.exit(fails ? 1 : 0);
