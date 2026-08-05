/* splats.js — R3 band classifier + splat pack loop (PURE logic, no GL/DOM).
 *
 * Three concentric rendering bands selected by APPARENT ANGULAR SIZE, with
 * hysteresis — rendering-only LOD (standing doctrine): SIMULATION resolution
 * never couples to camera zoom; nothing in this file touches physics/sim
 * state. viewer.js consumes frame() every rAF; the node test loads this file
 * headlessly (camera.js precedent: zero DOM, zero GL, zero fetch).
 *
 *   band 0  raymarch   — the body's leaf stays in the path tracer's SDF
 *   band 1  impostor   — analytic sphere hit test in the PT's miss branch
 *   band 2  splat      — GL_POINTS into an offscreen target, composited
 *
 * ── DISCLOSED Choices (each with WHY) ─────────────────────────────────────
 * Thresholds 24 px↑ / 20 px↓ and 3.0 px↑ / 2.4 px↓ — 24 and 3 are the
 *   approved plan's band boundaries; the 17–25% dead zones are an order of
 *   magnitude above the ~1.7%/frame apparent-size change of R2's exponential
 *   approach (K_SPEED = 1.0 s⁻¹, camera.js), so hand jitter cannot oscillate
 *   a band while a deliberate approach crosses each boundary exactly once.
 *   2.4 (not 2.5) because below ~2.4 px the impostor disc covers < 4.5 px,
 *   where the PT's Monte-Carlo edge aliasing already exceeds the splat's
 *   quantization — the splat is the better estimator there.
 * Euclidean distance (not view-space z) for apparent size — rotation
 *   invariance: panning the camera can never reclassify a band, eliminating
 *   a whole flicker class structurally. Cost: ≤ 1.7× paraxial underestimate
 *   at fov-45 screen corners — an edge body promotes to a richer band a
 *   little late; impostor geometry stays exact regardless.
 * Init (no history) resolves sticky zones toward the SAFER band (non-emitter
 *   → impostor, emitter → raymarch): classification is then independent of
 *   approach direction and deterministic under R0 capture pins.
 * EMITTER RULE — a body whose leaf is NEE-registered (the Sun, and Venus at
 *   737 K > the 700 K Draper registration gate) NEVER enters the impostor
 *   band and is never gated out of the SDF in the 3–24 px regime: removing
 *   an emitter's geometry there is the one change that can move 2+-bounce
 *   light on near-field surfaces (rays that used to terminate on the Sun
 *   would escape). Below 2.4 px the emitter subtends sub-pixel solid angle
 *   from every surface and NEE carries all its cast light, so a splat is
 *   safe. Consequence (by construction): every possible impostor body has
 *   zero incandescence — no NEE double-count is possible in the impostor
 *   path.
 * No RNG, no wall clock in any output that feeds rendering: bands are a pure
 *   function of (eye, poses, prev-band state); the ms timing returned by
 *   frame() feeds ONLY the dbg line and window._r3.stats.
 */
"use strict";
(() => {

// band ids
const B_RAY = 0, B_IMP = 1, B_SPL = 2;
// hysteresis thresholds (Choices — WHY in the header)
const RAY_ENTER = 24.0, RAY_EXIT = 20.0;    // raymarch ↔ impostor
const SPL_EXIT  = 3.0,  SPL_ENTER = 2.4;    // impostor ↔ splat (emitter: raymarch ↔ splat)

// Apparent pixel DIAMETER of a body of radius r (m) at distance d (m), on a
// canvas h pixels tall with vertical half-fov tangent tanHalf. Paraxial
// center mapping — the same h/(2·tanHalf) px-per-radian the ray generator
// implies (dir = fwd + up·(uv.y·tanHalf), uv over h pixels).
function apparentPx(r, d, tanHalf, h) {
  return (2.0 * r / Math.max(d, 1e-9)) * h / (2.0 * tanHalf);
}

// non-emitter classifier (bands {0,1,2}); prev = previous band (0/1/2)
function bandFor(px, prev) {
  if (px >= RAY_ENTER) return B_RAY;
  if (px <  SPL_ENTER) return B_SPL;
  if (px >= RAY_EXIT)  return prev === B_RAY ? B_RAY : B_IMP;  // [20,24) sticky-raymarch
  if (px >= SPL_EXIT)  return B_IMP;                            // [3,20) always impostor
  return prev === B_SPL ? B_SPL : B_IMP;                        // [2.4,3) sticky-splat
}
// NEE-emitter classifier (bands {0,2} ONLY — the emitter rule, header)
function bandForEmitter(px, prev) {
  if (px >= SPL_EXIT) return B_RAY;
  if (px <  SPL_ENTER) return B_SPL;
  return prev === B_SPL ? B_SPL : B_RAY;                        // [2.4,3) sticky
}

// Lambert-sphere disc-averaged reflectance phase function:
//   f(α) = (2/(3π))·(sin α + (π − α)·cos α),  f(0) = 2/3.
// This is the CLOSED-FORM EXPECTATION of the path tracer's own NEE cone
// estimator (alb·Le·ndl·G·neeExp) averaged over the body's camera-visible
// disc — verified numerically to 4+ digits at α = 0/60/90/120° (session
// evidence in the R3 report). NOT the (sin α + (π−α)cos α)/π variant, which
// is 1.5× too bright. House precedent for flux-at-observer reasoning:
// field/interface/detection.py's inverse-square derivation.
function lambertPhase(a) {
  return (2.0 / (3.0 * Math.PI)) * (Math.sin(a) + (Math.PI - a) * Math.cos(a));
}

// JS mirror of the ONE shared GLSL incandescence chunk (viewer.js
// INCANDESCENCE_GLSL): Planck × Kirchhoff × EMISSION_SCALE, Reinhard ×1.7.
// Used by the node continuity tests and splatFluxJS — statement-for-
// statement with the GLSL so the two can never disagree on the derivation.
const EMISSION_SCALE = 2400000000.0;
function incandescenceJS(T, emis) {
  if (T < 700.0) return [0, 0, 0];                 // Draper point
  const lam = [650e-9, 550e-9, 450e-9], e = [0, 0, 0];
  let m = 0;
  for (let c = 0; c < 3; c++) {
    const x = 1.4388e-2 / (lam[c] * T);            // c2/(λT)
    const L = emis[c] / (Math.pow(lam[c] / 650e-9, 5) * (Math.exp(x) - 1));
    e[c] = L * EMISSION_SCALE;
    if (e[c] > m) m = e[c];
  }
  // NON-DERIVED (audit): Reinhard + 1.7 gain is an exposure choice (the
  // shared chunk's own flag) — mirrored verbatim, never re-derived here.
  return [e[0] / (1 + m) * 1.7, e[1] / (1 + m) * 1.7, e[2] / (1 + m) * 1.7];
}

// integer sprite size (Choice: deterministic rasterization under R0's
// 3-render proof; ceil ⇒ sprite ≥ disc so the flux clamp never truncates)
function spriteSize(P) {
  return Math.min(3, Math.max(1, Math.ceil(P)));
}

// JS mirror of the splat vertex shader's closed-form radiance — the node
// continuity tests tie THIS to lambertPhase/incandescenceJS, and the R3
// report ties the GLSL to this (same expressions, same clamps).
//   o = { pos:[x,y,z] (eye-rebased), radius_m, albedo:[3], emisSelf:[3],
//         tempK, emitters:[{pos:[3] rebased, radius, temp, emis:[3],
//         vis:0|1}], neeExp, pxPerRad, uTemp }
// Flux conservation: total sprite flux = disc radiance × (π/4)P² — a
// receding body dims by exact inverse-square (P ∝ 1/d), never pops.
function splatFluxJS(o) {
  const d = Math.hypot(o.pos[0], o.pos[1], o.pos[2]);
  const P = (2.0 * o.radius_m / Math.max(d, 1e-9)) * o.pxPerRad;
  const n = spriteSize(P);
  const uT = o.uTemp || 0;
  const Lr = [0, 0, 0];
  for (const em of (o.emitters || [])) {
    const wc = [em.pos[0] - o.pos[0], em.pos[1] - o.pos[1], em.pos[2] - o.pos[2]];
    const dbe = Math.hypot(wc[0], wc[1], wc[2]);
    if (dbe <= em.radius * 1.001) continue;        // self / inside: no self-reflection
    const sinT = em.radius / dbe;
    const cosTm = Math.sqrt(Math.max(0, 1 - sinT * sinT));
    const G = 2.0 * (1.0 - cosTm);                 // the NEE block's own solid-angle factor
    const Le = incandescenceJS(em.temp + uT, em.emis);
    // phase angle at the body between dir-to-emitter and dir-to-camera
    const ca = Math.max(-1, Math.min(1,
      (wc[0] * -o.pos[0] + wc[1] * -o.pos[1] + wc[2] * -o.pos[2]) / (dbe * d)));
    const f = lambertPhase(Math.acos(ca));
    const v = em.vis ? 1 : 0;
    for (let c = 0; c < 3; c++) Lr[c] += o.albedo[c] * Le[c] * G * o.neeExp * f * v;
  }
  const Lth = incandescenceJS(o.tempK + uT, o.emisSelf);
  const spread = Math.min(1.0, (Math.PI / 4) * P * P / (n * n));
  const rgb = [0, 0, 0];
  for (let c = 0; c < 3; c++)
    rgb[c] = Math.min(16.0, (Lr[c] + Lth[c]) * spread);   // 16 = the PT's own firefly clamp
  return { P, n, rgb };
}

// segment a→b vs sphere (c, r): true when the OPEN segment passes through
// the sphere (occluder strictly between the endpoints).
function segOccluded(a, b, c, r) {
  const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
  const L = Math.hypot(dx, dy, dz);
  if (!(L > 0)) return false;
  const ux = dx / L, uy = dy / L, uz = dz / L;
  const ox = c[0] - a[0], oy = c[1] - a[1], oz = c[2] - a[2];
  const tca = ox * ux + oy * uy + oz * uz;
  if (tca <= 0 || tca >= L) return false;
  const d2 = (ox * ox + oy * oy + oz * oz) - tca * tca;
  return d2 < r * r;
}

const _now = (typeof performance !== "undefined" && performance.now)
  ? () => performance.now() : () => Date.now();

class SGSplats {
  constructor() { this.reset(); }

  // Reset ALL hysteresis state — viewer.js calls this on EVERY sceneId bump
  // (scene load, locateBody, capture-pin application, opacity rebuild …) so
  // a pinned R0 capture classifies MEMORYLESSLY from the pinned camera.
  reset() {
    this._bands = null; this._nb = 0;
    this.transitions = 0;                      // total band changes since reset
    this.transitionsByBody = null;             // Int32Array per body (S5 flicker proof)
  }

  /* Classify + occlude + pack one frame.
   * env = {
   *   eye:[x,y,z] float64 world;  poses:[[x,y,z]..] world per body;
   *   radius:[m..] per body (0/absent → the body never bands: stays band 0);
   *   emitterBody:Set|Array of NEE-registered body indices (the emitter rule);
   *   emitters:[{body, center:[x,y,z] world LIVE, radius}] — the SAME
   *     centres/radii the viewer's NEE upload uses (bounds radius, never
   *     radius_m — one emitter definition, no parallel math);
   *   tanHalf, h — the draw() camera terms.
   * }
   * → { bands:Int8Array, counts:[n0,n1,n2], px:Float64Array,
   *     dyn:Float32Array(nb*3) eye-rebased positions (float64 subtract →
   *     f32 cast, the R1-lite discipline),
   *     vis:Uint8Array(nb): bit0 = draw splat (band 2 ∧ eye-unoccluded);
   *     bit(k+1) = emitter k visible from the body (eclipse test),
   *     ms (dbg/stats ONLY — never fed to rendering), transitions }
   */
  frame(env) {
    const t0 = _now();
    const poses = env.poses, nb = poses.length, eye = env.eye;
    if (!this._bands || this._nb !== nb) {
      this._bands = new Int8Array(nb).fill(-1);  // -1 = no history (init rule)
      this.transitionsByBody = new Int32Array(nb);
      this._nb = nb;
    }
    const bands = this._bands;
    const pxScale = env.h / (2.0 * env.tanHalf);
    const emSet = env.emitterBody instanceof Set
      ? env.emitterBody : new Set(env.emitterBody || []);
    const counts = [0, 0, 0];
    const dyn = new Float32Array(nb * 3);
    const vis = new Uint8Array(nb);
    const px = new Float64Array(nb);
    // ── pass 1: rebase + classify ─────────────────────────────────────────
    for (let k = 0; k < nb; k++) {
      const p = poses[k];
      const dx = p[0] - eye[0], dy = p[1] - eye[1], dz = p[2] - eye[2]; // float64
      dyn[k * 3] = dx; dyn[k * 3 + 1] = dy; dyn[k * 3 + 2] = dz;       // f32 cast
      const r = env.radius[k] || 0;
      let b = B_RAY;
      if (r > 0) {
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const Pk = (2.0 * r / Math.max(d, 1e-9)) * pxScale;
        px[k] = Pk;
        const prev = bands[k];
        b = emSet.has(k)
          ? bandForEmitter(Pk, prev < 0 ? B_RAY : prev)   // init → safer band
          : bandFor(Pk, prev < 0 ? B_IMP : prev);
        if (prev >= 0 && prev !== b) { this.transitions++; this.transitionsByBody[k]++; }
      }
      bands[k] = b; counts[b]++;
    }
    // ── pass 2: occlusion for splat-band bodies ──────────────────────────
    // Occluder candidate list built ONCE (bands make it small by design):
    // band-0/1 bodies always; band-2 bodies are included too while nb is
    // small (≤ 256) so the 30-body theater tests against ALL spheres
    // exactly; at splat-buffer populations (10k/100k synthetic) restricting
    // to band-0/1 + emitter bodies keeps the loop O(n·small) — a disclosed
    // scaling Choice (splat-vs-splat eclipses at that population are below
    // one LSB of the additive target anyway).
    const ems = env.emitters || [];
    const occ = [];
    const allOcc = nb <= 256;
    for (let i = 0; i < nb; i++) {
      const r = env.radius[i] || 0;
      if (!(r > 0)) continue;
      if (allOcc || bands[i] !== B_SPL || emSet.has(i)) occ.push(i);
    }
    for (let k = 0; k < nb; k++) {
      if (bands[k] !== B_SPL) continue;
      let v = 1;                                   // bit0: draw
      for (let j = 0; j < occ.length; j++) {       // eye → splat vs band-0/1 geometry
        const i = occ[j];
        if (i === k || bands[i] === B_SPL) continue;   // splat-vs-splat: additive-only, disclosed
        if (segOccluded(eye, poses[k], poses[i], env.radius[i])) { v = 0; break; }
      }
      for (let e = 0; e < ems.length; e++) {       // body → emitter eclipse test
        const em = ems[e];
        if (em.body === k) continue;               // self: no bit (the VS self-skips too)
        let ev = 1;
        for (let j = 0; j < occ.length; j++) {
          const i = occ[j];
          if (i === k || i === em.body) continue;
          if (segOccluded(poses[k], em.center, poses[i], env.radius[i])) { ev = 0; break; }
        }
        if (ev) v |= (1 << (e + 1));
      }
      vis[k] = v;
    }
    const ms = _now() - t0;
    return { bands, counts, px, dyn, vis, ms, transitions: this.transitions };
  }

  // Deliverable-C microbench: index-derived golden-spiral positions (NO RNG,
  // fully deterministic), one Sun-like emitter at the origin, camera 2.2 AU
  // out. Measures the classify+occlude+pack loop exactly as shipped.
  bench(n, reps) {
    reps = reps || 100;
    const poses = new Array(n), radius = new Array(n);
    const GA = Math.PI * (3 - Math.sqrt(5));       // golden angle
    for (let i = 0; i < n; i++) {
      const z = 1 - 2 * (i + 0.5) / n, rho = Math.sqrt(Math.max(0, 1 - z * z));
      const th = GA * i;
      const R = 7.5e11 * (0.2 + 0.8 * ((i % 97) / 97));  // 1.5e11–7.5e11 m shells
      poses[i] = [R * rho * Math.cos(th), R * rho * Math.sin(th), R * z];
      radius[i] = 1e5 + (i % 13) * 3e4;            // 100–460 km bodies
    }
    poses[0] = [0, 0, 0]; radius[0] = 6.9634e8;    // a Sun (emitter, band 0)
    const env = {
      eye: [2e11, 1e11, 5e10], poses, radius,
      emitterBody: new Set([0]),
      emitters: [{ body: 0, center: [0, 0, 0], radius: 8.0079e8 }],
      tanHalf: Math.tan(22.5 * Math.PI / 180), h: 1080,
    };
    const times = [];
    this.reset();
    for (let r = 0; r < reps; r++) times.push(this.frame(env).ms);
    times.sort((a, b) => a - b);
    return { n, reps, median_ms: times[reps >> 1],
             p95_ms: times[Math.min(reps - 1, Math.floor(reps * 0.95))],
             min_ms: times[0], max_ms: times[reps - 1] };
  }
}

const api = { SGSplats, apparentPx, bandFor, bandForEmitter, lambertPhase,
              incandescenceJS, splatFluxJS, spriteSize, segOccluded,
              RAY_ENTER, RAY_EXIT, SPL_EXIT, SPL_ENTER, EMISSION_SCALE };
if (typeof window !== "undefined") window.SGSplats = api;
if (typeof module !== "undefined" && module.exports) module.exports = api;
})();
