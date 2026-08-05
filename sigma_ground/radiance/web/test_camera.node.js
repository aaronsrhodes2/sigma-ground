/* test_camera.node.js — THE headless node test camera.js's header promises.
 *
 * Run:  node sigma_ground/radiance/web/test_camera.node.js   (exit 0 = pass)
 * Or via pytest: test_camera.py shells out here (skips when node is absent —
 * environment-gated, the same honesty class as the network-gated ephemeris
 * test).
 *
 * The reference eye below is an INDEPENDENT float64 rewrite of viewer.js's
 * eyePos/frameBasis (different code style on purpose — a shared bug can't
 * hide in shared code). TOLERANCE DOCTRINE: every mode switch must
 * preserve the eye to 1e-9 RELATIVE TO THE PARAMETERIZATION SCALE
 * R = |eye − target| — the (az, el, R) trig round-trip is ~2 ulp OF THE
 * DISTANCE, so an eye 0.3 m from the origin following a body 1.5e11 m
 * away legitimately moves ~3e-5 m in absolute terms (2e-16 of R, ten
 * orders inside any pixel). Asserting 1e-9 of |eye| there would demand
 * more precision than float64 trig HAS at that distance.
 */
"use strict";
const path = require("path");
const { SGCamera, K_SPEED, D_FLOOR, STEP_CAP, MOVE_HOLD_S } =
  require(path.join(__dirname, "camera.js"));

// ── independent reference math ───────────────────────────────────────────
function refEye(cam) {
  const u = cam.up, L = Math.hypot(u[0], u[1], u[2]) || 1;
  const U = [u[0] / L, u[1] / L, u[2] / L];
  const H = Math.abs(U[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1];
  let A = [U[1] * H[2] - U[2] * H[1], U[2] * H[0] - U[0] * H[2],
           U[0] * H[1] - U[1] * H[0]];
  const La = Math.hypot(A[0], A[1], A[2]) || 1;
  A = [A[0] / La, A[1] / La, A[2] / La];
  const B = [U[1] * A[2] - U[2] * A[1], U[2] * A[0] - U[0] * A[2],
             U[0] * A[1] - U[1] * A[0]];
  const ce = Math.cos(cam.el), se = Math.sin(cam.el);
  const ca = Math.cos(cam.az), sa = Math.sin(cam.az);
  return [0, 1, 2].map((i) =>
    cam.target[i] + (A[i] * ca * ce + B[i] * sa * ce + U[i] * se) * cam.radius);
}
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mag = (a) => Math.hypot(a[0], a[1], a[2]);

let fails = 0, checks = 0;
function ok(name, cond, detail) {
  checks++;
  if (!cond) { fails++; console.log(`FAIL  ${name}  ${detail || ""}`); }
}
// eye continuity at the parameterization scale (see header)
function eyeOK(name, a, b, cam) {
  const scale = Math.max(mag(sub3(a, cam.target)), mag(a), 1e-30);
  const rel = mag(sub3(a, b)) / scale;
  ok(name, rel < 1e-9, `rel=${rel.toExponential(2)}`);
}

function mkEnv(bodies, poses, diag, nowRef) {
  return {
    bodies: () => bodies,
    poses: () => poses,
    bodyIndexByLabel: (n) => bodies.findIndex((b) => b.label === n),
    bodyRadius: (i) => bodies[i].radius_m || 1.0,
    sceneDiag: () => diag,
    now: () => nowRef.t,          // the input.js test-seam idiom
  };
}

// ═══ 1. transition continuity — every path, hostile geometries ════════════
const AU = 1.495978707e11;
const R_E = 6.371e6;
const earthPos = [1.4959787e11, 2.1e10, -4.4e10];
const cases = [
  { name: "tabletop",       az: 0.8,  el: 0.45, radius: 0.3,    target: [0, 0, 0] },
  { name: "el +clamp",      az: -2.1, el: 1.5,  radius: 12.0,   target: [3, -4, 5] },
  { name: "el -clamp",      az: 5.9,  el: -1.5, radius: 7.7,    target: [-2, 8, 1] },
  { name: "radius 1mm",     az: 1.234, el: -0.7, radius: 0.001, target: [0.5, 0.25, -0.125] },
  { name: "radius 5AU",     az: 2.5,  el: 0.9,  radius: 5 * AU, target: [0, 0, 0] },
  { name: "solar target",   az: 0.31, el: -1.1, radius: 2.0e7,  target: earthPos.slice() },
  { name: "5AU+far target", az: 4.2,  el: 1.5,  radius: 5 * AU, target: [1.5e11, -9e10, 3e10] },
];
for (const c of cases) {
  const bodies = [{ label: "Sun", radius_m: 6.9634e8 },
                  { label: "Earth", radius_m: R_E }];
  const poses = [{ pos: [0, 0, 0] }, { pos: earthPos }];
  const nowRef = { t: 0 };
  const cam = { az: c.az, el: c.el, radius: c.radius,
                target: c.target.slice(), up: [0, 0, 1] };
  const ctl = new SGCamera(cam, mkEnv(bodies, poses, 3e11, nowRef));
  const eye0 = refEye(cam);

  ctl.setMode("fly");                              // orbit → fly: identity
  eyeOK(`[${c.name}] orbit->fly`, eye0, ctl.state().eye, cam);
  ctl.tick(1 / 60);                                // no-input tick: eye fixed,
  eyeOK(`[${c.name}] fly tick fixed`, eye0, ctl.state().eye, cam);
  eyeOK(`[${c.name}] fly projection`, eye0, refEye(cam), cam);  // cam mirrors _eye
  ctl.setMode("orbit");                            // fly → orbit: re-anchor
  eyeOK(`[${c.name}] fly->orbit`, eye0, refEye(cam), cam);

  const eyePre = refEye(cam);                      // orbit → follow: entry solve
  ok(`[${c.name}] follow accepted`, ctl.follow("Earth") === true);
  eyeOK(`[${c.name}] orbit->follow`, eyePre, refEye(cam), cam);
  ok(`[${c.name}] follow target=body`,
     mag(sub3(cam.target, earthPos)) <= 1e-6 * mag(earthPos));

  const eyeChain = refEye(cam);                    // follow→fly→follow→orbit
  ctl.setMode("fly");
  eyeOK(`[${c.name}] follow->fly`, eyeChain, ctl.state().eye, cam);
  ctl.follow("Sun");
  eyeOK(`[${c.name}] fly->follow`, eyeChain, refEye(cam), cam);
  ctl.setMode("orbit");
  eyeOK(`[${c.name}] follow->orbit`, eyeChain, refEye(cam), cam);
}
// fly→orbit with every body BEHIND the look ray (the t<0 branch)
{
  const nowRef = { t: 0 };
  const cam = { az: 0, el: 0, radius: 10, target: [0, 0, 0], up: [0, 0, 1] };
  const ctl = new SGCamera(cam,
    mkEnv([{ label: "Ball", radius_m: 1 }], [{ pos: [100, 0, 0] }], 50, nowRef));
  ctl.setMode("fly");
  ctl.drag(314159, 0);                             // spin the look away
  const eye0 = ctl.state().eye.slice();
  ctl.setMode("orbit");
  eyeOK("fly->orbit body-behind", eye0, refEye(cam), cam);
  ok("fly->orbit radius > 0", cam.radius > 0);
}

// ═══ 2. speed law — playable numbers, floor, never NaN ════════════════════
{
  const nowRef = { t: 0 };
  const cam = { az: 0, el: 0, radius: 1, target: [0, 0, 0], up: [0, 0, 1] };
  const ctl = new SGCamera(cam,
    mkEnv([{ label: "Earth", radius_m: R_E }], [{ pos: [0, 0, 0] }], 1e12, nowRef));
  ctl.setMode("fly");
  const at = (d) => { ctl._eye = [R_E + d, 0, 0]; return ctl.speedNow(); };
  ok("speed @100 m = 100 m/s", Math.abs(at(100) - 100) < 1e-6);
  ok("speed @100 km = 1e5 m/s", Math.abs(at(1e5) - 1e5) < 1e-3);
  ok("speed @1 AU = 1 AU/s", Math.abs(at(AU) / AU - 1) < 1e-12);
  ok("floor @0.1 m", at(0.1) === K_SPEED * D_FLOOR);
  ok("floor inside body", at(-1000) === K_SPEED * D_FLOOR);
  const tCross = Math.log(AU / 100) / K_SPEED;     // exponential approach
  ok("playable: 1AU->100m ≈ 21 s", tCross > 15 && tCross < 30);
}
{ // body-less scene: sceneDiag fallback, cycle skips follow, all finite
  const nowRef = { t: 0 };
  const cam = { az: 0.8, el: 0.45, radius: 0.3, target: [0, 0, 0], up: [0, 0, 1] };
  const ctl = new SGCamera(cam, mkEnv([], [], 2.5, nowRef));
  ctl.setMode("fly");
  ok("empty: clearance = diag", ctl.clearance() === 2.5);
  ok("empty: speed finite", Number.isFinite(ctl.speedNow()) && ctl.speedNow() > 0);
  const ctl2 = new SGCamera({ az: 0, el: 0, radius: 1, target: [0, 0, 0],
                              up: [0, 0, 1] }, mkEnv([], [], 0, nowRef));
  ctl2.setMode("fly");
  ok("zero-diag: speed floored", ctl2.speedNow() === K_SPEED * D_FLOOR);
  ok("empty: follow refuses", ctl.follow(null) === false);
  ctl.setMode("orbit"); ctl.cycleMode();
  ok("empty: cycle -> fly", ctl.state().mode === "fly");
  ctl.tick(1 / 60);
  ok("empty: tick finite", ctl.state().eye.every(Number.isFinite));
}

// ═══ 3. step cap — the no-tunnel invariant under hostile dt ═══════════════
{
  const nowRef = { t: 1000 };
  const cam = { az: 0, el: 0, radius: 1, target: [0, 0, 0], up: [0, 0, 1] };
  const ctl = new SGCamera(cam,
    mkEnv([{ label: "Earth", radius_m: R_E }], [{ pos: [0, 0, 0] }], 1e12, nowRef));
  ctl.setMode("fly");
  ctl._eye = [R_E + 1e7, 0, 0];
  ctl._lookAz = -Math.PI / 2; ctl._lookEl = 0;     // forward = −x, at the planet
  ctl.k_throttle = 1e3;                            // K_MAX
  let violated = false, worstCross = 0;
  let prevEye = ctl._eye.slice(), prevClear = ctl.clearance();
  for (let i = 0; i < 400; i++) {
    nowRef.t += 50;
    ctl.handleIntent({ op: "cam_move", args: { dz: 1 } });
    ctl.tick(1.0);                                 // pathological 1 s frames
    const step = mag(sub3(ctl._eye, prevEye));
    const cap = STEP_CAP * Math.max(prevClear, D_FLOOR);
    const slack = 8 * 2.3e-16 * Math.max(mag(prevEye), 1);  // stored-eye ulp
    if (step > cap + slack) violated = true;
    const cl = ctl.clearance();
    if (prevClear >= 0 && cl < 0) worstCross = Math.max(worstCross, -cl);
    prevEye = ctl._eye.slice(); prevClear = cl;
  }
  ok("cap honored every hostile tick", !violated);
  // first outside→inside crossing ≤ 0.9·D_FLOOR (deliberate SLOW entry into
  // a body is disclosed design — D_FLOOR's "escape at floor speed" note —
  // tunnelling THROUGH surfaces from distance is what the cap forbids)
  ok("surface crossing ≤ 0.45 m", worstCross <= STEP_CAP * D_FLOOR + 1e-9,
     `worst=${worstCross}`);
  ok("hostile ticks stay finite", ctl.state().eye.every(Number.isFinite));
}

// ═══ 4. cam_move merge + decay (the per-component hold contract) ══════════
{
  const nowRef = { t: 0 };
  const cam = { az: 0, el: 0, radius: 5, target: [0, 0, 0], up: [0, 0, 1] };
  const ctl = new SGCamera(cam,
    mkEnv([{ label: "B", radius_m: 1 }], [{ pos: [1000, 0, 0] }], 100, nowRef));
  ctl.setMode("fly");
  ctl.handleIntent({ op: "cam_move", args: { dx: 1 } });
  nowRef.t = 20;
  ctl.handleIntent({ op: "cam_move", args: { dz: 1 } });
  ok("merge: dz never clobbers held dx", ctl._held.dx.v === 1 && ctl._held.dz.v === 1);
  nowRef.t = 90;
  const e0 = ctl.state().eye.slice();
  ctl.tick(0.001);
  ok("held components move the eye", mag(sub3(ctl.state().eye, e0)) > 0);
  nowRef.t = 90 + MOVE_HOLD_S * 1000 + 50;         // both stale
  ctl.tick(0.001);
  ok("decay zeroes stale components", ctl._held.dx.v === 0 && ctl._held.dz.v === 0);
  ctl.handleIntent({ op: "cam_move", args: { dx: NaN, dy: Infinity, dz: "1" } });
  ok("hostile args ignored", ctl._held.dx.v === 0 && ctl._held.dy.v === 0
     && ctl._held.dz.v === 0);
  ctl.handleIntent({ op: "cam_move", args: { dz: 7 } });
  ok("deflection clamped to [-1,1]", ctl._held.dz.v === 1);
}

// ═══ 5. zero regression — orbit mode is the E3 formulas, exactly (===) ════
{
  const nowRef = { t: 0 };
  const cam = { az: 0.8, el: 0.45, radius: 0.3, target: [0, 0, 0], up: [0, 0, 1] };
  const ctl = new SGCamera(cam, mkEnv([], [], 2.5, nowRef));
  let az = 0.8, el = 0.45, r = 0.3;                // viewer.js's shipped lines
  for (const [dx, dy] of [[3, -2], [-120, 55], [400, -900], [7, 1000]]) {
    ctl.drag(dx, dy);
    az -= dx * 0.01; el = Math.max(-1.5, Math.min(1.5, el + dy * 0.01));
  }
  ok("drag az bit-identical", cam.az === az);
  ok("drag el bit-identical (clamped)", cam.el === el);
  for (const dY of [120, -360, 1, 5000]) { ctl.wheel(dY); r *= Math.exp(dY * 0.0012); }
  ok("wheel radius bit-identical", cam.radius === r);
  ctl.handleIntent({ op: "cam_orbit", args: { daz: 0.1, delv: 2.0 } });
  az += 0.1; el = Math.max(-1.5, Math.min(1.5, el + 2.0));
  ok("cam_orbit intent bit-identical", cam.az === az && cam.el === el);
  ctl.handleIntent({ op: "cam_zoom", args: { factor: 1.03 } });
  r *= 1.03;
  ok("cam_zoom intent bit-identical", cam.radius === r);
  const snap = JSON.stringify(cam);
  ctl.handleIntent({ op: "cam_move", args: { dz: 1 } });
  ctl.tick(1 / 60);
  ok("cam_move is a no-op in orbit", JSON.stringify(cam) === snap);
  ctl.tick(0.5);
  ok("orbit tick never writes cam", JSON.stringify(cam) === snap);
}

// ═══ 6. follow — live pose, copy-not-alias, honest degradations ═══════════
{
  const nowRef = { t: 0 };
  const bodies = [{ label: "Moon", radius_m: 1.7374e6 }];
  const box = { arr: [{ pos: [1e8, 0, 0] }] };
  const env = mkEnv(bodies, box.arr, 1e9, nowRef);
  env.poses = () => box.arr;                       // live closure, like viewer.js
  const cam = { az: 0, el: 0, radius: 1e7, target: [0, 0, 0], up: [0, 0, 1] };
  const ctl = new SGCamera(cam, env);
  ok("follow adopts", ctl.follow("Moon") === true);
  box.arr = [{ pos: [1.001e8, 5e5, -2e5] }];       // the body moved THIS frame
  ctl.tick(1 / 60);
  ok("tracks this frame's pose",
     mag(sub3(cam.target, [1.001e8, 5e5, -2e5])) === 0);
  ok("target is a copy, not an alias", cam.target !== box.arr[0].pos);
  const snap = JSON.stringify(cam);
  bodies.length = 0;                               // label vanished
  ctl.tick(1 / 60);
  ok("label gone -> orbit in place",
     ctl.state().mode === "orbit" && JSON.stringify(cam) === snap);
  bodies.push({ label: "Moon", radius_m: 1.7374e6 });
  box.arr = [];                                    // static scene: no poses
  ctl.followAdopt("Moon");
  const snap2 = JSON.stringify(cam);
  ctl.tick(1 / 60);
  ok("static scene: follow holds still", JSON.stringify(cam) === snap2);
}

// ═══ 7. cycle order + resetToOrbit ════════════════════════════════════════
{
  const nowRef = { t: 0 };
  const cam = { az: 0.1, el: 0.2, radius: 5, target: [0, 0, 0], up: [0, 0, 1] };
  const ctl = new SGCamera(cam,
    mkEnv([{ label: "B", radius_m: 1 }], [{ pos: [50, 0, 0] }], 100, nowRef));
  ctl.cycleMode();
  ok("cycle: orbit->follow(nearest)",
     ctl.state().mode === "follow" && ctl.state().followLabel === "B");
  ctl.cycleMode();
  ok("cycle: follow->fly", ctl.state().mode === "fly");
  ctl.cycleMode();
  ok("cycle: fly->orbit", ctl.state().mode === "orbit");
  ctl.handleIntent({ op: "cam_mode", args: { mode: "fly" } });
  ctl.k_throttle = 500;
  ctl.resetToOrbit();
  ok("resetToOrbit: mode+throttle+held",
     ctl.state().mode === "orbit" && ctl.k_throttle === K_SPEED
     && ctl._held.dx.v === 0 && ctl._held.dy.v === 0 && ctl._held.dz.v === 0);
}

console.log(`camera.js node checks: ${checks - fails}/${checks} passed`);
process.exit(fails ? 1 : 0);
