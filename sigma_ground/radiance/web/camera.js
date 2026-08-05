/* camera.js — SGCamera: the R2 camera CONTROLLER (orbit / follow / fly).
 *
 * DOCTRINE (standing, 2026-08-03): cameras are ENGINE CITIZENS, never
 * world entities — no Matter, never an AppliedForce target, camera state
 * can NEVER touch physics. The server has no camera (cockpit refuses all
 * cam_* verbs as client-local). And the Captain's real-scale navigation
 * doctrine: distances stay REAL and undistorted; NAVIGATION adapts —
 * that is what the distance-proportional fly speed below exists for.
 *
 * OWNERSHIP (disclosed design call): there is ONE camera state — the
 * `cam` object viewer.js declares — held here BY REFERENCE, never
 * copied. In orbit/follow mode cam's fields (az/el/radius/target) are
 * authoritative. In fly mode the controller's float64 (_eye, _lookAz,
 * _lookEl) is authoritative and cam's fields are a per-tick DERIVED VIEW
 * (see tick()'s projection) — mode-dependent authority, never two
 * mirrors. DOM handlers stay in viewer.js and call controller methods
 * (drag/wheel), mirroring input.js's attach pattern: this file has ZERO
 * DOM listeners and ZERO fetch, so the headless node test needs no stubs.
 *
 * The ~10-line vector kit below is duplicated from viewer.js — accepted
 * duplication so this file loads headlessly under node (pinned; keep the
 * two in sync by eye, they are one line each).
 */
"use strict";
(() => {

// ── DISCLOSED Choices (R2) — every tuning constant, each with WHY ─────────
// K_SPEED = 1.0 s⁻¹ — full stick closes ~100% of the current clearance per
//   second: an exponential approach (the gap halves every ~0.7 s; the
//   per-frame step is ≈1.7% of the gap at 60 fps). Worked numbers on the
//   real solar theater: 100 km above Earth → 1.0e5 m/s; halfway
//   Earth→Moon → ≈1.9e8 m/s, self-braking on approach; ~1 AU from the
//   Sun's surface → ≈1 AU/s. NOT a physical velocity: the camera is an
//   engine citizen, not matter — no relativity claim attaches (between
//   planets this exceeds c, honestly, because nothing physical moves).
const K_SPEED = 1.0;
// D_FLOOR = 0.5 m — v_min = 0.5 m/s: tabletop scenes (plate/table,
//   0.1–1 m bodies) stay controllable and you are never stuck; it also
//   clamps clearance() < 0 (eye inside a body — you escape at floor
//   speed) and is the minimum fly-mode cam.radius, keeping the
//   view-scale-adaptive epsilons sane right at a surface. Chosen over
//   2 m for tabletop compatibility.
const D_FLOOR = 0.5;
// STEP_CAP = 0.9 — max per-frame displacement as a fraction of the
//   current clearance: the no-tunnel invariant, independent of throttle
//   and dt (at k_throttle=100 the raw step is 1.67× the gap per 60 fps
//   frame, and one stalled 1 s frame at k=1 crosses the whole gap —
//   the cap restores monotone approach for ANY k, any dt spike).
const STEP_CAP = 0.9;
// K_RANGE — fly-throttle clamp; the step cap makes the top end safe.
const K_MIN = 1e-3, K_MAX = 1e3;
// MOVE_HOLD_S = 0.1 s — held-move-component decay window, ≈ six 60 Hz
//   emission gaps: release coasts ≤100 ms (the client twin of the
//   analog-stick doctrine's release-decays-naturally). A client below
//   ~15 fps stutter-moves (its emission gap exceeds the window) —
//   disclosed and raisable, never raised silently.
const MOVE_HOLD_S = 0.1;
// DRAG_GAIN / WHEEL_GAIN / EL_CLAMP — inherited VERBATIM from viewer.js's
//   shipped mouse handlers (0.01 rad/px, exp(deltaY*0.0012), ±1.5 rad):
//   zero-regression constants, not new Choices. sin(1.5)=0.99749 < 0.999,
//   so inside the clamp draw()'s anti-gimbal up-flip never fires — the
//   pole is unreachable by pitch (yaw around instead, same as orbit).
const DRAG_GAIN = 0.01;
const WHEEL_GAIN = 0.0012;
const EL_CLAMP = 1.5;

// ── vector kit (duplicated from viewer.js — see header) ──────────────────
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1],
                         a[2] * b[0] - a[0] * b[2],
                         a[0] * b[1] - a[1] * b[0]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const norm = (a) => { const L = len(a) || 1; return [a[0] / L, a[1] / L, a[2] / L]; };
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// IDENTICAL operation order to viewer.js's frameBasis/eyePos — pinned:
// the identity mode transitions below depend on reproducing the orbit
// eye bit-for-bit (well, to ~2 ulp), so do NOT "simplify" these.
function frameBasis(up) {
  const U2 = norm(up), H = Math.abs(U2[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1];
  const A = norm(cross(U2, H)), B = cross(U2, A);
  return [A, B, U2];
}

class SGCamera {
  // cam: THE shared state object (viewer.js's `const cam`), by reference.
  // env: lazy closures over viewer module state —
  //   poses()            → live per-body poses ([{pos,...}])
  //   bodies()           → scene.bodies (labels; may be [])
  //   bodyIndexByLabel(n)→ index or -1
  //   bodyRadius(i)      → body radius in metres (locateBody's fallback chain)
  //   sceneDiag()        → scene bbox diagonal (clearance fallback for
  //                        body-less scenes — scene-scale speed, never Infinity)
  //   now()              → ms clock (OPTIONAL test seam, same idiom as
  //                        input.js's `now` constructor option)
  constructor(cam, env) {
    this.cam = cam;
    this.env = env || {};
    this.mode = "orbit";
    this.followLabel = null;
    this.k_throttle = K_SPEED;
    // fly-mode authoritative state (float64)
    this._eye = [0, 0, 0];
    this._lookAz = 0;
    this._lookEl = 0;
    // held cam_move components — per-component {v, t} so concurrent
    // emissions merge instead of clobbering (see handleIntent's cam_move)
    this._held = { dx: { v: 0, t: -Infinity },
                   dy: { v: 0, t: -Infinity },
                   dz: { v: 0, t: -Infinity } };
    this._now = (this.env.now)
      || (() => (typeof performance !== "undefined" ? performance.now() : Date.now()));
  }

  // unit target→eye direction for the orbit parameterization — the SAME
  // angle convention fly mode reuses (fwd = -_dir(lookAz, lookEl)), which
  // is exactly why orbit→fly is an identity transition.
  _dir(az, el) {
    const [A, B, Up] = frameBasis(this.cam.up);
    const ce = Math.cos(el), se = Math.sin(el);
    const planar = add(mul(A, Math.cos(az) * ce), mul(B, Math.sin(az) * ce));
    return add(planar, mul(Up, se));
  }

  // orbit-convention eye — operation-for-operation viewer.js's eyePos()
  eyePos() {
    const cam = this.cam;
    return add(cam.target, mul(this._dir(cam.az, cam.el), cam.radius));
  }

  _curEye() { return this.mode === "fly" ? this._eye.slice() : this.eyePos(); }

  // nearest body by SURFACE distance (|eye−pos| − radius); O(n) per call,
  // bodies lists are small (≤ ~30) — fine per tick.
  _nearest(eye) {
    const bodies = this.env.bodies ? this.env.bodies() : [];
    const poses = this.env.poses ? this.env.poses() : [];
    let best = -1, bestD = Infinity;
    for (let i = 0; i < bodies.length; i++) {
      const p = poses[i] && poses[i].pos; if (!p) continue;
      const d = len(sub(eye, p)) - this.env.bodyRadius(i);
      if (isFinite(d) && d < bestD) { bestD = d; best = i; }
    }
    return { idx: best, dist: bestD };
  }

  // surface clearance from the CURRENT eye; body-less scene → sceneDiag
  // (disclosed fallback: scene-scale speed, never Infinity / never stuck)
  clearance() {
    const n = this._nearest(this._curEye());
    if (n.idx >= 0) return n.dist;
    return this.env.sceneDiag ? this.env.sceneDiag() : 1.0;
  }

  speedNow() { return this.k_throttle * Math.max(this.clearance(), D_FLOOR); }

  // ── per-frame update — loop() calls this after bodyPoses refresh ──────
  tick(dt) {
    if (this.mode === "follow") {
      // Re-resolve by LABEL every tick (a preserve=true scene commit can
      // reorder bodies; an O(n) string scan per frame is nothing).
      const i = this.env.bodyIndexByLabel(this.followLabel);
      if (i < 0) { this.mode = "orbit"; return; }  // label gone → degrade
      //             IN PLACE: no cam writes = no teleport (noted in state())
      const poses = this.env.poses();
      const p = poses[i] && poses[i].pos; if (!p) return;
      this.cam.target = p.slice();      // MUST copy — never alias bodyPoses
      // az/el/radius untouched: drag/wheel/cam_orbit/cam_zoom orbit AROUND
      // the moving body. Animated scenes restart PT accumulation every
      // frame via draw()'s exposure key — same cheap-exposure behavior as
      // dragging; expected.
      return;
    }
    if (this.mode !== "fly") return;    // orbit: no-op — the controller
    //   NEVER touches cam.target in orbit, so legacy scene.camera.follow
    //   scenes (viewer loop writes it) stay pixel-identical.

    // decay held components not re-issued within MOVE_HOLD_S
    const now = this._now();
    for (const c of ["dx", "dy", "dz"])
      if (now - this._held[c].t > MOVE_HOLD_S * 1000) this._held[c].v = 0;

    const f = mul(this._dir(this._lookAz, this._lookEl), -1);     // forward
    const r = norm(cross(f, this.cam.up));                        // right
    const v = cross(r, f);                                        // up
    // (matches draw()'s right/camUp construction)
    const clPre = this.clearance();
    const s = this.k_throttle * Math.max(clPre, D_FLOOR);         // m/s
    const m = this._held;
    let d = add(add(mul(r, m.dx.v), mul(v, m.dy.v)), mul(f, m.dz.v));
    d = mul(d, s * (dt || 0));
    // STEP CAP — the no-tunnel invariant (see the Choices block)
    const L = len(d), cap = STEP_CAP * Math.max(clPre, D_FLOOR);
    if (L > cap) d = mul(d, cap / L);
    this._eye = add(this._eye, d);

    // PROJECTION into the ONE render-facing cam object: eyePos() then
    // returns _eye to ~2 ulp (≈3e-5 m at 1e11 m coordinates — 7 orders
    // inside the node test's 1e-9 relative tolerance; do NOT "fix" this
    // into a second mirror), fwd is exact, draw()'s exposure key restarts
    // itself on motion exactly as with a drag, uEyeWorld/R1-lite rebase is
    // untouched, and cam.radius = max(clearance, floor) makes the
    // view-scale-adaptive epsilons PROXIMITY-adaptive in fly — fine steps
    // skimming Earth, coarse in deep space.
    const cl = Math.max(this.clearance(), D_FLOOR);
    this.cam.az = this._lookAz;
    this.cam.el = this._lookEl;
    this.cam.radius = cl;
    this.cam.target = add(this._eye, mul(f, cl));
  }

  // ── mouse seams (viewer.js's handlers call these) ─────────────────────
  drag(dxPx, dyPx) {
    if (this.mode === "fly") {
      // Same gains, same sign as orbit — DERIVED, not guessed: with
      // up=[0,0,1] the legacy `az -= dx·0.01` IS the FPS look convention
      // (screen-right drag turns the view right). A flip, if ever wanted,
      // is one disclosed constant here.
      this._lookAz -= dxPx * DRAG_GAIN;
      this._lookEl = clamp(this._lookEl + dyPx * DRAG_GAIN, -EL_CLAMP, EL_CLAMP);
      return;
    }
    // orbit/follow — viewer.js's shipped mousemove lines, verbatim
    this.cam.az -= dxPx * DRAG_GAIN;
    this.cam.el = clamp(this.cam.el + dyPx * DRAG_GAIN, -EL_CLAMP, EL_CLAMP);
  }

  wheel(deltaY) {
    if (this.mode === "fly") {
      // Fly throttle rides the LEGACY zoom formula verbatim with
      // radius→k_throttle (disclosed): the zoom-OUT gesture (wheel-down,
      // factor>1, KeyX) = throttle UP ("bigger view = faster"); zoom-IN =
      // throttle DOWN (fine approach). One gesture rule, no new signs.
      this.k_throttle = clamp(this.k_throttle * Math.exp(deltaY * WHEEL_GAIN),
                              K_MIN, K_MAX);
      return;
    }
    this.cam.radius *= Math.exp(deltaY * WHEEL_GAIN);   // shipped line, verbatim
  }

  // ── engine-intent seams (the E3 tail hook routes through these) ───────
  handles(op) {
    return op === "cam_orbit" || op === "cam_zoom" || op === "cam_move"
        || op === "cam_mode" || op === "follow";
    // cam_pan stays PARKED (registry member, no binding, not wired) —
    // the tail hook's else-branch keeps parking it.
  }

  handleIntent(i) {
    const args = (i && i.args) || {};
    switch (i.op) {
      case "cam_orbit":
        // ONE angular verb for every mode (no cam_look: under the
        // identity map orbit-angles and fly-look are the same two
        // fields — mirrors the tilt one-verb-not-two doctrine).
        if (this.mode === "fly") {
          this._lookAz += (args.daz || 0);
          this._lookEl = clamp(this._lookEl + (args.delv || 0), -EL_CLAMP, EL_CLAMP);
        } else {
          // E3 tail hook's shipped lines, verbatim — zero regression
          this.cam.az += (args.daz || 0);
          this.cam.el = clamp(this.cam.el + (args.delv || 0), -EL_CLAMP, EL_CLAMP);
        }
        return;
      case "cam_zoom":
        if (this.mode === "fly")
          this.k_throttle = clamp(this.k_throttle * (args.factor || 1), K_MIN, K_MAX);
        else
          this.cam.radius *= (args.factor || 1);        // shipped line, verbatim
        return;
      case "cam_move": {
        // Fly only (no-op elsewhere — disclosed; a move verb has no
        // referent in orbit/follow). MERGE RULE (pinned — input.js
        // emissions carry ONLY the params present in the binding, no
        // default zero-fill): each component PRESENT with a numeric value
        // re-arms {v, t}; absent components are untouched, so KeyI's
        // {dz:1} never zeroes a concurrently-held KeyL {dx:1}. tick()
        // decays anything older than MOVE_HOLD_S; deadzoned sticks stop
        // emitting entirely → the decay handles release.
        const now = this._now();
        for (const c of ["dx", "dy", "dz"])
          if (typeof args[c] === "number" && isFinite(args[c]))
            this._held[c] = { v: clamp(args[c], -1, 1), t: now };
        return;
      }
      case "cam_mode": {
        const m = args.mode || "cycle";
        if (m === "cycle") this.cycleMode(); else this.setMode(m);
        return;
      }
      case "follow":
        // param-less engine verb → nearest body (a static bindings table
        // cannot name scene-dependent bodies; named follow is the
        // controller API: window._camctl.follow(label))
        this.follow(null);
        return;
    }
  }

  // ── mode transitions — every one preserves the exact eye (no teleports) ──
  setMode(m) {
    if (m === this.mode) return;                        // no-op
    if (m === "fly") {
      // orbit→fly / follow→fly: IDENTITY — the orbit camera already looks
      // along -_dir(az,el); copy, don't solve. Eye error 0, fwd error 0.
      this._eye = this.eyePos();
      this._lookAz = this.cam.az;
      this._lookEl = this.cam.el;
      this.mode = "fly";
      return;
    }
    if (m === "orbit") {
      if (this.mode === "fly") {
        // fly→orbit: re-anchor the target ON the look ray (disclosed
        // design call, over the nearest-body-CENTER anchor which would
        // ROTATE the view — an orientation teleport). Consequence,
        // disclosed: you now orbit a point in space, not a body — KeyF or
        // locate re-anchors at a body when wanted; view continuity
        // outranks orbit-subject aesthetics.
        const f = mul(this._dir(this._lookAz, this._lookEl), -1);
        const n = this._nearest(this._eye);
        const poses = this.env.poses ? this.env.poses() : [];
        const t = (n.idx >= 0 && poses[n.idx])
          ? dot(sub(poses[n.idx].pos, this._eye), f) : -1;
        const R = (t > 0) ? t : Math.max(this.clearance(), D_FLOOR);
        this.cam.target = add(this._eye, mul(f, R));
        this.cam.az = this._lookAz;
        this.cam.el = this._lookEl;
        this.cam.radius = R;
        // target sits ON the look ray ⇒ eyePos() reproduces _eye to ~2 ulp
        // and fwd is bit-identical ⇒ zero view jump.
      }
      // follow→orbit: stop tracking in place — cam already carries a
      // consistent target/az/el/radius; no cam writes = no teleport.
      this.mode = "orbit";
      this.followLabel = null;
      return;
    }
    if (m === "follow") this.follow(this.followLabel);  // last label, else nearest
  }

  cycleMode() {                    // one keybind/pad button: the full loop
    // orbit → follow(nearest) → fly → orbit; a BODY-LESS scene (water)
    // has nothing to follow, so the cycle skips straight to fly there —
    // honest degradation, not a stuck key (disclosed).
    if (this.mode === "orbit") { if (!this.follow(null)) this.setMode("fly"); }
    else if (this.mode === "follow") this.setMode("fly");
    else this.setMode("orbit");
  }

  follow(label) {
    const bodies = this.env.bodies ? this.env.bodies() : [];
    if (!bodies.length) return false;                   // nothing to follow
    const eye = this._curEye();
    let idx = (label != null) ? this.env.bodyIndexByLabel(label) : -1;
    if (idx < 0) idx = this._nearest(eye).idx;          // null/missing → nearest
    if (idx < 0) return false;
    const poses = this.env.poses();
    const p = poses[idx] && poses[idx].pos; if (!p) return false;
    // Continuous ENTRY solve: keep the eye EXACTLY where it is and aim at
    // the body — the ONE sanctioned angular cut (a target-locked camera
    // looks at its target by definition; user-commanded, the same class
    // as locateBody's existing jump). Solved el can reach ±π/2, past the
    // ±1.5 input clamp — deliberately NOT clamped here (eye continuity
    // wins); the next angular input clamps, a ≤4.06° pitch snap, near-pole
    // only — locate has the identical exposure.
    const T = p.slice();
    const D = sub(eye, T);
    const R = len(D);
    if (R > 0) {
      const Dh = mul(D, 1 / R);
      const [A, B, Up] = frameBasis(this.cam.up);
      const el = Math.asin(clamp(dot(Dh, Up), -1, 1));
      const ce = Math.cos(el);
      // near-pole: keep the prior az (locateBody's own convention)
      this.cam.az = (Math.abs(ce) > 1e-6) ? Math.atan2(dot(Dh, B), dot(Dh, A))
                                          : this.cam.az;
      this.cam.el = el;
      this.cam.radius = R;
    }
    this.cam.target = T;
    this.mode = "follow";
    this.followLabel = bodies[idx].label;
    return true;
  }

  // locateBody's hand-off: locate already framed the jump (locate IS the
  // one sanctioned camera jump) — ADOPT its geometry, never re-solve.
  followAdopt(label) {
    this.mode = "follow";
    this.followLabel = label;
  }

  // applyObj's !preserve hook: a fresh scene = fresh navigation
  // (disclosed: k_throttle resets too — a deep-space throttle carried
  // into a tabletop scene would be a hidden trap).
  resetToOrbit() {
    this.mode = "orbit";
    this.followLabel = null;
    this.k_throttle = K_SPEED;
    for (const c of ["dx", "dy", "dz"]) this._held[c] = { v: 0, t: -Infinity };
  }

  // ── readouts ──────────────────────────────────────────────────────────
  // dbg-line suffix — EMPTY in orbit so today's HUD strings stay
  // byte-identical; built only when a new mode is entered. This is a
  // debug-string tweak, NOT an E4 HUD panel.
  hudSuffix() {
    if (this.mode === "follow") return ` · follow ${this.followLabel}`;
    if (this.mode === "fly") {
      const d = this.clearance();
      return ` · fly v=${this.speedNow().toPrecision(3)} m/s`
           + ` · d=${d.toPrecision(3)} m · k=${this.k_throttle.toPrecision(3)}`;
    }
    return "";
  }

  state() {                        // the smoke test's primary readout
    return { mode: this.mode, followLabel: this.followLabel,
             k_throttle: this.k_throttle, clearance: this.clearance(),
             speed: this.speedNow(), eye: this._curEye() };
  }
}

if (typeof window !== "undefined") window.SGCamera = SGCamera;
if (typeof module !== "undefined" && module.exports)
  module.exports = { SGCamera, K_SPEED, D_FLOOR, STEP_CAP, MOVE_HOLD_S };
})();
