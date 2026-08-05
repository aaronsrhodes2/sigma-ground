/* input.js — the DEVICE layer of the E3 input stack (sigma-ground).
 *
 * Maps device chords (keyboard, mouse, Gamepad API) to GAME VERBS from the
 * closed registry in sigma_ground/engine/intents.py, via a JSON-remappable
 * binding table (data/input_bindings.json) validated against the generated
 * registry mirror (data/intent_verbs.json — regenerate with
 * `python -m sigma_ground.engine.intents`, never hand-edit).
 *
 * DOCTRINE (Golden Rule 7's input layer):
 *   - This file NEVER constructs force dicts. World verbs reduce
 *     SERVER-SIDE (engine/intents.py) — the client cannot honestly build
 *     an impulse vector because J = m·v needs the mass, and matter lives
 *     server-side. Emissions carry {verb, params, target} only.
 *   - Two channels, hard split: the embedder wires onWorldIntent and
 *     onEngineIntent explicitly. Engine intents (camera, pace, UI) can
 *     never touch physics; world intents can never bypass the server.
 *   - The analog-stick doctrine: continuous verbs re-issue at the cap
 *     while held (each becomes a short-duration force server-side), so a
 *     release decays naturally — never a new force kind.
 *   - Rate cap: 60 Hz per (chord, verb) — CLIENT politeness, not a
 *     security boundary (the server logs whatever actually applies).
 *     Suppressed EDGE events are counted in state().dropped — visible,
 *     never invisible.
 *   - Transport-agnostic: ZERO socket code here. The /world wiring is
 *     E4/R2's lane; in E3 the viewer parks world intents for inspection.
 *
 * Chord grammar (exact-modifier match):
 *   key:[Shift+|Ctrl+|Alt+]*<KeyboardEvent.code>   e.g. key:Shift+ArrowRight
 *   mouse:wheel | mouse:move                        (analog; move needs
 *                                                    pointer lock)
 *   pad:button:<n> | pad:axis:<n>
 * Analog chords (pad:axis:*, mouse:move) REQUIRE a "derive" block:
 *   {"derive": {param: {"scale": s, "deadzone": dz=0.15}}}
 *   value = sign(v)·(|v|−dz)/(1−dz)·scale  (post-deadzone rescale); no
 *   emission while inside the deadzone, so honest zeros never trip the
 *   server's nonzero validation.
 *
 * Deliberate E3 cuts (disclosed): no mouse-button chord base exists (so no
 * mousedown/mouseup handlers — nothing could consume them); mouse:move
 * reads movementX only (single-scalar derive; the two-axis look binding is
 * R2's camera-mode lane, and no default binding ships mouse:move); pointer
 * lock is a data-gated progressive enhancement (enablePointerLock AND a
 * mouse:move binding — default table ships neither, so drag-orbit and
 * click-to-pick UX are untouched).
 */
"use strict";
(() => {

const CHORD_RE = /^(key:((Shift|Ctrl|Alt)\+)*[A-Za-z0-9]+|mouse:(wheel|move)|pad:(button|axis):\d+)$/;
const ANALOG_RE = /^(pad:axis:\d+|mouse:move)$/;

class SGInput {
  constructor({verbs, bindings, onWorldIntent, onEngineIntent,
               getTarget = () => null, capHz = 60,
               enablePointerLock = false,
               now = () => performance.now()} = {}) {
    // Accept either the raw mirror/bindings files or their inner payloads.
    this.verbs = (verbs && verbs.verbs) || verbs || {};
    const raw = (bindings && bindings.bindings) || bindings || [];
    this.onWorldIntent = onWorldIntent || (() => {});
    this.onEngineIntent = onEngineIntent || (() => {});
    this.getTarget = getTarget;
    this.capHz = capHz;
    this.enablePointerLock = enablePointerLock;
    this._now = now;

    // Load-time validation against the registry mirror. A bad entry is
    // DISABLED and reported (problems + console.warn) — honest, never
    // silent; the rest of the table stays live.
    this.problems = [];
    this.bindings = [];
    const seen = new Set();
    for (const entry of raw) {
      const why = this._problem(entry, seen);
      if (why) {
        this.problems.push(why);
        console.warn("input.js: binding disabled —", why, entry);
        continue;
      }
      seen.add(entry.chord + "|" + entry.verb);
      const spec = this.verbs[entry.verb];
      this.bindings.push({
        chord: entry.chord, verb: entry.verb,
        params: entry.params || null,
        derive: entry.derive || null,
        channel: spec.channel,          // DERIVED — never stated in the JSON
        continuous: !!spec.continuous,
      });
    }

    // device state (listeners only MUTATE this; poll() is the single
    // emission path)
    this._keys = new Set();
    this._mods = {shift: false, ctrl: false, alt: false};
    this._wheel = 0;
    this._moveX = 0;
    this._prevActive = {};              // chord|verb -> active last poll
    this.lastEmit = {};                 // chord|verb -> last emit ms
    this.dropped = 0;                   // suppressed edge events (visible)
    this.pointerLocked = false;
    this._canvas = null;
    this._handlers = null;
  }

  _problem(entry, seen) {
    if (!entry || typeof entry !== "object") return "binding is not an object";
    if ("channel" in entry)
      return "'channel' is never stated in bindings — it is derived from " +
             "the verb registry (" + entry.chord + ")";
    if (!CHORD_RE.test(entry.chord || ""))
      return "bad chord " + JSON.stringify(entry.chord);
    const spec = this.verbs[entry.verb];
    if (!spec)
      return "unknown verb " + JSON.stringify(entry.verb) +
             " — not in the closed set";
    const declared = new Set((spec.params || []).map((p) => p.name));
    for (const k of Object.keys(entry.params || {}))
      if (!declared.has(k))
        return "verb '" + entry.verb + "' has no parameter '" + k + "'";
    for (const k of Object.keys(entry.derive || {}))
      if (!declared.has(k))
        return "verb '" + entry.verb + "' has no parameter '" + k + "' (derive)";
    if (entry.derive && !ANALOG_RE.test(entry.chord) &&
        entry.chord !== "mouse:wheel")
      return "derive needs an analog chord (pad:axis:*/mouse:wheel/" +
             "mouse:move), got '" + entry.chord + "'";
    if (ANALOG_RE.test(entry.chord) && !entry.derive)
      return "analog chord '" + entry.chord + "' needs a derive block " +
             "(a raw axis with fixed params would fire forever)";
    if (seen.has(entry.chord + "|" + entry.verb))
      return "duplicate chord+verb " + entry.chord + "|" + entry.verb;
    return null;
  }

  // ── device listeners (mutate state only — emission lives in poll()) ──
  attach(canvas) {
    this._canvas = canvas;
    const H = (this._handlers = {});
    const inEditor = () => {
      // The JSON editor / chat box must never fire verbs.
      const el = document.activeElement;
      return !!el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" ||
                      el.tagName === "SELECT" || el.isContentEditable);
    };
    H.keydown = (e) => {
      if (inEditor()) return;
      this._mods = {shift: e.shiftKey, ctrl: e.ctrlKey, alt: e.altKey};
      if (e.repeat) return;             // own repeat only — the cap is ours
      this._keys.add(e.code);
    };
    H.keyup = (e) => {                  // ALWAYS clears (even from an editor)
      this._mods = {shift: e.shiftKey, ctrl: e.ctrlKey, alt: e.altKey};
      this._keys.delete(e.code);
    };
    H.clear = () => {
      // alt-tab with a key held: the keyup goes to another window and the
      // chord would stick forever — blur AND visibilitychange clear ALL
      // held device state.
      this._keys.clear();
      this._mods = {shift: false, ctrl: false, alt: false};
      this._wheel = 0;
      this._moveX = 0;
    };
    window.addEventListener("keydown", H.keydown);
    window.addEventListener("keyup", H.keyup);
    window.addEventListener("blur", H.clear);
    document.addEventListener("visibilitychange", H.clear);

    const hasWheel = this.bindings.some((b) => b.chord === "mouse:wheel");
    const hasMove = this.bindings.some((b) => b.chord === "mouse:move");
    if (hasWheel) {
      H.wheel = (e) => { this._wheel += e.deltaY; };
      canvas.addEventListener("wheel", H.wheel, {passive: true});
    }
    if (hasMove) {
      H.mousemove = (e) => {
        if (this.pointerLocked) this._moveX += e.movementX;
      };
      canvas.addEventListener("mousemove", H.mousemove);
    }
    if (this.enablePointerLock && hasMove) {
      // progressive enhancement, DATA-gated: no mouse:move binding in the
      // table => no pointer-lock request, click-to-pick untouched.
      H.click = () => {
        if (!this.pointerLocked && canvas.requestPointerLock)
          canvas.requestPointerLock();
      };
      canvas.addEventListener("click", H.click);
    }
    H.plock = () => {
      this.pointerLocked = document.pointerLockElement === canvas;
    };
    document.addEventListener("pointerlockchange", H.plock);
  }

  detach() {
    const H = this._handlers;
    if (!H) return;
    window.removeEventListener("keydown", H.keydown);
    window.removeEventListener("keyup", H.keyup);
    window.removeEventListener("blur", H.clear);
    document.removeEventListener("visibilitychange", H.clear);
    if (H.wheel) this._canvas.removeEventListener("wheel", H.wheel);
    if (H.mousemove) this._canvas.removeEventListener("mousemove", H.mousemove);
    if (H.click) this._canvas.removeEventListener("click", H.click);
    document.removeEventListener("pointerlockchange", H.plock);
    this._handlers = null;
    H.clear();
  }

  // ── chord evaluation against current device state ──
  _evalChord(chord, pads) {
    if (chord.startsWith("key:")) {
      const parts = chord.slice(4).split("+");
      const code = parts.pop();
      const want = {shift: false, ctrl: false, alt: false};
      for (const m of parts) {
        if (m === "Shift") want.shift = true;
        else if (m === "Ctrl") want.ctrl = true;
        else if (m === "Alt") want.alt = true;
      }
      const m = this._mods;             // EXACT-modifier match
      return {active: this._keys.has(code) && m.shift === want.shift &&
                      m.ctrl === want.ctrl && m.alt === want.alt,
              value: 1};
    }
    if (chord === "mouse:wheel") {
      const v = this._wheel;
      return {active: v !== 0, value: v};
    }
    if (chord === "mouse:move") {
      const v = this._moveX;
      return {active: v !== 0, value: v};
    }
    let m = chord.match(/^pad:button:(\d+)$/);
    if (m) {
      const n = +m[1];
      for (const gp of pads)
        if (gp && gp.buttons[n] && gp.buttons[n].pressed)
          return {active: true, value: gp.buttons[n].value || 1};
      return {active: false, value: 0};
    }
    m = chord.match(/^pad:axis:(\d+)$/);
    if (m) {
      const n = +m[1];
      for (const gp of pads)
        if (gp && gp.axes.length > n)
          return {active: true, value: gp.axes[n]};   // deadzone gates later
      return {active: false, value: 0};
    }
    return {active: false, value: 0};
  }

  _deriveParams(binding, value) {
    // Post-deadzone rescale: sign(v)·(|v|−dz)/(1−dz)·scale. Returns null
    // when every derived param sits inside its deadzone (no emission — an
    // honest zero must never reach the server's nonzero validation).
    const out = Object.assign({}, binding.params || {});
    let any = false;
    for (const pname of Object.keys(binding.derive)) {
      const d = binding.derive[pname];
      const dz = d.deadzone === undefined ? 0.15 : d.deadzone;
      const av = Math.abs(value);
      if (av <= dz) continue;
      out[pname] = Math.sign(value) * ((av - dz) / (1 - dz)) * d.scale;
      any = true;
    }
    return any ? out : null;
  }

  // ── the SINGLE emission path — the render loop calls poll(dt) ──
  poll(_dtSeconds) {
    const pads = (typeof navigator !== "undefined" && navigator.getGamepads)
      ? Array.prototype.slice.call(navigator.getGamepads() || []) : [];
    const minGap = 1000 / this.capHz;
    for (const b of this.bindings) {
      const key = b.chord + "|" + b.verb;
      const ev = this._evalChord(b.chord, pads);
      let params = b.params ? Object.assign({}, b.params) : {};
      let active = ev.active;
      if (b.derive) {
        const derived = active ? this._deriveParams(b, ev.value) : null;
        if (derived) params = derived;
        else active = false;            // inside the deadzone = inactive
      }
      if (!active) { this._prevActive[key] = false; continue; }
      const was = this._prevActive[key];
      this._prevActive[key] = true;
      if (!b.continuous && was) continue;   // discrete: edge-trigger only
      const t = this._now();
      const lastT = this.lastEmit[key];
      if (lastT !== undefined && t - lastT < minGap) {
        // continuous verbs simply wait for the next tick; a suppressed
        // EDGE event is a lost press — counted, never invisible.
        if (!b.continuous) this.dropped += 1;
        continue;
      }
      this.lastEmit[key] = t;
      this._emit(b, params);
    }
    this._wheel = 0;                    // per-poll accumulators
    this._moveX = 0;
  }

  _emit(b, params) {
    const spec = this.verbs[b.verb];
    if (b.channel === "world") {
      const target = spec.needs_target ? this.getTarget() : null;
      this.onWorldIntent({verb: b.verb, params: params, target: target});
    } else {
      this.onEngineIntent({verb: b.verb, op: spec.op, args: params});
    }
  }

  // ── debug surface (window._input.state() in the console) ──
  state() {
    const pads = (typeof navigator !== "undefined" && navigator.getGamepads)
      ? Array.prototype.slice.call(navigator.getGamepads() || [])
          .filter(Boolean).length : 0;
    return {held: Array.from(this._keys),
            target: this.getTarget(),
            lastEmit: Object.assign({}, this.lastEmit),
            dropped: this.dropped,
            gamepads: pads,
            pointerLocked: this.pointerLocked,
            problems: this.problems.slice()};
  }

  // ── the one fetch helper — the embedder calls it, the class stays IO-free ──
  static load(bindingsUrl = "data/input_bindings.json",
              verbsUrl = "data/intent_verbs.json") {
    const get = (url) => fetch(url).then((r) => {
      if (!r.ok) throw new Error(url + ": HTTP " + r.status);
      return r.json();
    });
    return Promise.all([get(bindingsUrl), get(verbsUrl)])
      .then(([bindings, verbs]) => ({bindings: bindings, verbs: verbs}));
  }
}

window.SGInput = SGInput;
})();
