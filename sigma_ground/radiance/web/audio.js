/* audio.js — SGAudio: the DUMB synthesizer half of the E4 audio pipeline.
 *
 * The SERVER computes every physics number (engine/audio_events.py:
 * energy, frequency from acoustics.ring_frequency, Q, materials); this
 * file owns ONLY disclosed presentation Choices — the distance law, the
 * full-scale mapping, master volume — and the WebAudio plumbing. The
 * pure core (mediumOf/deriveVoice/CHOICES) is DOM-free and node-tested
 * (test_e4.node.js, the camera.js house pattern); the shell needs a
 * browser AudioContext.
 *
 * VACUUM SILENCE (doctrine): physics_env.medium == "vacuum" (or
 * pressure_pa == 0) means no medium propagates — nothing is heard.
 * Hearing through STRUCTURAL ATTACHMENT (the player bolted to what is
 * vibrating) is explicitly NOT implemented: attachment does not exist
 * until the game arc — a disclosed deferral, surfaced in every
 * suppressed event's reason, never a silent drop. Tabletop push-route
 * bundles stamp physics_env.medium = "air" (verified against the shipped
 * data/scene_*.json bundles), so world-mode scenes hit the explicit-stamp
 * arm; the no-physics_env fallback → "air" is the disclosed Choice for
 * older env-less scenes only.
 *
 * AUTOPLAY: browsers refuse audio before a user gesture. The context is
 * created/resumed on the first pointerdown/keydown; until it runs the
 * HUD audio panel says so ("muted until you interact") — a disclosure,
 * not a silent failure.
 *
 * MUSIC: registerMusicHook(fn) is a hook API ONLY — state → track name,
 * nothing plays, zero assets ship, and it makes NO physics claim (an
 * authored-asset channel with game-state hooks; E4 ships the hook).
 * The bell_strike_*.wav files in data/ are prior bell-chain work, not
 * music — untouched.
 */
"use strict";
(() => {

// ── disclosed presentation Choices (each {value, reason}) ────────────────
const CHOICES = {
  D_REF: { value: 1.0,
    reason: "reference distance (m): closer than arm's length stops " +
            "growing — a disclosed listener-model cap (no blowup at the " +
            "lens)" },
  E_REF: { value: 1e-3,
    reason: "full-scale acoustic energy (J): the flagship plate fall " +
            "(~4e-4 J acoustic) lands near -4 dBFS — audible headroom, " +
            "no clipping" },
  THUD_CENTER: { value: 150.0,
    reason: "unmapped-material fallback center (Hz) — explicitly NO " +
            "physics claim, an honest generic thud voice" },
  THUD_Q: { value: 8.0,
    reason: "fallback bandpass Q — broad, pitchless; no physics claim" },
  THUD_TAU: { value: 0.03,
    reason: "fallback decay (s) — a short dull thump; no physics claim" },
  VOICE_CAP: { value: 16,
    reason: "max concurrent voices; oldest culled, counted — flood " +
            "honesty on the render thread" },
  TAU_CAP: { value: 20.0,
    reason: "decay-time ceiling (s): a hostile/absurd wire Q cannot " +
            "schedule an effectively-eternal tone (tau = Q/(π·f) is " +
            "clamped here); every legitimate tabletop ring decays far " +
            "below this — a defensive presentation guard, no physics " +
            "claim" },
  DEFAULT_VOLUME: { value: 0.8,
    reason: "boot master volume — a comfort default, user-adjustable in " +
            "the HUD" },
};

// ── pure core (node-tested) ──────────────────────────────────────────────
function mediumOf(scene) {
  // SOURCE OF TRUTH: the scene stamps physics_env.medium explicitly
  // (theaters: plain/room → "air", void/deep_space → "vacuum"; shipped
  // push-route bundles stamp "air" too); the no-physics_env fallback →
  // "air" is the disclosed Choice for older env-less scenes.
  const env = scene && scene.physics_env;
  if (!env) return "air";
  return (env.medium === "vacuum" || env.pressure_pa === 0)
    ? "vacuum" : "air";
}

function deriveVoice(ev, listenerPos, medium, opts) {
  // ev: the SERVER's wire event (energy/frequency/Q already computed).
  // Returns {play, reason, kind, freq, q, tau, gain} — pure, no state.
  const o = opts || {};
  const volume = (o.volume !== undefined) ? o.volume : 1.0;
  if (medium === "vacuum") {
    return { play: false,
             reason: "vacuum — no medium propagates; hearing through " +
                     "structural attachment is NOT implemented until " +
                     "the game arc (disclosed)" };
  }
  const ae = ev && ev.acoustic_energy_j;
  if (!(typeof ae === "number" && isFinite(ae) && ae > 0)) {
    return { play: false, reason: "event carries no acoustic energy" };
  }
  const p = (ev.position || [0, 0, 0]);
  const l = listenerPos || [0, 0, 0];
  const dist = Math.hypot(p[0] - l[0], (p[1] || 0) - (l[1] || 0),
                          p[2] - l[2]);
  const D_REF = CHOICES.D_REF.value;
  const d = Math.max(dist, D_REF);          // clamped inside D_REF
  // amplitude ∝ sqrt(E) (full-scale at E_REF), 1/d amplitude falloff
  // = inverse-square in ENERGY — the disclosed distance law
  const gain = Math.min(1, Math.sqrt(ae / CHOICES.E_REF.value))
             * (D_REF / d) * volume;
  if (!(isFinite(gain) && gain >= 0)) {
    // DEFENSIVE GUARD (adversarially proven): a tampered/malformed event
    // position (NaN, a string, a short array) makes dist — and so gain —
    // NaN; WebAudio's setValueAtTime throws a RangeError on non-finite
    // values, which would escape inside ws.onmessage and drop the rest of
    // the frame's events. Refuse with the reason instead — never hand the
    // scheduler a non-finite number.
    return { play: false,
             reason: "malformed event position/energy produced a " +
                     "non-finite gain — refused (defensive presentation " +
                     "guard; tampered-wire class, disclosed)" };
  }
  const synth = ev.synth || {};
  if (synth.kind === "ring" && typeof synth.frequency_hz === "number"
      && isFinite(synth.frequency_hz) && synth.frequency_hz > 0) {
    const q = (typeof synth.q === "number" && isFinite(synth.q)
               && synth.q > 0) ? synth.q : 60.0;
    return { play: true, reason: null, kind: "ring",
             freq: synth.frequency_hz, q,
             // exp-decay const, ceiling-clamped (TAU_CAP): an absurd wire
             // Q must not schedule an effectively-eternal tone
             tau: Math.min(q / (Math.PI * synth.frequency_hz),
                           CHOICES.TAU_CAP.value),
             gain };
  }
  // thud fallback: frequency_hz null = the server made NO pitch claim
  return { play: true, reason: null, kind: "thud",
           freq: CHOICES.THUD_CENTER.value, q: CHOICES.THUD_Q.value,
           tau: CHOICES.THUD_TAU.value, gain };
}

// ── shell (browser only) ─────────────────────────────────────────────────
const S = {
  ctx: null, master: null, noiseBuf: null,
  muted: false, volume: CHOICES.DEFAULT_VOLUME.value,
  voicesBuilt: 0, culled: 0,
  suppressedVacuum: 0, suppressedPolicy: 0,
  lastReason: null, lastMedium: null,
  voices: [],                       // [{stopAt, nodes:[...]}]
  musicHook: null, musicTrack: null,
};

function _ensureCtx() {
  if (typeof window === "undefined") return null;
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) return null;
  if (!S.ctx) {
    try {
      S.ctx = new AC();
      S.master = S.ctx.createGain();
      S.master.gain.value = S.muted ? 0 : S.volume;
      S.master.connect(S.ctx.destination);
    } catch (_e) { S.ctx = null; }
  }
  if (S.ctx && S.ctx.state === "suspended") {
    try { S.ctx.resume(); } catch (_e) {}
  }
  return S.ctx;
}

// the autoplay gesture hook: first interaction wakes the context
if (typeof window !== "undefined") {
  const wake = () => { _ensureCtx(); _syncBag(); };
  window.addEventListener("pointerdown", wake, true);
  window.addEventListener("keydown", wake, true);
}

function _noise(ctx) {
  if (!S.noiseBuf) {
    const n = Math.floor(ctx.sampleRate * 0.5);
    S.noiseBuf = ctx.createBuffer(1, n, ctx.sampleRate);
    const ch = S.noiseBuf.getChannelData(0);
    // deterministic LCG noise — same buffer every session (no physics
    // claim either way; determinism is just tidier)
    let s = 22695477;
    for (let i = 0; i < n; i++) {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      ch[i] = (s / 0x3fffffff) - 1.0;
    }
  }
  return S.noiseBuf;
}

function _cull() {
  const cap = CHOICES.VOICE_CAP.value;
  while (S.voices.length > cap) {
    const v = S.voices.shift();
    for (const nd of v.nodes) { try { nd.stop ? nd.stop() : 0; } catch (_e) {}
                                try { nd.disconnect(); } catch (_e) {} }
    S.culled++;
  }
}

function _schedule(v) {
  const ctx = S.ctx;
  if (!ctx || ctx.state !== "running") {
    S.suppressedPolicy++;
    S.lastReason = "🔇 audio muted until you interact " +
                   "(browser autoplay policy)";
    return false;
  }
  const t0 = ctx.currentTime;
  const g = ctx.createGain();
  g.gain.setValueAtTime(v.gain, t0);
  g.gain.setTargetAtTime(0, t0, v.tau);
  g.connect(S.master);
  let src;
  if (v.kind === "ring") {
    src = ctx.createOscillator();
    src.type = "sine";
    src.frequency.value = v.freq;
  } else {
    src = ctx.createBufferSource();
    src.buffer = _noise(ctx);
    src.loop = true;
    const bp = ctx.createBiquadFilter();
    bp.type = "bandpass";
    bp.frequency.value = v.freq;
    bp.Q.value = v.q;
    src.connect(bp); bp.connect(g);
  }
  if (v.kind === "ring") src.connect(g);
  const entry = { nodes: [src, g] };
  src.onended = () => {             // natural end: leave the live list
    const i = S.voices.indexOf(entry);
    if (i >= 0) S.voices.splice(i, 1);
    try { g.disconnect(); } catch (_e) {}
  };
  src.start(t0);
  src.stop(t0 + 8 * v.tau);         // amplitude down ~e^-8 — inaudible
  S.voices.push(entry);
  S.voicesBuilt++;
  _cull();
  return true;
}

function _syncBag(medium) {
  if (typeof window === "undefined") return;
  const bag = window._hudState;
  if (!bag) return;
  if (medium !== undefined) S.lastMedium = medium;
  const running = S.ctx && S.ctx.state === "running";
  bag.audio = {
    status: S.muted ? "🔇 muted"
      : running ? "♪ live"
      : "🔇 audio muted until you interact (browser autoplay policy)",
    muted: S.muted, volume: S.volume,
    voicesBuilt: S.voicesBuilt,
    medium: S.lastMedium || "—",
  };
  if (window._hud) window._hud.touch();
}

const SGAudio = {
  CHOICES, mediumOf, deriveVoice,

  handleEvent(ev, eye, scene) {
    const medium = mediumOf(scene);
    const v = deriveVoice(ev, eye, medium,
                          { volume: 1.0 });   // master volume lives on the
    //                                          masterGain node (mute/vol)
    if (!v.play) {
      if (medium === "vacuum") S.suppressedVacuum++;
      else S.suppressedPolicy++;
      S.lastReason = v.reason;
    } else {
      _schedule(v);
    }
    _syncBag(medium);
    return v;
  },

  setMuted(m) {
    S.muted = !!m;
    if (S.master) S.master.gain.value = S.muted ? 0 : S.volume;
    _syncBag();
  },
  setVolume(x) {
    const v = Math.max(0, Math.min(1, +x || 0));
    S.volume = v;
    if (S.master && !S.muted) S.master.gain.value = v;
    _syncBag();
  },

  registerMusicHook(fn) {           // hook API ONLY — zero assets, zero
    S.musicHook = (typeof fn === "function") ? fn : null;   // physics claim
  },
  musicState(worldState) {
    S.musicTrack = S.musicHook ? (S.musicHook(worldState) || null) : null;
    return S.musicTrack;
  },

  state() {                         // the smoke test's assertion surface:
    // audibility is a HUMAN-EAR check; the assertion is that the WebAudio
    // GRAPH was built (ctxState + voice counts) — disclosed.
    return { ctxState: S.ctx ? S.ctx.state : "none",
             voicesBuilt: S.voicesBuilt, nodesLive: S.voices.length,
             culled: S.culled, muted: S.muted, volume: S.volume,
             suppressedVacuum: S.suppressedVacuum,
             suppressedPolicy: S.suppressedPolicy,
             lastReason: S.lastReason, medium: S.lastMedium,
             musicTrack: S.musicTrack };
  },
};

if (typeof window !== "undefined") window.SGAudio = SGAudio;
if (typeof module !== "undefined" && module.exports)
  module.exports = { mediumOf, deriveVoice, CHOICES, SGAudio };
})();
