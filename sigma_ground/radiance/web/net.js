/* net.js — SGWorldNet: the viewer's /world protocol client (E4).
 *
 * One class, DOM-free (loads under plain node; wsFactory is the test
 * seam — the same house pattern as camera.js/test_camera.node.js). The
 * viewer owns all DOM writes; this file owns the wire protocol only:
 *
 *   attach → ready → hello frame → broadcast frames (+ additive audio
 *   "events" — see mcp/cockpit.py's module docstring for the wire shape)
 *   intent/apply → per-socket "applied"/"refusal" tickets (id-matched)
 *   pace/pause/resume → protocol echoes (id-matched when sent as intents)
 *   save/hash → id-matched promise resolution
 *
 * REFUSAL CONTRACT (the seam answer): EVERY inbound {"type":"refusal"} —
 * with or without an id (protocol-level attach refusals have none) — is
 * appended to the tickets ring (id: null) AND forwarded to onTicket.
 * Never a silent drop; the viewer's transient applymsg line is IN
 * ADDITION to the persistent HUD tickets panel.
 *
 * HONEST STATIC-SERVER FAILURE: serve.py (untouched, pure static) 404s
 * the websocket upgrade, so the socket errors/closes before "ready" ever
 * arrives. We track everReady and report ONE honest reason ("world mode
 * needs the cockpit — launch: python -m sigma_ground.mcp.cockpit") for
 * every pre-ready failure — 404 and network-down alike (disclosed: the
 * client cannot tell them apart from a failed ws handshake).
 *
 * Events can never be injected client→server: no such message type
 * exists in the protocol.
 */
"use strict";
(() => {

const TICKET_CAP = 50;              // disclosed ring cap (drop-oldest)
const NO_COCKPIT_MSG =
  "world mode needs the cockpit — launch: python -m sigma_ground.mcp.cockpit";

class SGWorldNet {
  constructor(opts = {}) {
    this.url = opts.url || ((typeof location !== "undefined")
      ? `ws://${location.host}/world` : null);
    this.world = opts.world;
    this.items = opts.items || null;
    this.pushedIndex = (opts.pushedIndex !== undefined)
      ? opts.pushedIndex : undefined;
    this.onReady = opts.onReady || (() => {});
    this.onFrame = opts.onFrame || (() => {});
    this.onEvent = opts.onEvent || (() => {});
    this.onTicket = opts.onTicket || (() => {});
    this.onProto = opts.onProto || (() => {});
    this.onClose = opts.onClose || (() => {});
    this._wsFactory = opts.wsFactory || ((url) => new WebSocket(url));
    this.ws = null;
    this.everReady = false;
    this._live = false;
    this._closed = false;
    this._seq = 0;                  // per-connection monotone ticket ids
    this._pending = {};             // id -> {kind, verb, target, summary}
    this._await = {};               // id -> {resolve, reject} (save/hash)
    this.tickets = [];              // ring, cap TICKET_CAP
    this.nframes = 0;
  }

  get live() { return this._live; }

  connect() {
    const ws = this.ws = this._wsFactory(this.url);
    ws.onopen = () => {
      const attach = { type: "attach", world: this.world };
      if (this.items) attach.items = this.items;
      if (this.pushedIndex !== undefined)
        attach.pushed_index = this.pushedIndex;
      this._send(attach);
    };
    ws.onmessage = (e) => {
      let m;
      try { m = JSON.parse(e.data); } catch (_err) { return; }
      this._onmsg(m);
    };
    ws.onerror = () => {};          // the close handler owns reporting
    ws.onclose = () => this._onclosed();
    return this;
  }

  _send(obj) {
    if (this.ws && this.ws.readyState === 1) {
      this.ws.send(JSON.stringify(obj));
      return true;
    }
    // ws.readyState CONNECTING (0): sends before open are a caller bug —
    // surfaced as an honest local ticket, never a throw
    this._ticket({ id: null, ok: false,
                   text: "not connected — message dropped: "
                         + (obj && obj.type) });
    return false;
  }

  _ticket(t) {
    this.tickets.push(t);
    if (this.tickets.length > TICKET_CAP) this.tickets.shift();
    this.onTicket(t);
  }

  _onmsg(m) {
    if (!m || typeof m !== "object") return;
    const t = m.type;
    if (t === "ready") {
      this.everReady = true;
      this._live = true;
      this.onReady(m);
    } else if (t === "frame") {
      this.nframes++;
      this.onFrame(m);
      for (const ev of (m.events || [])) this.onEvent(ev);
    } else if (t === "applied") {
      const sent = this._pending[m.id]; delete this._pending[m.id];
      this._ticket({ id: (m.id !== undefined ? m.id : null), ok: true,
                     icon: "✓",
                     verb: sent ? sent.verb : (m.force && m.force.kind),
                     target: m.target,
                     text: `applied ${(sent && sent.summary)
                            || (m.force && m.force.kind) || "?"}`
                           + ` @ step ${m.step_index}`,
                     sent: sent || null });
    } else if (t === "refusal") {
      const id = (m.id !== undefined && m.id !== null) ? m.id : null;
      const sent = id !== null ? this._pending[id] : null;
      if (id !== null) delete this._pending[id];
      if (id !== null && this._await[id]) {       // a refused save/hash
        this._await[id].reject(new Error(m.text || "refused"));
        delete this._await[id];
      }
      this._ticket({ id, ok: false, icon: "✗",
                     verb: sent ? sent.verb : null,
                     target: sent ? sent.target : null,
                     text: (sent ? `${sent.summary} → refused: ` : "")
                           + (m.text || "(no reason given)"),
                     sent: sent || null });
    } else if (t === "pace" || t === "paused" || t === "resumed") {
      // protocol echoes; id-carrying ones (intent-routed) get a ticket so
      // keyboard pace/pause chords surface in the HUD log too
      if (m.id !== undefined && m.id !== null) {
        const sent = this._pending[m.id]; delete this._pending[m.id];
        this._ticket({ id: m.id, ok: true, icon: "✓",
                       verb: sent ? sent.verb : t, target: null,
                       text: t === "pace" ? `pace → ${m.factor}×` : t,
                       sent: sent || null });
      }
      this.onProto(m);
    } else if (t === "save" || t === "hash") {
      const w = this._await[m.id];
      if (w) { delete this._await[m.id];
               w.resolve(t === "save" ? m.data : m.value); }
    }
  }

  _onclosed() {
    if (this._closed) return;
    this._closed = true;
    this._live = false;
    for (const id of Object.keys(this._await)) {
      this._await[id].reject(new Error("connection closed"));
      delete this._await[id];
    }
    this.onClose({ everReady: this.everReady,
                   reason: this.everReady ? "connection closed"
                                          : NO_COCKPIT_MSG });
  }

  // ── the two world channels (both stay working — deliverable A) ──
  intent(verb, params, target) {
    const id = ++this._seq;
    this._pending[id] = { kind: "intent", verb, target,
                          summary: `${verb}${target ? " " + target : ""}` };
    this._send({ type: "intent", id, verb, params: params || {},
                 target: (target === undefined ? null : target) });
    return id;
  }

  apply(target, force) {            // RAW channel: a full wire dict
    const id = ++this._seq;
    this._pending[id] = { kind: "apply", verb: force && force.kind, target,
                          summary: `${(force && force.kind) || "?"}`
                                   + `${target ? " " + target : ""}` };
    this._send({ type: "apply", id, target: (target === undefined
                                             ? null : target), force });
    return id;
  }

  // ── protocol ops (bare — the HUD reads SERVER echoes, never the UI) ──
  pace(factor) { this._send({ type: "pace", factor }); }
  pause() { this._send({ type: "pause" }); }
  resume() { this._send({ type: "resume" }); }

  _awaited(type) {
    const id = ++this._seq;
    return new Promise((resolve, reject) => {
      this._await[id] = { resolve, reject };
      if (!this._send({ type, id })) {
        delete this._await[id];
        reject(new Error("not connected"));
      }
    });
  }

  save() { return this._awaited("save"); }
  hashNow() { return this._awaited("hash"); }

  detach() {                        // the world PERSISTS server-side
    try { this._send({ type: "detach" }); } catch (_e) {}
    try { if (this.ws) this.ws.close(); } catch (_e) {}
  }

  state() {                         // debug surface: window._worldnet
    return { live: this._live, everReady: this.everReady,
             world: this.world, nframes: this.nframes,
             tickets: this.tickets.length,
             pending: Object.keys(this._pending).length };
  }
}

SGWorldNet.NO_COCKPIT_MSG = NO_COCKPIT_MSG;
SGWorldNet.TICKET_CAP = TICKET_CAP;

if (typeof window !== "undefined") window.SGWorldNet = SGWorldNet;
if (typeof module !== "undefined" && module.exports)
  module.exports = { SGWorldNet, NO_COCKPIT_MSG, TICKET_CAP };
})();
