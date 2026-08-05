/* hud.js — SGHud: dockable HUD panels AS DATA (E4).
 *
 * "formalizes the viewer's existing overlay idiom (energy readout, plug
 * disclosure, variant slider) into hud.js: dockable panels driven by the
 * snapshot and event streams, specified as DATA so an LLM can compose
 * new panels without touching shader/render code." (the plan, verbatim)
 *
 * THE STATE BAG — window._hudState, domain-per-writer (the cross-lane
 * seam). Writers create/extend their OWN domain then call
 * window._hud && _hud.touch():
 *   viewer world block → world {mode,name,err}, frame {step,t,pace,
 *     paused,settled,energy_j,momentum,angular_momentum}, tickets
 *     [ring 50], events [ring 32: wire event + {heard,silenced_reason}]
 *   audio.js → audio {status,muted,volume,voicesBuilt,medium}
 *   viewer loop → camera (camctl.state() STRUCTURED fields — hudSuffix()
 *     is NEVER string-parsed), playback {simTime,tEnd,playing}
 * hud.js READS ONLY; renders on a dirty flag throttled to HUD_HZ.
 *
 * PANEL SPEC = pure JSON. Formats and actions are STRING NAMES resolved
 * against CLOSED sets (the applied.KINDS pattern) — that is what makes
 * specs LLM-composable AND what makes data unable to invoke arbitrary
 * code. Unknown kind/format/action → the row renders DISABLED naming the
 * offender + a problems[] entry + console.warn — never a throw, never
 * silence (input.js's disabled-and-reported idiom).
 *
 * SHIPPED PANELS live in the PANELS const below (DESIGN CALL, disclosed:
 * not a fetched hud_panels.json — avoiding the offline/bad-JSON failure
 * mode; addPanel() is the runtime extension surface and a follow-up can
 * externalize the array verbatim).
 *
 * ZERO-REGRESSION RULE: SGHud mounts ONLY on world attach and unmounts
 * on detach; the legacy #energy/#tval/#dbg/#plugs overlays are untouched
 * and keep serving recorded//live modes. The visible world-mode overlap
 * is the disclosed parity-proving duplication — retirement is
 * post-parity, explicitly NOT E4 (the /live doctrine applied to
 * overlays).
 *
 * Pure core (getPath/fmt/renderRows/renderFeed/validatePanel) is DOM-free
 * and node-tested (test_e4.node.js).
 */
"use strict";
(() => {

const HUD_HZ = 10;                  // Choice: readable, not a hot-path cost
const DOCKS = ["tl", "tr", "bl", "br"];
const ROW_KINDS = ["scalar", "button", "slider"];

// ── pure core ────────────────────────────────────────────────────────────
function getPath(obj, path) {
  if (obj == null || !path) return undefined;
  let cur = obj;
  for (const part of String(path).split(".")) {
    if (cur == null) return undefined;
    cur = cur[part];
  }
  return cur;
}

// fmt: value + format-name → display string, "—" for absent values,
// null for an UNKNOWN format name (the caller renders the row disabled).
function fmt(v, spec) {
  if (spec !== undefined && spec !== null && typeof spec !== "string")
    return null;
  const known = !spec || spec === "str" || spec === "int" || spec === "hz"
    || /^(fixed|sig|exp|mag):\d+$/.test(spec)
    || /^bool:[^|]*\|.*$/.test(spec);
  if (!known) return null;
  if (v === undefined || v === null) return "—";
  if (!spec || spec === "str") return String(v);
  let m;
  if ((m = /^bool:([^|]*)\|(.*)$/.exec(spec))) return v ? m[1] : m[2];
  if (spec === "int")
    return (typeof v === "number" && isFinite(v))
      ? String(Math.round(v)) : "—";
  if (spec === "hz")
    return (typeof v === "number" && isFinite(v))
      ? v.toFixed(0) + " Hz" : "—";
  if ((m = /^mag:(\d+)$/.exec(spec))) {
    if (!Array.isArray(v)) return "—";
    const h = Math.hypot.apply(null, v);
    return isFinite(h) ? h.toPrecision(+m[1]) : "—";
  }
  if (!(typeof v === "number" && isFinite(v))) return "—";
  if ((m = /^fixed:(\d+)$/.exec(spec))) return v.toFixed(+m[1]);
  if ((m = /^sig:(\d+)$/.exec(spec))) return v.toPrecision(+m[1]);
  if ((m = /^exp:(\d+)$/.exec(spec))) return v.toExponential(+m[1]);
  return String(v);                 // unreachable (known covers all)
}

// renderRows: spec + bag → [{label, text, disabled, control?, action?}]
// NEVER throws; unknown names come back as disabled rows naming the
// offender.
function renderRows(spec, bag, actionNames) {
  const out = [];
  const acts = actionNames || null;
  for (const row of (spec && spec.rows) || []) {
    const kind = row.kind || "scalar";
    const label = row.label !== undefined ? String(row.label) : "";
    if (ROW_KINDS.indexOf(kind) < 0) {
      out.push({ label, text: "unknown kind '" + kind + "'",
                 disabled: true });
      continue;
    }
    if (kind === "button" || kind === "slider") {
      const bad = acts && acts.indexOf(row.action) < 0;
      out.push({ label,
                 text: bad ? "unknown action '" + row.action + "'" : "",
                 control: kind, action: row.action,
                 min: row.min, max: row.max, scale: row.scale,
                 disabled: !!bad });
      continue;
    }
    const text = fmt(getPath(bag, row.path), row.format);
    if (text === null) {
      out.push({ label, text: "unknown format '" + row.format + "'",
                 disabled: true });
      continue;
    }
    out.push({ label,
               text: text === "—" ? text : text + (row.suffix || ""),
               disabled: false });
  }
  return out;
}

// renderFeed: {feed:{path,limit,line}} → [string] newest-first; the line
// template supports {field} and {field:format} (dot paths into each item).
function renderFeed(spec, bag) {
  const f = spec && spec.feed;
  if (!f || !f.path) return [];
  const arr = getPath(bag, f.path);
  if (!Array.isArray(arr)) return [];
  const lim = f.limit || 8;
  return arr.slice(-lim).map((item) =>
    String(f.line || "").replace(
      /\{([A-Za-z0-9_.]+)(?::([^{}]+))?\}/g,
      (_all, p, fs) => {
        const t = fmt(getPath(item, p), fs);
        return t === null ? "?" + fs + "?" : t;
      })
  ).reverse();
}

function validatePanel(spec) {      // null = valid, else the reason
  if (!spec || typeof spec !== "object") return "panel is not an object";
  if (!spec.id || typeof spec.id !== "string") return "panel needs an id";
  if (DOCKS.indexOf(spec.dock) < 0)
    return "panel '" + spec.id + "': dock must be one of " + DOCKS.join("/");
  if (spec.rows !== undefined && !Array.isArray(spec.rows))
    return "panel '" + spec.id + "': rows must be an array";
  if (spec.feed !== undefined
      && !(spec.feed && typeof spec.feed.path === "string"))
    return "panel '" + spec.id + "': feed needs a path";
  if (!spec.rows && !spec.feed)
    return "panel '" + spec.id + "': needs rows and/or feed";
  return null;
}

// ── the shipped panels (pure data — the LLM-composability surface) ──────
const PANELS = [
  { id: "world_status", title: "world", dock: "tr", rows: [
    { label: "world", path: "world.name" },
    { label: "step", path: "frame.step", format: "int" },
    { label: "t", path: "frame.t", format: "fixed:2", suffix: " s" },
    { label: "pace", path: "frame.pace", format: "fixed:2", suffix: "×" },
    { label: "state", path: "frame.paused", format: "bool:paused|running" },
    { label: "settled", path: "frame.settled", format: "bool:yes|no" },
    { kind: "button", label: "⏸/▶", action: "world.pausetoggle" },
    { kind: "slider", label: "pace", action: "world.pace",
      min: 0.1, max: 10, scale: "log" },
    { kind: "button", label: "💾 save", action: "world.save" }] },
  { id: "dynamics", title: "dynamics", dock: "tr", rows: [
    { label: "KE", path: "frame.energy_j", format: "sig:3", suffix: " J" },
    { label: "|p|", path: "frame.momentum", format: "mag:3",
      suffix: " kg·m/s" },
    { label: "|L|", path: "frame.angular_momentum", format: "mag:3",
      suffix: " kg·m²/s" }] },
  { id: "tickets", title: "tickets", dock: "br",
    feed: { path: "tickets", limit: 8,
            line: "#{id} {icon} {text}" } },
  { id: "audio_feed", title: "audio", dock: "bl", rows: [
    { label: "status", path: "audio.status" },
    { label: "medium", path: "audio.medium" },
    { kind: "button", label: "🔇 mute", action: "audio.mute" },
    { kind: "slider", label: "vol", action: "audio.volume",
      min: 0, max: 1 }],
    feed: { path: "events", limit: 6,
            line: "{kind} {body} on {surface} · " +
                  "{impact_speed_m_s:fixed:2} m/s · " +
                  "{synth.frequency_hz:hz} · {heard:bool:♪|silent}" } },
  { id: "camera", title: "camera", dock: "tr", rows: [
    { label: "mode", path: "camera.mode" },
    { label: "follow", path: "camera.followLabel" },
    { label: "speed", path: "camera.speed", format: "sig:3",
      suffix: " m/s" },
    { label: "clearance", path: "camera.clearance", format: "sig:3",
      suffix: " m" }] },
];

// ── shell (browser only) ────────────────────────────────────────────────
const CSS = `
#hudroot { position:absolute; inset:0; pointer-events:none; z-index:4;
  font:11px/1.5 ui-monospace,Menlo,Consolas,monospace; }
#hudroot .hud-dock { position:absolute; display:flex; flex-direction:column;
  gap:8px; max-width:min(340px,44%); }
#hudroot .hud-dock.tl { top:10px; left:12px; }
#hudroot .hud-dock.tr { top:10px; right:12px; align-items:flex-end; }
#hudroot .hud-dock.bl { bottom:10px; left:12px; }
#hudroot .hud-dock.br { bottom:10px; right:12px; align-items:flex-end; }
#hudroot .hud-panel { background:rgba(5,6,8,0.72); border:1px solid #26262c;
  border-radius:8px; padding:7px 10px; min-width:170px; max-width:100%;
  color:#c8c8cc; text-shadow:0 0 6px #000; }
#hudroot .hud-title { color:#00CCFF; font-size:10px; letter-spacing:.5px;
  text-transform:uppercase; margin-bottom:3px; }
#hudroot .hud-row { display:flex; gap:8px; justify-content:space-between; }
#hudroot .hud-row .hl { color:#6a6a72; }
#hudroot .hud-row .hv { color:#e8e8ea; text-align:right; overflow:hidden;
  text-overflow:ellipsis; white-space:nowrap; max-width:220px; }
#hudroot .hud-row.disabled .hv { color:#FFAA00; }
#hudroot .hud-ctl { pointer-events:auto; display:flex; gap:6px;
  align-items:center; margin-top:3px; }
#hudroot .hud-ctl button { background:#16161a; color:#e8e8ea;
  border:1px solid #26262c; border-radius:5px; padding:2px 8px;
  cursor:pointer; font:inherit; }
#hudroot .hud-ctl button:hover { border-color:#00CCFF; }
#hudroot .hud-ctl button:disabled { color:#FFAA00; cursor:default; }
#hudroot .hud-ctl input[type=range] { flex:1; min-width:70px;
  accent-color:#00CCFF; }
#hudroot .hud-feed { margin-top:4px; border-top:1px solid #1c1c20;
  padding-top:3px; max-width:320px; }
#hudroot .hud-feed div { color:#9a9aa2; overflow:hidden;
  text-overflow:ellipsis; white-space:nowrap; }
#hudroot .hud-feed div.bad { color:#FF7766; }
`;

const SGHud = {
  PANELS,
  getPath, fmt, renderRows, renderFeed, validatePanel,
  problems: [],
  _mounted: false, _root: null, _specs: null, _actions: null,
  _dirty: false, _timer: null, _els: {},

  mount(container, specs, opts) {
    if (this._mounted) this.unmount();
    if (typeof document === "undefined" || !container) return false;
    if (!document.getElementById("hudcss")) {
      const st = document.createElement("style");
      st.id = "hudcss";
      st.textContent = CSS;
      document.head.appendChild(st);
    }
    this._actions = (opts && opts.actions) || {};
    this._bagFn = (opts && opts.bag) || (() => window._hudState || {});
    // container IS the root (#hudroot ships statically in index.html —
    // mount populates it, unmount empties it, the div itself stays)
    this._root = container;
    if (!this._root.id) this._root.id = "hudroot";
    this._root.innerHTML = "";
    this._docks = {};
    for (const d of DOCKS) {
      const el = document.createElement("div");
      el.className = "hud-dock " + d;
      this._docks[d] = el;
      this._root.appendChild(el);
    }
    this._specs = [];
    this._els = {};
    this._mounted = true;
    for (const s of (specs || PANELS)) this.addPanel(s);
    window._hud = this;
    this._timer = setInterval(() => {
      if (this._dirty) { this._dirty = false; this._render(); }
    }, 1000 / HUD_HZ);
    this.touch();
    return true;
  },

  // THE composability seam: adding a panel = adding a spec object —
  // zero render-code edits (proven by the node test's novel spec).
  addPanel(spec) {
    const why = validatePanel(spec);
    if (why) {
      this.problems.push(why);
      if (typeof console !== "undefined") console.warn("hud.js:", why);
      return false;
    }
    this._specs.push(spec);
    if (this._mounted) { this._buildPanel(spec); this.touch(); }
    return true;
  },

  _buildPanel(spec) {
    const p = document.createElement("div");
    p.className = "hud-panel";
    p.dataset.panel = spec.id;
    const title = document.createElement("div");
    title.className = "hud-title";
    title.textContent = spec.title || spec.id;
    p.appendChild(title);
    const rowsEl = document.createElement("div");
    p.appendChild(rowsEl);
    let feedEl = null;
    if (spec.feed) {
      feedEl = document.createElement("div");
      feedEl.className = "hud-feed";
      p.appendChild(feedEl);
    }
    this._docks[spec.dock].appendChild(p);
    this._els[spec.id] = { rowsEl, feedEl, controlsBuilt: false };
  },

  _render() {
    const bag = this._bagFn();
    const actNames = Object.keys(this._actions);
    for (const spec of this._specs) {
      const els = this._els[spec.id];
      if (!els) continue;
      const rows = renderRows(spec, bag, actNames);
      if (!els.controlsBuilt) {
        els.rowsEl.innerHTML = "";
        els.rowEls = [];
        rows.forEach((r) => {
          if (r.control) {
            const div = document.createElement("div");
            div.className = "hud-ctl";
            if (r.control === "button") {
              const b = document.createElement("button");
              b.textContent = r.label;
              if (r.disabled) { b.disabled = true; b.title = r.text; }
              else b.addEventListener("click",
                () => { try { this._actions[r.action](); }
                        catch (e) { console.warn("hud action:", e); } });
              div.appendChild(b);
            } else {
              const lbl = document.createElement("span");
              lbl.className = "hl";
              lbl.textContent = r.label;
              const sl = document.createElement("input");
              sl.type = "range";
              const log = r.scale === "log";
              sl.min = log ? Math.log10(r.min) : (r.min ?? 0);
              sl.max = log ? Math.log10(r.max) : (r.max ?? 1);
              sl.step = "0.01";
              sl.value = log ? "0" : sl.max;
              if (r.disabled) sl.disabled = true;
              else sl.addEventListener("input", () => {
                const v = log ? Math.pow(10, parseFloat(sl.value))
                              : parseFloat(sl.value);
                try { this._actions[r.action](v); }
                catch (e) { console.warn("hud action:", e); }
              });
              div.append(lbl, sl);
            }
            els.rowsEl.appendChild(div);
            els.rowEls.push(null);          // controls: no per-frame text
          } else {
            const div = document.createElement("div");
            div.className = "hud-row" + (r.disabled ? " disabled" : "");
            const hl = document.createElement("span");
            hl.className = "hl"; hl.textContent = r.label;
            const hv = document.createElement("span");
            hv.className = "hv";
            div.append(hl, hv);
            els.rowsEl.appendChild(div);
            els.rowEls.push(hv);
          }
        });
        els.controlsBuilt = true;
      }
      rows.forEach((r, i) => {
        const el = els.rowEls[i];
        if (el && el.textContent !== r.text) el.textContent = r.text;
      });
      if (els.feedEl) {
        const lines = renderFeed(spec, bag);
        if (els._feedCache !== lines.join("\n")) {
          els._feedCache = lines.join("\n");
          els.feedEl.innerHTML = "";
          for (const ln of lines) {
            const d = document.createElement("div");
            if (ln.indexOf("✗") >= 0 || ln.indexOf("refused") >= 0)
              d.className = "bad";          // refusals highlighted
            d.textContent = ln;
            d.title = ln;                   // full text on hover
            els.feedEl.appendChild(d);
          }
        }
      }
    }
  },

  touch() { this._dirty = true; },

  unmount() {
    if (this._timer) { clearInterval(this._timer); this._timer = null; }
    if (this._root) this._root.innerHTML = "";   // container div stays
    this._root = null;
    this._els = {};
    this._specs = null;
    this._mounted = false;
    if (typeof window !== "undefined" && window._hud === this)
      window._hud = undefined;
  },
};

if (typeof window !== "undefined") window.SGHud = SGHud;
if (typeof module !== "undefined" && module.exports)
  module.exports = { SGHud, getPath, fmt, renderRows, renderFeed,
                     validatePanel, PANELS };
})();
