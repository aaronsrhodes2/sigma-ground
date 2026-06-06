/* Mentat chat — the text view beside Radiance's simulation view.
 *
 * One text input → the front door (POST /chat → ASK / SIMULATE / RENDER). A
 * RENDER writes data/<slug>.json server-side and returns its slug; the client
 * drops it straight into the SHARED viewer via window.Radiance.load — no iframe,
 * no reload, PT state preserved. Multi-turn ("drop it" → "yes") threads on a
 * stable session id. Speech-to-text is the browser Web Speech API; swap the
 * recognizer for a MediaRecorder → POST /stt (local Whisper) seam to stay
 * fully offline. Zero dependencies, IIFE like viewer.js.
 */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const log = $("chatlog"), input = $("chatinput"), sendBtn = $("chatsend"),
        mic = $("mic"), statusEl = $("chatstatus"), pane = $("chatpane"), bChat = $("b-chat");
  if (!log || !input) return;                       // viewer-only page; nothing to wire

  // ── explicit lane (static routing): auto | ask | simulate | render ──
  let mode = "auto";
  const SLASH = { ask:"ask", question:"ask", simulate:"simulate", sim:"simulate",
                  render:"render", draw:"render", auto:"auto" };
  const modeBtns = [...document.querySelectorAll(".modebtn")];
  function setMode(m) { mode = m;
    modeBtns.forEach(b => b.classList.toggle("on", b.dataset.mode === m));
    input.placeholder =
        m === "render"   ? "describe a sim to watch — “drop a feather from 8 feet”…"
      : m === "simulate" ? "describe a physics sim — “how fast does a steel ball fall from 10 km”…"
      : m === "ask"      ? "ask a physics question…"
      :                    "ask, or “drop a feather from 8 feet” — pick a lane above";
  }
  modeBtns.forEach(b => b.addEventListener("click", () => setMode(b.dataset.mode)));

  // Stable per-tab session so a later "yes" renders the sim the previous turn set up.
  const SID = localStorage.getItem("mentat_sid")
    || (v => (localStorage.setItem("mentat_sid", v), v))(
         (crypto.randomUUID && crypto.randomUUID()) || (Date.now() + "_" + Math.random()));

  // ── bubbles (text on black, NO fill; role = colour + side) ──
  function bubble(role, text) {
    const el = document.createElement("div");
    el.className = "msg " + role;
    el.textContent = text;                          // textContent → no HTML injection
    log.appendChild(el); log.scrollTop = log.scrollHeight;
    return el;
  }
  const status = (m) => { statusEl.textContent = m || ""; };
  function roleFor(env) {
    if (env.error || env.intent === "error") return "error";
    if (env.intent === "clarify") return "clarify";
    return "mentat";                                // ask / simulate / render → green
  }

  // The window.Radiance hook is defined at the very end of viewer.js's IIFE; wait
  // for it so the first render-triggering message can't race the viewer's own boot.
  function ensureRadiance() {
    if (window.Radiance && window.Radiance.load) return Promise.resolve();
    return new Promise(res => window.addEventListener("radiance-ready", res, { once: true }));
  }

  // ── send → dispatch → render ──
  let busy = false;
  function setBusy(b) { busy = b; input.disabled = b; sendBtn.disabled = b; }

  async function send(text) {
    text = (text || "").trim();
    if (!text || busy) return;
    // a leading /ask /simulate /render (or /question /sim /draw /auto) forces the lane
    let msgMode = mode;
    const m = text.match(/^\/(ask|question|simulate|sim|render|draw|auto)\b[\s:]*/i);
    if (m) { msgMode = SLASH[m[1].toLowerCase()]; setMode(msgMode); text = text.slice(m[0].length).trim(); }
    if (!text) { setBusy(false); return; }
    bubble("user", (msgMode !== "auto" ? "/" + msgMode + " " : "") + text);
    input.value = ""; setBusy(true);
    const pend = bubble("pending", msgMode === "render" ? "Mentat is rendering…" : "Mentat is thinking…");
    try {
      const r = await fetch("/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, session_id: SID, use_llm: true, mode: msgMode }),
      });
      const env = await r.json();
      pend.remove();
      if (env.degraded) bubble("clarify", "(ollama unavailable — deterministic mode)");
      bubble(roleFor(env), env.text || "(no reply)");

      // A RENDER landed → drop the saved sim into the simulation view, instantly.
      if (env.saved && env.saved.slug) {
        await ensureRadiance();
        try {
          await window.Radiance.load("data/" + env.saved.slug + ".json",
                                     env.saved.title || env.saved.slug);
          bubble("dim", "▶ loaded in the sim — press play, orbit, or open ⟨ / ⟩ behind.");
        } catch (e) {
          bubble("error", "rendered the data, but the viewer couldn't load it: " + e);
        }
      }
    } catch (e) {
      pend.remove();
      bubble("error", "offline — is the server running? (" + e + ")");
    } finally { setBusy(false); input.focus(); }
  }

  // ── speech-to-text: browser Web Speech API ──
  // NOTE: in Chrome this routes audio to Google's cloud. For a fully-local mic,
  // replace the recognizer below with a MediaRecorder → POST /stt (Whisper) seam.
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  let rec = null, listening = false;
  if (SR) {
    rec = new SR();
    rec.lang = "en-US"; rec.interimResults = false; rec.maxAlternatives = 1;
    rec.onresult = (e) => { const t = e.results[0][0].transcript; status(""); if (t) send(t); };
    rec.onerror = (e) => status("mic: " + (e.error || "error"));
    rec.onend = () => { listening = false; mic.classList.remove("rec");
                        if (statusEl.textContent === "● listening…") status(""); };
  } else {
    mic.disabled = true; mic.title = "Voice input not supported in this browser";
  }
  function startMic() {
    if (!rec || listening || busy) return;
    try { rec.start(); listening = true; mic.classList.add("rec"); status("● listening…"); }
    catch (e) { status("mic: " + e.message); }
  }

  // ── wiring ──
  sendBtn.addEventListener("click", () => send(input.value));
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); send(input.value); } });
  mic.addEventListener("click", () => listening ? rec.stop() : startMic());
  bChat.addEventListener("click", () => {
    const hidden = pane.classList.toggle("collapsed");
    bChat.classList.toggle("on", !hidden);
    if (!hidden) input.focus();
  });

  // ── init: viewer-focused on a deep-link (?scene=), cockpit otherwise ──
  if (new URLSearchParams(location.search).get("scene")) {
    pane.classList.add("collapsed"); bChat.classList.remove("on");
  } else {
    bubble("mentat", "I'm Mentat. Ask me physics, or describe a simulation — try "
      + "'how fast does a 5 cm steel ball hit the ground from 10 km?' or "
      + "'drop a feather from 8 feet'.");
    bubble("dim", "Pick a lane above (or type /ask /simulate /render): "
      + "render runs the sim AND loads the watchable .json right here — no 'yes' needed."
      + (SR ? " 🎤 click the mic to talk." : ""));
    input.focus();
  }
})();
