/* Mentat — the agent view.
 *
 * One input → the front door (POST /chat → ASK / SIMULATE / RENDER). Auto-routed
 * by default; a quiet lane override (or a leading /ask /simulate /render) forces
 * it. Answers render as structured turns: a grounded value, the worked provenance
 * tucked behind a disclosure, and — when a sim is renderable — a "▶ Render this"
 * action (no "type yes" dance) or an inline render card that loads the watchable
 * 3D fall into the immersive viewer (viewer.js, window.Radiance.load). Multi-turn
 * threads on a stable session id. Zero dependencies, IIFE like viewer.js.
 */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const thread = $("thread"), input = $("chatinput"), sendBtn = $("chatsend"),
        mic = $("mic"), statusEl = $("chatstatus"), scroller = $("scroll"),
        viewer = $("viewer"), openV = $("open-viewer"), closeV = $("close-viewer"),
        bLib = $("b-lib");
  if (!thread || !input) return;                      // not the agent page

  // ── per-tab session (so a later "yes"/"render that" finds the pending sim) ──
  const SID = localStorage.getItem("mentat_sid")
    || ((v) => (localStorage.setItem("mentat_sid", v), v))(
         (crypto.randomUUID && crypto.randomUUID()) || (Date.now() + "_" + Math.random()));

  // ── lane override (auto | ask | simulate | render) ──
  let mode = "auto";
  const SLASH = { ask:"ask", question:"ask", simulate:"simulate", sim:"simulate",
                  render:"render", draw:"render", auto:"auto" };
  const laneBtns = [...document.querySelectorAll(".lane")];
  function setMode(m) { mode = m;
    laneBtns.forEach((b) => b.classList.toggle("on", b.dataset.mode === m)); }
  laneBtns.forEach((b) => b.addEventListener("click", () => { setMode(b.dataset.mode); input.focus(); }));

  // ── viewer overlay ──
  const showViewer = () => viewer.classList.add("open");
  const hideViewer = () => { viewer.classList.remove("open"); input.focus(); };
  openV && openV.addEventListener("click", showViewer);
  closeV && closeV.addEventListener("click", hideViewer);
  bLib && bLib.addEventListener("click", showViewer);   // the library lives inside the viewer
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && viewer.classList.contains("open")) hideViewer(); });

  // viewer.js publishes window.Radiance.load after it boots; wait for it so the
  // first render can't race the viewer's init.
  const ensureRadiance = () => (window.Radiance && window.Radiance.load)
    ? Promise.resolve()
    : new Promise((res) => window.addEventListener("radiance-ready", res, { once: true }));

  // ── small DOM helpers (textContent everywhere → no injection) ──
  const el = (tag, cls, txt) => { const e = document.createElement(tag);
    if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
  const scrollDown = () => { scroller.scrollTop = scroller.scrollHeight; };
  // envelope.text should be a string, but guard so a structured answer can never
  // render as "[object Object]".
  const asText = (x) => typeof x === "string" ? x
    : (x && typeof x === "object" && x.answer_text) ? x.answer_text
    : (x == null ? "" : String(x));
  function turn(role) {
    const t = el("div", "turn " + role);
    t.appendChild(el("div", "who", role === "user" ? "You"
      : role === "error" ? "Mentat · offline" : "Mentat"));
    thread.appendChild(t); scrollDown(); return t;
  }
  const body = (t, txt) => { const b = el("div", "body", txt); t.appendChild(b); scrollDown(); return b; };

  // ── format a SIMULATE envelope: lead sentence + collapsible work + render action ──
  function renderSim(t, env, text) {
    let route = "", rest = text;
    const m = rest.match(/^\[routed → ([\s\S]+?)\]\s*\n+/);
    if (m) { route = m[1]; rest = rest.slice(m[0].length); }
    rest = rest.replace(/\n*Do you want Radiance to render[\s\S]*?Reply 'yes'\.?/i, "").trim();
    let lead = rest, work = "";
    const bi = rest.indexOf("\n  •");
    if (bi >= 0) { lead = rest.slice(0, bi).trim(); work = rest.slice(bi).trim(); }
    t.classList.add("answer");
    body(t, lead).classList.add("lead");
    if (work || route) {
      const d = el("details", "work");
      d.appendChild(el("summary", null, "show the working"));
      if (route) d.appendChild(el("div", "route mono", "routed → " + route));
      if (work) { const pre = el("pre"); pre.textContent = work; d.appendChild(pre); }
      t.appendChild(d);
    }
    if (env.source) t.appendChild(el("div", "src", "source: " + env.source));
    if (env.can_render && !(env.saved && env.saved.slug)) {
      const row = el("div");
      const btn = el("button", "open", "▶ Render this in 3D");
      btn.addEventListener("click", () => send("yes"));     // confirms the pending render
      row.appendChild(btn); t.appendChild(row);
    }
    scrollDown();
  }

  // ── format an ASK envelope: the grounded answer + its source ──
  function renderAsk(t, env, text) {
    body(t, text);
    if (env.source) t.appendChild(el("div", "src", "source: " + env.source));
  }

  // ── an inline render card: the watchable sim, loaded + openable ──
  async function attachRender(t, saved) {
    const card = el("div", "rcard");
    card.appendChild(el("span", "play", "▶"));
    const meta = el("div", "meta");
    meta.appendChild(el("b", null, saved.title || saved.slug));
    meta.appendChild(el("div", "d", "watchable 3D fall — click to watch"));
    card.appendChild(meta);
    card.appendChild(el("span", "open", "Watch ›"));
    t.appendChild(card); scrollDown();
    const watch = async () => {
      try {
        await ensureRadiance();
        await window.Radiance.load("data/" + saved.slug + ".json", saved.title || saved.slug);
        showViewer();
      } catch (e) { body(turn("error"), "rendered the data, but the viewer couldn't load it: " + e); }
    };
    card.addEventListener("click", watch);
    await watch();                                   // render → load + open straightaway
  }

  function renderEnvelope(env) {
    const role = (env.error || env.intent === "error") ? "error"
      : env.intent === "clarify" ? "clarify" : "mentat";
    if (env.degraded) body(turn("clarify"), "(ollama unavailable — deterministic mode)");
    const t = turn(role);
    const text = asText(env.text) || "(no reply)";
    if (role === "mentat" && env.intent === "simulate" && /\[routed →/.test(text)) renderSim(t, env, text);
    else if (role === "mentat" && env.intent === "ask") renderAsk(t, env, text);
    else body(t, text);
    if (env.saved && env.saved.slug) attachRender(t, env.saved);
  }

  // ── send → dispatch → render ──
  let busy = false;
  const setBusy = (b) => { busy = b; input.disabled = b; sendBtn.disabled = b; };

  async function send(text) {
    text = (text || "").trim();
    if (!text || busy) return;
    let msgMode = mode;
    const m = text.match(/^\/(ask|question|simulate|sim|render|draw|auto)\b[\s:]*/i);
    if (m) { msgMode = SLASH[m[1].toLowerCase()]; setMode(msgMode); text = text.slice(m[0].length).trim(); }
    if (!text) return;
    body(turn("user"), text);
    input.value = ""; autoGrow(); setBusy(true);
    const pend = turn("pending"); body(pend, msgMode === "render" ? "Rendering…" : "Thinking…");
    try {
      const r = await fetch("/chat", { method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, session_id: SID, use_llm: true, mode: msgMode }) });
      const env = await r.json();
      pend.remove();
      renderEnvelope(env);
    } catch (e) {
      pend.remove();
      body(turn("error"), "offline — is the server running? (" + e + ")");
    } finally { setBusy(false); input.focus(); }
  }

  // ── composer: textarea auto-grow, Enter sends / Shift+Enter newline ──
  function autoGrow() { input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px"; }
  input.addEventListener("input", autoGrow);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input.value); } });
  sendBtn.addEventListener("click", () => send(input.value));

  // ── speech-to-text (browser Web Speech API; swap for /stt + Whisper to stay local) ──
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  let rec = null, listening = false;
  const status = (m) => { statusEl.textContent = m || ""; };
  if (SR) {
    rec = new SR(); rec.lang = "en-US"; rec.interimResults = false; rec.maxAlternatives = 1;
    rec.onresult = (e) => { const txt = e.results[0][0].transcript; status(""); if (txt) send(txt); };
    rec.onerror = (e) => status("mic: " + (e.error || "error"));
    rec.onend = () => { listening = false; mic.classList.remove("rec");
      if (statusEl.textContent === "● listening…") status(""); };
    mic.addEventListener("click", () => {
      if (listening) { rec.stop(); return; }
      if (busy) return;
      try { rec.start(); listening = true; mic.classList.add("rec"); status("● listening…"); }
      catch (e) { status("mic: " + e.message); } });
  } else { mic.disabled = true; mic.title = "Voice input not supported in this browser"; }

  // ── welcome: skill callouts (fill the input) + suggestions (send on click) ──
  function welcome() {
    const t = turn("mentat");
    body(t, "I'm Mentat. I answer physics & chemistry from grounded models — with "
      + "the working shown — run simulations, and render watchable 3D falls. Ask me "
      + "anything, or start here:");
    const skills = el("div", "skills");
    [["Ask physics", "escape velocity · half-life · a photon's energy", "ask",
      "What's the escape velocity of Mars?"],
     ["Run a simulation", "drop / fall / terminal velocity — numbers + provenance", "simulate",
      "how fast does a 5 cm steel ball hit the ground from 10 km?"],
     ["Render a fall", "build a watchable 3D sim and play it here", "render",
      "drop a feather from 8 feet and watch it fall"]
    ].forEach(([st, sd, m, q]) => {
      const b = el("button", "skill"); b.type = "button";
      b.appendChild(el("div", "st", st)); b.appendChild(el("div", "sd", sd));
      b.addEventListener("click", () => { setMode(m); input.value = q; input.focus(); autoGrow(); });
      skills.appendChild(b); });
    t.appendChild(skills);
    const sug = el("div", "suggest");
    ["Schwarzschild radius of the Sun", "does an iron sphere heat up falling from 30 km?",
     "pH of 0.01 M HCl", "drop a brick from 3 m"].forEach((q) => {
      const c = el("button", "chip", q); c.type = "button";
      c.addEventListener("click", () => send(q)); sug.appendChild(c); });
    t.appendChild(sug); scrollDown();
  }

  // ── init: deep-link (?scene=) opens the viewer on that scene; else the welcome ──
  const deep = new URLSearchParams(location.search).get("scene");
  if (deep) {
    body(turn("mentat"), "Loading “" + deep + "” in the viewer…");
    ensureRadiance().then(() => window.Radiance.load("data/" + deep + ".json", deep)
      .then(showViewer).catch((e) => body(turn("error"), "couldn't load that scene: " + e)));
  } else {
    welcome(); input.focus();
  }
})();
