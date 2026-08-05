/* test_e4.node.js — headless node checks for the E4 client stack.
 *
 * Run:  node sigma_ground/radiance/web/test_e4.node.js   (exit 0 = pass)
 * Or via pytest: test_e4.py shells out here (skips when node is absent —
 * the test_camera.py environment-gated house pattern).
 *
 * Covers the DOM-free pure cores:
 *   net.js   — SGWorldNet against a FAKE wsFactory (the injected test
 *              seam): attach payload shape, ticket id correlation,
 *              refusal-without-id → tickets id null, save promise
 *              resolution, the honest everReady-false close reason.
 *   audio.js — mediumOf truth table (vacuum / pressure 0 / absent env),
 *              deriveVoice: the disclosed gain law (halves at 2×D_REF,
 *              clamped inside D_REF), vacuum → play:false with the
 *              structural-attachment disclosure, thud fallback on a null
 *              frequency (no pitch claim), tau == q/(π·f) exact.
 *   hud.js   — getPath/fmt edge cases, every SHIPPED panel renders, the
 *              NOVEL-spec LLM-composability proof (a spec never seen by
 *              the shipped array renders with zero render-code edits),
 *              unknown kind/format/action → disabled rows, never a throw.
 */
"use strict";
const path = require("path");
const { SGWorldNet, NO_COCKPIT_MSG } =
  require(path.join(__dirname, "net.js"));
const { mediumOf, deriveVoice, CHOICES } =
  require(path.join(__dirname, "audio.js"));
const { getPath, fmt, renderRows, renderFeed, validatePanel, PANELS } =
  require(path.join(__dirname, "hud.js"));

let fails = 0, checks = 0;
function ok(name, cond, detail) {
  checks++;
  if (!cond) { fails++; console.log(`FAIL  ${name}  ${detail || ""}`); }
}

// ═══ net.js — the fake websocket seam ═══════════════════════════════════
function fakeWS() {
  const ws = { sent: [], readyState: 0,
               send(s) { ws.sent.push(JSON.parse(s)); },
               close() { ws.readyState = 3;
                         if (ws.onclose) ws.onclose({}); } };
  return ws;
}
function msg(ws, obj) { ws.onmessage({ data: JSON.stringify(obj) }); }

(function testNet() {
  const readies = [], frames = [], events = [], tickets = [], protos = [];
  let closed = null;
  let ws;
  const net = new SGWorldNet({
    url: "ws://test/world", world: "w-test",
    items: ["a", "b"], pushedIndex: 1,
    onReady: (m) => readies.push(m),
    onFrame: (m) => frames.push(m),
    onEvent: (e) => events.push(e),
    onTicket: (t) => tickets.push(t),
    onProto: (m) => protos.push(m),
    onClose: (i) => { closed = i; },
    wsFactory: (url) => { ws = fakeWS(); ws.url = url; return ws; },
  });
  net.connect();
  ok("net.url", ws.url === "ws://test/world");
  ws.readyState = 1;
  ws.onopen();
  ok("net.attach-shape", JSON.stringify(ws.sent[0]) === JSON.stringify(
    { type: "attach", world: "w-test", items: ["a", "b"],
      pushed_index: 1 }), JSON.stringify(ws.sent[0]));

  msg(ws, { type: "ready", world: "w-test", labels: ["a", "b"] });
  ok("net.ready", readies.length === 1 && net.live && net.everReady);

  // frames fan out events one at a time
  msg(ws, { type: "frame", step: 3, bodies: [],
            events: [{ kind: "impact", body: "b" },
                     { kind: "bounce", body: "b" }] });
  ok("net.frame", frames.length === 1 && net.nframes === 1);
  ok("net.events-fanout", events.length === 2
     && events[0].kind === "impact" && events[1].kind === "bounce");

  // intent → applied ticket, id-correlated
  const id1 = net.intent("shove", { speed_m_s: 2.0 }, "b");
  ok("net.intent-wire", ws.sent[1].type === "intent"
     && ws.sent[1].id === id1 && ws.sent[1].verb === "shove"
     && ws.sent[1].target === "b");
  msg(ws, { type: "applied", id: id1, step_index: 7, target: "b",
            force: { kind: "impulse" } });
  ok("net.applied-ticket", tickets.length === 1 && tickets[0].id === id1
     && tickets[0].ok === true && /shove b/.test(tickets[0].text));

  // refusal WITH id names the sent summary
  const id2 = net.apply("b", { kind: "source", target: "b",
                               variables: {} });
  msg(ws, { type: "refusal", id: id2, text: "no thermal solver" });
  ok("net.refusal-ticket", tickets.length === 2 && tickets[1].id === id2
     && tickets[1].ok === false
     && /source b → refused: no thermal solver/.test(tickets[1].text));

  // refusal WITHOUT id (protocol-level) → ticket with id null, NEVER a
  // silent drop
  msg(ws, { type: "refusal", text: "world already exists" });
  ok("net.refusal-no-id", tickets.length === 3 && tickets[2].id === null
     && /world already exists/.test(tickets[2].text));

  // id-carrying protocol echo (intent-routed pace) → ok ticket + onProto
  const id3 = net.intent("pace", { factor: 2.0 }, null);
  msg(ws, { type: "pace", factor: 2.0, id: id3 });
  ok("net.pace-ticket", tickets.length === 4 && tickets[3].id === id3
     && tickets[3].ok === true && protos.length === 1);
  // bare protocol echo → onProto only, no ticket
  msg(ws, { type: "paused" });
  ok("net.bare-proto", protos.length === 2 && tickets.length === 4);

  // save promise resolves with the data payload
  let saved = null;
  net.save().then((d) => { saved = d; });
  const saveId = ws.sent[ws.sent.length - 1].id;
  msg(ws, { type: "save", id: saveId, data: { step_count: 9 } });
  Promise.resolve().then(() => {
    ok("net.save-promise", saved && saved.step_count === 9);

    // close AFTER ready: plain reason
    ws.close();
    ok("net.close-after-ready", closed && closed.everReady === true
       && !net.live);

    // a fresh net closed BEFORE ready: the honest cockpit message
    let closed2 = null, ws2;
    const net2 = new SGWorldNet({
      url: "ws://test/world", world: "w2",
      onClose: (i) => { closed2 = i; },
      wsFactory: () => { ws2 = fakeWS(); return ws2; },
    });
    net2.connect();
    ws2.close();                      // the static server's 404 path
    ok("net.close-pre-ready", closed2 && closed2.everReady === false
       && closed2.reason === NO_COCKPIT_MSG
       && /cockpit/.test(closed2.reason), closed2 && closed2.reason);

    finishAudio();
  });
})();

// ═══ audio.js — pure core ═══════════════════════════════════════════════
function finishAudio() {
  // mediumOf truth table
  ok("audio.medium-absent", mediumOf({}) === "air");
  ok("audio.medium-null", mediumOf(null) === "air");
  ok("audio.medium-air",
     mediumOf({ physics_env: { medium: "air" } }) === "air");
  ok("audio.medium-vacuum",
     mediumOf({ physics_env: { medium: "vacuum" } }) === "vacuum");
  ok("audio.medium-p0",
     mediumOf({ physics_env: { medium: "unknown", pressure_pa: 0 } })
     === "vacuum");

  const EV = { kind: "impact", position: [0, 0, 0],
               acoustic_energy_j: CHOICES.E_REF.value,   // full scale
               synth: { kind: "ring", frequency_hz: 800.0, q: 300.0 } };
  const D = CHOICES.D_REF.value;

  // vacuum: play false, the structural-attachment disclosure present
  const vac = deriveVoice(EV, [0, 0, 0], "vacuum", {});
  ok("audio.vacuum-silent", vac.play === false
     && /vacuum/.test(vac.reason)
     && /structural attachment/.test(vac.reason), vac.reason);

  // gain law: full-scale at D_REF, halves at 2×D_REF, clamped inside
  const g1 = deriveVoice(EV, [D, 0, 0], "air", { volume: 1 }).gain;
  const g2 = deriveVoice(EV, [2 * D, 0, 0], "air", { volume: 1 }).gain;
  const g0 = deriveVoice(EV, [0.1 * D, 0, 0], "air", { volume: 1 }).gain;
  ok("audio.gain-fullscale", g1 === 1.0, g1);
  ok("audio.gain-halves", Math.abs(g2 - 0.5) < 1e-12, g2);
  ok("audio.gain-clamped", g0 === g1, g0);

  // quarter energy → half amplitude (amplitude ∝ sqrt(E))
  const eq = deriveVoice(Object.assign({}, EV,
    { acoustic_energy_j: CHOICES.E_REF.value / 4 }),
    [D, 0, 0], "air", { volume: 1 }).gain;
  ok("audio.gain-sqrtE", Math.abs(eq - 0.5) < 1e-12, eq);

  // tau == q/(π·f) EXACT
  const v = deriveVoice(EV, [D, 0, 0], "air", {});
  ok("audio.tau-exact", v.tau === 300.0 / (Math.PI * 800.0), v.tau);
  ok("audio.kind-ring", v.kind === "ring" && v.freq === 800.0);

  // thud fallback: a null frequency makes NO pitch claim
  const thud = deriveVoice({ position: [0, 0, 0],
    acoustic_energy_j: 1e-4,
    synth: { kind: "thud", frequency_hz: null, q: null } },
    [0, 0, 0], "air", {});
  ok("audio.thud-fallback", thud.play === true && thud.kind === "thud"
     && thud.freq === CHOICES.THUD_CENTER.value
     && thud.tau === CHOICES.THUD_TAU.value);

  // no acoustic energy → honest no-play
  const dead = deriveVoice({ position: [0, 0, 0] }, [0, 0, 0], "air", {});
  ok("audio.no-energy", dead.play === false);

  // ── ADVERSARIAL REGRESSIONS (E4 refutation pass) ──
  // hostile position (NaN / string / short array) used to yield
  // play:true with gain NaN — WebAudio's setValueAtTime throws a
  // RangeError on non-finite values inside ws.onmessage. Now: refused
  // with the reason, never a non-finite number to the scheduler.
  for (const [label, pos] of [["nan", [NaN, 0, 0]], ["str", "over there"],
                              ["short", [1]], ["obj", { x: 1 }]]) {
    const h = deriveVoice(Object.assign({}, EV, { position: pos }),
                          [0, 0, 0], "air", {});
    ok("audio.hostile-pos-" + label, h.play === false
       && /non-finite gain/.test(h.reason), JSON.stringify(h));
  }
  const nanEye = deriveVoice(EV, [NaN, 0, 0], "air", {});
  ok("audio.hostile-listener", nanEye.play === false
     && /non-finite gain/.test(nanEye.reason));
  // an absurd finite wire Q cannot schedule an effectively-eternal tone
  const hq = deriveVoice(Object.assign({}, EV,
    { synth: { kind: "ring", frequency_hz: 800.0, q: 1e12 } }),
    [D, 0, 0], "air", {});
  ok("audio.tau-capped", hq.play === true
     && hq.tau === CHOICES.TAU_CAP.value, hq.tau);

  finishHud();
}

// ═══ hud.js — pure core ═════════════════════════════════════════════════
function finishHud() {
  // getPath: dots, arrays, missing
  const bag = { a: { b: [10, { c: 7 }] } };
  ok("hud.getpath", getPath(bag, "a.b.1.c") === 7);
  ok("hud.getpath-arr", getPath(bag, "a.b.0") === 10);
  ok("hud.getpath-missing", getPath(bag, "a.z.c") === undefined);
  ok("hud.getpath-null", getPath(null, "a") === undefined);

  // fmt: the closed format set
  ok("hud.fmt-dash", fmt(undefined, "fixed:2") === "—");
  ok("hud.fmt-fixed", fmt(1.2345, "fixed:2") === "1.23");
  ok("hud.fmt-sig", fmt(1234.5, "sig:3") === "1.23e+3");
  ok("hud.fmt-int", fmt(3.7, "int") === "4");
  ok("hud.fmt-mag", fmt([3, 0, 4], "mag:3") === "5.00");
  ok("hud.fmt-bool", fmt(true, "bool:paused|running") === "paused"
     && fmt(false, "bool:paused|running") === "running");
  ok("hud.fmt-hz-null", fmt(null, "hz") === "—");
  ok("hud.fmt-hz", fmt(812.4, "hz") === "812 Hz");
  ok("hud.fmt-unknown", fmt(1.0, "florps:3") === null);   // detectable

  // every SHIPPED panel renders against a realistic bag — no throws
  const shipBag = {
    world: { mode: "live", name: "w-x", err: null },
    frame: { step: 12, t: 0.4, pace: 1, paused: false, settled: false,
             energy_j: 0.92, momentum: [1, 0, 0],
             angular_momentum: [0, 0.5, 0] },
    tickets: [{ id: 1, icon: "✓", text: "applied shove plate @ step 3" },
              { id: 2, icon: "✗", text: "source plate → refused: no" }],
    events: [{ kind: "impact", body: "plate", surface: "floor",
               impact_speed_m_s: 1.83,
               synth: { frequency_hz: 812.4 }, heard: true }],
    audio: { status: "♪ live", medium: "air" },
    camera: { mode: "orbit", followLabel: null, speed: 1.2,
              clearance: 0.4 },
  };
  const actions = ["world.pausetoggle", "world.pace", "world.save",
                   "audio.mute", "audio.volume"];
  for (const spec of PANELS) {
    ok("hud.valid-" + spec.id, validatePanel(spec) === null);
    let rows, feed;
    let threw = false;
    try { rows = renderRows(spec, shipBag, actions);
          feed = renderFeed(spec, shipBag); }
    catch (e) { threw = true; }
    ok("hud.render-" + spec.id, !threw && Array.isArray(rows));
    if (spec.rows)
      ok("hud.rows-clean-" + spec.id, rows.every((r) => !r.disabled),
         JSON.stringify(rows.filter((r) => r.disabled)));
  }
  // shipped feed lines carry the real fields
  const tfeed = renderFeed(PANELS.find((p) => p.id === "tickets"), shipBag);
  ok("hud.ticket-line", tfeed.length === 2
     && tfeed[0] === "#2 ✗ source plate → refused: no", tfeed[0]);
  const afeed = renderFeed(PANELS.find((p) => p.id === "audio_feed"),
                           shipBag);
  ok("hud.event-line",
     afeed[0] === "impact plate on floor · 1.83 m/s · 812 Hz · ♪",
     afeed[0]);

  // ── THE LLM-COMPOSABILITY PROOF: a NOVEL spec never seen by the
  //    shipped array renders correctly with ZERO render-code edits ──
  const novel = { id: "novel_probe", title: "novel", dock: "bl",
    rows: [{ label: "impact E", path: "frame.energy_j",
             format: "sig:2", suffix: " J" },
           { label: "state", path: "frame.paused",
             format: "bool:❚❚|▶" }],
    feed: { path: "events", limit: 2,
            line: "{body}: {impact_speed_m_s:fixed:1} m/s" } };
  ok("hud.novel-valid", validatePanel(novel) === null);
  const nrows = renderRows(novel, shipBag, actions);
  ok("hud.novel-rows", nrows.length === 2
     && nrows[0].text === "0.92 J" && nrows[1].text === "▶",
     JSON.stringify(nrows));
  const nfeed = renderFeed(novel, shipBag);
  ok("hud.novel-feed", nfeed[0] === "plate: 1.8 m/s", nfeed[0]);

  // unknown kind/format/action → DISABLED rows naming the offender,
  // never a throw, never silence
  const hostile = { id: "hostile", title: "h", dock: "tl", rows: [
    { kind: "hologram", label: "x" },
    { label: "y", path: "frame.t", format: "florps:9" },
    { kind: "button", label: "z", action: "world.selfdestruct" }] };
  let threw = false, hrows = null;
  try { hrows = renderRows(hostile, shipBag, actions); }
  catch (e) { threw = true; }
  ok("hud.hostile-no-throw", !threw);
  ok("hud.hostile-kind", hrows[0].disabled
     && /hologram/.test(hrows[0].text));
  ok("hud.hostile-format", hrows[1].disabled
     && /florps:9/.test(hrows[1].text));
  ok("hud.hostile-action", hrows[2].disabled
     && /selfdestruct/.test(hrows[2].text));

  // bad panel specs are refused with reasons (validatePanel)
  ok("hud.bad-dock",
     /dock/.test(validatePanel({ id: "x", dock: "center", rows: [] })));
  ok("hud.bad-empty", validatePanel({ id: "x", dock: "tl" }) !== null);

  console.log(`${checks - fails}/${checks} checks passed`);
  if (fails) { console.log(`${fails} FAILED`); process.exit(1); }
  console.log("passed");
}
