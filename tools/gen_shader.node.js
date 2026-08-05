/* gen_shader.node.js — OFFLINE shader-text generator (R4 bisect instrument).
 *
 * Loads the real viewer.js under a minimal DOM/GL stub with ?svodry=1 so
 * buildProgram stashes the generated PT + shim shader TEXTS without ever
 * compiling (a wedged ANGLE compile blocks the shared same-site renderer
 * main thread — measured this session — so text generation must not touch
 * a browser at all).
 *
 *   node tools/gen_shader.node.js <scene.json> <out-prefix> [svo=1]
 *
 * Writes <out-prefix>_pt.glsl and <out-prefix>_fast.glsl.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const WEB = path.join(__dirname, "..", "sigma_ground", "radiance", "web");

const [scenePath, outPrefix, svoFlag] = process.argv.slice(2);
if (!scenePath || !outPrefix) {
  console.error("usage: node tools/gen_shader.node.js <scene.json> <out-prefix> [svo=1]");
  process.exit(2);
}
const svo = svoFlag === "svo=1" ? "1" : "0";

// ── minimal DOM/GL stubs ──────────────────────────────────────────────────
function makeEl() {
  const el = {
    addEventListener() {}, appendChild() {}, append() {}, remove() {},
    classList: { toggle: () => false, add() {}, remove() {}, contains: () => false },
    style: {}, options: [], dataset: {}, innerHTML: "", textContent: "",
    value: "0", title: "",
    querySelector: () => null, getBoundingClientRect: () => ({ width: 0, height: 0, left: 0, top: 0 }),
  };
  return el;
}
const els = new Map();
const glStub = new Proxy({}, {
  get(_t, prop) {
    if (prop === "getParameter") return () => [1, 1024];
    if (prop === "getExtension") return () => ({ loseContext() {} });
    if (prop === "getShaderParameter" || prop === "getProgramParameter") return () => true;
    if (prop === "getUniformLocation" || prop === "getAttribLocation") return () => null;
    if (typeof prop === "string" && /^[A-Z0-9_]+$/.test(prop)) return 1;   // GL enums
    return () => ({});
  },
});
const canvas = Object.assign(makeEl(), {
  getContext: () => glStub, width: 64, height: 64, clientWidth: 64, clientHeight: 64,
});
global.window = global;
global.document = {
  getElementById(id) {
    if (id === "gl") return canvas;
    if (!els.has(id)) els.set(id, makeEl());
    return els.get(id);
  },
  createElement: () => makeEl(),
};
global.location = { search: `?svodry=1&svo=${svo}`, host: "offline" };
Object.defineProperty(global, "navigator", {
  value: { userAgent: "node-genshader", sendBeacon() {} }, configurable: true });
global.requestAnimationFrame = () => {};
global.fetch = () => new Promise(() => {});      // load() never resolves — we drive applyObj directly
if (typeof global.Event === "undefined") global.Event = class Event { constructor(n) { this.type = n; } };
global.dispatchEvent = () => {};
global.addEventListener = () => {};
global.devicePixelRatio = 1;
global.innerWidth = 640; global.innerHeight = 400;

// real modules the viewer consumes
require(path.join(WEB, "splats.js"));            // window.SGSplats (banding fidelity)
require(path.join(WEB, "svo.js"));               // window.SGSVO
require(path.join(WEB, "viewer.js"));            // wires window.Radiance

const o = JSON.parse(fs.readFileSync(scenePath, "utf-8"));
window.Radiance.applyObj(o, "gen", false);
if (!window.__shaders) { console.error("no __shaders — svodry path did not run"); process.exit(3); }
fs.writeFileSync(outPrefix + "_pt.glsl", window.__shaders.pt);
fs.writeFileSync(outPrefix + "_fast.glsl", window.__shaders.fast);
console.log(JSON.stringify({ pt_len: window.__shaders.pt.length,
                             fast_len: window.__shaders.fast.length }));
process.exit(0);
