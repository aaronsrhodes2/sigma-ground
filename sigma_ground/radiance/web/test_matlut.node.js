/* test_matlut.node.js — headless invariants for R4.7 Phase 1 (the legacy-
 * leaf material LUT that generalizes SVO's texelFetch pattern to ordinary
 * analytic CSG leaves, gated behind matLutEnabled/?matlut=1).
 *
 * Run:  node sigma_ground/radiance/web/test_matlut.node.js   (exit 0 = pass)
 *
 * viewer.js is not DOM-free (unlike svo.js/net.js/audio.js/hud.js, which
 * the other node suites require directly) — this file establishes the same
 * minimal DOM/GL stub tools/gen_shader.node.js already proved out, so
 * buildPathShader/buildMaterialPalette/buildMatLutArray can run headlessly
 * with zero browser cost.
 *
 * Covers the 3 gaps flagged during R4.7 Phase 1's review (verify_svo_
 * parity.py validates WHICH material wins geometrically, never the numeric
 * LUT payload; and the id-space/leaf-scoping claims were argued in prose,
 * not exercised):
 *   1. LUT-value correctness — a leaf's exact material fields survive into
 *      the packed Float32Array unchanged.
 *   2. The 256-overflow boundary — 300 uniquely-colored legacy leaves: the
 *      256 that fit the palette route through the LUT, the 44 that don't
 *      keep their own correct per-leaf fallback line, and no leaf's
 *      material gets silently hijacked by the wrong id range.
 *   3. Shared material / divergent temperature — two leaves sharing one
 *      material label (so colorfn/etc. correctly dedupe them) must still
 *      report INDEPENDENT temperatures (tempfn/bodyTempFn must NOT dedupe).
 */
"use strict";
const path = require("path");
const WEB = __dirname;

function makeEl() {
  return {
    addEventListener() {}, appendChild() {}, append() {}, remove() {},
    classList: { toggle: () => false, add() {}, remove() {}, contains: () => false },
    style: {}, options: [], dataset: {}, innerHTML: "", textContent: "",
    value: "0", title: "",
    querySelector: () => null, getBoundingClientRect: () => ({ width: 0, height: 0, left: 0, top: 0 }),
  };
}
const els = new Map();
const glStub = new Proxy({}, {
  get(_t, prop) {
    if (prop === "getParameter") return () => [1, 1024];
    if (prop === "getExtension") return () => ({ loseContext() {} });
    if (prop === "getShaderParameter" || prop === "getProgramParameter") return () => true;
    if (prop === "getUniformLocation" || prop === "getAttribLocation") return () => null;
    if (typeof prop === "string" && /^[A-Z0-9_]+$/.test(prop)) return 1;
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
global.location = { search: "?svodry=1&svo=0&matlut=1", host: "offline" };
Object.defineProperty(global, "navigator", {
  value: { userAgent: "node-test-matlut", sendBeacon() {} }, configurable: true });
global.requestAnimationFrame = () => {};
global.fetch = () => new Promise(() => {});
if (typeof global.Event === "undefined") global.Event = class Event { constructor(n) { this.type = n; } };
global.dispatchEvent = () => {};
global.addEventListener = () => {};
global.devicePixelRatio = 1;
global.innerWidth = 640; global.innerHeight = 400;

const S = require(path.join(WEB, "splats.js")) || {};
require(path.join(WEB, "svo.js"));
require(path.join(WEB, "viewer.js"));   // matLutEnabled starts TRUE; _assertMatLutGate already ran
const SVO = window.SGSVO;
const R = window.Radiance;

let fails = 0, checks = 0;
function ok(name, cond, detail) {
  checks++;
  if (!cond) { fails++; console.log(`FAIL  ${name}  ${detail || ""}`); }
}
function f32(x) { return Math.fround(x); }

function scene(leaves, materials) {
  return {
    csg_leaves: leaves, materials,
    bodies: leaves.map((_, i) => ({ label: `b${i}` })),
    bbox: [[-5, 5], [-5, 5], [-5, 5]],
  };
}
function sphereLeaf(i, material, extra) {
  return Object.assign({ op: "add", material, body: i,
    shape: { type: "Sphere", center: [i * 2, 0, 0], radius: 0.4 } }, extra || {});
}

// ═══ 1. LUT-value correctness ════════════════════════════════════════════
(function testLutValues() {
  const mats = { distinct_mat: { color_rgb: [0.11, 0.22, 0.33], metal: true,
    emissivity_rgb: [0.44, 0.55, 0.66], refractive_index: 1.5, opacity: 0.75 } };
  const sc = scene([sphereLeaf(0, "distinct_mat")], mats);
  const pal = SVO.buildMaterialPalette(sc.csg_leaves.map(l => l.material));
  ok("palette has exactly the one label", pal.palette.length === 1 && pal.palette[0] === "distinct_mat",
     JSON.stringify(pal.palette));
  const lut = R.buildMatLutArray(sc, pal.palette);
  const id = pal.palIdx.get("distinct_mat");
  ok("color.r exact", lut[(0 * 256 + id) * 4 + 0] === f32(0.11), lut[(0 * 256 + id) * 4 + 0]);
  ok("color.g exact", lut[(0 * 256 + id) * 4 + 1] === f32(0.22), lut[(0 * 256 + id) * 4 + 1]);
  ok("color.b exact", lut[(0 * 256 + id) * 4 + 2] === f32(0.33), lut[(0 * 256 + id) * 4 + 2]);
  ok("metal flag exact (row0.a)", lut[(0 * 256 + id) * 4 + 3] === 1, lut[(0 * 256 + id) * 4 + 3]);
  ok("emissivity.r exact", lut[(1 * 256 + id) * 4 + 0] === f32(0.44), lut[(1 * 256 + id) * 4 + 0]);
  ok("emissivity.g exact", lut[(1 * 256 + id) * 4 + 1] === f32(0.55), lut[(1 * 256 + id) * 4 + 1]);
  ok("emissivity.b exact", lut[(1 * 256 + id) * 4 + 2] === f32(0.66), lut[(1 * 256 + id) * 4 + 2]);
  ok("ior exact (row1.a)", lut[(1 * 256 + id) * 4 + 3] === f32(1.5), lut[(1 * 256 + id) * 4 + 3]);
  ok("opacity exact (row2.x)", lut[(2 * 256 + id) * 4 + 0] === f32(0.75), lut[(2 * 256 + id) * 4 + 0]);
})();

// ═══ 2. 256-leaf overflow boundary — no id-range hijacking ════════════════
(function testOverflowBoundary() {
  const N = 300;
  const mats = {};
  const leaves = [];
  for (let i = 0; i < N; i++) {
    const label = `uniq_${i}`;
    mats[label] = { color_rgb: [f32(i / N), f32(1 - i / N), 0.5] };
    leaves.push(sphereLeaf(i, label));
  }
  const sc = scene(leaves, mats);
  const pal = SVO.buildMaterialPalette(sc.csg_leaves.map(l => l.material));
  ok("palette capped at 256", pal.palette.length === 256, pal.palette.length);
  ok("overflow is exactly the remaining 44", pal.overflow.length === N - 256, pal.overflow.length);
  ok("overflow preserves first-seen order (leaf 256 is the first overflow)",
     pal.overflow[0] === "uniq_256", pal.overflow[0]);

  const shader = R.buildPathShaderFor(sc);
  ok("shader generated for the 300-leaf overflow scene", !!shader);
  if (shader) {
    // a PALETTE-FIT leaf (index 0, material uniq_0, palette id 0) must NOT
    // carry its own inline colorfn line any more -- it's LUT-routed.
    ok("palette-fit leaf 0 has no inline colorfn line (routed via LUT)",
       !new RegExp("if\\(m==0\\) return vec3\\(").test(shader));
    // an OVERFLOW leaf (index 256+, e.g. 299) MUST keep its own correct
    // inline colorfn line -- the promised graceful fallback, not a crash
    // and not a silently wrong color.
    const c299 = mats["uniq_299"].color_rgb;
    const expected = `if(m==299) return vec3(${c299[0].toFixed(8)},${c299[1].toFixed(8)},${c299[2].toFixed(8)});`;
    ok("overflow leaf 299 keeps its OWN correct inline colorfn line",
       shader.indexOf(expected) >= 0, "expected: " + expected);
    // the overflow leaf's `mat` must stay its RAW, non-negative index (299)
    // -- never accidentally negative (the LUT range) or otherwise hijacked.
    ok("overflow leaf 299's matsel assigns its raw index, not a LUT id",
       /mat=299;/.test(shader) && !/mat=-300;/.test(shader));
  }
})();

// ═══ 3. shared material, divergent temperature — tempfn must NOT dedupe ═══
(function testSharedMaterialDivergentTemp() {
  const mats = { shared_mat: { color_rgb: [0.6, 0.6, 0.6] } };
  const leaves = [
    sphereLeaf(0, "shared_mat", { temperature_k: 500 }),
    sphereLeaf(1, "shared_mat", { temperature_k: 900 }),
  ];
  const sc = scene(leaves, mats);
  const shader = R.buildPathShaderFor(sc);
  ok("shader generated for the shared-material scene", !!shader);
  if (shader) {
    // colorfn: both leaves share material -> only ONE leaf should still
    // carry a per-leaf inline line at most (both may be palette-routed and
    // carry NEITHER, but they must never each carry their own DIFFERING
    // inline color line if the material truly deduped) -- the real
    // assertion that matters here is temperature, checked below; this is
    // a sanity check that the palette dedup actually happened (one entry).
    const pal = SVO.buildMaterialPalette(sc.csg_leaves.map(l => l.material));
    ok("shared material deduped to ONE palette entry", pal.palette.length === 1, pal.palette.length);
    // tempfn/bodyTempFn: must NOT be deduped -- both leaves' own distinct
    // temperatures must still appear, keyed by leafId (matLutActive).
    ok("leaf 0's own temperature (500K) present, keyed by leafId",
       /if\(leafId==0\) return 500\.00000000;/.test(shader));
    ok("leaf 1's own temperature (900K) present, keyed by leafId",
       /if\(leafId==1\) return 900\.00000000;/.test(shader));
  }
})();

console.log(`${checks - fails}/${checks} checks pass`);
process.exit(fails ? 1 : 0);
