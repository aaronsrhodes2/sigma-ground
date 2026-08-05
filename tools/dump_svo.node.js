/* dump_svo.node.js — fixture generator for tools/verify_svo_parity.py.
 *
 * Builds the ACTUAL svo.js scene pack (the same builder viewer.js runs at
 * load) for a scene's SVO-eligible Voxel leaves and dumps it to disk so the
 * Python parity twin replays traversal over the REAL builder output — one
 * builder, no parallel Python reimplementation that could drift.
 *
 *   node tools/dump_svo.node.js <scene.json> <outdir>
 *
 * Writes: pool.bin (u32 LE), atlas.bin (u8, x-fastest 3-D), meta.json.
 * Eligibility mirrors viewer.js's rule exactly: top-level Voxel shape,
 * body-bound, op:add (or leaf 0).
 */
"use strict";
const fs = require("fs");
const path = require("path");
const S = require(path.join(__dirname, "..", "sigma_ground", "radiance", "web", "svo.js"));

const [scenePath, outDir] = process.argv.slice(2);
if (!scenePath || !outDir) { console.error("usage: node dump_svo.node.js <scene.json> <outdir>"); process.exit(2); }
const o = JSON.parse(fs.readFileSync(scenePath, "utf-8"));
const sc = o.kind === "trajectory" ? o.scene : o;
const bodies = sc.bodies || [];
const eligible = [];
(sc.csg_leaves || []).forEach((l, i) => {
  const s = l.shape;
  if (s && s.type === "Voxel" && l.body != null && (i === 0 || !l.op || l.op === "add")) {
    if (s.sdf_ref != null) {                       // R4.5 payload sharing
      const src = sc.csg_leaves[s.sdf_ref].shape;
      s.sdf_b64 = src.sdf_b64;
    }
    eligible.push({ leaf: i, body: l.body, shape: s, material: l.material,
                    pivot: (bodies[l.body] && bodies[l.body].pivot) || [0, 0, 0],
                    bounds: l.bounds || null });
  }
});
if (!eligible.length) { console.error("no SVO-eligible Voxel leaves"); process.exit(3); }
const packed = S.packScene(eligible);
fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(path.join(outDir, "pool.bin"), Buffer.from(packed.pool.buffer));
fs.writeFileSync(path.join(outDir, "atlas.bin"), Buffer.from(packed.atlas.buffer));
fs.writeFileSync(path.join(outDir, "meta.json"), JSON.stringify({
  atlasDim: packed.atlasDim, bpa: packed.bpa, palette: packed.palette,
  bodies: packed.bodies, stats: packed.stats,
  eligible: eligible.map(e => ({ leaf: e.leaf, body: e.body, material: e.material,
    pivot: e.pivot, bounds: e.bounds,
    dims: e.shape.dims, voxel_size: e.shape.voxel_size, band: e.shape.band,
    center: e.shape.center })),
}, null, 1));
console.log(JSON.stringify(packed.stats));
