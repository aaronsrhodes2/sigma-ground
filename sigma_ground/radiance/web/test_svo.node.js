/* test_svo.node.js — headless invariants for svo.js (R4.1).
 *
 * Run:  node sigma_ground/radiance/web/test_svo.node.js   (exit 0 = pass)
 *
 * Covers (blueprint R4.1): builder invariants (root brick, contiguous
 * 8-child blocks, byte preservation of the shipped finest-level payload),
 * the Sec-1 texel math at cell centers, brick-seam C0 continuity at matched
 * level (the border-ring guarantee, checked numerically), uniform collapse
 * on a synthetic sphere, dense-vs-SVO agreement at matched (finest) LOD,
 * palette refusal, content-hash tree sharing, and the GLSL chunk's ANGLE-law
 * hygiene (no uniform arrays, texelFetch-only per-body data).
 */
"use strict";
const path = require("path");
const fs = require("fs");
const S = require(path.join(__dirname, "svo.js"));

let fails = 0, checks = 0;
function ok(name, cond, detail) {
  checks++;
  if (!cond) { fails++; console.log(`FAIL  ${name}  ${detail || ""}`); }
}

// ── synthetic sphere grid (deterministic, no RNG) ─────────────────────────
function sphereShape(n, r_cells) {
  const vs = 0.01, band = 6 * vs;
  const g = new Int8Array(n * n * n);
  const c = (n - 1) / 2;
  for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const d = (Math.hypot(i - c, j - c, k - c) - r_cells) * vs;   // metres
    g[(k * n + j) * n + i] = Math.max(-127, Math.min(127, Math.round(d * 127 / band)));
  }
  // base64 the payload exactly like scene_export (x-fastest already)
  const b64 = Buffer.from(Buffer.from(g.buffer)).toString("base64");
  return { type: "Voxel", center: [0, 0, 0], dims: [n, n, n], voxel_size: vs,
           band, encoding: "int8-snorm-xfast", sdf_b64: b64 };
}

// ── 1. builder invariants on the synthetic sphere ─────────────────────────
{
  const sh = sphereShape(40, 14);                  // 40^3 → L=2, S=64
  const t = S.buildTree(sh);
  ok("sphere: L=2 for 40^3", t.L === 2, `L=${t.L}`);
  ok("sphere: root exists at node 1", !!t.nodes[1]);
  ok("sphere: root is MIXED", ((t.nodes[1].b >> 16) & 3) === 2);
  ok("sphere: root ALWAYS has a brick (invariant)", t.nodes[1].brick !== S.NO_BRICK);
  // contiguous 8-child blocks + level consistency
  let contigOK = true, levelOK = true;
  for (let i = 1; i < t.nodes.length; i++) {
    const nd = t.nodes[i];
    if (nd.child > 0) {
      for (let o = 0; o < 8; o++) {
        const ch = t.nodes[nd.child + o];
        if (!ch) { contigOK = false; break; }
        if (ch.l !== nd.l + 1) levelOK = false;
      }
    }
  }
  ok("sphere: contiguous 8-child blocks", contigOK);
  ok("sphere: child level = parent+1", levelOK);
  // uniform collapse really fires: EMPTY here; SOLID needs a bigger sphere
  // (a r=14-cell sphere in 40^3 leaves no brick-aligned 16^3 block fully
  // interior — measured, not assumed) — asserted on the 64^3 sphere below
  ok("sphere: EMPTY collapse fired", t.stats.empty > 0, JSON.stringify(t.stats));
  // L=3 case: r=34 in 96^3 (S=128) has provably-aligned interior 16-blocks
  // (aligned block far-corner √3·15.5 + 6 < 34) and saturated
  // padding blocks — both collapse classes must fire, and sparsity must beat
  // the padded-dense lattice by a wide margin at this size
  const tBig = S.buildTree(sphereShape(96, 34));
  ok("sphere96: SOLID collapse fired", tBig.stats.solid > 0, JSON.stringify(tBig.stats));
  ok("sphere96: EMPTY collapse fired", tBig.stats.empty > 0);
  ok("sphere96: L=3", tBig.L === 3, `L=${tBig.L}`);
  // HONEST ECONOMICS (blueprint V1 disclosure): at small tiers sparsity buys
  // ~nothing vs the raw dense grid — the assertion is against the PADDED
  // root-cube lattice the tree actually replaces; the dense ratio is
  // REPORTED, never asserted favorable at this size.
  const padded = tBig.S * tBig.S * tBig.S;
  ok("sphere96: brick payload < padded root-cube lattice",
     tBig.stats.brick_bytes < padded, `${tBig.stats.brick_bytes} vs ${padded}`);
  console.log(`  info: sphere96 brick payload ${(tBig.stats.brick_bytes / 1024).toFixed(0)} KB vs `
    + `dense ${(tBig.stats.dense_bytes / 1024).toFixed(0)} KB (ratio `
    + `${(tBig.stats.brick_bytes / tBig.stats.dense_bytes).toFixed(2)}x — small-tier honesty, blueprint V1)`);
  ok("pad=0 on an exact 16-multiple (32^3)", JSON.stringify(S.buildTree(sphereShape(32, 10)).pad) === "[0,0,0]");
  // every MIXED node has a brick; uniform nodes never do
  let brickRule = true;
  for (let i = 1; i < t.nodes.length; i++) {
    const nd = t.nodes[i], cls = (nd.b >> 16) & 3;
    if (cls === 2 && nd.brick === S.NO_BRICK) brickRule = false;
    if (cls !== 2 && nd.brick !== S.NO_BRICK) brickRule = false;
  }
  ok("sphere: brick ⇔ MIXED rule", brickRule);
}

// ── 2. finest-level bricks preserve the shipped bytes verbatim ────────────
{
  const sh = sphereShape(40, 14);
  const g = S.b64ToI8(sh.sdf_b64);
  const t = S.buildTree(sh);
  const n = 40, pad = t.pad;
  let byteOK = true, tested = 0;
  for (let i = 1; i < t.nodes.length && byteOK; i++) {
    const nd = t.nodes[i];
    if (nd.l !== t.L || nd.brick === S.NO_BRICK) continue;
    const bk = t.bricks[nd.brick];
    for (let z = 1; z < 17 && byteOK; z++) for (let y = 1; y < 17 && byteOK; y++) for (let x = 1; x < 17; x++) {
      const fx = nd.p[0] * 16 + x - 1, fy = nd.p[1] * 16 + y - 1, fz = nd.p[2] * 16 + z - 1;
      const gi = fx - pad[0], gj = fy - pad[1], gk = fz - pad[2];
      if (gi < 0 || gj < 0 || gk < 0 || gi >= n || gj >= n || gk >= n) continue;
      tested++;
      const want = g[(gk * n + gj) * n + gi] + 128;
      if (bk[(z * 18 + y) * 18 + x] !== want) { byteOK = false; }
    }
  }
  ok("finest bricks: shipped payload byte-for-byte", byteOK && tested > 1000, `tested=${tested}`);
}

// ── 3. packScene + jsSVODist: cell-center identity + matched-LOD agreement ─
{
  const sh = sphereShape(40, 14);
  const g = S.b64ToI8(sh.sdf_b64);
  const packed = S.packScene([{ leaf: 0, body: 0, shape: sh, material: "iron",
                                pivot: [0, 0, 0], bounds: null }]);
  const B = packed.bodies[0];
  const n = 40, vs = sh.voxel_size;
  ok("pack: rootNode is 1 (first tree)", B.rootNode === 1, `${B.rootNode}`);
  ok("pack: rootSpan = 64·vs", Math.abs(B.rootSpan - 64 * vs) < 1e-12);
  ok("pack: band0 = band·2^L", Math.abs(B.band0 - sh.band * 4) < 1e-12);
  // cell-center identity: SVO trilinear at a fine-cell center == the byte
  const tree = S.buildTree(sh);                    // same content → same pad
  ok("pack: pad = 12 cells each side", JSON.stringify(tree.pad) === "[12,12,12]",
     JSON.stringify(tree.pad));
  // exactness holds where the descent reaches a level-L brick; uniform-
  // collapsed cells take the CORRECTED ancestor fallback (safety bounds)
  const guar = sh.band - 0.87 * vs;                // classification guarantee
  let maxd = 0, tested = 0, fb = 0, fbBad = 0;
  for (let k = 2; k < n - 2; k += 3) for (let j = 2; j < n - 2; j += 3) for (let i = 2; i < n - 2; i += 3) {
    const q = [B.lo[0] + (tree.pad[0] + i + 0.5) * vs,
               B.lo[1] + (tree.pad[1] + j + 0.5) * vs,
               B.lo[2] + (tree.pad[2] + k + 0.5) * vs];
    const want = g[(k * n + j) * n + i] / 127 * sh.band;
    const r = S.jsSVODist(packed, 0, q);
    if (r.level === B.L) { maxd = Math.max(maxd, Math.abs(r.d - want)); tested++; }
    else {
      fb++;                                        // uniform fallback: sign-safe, never a phantom hit
      if (!(r.d >= guar * 0.999 || r.d <= -guar * 0.999)) fbBad++;
    }
  }
  ok(`cell centers (level-L path): SVO == dense byte (max|d| ${maxd.toExponential(2)})`,
     tested > 500 && maxd < 1e-3 * sh.band, `tested=${tested}`);
  ok(`cell centers (fallback path): |d| ≥ guarantee (${fb} cells)`, fbBad === 0, `bad=${fbBad}`);
}

// helper reused below: dense continuous kernel in metres at shape-local point
function denseD(sh, g, q) {
  const n = sh.dims, vs = sh.voxel_size, c = sh.center, sc = sh.band / 127;
  const nx = n[0], ny = n[1], nz = n[2];
  const fi = (q[0] - c[0]) / vs + (nx - 1) * 0.5,
        fj = (q[1] - c[1]) / vs + (ny - 1) * 0.5,
        fk = (q[2] - c[2]) / vs + (nz - 1) * 0.5;
  const ci = Math.min(Math.max(fi, 0), nx - 1), cj = Math.min(Math.max(fj, 0), ny - 1),
        ck = Math.min(Math.max(fk, 0), nz - 1);
  const i0 = Math.floor(ci), j0 = Math.floor(cj), k0 = Math.floor(ck);
  const i1 = Math.min(i0 + 1, nx - 1), j1 = Math.min(j0 + 1, ny - 1), k1 = Math.min(k0 + 1, nz - 1);
  const di = ci - i0, dj = cj - j0, dk = ck - k0;
  const at = (i, j, k) => g[(k * ny + j) * nx + i] * sc;
  const c00 = at(i0, j0, k0) * (1 - di) + at(i1, j0, k0) * di,
        c01 = at(i0, j0, k1) * (1 - di) + at(i1, j0, k1) * di,
        c10 = at(i0, j1, k0) * (1 - di) + at(i1, j1, k0) * di,
        c11 = at(i0, j1, k1) * (1 - di) + at(i1, j1, k1) * di;
  const val = (c00 * (1 - dj) + c10 * dj) * (1 - dk) + (c01 * (1 - dj) + c11 * dj) * dk;
  if (ci === fi && cj === fj && ck === fk) return val;
  const bx = c[0] + (ci - (nx - 1) * 0.5) * vs, by = c[1] + (cj - (ny - 1) * 0.5) * vs,
        bz = c[2] + (ck - (nz - 1) * 0.5) * vs;
  return Math.max(val, 0) + Math.hypot(q[0] - bx, q[1] - by, q[2] - bz);
}

// ── 4. matched-LOD in-grid agreement + brick-seam C0 (deterministic sweep) ─
{
  const sh = sphereShape(40, 14);
  const g = S.b64ToI8(sh.sdf_b64);
  const packed = S.packScene([{ leaf: 0, body: 0, shape: sh, material: "iron",
                                pivot: [0, 0, 0], bounds: null }]);
  const vs = sh.voxel_size;
  // in-grid interior points on an irrational-stride lattice (LCG-free, deterministic)
  const L = packed.bodies[0].L;
  let maxInGrid = 0, nExact = 0, overshootBad = 0, signBad = 0;
  const guar = sh.band - 0.87 * vs;
  for (let s = 0; s < 4000; s++) {
    const q = [((s * 0.6180339887) % 1) * 36 * vs - 18 * vs,
               ((s * 0.7548776662) % 1) * 36 * vs - 18 * vs,
               ((s * 0.5698402910) % 1) * 36 * vs - 18 * vs];
    const dd = denseD(sh, g, q);
    const r = S.jsSVODist(packed, 0, q);
    if (r.level === L) { maxInGrid = Math.max(maxInGrid, Math.abs(r.d - dd)); nExact++; }
    else {
      // corrected fallback: sign-safe AND never overshoots the dense value
      // by more than one fine cell (the derived conservative-step bound)
      if (!(r.d >= guar * 0.99 || r.d <= -guar * 0.99)) signBad++;
      if (r.d > 0 && r.d > dd + vs) overshootBad++;
      if (r.d < 0 && r.d < dd - vs) overshootBad++;
    }
  }
  // identical bytes + identical trilinear ⇒ float-rounding-level agreement
  ok(`matched-LOD in-grid (level-L path, n=${nExact}): max|svo−dense| = ${maxInGrid.toExponential(2)}`,
     nExact > 1000 && maxInGrid < 1e-9 + 1e-6 * sh.band);
  ok("fallback path: |d| ≥ guarantee (no phantom-hit class)", signBad === 0, `bad=${signBad}`);
  ok("fallback path: conservative (≤ dense + 1 fine cell)", overshootBad === 0, `bad=${overshootBad}`);
  // brick-seam C0: cross every 16-cell seam plane on fine lines; border ring
  // duplication must make the interpolant continuous across siblings
  let maxJump = 0;
  const seams = [16, 32, 48];                      // fine-cell seam planes (root lattice)
  for (const sp of seams) {
    for (let s = 0; s < 400; s++) {
      const y = ((s * 0.6180339887) % 1) * 60 + 2, z = ((s * 0.7548776662) % 1) * 60 + 2;
      const xw = (sp - 12) * vs - 20 * vs;         // root-lattice → shape-local (pad 12, center 20)
      const qa = [xw - 1e-7, (y - 12 - 20 + 0.5) * vs, (z - 12 - 20 + 0.5) * vs];
      const qb = [xw + 1e-7, qa[1], qa[2]];
      const da = S.jsSVODist(packed, 0, qa), db = S.jsSVODist(packed, 0, qb);
      if (da.level === db.level)                   // matched level only (seam rule)
        maxJump = Math.max(maxJump, Math.abs(da.d - db.d));
    }
  }
  ok(`brick-seam C0 at matched level: max jump ${maxJump.toExponential(2)}`,
     maxJump < 1e-5 * sh.band);
  // LOD-seam magnitude (coarse cap): documented, not hidden — reported only
  let maxLod = 0;
  for (let s = 0; s < 2000; s++) {
    const q = [((s * 0.6180339887) % 1) * 36 * vs - 18 * vs,
               ((s * 0.7548776662) % 1) * 36 * vs - 18 * vs,
               ((s * 0.5698402910) % 1) * 36 * vs - 18 * vs];
    const fine = S.jsSVODist(packed, 0, q, packed.bodies[0].L).d;
    const coarse = S.jsSVODist(packed, 0, q, 0).d;
    maxLod = Math.max(maxLod, Math.abs(fine - coarse));
  }
  console.log(`  info: root-vs-finest LOD max|Δd| = ${maxLod.toExponential(2)} m `
    + `(root cell ${(vs * 4).toExponential(2)} m) — the documented LOD-seam tolerance class`);
  ok("LOD delta bounded by ~2 root cells", maxLod < 2.5 * vs * Math.pow(2, packed.bodies[0].L));
}

// ── 5. content-hash sharing + palette rules ───────────────────────────────
{
  const sh = sphereShape(24, 8);
  const shB = JSON.parse(JSON.stringify(sh));      // same payload, distinct object
  const packed = S.packScene([
    { leaf: 0, body: 0, shape: sh, material: "iron", pivot: [0, 0, 0], bounds: null },
    { leaf: 1, body: 1, shape: shB, material: "wood", pivot: [0, 0, 0], bounds: null },
  ]);
  ok("dedupe: 2 bodies, 1 tree", packed.stats.trees === 1 && packed.stats.bodies === 2);
  ok("dedupe: shared rootNode", packed.bodies[0].rootNode === packed.bodies[1].rootNode);
  ok("palette: distinct ids on a shared tree",
     packed.bodies[0].matId === 0 && packed.bodies[1].matId === 1);
  let threw = false;
  try {
    const many = [];
    for (let i = 0; i < 257; i++)
      many.push({ leaf: i, body: i, shape: sh, material: "m" + i, pivot: [0, 0, 0], bounds: null });
    S.packScene(many);
  } catch (e) { threw = /palette overflow/.test(String(e)); }
  ok("palette: REFUSES loudly past 256", threw);
}

// ── 6. GLSL chunk hygiene (the ANGLE law, string-checked) ─────────────────
{
  const glsl = S.glslSVO() + S.glslSVOMapTerm();
  ok("glsl: no uniform arrays", !/uniform[^;]*\[/.test(glsl), "uniform array found");
  ok("glsl: no dynamically-indexed local arrays", !/\w+\[[ ]*[a-z]\w*[ ]*\]/.test(
     glsl.replace(/texelFetch[^;]*;/g, "").replace(/ivec2\([^)]*\)/g, "")),
     "local array indexing found");
  ok("glsl: body data rides texelFetch", /texelFetch\(uSVOBody/.test(glsl));
  ok("glsl: nodes ride texelFetch", /texelFetch\(uSVONodes/.test(glsl));
  ok("glsl: break-driven compile-time loop", /for\(int l = 0; l < MAX_SVO_LEVELS/.test(glsl));
  ok("glsl: NO_BRICK sentinel", /0xFFFFFFFFu/.test(glsl));
  ok("glsl: gate buffer honored in mapD term", /uGateBuffer/.test(glsl));
  // implicit-LOD texture() is a GRADIENT op: inside the nested dynamic loops
  // it wedges the D3D compile (measured live on the theater) — textureLod only
  ok("glsl: no implicit-LOD texture() (compile-wedge class)",
     !/[^a-zA-Z]texture\(/.test(glsl), "implicit-LOD sample found");
}

// ── 7. the real shipped payload: vox_chair round-trip ─────────────────────
{
  const p = path.join(__dirname, "data", "vox_chair.json");
  if (fs.existsSync(p)) {
    const o = JSON.parse(fs.readFileSync(p, "utf-8"));
    const sc = o.kind === "trajectory" ? o.scene : o;
    const leaf = sc.csg_leaves.find(l => l.shape && l.shape.type === "Voxel");
    const sh = leaf.shape;
    const g = S.b64ToI8(sh.sdf_b64);
    const packed = S.packScene([{ leaf: 0, body: leaf.body, shape: sh,
      material: leaf.material, pivot: [0, 0, 0], bounds: leaf.bounds || null }]);
    const st = packed.stats;
    const L = packed.bodies[0].L, vs = sh.voxel_size;
    const guar = sh.band - 0.87 * vs;
    let maxd = 0, nExact = 0, bad = 0;
    for (let s = 0; s < 4000; s++) {
      const q = [sh.center[0] + (((s * 0.6180339887) % 1) - 0.5) * sh.dims[0] * vs * 0.94,
                 sh.center[1] + (((s * 0.7548776662) % 1) - 0.5) * sh.dims[1] * vs * 0.94,
                 sh.center[2] + (((s * 0.5698402910) % 1) - 0.5) * sh.dims[2] * vs * 0.94];
      const dd = denseD(sh, g, q);
      const r = S.jsSVODist(packed, 0, q);
      if (r.level === L) { maxd = Math.max(maxd, Math.abs(r.d - dd)); nExact++; }
      else {
        if (!(r.d >= guar * 0.99 || r.d <= -guar * 0.99)) bad++;
        if (r.d > 0 && r.d > dd + vs) bad++;
        if (r.d < 0 && r.d < dd - vs) bad++;
      }
    }
    ok(`vox_chair: matched-LOD (level-L, n=${nExact}) max|svo−dense| ${maxd.toExponential(2)}`,
       nExact > 500 && maxd < 1e-9 + 1e-6 * sh.band);
    ok("vox_chair: fallback sign-safe + conservative", bad === 0, `bad=${bad}`);
    console.log(`  info: vox_chair tree — nodes ${st.nodes}, bricks ${st.bricks} `
      + `(${(st.brick_payload_bytes / 1024).toFixed(1)} KB vs dense ${(st.dense_bytes / 1024).toFixed(1)} KB), atlas ${st.atlas_dim}^3`);
  } else {
    console.log("  info: data/vox_chair.json not found — shipped-payload check skipped");
  }
}

// ── 8. R4.4 calcN level-pin GLSL invariants (Sec 2) ──────────────────────
{
  const glsl = S.glslSVO();
  const term = S.glslSVOMapTerm();
  ok("pin: gSvoPin declared and consumed in the descent",
     /int gSvoPin = -1;/.test(glsl)
     && /if\(gSvoPin >= 0\) cap = min\(cap, gSvoPin\);/.test(glsl));
  ok("pin: svoDist latches level + cell size for the winner latch",
     /gSvoLvl = blvl;/.test(glsl) && /gSvoVs = /.test(glsl));
  ok("pin: mapD term RESETS the winner latch before the body loop",
     /gSvoHit = 0\.0; gSvoHitLvl = -1;\n\s*for\(int k=0;/.test(term));
  ok("pin: mapD term latches winner only on min-win",
     /if\(sk < d\)\{ d = sk; mat = mk; gSvoHit = gSvoVs; gSvoHitLvl = gSvoLvl; \}/.test(term));
  // the ANGLE law still holds after the pin edits: no uniform arrays, no
  // dynamically-indexed local arrays anywhere in the SVO chunk
  ok("pin: still zero uniform arrays in the SVO chunk",
     !/uniform[^;]*\[/.test(glsl) && !/uniform[^;]*\[/.test(term));
}

console.log(`\n${checks - fails}/${checks} checks pass`);
process.exit(fails ? 1 : 0);
