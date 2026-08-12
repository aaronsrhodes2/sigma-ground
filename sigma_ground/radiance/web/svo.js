/* svo.js — R4 octree-of-bricks (SVO) builder + traversal mirror (PURE logic).
 *
 * Blueprint verdicts implemented here (misc R4 panel, all settled):
 *   V1  octree-of-16^3-bricks: 18^3 stored CELL-CENTERED bricks (1-cell
 *       border ring per face) in ONE R8 UNORM LINEAR 3-D atlas — the proven
 *       makeVoxTex parameters; 128-bit nodes in an RGBA32UI 2-D pool,
 *       texelFetch only, stackless restart point descent.
 *   V2  PER-BODY trees anchored in body frames; interplanetary vacuum is
 *       R3's job (splats/impostors), never traversed here.
 *   V4  16^3 usable cells stored 18^3 cell-centered: preserves the shipped
 *       int8-snorm payloads byte-for-byte at the finest level and the exact
 *       jsVoxelSDF/Python twin math.
 *   V5  per-LEVEL band: band_l = band_leaf · 2^(L−l); one shader multiply,
 *       byte-identical to _VOXEL_BAND_CELLS=6.0 at the finest level.
 *   V6  ONE generic svoDist() + an RGBA32F body-table texture; ZERO uniform
 *       arrays, ZERO dynamically-indexed local arrays (the ANGLE law) —
 *       every per-frame per-body datum rides texelFetch.
 *   V8  JS-first builder: trees are built AT LOAD TIME from the EXISTING
 *       sdf_b64 payloads — zero scene-format changes for R4.1–R4.5.
 *
 * No GL, no DOM, no fetch, no RNG, no wall clock: everything below is a pure
 * function of the decoded grid bytes (node-testable, splats.js precedent).
 * viewer.js consumes buildSceneSVO() + glslSVO(); test_svo.node.js and
 * tools/verify_svo_parity.py consume the same functions headlessly.
 *
 * COORDINATES (one convention, stated once):
 *   grid lo  = shape.center − dims·vs/2  (leaf-local; cell (i,j,k) center at
 *              lo + (idx+0.5)·vs — kernel.voxel.Voxel's own lattice).
 *   root cube: side S = 16·2^L cells, L = ceil(log2(ceil(maxdim/16))); the
 *              grid is centered in the cube on WHOLE-CELL offsets (pad), so
 *              fine cells stay on the shipped lattice byte-for-byte.
 *   svoDist operates PIVOT-FREE: q = qrotInv(quat, p − bodyPos); the caller
 *              bakes lo_table = (grid lo − pad·vs) − pivot into body row t2.
 *
 * PADDING CELLS (root-lattice cells outside the grid box) are evaluated with
 * the SHIPPED dense outside-box kernel at the cell center — max(boundary,0)
 * + distance-to-box — then quantized; the finest level is therefore a
 * straight lattice extension of the shipped payload, never invented data.
 * Root-cube EDGE border texels use the SAME kernel continuation (adversarial
 * R4 refutation finding: a flush-face grid — pad 0 with matter inside the
 * band of a face — made the old "+127 beyond the cube" assumption claim
 * empty space over real matter, overshooting dense by up to 11 fine cells).
 */
"use strict";
(() => {

const NO_BRICK = 0xFFFFFFFF;
const MAX_SVO_LEVELS = 12;          // compile-time descent bound (spans 16·2^12 cells)
const SVO_MAT_BASE = 256;           // mat ids ≥ this are SVO: 256 + (body<<8) + palette_id
const MAX_SVO_BODIES = 512;         // mapD loop compile-time bound (uniform-int broken)
const BRICK = 16, STORED = 18;      // usable cells / stored cells per brick edge
const BRICK_BYTES = STORED * STORED * STORED;   // 5,832

// byte → signed (identical to viewer.js b64ToI8; atob exists in node ≥ 16)
function b64ToI8(b64) {
  const bin = atob(b64), a = new Int8Array(bin.length);
  for (let i = 0; i < bin.length; i++) a[i] = (bin.charCodeAt(i) << 24) >> 24;
  return a;
}
// FNV-1a over the payload string — the content-hash tree dedupe key (V7).
function contentHash(shape) {
  const s = `${shape.dims.join(",")}|${shape.voxel_size}|${shape.band}|${shape.sdf_b64}`;
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  return (h >>> 0).toString(16);
}

// ── the shipped continuous kernel, in FINE-CELL index units ───────────────
// g: Int8Array (x-fastest), query at fractional index (fi,fj,fk) — jsVoxelSDF
// verbatim modulo units: returns i8-units at the FINEST band (metres ·
// 127/band); outside-box distance converts via k = 127/bandCells.
function kernelI8(g, nx, ny, nz, fi, fj, fk, kDist) {
  const ci = Math.min(Math.max(fi, 0), nx - 1),
        cj = Math.min(Math.max(fj, 0), ny - 1),
        ck = Math.min(Math.max(fk, 0), nz - 1);
  const i0 = Math.floor(ci), j0 = Math.floor(cj), k0 = Math.floor(ck);
  const i1 = Math.min(i0 + 1, nx - 1), j1 = Math.min(j0 + 1, ny - 1), k1 = Math.min(k0 + 1, nz - 1);
  const di = ci - i0, dj = cj - j0, dk = ck - k0;
  const at = (i, j, k) => g[(k * ny + j) * nx + i];
  const c00 = at(i0, j0, k0) * (1 - di) + at(i1, j0, k0) * di,
        c01 = at(i0, j0, k1) * (1 - di) + at(i1, j0, k1) * di,
        c10 = at(i0, j1, k0) * (1 - di) + at(i1, j1, k0) * di,
        c11 = at(i0, j1, k1) * (1 - di) + at(i1, j1, k1) * di;
  const val = (c00 * (1 - dj) + c10 * dj) * (1 - dk) + (c01 * (1 - dj) + c11 * dj) * dk;
  if (ci === fi && cj === fj && ck === fk) return val;
  return Math.max(val, 0) + Math.hypot(fi - ci, fj - cj, fk - ck) * kDist;
}

/* Build ONE per-body tree from a Voxel shape dict ({dims, voxel_size, band,
 * sdf_b64 | _grid}). Local node indexing is 1-based (0 reserved); packScene
 * rebases into the global pool. Every MIXED node carries a brick (LOD
 * residency down the whole ladder; the ROOT brick invariant included).
 * Returns { L, S, pad, nodes:[{child,brick,b}], bricks:[Uint8Array(5832)],
 *           stats } — bricks are 18^3 stored blocks, x-fastest.            */
function buildTree(shape) {
  const [nx, ny, nz] = shape.dims;
  const vs = shape.voxel_size, band = shape.band;
  const g = shape._grid || b64ToI8(shape.sdf_b64);
  const bandCells = band / vs;                    // 6.0 for every shipped export
  const kDist = 127 / bandCells;
  const maxdim = Math.max(nx, ny, nz);
  const L = Math.max(0, Math.ceil(Math.log2(Math.max(1, Math.ceil(maxdim / BRICK)))));
  const S = BRICK * (1 << L);                     // root cube side, fine cells
  const pad = [(S - nx) >> 1, (S - ny) >> 1, (S - nz) >> 1];   // whole-cell centering

  // padded fine lattice, i8 (S^3): in-grid bytes verbatim; padding via kernel
  const fine = new Int8Array(S * S * S);
  for (let z = 0; z < S; z++) for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const gi = x - pad[0], gj = y - pad[1], gk = z - pad[2];
    let v;
    if (gi >= 0 && gi < nx && gj >= 0 && gj < ny && gk >= 0 && gk < nz) {
      v = g[(gk * ny + gj) * nx + gi];
    } else {
      v = Math.round(kernelI8(g, nx, ny, nz, gi, gj, gk, kDist));
      v = Math.max(-127, Math.min(127, v));
    }
    fine[(z * S + y) * S + x] = v;
  }
  // beyond-cube lattice points continue via the SAME kernel the padding used
  // (never a hard +127: a flush-face grid carries matter within the band of
  // the cube face, and the dense twin's outside-box formula sees it)
  const fineAt = (x, y, z) => {
    if (x >= 0 && y >= 0 && z >= 0 && x < S && y < S && z < S)
      return fine[(z * S + y) * S + x];
    const v = Math.round(kernelI8(g, nx, ny, nz, x - pad[0], y - pad[1], z - pad[2], kDist));
    return Math.max(-127, Math.min(127, v));
  };

  // min/max brick pyramid for classification: level L bricks are 16^3 fine
  // blocks; each coarser level is a 2×2×2 reduction. nb[l] = bricks across.
  const mm = [];                                  // mm[l] = {min,max Int16Array}
  {
    const nbL = 1 << L;
    let mn = new Int16Array(nbL * nbL * nbL).fill(32767),
        mx = new Int16Array(nbL * nbL * nbL).fill(-32768);
    for (let z = 0; z < S; z++) for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
      const v = fine[(z * S + y) * S + x];
      const bi = ((z >> 4) * nbL + (y >> 4)) * nbL + (x >> 4);
      if (v < mn[bi]) mn[bi] = v;
      if (v > mx[bi]) mx[bi] = v;
    }
    mm[L] = { mn, mx };
    for (let l = L - 1; l >= 0; l--) {
      const n = 1 << l, N2 = 1 << (l + 1);
      const mn2 = new Int16Array(n * n * n).fill(32767),
            mx2 = new Int16Array(n * n * n).fill(-32768);
      for (let z = 0; z < N2; z++) for (let y = 0; y < N2; y++) for (let x = 0; x < N2; x++) {
        const v0 = mm[l + 1].mn[((z * N2) + y) * N2 + x], v1 = mm[l + 1].mx[((z * N2) + y) * N2 + x];
        const bi = (((z >> 1) * n) + (y >> 1)) * n + (x >> 1);
        if (v0 < mn2[bi]) mn2[bi] = v0;
        if (v1 > mx2[bi]) mx2[bi] = v1;
      }
      mm[l] = { mn: mn2, mx: mx2 };
    }
  }
  // class of node (l, px,py,pz): px in [0, 2^l)
  function classOf(l, px, py, pz) {
    const n = 1 << l;
    const mn = mm[l].mn[(pz * n + py) * n + px], mx = mm[l].mx[(pz * n + py) * n + px];
    if (mn === 127) return 0;                     // uniform-EMPTY (all saturated +)
    if (mx === -127) return 1;                    // uniform-SOLID
    return 2;                                     // MIXED
  }

  // level-l stored-cell value (cx..) in the node-INDEPENDENT level lattice
  // (16·2^l cells across the cube), i8 at band_l. l==L: fine bytes verbatim.
  function cellVal(l, cx, cy, cz) {
    // out-of-cube coords fall through to the kernel continuation below (fine
    // level: fineAt's own continuation) — no hard +127 (flush-face honesty)
    if (l === L) return fineAt(cx, cy, cz);
    const sc = 1 << (L - l);                      // fine cells per level-l cell
    // level-l cell center in fine-index coords of the GRID:
    const fi = (cx + 0.5) * sc - pad[0] - 0.5,
          fj = (cy + 0.5) * sc - pad[1] - 0.5,
          fk = (cz + 0.5) * sc - pad[2] - 0.5;
    const v = kernelI8(g, nx, ny, nz, fi, fj, fk, kDist) / sc;   // rescale to band_l
    return Math.max(-127, Math.min(127, Math.round(v)));
  }

  // BFS allocation: node 1 = root; MIXED children allocated as contiguous 8.
  const nodes = [null, { child: 0, brick: NO_BRICK, b: (2 << 16), l: 0, p: [0, 0, 0] }];
  const bricks = [];
  const queue = [1];
  let nEmpty = 0, nSolid = 0, nMixed = 1;
  while (queue.length) {
    const idx = queue.shift();
    const nd = nodes[idx];
    // brick for every MIXED node (root invariant included): pack 18^3 stored
    const bk = new Uint8Array(BRICK_BYTES);
    const ox = nd.p[0] * BRICK, oy = nd.p[1] * BRICK, oz = nd.p[2] * BRICK;
    for (let z = 0; z < STORED; z++) for (let y = 0; y < STORED; y++) for (let x = 0; x < STORED; x++)
      bk[(z * STORED + y) * STORED + x] =
        cellVal(nd.l, ox + x - 1, oy + y - 1, oz + z - 1) + 128;
    nd.brick = bricks.length; bricks.push(bk);
    if (nd.l >= L) { nd.child = 0; continue; }    // finest MIXED leaf
    nd.child = nodes.length;
    let mask = 0;
    for (let o = 0; o < 8; o++) {
      const cp = [nd.p[0] * 2 + (o & 1), nd.p[1] * 2 + ((o >> 1) & 1), nd.p[2] * 2 + ((o >> 2) & 1)];
      const cls = classOf(nd.l + 1, cp[0], cp[1], cp[2]);
      nodes.push({ child: 0, brick: NO_BRICK, b: (cls << 16), l: nd.l + 1, p: cp });
      if (cls === 2) { mask |= (1 << o); nMixed++; }
      else if (cls === 0) nEmpty++;
      else nSolid++;
    }
    nd.b |= (mask << 8);
    for (let o = 0; o < 8; o++)
      if ((mask >> o) & 1) queue.push(nd.child + o);
  }
  return {
    L, S, pad, nodes, bricks,
    vs, band,
    stats: { nodes: nodes.length - 1, bricks: bricks.length,
             mixed: nMixed, empty: nEmpty, solid: nSolid,
             node_bytes: (nodes.length - 1) * 16,
             brick_bytes: bricks.length * BRICK_BYTES,
             dense_bytes: nx * ny * nz },
  };
}

/* Deduplicate a list of material-label strings into a stable-order palette
 * (first-seen order, id = insertion index) — the dedup core packScene() has
 * always done for its own SVO-only material set, extracted here (R4.7 Phase
 * 1) so viewer.js's buildPathShader can run the IDENTICAL algorithm over the
 * full legacy-leaf material list (a completely separate call, separate
 * palette — see buildPathShader's svoMatBase/matLutEnabled comments for why
 * the two stay two independent palettes rather than one unified one).
 * Pure: no throw, no GL, no scene coupling — labels may repeat (every leaf's
 * raw material string, duplicates and all) and order is exactly first-seen.
 * Overflow past 256 distinct labels does NOT throw here, unlike packScene's
 * own hard-throw-on-overflow (an SVO scene that can't fit its materials is a
 * genuine builder-contract violation, V6/V8's original rule — packScene
 * keeps that behaviour verbatim, below). This function is general-purpose:
 * an otherwise-valid scene must never fail to COMPILE just because an
 * internal LUT optimization ran out of palette slots, so it caps
 * registration at 256 and reports the rest via `overflow` (deduped, first-
 * seen order too) for the caller to fall back on — buildPathShader's rule is
 * to warn once via console.warn and leave that leaf's material on its
 * unmodified legacy per-leaf codegen chain. */
function buildMaterialPalette(labels){
  const palette=[], palIdx=new Map(), overflowSeen=new Set(), overflow=[];
  for(const label of labels){
    if(palIdx.has(label) || overflowSeen.has(label)) continue;   // already placed either way
    if(palette.length<256){ palIdx.set(label, palette.length); palette.push(label); }
    else{ overflowSeen.add(label); overflow.push(label); }
  }
  return {palette, palIdx, overflow};
}
/* Pack deduped trees + per-body rows into the scene-level GPU payload.
 * eligible: [{leaf, body, shape, material, pivot:[3], bounds}] — the
 * viewer's svo-routed leaves. Returns null when none.
 *   pool      Uint32Array (4 u32 per node; node 0 reserved, all-zero)
 *   atlas     Uint8Array (dim^3, x-fastest 3-D layout)
 *   bodies    [{body, leaf, rootNode, lo:[3], rootSpan, band0, L, matId,
 *               gateCenterOffset? — folded into gateRadius, gateRadius}]
 *   palette   [material labels] (global 8-bit ids; REFUSES loudly past 256) */
function packScene(eligible) {
  if (!eligible || !eligible.length) return null;
  if (eligible.length > MAX_SVO_BODIES)
    throw new Error(`svo: ${eligible.length} SVO bodies exceed MAX_SVO_BODIES=`
      + `${MAX_SVO_BODIES} — REFUSING (mapD loop + body-table height bound)`);
  const byHash = new Map();
  // R4.7: dedup core delegated to buildMaterialPalette (same first-seen-
  // order algorithm this loop always ran inline) — packScene keeps its OWN
  // hard-throw-on-overflow (a real builder-contract violation for the SVO
  // path, V6/V8's original rule), unlike buildMaterialPalette's general-
  // purpose caller-decides overflow contract.
  const { palette, palIdx, overflow } = buildMaterialPalette(eligible.map(e => e.material));
  if (overflow.length)
    throw new Error(`svo: material palette overflow (>256 labels) — REFUSING (builder rule, V6/V8)`);
  const bodies = [];
  let totNodes = 0, totBricks = 0;
  for (const e of eligible) {
    const h = contentHash(e.shape);
    if (!byHash.has(h)) {
      const t = buildTree(e.shape);
      byHash.set(h, t);
      totNodes += t.nodes.length - 1;
      totBricks += t.bricks.length;
    }
  }
  // global layout: node 0 reserved; per-tree bases in insertion order
  const pool = new Uint32Array((totNodes + 1) * 4);
  const bpa = Math.max(1, Math.ceil(Math.cbrt(totBricks)));
  const dim = STORED * bpa;
  const atlas = new Uint8Array(dim * dim * dim);
  let nodeBase = 1, brickBase = 0;
  for (const [h, t] of byHash) {
    t._rootGlobal = nodeBase;
    for (let i = 1; i < t.nodes.length; i++) {
      const nd = t.nodes[i], gi = nodeBase + (i - 1);
      pool[gi * 4] = nd.child > 0 ? (nodeBase + nd.child - 1) >>> 0 : 0;
      pool[gi * 4 + 1] = nd.brick === NO_BRICK ? NO_BRICK : (brickBase + nd.brick) >>> 0;
      pool[gi * 4 + 2] = nd.b >>> 0;
      pool[gi * 4 + 3] = 0;
    }
    for (let s = 0; s < t.bricks.length; s++) {
      const gs = brickBase + s;
      const ox = STORED * (gs % bpa),
            oy = STORED * (Math.floor(gs / bpa) % bpa),
            oz = STORED * Math.floor(gs / (bpa * bpa));
      const bk = t.bricks[s];
      for (let z = 0; z < STORED; z++) for (let y = 0; y < STORED; y++) {
        const src = (z * STORED + y) * STORED;
        const dst = ((oz + z) * dim + (oy + y)) * dim + ox;
        atlas.set(bk.subarray(src, src + STORED), dst);
      }
    }
    nodeBase += t.nodes.length - 1;
    brickBase += t.bricks.length;
  }
  for (const e of eligible) {
    const t = byHash.get(contentHash(e.shape));
    const s = e.shape, c = s.center, piv = e.pivot || [0, 0, 0];
    const [nx, ny, nz] = s.dims, vs = s.voxel_size;
    // grid lo (leaf-local) minus root padding, minus pivot (svoDist is pivot-free)
    const lo = [
      c[0] - nx * vs * 0.5 - t.pad[0] * vs - piv[0],
      c[1] - ny * vs * 0.5 - t.pad[1] * vs - piv[1],
      c[2] - nz * vs * 0.5 - t.pad[2] * vs - piv[2],
    ];
    // matId always defined: palIdx was built above from this SAME
    // eligible.map(e=>e.material) list, and the overflow throw already
    // returned before this loop if any label didn't fit.
    const matId = palIdx.get(e.material);
    // stamp the palette id into every node of this tree's b bits 0-7 is per
    // NODE, but trees are SHARED across bodies with possibly different
    // materials (yard instancing). The node-resident id therefore stays 0
    // and the BODY-level id is carried in the body row (t3 is full — packed
    // into lodCap's integer: t3.y = lodCap + 256*matId). Single-material
    // bodies are the only R4.1–R4.5 case, so per-cell ids wait for R4.6's
    // material bricks; the descent's nb&255 is used only when a per-node id
    // is actually written (R4.6 bake path).
    const bnd = e.bounds || null;
    // fallback bound = |cube center| + half-diagonal (adversarial R4
    // refutation: the old |lo| + half-diagonal form UNDERSTATES for a cube
    // not centered on the body origin — lo is the MIN corner, so e.g. lo=0
    // gives 0.866·span against a true reach of √3·span — a gate that culls
    // real matter). deriveBounds returns null for small bodies, so this
    // branch IS viewer-reachable.
    const span = t.S * vs;
    const gateRadius = bnd
      ? bnd.radius + Math.hypot(bnd.center[0] - piv[0], bnd.center[1] - piv[1], bnd.center[2] - piv[2])
      : 0.5 * Math.sqrt(3) * span
        + Math.hypot(lo[0] + 0.5 * span, lo[1] + 0.5 * span, lo[2] + 0.5 * span);
    bodies.push({
      body: e.body, leaf: e.leaf, rootNode: t._rootGlobal,
      lo, rootSpan: t.S * vs, band0: s.band * Math.pow(2, t.L),
      L: t.L, matId, gateRadius,
    });
  }
  const stats = {
    trees: byHash.size, bodies: bodies.length,
    nodes: totNodes, bricks: totBricks,
    node_bytes: (totNodes + 1) * 16, atlas_dim: dim, bpa,
    atlas_bytes: atlas.length, brick_payload_bytes: totBricks * BRICK_BYTES,
    dense_bytes: [...byHash.values()].reduce((a, t) => a + t.stats.dense_bytes, 0),
    per_tree: [...byHash.entries()].map(([h, t]) => ({ hash: h, ...t.stats })),
  };
  return { pool, atlas, atlasDim: dim, bpa, bodies, palette, stats };
}

// ── the GLSL chunk (ONE source — viewer.js emits this text verbatim) ──────
// Everything texelFetch-driven; the descent loop is the R4.0-probed pattern
// (uniform-int-bounded, break-driven, computed texel coords). gSvoVs latches
// the deepest-brick cell size for calcN's level pin (R4.4, Sec 2).
function glslSVO(matBase) {
  const base = matBase!==undefined ? matBase : SVO_MAT_BASE;
  return `
uniform highp usampler2D uSVONodes;   // 128-bit nodes: R=child G=brick B=meta A=rsvd
uniform highp sampler3D uSdfAtlas;    // ONE R8 UNORM LINEAR atlas of 18^3 bricks
uniform highp sampler2D uMatLUT;      // 256x4 RGBA32F material properties
uniform highp sampler2D uSVOBody;     // 4 texels per body row (Sec 1 layout)
uniform int uNumSvoBodies;
uniform float uAtlasDim;
uniform int uAtlasBpa;
float gSvoVs = 0.0;                   // deepest-brick cell size at the last SVO sample
int gSvoLvl = 0;                      // level of that sample (R4.4 calcN pin)
float gSvoHit = 0.0;                  // WINNING body's cell size at the last mapD (0 = non-SVO winner)
int gSvoHitLvl = -1;                  // its level (-1 = non-SVO winner)
int gSvoPin = -1;                     // >=0: cap every descent at this level (calcN's 4 taps —
                                      // Sec 2's level pin: kills cross-LOD normal seams)
#define NO_BRICK 0xFFFFFFFFu
#define MAX_SVO_LEVELS ${MAX_SVO_LEVELS}
#define SVO_MAT_BASE ${base}
#define MAX_SVO_BODIES ${MAX_SVO_BODIES}
uvec4 nodeAt(uint i){ return texelFetch(uSVONodes, ivec2(int(i & 4095u), int(i >> 12)), 0); }
vec4 BT(int k, int t){ return texelFetch(uSVOBody, ivec2(t, k), 0); }
vec3 brickOrigin(uint s){ int b=uAtlasBpa; int si=int(s);
  return vec3(18*(si%b), 18*((si/b)%b), 18*(si/(b*b))); }
float sampleBrick(uint brick, vec3 u, int lvl, float band0){
  float side = exp2(float(lvl));
  vec3 f = fract(u * side);                        // position inside this node [0,1)
  vec3 tc = (brickOrigin(brick) + 1.0 + f*16.0) / uAtlasDim;   // Sec 1 texel math
  // textureLod (NOT texture): the atlas is single-mip so the value is
  // identical, but implicit-LOD sampling is a GRADIENT op — inside the
  // nested dynamic loops (march -> body loop -> descent) it forces the
  // D3D compiler toward full unroll and the compile wedges (measured live
  // on the theater, this session — the blueprint's risk (1) bisect).
  return (textureLod(uSdfAtlas, tc, 0.0).r*255.0 - 128.0)/127.0 * (band0 * exp2(-float(lvl)));
}
float svoDist(int k, vec3 p, out int matOut){
  vec4 t0=BT(k,0), t1=BT(k,1), t2=BT(k,2), t3=BT(k,3);
  vec3 q  = qrotInv(t1, p - t0.xyz);               // the bdy{k} rebase, verbatim
  vec3 u  = (q - t2.xyz) / t2.w;
  vec3 cl = clamp(u, 0.0, 0.99999994);             // 1-ulp shy of 1.0: an exact
                                                   // face hit must not descend
                                                   // into the wrong octant
  uint node = uint(t3.x); uint brick = NO_BRICK; int blvl = 0; uint nb = 0u;
  int cap = int(t3.y) & 255;                  // t3.y packs lodCap | matId<<8 | L<<16
  if(gSvoPin >= 0) cap = min(cap, gSvoPin);   // calcN tap pin (R4.4, Sec 2)
  int Lf  = (int(t3.y) >> 16) & 255;          // FINEST level (guarantee lives there)
  uint lastCls = 2u;
  for(int l = 0; l < MAX_SVO_LEVELS; l++){         // break-driven — the R4.0-
    uvec4 nd = nodeAt(node);                       // probed ANGLE-safe pattern
    if(nd.g != NO_BRICK){ brick = nd.g; blvl = l; nb = nd.b; }
    lastCls = (nd.b >> 16) & 3u;
    if(lastCls != 2u || l >= cap || nd.r == 0u) break;   // uniform / LOD cap / leaf
    ivec3 c = ivec3(floor(cl * exp2(float(l+1)))) & 1;
    node = nd.r + uint(c.x | (c.y << 1) | (c.z << 2));
  }
  float d = sampleBrick(brick, cl, blvl, t3.w);    // deepest RESIDENT brick —
                                                   // always exists (root invariant)
  gSvoVs = t2.w / (16.0 * exp2(float(blvl)));      // cell size at the sampled level
  gSvoLvl = blvl;
  if(lastCls != 2u){
    // uniform-node fallback: the ancestor brick's point-sampled cells can
    // exceed true distance by ~one ancestor cell (trilinear-of-centers
    // bound). Subtract that cell, floored at the CLASSIFICATION guarantee
    // (all fine cells saturated ⇒ |true| ≥ band_fine − (√3/2)·vs_fine) —
    // sphere-trace-safe at ANY tree depth, still skips empty space in
    // ancestor-cell-sized strides. Uniform-SOLID is the mirror image.
    // The guarantee is a FINEST-level (L) fact: classification reduces fine
    // cells, so the floor must use L, NEVER the lodCap (adversarial R4
    // refutation: the cap-scaled floor overshot dense by 1.5-1.8 cells-at-cap
    // at cap=L-2 on both shipped voxel scene classes — tunneling class).
    float vsf = t2.w / (16.0 * exp2(float(Lf)));
    float guar = t3.w * exp2(-float(Lf)) - 0.87 * vsf;
    d = (lastCls == 1u) ? min(d + gSvoVs, -guar) : max(d - gSvoVs, guar);
  }
  matOut = SVO_MAT_BASE + (k << 8) + ((int(t3.y) >> 8) & 255);
  if(cl != u){                                     // outside root box: glslVoxel's
    vec3 e = max(max(t2.xyz - q, q - (t2.xyz + vec3(t2.w))), vec3(0.0));
    return max(d, 0.0) + length(e);                // exact outside-box kernel
  }
  return d;
}
`;
}
// the mapD() SVO term — appended after the legacy compose/matsel chains.
// gSvoHit/gSvoHitLvl latch the WINNER of this mapD call (reset first, so a
// non-SVO winner reads 0.0/-1): march()'s final mapOnly at the accepted hit
// point leaves them describing the hit's level, which the svoActive calcN
// variant (viewer.js) consumes for h = max(uCalcNH, 0.5·vs_hit) + the tap
// pin — the R4.4 faceting mechanism (Sec 2/Sec 7).
function glslSVOMapTerm() {
  return `  gSvoHit = 0.0; gSvoHitLvl = -1;
  for(int k=0;k<MAX_SVO_BODIES;k++){ if(k>=uNumSvoBodies) break;
    vec4 gt0=BT(k,0);
    float gbd = length(p - gt0.xyz) - gt0.w;       // SAME bounding-sphere gate as bd{i}
    if(gbd > uGateBuffer){ d = min(d, gbd); continue; }   // the GATE_BUFFER lesson
    int mk; float sk = svoDist(k, p, mk);
    if(sk < d){ d = sk; mat = mk; gSvoHit = gSvoVs; gSvoHitLvl = gSvoLvl; } }
`;
}

// ── JS mirror of the GLSL descent (float32-emulating at the branch points) ─
// packed = packScene() output; row = index into packed.bodies; q = body-local
// PIVOT-FREE point (the caller applies qrotInv(quat, p - pos), as svoDist
// does). Returns {d, level, node, brick, outside} — d in metres.
const f32 = Math.fround;
function jsSVODist(packed, row, q, lodCap) {
  const B = packed.bodies[row];
  const u = [f32((q[0] - B.lo[0]) / B.rootSpan),
             f32((q[1] - B.lo[1]) / B.rootSpan),
             f32((q[2] - B.lo[2]) / B.rootSpan)];
  const CL = 0.99999994;
  const cl = u.map(v => Math.min(Math.max(v, 0), CL));
  let node = B.rootNode, brick = NO_BRICK, blvl = 0, lastCls = 2;
  const cap = lodCap === undefined ? B.L : lodCap;
  for (let l = 0; l < MAX_SVO_LEVELS; l++) {
    const child = packed.pool[node * 4], bptr = packed.pool[node * 4 + 1], b = packed.pool[node * 4 + 2];
    if (bptr !== NO_BRICK) { brick = bptr; blvl = l; }
    lastCls = (b >>> 16) & 3;
    if (lastCls !== 2 || l >= cap || child === 0) break;
    const s = f32(Math.pow(2, l + 1));
    const cx = Math.floor(f32(cl[0] * s)) & 1,
          cy = Math.floor(f32(cl[1] * s)) & 1,
          cz = Math.floor(f32(cl[2] * s)) & 1;
    node = child + (cx | (cy << 1) | (cz << 2));
  }
  // sampleBrick mirror (float64 interpolation; hardware-filter tolerance is
  // the R4.0 probe's measured 2-LSB bound, documented for the parity tool)
  const side = Math.pow(2, blvl);
  const dim = packed.atlasDim, bpa = packed.bpa;
  const ox = STORED * (brick % bpa),
        oy = STORED * (Math.floor(brick / bpa) % bpa),
        oz = STORED * Math.floor(brick / (bpa * bpa));
  const fx = f32(cl[0] * side) % 1, fy = f32(cl[1] * side) % 1, fz = f32(cl[2] * side) % 1;
  const sx = 1 + fx * 16, sy = 1 + fy * 16, sz = 1 + fz * 16;
  const ix = Math.floor(sx - 0.5), wx = sx - 0.5 - ix;
  const iy = Math.floor(sy - 0.5), wy = sy - 0.5 - iy;
  const iz = Math.floor(sz - 0.5), wz = sz - 0.5 - iz;
  const at = (x, y, z) => packed.atlas[((oz + z) * dim + (oy + y)) * dim + (ox + x)];
  let v = 0;
  for (let dz = 0; dz < 2; dz++) for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++)
    v += at(ix + dx, iy + dy, iz + dz) *
         (dx ? wx : 1 - wx) * (dy ? wy : 1 - wy) * (dz ? wz : 1 - wz);
  const band_l = B.band0 * Math.pow(2, -blvl);
  let d = (v - 128) / 127 * band_l;
  const vsl = B.rootSpan / (16 * Math.pow(2, blvl));
  if (lastCls !== 2) {                             // uniform-fallback step correction
    // floor at the FINEST-level classification guarantee (L, never cap —
    // the cap-scaled floor overshot dense at pinned LOD; see glslSVO note)
    const vsf = B.rootSpan / (16 * Math.pow(2, B.L));
    const guar = B.band0 * Math.pow(2, -B.L) - 0.87 * vsf;
    d = (lastCls === 1) ? Math.min(d + vsl, -guar) : Math.max(d - vsl, guar);
  }
  const outside = (cl[0] !== u[0] || cl[1] !== u[1] || cl[2] !== u[2]);
  if (outside) {
    const hi = [B.lo[0] + B.rootSpan, B.lo[1] + B.rootSpan, B.lo[2] + B.rootSpan];
    const e = [Math.max(Math.max(B.lo[0] - q[0], q[0] - hi[0]), 0),
               Math.max(Math.max(B.lo[1] - q[1], q[1] - hi[1]), 0),
               Math.max(Math.max(B.lo[2] - q[2], q[2] - hi[2]), 0)];
    d = Math.max(d, 0) + Math.hypot(e[0], e[1], e[2]);
  }
  return { d, level: blvl, node, brick, outside };
}

const api = { b64ToI8, contentHash, buildTree, packScene, jsSVODist,
              glslSVO, glslSVOMapTerm, kernelI8, buildMaterialPalette,
              NO_BRICK, MAX_SVO_LEVELS, SVO_MAT_BASE, MAX_SVO_BODIES,
              BRICK, STORED, BRICK_BYTES };
if (typeof window !== "undefined") window.SGSVO = api;
if (typeof module !== "undefined" && module.exports) module.exports = api;
})();
