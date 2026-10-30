// MineBench's voxel.exec (github.com/Ammaar-Alam/minebench lib/ai/tools/voxelExec.ts), with MATERIALS instead of Minecraft blocks.
// The model's JavaScript calls block(x,y,z,m), box(x1,y1,z1,x2,y2,z2,m) (filled, inclusive), line(x1,y1,z1,x2,y2,z2,m), rng()
// (xorshift32 from the seed), Math, GRID_SIZE. Coordinates are integers, y up, as in Minecraft. `m` is any material name: this runtime
// does not judge it; deckard/voxel_exec.py resolves every name through sigma-ground and refuses the ones it cannot ground. "air"
// clears a cell (carving), since air is what every unset cell already is.
//
// Run by voxel_exec.py as `node --permission -e <this file>` (no file, network or child-process access for the model's code), with
// {code, gridSize, seed} on stdin. Prints {names, cells: [x,y,z,nameIndex, ...], placed, outside, ms} on stdout.
"use strict";
let src = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (d) => { src += d; });
process.stdin.on("end", () => {
  const input = JSON.parse(src);
  const G = input.gridSize || 256;
  const t0 = Date.now();
  let s = (input.seed | 0) || 1;
  const rng = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 0x1_0000_0000; };
  const T = (v) => { if (typeof v !== "number" || !Number.isFinite(v)) throw new Error("Non-finite numeric argument"); return Math.trunc(v); };
  const names = [], nameIndex = new Map();
  const voxels = new Map();            // last write wins, like a world
  let placed = 0, outside = 0;
  const put = (a, b, c, m) => {
    placed++;
    if (a < 0 || b < 0 || c < 0 || a >= G || b >= G || c >= G) { outside++; return; }
    const key = (a * G + b) * G + c;
    m = String(m).trim().toLowerCase();
    if (m === "air") { voxels.delete(key); return; }
    let i = nameIndex.get(m);
    if (i === undefined) { i = names.length; names.push(m); nameIndex.set(m, i); }
    voxels.set(key, i);
  };
  const block = (a, b, c, m) => put(T(a), T(b), T(c), m);
  const box = (a1, b1, c1, a2, b2, c2, m) => {
    [a1, b1, c1, a2, b2, c2] = [a1, b1, c1, a2, b2, c2].map(T);
    for (let a = Math.min(a1, a2); a <= Math.max(a1, a2); a++)
      for (let b = Math.min(b1, b2); b <= Math.max(b1, b2); b++)
        for (let c = Math.min(c1, c2); c <= Math.max(c1, c2); c++) put(a, b, c, m);
  };
  const line = (...args) => {
    let p, q, m;
    if (args.length === 3) { p = [args[0].x, args[0].y, args[0].z]; q = [args[1].x, args[1].y, args[1].z]; m = args[2]; }
    else { p = args.slice(0, 3); q = args.slice(3, 6); m = args[6]; }
    p = p.map(T); q = q.map(T);
    const n = Math.max(...p.map((v, k) => Math.abs(q[k] - v)), 1);
    for (let k = 0; k <= n; k++) put(...p.map((v, j) => Math.round(v + (q[j] - v) * k / n)), m);
  };
  new Function("block", "box", "line", "rng", "Math", "GRID_SIZE", `"use strict";\n${input.code}`)(block, box, line, rng, Math, G);

  const cells = new Array(voxels.size * 4);
  let j = 0;
  for (const [key, i] of voxels) {
    cells[j++] = Math.floor(key / G / G); cells[j++] = Math.floor(key / G) % G; cells[j++] = key % G; cells[j++] = i;
  }
  process.stdout.write(JSON.stringify({ names, cells, placed, outside, ms: Date.now() - t0 }));
});
