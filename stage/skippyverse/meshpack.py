"""Pack a glTF/GLB model into a small payload a game client can draw: triangles only, simplified, quantised, deflated.

    python stage/skippyverse/meshpack.py <model.gltf|glb> [--target-tris 3000] [--out payload.bin]

The payload is the visual half of swapping a placeholder box for the real thing: the server sends it once per model, the client
builds a runtime mesh. Collision stays the box (or a later hull); nothing here is physics.

Frame of the result: metres, +Z up, origin on the floor under the footprint centre (the card's frame, design section 4); the glTF
y-up axes are turned to z-up the way measure.py does it. The front of a found model is unknown, so it is left as authored.

Wire format (little endian, then zlib-deflated):
    magic 'SVM1' | u16 n_verts | u16 n_tris | f32 min[3] | f32 max[3] | u16 verts[n_verts][3] (quantised across min..max)
    | u16 idx[n_tris][3] | u8 rgb[n_tris][3] (flat colour per triangle: the material's base colour)
Simplification is vertex clustering on a grid sized to hit the target: crude, keeps the silhouette and the size, drops detail.
Textures are not carried; that is the next step, and flat colours are what mark this payload as the untextured version.
"""
from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skippyverse import measure as M  # noqa: E402

MAGIC = b"SVM1"
_IDX = {5121: ("B", 1), 5123: ("H", 2), 5125: ("I", 4)}


def _indices(doc, buffers, idx):
    acc = doc["accessors"][idx]
    fmt, size = _IDX[acc["componentType"]]
    bv = doc["bufferViews"][acc["bufferView"]]
    buf = buffers[bv["buffer"]]
    start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride") or size
    if stride == size:
        return list(struct.unpack_from(f"<{acc['count']}{fmt}", buf, start))
    return [struct.unpack_from("<" + fmt, buf, start + i * stride)[0] for i in range(acc["count"])]


def _base_colour(doc, prim):
    mi = prim.get("material")
    if mi is None:
        return (200, 200, 200)
    pbr = doc.get("materials", [{}])[mi].get("pbrMetallicRoughness", {})
    c = pbr.get("baseColorFactor", [0.8, 0.8, 0.8, 1])
    return tuple(max(0, min(255, int(round(255 * v ** (1 / 2.2))))) for v in c[:3])      # linear -> display


def triangles(path):
    """[(p0, p1, p2, rgb)] in metres, z up, with the node transforms applied (not yet put on the floor)."""
    doc, buffers = M._load_gltf(Path(path))
    nodes = doc.get("nodes", [])
    scenes = doc.get("scenes", [])
    if scenes:
        roots = scenes[doc.get("scene", 0)].get("nodes", [])
    else:
        children = {c for n in nodes for c in n.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in children]
    canon = M._canon("y", 1.0)
    out = []

    def local(n):
        if "matrix" in n:
            c = n["matrix"]
            return [c[0], c[4], c[8], c[12], c[1], c[5], c[9], c[13], c[2], c[6], c[10], c[14], c[3], c[7], c[11], c[15]]
        m = M.mat_translate(n.get("translation", (0, 0, 0)))
        m = M.mat_mul(m, M.mat_quat(n.get("rotation", (0, 0, 0, 1))))
        return M.mat_mul(m, M.mat_scale(n.get("scale", (1, 1, 1))))

    def visit(i, parent, depth=0):
        if depth > 64:
            raise M.MeasureError("hierarchy too deep")
        n = nodes[i]
        world = M.mat_mul(parent, local(n))
        if "mesh" in n:
            full = M.mat_mul(canon, world)
            for prim in doc["meshes"][n["mesh"]].get("primitives", []):
                if prim.get("mode", 4) != 4 or "POSITION" not in prim.get("attributes", {}):
                    continue
                pts, _, _ = M._accessor_points(doc, buffers, prim["attributes"]["POSITION"])
                if pts is None:
                    continue
                pts = [M.mat_apply(full, p) for p in pts]
                ix = _indices(doc, buffers, prim["indices"]) if "indices" in prim else list(range(len(pts)))
                rgb = _base_colour(doc, prim)
                for k in range(0, len(ix) - 2, 3):
                    out.append((pts[ix[k]], pts[ix[k + 1]], pts[ix[k + 2]], rgb))
        for c in n.get("children", []):
            visit(c, world, depth + 1)

    for r in roots:
        visit(r, M.IDENT)
    return out


def on_floor(tris):
    """Shift so the footprint centre is at x = y = 0 and the lowest point at z = 0."""
    if not tris:
        return []
    pts = [p for t in tris for p in t[:3]]
    cx = (min(p[0] for p in pts) + max(p[0] for p in pts)) / 2
    cy = (min(p[1] for p in pts) + max(p[1] for p in pts)) / 2
    z0 = min(p[2] for p in pts)
    return [tuple((p[0] - cx, p[1] - cy, p[2] - z0) for p in t[:3]) + (t[3],) for t in tris]


def simplify(tris, target):
    """Vertex clustering: snap vertices to a grid, drop triangles that collapse, drop duplicates. Returns (verts, faces, colours)."""
    if not tris:
        return [], [], []
    pts = [p for t in tris for p in t[:3]]
    mn = tuple(min(p[i] for p in pts) for i in range(3))
    mx = tuple(max(p[i] for p in pts) for i in range(3))
    ext = max(mx[i] - mn[i] for i in range(3)) or 1.0

    def cluster(cells):
        # cells along the longest axis; a thinner axis gets proportionally fewer but never fewer than 2, so a thin shape is not flattened
        n = [max(2, round(cells * (mx[i] - mn[i]) / ext)) for i in range(3)]
        h = [((mx[i] - mn[i]) / n[i]) or 1.0 for i in range(3)]
        vid, order, faces, cols, seen, acc = {}, [], [], [], set(), {}
        for a, b, c, rgb in tris:
            key = []
            for p in (a, b, c):
                k = tuple(min(n[i] - 1, int((p[i] - mn[i]) / h[i])) for i in range(3))
                if k not in vid:
                    vid[k] = len(order)
                    order.append(k)
                    acc[k] = [0.0, 0.0, 0.0, 0]
                s = acc[k]
                s[0] += p[0]
                s[1] += p[1]
                s[2] += p[2]
                s[3] += 1
                key.append(vid[k])
            if len(set(key)) < 3:
                continue
            canon = tuple(sorted(key))
            if canon in seen:
                continue
            seen.add(canon)
            faces.append(tuple(key))
            cols.append(rgb)
        pos = [(acc[k][0] / acc[k][3], acc[k][1] / acc[k][3], acc[k][2] / acc[k][3]) for k in order]
        return pos, faces, cols

    lo, hi = 4, 512
    best = cluster(lo)
    while lo < hi:                                           # the finest grid that stays at or under the target
        mid = (lo + hi + 1) // 2
        got = cluster(mid)
        if len(got[1]) <= target:
            best, lo = got, mid
        else:
            hi = mid - 1
    cells = 4
    while not best[1] and cells < 4096:                      # a thin shape collapses on a coarse grid: refine until something is left
        cells *= 2
        best = cluster(cells)
    return best


def pack(path, target_tris=3000):
    tris = on_floor(triangles(path))
    pos, faces, cols = simplify(tris, target_tris)
    if not faces or len(pos) > 65535:
        raise M.MeasureError("nothing to pack, or too many vertices for u16")
    mn = [min(p[i] for p in pos) for i in range(3)]
    mx = [max(p[i] for p in pos) for i in range(3)]
    q = [[int(round(65535 * (p[i] - mn[i]) / ((mx[i] - mn[i]) or 1.0))) for i in range(3)] for p in pos]
    body = bytearray()
    body += MAGIC + struct.pack("<HH6f", len(pos), len(faces), *mn, *mx)
    for v in q:
        body += struct.pack("<3H", *v)
    for f in faces:
        body += struct.pack("<3H", *f)
    for c in cols:
        body += struct.pack("<3B", *c)
    blob = zlib.compress(bytes(body), 9)
    info = {"tris_in": len(tris), "tris_out": len(faces), "verts": len(pos),
            "bbox_m": [round(mx[i] - mn[i], 4) for i in range(3)], "raw_bytes": len(body), "bytes": len(blob)}
    return blob, info


def unpack(blob):
    """The inverse, for tests: (verts in metres, faces, colours)."""
    b = zlib.decompress(blob)
    if b[:4] != MAGIC:
        raise M.MeasureError("not an SVM1 payload")
    nv, nt = struct.unpack_from("<HH", b, 4)
    f = struct.unpack_from("<6f", b, 8)
    mn, mx = f[:3], f[3:]
    off = 32
    verts = []
    for i in range(nv):
        q = struct.unpack_from("<3H", b, off + 6 * i)
        verts.append(tuple(mn[k] + (mx[k] - mn[k]) * q[k] / 65535 for k in range(3)))
    off += 6 * nv
    faces = [struct.unpack_from("<3H", b, off + 6 * i) for i in range(nt)]
    off += 6 * nt
    cols = [struct.unpack_from("<3B", b, off + 3 * i) for i in range(nt)]
    return verts, faces, cols


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--target-tris", type=int, default=3000)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    blob, info = pack(a.model, a.target_tris)
    print(info)
    if a.out:
        Path(a.out).write_bytes(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
