"""Pack a glTF/GLB model into a small payload a game client can draw: triangles only, simplified, quantised, deflated.

    python stage/skippyverse/meshpack.py <model.gltf|glb> [--target-tris 3000] [--out payload.bin]

The payload is the visual half of swapping a placeholder box for the real thing: the server sends it once per model, the client
builds a runtime mesh. Collision stays the box (or a later hull); nothing here is physics.

Frame of the result: metres, +Z up, origin on the floor under the footprint centre (the card's frame, design section 4); the glTF
y-up axes are turned to z-up the way measure.py does it. The front of a found model is unknown, so it is left as authored.

Wire format (little endian, then compressed: zlib by default; codec="lzma" for a Garry's Mod client, see lzma_alone below):
    magic 'SVM1' | u16 n_verts | u16 n_tris | f32 min[3] | f32 max[3] | u16 verts[n_verts][3] (quantised across min..max)
    | u16 idx[n_tris][3] | u8 rgb[n_tris][3] (flat colour per triangle: the material's base colour)
Simplification is vertex clustering on a grid sized to hit the target: crude, keeps the silhouette and the size, drops detail.
Textures are not carried; that is the next step, and flat colours are what mark this payload as the untextured version.
"""
from __future__ import annotations

import argparse
import lzma
import math
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


_UNREADABLE_EXT = {"EXT_meshopt_compression": "meshopt"}       # no decoder for these; Draco is decoded when DracoPy is installed (measure.decode_draco_primitive)


def gltf_unreadable_why(path):
    """None when this glTF/GLB's geometry can be read (plain accessors, or Draco with the decoder installed and every Draco buffer decoding);
    else a short reason."""
    doc, buffers = M._load_gltf(Path(path))
    used = set(doc.get("extensionsRequired", [])) | set(doc.get("extensionsUsed", []))
    for ext, name in _UNREADABLE_EXT.items():
        if ext in used and any(ext in (pr.get("extensions") or {}) for m in doc.get("meshes", []) for pr in m.get("primitives", [])):
            return f"its geometry is {name}-compressed, which this packer cannot decode"
    if M.DRACO_EXT in used or any(M.draco_ext(pr) is not None for m in doc.get("meshes", []) for pr in m.get("primitives", [])):
        for m in doc.get("meshes", []):
            for pr in m.get("primitives", []):
                if M.draco_ext(pr) is not None:
                    try:
                        M.decode_draco_primitive(doc, buffers, pr)
                    except M.MeasureError as e:
                        return str(e) if "Draco" in str(e) else f"its Draco geometry cannot be read ({e})"
    return None


def gltf_readable(path) -> bool:
    """Never raises."""
    try:
        return gltf_unreadable_why(path) is None
    except (M.MeasureError, OSError, ValueError, KeyError):
        return False


def triangles(path):
    """[(p0, p1, p2, rgb)] in metres, z up, with the node transforms applied (not yet put on the floor). glTF/GLB, or a Gazebo SDF
    made of box/sphere/cylinder visuals (sdf_triangles)."""
    if Path(path).suffix.lower() == ".sdf":
        return sdf_triangles(path)
    if Path(path).suffix.lower() in (".dae", ".obj"):
        return mesh_file_triangles(path, M.IDENT, (200, 200, 200))
    why = gltf_unreadable_why(path)
    if why:
        raise M.MeasureError(why)
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
                if M.draco_ext(prim) is not None:
                    pts, ix = M.decode_draco_primitive(doc, buffers, prim)
                else:
                    pts, _, _ = M._accessor_points(doc, buffers, prim["attributes"]["POSITION"])
                    if pts is None:
                        continue
                    ix = None
                pts = [M.mat_apply(full, p) for p in pts]
                if ix is None:
                    ix = _indices(doc, buffers, prim["indices"]) if "indices" in prim else list(range(len(pts)))
                rgb = _base_colour(doc, prim)
                for k in range(0, len(ix) - 2, 3):
                    out.append((pts[ix[k]], pts[ix[k + 1]], pts[ix[k + 2]], rgb))
        for c in n.get("children", []):
            visit(c, world, depth + 1)

    for r in roots:
        visit(r, M.IDENT)
    return out


# ----------------------------------------------------------------------------------------------------------------- Collada and OBJ
def obj_triangles(path, post, rgb):
    """OBJ: +Z up, metres (nothing is declared; the same assumption measure_obj makes). Faces are fanned."""
    total = M.mat_mul(post, M._canon("z", None))
    pts, out = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            w = line.split()
            if line.startswith("v ") and len(w) >= 4:
                pts.append(M.mat_apply(total, (float(w[1]), float(w[2]), float(w[3]))))
            elif line.startswith("f ") and len(w) >= 4:
                ix = [int(t.split("/")[0]) for t in w[1:]]
                ix = [i - 1 if i > 0 else len(pts) + i for i in ix]
                for k in range(1, len(ix) - 1):
                    out.append((pts[ix[0]], pts[ix[k]], pts[ix[k + 1]], rgb))
    return out


def dae_triangles(path, post, rgb):
    """Collada <triangles>/<polylist>/<polygons> (positions only; the material is not read), with the file's unit and up axis and the scene's node
    transforms applied, the way measure_dae measures them."""
    root = M.ET.parse(str(path)).getroot()
    unit_m, up = 1.0, "y"
    for el in root.iter():
        t = M._local(el.tag)
        if t == "unit" and el.get("meter"):
            unit_m = float(el.get("meter"))
        elif t == "up_axis" and el.text:
            up = el.text.strip()[0].lower()
    if up not in M.UP_TO_Z:
        raise M.MeasureError(f"unknown up_axis {up!r}")
    total = M.mat_mul(post, M._canon(up, unit_m))
    sources = {}
    for el in root.iter():
        if M._local(el.tag) == "source":
            fa = next((c for c in el.iter() if M._local(c.tag) == "float_array"), None)
            acc = next((c for c in el.iter() if M._local(c.tag) == "accessor"), None)
            if fa is not None:
                sources[el.get("id")] = (M._floats(fa.text), int(acc.get("stride")) if acc is not None and acc.get("stride") else 3)
    geoms = {}
    for g in root.iter():
        if M._local(g.tag) != "geometry":
            continue
        tris = []
        for mesh in (c for c in g if M._local(c.tag) == "mesh"):
            vpos = {}
            for vert in (c for c in mesh if M._local(c.tag) == "vertices"):
                for inp in (c for c in vert if M._local(c.tag) == "input" and c.get("semantic") == "POSITION"):
                    vals, stride = sources.get(inp.get("source", "").lstrip("#"), ([], 3))
                    vpos[vert.get("id")] = [(vals[i], vals[i + 1], vals[i + 2]) for i in range(0, len(vals) - 2, stride)]
            for prim in (c for c in mesh if M._local(c.tag) in ("triangles", "polylist", "polygons")):
                ins = [c for c in prim if M._local(c.tag) == "input"]
                vin = next((c for c in ins if c.get("semantic") == "VERTEX"), None)
                if vin is None or vin.get("source", "").lstrip("#") not in vpos:
                    continue
                pos, off = vpos[vin.get("source").lstrip("#")], int(vin.get("offset", 0))
                stride = max(int(c.get("offset", 0)) for c in ins) + 1
                kind = M._local(prim.tag)
                rings = []
                if kind == "polygons":
                    for p in (q for q in prim if M._local(q.tag) == "p"):
                        v = [int(x) for x in p.text.split()]
                        rings.append([v[i] for i in range(off, len(v), stride)])
                else:
                    flat = [int(x) for x in next(q.text for q in prim if M._local(q.tag) == "p").split()]
                    if kind == "triangles":
                        counts = [3] * (len(flat) // (3 * stride))
                    else:
                        counts = [int(x) for x in next(q.text for q in prim if M._local(q.tag) == "vcount").split()]
                    k = 0
                    for n in counts:
                        rings.append([flat[(k + j) * stride + off] for j in range(n)])
                        k += n
                for r in rings:
                    for j in range(1, len(r) - 1):
                        tris.append((pos[r[0]], pos[r[j]], pos[r[j + 1]]))
        geoms[g.get("id")] = tris

    def node_matrix(node):
        m = M.IDENT
        for c in node:
            t, v = M._local(c.tag), M._floats(c.text)
            if t == "matrix":
                if len(v) != 16:
                    raise M.MeasureError("Collada matrix needs 16 numbers")
                m = M.mat_mul(m, v)
            elif t == "translate":
                m = M.mat_mul(m, M.mat_translate(v[:3]))
            elif t == "rotate":
                m = M.mat_mul(m, M.mat_axis_angle(v[:3], v[3]))
            elif t == "scale":
                m = M.mat_mul(m, M.mat_scale(v[:3]))
            elif t in ("lookat", "skew"):
                raise M.MeasureError(f"Collada <{t}> transforms are not supported")
        return m

    nodes_by_id = {n.get("id"): n for n in root.iter() if M._local(n.tag) == "node" and n.get("id")}
    out, seen = [], [0]

    def emit(gid, world):
        full = M.mat_mul(total, world)
        for a, b, c in geoms.get(gid, []):
            out.append((M.mat_apply(full, a), M.mat_apply(full, b), M.mat_apply(full, c), rgb))

    def visit(node, parent, depth=0):
        if depth > 64:
            raise M.MeasureError("Collada node hierarchy too deep or cyclic")
        world = M.mat_mul(parent, node_matrix(node))
        for c in node:
            t = M._local(c.tag)
            if t == "instance_geometry":
                seen[0] += 1
                emit(c.get("url", "").lstrip("#"), world)
            elif t == "instance_node":
                tgt = nodes_by_id.get(c.get("url", "").lstrip("#"))
                if tgt is not None:
                    visit(tgt, world, depth + 1)
            elif t == "node":
                visit(c, world, depth + 1)
    for vs in root.iter():
        if M._local(vs.tag) == "visual_scene":
            for n in (c for c in vs if M._local(c.tag) == "node"):
                visit(n, M.IDENT)
    if not seen[0]:
        for gid in geoms:
            emit(gid, M.IDENT)
    return out


def mesh_file_triangles(path, post, rgb):
    ext = Path(path).suffix.lower()
    if ext == ".dae":
        return dae_triangles(path, post, rgb)
    if ext == ".obj":
        return obj_triangles(path, post, rgb)
    raise M.MeasureError(f"{ext} meshes are not read")


# ----------------------------------------------------------------------------------------------------------------- Gazebo SDF primitives
# What an SDF draws: its own box/sphere/cylinder and <mesh> visuals that are Collada or OBJ (colours: the visual's diffuse/ambient, else grey;
# textures and material scripts are not read). Anything else (STL, glTF inside an SDF) is NOT read: such a model stays a box in the game
# (sdf_drawable says which is which, before anything is promised).
_BOX_FACES = (((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
              ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),
              ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)))        # counter-clockwise from outside
SDF_SEGMENTS = 16


def _sdf_visuals(path):
    """-> [(world matrix, geometry element, rgb)] of every <visual> of an SDF model (collisions are not drawn)."""
    root = M.ET.parse(str(path)).getroot()
    model = next((c for c in root.iter() if M._local(c.tag) == "model"), None)
    if model is None:
        raise M.MeasureError("no <model> in the SDF")
    for c in model.iter():
        if M._local(c.tag) == "include" or (M._local(c.tag) == "model" and c is not model):
            raise M.MeasureError("nested models and <include> are not supported")
    out = []
    model_m = M._sdf_pose(model)
    for link in (c for c in model if M._local(c.tag) == "link"):
        link_m = M.mat_mul(model_m, M._sdf_pose(link))
        for vis in (c for c in link if M._local(c.tag) == "visual"):
            geo = next((c for c in vis if M._local(c.tag) == "geometry"), None)
            kind = next(iter(geo), None) if geo is not None else None
            if kind is None:
                continue
            rgb = (200, 200, 200)
            mat = next((c for c in vis if M._local(c.tag) == "material"), None)
            if mat is not None:
                for tag in ("diffuse", "ambient"):
                    el = next((c for c in mat if M._local(c.tag) == tag), None)
                    v = M._floats(el.text) if el is not None else []
                    if len(v) >= 3:
                        rgb = tuple(max(0, min(255, int(round(255 * x)))) for x in v[:3])
                        break
            out.append((M.mat_mul(link_m, M._sdf_pose(vis)), kind, rgb))
    return out


def sdf_drawable(path) -> bool:
    """True when every visual of the SDF can be drawn by sdf_triangles: box/sphere/cylinder, or a mesh visual whose .dae/.obj file is in
    the model. Never raises (a model that is not drawable stays a box in the game)."""
    try:
        path = Path(path)
        vis = _sdf_visuals(path)
        if not vis:
            return False
        for _, kind, _ in vis:
            t = M._local(kind.tag)
            if t == "mesh":
                uri = next((c.text for c in kind if M._local(c.tag) == "uri"), None)
                if not uri or M._find_mesh_file(uri, path.parent, path.parent).suffix.lower() not in (".dae", ".obj"):
                    return False
            elif t not in ("box", "sphere", "cylinder"):
                return False
        return True
    except (M.MeasureError, OSError, ValueError, M.ET.ParseError, StopIteration):
        return False


def _quads(tris, rgb, a, b, c, d):
    tris.append((a, b, c, rgb))
    tris.append((a, c, d, rgb))


def sdf_triangles(path):
    """[(p0, p1, p2, rgb)] in metres, z up (the SDF's own frame), for box/sphere/cylinder visuals; anything else raises MeasureError."""
    out = []
    for m, kind, rgb in _sdf_visuals(path):
        t = M._local(kind.tag)

        def put(p):
            return M.mat_apply(m, p)
        if t == "box":
            h = [v / 2 for v in M._floats(next(c.text for c in kind if M._local(c.tag) == "size"))]
            for f in _BOX_FACES:
                _quads(out, rgb, *[put(tuple(h[i] * (1 if q[i] else -1) for i in range(3))) for q in f])
        elif t == "cylinder":
            r = float(next(c.text for c in kind if M._local(c.tag) == "radius"))
            hl = float(next(c.text for c in kind if M._local(c.tag) == "length")) / 2
            n = SDF_SEGMENTS
            ring = [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)) for i in range(n + 1)]
            for i in range(n):
                (x0, y0), (x1, y1) = ring[i], ring[i + 1]
                _quads(out, rgb, put((x0, y0, -hl)), put((x1, y1, -hl)), put((x1, y1, hl)), put((x0, y0, hl)))
                out.append((put((0, 0, hl)), put((x0, y0, hl)), put((x1, y1, hl)), rgb))
                out.append((put((0, 0, -hl)), put((x1, y1, -hl)), put((x0, y0, -hl)), rgb))
        elif t == "sphere":
            r = float(next(c.text for c in kind if M._local(c.tag) == "radius"))
            nlat, nlon = 8, 12

            def pt(i, j):
                th, ph = math.pi * i / nlat, 2 * math.pi * j / nlon
                return put((r * math.sin(th) * math.cos(ph), r * math.sin(th) * math.sin(ph), r * math.cos(th)))
            for i in range(nlat):
                for j in range(nlon):
                    a, b, c, d = pt(i, j), pt(i + 1, j), pt(i + 1, j + 1), pt(i, j + 1)
                    if i < nlat - 1:
                        out.append((a, b, c, rgb))                    # (the last ring meets the bottom pole: b == c, so no triangle here)
                    if i > 0:
                        out.append((a, c, d, rgb))                    # (the first ring meets the top pole: a == d)
        elif t == "mesh":
            uri = next((c.text for c in kind if M._local(c.tag) == "uri"), None)
            if not uri:
                raise M.MeasureError("mesh visual without a <uri>")
            sc = next((c for c in kind if M._local(c.tag) == "scale"), None)
            sv = M._floats(sc.text) if sc is not None and (sc.text or "").strip() else [1.0, 1.0, 1.0]
            if len(sv) != 3:
                raise M.MeasureError("<scale> needs 3 numbers")
            f = M._find_mesh_file(uri, Path(path).parent, Path(path).parent)
            out += mesh_file_triangles(f, M.mat_mul(m, M.mat_scale(sv)), rgb)
        else:
            raise M.MeasureError(f"SDF geometry <{t}> cannot be drawn")
    if not out:
        raise M.MeasureError("the SDF has nothing to draw")
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


def lzma_alone(body: bytes) -> bytes:
    """`body` as the legacy .lzma stream with the uncompressed size KNOWN in its header (props 5 B | u64 size | stream): the one framing
    Garry's Mod's util.Decompress accepts. Verified in the real game 2026-10-11: the size field must hold the real length (-1 / unknown
    makes util.Decompress return nothing); xz and a header-less stream are refused as well. util.Compress emits exactly this framing."""
    a = lzma.compress(body, format=lzma.FORMAT_ALONE,
                      filters=[{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 18, "lc": 3, "lp": 0, "pb": 2}])
    return a[:5] + struct.pack("<Q", len(body)) + a[13:]


def unlzma_alone(blob: bytes) -> bytes:
    """The inverse, for tests (liblzma wants the unknown-size form of a stream that ends with an end marker, which ours does)."""
    return lzma.decompress(blob[:5] + bytes([255]) * 8 + blob[13:], format=lzma.FORMAT_ALONE)


def pack(path, target_tris=3000, codec="zlib", scale=1.0):
    """`scale` multiplies every coordinate (NASA files whose published size sets the scale); 1.0 = as authored."""
    raw = triangles(path)
    if scale != 1.0:
        raw = [tuple(tuple(c * scale for c in p) for p in t[:3]) + (t[3],) for t in raw]
    tris = on_floor(raw)
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
    if codec == "lzma":
        blob = lzma_alone(bytes(body))
    elif codec == "zlib":
        blob = zlib.compress(bytes(body), 9)
    else:
        raise ValueError(f"unknown codec {codec!r}")
    info = {"tris_in": len(tris), "tris_out": len(faces), "verts": len(pos),
            "bbox_m": [round(mx[i] - mn[i], 4) for i in range(3)], "raw_bytes": len(body), "bytes": len(blob), "codec": codec}
    return blob, info


def unpack(blob):
    """The inverse, for tests: (verts in metres, faces, colours)."""
    b = unlzma_alone(blob) if blob[:1] == bytes([0x5D]) else zlib.decompress(blob)         # zlib streams start 0x78; legacy lzma props 0x5d
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
    ap.add_argument("--codec", default="zlib", choices=["zlib", "lzma"])
    a = ap.parse_args(argv)
    blob, info = pack(a.model, a.target_tris, a.codec)
    print(info)
    if a.out:
        Path(a.out).write_bytes(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
