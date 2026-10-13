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
from skippyverse import surface as S  # noqa: E402

MAGIC = b"SVM1"
MAGIC3 = b"SVM3"
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
    made of box/sphere/cylinder/mesh visuals. The colour is the material's (with a base-colour texture sampled when it can be read)."""
    return [t[:4] for t in surface_triangles(path)]


def _gltf_image_bytes(doc, buffers, base, tex_index):
    """The bytes of glTF texture `tex_index`'s image (an external file next to the model, a data: URI or a bufferView), or None."""
    try:
        tex = doc["textures"][tex_index]
        src = tex.get("source")
        if src is None:
            src = ((tex.get("extensions") or {}).get("EXT_texture_webp") or {}).get("source")
        img = doc["images"][src]
        if img.get("uri") is not None:
            uri = img["uri"]
            if uri.startswith("data:"):
                import base64
                return base64.b64decode(uri.split(",", 1)[1])
            f = Path(base) / M._unquote(uri)
            return f.read_bytes() if f.is_file() else None
        bv = doc["bufferViews"][img["bufferView"]]
        buf = buffers[bv["buffer"]]
        start = bv.get("byteOffset", 0)
        return bytes(buf[start: start + bv["byteLength"]]) if buf is not None else None
    except (KeyError, IndexError, TypeError, OSError, ValueError):
        return None


def _gltf_uv(doc, buffers, acc_index):
    """[(u, v)] of a TEXCOORD accessor (float, or normalised ubyte/ushort), or None."""
    try:
        acc = doc["accessors"][acc_index]
        if acc.get("type") != "VEC2" or acc.get("sparse") or acc.get("bufferView") is None:
            return None
        ct = acc["componentType"]
        fmt, size, norm = {5126: ("f", 4, 1.0), 5121: ("B", 1, 255.0), 5123: ("H", 2, 65535.0)}[ct]
        bv = doc["bufferViews"][acc["bufferView"]]
        buf = buffers[bv["buffer"]]
        stride = bv.get("byteStride") or 2 * size
        start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        n = acc["count"]
        if buf is None or (n and start + (n - 1) * stride + 2 * size > len(buf)):
            return None
        out = []
        for i in range(n):
            u, v = struct.unpack_from("<2" + fmt, buf, start + i * stride)
            out.append((u / norm, v / norm))
        return out
    except (KeyError, IndexError, TypeError, struct.error):
        return None


class _GltfMaterial:
    """What one glTF material says about a surface: the colour factor, a base-colour texture, the metal-rough texture (G rough, B metal),
    the emissive colour, the alpha mode."""

    def __init__(self, doc, buffers, base, mi, cache):
        m = (doc.get("materials") or [{}])[mi] if mi is not None and mi < len(doc.get("materials") or []) else {}
        pbr = m.get("pbrMetallicRoughness") or {}
        self.factor = list(pbr.get("baseColorFactor") or [1.0, 1.0, 1.0, 1.0])
        self.tex = self.mr = None
        self.uv_set, self.mr_set = 0, 0
        bt, mt = pbr.get("baseColorTexture"), pbr.get("metallicRoughnessTexture")
        if bt is not None:
            self.tex = self._tex(doc, buffers, base, bt.get("index"), cache)
            self.uv_set = bt.get("texCoord", 0)
        if mt is not None:
            self.mr = self._tex(doc, buffers, base, mt.get("index"), cache)
            self.mr_set = mt.get("texCoord", 0)
        self.has_factor = "baseColorFactor" in pbr
        self.metal_f, self.rough_f = pbr.get("metallicFactor"), pbr.get("roughnessFactor")
        e = m.get("emissiveFactor") or [0, 0, 0]
        self.emit_rgb = tuple(S.to_display(c) for c in e[:3])
        self.emit = S.lum255(self.emit_rgb) if max(e[:3] or [0]) > 0.02 else 0.0
        self.mode = m.get("alphaMode", "OPAQUE")
        self.cutoff = m.get("alphaCutoff", 0.5)

    @staticmethod
    def _tex(doc, buffers, base, index, cache):
        if index is None:
            return None
        key = (id(doc), index)
        if key not in cache:
            cache[key] = S.load_texture(_gltf_image_bytes(doc, buffers, base, index))
        return cache[key]


def _gltf_surface(mat, uvs, mr_uvs):
    """-> (rgb, cue) for a triangle, or None when an alpha-masked texel is cut out."""
    f = mat.factor
    rgb = S.tri_colour(tuple(f[:3]), mat.tex, uvs)
    alpha = f[3] if len(f) > 3 else 1.0
    if mat.tex is not None and uvs and (mat.mode != "OPAQUE"):
        cu = sum(p[0] for p in uvs) / 3.0
        cv = sum(p[1] for p in uvs) / 3.0
        alpha *= mat.tex.sample(cu, cv)[3] / 255.0
    if mat.mode == "MASK" and alpha < mat.cutoff:
        return None
    metal, rough = mat.metal_f, mat.rough_f
    if mat.mr is not None and mr_uvs:
        cu = sum(p[0] for p in mr_uvs) / 3.0
        cv = sum(p[1] for p in mr_uvs) / 3.0
        s = mat.mr.sample(cu, cv)
        metal = (1.0 if metal is None else metal) * s[2] / 255.0
        rough = (1.0 if rough is None else rough) * s[1] / 255.0
    if mat.emit > 0.05:
        rgb = mat.emit_rgb
    cue = S.pbr_cue(metal, rough, mat.emit)
    a255 = 255 if mat.mode != "BLEND" else max(0, min(255, int(round(255 * alpha))))
    return rgb, cue[:3] + (a255,)


def surface_triangles(path):
    """[(p0, p1, p2, rgb, cue)] as triangles() but with the material cue (spec, metal, emit, alpha bytes: surface.pbr_cue) of every triangle."""
    if Path(path).suffix.lower() == ".sdf":
        return [t if len(t) == 5 else t + (S.DEFAULT_CUE,) for t in sdf_triangles(path)]
    if Path(path).suffix.lower() in (".dae", ".obj"):
        return mesh_file_triangles(path, M.IDENT, None)
    why = gltf_unreadable_why(path)
    if why:
        raise M.MeasureError(why)
    doc, buffers = M._load_gltf(Path(path))
    base = Path(path).parent
    nodes = doc.get("nodes", [])
    scenes = doc.get("scenes", [])
    if scenes:
        roots = scenes[doc.get("scene", 0)].get("nodes", [])
    else:
        children = {c for n in nodes for c in n.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in children]
    canon = M._canon("y", 1.0)
    out = []
    tex_cache, mats = {}, {}

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
                uv_all = None
                if M.draco_ext(prim) is not None:
                    pts, ix = M.decode_draco_primitive(doc, buffers, prim)
                    uv_all = {0: M.draco_uvs(doc, buffers, prim)}
                else:
                    pts, _, _ = M._accessor_points(doc, buffers, prim["attributes"]["POSITION"])
                    if pts is None:
                        continue
                    ix = None
                pts = [M.mat_apply(full, p) for p in pts]
                if ix is None:
                    ix = _indices(doc, buffers, prim["indices"]) if "indices" in prim else list(range(len(pts)))
                mi = prim.get("material")
                if mi not in mats:
                    mats[mi] = _GltfMaterial(doc, buffers, base, mi, tex_cache)
                mat = mats[mi]
                uvs = {}

                def uvset(k):
                    if k not in uvs:
                        if uv_all is not None and k in uv_all:
                            uvs[k] = uv_all[k]
                        else:
                            a = prim["attributes"].get(f"TEXCOORD_{k}")
                            uvs[k] = _gltf_uv(doc, buffers, a) if a is not None else None
                    return uvs[k]
                want_tex = mat.tex is not None or mat.mr is not None or mat.mode != "OPAQUE"
                uv_b = uvset(mat.uv_set) if want_tex else None
                uv_m = uvset(mat.mr_set) if mat.mr is not None else None
                flat = None
                if not want_tex or (uv_b is None and uv_m is None):
                    flat = _gltf_surface(mat, None, None)             # no texture to read (or no UVs): one colour for the whole primitive
                for k in range(0, len(ix) - 2, 3):
                    a, b, c = ix[k], ix[k + 1], ix[k + 2]
                    if flat is not None:
                        sf = flat
                    else:
                        sf = _gltf_surface(mat, [uv_b[a], uv_b[b], uv_b[c]] if uv_b else None, [uv_m[a], uv_m[b], uv_m[c]] if uv_m else None)
                    if sf is None:
                        continue
                    out.append((pts[a], pts[b], pts[c], sf[0], sf[1]))
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
                    out.append((pts[ix[0]], pts[ix[k]], pts[ix[k + 1]], rgb or (200, 200, 200), S.DEFAULT_CUE))
    return out


def _dae_texture_file(name, dae_path):
    """The image file a Collada <init_from> names, or a sibling of the same stem (the .dae often says .tga and ships a .png), or None."""
    if not name:
        return None
    n = M._unquote(name.strip()).replace("file://", "").replace("\\", "/")
    base = Path(dae_path).parent
    roots = [base, base.parent]
    stem = Path(n).stem.lower()
    for r in roots:
        f = r / n
        if f.is_file():
            return f
    want = {Path(n).name.lower(), stem}
    for r in roots:
        for f in (sorted(r.rglob("*")) if r.is_dir() else []):
            if f.is_file() and f.suffix.lower() in S.IMAGE_SUFFIXES and (f.name.lower() in want or f.stem.lower() in want):
                return f
    return None


def _dae_materials(root, dae_path):
    """{material id: {"factor": (r, g, b) linear, "tex": Texture | None, "cue": (spec, metal, emit, alpha)}} from the file's effects."""
    images = {}
    for im in root.iter():
        if M._local(im.tag) == "image":
            init = next((c.text for c in im if M._local(c.tag) == "init_from" and c.text), None)
            if im.get("id") and init:
                images[im.get("id")] = init
    effects = {}
    for ef in root.iter():
        if M._local(ef.tag) != "effect":
            continue
        params = {}
        for np_ in ef.iter():
            if M._local(np_.tag) == "newparam" and np_.get("sid"):
                src = next((c.text.strip() for c in np_.iter() if M._local(c.tag) in ("source", "init_from") and c.text), None)
                params[np_.get("sid")] = src
        tech = next((c for c in ef.iter() if M._local(c.tag) in ("lambert", "phong", "blinn", "constant")), None)
        if tech is None:
            continue

        def slot(name, tech=tech):
            el = next((c for c in tech if M._local(c.tag) == name), None)
            if el is None:
                return None, None
            col = next((c for c in el if M._local(c.tag) == "color"), None)
            tx = next((c for c in el if M._local(c.tag) == "texture"), None)
            return (M._floats(col.text) if col is not None else None), (tx.get("texture") if tx is not None else None)
        dif, dif_tex = slot("diffuse")
        spec, _ = slot("specular")
        emi, _ = slot("emission")
        sh = next((c for c in tech if M._local(c.tag) == "shininess"), None)
        shin = next((float(c.text) for c in sh.iter() if M._local(c.tag) == "float" and c.text), 20.0) if sh is not None else 20.0
        effects[ef.get("id")] = (dif, dif_tex, spec, emi, shin, params, M._local(tech.tag))
    out = {}
    for mat in root.iter():
        if M._local(mat.tag) != "material":
            continue
        inst = next((c for c in mat if M._local(c.tag) == "instance_effect"), None)
        eff = effects.get((inst.get("url", "") if inst is not None else "").lstrip("#"))
        if eff is None:
            continue
        dif, dif_tex, spec, emi, shin, params, kind = eff
        tex = None
        if dif_tex:
            ref = dif_tex
            for _ in range(3):                                             # sampler -> surface -> image
                nxt = params.get(ref)
                if nxt is None:
                    break
                ref = nxt
            f = _dae_texture_file(images.get(ref) or ref, dae_path)
            if f is not None:
                tex = S.load_texture(f.read_bytes())
        factor = tuple(dif[:3]) if dif and len(dif) >= 3 else (1.0, 1.0, 1.0)
        if tex is not None:
            factor = (1.0, 1.0, 1.0)
        sp = S.lum255([255 * c for c in spec[:3]]) if spec and len(spec) >= 3 and kind != "lambert" else 0.08
        gloss = max(0.0, min(1.0, sp * min(1.0, shin / 50.0)))
        em = S.lum255([255 * c for c in emi[:3]]) if emi and len(emi) >= 3 else 0.0
        out[mat.get("id")] = {"factor": factor, "tex": tex, "cue": (int(round(255 * gloss)), 0, int(round(255 * em)) if em > 0.05 else 0, 255),
                              "plain_white": tex is None and min(factor) > 0.95}
    return out


def _dae_loose_texture(dae_path):
    """A model whose Collada says plain white but ships a diffuse image in materials/textures (a Gazebo material script): that image, or None."""
    base = Path(dae_path).parent
    for r in (base, base.parent):
        d = r / "materials" / "textures"
        if d.is_dir():
            files = [f for f in sorted(d.iterdir()) if f.is_file()]
            pick = [f for f in files if any(k in f.name.lower() for k in ("diffuse", "albedo", "color", "colour", "basecolor"))] or files
            for f in pick:
                t = S.load_texture(f.read_bytes())
                if t is not None:
                    return t
    return None


def dae_triangles(path, post, rgb, submesh=None):
    """Collada <triangles>/<polylist>/<polygons> with the file's unit and up axis and the scene's node transforms applied, the way measure_dae
    measures them. Colours: each primitive's material (a diffuse colour, or its texture sampled at the UV centroid of the triangle), unless
    `rgb` is given (a Gazebo visual's own colour, which wins); (p0, p1, p2, rgb, cue). `submesh`: draw only the node/geometry of that name (an SDF's
    <submesh><name>: without it a file of ten parts used by ten visuals was drawn ten times over)."""
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
    try:
        materials = _dae_materials(root, path)
    except Exception:                                                      # a colour is a nicety: never lose the shape over it
        materials = {}
    sources = {}
    for el in root.iter():
        if M._local(el.tag) == "source":
            fa = next((c for c in el.iter() if M._local(c.tag) == "float_array"), None)
            acc = next((c for c in el.iter() if M._local(c.tag) == "accessor"), None)
            if fa is not None:
                sources[el.get("id")] = (M._floats(fa.text), int(acc.get("stride")) if acc is not None and acc.get("stride") else 3)
    geoms, gnames = {}, {}
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
                tin = next((c for c in ins if c.get("semantic") == "TEXCOORD"), None)
                uvs, uoff = None, 0
                if tin is not None:
                    vals, ustride = sources.get(tin.get("source", "").lstrip("#"), ([], 2))
                    uvs = [(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, ustride)]
                    uoff = int(tin.get("offset", 0))
                sym = prim.get("material")
                kind = M._local(prim.tag)
                rings = []
                if kind == "polygons":
                    for p in (q for q in prim if M._local(q.tag) == "p"):
                        v = [int(x) for x in p.text.split()]
                        rings.append([(v[i], v[i - off + uoff] if uvs else None) for i in range(off, len(v), stride)])
                else:
                    flat = [int(x) for x in next(q.text for q in prim if M._local(q.tag) == "p").split()]
                    if kind == "triangles":
                        counts = [3] * (len(flat) // (3 * stride))
                    else:
                        counts = [int(x) for x in next(q.text for q in prim if M._local(q.tag) == "vcount").split()]
                    k = 0
                    for n in counts:
                        rings.append([(flat[(k + j) * stride + off], flat[(k + j) * stride + uoff] if uvs else None) for j in range(n)])
                        k += n
                for r in rings:
                    for j in range(1, len(r) - 1):
                        tri = (r[0], r[j], r[j + 1])
                        tuv = tuple(uvs[t[1]] for t in tri) if uvs and all(t[1] is not None and t[1] < len(uvs) for t in tri) else None
                        tris.append((pos[r[0][0]], pos[r[j][0]], pos[r[j + 1][0]], sym, tuv))
        geoms[g.get("id")] = tris
        gnames[g.get("id")] = (g.get("name"), g.get("id"))

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
    state = {"loose": None, "tried": False}

    def surface_of(mat, tuv):
        if rgb is not None:
            return rgb, S.DEFAULT_CUE
        if mat is None:
            return (200, 200, 200), S.DEFAULT_CUE
        tex = mat["tex"]
        if tex is None and mat["plain_white"]:
            if not state["tried"]:
                state["tried"], state["loose"] = True, _dae_loose_texture(path)
            tex = state["loose"]
        return S.tri_colour(mat["factor"], tex, tuv, flip_v=True), mat["cue"]

    def wanted(gid, node):
        if not submesh:
            return True
        names = {node.get("name") if node is not None else None, node.get("id") if node is not None else None, *gnames.get(gid, ())}
        names |= {n[:-5] for n in names if isinstance(n, str) and n.endswith("-mesh")}
        return submesh in names

    def emit(gid, world, binding):
        full = M.mat_mul(total, world)
        cache = {}
        for a, b, c, sym, tuv in geoms.get(gid, []):
            mat = materials.get(binding.get(sym, sym)) or materials.get(sym)
            if tuv is None and sym in cache:
                sf = cache[sym]
            else:
                sf = surface_of(mat, tuv)
                if tuv is None:
                    cache[sym] = sf
            out.append((M.mat_apply(full, a), M.mat_apply(full, b), M.mat_apply(full, c), sf[0], sf[1]))

    def bindings(inst):
        b = {}
        for im in inst.iter():
            if M._local(im.tag) == "instance_material" and im.get("symbol"):
                b[im.get("symbol")] = (im.get("target") or "").lstrip("#")
        return b

    def visit(node, parent, depth=0):
        if depth > 64:
            raise M.MeasureError("Collada node hierarchy too deep or cyclic")
        world = M.mat_mul(parent, node_matrix(node))
        for c in node:
            t = M._local(c.tag)
            if t == "instance_geometry":
                seen[0] += 1
                if wanted(c.get("url", "").lstrip("#"), node):
                    emit(c.get("url", "").lstrip("#"), world, bindings(c))
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
            if wanted(gid, None):
                emit(gid, M.IDENT, {})
    return out


def mesh_file_triangles(path, post, rgb, submesh=None):
    ext = Path(path).suffix.lower()
    if ext == ".dae":
        return dae_triangles(path, post, rgb, submesh)
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
            rgb = None
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
    for m, kind, vis_rgb in _sdf_visuals(path):
        t = M._local(kind.tag)
        rgb = vis_rgb or (200, 200, 200)

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
            sub = next((c for c in kind if M._local(c.tag) == "submesh"), None)
            sub_name = next((c.text.strip() for c in sub if M._local(c.tag) == "name" and c.text), None) if sub is not None else None
            out += mesh_file_triangles(f, M.mat_mul(m, M.mat_scale(sv)), vis_rgb, sub_name)
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
    return [tuple((p[0] - cx, p[1] - cy, p[2] - z0) for p in t[:3]) + tuple(t[3:]) for t in tris]


def _area(a, b, c):
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    x, y, z = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    return 0.5 * math.sqrt(x * x + y * y + z * z)


def _qem_point(q, centroid, lo, hi):
    """The point minimising the summed squared distance to the planes of `q` (see simplify_surface), pulled a little toward the cluster's centroid
    (a flat or one-plane cluster has no best point: the pull keeps it where the surface is) and kept within half a cell of the cell."""
    w = q[10]
    if w <= 0:
        return centroid
    lam = 1e-2 * (q[0] + q[3] + q[5]) / 3.0 + 1e-18
    a = [[q[0] + lam, q[1], q[2]], [q[1], q[3] + lam, q[4]], [q[2], q[4], q[5] + lam]]
    rhs = [-q[6] + lam * centroid[0], -q[7] + lam * centroid[1], -q[8] + lam * centroid[2]]
    m = [a[i] + [rhs[i]] for i in range(3)]                                   # Gauss-Jordan with pivoting
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-30:
            return centroid
        m[col], m[piv] = m[piv], m[col]
        for r in range(3):
            if r != col:
                f = m[r][col] / m[col][col]
                for c in range(col, 4):
                    m[r][c] -= f * m[col][c]
    p = [m[i][3] / m[i][i] for i in range(3)]
    return tuple(min(hi[i] + 0.5 * (hi[i] - lo[i]), max(lo[i] - 0.5 * (hi[i] - lo[i]), p[i])) for i in range(3))


def simplify_surface(tris, target, qem=True):
    """Vertex clustering: snap vertices to a grid, drop triangles that collapse, drop duplicates. `tris` are (p0, p1, p2, rgb[, cue]).
    Returns (verts, faces, colours, cues, areas). A surviving face takes the DOMINANT colour among every original triangle (the collapsed
    ones too) that touches its three vertices or landed on it (colours binned 16 per channel, area-weighted; the face's own triangles count
    double), and the mean colour and cue inside that winning bin: detail the clustering removed still decides the colour, but unlike a plain
    mean a red frosting next to a brown sponge does not become purple.
    `qem`: each cluster's vertex is placed where the planes of its triangles meet best (the quadric error minimiser, Lindstrom / Garland-Heckbert
    clustering) instead of at their centroid, so corners, edges and thin parts keep their place and a box stays a box."""
    if not tris:
        return [], [], [], [], []
    pts = [p for t in tris for p in t[:3]]
    mn = tuple(min(p[i] for p in pts) for i in range(3))
    mx = tuple(max(p[i] for p in pts) for i in range(3))
    ext = max(mx[i] - mn[i] for i in range(3)) or 1.0
    dflt = S.DEFAULT_CUE

    def cluster(cells, full):
        # cells along the longest axis; a thinner axis gets proportionally fewer but never fewer than 2, so a thin shape is not flattened
        n = [max(2, round(cells * (mx[i] - mn[i]) / ext)) for i in range(3)]
        h = [((mx[i] - mn[i]) / n[i]) or 1.0 for i in range(3)]
        vid, order, faces, seen, acc = {}, [], [], {}, {}
        fbins, vbins = [], []                                 # per face and per vertex: {bin: [w, r, g, b, spec, metal, emit, alpha] weighted sums}
        quad = []                                             # per vertex: [Axx, Axy, Axz, Ayy, Ayz, Azz, bx, by, bz, c, w] (the plane quadrics, area-weighted)
        for t in tris:
            a, b, c, rgb = t[0], t[1], t[2], t[3]
            key = []
            for p in (a, b, c):
                k = tuple(min(n[i] - 1, int((p[i] - mn[i]) / h[i])) for i in range(3))
                if k not in vid:
                    vid[k] = len(order)
                    order.append(k)
                    acc[k] = [0.0, 0.0, 0.0, 0]
                    if full:
                        vbins.append({})
                        quad.append([0.0] * 11)
                s = acc[k]
                s[0] += p[0]
                s[1] += p[1]
                s[2] += p[2]
                s[3] += 1
                key.append(vid[k])
            bk = vals = None
            if full:
                w = _area(a, b, c) + 1e-12
                if qem:
                    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
                    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
                    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
                    nl = math.sqrt(nx * nx + ny * ny + nz * nz)
                    if nl > 1e-18:
                        nx, ny, nz = nx / nl, ny / nl, nz / nl
                        d = -(nx * a[0] + ny * a[1] + nz * a[2])
                        qv = (w * nx * nx, w * nx * ny, w * nx * nz, w * ny * ny, w * ny * nz, w * nz * nz, w * d * nx, w * d * ny, w * d * nz, w * d * d, w)
                        for v in set(key):
                            qq = quad[v]
                            for j in range(11):
                                qq[j] += qv[j]
                cue = t[4] if len(t) > 4 else dflt
                vals = [w, w * rgb[0], w * rgb[1], w * rgb[2], w * cue[0], w * cue[1], w * cue[2], w * cue[3]]
                bk = (rgb[0] >> 4, rgb[1] >> 4, rgb[2] >> 4, cue[3] < 250)
                for v in set(key):
                    d = vbins[v]
                    sv = d.get(bk)
                    if sv is None:
                        d[bk] = list(vals)
                    else:
                        for j in range(8):
                            sv[j] += vals[j]
            if len(set(key)) < 3:
                continue
            canon = tuple(sorted(key))
            fi = seen.get(canon)
            if fi is None:
                seen[canon] = len(faces)
                faces.append(tuple(key))
                if full:
                    fbins.append({bk: list(vals)})
            elif full:
                d = fbins[fi]
                sv = d.get(bk)
                if sv is None:
                    d[bk] = list(vals)
                else:
                    for j in range(8):
                        sv[j] += vals[j]
        pos = [(acc[k][0] / acc[k][3], acc[k][1] / acc[k][3], acc[k][2] / acc[k][3]) for k in order]
        if not full:
            return pos, faces, None, None, None
        if qem:
            for i, k in enumerate(order):
                pos[i] = _qem_point(quad[i], pos[i], [mn[j] + k[j] * h[j] for j in range(3)], [mn[j] + (k[j] + 1) * h[j] for j in range(3)])
        cols, cues, areas = [], [], []
        for f, own in zip(faces, fbins):
            tot = {}
            for d, mult in ((own, OWN_WEIGHT), (vbins[f[0]], 1.0), (vbins[f[1]], 1.0), (vbins[f[2]], 1.0)) if not OWN_ONLY else ((own, 1.0),):
                for bk, sv in d.items():
                    t = tot.get(bk)
                    if t is None:
                        tot[bk] = [mult * x for x in sv]
                    else:
                        for j in range(8):
                            t[j] += mult * sv[j]
            best = max(tot.values(), key=lambda t: t[0])
            w = best[0]
            cols.append((int(round(best[1] / w)), int(round(best[2] / w)), int(round(best[3] / w))))
            cues.append((int(round(best[4] / w)), int(round(best[5] / w)), int(round(best[6] / w)), int(round(best[7] / w))))
            areas.append(_area(pos[f[0]], pos[f[1]], pos[f[2]]) + 1e-12)
        return pos, faces, cols, cues, areas

    lo, hi = 4, 512
    best_cells = lo
    while lo < hi:                                           # the finest grid that stays at or under the target
        mid = (lo + hi + 1) // 2
        if len(cluster(mid, False)[1]) <= target:
            best_cells, lo = mid, mid
        else:
            hi = mid - 1
    best = cluster(best_cells, False)
    cells = 4
    while not best[1] and cells < 4096:                      # a thin shape collapses on a coarse grid: refine until something is left
        cells *= 2
        best = cluster(cells, False)
        best_cells = cells
    return cluster(best_cells, True)


def simplify(tris, target):
    """simplify_surface without the cues: (verts, faces, colours)."""
    pos, faces, cols, _, _ = simplify_surface(tris, target, qem=False)          # (the original centroid clustering: the legacy SVM1 payload is unchanged)
    return pos, faces, cols


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


def _compress(body: bytes, codec: str) -> bytes:
    if codec == "lzma":
        return lzma_alone(bytes(body))
    if codec == "zlib":
        return zlib.compress(bytes(body), 9)
    raise ValueError(f"unknown codec {codec!r}")


MAX_PAL = 24
OWN_ONLY = False                               # a face takes its colour from the triangles that landed on it (not the neighbourhood: interior layers would win)
OWN_WEIGHT = 6.0                              # the triangles that landed on a face count this much more than its vertices' other neighbours
SVM3_MAX_VERTS = 12000                        # what the game client accepts (svm1.lua MAX_VERTS3)


def pack(path, target_tris=3000, codec="zlib", scale=1.0, fmt="svm1"):
    """`scale` multiplies every coordinate (NASA files whose published size sets the scale); 1.0 = as authored.
    fmt "svm1": flat colour per triangle (the original payload). fmt "svm3": colour regions (a palette of <= 24 entries, each with a gloss, a
    metal level, a glow and an opacity), smooth vertex normals with creases split, a palette index per triangle (see the module docstring)."""
    raw = surface_triangles(path)
    if scale != 1.0:
        raw = [tuple(tuple(c * scale for c in p) for p in t[:3]) + tuple(t[3:]) for t in raw]
    tris = on_floor(raw)
    if fmt == "svm3":
        return _pack3(tris, target_tris, codec)
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
    blob = _compress(bytes(body), codec)
    info = {"tris_in": len(tris), "tris_out": len(faces), "verts": len(pos),
            "bbox_m": [round(mx[i] - mn[i], 4) for i in range(3)], "raw_bytes": len(body), "bytes": len(blob), "codec": codec}
    return blob, info


def _pack3(tris, target_tris, codec):
    target = target_tris
    while True:
        pos, faces, cols, cues, areas = simplify_surface(tris, target)
        if not faces:
            raise M.MeasureError("nothing to pack")
        pal, pidx = S.build_palette([(cols[i], cues[i], areas[i]) for i in range(len(faces))], MAX_PAL)
        order = sorted(range(len(faces)), key=lambda i: (pidx[i], i))               # runs of one palette entry compress to nothing
        faces = [faces[i] for i in order]
        pidx = [pidx[i] for i in order]
        vpos, vnrm, faces = S.smooth_normals(pos, faces)
        if len(vpos) <= SVM3_MAX_VERTS and len(vpos) <= 65535:
            break
        target = int(target * 0.8)
        if target < 50:
            raise M.MeasureError("too many vertices for one payload")
    mn = [min(p[i] for p in vpos) for i in range(3)]
    mx = [max(p[i] for p in vpos) for i in range(3)]
    body = bytearray()
    body += MAGIC3 + struct.pack("<HH6f", len(vpos), len(faces), *mn, *mx)
    body += struct.pack("<B", len(pal))
    for e in pal:
        body += struct.pack("<7B", *e)
    for p in vpos:
        body += struct.pack("<3H", *[int(round(65535 * (p[i] - mn[i]) / ((mx[i] - mn[i]) or 1.0))) for i in range(3)])
    for f in faces:
        body += struct.pack("<3H", *f)
    body += bytes(pidx)
    for n in vnrm:
        body += struct.pack("<3b", *[max(-127, min(127, int(round(127 * c)))) for c in n])
    blob = _compress(bytes(body), codec)
    info = {"tris_in": len(tris), "tris_out": len(faces), "verts": len(vpos), "palette": len(pal),
            "bbox_m": [round(mx[i] - mn[i], 4) for i in range(3)], "raw_bytes": len(body), "bytes": len(blob), "codec": codec, "format": "svm3"}
    return blob, info


def _inflate(blob):
    return unlzma_alone(blob) if blob[:1] == bytes([0x5D]) else zlib.decompress(blob)         # zlib streams start 0x78; legacy lzma props 0x5d


def unpack3(blob):
    """The inverse of an SVM3 payload, for tests and tools: dict(verts, normals, faces, palette, face_pal, bbox)."""
    b = _inflate(blob)
    if b[:4] != MAGIC3:
        raise M.MeasureError("not an SVM3 payload")
    nv, nt = struct.unpack_from("<HH", b, 4)
    f = struct.unpack_from("<6f", b, 8)
    mn, mx = f[:3], f[3:]
    npal = b[32]
    off = 33
    pal = [struct.unpack_from("<7B", b, off + 7 * i) for i in range(npal)]
    off += 7 * npal
    verts = []
    for i in range(nv):
        q = struct.unpack_from("<3H", b, off + 6 * i)
        verts.append(tuple(mn[k] + (mx[k] - mn[k]) * q[k] / 65535 for k in range(3)))
    off += 6 * nv
    faces = [struct.unpack_from("<3H", b, off + 6 * i) for i in range(nt)]
    off += 6 * nt
    face_pal = list(b[off: off + nt])
    off += nt
    nrm = [tuple(c / 127.0 for c in struct.unpack_from("<3b", b, off + 3 * i)) for i in range(nv)]
    return {"verts": verts, "normals": nrm, "faces": faces, "palette": pal, "face_pal": face_pal, "bbox": (mn, mx)}


def unpack(blob):
    """The inverse, for tests: (verts in metres, faces, colours). An SVM3 payload gives each triangle its palette colour."""
    b = _inflate(blob)
    if b[:4] == MAGIC3:
        d = unpack3(blob)
        return d["verts"], d["faces"], [tuple(d["palette"][i][:3]) for i in d["face_pal"]]
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
    ap.add_argument("--fmt", default="svm3", choices=["svm1", "svm3"])
    a = ap.parse_args(argv)
    blob, info = pack(a.model, a.target_tris, a.codec, fmt=a.fmt)
    print(info)
    if a.out:
        Path(a.out).write_bytes(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
