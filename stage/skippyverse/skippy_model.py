"""The canonical Skippy ("Skippy the Magnificent" 3D-print kit) as SkippyVerse avatar payloads: per-part, decimated, coloured, assembled.

    python stage/skippyverse/skippy_model.py <kit dir> [--out stage/skippyverse/private/skippy] [--height 0.6] [--preview]

The kit is a personal print kit of unknown licence: it and everything derived from it (payloads, previews, manifest) lives under the
gitignored stage/skippyverse/private/ and never goes to git. This module is code only and imports with the data absent.

Inputs (kit dir): `Individual Parts/*.stl` (binary STL, millimetres, ONE shared frame, y up: the parts already sit in their assembled
pose) and `Skippy_AMS_Coloured.3mf` (Bambu Studio project). What is read from where
  * geometry          : the STLs (the 3MF meshes are the same triangles, in the same order, shifted by the part's `source_offset`
                        in Metadata/model_settings.config; checked on a sample of triangles before any colour is trusted);
  * colour, REAL      : the 3MF. Each part has a default extruder (model_settings.config) and some triangles carry `paint_color`, a
                        Slic3r/Bambu triangle-selector tree in hex (see decode_paint); extruder -> colour is `filament_colour` in
                        Metadata/project_settings.config (silver #C0C0C0, navy #0000A0, yellow #FCEF03, red #FF0000, black, white);
  * colour, GUESSED   : nothing is invented per triangle; palette entries are the printer's filament colours, flat-shaded.
Decimation: weld the unindexed STL, one vertex-clustering pass down to ~6x the budget, then quadric-error edge collapse (Garland-Heckbert,
candidates: both ends and the midpoint) with colour borders and open borders held by extra penalty planes, and a flip check; then
smooth vertex normals with creases split (angle > 40 deg), so a rim or a hat edge stays sharp.

Payload: `SVM2`, SVM1 (stage/skippyverse/meshpack.py) plus per-vertex normals, LZMA in the framing Garry's Mod's util.Decompress reads:
    magic 'SVM2' | u16 n_verts | u16 n_tris | f32 min[3] | f32 max[3] | u16 verts[n][3] (quantised min..max) | u16 idx[t][3]
    | u8 rgb[t][3] (flat per triangle) | i8 normals[n][3] (snorm: n = v / 127)
Frame of every part: metres, +Z up, +X the avatar's front, origin on the floor under the footprint centre of the WHOLE figure, so all part
payloads are placed at the same origin and line up without per-part offsets (each part's own quantisation box is in its header).
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import struct
import sys
import zipfile
from array import array
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skippyverse import meshpack  # noqa: E402

MAGIC2 = b"SVM2"
PARTS = {  # role -> (STL stem in `Individual Parts`, triangle budget)
    "can": ("Skippy_can", 700),
    "jacket": ("Skippy_jacket", 3000),
    "head": ("Skippy_head_keyed", 4200),
    "hat": ("Skippy_hat", 1600),
    "hand_a": ("Skippy_hand_left", 1000),
    "hand_b": ("Skippy_hand_right", 600),
}


class ModelError(ValueError):
    pass


# ----------------------------------------------------------------------------------------------------------------------- STL

def read_stl(path_or_bytes):
    """Triangles as a flat array('f') of 9 floats each (x0 y0 z0 x1 ... z2). Binary or ASCII STL."""
    d = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else Path(path_or_bytes).read_bytes()
    out = array("f")
    if len(d) >= 84:
        n = struct.unpack_from("<I", d, 80)[0]
        if len(d) == 84 + 50 * n:                                     # exact binary size: a header may say "solid", the size cannot lie
            mv = memoryview(d)[84:]
            for r in struct.iter_unpack("<12xfffffffff2x", mv):
                out.extend(r)
            return out
    txt = d.decode("ascii", "replace")
    nums = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", txt)
    if not nums:
        raise ModelError("not a binary or ASCII STL")
    for t in nums:
        out.extend(float(v) for v in t)
    if len(out) % 9:
        raise ModelError("ASCII STL with a partial facet")
    return out


# ----------------------------------------------------------------------------------------------------------------------- 3MF paint

def decode_paint(code: str):
    """Area share per extruder state of one triangle's `paint_color`: {state: fraction}, state 0 = unpainted (object's default extruder).

    The string is a Slic3r TriangleSelector tree, hex nibbles read from the LAST character to the first, depth first. Nibble `c`:
    `c & 3` = number of split sides (0 = leaf, else the triangle splits into `(c & 3) + 1` children, bits 2-3 say which side is special);
    a leaf's state is `c >> 2` (0..2), and the value 3 means 'extended': the NEXT nibble n gives state 3 + n. So "4" = extruder 1, "8" = 2,
    "0C" = 3, "1C" = 4, "2C" = 5, "3C" = 6. Child areas are taken as 1/2 each for 2 children, 1/2,1/4,1/4 for 3, 1/4 each for 4 (exact for
    the 2- and 4-way splits, a close approximation for 3-way). The tree must consume the whole string, else ModelError."""
    if not code:
        return {0: 1.0}
    nib = [int(ch, 16) for ch in reversed(code)]
    pos = [0]
    acc: Counter = Counter()

    def node(frac, depth):
        if pos[0] >= len(nib) or depth > 40:
            raise ModelError("bad paint_color " + code)
        c = nib[pos[0]]
        pos[0] += 1
        k = c & 3
        if k == 0:
            s = c >> 2
            if s == 3:
                if pos[0] >= len(nib):
                    raise ModelError("bad paint_color " + code)
                s = 3 + nib[pos[0]]
                pos[0] += 1
            acc[s] += frac
            return
        shares = {1: (0.5, 0.5), 2: (0.5, 0.25, 0.25), 3: (0.25, 0.25, 0.25, 0.25)}[k]
        for sh in shares:
            node(frac * sh, depth + 1)

    node(1.0, 0)
    if pos[0] != len(nib):
        raise ModelError("trailing nibbles in paint_color " + code)
    return dict(acc)


def dominant_state(code: str, _cache={}):
    r = _cache.get(code)
    if r is None:
        sh = decode_paint(code)
        r = max(sorted(sh), key=lambda s: sh[s])
        if len(_cache) < 200000:
            _cache[code] = r
    return r


_TRI = re.compile(rb'<triangle v1="\d+" v2="\d+" v3="\d+"(?: paint_color="([0-9A-Fa-f]*)")?')


def read_3mf(path):
    """{'filaments': ['#C0C0C0', ...], 'objects': {name: {'extruder': int, 'offset': (x, y, z), 'file': 'object_N.model'}}} for the printed parts;
    the paint is read lazily with `object_states`."""
    z = zipfile.ZipFile(path)
    cfg = z.read("Metadata/model_settings.config").decode()
    proj = json.loads(z.read("Metadata/project_settings.config").decode())
    main = z.read("3D/3dmodel.model").decode()
    comp = {}                                           # build object id -> mesh file
    for m in re.finditer(r'<object id="(\d+)"[^>]*>\s*<components>\s*<component p:path="([^"]+)"', main):
        comp[m.group(1)] = m.group(2).lstrip("/")
    objs = {}
    for m in re.finditer(r'<object id="(\d+)">(.*?)</object>', cfg, re.S):
        body = m.group(2)
        name = re.search(r'<metadata key="name" value="([^"]*)"', body).group(1)
        ext = int(re.search(r'<metadata key="extruder" value="(\d+)"', body).group(1))
        off = tuple(float(re.search(rf'source_offset_{a}" value="([-\d.e+]+)"', body).group(1)) for a in "xyz")
        if m.group(1) in comp:
            objs.setdefault(name, []).append({"extruder": ext, "offset": off, "file": comp[m.group(1)]})
    return {"filaments": proj["filament_colour"], "objects": objs, "zip": z}


def object_states(kit, name, flat, max_unmatched=0.001):
    """Per-triangle extruder state (1-based) of a printed part in the STL's triangle order, plus the number of STL triangles the 3MF lacks.

    The 3MF mesh is the STL's, shifted by the part's source_offset, with degenerate triangles removed (the jacket loses 2 of 726068): so the
    two lists are walked together, each 3MF triangle matched to the next STL triangle (within 8) whose first corner is the same point. An STL
    triangle skipped that way takes the part's default extruder. More than `max_unmatched` of the triangles unmatched is a ModelError (the
    colours would not be trustworthy)."""
    o = kit["objects"][name][0]
    d = kit["zip"].read(o["file"])
    vs = [(float(a), float(b), float(c)) for a, b, c in re.findall(rb'<vertex x="([^"]+)" y="([^"]+)" z="([^"]+)"', d)]
    off = o["offset"]
    n = len(flat) // 9
    dflt = o["extruder"]
    out = [dflt] * n
    s = 0
    unmatched = 0
    nt = 0
    for m in re.finditer(rb'<triangle v1="(\d+)" v2="\d+" v3="\d+"(?: paint_color="([0-9A-Fa-f]*)")?', d):
        nt += 1
        v = vs[int(m.group(1))]
        hit = -1
        for k in range(s, min(n, s + 8)):
            if (abs(v[0] + off[0] - flat[9 * k]) < 1e-3 and abs(v[1] + off[1] - flat[9 * k + 1]) < 1e-3
                    and abs(v[2] + off[2] - flat[9 * k + 2]) < 1e-3):
                hit = k
                break
        if hit < 0:
            unmatched += 1
            continue
        unmatched += hit - s
        s = hit + 1
        c = m.group(2)
        if c:
            out[hit] = dominant_state(c.decode()) or dflt
    unmatched += n - s
    if unmatched > max_unmatched * n:
        raise ModelError(f"{name}: 3MF and STL do not line up ({unmatched} of {n} triangles unmatched); colours cannot be trusted")
    return out, unmatched


def hex_rgb(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# ----------------------------------------------------------------------------------------------------------------------- mesh ops

def weld(flat, cols):
    """Unindexed triangles -> (verts[(x,y,z)], faces[(a,b,c)], cols); drops degenerate faces (repeated index)."""
    vid, verts, faces, fc = {}, [], [], []
    for t in range(len(flat) // 9):
        ids = []
        for k in range(3):
            p = (flat[9 * t + 3 * k], flat[9 * t + 3 * k + 1], flat[9 * t + 3 * k + 2])
            i = vid.get(p)
            if i is None:
                i = vid[p] = len(verts)
                verts.append(p)
            ids.append(i)
        if ids[0] != ids[1] and ids[1] != ids[2] and ids[0] != ids[2]:
            faces.append(tuple(ids))
            fc.append(cols[t])
    return verts, faces, fc


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _area_normal(p, q, r):
    return _cross(_sub(q, p), _sub(r, p))                       # length = 2 * area


def surface_area(verts, faces):
    return 0.5 * sum(math.sqrt(_dot(n, n)) for n in (_area_normal(verts[a], verts[b], verts[c]) for a, b, c in faces))


def cluster(verts, faces, cols, h):
    """Vertex clustering on a grid of pitch h: vertices in a cell merge to their mean; collapsed and duplicate faces go; a merged face
    keeps the majority colour of the faces it stands for."""
    mn = [min(v[i] for v in verts) for i in range(3)]
    cell, pos_acc, new = {}, [], []
    cid = [0] * len(verts)
    for i, v in enumerate(verts):
        k = (int((v[0] - mn[0]) / h), int((v[1] - mn[1]) / h), int((v[2] - mn[2]) / h))
        j = cell.get(k)
        if j is None:
            j = cell[k] = len(pos_acc)
            pos_acc.append([0.0, 0.0, 0.0, 0])
        s = pos_acc[j]
        s[0] += v[0]
        s[1] += v[1]
        s[2] += v[2]
        s[3] += 1
        cid[i] = j
    new = [(s[0] / s[3], s[1] / s[3], s[2] / s[3]) for s in pos_acc]
    votes = {}
    for (a, b, c), col in zip(faces, cols):
        a, b, c = cid[a], cid[b], cid[c]
        if a == b or b == c or a == c:
            continue
        key = (a, b, c) if a < b and a < c else (b, c, a) if b < c else (c, a, b)       # canonical rotation keeps orientation
        v = votes.get(key)
        if v is None:
            votes[key] = {col: 1}
        else:
            v[col] = v.get(col, 0) + 1
    out_f = list(votes)
    out_c = [max(sorted(v), key=lambda c: v[c]) for v in votes.values()]
    return new, out_f, out_c


def _plane_quadric(n, d, w):
    a, b, c = n
    return [w * a * a, w * a * b, w * a * c, w * a * d, w * b * b, w * b * c, w * b * d, w * c * c, w * c * d, w * d * d]


def _qerr(q, p):
    x, y, z = p
    return (q[0] * x * x + 2 * q[1] * x * y + 2 * q[2] * x * z + 2 * q[3] * x + q[4] * y * y + 2 * q[5] * y * z + 2 * q[6] * y
            + q[7] * z * z + 2 * q[8] * z + q[9])


def qem(verts, faces, cols, target, border_weight=250.0, flip_dot=0.2):
    """Quadric-error edge collapse down to `target` faces (or until nothing legal is left). Returns (verts, faces, cols), orientation kept."""
    P = [list(v) for v in verts]
    F = [list(f) for f in faces]
    C = list(cols)
    alive = [True] * len(F)
    vf = [set() for _ in P]
    for fi, f in enumerate(F):
        for v in f:
            vf[v].add(fi)
    Q = [[0.0] * 10 for _ in P]
    edge_faces = {}
    for fi, (a, b, c) in enumerate(F):
        n = _area_normal(P[a], P[b], P[c])
        ln = math.sqrt(_dot(n, n))
        if ln == 0:
            continue
        un = (n[0] / ln, n[1] / ln, n[2] / ln)
        d = -_dot(un, P[a])
        q = _plane_quadric(un, d, ln / 2)
        for v in (a, b, c):
            qq = Q[v]
            for i in range(10):
                qq[i] += q[i]
        for e in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault((min(e), max(e)), []).append(fi)
    scale = 0.0
    for (a, b), fl in edge_faces.items():                       # penalty planes along colour borders and open borders
        border = len(fl) != 2 or C[fl[0]] != C[fl[1]]
        if not border:
            continue
        pa, pb = P[a], P[b]
        e = _sub(pb, pa)
        el = math.sqrt(_dot(e, e))
        if el == 0:
            continue
        for fi in fl:
            fa, fb, fc = F[fi]
            n = _area_normal(P[fa], P[fb], P[fc])
            ln = math.sqrt(_dot(n, n))
            if ln == 0:
                continue
            pn = _cross(e, n)
            pl = math.sqrt(_dot(pn, pn))
            if pl == 0:
                continue
            un = (pn[0] / pl, pn[1] / pl, pn[2] / pl)
            q = _plane_quadric(un, -_dot(un, pa), border_weight * el * el)
            for v in (a, b):
                qq = Q[v]
                for i in range(10):
                    qq[i] += q[i]
    ver = [0] * len(P)
    dead = [False] * len(P)

    def nbrs(v):
        s = set()
        for fi in vf[v]:
            s.update(F[fi])
        s.discard(v)
        return s

    def candidate(a, b):
        q = [Q[a][i] + Q[b][i] for i in range(10)]
        best = None
        pa, pb = P[a], P[b]
        for p in (pa, pb, ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, (pa[2] + pb[2]) / 2)):
            e = _qerr(q, p)
            if best is None or e < best[0]:
                best = (e, p)
        return best

    heap = []
    cnt = 0

    def push(a, b):
        nonlocal cnt
        e, p = candidate(a, b)
        cnt += 1
        heapq.heappush(heap, (e, cnt, a, b, ver[a], ver[b], p))

    for (a, b) in edge_faces:
        push(a, b)
    nalive = len(F)
    while nalive > target and heap:
        e, _, a, b, va, vb, p = heapq.heappop(heap)
        if dead[a] or dead[b] or ver[a] != va or ver[b] != vb:
            continue
        shared = vf[a] & vf[b]
        if not shared or len(nbrs(a) & nbrs(b)) != len(shared):
            continue
        ok = True
        for v in (a, b):                                        # no foldover, no sliver
            for fi in vf[v]:
                if fi in shared:
                    continue
                f = F[fi]
                old = _area_normal(P[f[0]], P[f[1]], P[f[2]])
                q = [p if x in (a, b) else P[x] for x in f]
                new = _area_normal(q[0], q[1], q[2])
                lo, ln = math.sqrt(_dot(old, old)), math.sqrt(_dot(new, new))
                if ln < 1e-12 or (lo > 0 and _dot(old, new) / (lo * ln) < flip_dot):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        P[a] = list(p)
        for i in range(10):
            Q[a][i] += Q[b][i]
        for fi in list(vf[b]):
            f = F[fi]
            if fi in shared:
                alive[fi] = False
                nalive -= 1
                for v in f:
                    if v != b:
                        vf[v].discard(fi)
                continue
            F[fi] = [a if x == b else x for x in f]
            vf[a].add(fi)
        vf[b].clear()
        dead[b] = True
        ver[a] += 1
        for n in nbrs(a):
            push(min(a, n), max(a, n))
    remap, nv, nf, nc = {}, [], [], []
    for fi, f in enumerate(F):
        if not alive[fi]:
            continue
        ids = []
        for v in f:
            if v not in remap:
                remap[v] = len(nv)
                nv.append(tuple(P[v]))
            ids.append(remap[v])
        nf.append(tuple(ids))
        nc.append(C[fi])
    return nv, nf, nc


def decimate(flat, cols, target):
    """STL triangles + per-triangle colour -> (verts, faces, cols) of about `target` faces."""
    verts, faces, fc = weld(flat, cols)
    pre = target * 6
    if len(faces) > pre * 1.5:
        h = math.sqrt(surface_area(verts, faces) * 1.9 / pre)
        verts, faces, fc = cluster(verts, faces, fc, h)
    for flip in (0.2, -0.1, -0.5):                       # a pass can stall on flip checks: relax them step by step, but only if it must
        verts, faces, fc = qem(verts, faces, fc, target, flip_dot=flip)
        if len(faces) <= target * 1.03:
            break
    return verts, faces, fc


def shade(verts, faces, crease_deg=40.0):
    """Smooth vertex normals with creases split: a vertex is duplicated per group of incident faces whose normals lie within crease_deg of
    the group's first. Returns (verts, normals, faces, face_map) with face_map[i] = source face index (to carry colours)."""
    fn = []
    for a, b, c in faces:
        n = _area_normal(verts[a], verts[b], verts[c])
        fn.append(n)
    inc = [[] for _ in verts]
    for fi, f in enumerate(faces):
        for v in f:
            inc[v].append(fi)
    cs = math.cos(math.radians(crease_deg))

    def unit(n):
        ln = math.sqrt(_dot(n, n)) or 1.0
        return (n[0] / ln, n[1] / ln, n[2] / ln)

    out_v, out_n, slot = [], [], {}
    for v, fl in enumerate(inc):
        groups = []                                              # [(ref unit normal, sum normal, [faces])]
        for fi in fl:
            u = unit(fn[fi])
            for g in groups:
                if _dot(u, g[0]) >= cs:
                    g[1][0] += fn[fi][0]
                    g[1][1] += fn[fi][1]
                    g[1][2] += fn[fi][2]
                    g[2].append(fi)
                    break
            else:
                groups.append((u, list(fn[fi]), [fi]))
        for g in groups:
            j = len(out_v)
            out_v.append(verts[v])
            out_n.append(unit(g[1]))
            for fi in g[2]:
                slot[(v, fi)] = j
    nf = [tuple(slot[(v, fi)] for v in f) for fi, f in enumerate(faces)]
    return out_v, out_n, nf


# ----------------------------------------------------------------------------------------------------------------------- frame, pack

def frame_matrix(front_xy):
    """Source (x, y up, z) mm -> game (X front, Y left, Z up): up = y -> Z; the source horizontal direction `front_xy` = (sx, sz) -> +X.
    Returns a function p -> (X, Y, Z) (unscaled, unshifted). It is a proper rotation."""
    fx, fz = front_xy
    ln = math.hypot(fx, fz) or 1.0
    fx, fz = fx / ln, fz / ln
    # source (x, y up, z) is right-handed; Y (left) must satisfy front x left = up, which gives left = (fz, -fx) in the (x, z) plane
    lx, lz = fz, -fx

    def f(p):
        return (p[0] * fx + p[2] * fz, p[0] * lx + p[2] * lz, p[1])

    return f


def pack2(verts, normals, faces, cols, codec="lzma"):
    """One part as an SVM2 payload (see module docstring). Returns (blob, info)."""
    if not faces or len(verts) > 65535:
        raise ModelError("nothing to pack, or too many vertices for u16")
    mn = [min(p[i] for p in verts) for i in range(3)]
    mx = [max(p[i] for p in verts) for i in range(3)]
    body = bytearray(MAGIC2 + struct.pack("<HH6f", len(verts), len(faces), *mn, *mx))
    for p in verts:
        body += struct.pack("<3H", *[int(round(65535 * (p[i] - mn[i]) / ((mx[i] - mn[i]) or 1.0))) for i in range(3)])
    for f in faces:
        body += struct.pack("<3H", *f)
    for c in cols:
        body += struct.pack("<3B", *c)
    for n in normals:
        body += struct.pack("<3b", *[max(-127, min(127, int(round(127 * x)))) for x in n])
    blob = meshpack.lzma_alone(bytes(body)) if codec == "lzma" else meshpack.zlib.compress(bytes(body), 9)
    return blob, {"verts": len(verts), "tris": len(faces), "raw_bytes": len(body), "bytes": len(blob),
                  "min": [round(x, 6) for x in mn], "max": [round(x, 6) for x in mx]}


def unpack2(blob):
    """Inverse of pack2 for tests/previews: (verts m, normals, faces, rgb per tri)."""
    b = meshpack.unlzma_alone(blob) if blob[:1] == bytes([0x5D]) else meshpack.zlib.decompress(blob)
    if b[:4] != MAGIC2:
        raise ModelError("not an SVM2 payload")
    nv, nt = struct.unpack_from("<HH", b, 4)
    f = struct.unpack_from("<6f", b, 8)
    mn, mx = f[:3], f[3:]
    off = 32
    verts = [tuple(mn[k] + (mx[k] - mn[k]) * q / 65535 for k, q in enumerate(struct.unpack_from("<3H", b, off + 6 * i))) for i in range(nv)]
    off += 6 * nv
    faces = [struct.unpack_from("<3H", b, off + 6 * i) for i in range(nt)]
    off += 6 * nt
    cols = [struct.unpack_from("<3B", b, off + 3 * i) for i in range(nt)]
    off += 3 * nt
    normals = [tuple(x / 127 for x in struct.unpack_from("<3b", b, off + 3 * i)) for i in range(nv)]
    return verts, normals, faces, cols


# ----------------------------------------------------------------------------------------------------------------------- preview

def render(parts, view, size=(360, 480), ss=2, light=(0.35, -0.5, 0.8)):
    """Software rasteriser: flat/Lambert shading, z-buffer, supersampled. `parts` = [(verts, faces, rgb per tri)] in game metres.
    view = (azimuth_deg, elevation_deg): azimuth 0 looks at the front (+X face) from +X. Returns a PIL image (needs PIL) or the raw rows."""
    az, el = math.radians(view[0]), math.radians(view[1])
    pts = [p for v, _, _ in parts for p in v]
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    ctr = [(lo[i] + hi[i]) / 2 for i in range(3)]
    ce, se = math.cos(el), math.sin(el)

    def cam(p):                                                  # camera space: x right, y up, z toward viewer
        x, y, z = p[0] - ctr[0], p[1] - ctr[1], p[2] - ctr[2]
        # view from azimuth: the viewer sits at (cos az, sin az) horizontally; right = (-sin az, cos az)
        r = -x * math.sin(az) + y * math.cos(az)
        d = x * math.cos(az) + y * math.sin(az)                  # toward viewer horizontally
        return (r, z * ce - d * se, d * ce + z * se)

    W, H = size[0] * ss, size[1] * ss
    cp = [[cam(p) for p in v] for v, _, _ in parts]
    ext = max(max(c[0] for vs in cp for c in vs) - min(c[0] for vs in cp for c in vs),
              max(c[1] for vs in cp for c in vs) - min(c[1] for vs in cp for c in vs)) or 1.0
    sc = 0.9 * min(W, H * 1.0) / ext
    L = light
    ll = math.sqrt(_dot(L, L))
    L = (L[0] / ll, L[1] / ll, L[2] / ll)
    zb = [-1e9] * (W * H)
    img = [(120, 120, 128)] * (W * H)
    for (verts, faces, cols), vs in zip(parts, cp):
        sp = [(W / 2 + c[0] * sc, H / 2 - c[1] * sc, c[2]) for c in vs]
        for fi, (a, b, c) in enumerate(faces):
            A, B, Cc = sp[a], sp[b], sp[c]
            area = (B[0] - A[0]) * (Cc[1] - A[1]) - (B[1] - A[1]) * (Cc[0] - A[0])
            if area >= 0:                                        # back-facing in screen space (y flipped): CCW in game frame -> area < 0
                continue
            n = _area_normal(verts[a], verts[b], verts[c])
            nl = math.sqrt(_dot(n, n)) or 1.0
            lam = max(0.0, _dot((n[0] / nl, n[1] / nl, n[2] / nl), L))
            k = 0.35 + 0.65 * lam
            col = cols[fi]
            rgb = (int(col[0] * k), int(col[1] * k), int(col[2] * k))
            x0, x1 = max(0, int(min(A[0], B[0], Cc[0]))), min(W - 1, int(max(A[0], B[0], Cc[0])) + 1)
            y0, y1 = max(0, int(min(A[1], B[1], Cc[1]))), min(H - 1, int(max(A[1], B[1], Cc[1])) + 1)
            ia = 1.0 / area
            for y in range(y0, y1 + 1):
                py = y + 0.5
                for x in range(x0, x1 + 1):
                    px = x + 0.5
                    w0 = ((B[0] - px) * (Cc[1] - py) - (B[1] - py) * (Cc[0] - px)) * ia
                    w1 = ((Cc[0] - px) * (A[1] - py) - (Cc[1] - py) * (A[0] - px)) * ia
                    w2 = 1.0 - w0 - w1
                    if w0 < 0 or w1 < 0 or w2 < 0:
                        continue
                    z = w0 * A[2] + w1 * B[2] + w2 * Cc[2]
                    i = y * W + x
                    if z > zb[i]:
                        zb[i] = z
                        img[i] = rgb
    try:
        from PIL import Image
    except ImportError:
        return img, (W, H)
    im = Image.new("RGB", (W, H))
    im.putdata(img)
    return im.resize(size, Image.LANCZOS) if ss > 1 else im


# ----------------------------------------------------------------------------------------------------------------------- build

def build(kit_dir, out_dir=None, height_m=0.6, budgets=None, codec="lzma", progress=print):
    """Read the kit, decimate every part, assemble, pack. Returns (manifest, {role: blob}, {role: (verts, faces, cols, normals)}).
    height_m is the total figure height (floor to hat top); the can's size follows from the real proportions."""
    kit_dir = Path(kit_dir)
    kit = read_3mf(kit_dir / "Skippy_AMS_Coloured.3mf")
    fil = [hex_rgb(h) for h in kit["filaments"]]
    raw = {}
    for role, (stem, _) in PARTS.items():
        name = stem + ".stl"
        flat = read_stl(kit_dir / "Individual Parts" / name)
        st, unm = object_states(kit, name, flat)
        raw[role] = (flat, [fil[s - 1] for s in st], st)
        progress(f"read {name}: {len(flat) // 9} tris, states {dict(Counter(st))}, {unm} unmatched")
    # whole-figure box (source frame, mm) for scale and footprint centre
    lo = [1e9] * 3
    hi = [-1e9] * 3
    for flat, _, _ in raw.values():
        for i in range(0, len(flat), 3):
            for k in range(3):
                v = flat[i + k]
                if v < lo[k]:
                    lo[k] = v
                if v > hi[k]:
                    hi[k] = v
    # front: from the head's centre to its red (tongue) triangles, horizontal, source (x, z)
    hflat, hcols, hst = raw["head"]
    red = fil.index(max(fil, key=lambda c: c[0] - c[1] - c[2]))             # the reddest filament
    hc = [(min(hflat[i] for i in range(k, len(hflat), 3)) + max(hflat[i] for i in range(k, len(hflat), 3))) / 2 for k in range(3)]
    sx = sz = 0.0
    n_red = 0
    for t, s in enumerate(hst):
        if s - 1 == red:
            n_red += 1
            for v in range(3):
                sx += hflat[9 * t + 3 * v] - hc[0]
                sz += hflat[9 * t + 3 * v + 2] - hc[2]
    if not n_red:
        raise ModelError("no red tongue triangles: cannot orient the figure")
    to_game = frame_matrix((sx, sz))
    s = height_m / (hi[1] - lo[1])
    corners = [to_game((x, y, z)) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    fcx = (min(c[0] for c in corners) + max(c[0] for c in corners)) / 2
    fcy = (min(c[1] for c in corners) + max(c[1] for c in corners)) / 2

    def to_world(p):
        q = to_game(p)
        return ((q[0] - fcx) * s, (q[1] - fcy) * s, (p[1] - lo[1]) * s)

    blobs, geo, manifest_parts = {}, {}, []
    for role, (stem, budget) in PARTS.items():
        budget = (budgets or {}).get(role, budget)
        flat, cols, _ = raw[role]
        v, f, c = decimate(flat, cols, budget)
        v = [to_world(p) for p in v]
        v, nrm, f2 = shade(v, f)
        fcols = c
        blob, info = pack2(v, nrm, f2, fcols, codec)
        blobs[role] = blob
        geo[role] = (v, f2, fcols, nrm)
        palette = sorted(set(fcols))
        manifest_parts.append({"role": role, "source": stem + ".stl", "file": f"{role}.svm2", "budget": budget, **info,
                               "palette_rgb": [list(p) for p in palette], "origin": [0, 0, 0]})
        progress(f"{role}: {info['tris']} tris, {info['verts']} verts, {info['bytes']} B")
    # pivots (game metres)
    def part_pts(role):
        return geo[role][0]

    hv = part_pts("head")
    head_lo = min(p[2] for p in hv)
    hat = part_pts("hat")
    hand_roles = {r: part_pts(r) for r in ("hand_a", "hand_b")}
    ext = {r: max(p[2] for p in v) - min(p[2] for p in v) for r, v in hand_roles.items()}
    point = max(ext, key=lambda r: ext[r])
    rest = "hand_b" if point == "hand_a" else "hand_a"
    pv = hand_roles[point]
    tip = max(pv, key=lambda p: p[2])
    lowz = sorted(p[2] for p in pv)[max(1, len(pv) // 10)]
    low = [p for p in pv if p[2] <= lowz]
    wrist = tuple(sum(p[i] for p in low) / len(low) for i in range(3))

    def centre_xy(pts, zmax=None):
        sel = [p for p in pts if zmax is None or p[2] <= zmax]
        return (sum(p[0] for p in sel) / len(sel), sum(p[1] for p in sel) / len(sel))

    hx, hy = centre_xy(hv, head_lo + 0.1 * (max(p[2] for p in hv) - head_lo))
    head_hi = max(p[2] for p in hv)
    bx, by = centre_xy([p for p in hv if p[2] >= head_hi - 0.1 * (head_hi - head_lo)])
    can_v = part_pts("can")
    manifest = {
        "format": "SVM2", "units": "m", "frame": "x front, y left, z up, origin on the floor under the footprint centre of the whole figure",
        "scale_m_per_mm": s, "figure_height_m": round(max(p[2] for r in geo for p in geo[r][0]), 4),
        "can_height_m": round(max(p[2] for p in can_v) - min(p[2] for p in can_v), 4),
        "orientation_rule": "front = horizontal direction from the head's centre to its red (tongue) triangles",
        "colours": {"filaments": kit["filaments"], "source": "3MF project_settings.config filament_colour; per triangle from 3MF extruder + paint_color"},
        "parts": manifest_parts,
        "pivots": {
            "head": {"pivot": [round(hx, 4), round(hy, 4), round(head_lo, 4)], "note": "base of the head (neck), tilt/turn about this point"},
            "hat": {"pivot": [round(bx, 4), round(by, 4), round(head_hi, 4)],
                    "top": [round(sum(p[0] for p in hat) / len(hat), 4), round(sum(p[1] for p in hat) / len(hat), 4), round(max(p[2] for p in hat), 4)],
                    "note": "pivot = top centre of the head (the hat sits on it), tilt about this; top = the hat's highest point"},
            "hand_point": {"part": point, "wrist": [round(x, 4) for x in wrist], "tip": [round(x, 4) for x in tip],
                           "note": "the raised, pointing hand: rotate about wrist; the finger tip is the highest vertex"},
            "hand_rest": {"part": rest},
        },
    }
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for role, blob in blobs.items():
            (out / f"{role}.svm2").write_bytes(blob)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest, blobs, geo


def preview(geo, out_dir, views=(("front", (0, 8)), ("three_quarter", (35, 15)), ("side", (90, 8)), ("back", (180, 8)))):
    out = Path(out_dir)
    parts = [(v, f, c) for v, f, c, _ in geo.values()]
    paths = []
    for name, vw in views:
        im = render(parts, vw)
        p = out / f"preview_{name}.png"
        im.save(p)
        paths.append(str(p))
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("kit")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "private" / "skippy"))
    ap.add_argument("--height", type=float, default=0.6)
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args(argv)
    manifest, _, geo = build(a.kit, a.out, a.height)
    tot = sum(p["tris"] for p in manifest["parts"])
    print(f"total {tot} tris, {sum(p['bytes'] for p in manifest['parts'])} B; figure {manifest['figure_height_m']} m, can {manifest['can_height_m']} m")
    if a.preview:
        print(preview(geo, a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
