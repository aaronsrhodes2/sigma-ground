"""Surface for the found-object payload (SVM3): colour from textures without shipping them, material cues, palette, smooth normals.

Pure Python plus (optionally) Pillow, which only decodes the texture at pack time (pip install pillow, official PyPI). Nothing here touches
the network. Without Pillow a textured material falls back to its base colour factor, exactly the old behaviour.

    Texture(bytes)               a decoded, box-reduced base-colour / metal-rough image; .sample(u, v) -> (r, g, b, a) 0..255
    tri_colour(...)              one display colour for a triangle: base factor (linear) x texture at the UV centroid
    pbr_cue(metal, rough, emit)  (spec, metal, emit, alpha) bytes: what the client turns into a gloss, a metal look and a glow
    build_palette(items, n)      colour regions: <= n (r, g, b, spec, metal, emit) entries and an entry index per triangle
    smooth_normals(v, f)         vertex normals with creases split (a box edge stays sharp), the way Skippy's body does it

What is LOST, said once: the texture's fine detail (a label, a grain, a logo) is averaged away; each triangle gets ONE flat colour, and the
palette snaps those to a few regions. Large smooth colour areas, gradients of tone and the main material split survive; text and
decals do not.
"""
from __future__ import annotations

import io
import math
import os

GAMMA = 2.2
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tga", ".bmp", ".webp")
LIN = [(i / 255.0) ** GAMMA for i in range(256)]


def to_display(lin):
    """Linear 0..1 -> display byte."""
    return max(0, min(255, int(round(255.0 * max(0.0, lin) ** (1.0 / GAMMA)))))


def have_pillow() -> bool:
    try:
        import PIL  # noqa: F401
        return True
    except Exception:
        return False


class Texture:
    """An image reduced to at most `max_side` pixels (box filter: the reduction IS the averaging), sampled bilinearly with repeat wrap."""

    def __init__(self, data: bytes, max_side: int = 256):
        from PIL import Image
        im = Image.open(io.BytesIO(data))
        im.load()
        im = im.convert("RGBA")
        k = max(1, max(im.size) // max_side)
        if k > 1:
            im = im.reduce(k)
        self.w, self.h = im.size
        self.px = im.tobytes()

    def mean(self):
        n = self.w * self.h
        s = [0, 0, 0]
        for i in range(0, 4 * n, 4):
            s[0] += self.px[i]
            s[1] += self.px[i + 1]
            s[2] += self.px[i + 2]
        return tuple(x / n for x in s)

    def sample(self, u: float, v: float, flip_v: bool = False):
        """Bilinear, wrapping. `v` runs down the image (glTF); `flip_v` for Collada/OBJ, where v runs up."""
        w, h = self.w, self.h
        x = (u % 1.0) * w - 0.5
        y = ((1.0 - v if flip_v else v) % 1.0) * h - 0.5
        x0, y0 = math.floor(x), math.floor(y)
        fx, fy = x - x0, y - y0
        x1, y1 = (x0 + 1) % w, (y0 + 1) % h
        x0, y0 = x0 % w, y0 % h
        px = self.px
        i00, i10, i01, i11 = 4 * (y0 * w + x0), 4 * (y0 * w + x1), 4 * (y1 * w + x0), 4 * (y1 * w + x1)
        w00, w10, w01, w11 = (1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy
        return tuple(px[i00 + c] * w00 + px[i10 + c] * w10 + px[i01 + c] * w01 + px[i11 + c] * w11 for c in range(4))


def load_texture(data):
    """A Texture, or None when there is no data, no Pillow, or the image cannot be decoded (never raises). SKIPPY_MESH_LEGACY=1 (the original
    flat-colour payload, used for the "before" pictures) reads no textures at all."""
    if not data or os.environ.get("SKIPPY_MESH_LEGACY") == "1":
        return None
    try:
        return Texture(bytes(data))
    except Exception:
        return None


def tri_colour(factor_lin, tex, uvs, flip_v=False):
    """Display (r, g, b) for one triangle: the base colour factor (linear, 3 numbers) times the texture's colour at the centroid of `uvs`
    (three (u, v) pairs; one sample from the reduced image, or four when the triangle is large in UV space). No texture: the factor alone."""
    if tex is None or not uvs:
        return tuple(to_display(c) for c in factor_lin[:3])
    cu = (uvs[0][0] + uvs[1][0] + uvs[2][0]) / 3.0
    cv = (uvs[0][1] + uvs[1][1] + uvs[2][1]) / 3.0
    pts = [(cu, cv)]
    span = max(abs(uvs[0][0] - uvs[1][0]), abs(uvs[1][0] - uvs[2][0]), abs(uvs[0][1] - uvs[1][1]), abs(uvs[1][1] - uvs[2][1]))
    if span * max(tex.w, tex.h) > 6 and span < 0.5:                  # a triangle that covers several texels: average a few points
        for a, b, c in ((0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6)):
            pts.append((a * uvs[0][0] + b * uvs[1][0] + c * uvs[2][0], a * uvs[0][1] + b * uvs[1][1] + c * uvs[2][1]))
    acc = [0.0, 0.0, 0.0]
    for u, v in pts:
        s = tex.sample(u, v, flip_v)
        for k in range(3):
            acc[k] += LIN[min(255, int(s[k] + 0.5))]
    return tuple(to_display(factor_lin[k] * acc[k] / len(pts)) for k in range(3))


# ----------------------------------------------------------------------------------------------------------------- material cues
def pbr_cue(metal, rough, emit=0.0, alpha=255):
    """glTF metallic / roughness factors (0..1; None = unknown) -> (spec, metal, emit, alpha) bytes.
    spec is the strength of the highlight (a polished surface high, a rough one low); metal says how much the colour tints the highlight and
    how little diffuse the surface has; emit (0..1 luminance of the emissive colour) makes it glow. Unknown metal counts as 0 (a model that
    does not say is almost never a bare metal), unknown roughness as 0.6."""
    m = 0.0 if metal is None else max(0.0, min(1.0, metal))
    r = 0.6 if rough is None else max(0.0, min(1.0, rough))
    spec = (1.0 - r) ** 1.5
    return (int(round(255 * spec)), int(round(255 * m)), int(round(255 * max(0.0, min(1.0, emit)))), alpha)


DEFAULT_CUE = (40, 0, 0, 255)


def lum255(rgb):
    return (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255.0


# ----------------------------------------------------------------------------------------------------------------- palette
def _class_of(cue):
    return (min(4, cue[0] * 5 // 256), min(2, cue[1] * 3 // 256), 1 if cue[2] > 20 else 0, 1 if cue[3] < 250 else 0)


def _median_cut(pts, n):
    """pts: [(r, g, b, w, idx)]. -> list of lists of pts (<= n)."""
    boxes = [pts]
    while len(boxes) < n:
        best, bi, bscore = None, -1, 0.0
        for i, b in enumerate(boxes):
            if len(b) < 2:
                continue
            for ch in range(3):
                lo = min(p[ch] for p in b)
                hi = max(p[ch] for p in b)
                score = (hi - lo) * math.sqrt(sum(p[3] for p in b))
                if score > bscore:
                    best, bi, bscore = ch, i, score
        if best is None or bscore < 6:
            break
        b = sorted(boxes[bi], key=lambda p: p[best])
        half = sum(p[3] for p in b) / 2.0
        acc, cut = 0.0, 1
        for k, p in enumerate(b):
            acc += p[3]
            if acc >= half:
                cut = max(1, min(len(b) - 1, k + 1))
                break
        boxes[bi:bi + 1] = [b[:cut], b[cut:]]
    return boxes


def build_palette(items, max_n=24):
    """items: [(rgb, cue, weight)] per triangle -> (palette [(r, g, b, spec, metal, emit, alpha)], index per triangle).
    Triangles are first split by material class (spec level, metal level, glows or not), then each class is cut into colour regions with the
    median-cut method, the entries shared out by area; every class keeps at least one entry (up to max_n classes; extra classes merge into
    the nearest by spec)."""
    classes = {}
    for i, (rgb, cue, w) in enumerate(items):
        classes.setdefault(_class_of(cue), []).append((rgb[0], rgb[1], rgb[2], max(w, 1e-12), i))
    keys = sorted(classes, key=lambda k: -sum(p[3] for p in classes[k]))
    if len(keys) > max_n:
        keep = keys[:max_n]
        for k in keys[max_n:]:
            tgt = min(keep, key=lambda q: abs(q[0] - k[0]) + abs(q[1] - k[1]) + 3 * abs(q[2] - k[2]) + 5 * abs(q[3] - k[3]))
            classes[tgt] += classes.pop(k)
        keys = keep
    total = sum(sum(p[3] for p in classes[k]) for k in keys) or 1.0
    spare = max_n - len(keys)
    quota = {}
    for k in keys:
        share = sum(p[3] for p in classes[k]) / total
        quota[k] = 1 + int(spare * share + 0.5)
    while sum(quota.values()) > max_n:
        k = max(keys, key=lambda q: quota[q])
        quota[k] -= 1
    palette, index = [], [0] * len(items)
    for k in keys:
        for box in _median_cut(classes[k], quota[k]):
            wsum = sum(p[3] for p in box)
            r = sum(p[0] * p[3] for p in box) / wsum
            g = sum(p[1] * p[3] for p in box) / wsum
            b = sum(p[2] * p[3] for p in box) / wsum
            cue = [sum(items[p[4]][1][c] * p[3] for p in box) / wsum for c in range(4)]
            for p in box:
                index[p[4]] = len(palette)
            palette.append((int(round(r)), int(round(g)), int(round(b)), int(round(cue[0])), int(round(cue[1])), int(round(cue[2])), int(round(cue[3]))))
    return palette, index


# ----------------------------------------------------------------------------------------------------------------- normals
def smooth_normals(verts, faces, crease_deg=40.0):
    """Vertex normals for a triangle soup that shares vertices: the faces around a vertex are grouped by direction (within `crease_deg` of the
    group's first face) and each group gets its own copy of the vertex with the area-weighted mean normal, so a smooth curve is shaded
    smooth and a box edge stays sharp. Returns (positions, unit normals, faces) with the vertices duplicated where needed."""
    cos_c = math.cos(math.radians(crease_deg))
    fn = []
    for a, b, c in faces:
        p, q, r = verts[a], verts[b], verts[c]
        ux, uy, uz = q[0] - p[0], q[1] - p[1], q[2] - p[2]
        vx, vy, vz = r[0] - p[0], r[1] - p[1], r[2] - p[2]
        n = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
        fn.append(n)
    incident = {}
    for fi, f in enumerate(faces):
        for k in range(3):
            incident.setdefault(f[k], []).append((fi, k))
    out_v, out_n, out_f = [], [], [list(f) for f in faces]
    for vi, inc in incident.items():
        groups = []                                                  # [unit of the first face, sum normal, [(fi, k)]]
        for fi, k in inc:
            n = fn[fi]
            ln = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
            if ln < 1e-18:
                u = (0.0, 0.0, 0.0)
            else:
                u = (n[0] / ln, n[1] / ln, n[2] / ln)
            for g in groups:
                if u == (0.0, 0.0, 0.0) or g[0] == (0.0, 0.0, 0.0) or g[0][0] * u[0] + g[0][1] * u[1] + g[0][2] * u[2] >= cos_c:
                    g[1][0] += n[0]
                    g[1][1] += n[1]
                    g[1][2] += n[2]
                    g[2].append((fi, k))
                    break
            else:
                groups.append([u, list(n), [(fi, k)]])
        for g in groups:
            s = g[1]
            ln = math.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2) or 1.0
            new = len(out_v)
            out_v.append(verts[vi])
            out_n.append((s[0] / ln, s[1] / ln, s[2] / ln) if ln > 1e-18 else (0.0, 0.0, 1.0))
            for fi, k in g[2]:
                out_f[fi][k] = new
    return out_v, out_n, [tuple(f) for f in out_f]
