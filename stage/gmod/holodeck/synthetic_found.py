"""A made-up found object for proving the mesh path in the real game: "chroma pillar", a 0.4 x 0.2 x 0.8 m block whose six faces are six
different colours, served as if by Poly Haven (offline: nothing is downloaded). Real Poly Haven models are textured, and the textures are
not carried yet, so they come out grey; this one has a base colour per face, which shows colour, axis and winding in one picture.

Frame of the glTF (y up): +X red, -X cyan, +Y (top) green, -Y (bottom) magenta, +Z blue, -Z yellow. In the game, after the
payload's frame and the mesh's quarter turn, the object's local +x (its "front") is the BLUE face, local +y the RED one.

    python stage/gmod/holodeck/shots.py "add a chroma pillar" --tag pillar --start-bridge --synthetic --two-phase --outline
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from skippyverse import cards as C  # noqa: E402

NAME = "chroma pillar"
PID = "chroma_pillar"
SIZE = (0.4, 0.2, 0.8)             # glTF x (card w), glTF z (card d), glTF y (card h)
FACES = [("+x", (1, 0, 0)), ("-x", (0, 1, 1)), ("+y", (0, 1, 0)), ("-y", (1, 0, 1)), ("+z", (0, 0, 1)), ("-z", (1, 1, 0))]


def _face_corners(axis: str):
    """Four corners of one face of the block, in glTF coordinates, ordered counter-clockwise seen from outside."""
    ai, sign = "xyz".index(axis[1]), 1 if axis[0] == "+" else -1
    half = (SIZE[0] / 2, SIZE[2] / 2, SIZE[1] / 2)           # x, y (height), z (depth) half extents
    centre_y = SIZE[2] / 2
    u, v = [i for i in range(3) if i != ai]
    pts = []
    for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        p = [0.0, 0.0, 0.0]
        p[ai] = sign * half[ai]
        p[u], p[v] = su * half[u], sv * half[v]
        p[1] += centre_y
        pts.append(tuple(p))
    a, b, c = pts[0], pts[1], pts[2]
    cr = ((b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]), (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
          (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    if cr[ai] * sign < 0:
        pts.reverse()
    return pts


def routes():
    pos, idx, prims, mats, off = b"", b"", [], [], 0
    accessors, views = [], []
    for k, (axis, rgb) in enumerate(FACES):
        pts = _face_corners(axis)
        pb = b"".join(struct.pack("<3f", *p) for p in pts)
        ib = struct.pack("<6H", 0, 1, 2, 0, 2, 3)
        lo = [min(p[i] for p in pts) for i in range(3)]
        hi = [max(p[i] for p in pts) for i in range(3)]
        views.append({"buffer": 0, "byteOffset": len(pos), "byteLength": len(pb)})
        views.append({"buffer": 0, "byteOffset": 0, "byteLength": len(ib)})        # patched below
        pos += pb
        accessors.append({"bufferView": 2 * k, "componentType": 5126, "count": 4, "type": "VEC3", "min": lo, "max": hi})
        accessors.append({"bufferView": 2 * k + 1, "componentType": 5123, "count": 6, "type": "SCALAR"})
        prims.append({"attributes": {"POSITION": 2 * k}, "indices": 2 * k + 1, "material": k})
        mats.append({"pbrMetallicRoughness": {"baseColorFactor": [*rgb, 1.0]}})
        idx += ib
    base = len(pos)
    for k in range(len(FACES)):
        views[2 * k + 1]["byteOffset"] = base + 12 * k
    blob = pos + idx
    doc = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(blob), "uri": f"{PID}.bin"}], "bufferViews": views, "accessors": accessors,
           "materials": mats, "meshes": [{"primitives": prims}], "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}]}
    gl = json.dumps(doc).encode()
    files = {"gltf": {"1k": {"gltf": {"url": f"https://dl.example/{PID}_1k.gltf", "size": len(gl), "include": {
        f"{PID}.bin": {"url": f"https://dl.example/{PID}.bin", "size": len(blob)}}}}}}
    return {f"https://api.polyhaven.com/files/{PID}": json.dumps(files).encode(), f"https://dl.example/{PID}_1k.gltf": gl,
            f"https://dl.example/{PID}.bin": blob}


class Fetch:
    def __init__(self):
        self.r = routes()

    def __call__(self, url, *, max_bytes):
        return self.r[url]


def card():
    w, d, h = SIZE
    return C.make_card(id=f"found.poly-haven.{C.slug(NAME)}", name=NAME,
                       size={"w": w, "d": d, "h": h, "basis": "sourced", "source": "synthetic"}, category="misc",
                       visual={"class": "found", "style": "photoscan-or-pbr", "blocky": False, "lods": []},
                       provenance={"source": "Poly Haven", "license": "CC0", "author": "synthetic", "url": f"https://polyhaven.com/a/{PID}"},
                       mass=C.prior_mass({"w": w, "d": d, "h": h, "basis": "sourced", "source": "synthetic"}), flags=["front_unknown"])
