"""Bounding boxes of mesh files, in the file's own units, pure standard library. Geometry only, no physics, no network.

Formats: glTF/GLB (node transforms: matrix or translation/rotation/scale, applied properly), Wavefront OBJ, STL (binary and ascii),
Collada .dae (float_array positions, node transforms, <unit meter=...>, <up_axis>).

Everything is reported in ONE canonical frame: +Z up (glTF and Collada Y_UP/X_UP files are rotated into it), units multiplied by the
file's declared unit when it has one. Where the format declares no unit (OBJ, STL) the numbers stay in file units and
`unit_known` is False: the caller decides whether the unit is known from somewhere else (an SDF, a convention) or guessed.

    m = measure_file(path)            # -> Measurement
    m.extent                          # (dx, dy, dz) in the canonical frame
    m.size_whd()                      # (w, d, h) = (dx, dy, dz): the card convention
    m.normalise_matrix()              # 4x4 (row-major, 16 floats): source-world coords -> floor under footprint centre, +Z up
"""
from __future__ import annotations

import json
import math
import re
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

Matrix = list  # 16 floats, row-major 4x4


class MeasureError(ValueError):
    """The file cannot be measured (unsupported feature, truncated, empty)."""


# ----------------------------------------------------------------------------------------------------------------- matrices
IDENT: Matrix = [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]


def mat_mul(a: Matrix, b: Matrix) -> Matrix:
    return [sum(a[4 * r + k] * b[4 * k + c] for k in range(4)) for r in range(4) for c in range(4)]


def mat_apply(m: Matrix, p) -> tuple:
    x, y, z = p
    return (m[0] * x + m[1] * y + m[2] * z + m[3],
            m[4] * x + m[5] * y + m[6] * z + m[7],
            m[8] * x + m[9] * y + m[10] * z + m[11])


def mat_translate(t) -> Matrix:
    return [1.0, 0, 0, t[0], 0, 1.0, 0, t[1], 0, 0, 1.0, t[2], 0, 0, 0, 1.0]


def mat_scale(s) -> Matrix:
    return [s[0], 0, 0, 0, 0, s[1], 0, 0, 0, 0, s[2], 0, 0, 0, 0, 1.0]


def mat_quat(q) -> Matrix:
    """Unit-normalised quaternion (x, y, z, w) -> rotation matrix."""
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0,
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0,
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0,
            0, 0, 0, 1.0]


def mat_axis_angle(axis, deg: float) -> Matrix:
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / n, y / n, z / n
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    t = 1 - c
    return [t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0,
            t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0,
            t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0,
            0, 0, 0, 1.0]


def mat_rpy(roll: float, pitch: float, yaw: float) -> Matrix:
    """Radians; R = Rz(yaw) Ry(pitch) Rx(roll), the SDF convention."""
    ex = mat_axis_angle((1, 0, 0), math.degrees(roll))
    ey = mat_axis_angle((0, 1, 0), math.degrees(pitch))
    ez = mat_axis_angle((0, 0, 1), math.degrees(yaw))
    return mat_mul(ez, mat_mul(ey, ex))


def axis_aligned(m: Matrix, tol: float = 1e-9) -> bool:
    """True when the linear part maps each axis onto an axis (a scale, mirror, or multiple-of-90-degree rotation): the transformed
    bounding box of a box is then exactly the box of the transformed corners."""
    for r in range(3):
        if sum(abs(m[4 * r + c]) > tol for c in range(3)) != 1:
            return False
    for c in range(3):
        if sum(abs(m[4 * r + c]) > tol for r in range(3)) != 1:
            return False
    return True


# the rotations that take a file's up axis to +Z
UP_TO_Z = {
    "z": IDENT,
    "y": [1.0, 0, 0, 0, 0, 0, -1.0, 0, 0, 1.0, 0, 0, 0, 0, 0, 1.0],       # (x, y, z) -> (x, -z, y)
    "x": [0, 1.0, 0, 0, 0, 0, 1.0, 0, 1.0, 0, 0, 0, 0, 0, 0, 1.0],         # (x, y, z) -> (y, z, x)
}


# ----------------------------------------------------------------------------------------------------------------- the box
class Box:
    """An accumulating axis-aligned bounding box."""

    def __init__(self):
        self.lo = [math.inf] * 3
        self.hi = [-math.inf] * 3
        self.n = 0

    def add(self, p) -> None:
        for i in range(3):
            v = p[i]
            if v < self.lo[i]:
                self.lo[i] = v
            if v > self.hi[i]:
                self.hi[i] = v
        self.n += 1

    def add_points(self, m: Matrix, pts) -> None:
        for p in pts:
            self.add(mat_apply(m, p))

    def add_corners(self, m: Matrix, lo, hi) -> None:
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                for z in (lo[2], hi[2]):
                    self.add(mat_apply(m, (x, y, z)))

    def union(self, other: "Box") -> None:
        if other.n:
            self.add(other.lo)
            self.add(other.hi)
            self.n += other.n - 2

    def empty(self) -> bool:
        return self.n == 0


class Measurement:
    """Result of measuring one file (or a union of files). Box is in the canonical frame: +Z up, metres if `unit_known`."""

    def __init__(self, box: Box, *, fmt: str, up_in: str | None, unit_m: float | None, unit_known: bool, canon: Matrix, notes=()):
        if box.empty():
            raise MeasureError("no geometry found")
        self.box, self.fmt, self.up_in, self.unit_m, self.unit_known, self.canon = box, fmt, up_in, unit_m, unit_known, canon
        self.notes = list(notes)

    @property
    def mins(self):
        return tuple(self.box.lo)

    @property
    def maxs(self):
        return tuple(self.box.hi)

    @property
    def extent(self):
        return tuple(h - l for l, h in zip(self.box.lo, self.box.hi))

    def size_whd(self):
        dx, dy, dz = self.extent
        return dx, dy, dz

    def normalise_matrix(self) -> Matrix:
        """Source-world coordinates -> origin on the floor under the footprint centre: canon first, then translate."""
        cx = (self.box.lo[0] + self.box.hi[0]) / 2
        cy = (self.box.lo[1] + self.box.hi[1]) / 2
        return mat_mul(mat_translate((-cx, -cy, -self.box.lo[2])), self.canon)

    def to_dict(self) -> dict:
        dx, dy, dz = self.extent
        return {"format": self.fmt, "up_in": self.up_in, "unit_m": self.unit_m, "unit_known": self.unit_known,
                "extent": [dx, dy, dz], "mins": list(self.mins), "maxs": list(self.maxs), "size_whd": [dx, dy, dz],
                "normalise_matrix": self.normalise_matrix(), "notes": self.notes}


def _canon(up: str, unit_m: float | None) -> Matrix:
    s = unit_m if unit_m else 1.0
    return mat_mul(UP_TO_Z[up], mat_scale((s, s, s)))


# ----------------------------------------------------------------------------------------------------------------- glTF / GLB
_GLB_MAGIC = 0x46546C67
_COMP = {5126: ("f", 4)}


def _load_gltf(path: Path) -> tuple[dict, list[bytes | None]]:
    raw = path.read_bytes()
    base = path.parent
    glb_bin = None
    if raw[:4] == b"glTF":
        magic, version, length = struct.unpack_from("<III", raw, 0)
        if magic != _GLB_MAGIC or version != 2:
            raise MeasureError("not a glTF 2 binary")
        off, doc = 12, None
        while off + 8 <= min(len(raw), length):
            clen, ctype = struct.unpack_from("<II", raw, off)
            chunk = raw[off + 8: off + 8 + clen]
            if ctype == 0x4E4F534A:
                doc = json.loads(chunk.decode("utf-8"))
            elif ctype == 0x004E4942 and glb_bin is None:
                glb_bin = chunk
            off += 8 + clen + (-clen % 4)
        if doc is None:
            raise MeasureError("GLB has no JSON chunk")
    else:
        doc = json.loads(raw.decode("utf-8"))
    buffers: list[bytes | None] = []
    for i, b in enumerate(doc.get("buffers", [])):
        uri = b.get("uri")
        if uri is None:
            buffers.append(glb_bin)
        elif uri.startswith("data:"):
            import base64
            buffers.append(base64.b64decode(uri.split(",", 1)[1]))
        else:
            p = base / _unquote(uri)
            buffers.append(p.read_bytes() if p.is_file() else None)
    return doc, buffers


# ---------------------------------------------------------------------------------------------------------------- Draco (KHR_draco_mesh_compression)
# Most of NASA's glb files store each primitive's geometry as a Draco buffer. The decoder is the official Apache-2.0 PyPI package DracoPy
# (Google's Draco library; `pip install DracoPy`). It is optional: without it this module imports and works, and a Draco model stays a box.
DRACO_EXT = "KHR_draco_mesh_compression"
_DRACO_CACHE: dict = {}


def draco_available() -> bool:
    try:
        import DracoPy  # noqa: F401
        return True
    except Exception:
        return False


def draco_ext(prim: dict):
    """The primitive's KHR_draco_mesh_compression extension dict, or None."""
    return (prim.get("extensions") or {}).get(DRACO_EXT)


def decode_draco_primitive(doc: dict, buffers: list, prim: dict):
    """([(x, y, z)], [i0, i1, i2, ...]) for one Draco primitive (positions already dequantised by the decoder, whatever the quantisation).
    Raises MeasureError when the decoder is absent, the extension is malformed or the buffer is corrupt: never anything else."""
    ext = draco_ext(prim)
    if not isinstance(ext, dict) or not isinstance(ext.get("bufferView"), int):
        raise MeasureError("Draco primitive has no usable bufferView")
    try:
        import DracoPy
    except Exception:
        raise MeasureError("its geometry is Draco-compressed and the Draco decoder (pip install DracoPy) is not installed") from None
    try:
        bv = doc["bufferViews"][ext["bufferView"]]
        buf = buffers[bv["buffer"]]
        if buf is None:
            raise MeasureError("Draco buffer is missing")
        start = bv.get("byteOffset", 0)
        data = bytes(buf[start: start + bv["byteLength"]])
        if len(data) != bv["byteLength"]:
            raise MeasureError("Draco buffer runs past the end of its buffer")
        key = (len(data), zlib.crc32(data))
        hit = _DRACO_CACHE.get(key)
        if hit is None:
            m = DracoPy.decode(data)
            pts = m.points
            faces = m.faces
            if pts is None or len(pts) == 0 or pts.ndim != 2 or pts.shape[1] != 3:
                raise MeasureError("Draco buffer holds no 3D positions")
            n = len(pts)
            flat = [] if faces is None else [int(i) for i in faces.reshape(-1)]
            if len(flat) % 3 or (flat and (min(flat) < 0 or max(flat) >= n)):
                raise MeasureError("Draco faces are malformed")
            hit = ([(float(a), float(b), float(c)) for a, b, c in pts.tolist()], flat)
            if len(_DRACO_CACHE) >= 6:
                _DRACO_CACHE.pop(next(iter(_DRACO_CACHE)))
            _DRACO_CACHE[key] = hit
        return hit
    except MeasureError:
        raise
    except Exception as e:                                       # DracoPy raises its own exception types on a bad buffer
        raise MeasureError(f"Draco buffer is corrupt ({type(e).__name__}: {e})") from None


def _accessor_points(doc: dict, buffers: list, idx: int):
    """(points or None, min or None, max or None) for a VEC3 float POSITION accessor."""
    acc = doc["accessors"][idx]
    amin, amax = acc.get("min"), acc.get("max")
    if acc.get("type") != "VEC3":
        raise MeasureError("POSITION accessor is not VEC3")
    if acc.get("componentType") not in _COMP or acc.get("sparse"):
        return None, amin, amax
    bv_i = acc.get("bufferView")
    if bv_i is None:
        return None, amin, amax
    bv = doc["bufferViews"][bv_i]
    buf = buffers[bv["buffer"]] if bv["buffer"] < len(buffers) else None
    if buf is None:
        return None, amin, amax
    count = acc["count"]
    stride = bv.get("byteStride") or 12
    start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    if start + (count - 1) * stride + 12 > len(buf) if count else False:
        raise MeasureError("accessor runs past the end of its buffer")
    if stride == 12:
        flat = struct.unpack_from(f"<{3 * count}f", buf, start)
        return [(flat[i], flat[i + 1], flat[i + 2]) for i in range(0, 3 * count, 3)], amin, amax
    return [struct.unpack_from("<3f", buf, start + i * stride) for i in range(count)], amin, amax


def measure_gltf(path, *, post: Matrix = IDENT) -> Measurement:
    path = Path(path)
    doc, buffers = _load_gltf(path)
    nodes = doc.get("nodes", [])
    scenes = doc.get("scenes", [])
    if scenes:
        roots = scenes[doc.get("scene", 0)].get("nodes", [])
    else:
        children = {c for n in nodes for c in n.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in children]
    canon = _canon("y", 1.0)                                  # glTF is y-up and in metres by specification
    total = mat_mul(post, canon)
    box = Box()
    notes: list[str] = []

    def local(n: dict) -> Matrix:
        if "matrix" in n:
            c = n["matrix"]                                    # column-major
            return [c[0], c[4], c[8], c[12], c[1], c[5], c[9], c[13], c[2], c[6], c[10], c[14], c[3], c[7], c[11], c[15]]
        m = mat_translate(n.get("translation", (0, 0, 0)))
        m = mat_mul(m, mat_quat(n.get("rotation", (0, 0, 0, 1))))
        return mat_mul(m, mat_scale(n.get("scale", (1, 1, 1))))

    def visit(i: int, parent: Matrix, depth: int = 0) -> None:
        if depth > 64:
            raise MeasureError("glTF node hierarchy too deep or cyclic")
        n = nodes[i]
        world = mat_mul(parent, local(n))
        if "mesh" in n:
            full = mat_mul(total, world)
            for prim in doc["meshes"][n["mesh"]].get("primitives", []):
                pos = prim.get("attributes", {}).get("POSITION")
                if pos is None:
                    continue
                if draco_ext(prim) is not None and draco_available():
                    try:
                        dpts, _ = decode_draco_primitive(doc, buffers, prim)
                        box.add_points(full, dpts)
                        continue
                    except MeasureError as e:                    # corrupt Draco: fall back to the accessor min/max, say so
                        notes.append(str(e))
                pts, amin, amax = _accessor_points(doc, buffers, pos)
                if pts is not None and not (amin and amax and axis_aligned(full)):
                    box.add_points(full, pts)
                elif amin and amax:
                    box.add_corners(full, amin, amax)
                    if pts is None:
                        notes.append("some vertex data unavailable: used the accessor min/max")
                else:
                    raise MeasureError("POSITION data and min/max both unavailable")
        for c in n.get("children", []):
            visit(c, world, depth + 1)

    for r in roots:
        visit(r, IDENT)
    return Measurement(box, fmt="gltf", up_in="y", unit_m=1.0, unit_known=True, canon=total, notes=notes)


# ----------------------------------------------------------------------------------------------------------------- OBJ / STL
def measure_obj(path, *, post: Matrix = IDENT) -> Measurement:
    box = Box()
    total = mat_mul(post, _canon("z", None))
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                try:
                    box.add(mat_apply(total, (float(parts[1]), float(parts[2]), float(parts[3]))))
                except (IndexError, ValueError):
                    raise MeasureError(f"bad OBJ vertex line: {line[:40]!r}") from None
    return Measurement(box, fmt="obj", up_in=None, unit_m=None, unit_known=False, canon=total,
                       notes=["OBJ declares no unit and no up axis: +Z up assumed"])


def measure_stl(path, *, post: Matrix = IDENT) -> Measurement:
    raw = Path(path).read_bytes()
    total = mat_mul(post, _canon("z", None))
    box = Box()
    binary = False
    if len(raw) >= 84:
        n = struct.unpack_from("<I", raw, 80)[0]
        binary = len(raw) == 84 + 50 * n
    if binary:
        for i in range(n):
            v = struct.unpack_from("<12f", raw, 84 + 50 * i)       # normal (3) + three vertices (9)
            for k in (3, 6, 9):
                box.add(mat_apply(total, v[k:k + 3]))
    else:
        for m in re.finditer(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", raw):
            box.add(mat_apply(total, tuple(float(g) for g in m.groups())))
    return Measurement(box, fmt="stl", up_in=None, unit_m=None, unit_known=False, canon=total,
                       notes=["STL declares no unit and no up axis: +Z up assumed"])


# ----------------------------------------------------------------------------------------------------------------- Collada
def _unquote(s: str) -> str:
    """%XX decoding of a relative file reference (no network module needed)."""
    return re.sub(r"(?:%[0-9A-Fa-f]{2})+", lambda m: bytes.fromhex(m.group(0).replace("%", "")).decode("utf-8", "replace"), s)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _floats(text: str | None) -> list[float]:
    return [float(t) for t in (text or "").split()]


def measure_dae(path, *, post: Matrix = IDENT) -> Measurement:
    try:
        root = ET.parse(str(path)).getroot()
    except ET.ParseError as e:
        raise MeasureError(f"bad Collada XML: {e}") from None
    unit_m, up = 1.0, "y"                                      # Collada defaults: 1 metre, Y_UP
    for el in root.iter():
        t = _local(el.tag)
        if t == "unit" and el.get("meter"):
            unit_m = float(el.get("meter"))
        elif t == "up_axis" and el.text:
            up = el.text.strip()[0].lower()
    if up not in UP_TO_Z:
        raise MeasureError(f"unknown up_axis {up!r}")
    total = mat_mul(post, _canon(up, unit_m))

    sources: dict[str, tuple[list[float], int]] = {}
    for el in root.iter():
        if _local(el.tag) == "source":
            fa = next((c for c in el.iter() if _local(c.tag) == "float_array"), None)
            stride = 3
            acc = next((c for c in el.iter() if _local(c.tag) == "accessor"), None)
            if acc is not None and acc.get("stride"):
                stride = int(acc.get("stride"))
            if fa is not None:
                sources[el.get("id")] = (_floats(fa.text), stride)
    geoms: dict[str, list[tuple[float, float, float]]] = {}
    for g in root.iter():
        if _local(g.tag) != "geometry":
            continue
        pts: list = []
        for mesh in (c for c in g if _local(c.tag) == "mesh"):
            for vert in (c for c in mesh if _local(c.tag) == "vertices"):
                for inp in (c for c in vert if _local(c.tag) == "input" and c.get("semantic") == "POSITION"):
                    vals, stride = sources.get(inp.get("source", "").lstrip("#"), ([], 3))
                    pts += [(vals[i], vals[i + 1], vals[i + 2]) for i in range(0, len(vals) - 2, stride)]
        geoms[g.get("id")] = pts

    def node_matrix(node) -> Matrix:
        m = IDENT
        for c in node:
            t = _local(c.tag)
            v = _floats(c.text)
            if t == "matrix":
                if len(v) != 16:
                    raise MeasureError("Collada matrix needs 16 numbers")
                m = mat_mul(m, v)
            elif t == "translate":
                m = mat_mul(m, mat_translate(v[:3]))
            elif t == "rotate":
                m = mat_mul(m, mat_axis_angle(v[:3], v[3]))
            elif t == "scale":
                m = mat_mul(m, mat_scale(v[:3]))
            elif t in ("lookat", "skew"):
                raise MeasureError(f"Collada <{t}> transforms are not supported")
        return m

    nodes_by_id = {n.get("id"): n for n in root.iter() if _local(n.tag) == "node" and n.get("id")}
    box = Box()
    seen_instances = 0

    def visit(node, parent: Matrix, depth: int = 0) -> None:
        nonlocal seen_instances
        if depth > 64:
            raise MeasureError("Collada node hierarchy too deep or cyclic")
        world = mat_mul(parent, node_matrix(node))
        for c in node:
            t = _local(c.tag)
            if t == "instance_geometry":
                seen_instances += 1
                box.add_points(mat_mul(total, world), geoms.get(c.get("url", "").lstrip("#"), []))
            elif t == "instance_node":
                tgt = nodes_by_id.get(c.get("url", "").lstrip("#"))
                if tgt is not None:
                    visit(tgt, world, depth + 1)
            elif t == "node":
                visit(c, world, depth + 1)

    scene_nodes = []
    for vs in root.iter():
        if _local(vs.tag) == "visual_scene":
            scene_nodes += [c for c in vs if _local(c.tag) == "node"]
    for n in scene_nodes:
        visit(n, IDENT)
    notes = []
    if not seen_instances:                                     # no scene instances found: every geometry as it is
        for pts in geoms.values():
            box.add_points(total, pts)
        notes.append("no geometry instances in the visual scene: all geometries measured untransformed")
    return Measurement(box, fmt="dae", up_in=up, unit_m=unit_m, unit_known=True, canon=total, notes=notes)


# ----------------------------------------------------------------------------------------------------------------- dispatch
MEASURERS = {".glb": measure_gltf, ".gltf": measure_gltf, ".obj": measure_obj, ".stl": measure_stl, ".dae": measure_dae}


def measure_file(path, *, post: Matrix = IDENT) -> Measurement:
    """Bounding box of one mesh file. `post` (4x4) is applied after the canonical frame (an SDF scale/pose goes here)."""
    p = Path(path)
    fn = MEASURERS.get(p.suffix.lower())
    if fn is None:
        raise MeasureError(f"no measurer for {p.suffix!r} files")
    try:
        return fn(p, post=post)
    except MeasureError:
        raise
    except (json.JSONDecodeError, struct.error, UnicodeDecodeError, KeyError, IndexError, TypeError, ValueError) as e:
        raise MeasureError(f"{p.name}: malformed or truncated ({type(e).__name__}: {e})") from e


def union(parts: list[Measurement]) -> Measurement:
    """Union of several measurements already in the same frame (all with the same unit knowledge)."""
    if not parts:
        raise MeasureError("nothing to measure")
    box = Box()
    for m in parts:
        box.union(m.box)
    first = parts[0]
    return Measurement(box, fmt="+".join(sorted({m.fmt for m in parts})), up_in=first.up_in, unit_m=first.unit_m,
                       unit_known=all(m.unit_known for m in parts), canon=first.canon,
                       notes=[n for m in parts for n in m.notes])


# ----------------------------------------------------------------------------------------------------------------- SDF (Gazebo Fuel)
def _sdf_pose(el) -> Matrix:
    """<pose>x y z r p y</pose> (radians unless degrees="true"), or x y z qx qy qz qw with rotation_format="quat_xyzw"."""
    p = next((c for c in el if _local(c.tag) == "pose"), None)
    if p is None or not (p.text or "").strip():
        return IDENT
    v = _floats(p.text)
    if p.get("rotation_format") == "quat_xyzw":
        if len(v) != 7:
            raise MeasureError("quaternion pose needs 7 numbers")
        return mat_mul(mat_translate(v[:3]), mat_quat(v[3:7]))
    if len(v) != 6:
        raise MeasureError("pose needs 6 numbers")
    r, pi, y = (math.radians(a) for a in v[3:6]) if str(p.get("degrees", "")).lower() == "true" else v[3:6]
    return mat_mul(mat_translate(v[:3]), mat_rpy(r, pi, y))


def _find_mesh_file(uri: str, root: Path, sdf_dir: Path) -> Path:
    u = uri.strip()
    for pre in ("model://", "file://"):
        if u.startswith(pre):
            u = u[len(pre):]
            if pre == "model://" and "/" in u:
                u = u.split("/", 1)[1]
            break
    u = _unquote(u)
    for base in (root, sdf_dir):
        cand = (base / u)
        if cand.is_file():
            return cand
    hits = [p for p in root.rglob(Path(u).name) if p.is_file()]
    if len(hits) == 1:
        return hits[0]
    raise MeasureError(f"mesh {uri!r} not found in the model")


def measure_sdf(sdf_path, root=None) -> Measurement:
    """Union bounding box of every <visual> of an SDF model, with its <scale>, link/visual/model poses; +Z up, metres. Mesh formats
    that declare no unit (OBJ, STL) are taken as metres, which is the SDF convention. Nested models, includes, and geometry types
    other than mesh/box/sphere/cylinder raise MeasureError rather than guess."""
    sdf_path = Path(sdf_path)
    root = Path(root) if root else sdf_path.parent
    try:
        top = ET.parse(str(sdf_path)).getroot()
    except ET.ParseError as e:
        raise MeasureError(f"bad SDF XML: {e}") from None
    model = next((c for c in top.iter() if _local(c.tag) == "model"), None)
    if model is None:
        raise MeasureError("no <model> in the SDF")
    for c in model.iter():
        if _local(c.tag) == "include" or (_local(c.tag) == "model" and c is not model):
            raise MeasureError("nested models and <include> are not supported")
    parts: list[Measurement] = []
    notes: list[str] = []
    model_m = _sdf_pose(model)
    for link in (c for c in model if _local(c.tag) == "link"):
        link_m = mat_mul(model_m, _sdf_pose(link))
        for vis in (c for c in link if _local(c.tag) == "visual"):
            m = mat_mul(link_m, _sdf_pose(vis))
            geo = next((c for c in vis if _local(c.tag) == "geometry"), None)
            if geo is None:
                continue
            kind = next((c for c in geo), None)
            if kind is None:
                continue
            t = _local(kind.tag)
            if t == "mesh":
                uri = next((c.text for c in kind if _local(c.tag) == "uri"), None)
                if not uri:
                    raise MeasureError("mesh visual without a <uri>")
                sc = next((c for c in kind if _local(c.tag) == "scale"), None)
                s = _floats(sc.text) if sc is not None and (sc.text or "").strip() else [1.0, 1.0, 1.0]
                if len(s) != 3:
                    raise MeasureError("<scale> needs 3 numbers")
                post = mat_mul(m, mat_scale(s))
                meas = measure_file(_find_mesh_file(uri, root, sdf_path.parent), post=post)
                if s != [1.0, 1.0, 1.0]:
                    notes.append(f"SDF scale {s} applied to {Path(uri).name}")
                notes += meas.notes
                parts.append(meas)
            elif t in ("box", "sphere", "cylinder"):
                b = Box()
                if t == "box":
                    sz = _floats(next(c.text for c in kind if _local(c.tag) == "size"))
                    h = [v / 2 for v in sz]
                elif t == "sphere":
                    r = float(next(c.text for c in kind if _local(c.tag) == "radius"))
                    h = [r, r, r]
                else:
                    r = float(next(c.text for c in kind if _local(c.tag) == "radius"))
                    length = float(next(c.text for c in kind if _local(c.tag) == "length"))
                    h = [r, r, length / 2]
                b.add_corners(m, [-v for v in h], h)
                parts.append(Measurement(b, fmt=t, up_in="z", unit_m=1.0, unit_known=True, canon=IDENT))
            else:
                raise MeasureError(f"SDF geometry <{t}> is not supported")
    if not parts:
        raise MeasureError("the SDF has no measurable visuals")
    box = Box()
    for p in parts:
        box.union(p.box)
    return Measurement(box, fmt="sdf", up_in="z", unit_m=1.0, unit_known=True, canon=IDENT,
                       notes=list(dict.fromkeys(notes + ["per-visual SDF scale and poses baked in; unit-less meshes taken as metres"])))
