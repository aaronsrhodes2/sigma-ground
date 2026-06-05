"""Deckard's compile step — a researched ConstructSpec → layered SDF matter.

Fits the primitive kit (here: capped cylinders for a layered vessel) to the
spec's cited dimensions, assembles the layered CSG via the geometry kernel
(`sigma_ground.kernel`), and integrates the result's physical properties —
mass, centre of mass, inertia tensor — by sampling `material_at` over the
bounding box with a per-material density.

The matter is *self-validating*: every layer's volume also has a closed-form
(cylinder/annulus) value, and the SDF integrator is cross-checked against it.
Agreement = the general integrator is trustworthy (the same discipline as
Materia's two-method energy check). For arbitrary constructs only the
integrator exists — validated here on a shape whose answer we know exactly.

`compile(spec)` dispatches on `spec.kind`; the `layered_vessel` kit is
implemented today, and new primitive kits plug in behind the same gate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..kernel.shapes import Cylinder, Sphere, Box, Cone, Torus, Ellipsoid
from ..kernel.csg import ComposedSDF
from .sources import density_of as _density_of


@dataclass
class Layer:
    name: str
    material: str
    density_kg_m3: float
    volume_m3: float        # closed-form (exact)
    mass_kg: float
    source: str = ""


class _LayerStack:
    """Minimal structure (just `.layers` + `._operations`) for ComposedSDF."""
    def __init__(self):
        self.layers = []
        self._operations = []

    def add(self, shape, material, op="add"):
        self.layers.append((shape, material))
        self._operations.append(op)


def _cyl(radius, z0, z1):
    """Capped cylinder (axis z) spanning [z0, z1]."""
    return Cylinder(radius, z1 - z0, center=(0.0, 0.0, 0.5 * (z0 + z1)))


# ── The SDF integrator (general) ────────────────────────────────────────
def _integrate(composed, density, bbox, n):
    """Mass, centre of mass, and inertia (about CoM) by grid sampling.

    Returns (mass, (cx,cy,cz), (Ixx,Iyy,Izz), {label: volume}).
    """
    (x0, x1), (y0, y1), (z0, z1) = bbox
    dx, dy, dz = (x1 - x0) / n, (y1 - y0) / n, (z1 - z0) / n
    cell_v = dx * dy * dz

    M = 0.0
    Sx = Sy = Sz = 0.0                       # first moments  Σ dm r
    Jxx = Jyy = Jzz = 0.0                    # inertia about origin
    vol = {}

    for iz in range(n):
        z = z0 + (iz + 0.5) * dz
        for iy in range(n):
            y = y0 + (iy + 0.5) * dy
            for ix in range(n):
                x = x0 + (ix + 0.5) * dx
                label = composed.material_at(x, y, z)
                if label is None:
                    continue
                rho = density.get(label, 0.0)
                if rho <= 0.0:
                    continue
                dm = rho * cell_v
                M += dm
                Sx += dm * x; Sy += dm * y; Sz += dm * z
                Jxx += dm * (y * y + z * z)
                Jyy += dm * (x * x + z * z)
                Jzz += dm * (x * x + y * y)
                vol[label] = vol.get(label, 0.0) + cell_v

    if M <= 0.0:
        return 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), vol
    cx, cy, cz = Sx / M, Sy / M, Sz / M
    # parallel-axis shift (origin → CoM)
    Ixx = Jxx - M * (cy * cy + cz * cz)
    Iyy = Jyy - M * (cx * cx + cz * cz)
    Izz = Jzz - M * (cx * cx + cy * cy)
    return M, (cx, cy, cz), (Ixx, Iyy, Izz), vol


# ── The compiled matter ─────────────────────────────────────────────────
@dataclass
class Construct:
    name: str
    composed: object                # ComposedSDF
    density_by_label: dict
    layers: list                    # list[Layer]
    mass_kg: float                  # closed-form total
    com_m: tuple                    # centre of mass (analytic)
    inertia_kgm2: tuple             # (Ixx, Iyy, Izz) about CoM, from the integrator
    bbox: tuple
    validation: dict = field(default_factory=dict)
    identified: bool = True
    source: str = ""

    # consumers (Materia / Radiance) query the field:
    def sdf(self, x, y, z):
        return self.composed.sdf(x, y, z)

    def material_at(self, x, y, z):
        return self.composed.material_at(x, y, z)

    def density_at(self, x, y, z):
        return self.density_by_label.get(self.material_at(x, y, z), 0.0)

    def render(self) -> str:
        flag = "" if self.identified else "  [UNIDENTIFIED — best-guess]"
        lines = [f"━━ Deckard · {self.name} ━━{flag}", f"  {self.source}", ""]
        for L in self.layers:
            lines.append(f"  • {L.name:8s} {L.material:16s} "
                         f"V={L.volume_m3*1e6:8.2f} cm³  m={L.mass_kg*1000:7.1f} g  "
                         f"(ρ={L.density_kg_m3:.0f})")
            lines.append(f"        {L.source}")
        cx, cy, cz = self.com_m
        Ixx, Iyy, Izz = self.inertia_kgm2
        lines += [
            "",
            f"  total mass : {self.mass_kg*1000:.1f} g",
            f"  centre of mass : ({cx*1000:.2f}, {cy*1000:.2f}, {cz*1000:.2f}) mm "
            f"(on-axis, {cz*1000:.1f} mm up)",
            f"  inertia @ CoM : Ixx={Ixx*1e6:.1f}, Iyy={Iyy*1e6:.1f}, "
            f"Izz={Izz*1e6:.1f}  (×10⁻⁶ kg·m²)",
        ]
        v = self.validation
        if v:
            mark = "✓ PASS" if v.get("passed") else "✗ FAIL"
            lines += ["", f"  self-check {mark}: {v.get('note', '')}"]
        return "\n".join(lines)


# ── compile: ConstructSpec → Construct ──────────────────────────────────
def _layer(spec, name):
    """The ConstructSpec layer with this engine label (raises if missing)."""
    for L in spec.layers:
        if L.name == name:
            return L
    raise KeyError(f"layer '{name}' missing from spec '{spec.name}'")


def compile(spec, resolution: int = 64, tolerance: float = 0.05) -> Construct:
    """Compile a ConstructSpec into validated Construct matter.

    Dispatches on ``spec.kind``. Today the ``layered_vessel`` kit is supported;
    new primitive kits register here behind the same self-check gate.
    """
    if getattr(spec, "parts", None):
        return _compile_parts(spec, resolution, tolerance)
    kind = getattr(spec, "kind", "layered_vessel")
    if kind == "layered_vessel":
        return _compile_vessel(spec, resolution, tolerance)
    raise NotImplementedError(f"compile: no kit for kind '{kind}' yet")


# Back-compat alias (the old entry point name).
compile_vessel = compile


# ── general primitive kit ───────────────────────────────────────────────
def _shape_from(part, center=None):
    """Instantiate a kernel shape from a spec Part (sphere/cylinder/box/cone)."""
    d = {k: f.value for k, f in part.dims.items()}
    c = tuple(center) if center is not None else tuple(part.center_m)
    s = (part.shape or "").lower()
    if s == "sphere":
        return Sphere(d["radius_m"], center=c)
    if s == "cylinder":
        return Cylinder(d["radius_m"], d["height_m"], center=c)
    if s == "box":
        return Box(d["x_m"], d["y_m"], d["z_m"], center=c)
    if s == "cone":
        return Cone(d["radius_m"], d["height_m"], center=c)
    if s == "torus":
        return Torus(d["major_radius_m"], d["minor_radius_m"], center=c)
    if s == "ellipsoid":
        return Ellipsoid(d["rx_m"], d["ry_m"], d["rz_m"], center=c)
    raise ValueError(f"unsupported primitive shape '{part.shape}' in part '{part.name}'")


def _half_extent(part):
    """Axis-aligned half-extents (hx, hy, hz) of a part's shape — a *tight* bbox,
    so the grid integrator isn't wasted on empty space around high-aspect shapes."""
    d = {k: f.value for k, f in part.dims.items()}
    s = (part.shape or "").lower()
    if s == "sphere":
        r = d["radius_m"]
        return (r, r, r)
    if s in ("cylinder", "cone"):
        return (d["radius_m"], d["radius_m"], d["height_m"] / 2.0)
    if s == "box":
        return (d["x_m"] / 2.0, d["y_m"] / 2.0, d["z_m"] / 2.0)
    if s == "torus":
        rr = d["major_radius_m"] + d["minor_radius_m"]
        return (rr, rr, d["minor_radius_m"])
    if s == "ellipsoid":
        return (d["rx_m"], d["ry_m"], d["rz_m"])
    raise ValueError(f"no AABB for shape '{part.shape}'")


# ── rotation (Euler degrees) — rotate the SDF query into the shape's frame ──
def _matmul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))


def _rotation(euler_deg):
    """3x3 rotation R = Rz·Ry·Rx from Euler angles in degrees."""
    rx, ry, rz = (math.radians(a) for a in euler_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = ((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx))
    Ry = ((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy))
    Rz = ((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0))
    return _matmul(Rz, _matmul(Ry, Rx))


class _Rotated:
    """A kernel shape rotated by R about its centre (for SDF queries).

    volume()/bounding_radius() are rotation-invariant (delegated); the SDF is
    evaluated at the inverse-rotated query point (R^T · (p - c) + c).
    """
    def __init__(self, base, R):
        self._base = base
        self._R = R
        self.center = base.center

    def surface_distance(self, px, py, pz):
        cx, cy, cz = self.center
        dx, dy, dz = px - cx, py - cy, pz - cz
        R = self._R
        lx = R[0][0] * dx + R[1][0] * dy + R[2][0] * dz   # R^T · d
        ly = R[0][1] * dx + R[1][1] * dy + R[2][1] * dz
        lz = R[0][2] * dx + R[1][2] * dy + R[2][2] * dz
        return self._base.surface_distance(cx + lx, cy + ly, cz + lz)

    def volume(self):
        return self._base.volume()

    def bounding_radius(self):
        return self._base.bounding_radius()


def _rotated_half_extent(he, R):
    """AABB half-extents of a box [±he] rotated by R (max |coord| over 8 corners)."""
    hx, hy, hz = he
    mx = my = mz = 0.0
    for sx in (-hx, hx):
        for sy in (-hy, hy):
            for sz in (-hz, hz):
                wx = R[0][0] * sx + R[0][1] * sy + R[0][2] * sz
                wy = R[1][0] * sx + R[1][1] * sy + R[1][2] * sz
                wz = R[2][0] * sx + R[2][1] * sy + R[2][2] * sz
                mx = max(mx, abs(wx)); my = max(my, abs(wy)); mz = max(mz, abs(wz))
    return (mx, my, mz)


# ── fluid fill: liquid floods a cavity bottom-up, gas settles on top (−z) ──
class _Clipped:
    """A shape intersected with a horizontal half-space — the cavity kept on one
    side of a fill plane. ``sign=+1`` keeps the part BELOW ``level`` (the liquid
    that sinks); ``sign=-1`` keeps it ABOVE (the gas on top). Volume and centre
    are precomputed when the fill is solved; the SDF is the CSG-intersection max.
    """
    def __init__(self, base, axis, sign, level, volume, center):
        self._base = base
        self._axis = axis
        self._sign = sign
        self._level = level
        self._volume = volume
        self.center = center

    def surface_distance(self, px, py, pz):
        d = self._base.surface_distance(px, py, pz)
        half = self._sign * ((px, py, pz)[self._axis] - self._level)
        return half if half > d else d          # max(d_base, d_halfspace) = ∩

    def volume(self):
        return self._volume

    def bounding_radius(self):
        return self._base.bounding_radius()


def _gas_density(material):
    """Grounded gas density (kg/m³); defaults to air if unknown — never the
    estimated-solid fallback, so a headspace can't accidentally weigh like rock."""
    f = _density_of(material, allow_web=False)
    return f.value if (f is not None and not f.estimated) else 1.225


def _fill_solve(base, aabb, fraction, axis=2, n=48):
    """Solve the fill plane on ``axis`` so the cavity below it holds ``fraction``
    of the cavity volume; return (level, below_vol, below_com, above_vol, above_com).

    Samples the cavity interior once; the plane is the fraction-quantile of the
    interior cells' coordinates — exact for a prismatic cavity, robust for any
    shape (cone, sphere, …). Volumes are partitioned from the analytic total so
    below+above == base.volume() exactly.
    """
    (x0, x1), (y0, y1), (z0, z1) = aabb
    dx, dy, dz = (x1 - x0) / n, (y1 - y0) / n, (z1 - z0) / n
    cells = []
    for iz in range(n):
        z = z0 + (iz + 0.5) * dz
        for iy in range(n):
            y = y0 + (iy + 0.5) * dy
            for ix in range(n):
                x = x0 + (ix + 0.5) * dx
                if base.surface_distance(x, y, z) < 0.0:
                    cells.append((x, y, z))
    V = base.volume()
    if not cells:
        return aabb[axis][0], 0.0, base.center, 0.0, base.center
    cells.sort(key=lambda p: p[axis])
    frac = min(1.0, max(0.0, fraction))
    k = int(round(frac * len(cells)))           # this many interior cells are liquid
    below, above = cells[:k], cells[k:]

    def com(pts, fallback):
        if not pts:
            return fallback
        m = len(pts)
        return (sum(p[0] for p in pts) / m, sum(p[1] for p in pts) / m,
                sum(p[2] for p in pts) / m)

    if k <= 0:
        level = aabb[axis][0]
    elif k >= len(cells):
        level = aabb[axis][1]
    else:
        level = 0.5 * (below[-1][axis] + above[0][axis])
    below_vol = (len(below) / len(cells)) * V
    return (level, below_vol, com(below, base.center),
            V - below_vol, com(above, base.center))


def _expand_fill(fp, by_name, centers, n=48):
    """Expand a fluid ``fill`` part into concrete liquid (+ gas) build items that
    flood the named cavity. Gravity (−z) settles the liquid to the bottom; the
    gas, lighter, takes the headspace on top. Returns a list of item dicts."""
    ref = (fp.fill or {}).get("of")
    cav = by_name.get(ref)
    if cav is None:
        raise ValueError(f"fill part '{fp.name}' references unknown cavity '{ref}'")
    base = _shape_from(cav, centers[cav.name])
    he = _half_extent(cav)
    euler = getattr(cav, "euler_deg", (0.0, 0.0, 0.0))
    if any(euler):
        R = _rotation(euler)
        base = _Rotated(base, R)
        he = _rotated_half_extent(he, R)
    c = base.center
    aabb = ((c[0] - he[0], c[0] + he[0]), (c[1] - he[1], c[1] + he[1]),
            (c[2] - he[2], c[2] + he[2]))
    fraction = (fp.fill or {}).get("fraction", 1.0)
    level, v_below, com_below, v_above, com_above = _fill_solve(base, aabb, fraction, 2, n)

    items = [dict(name=fp.name, shape=_Clipped(base, 2, 1.0, level, v_below, com_below),
                  material=fp.material, rho=fp.density.value, op="add", he=he,
                  cite=f"{fp.density.cite()} (fills {ref} to {fraction:g})")]
    if v_above > 0.0 and fraction < 1.0:
        gas_mat = (fp.fill or {}).get("gas", "air")
        gas_rho = _gas_density(gas_mat)
        items.append(dict(name=f"{fp.name}__gas",
                          shape=_Clipped(base, 2, -1.0, level, v_above, com_above),
                          material=gas_mat, rho=gas_rho, op="add", he=he,
                          cite=f"{gas_mat} headspace on top ({gas_rho:g} kg/m³)"))
    return items


# ── conforming solids: a part yields where it would overlap its mate ──────
class _Subtracted:
    """Shape A with shape B carved out — A − B (= A ∩ ¬B). A conforming solid
    yields wherever it would overlap its target: its surface becomes the
    target's surface there, so distinct solids share an *exact matching*
    (congruent) interface instead of clipping through each other.
    """
    def __init__(self, base, cut, volume, center):
        self._base = base
        self._cut = cut
        self._volume = volume
        self.center = center

    def surface_distance(self, px, py, pz):
        d = self._base.surface_distance(px, py, pz)
        dc = -self._cut.surface_distance(px, py, pz)     # ¬B  (outside the cut)
        return dc if dc > d else d                        # max(d_A, ¬d_B) = A ∩ ¬B

    def volume(self):
        return self._volume

    def bounding_radius(self):
        return self._base.bounding_radius()


def _make_conform(base, cut, aabb, n=44):
    """Carve ``cut`` out of ``base`` (base − cut); sample the result's volume and
    centroid over ``aabb`` so the conforming part reports the matter it actually
    keeps after yielding the overlap."""
    (x0, x1), (y0, y1), (z0, z1) = aabb
    dx, dy, dz = (x1 - x0) / n, (y1 - y0) / n, (z1 - z0) / n
    cell = dx * dy * dz
    cnt = 0
    sx = sy = sz = 0.0
    for iz in range(n):
        z = z0 + (iz + 0.5) * dz
        for iy in range(n):
            y = y0 + (iy + 0.5) * dy
            for ix in range(n):
                x = x0 + (ix + 0.5) * dx
                if base.surface_distance(x, y, z) < 0.0 and cut.surface_distance(x, y, z) >= 0.0:
                    cnt += 1
                    sx += x; sy += y; sz += z
    if cnt == 0:
        return _Subtracted(base, cut, 0.0, base.center)
    return _Subtracted(base, cut, cnt * cell, (sx / cnt, sy / cnt, sz / cnt))


# ── attachment: parts mate at anchors (touch at an interface, no overlap) ──
def _local_anchor(part, name):
    """A named anchor point in the part's LOCAL frame (before rotate/translate)."""
    d = {k: f.value for k, f in part.dims.items()}
    s = (part.shape or "").lower()
    name = (name or "center").lower()
    if name in ("center", "centre"):
        return (0.0, 0.0, 0.0)
    if s in ("cylinder", "cone"):
        h = d.get("height_m", 0.0)
        return {"top": (0.0, 0.0, h / 2.0),
                "bottom": (0.0, 0.0, -h / 2.0)}.get(name, (0.0, 0.0, 0.0))
    if s == "box":
        x, y, z = d.get("x_m", 0.0), d.get("y_m", 0.0), d.get("z_m", 0.0)
        return {"top": (0.0, 0.0, z / 2.0), "bottom": (0.0, 0.0, -z / 2.0),
                "+x": (x / 2.0, 0.0, 0.0), "-x": (-x / 2.0, 0.0, 0.0),
                "+y": (0.0, y / 2.0, 0.0), "-y": (0.0, -y / 2.0, 0.0)}.get(name, (0.0, 0.0, 0.0))
    if s in ("sphere", "ellipsoid"):
        r = d.get("radius_m") or d.get("rz_m", 0.0)
        rx = d.get("rx_m", r); ry = d.get("ry_m", r)
        return {"top": (0.0, 0.0, r), "bottom": (0.0, 0.0, -r),
                "+x": (rx, 0.0, 0.0), "-x": (-rx, 0.0, 0.0),
                "+y": (0.0, ry, 0.0), "-y": (0.0, -ry, 0.0)}.get(name, (0.0, 0.0, 0.0))
    return (0.0, 0.0, 0.0)


def _apply(R, v):
    """R · v (forward rotation of a vector)."""
    return (R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
            R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
            R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2])


def _resolve_positions(parts):
    """World centre of each part, honouring ``attach`` so anchors mate exactly
    (parts touch at a shared interface — no overlap, no gap). Returns {name: center}.
    """
    by_name = {p.name: p for p in parts}
    R = {p.name: _rotation(getattr(p, "euler_deg", (0.0, 0.0, 0.0))) for p in parts}
    centers = {}

    def world_anchor(pname, anchor):
        la = _apply(R[pname], _local_anchor(by_name[pname], anchor))
        c = centers[pname]
        return (c[0] + la[0], c[1] + la[1], c[2] + la[2])

    def resolve(name, stack):
        if name in centers:
            return
        p = by_name[name]
        att = getattr(p, "attach", None)
        if not att or att.get("to") not in by_name or att.get("to") in stack:
            centers[name] = tuple(float(x) for x in p.center_m)   # root / no-attach / bad ref
            return
        resolve(att["to"], stack | {name})
        target = world_anchor(att["to"], att.get("their", "top"))
        mine = _apply(R[name], _local_anchor(p, att.get("my", "bottom")))
        centers[name] = (target[0] - mine[0], target[1] - mine[1], target[2] - mine[2])

    for p in parts:
        resolve(p.name, frozenset())
    return centers


def _interfaces(composed, name_to_material, bbox, n):
    """Every mated surface pair: adjacent distinct materials + contact area (m²).

    Walks the composed construct on a grid and tallies each face where the
    material changes — solid↔solid, solid↔liquid, liquid↔gas, solid↔gas, and
    anything↔ambient-air (outside). The boundary between two materials *is* the
    interface.
    """
    (x0, x1), (y0, y1), (z0, z1) = bbox
    dx, dy, dz = (x1 - x0) / n, (y1 - y0) / n, (z1 - z0) / n

    def mat(ix, iy, iz):
        name = composed.material_at(x0 + (ix + 0.5) * dx, y0 + (iy + 0.5) * dy,
                                    z0 + (iz + 0.5) * dz)
        return name_to_material.get(name, "air") if name else "air"  # outside = ambient air

    pairs = {}

    def add(a, b, area):
        if a == b:
            return
        k = tuple(sorted((a, b)))
        pairs[k] = pairs.get(k, 0.0) + area

    ax, ay, az = dy * dz, dx * dz, dx * dy
    prev = None
    for iz in range(n):
        cur = [[mat(ix, iy, iz) for ix in range(n)] for iy in range(n)]
        for iy in range(n):
            for ix in range(n):
                a = cur[iy][ix]
                if ix + 1 < n:
                    add(a, cur[iy][ix + 1], ax)
                if iy + 1 < n:
                    add(a, cur[iy + 1][ix], ay)
                if prev is not None:
                    add(a, prev[iy][ix], az)
        prev = cur
    return [{"between": list(k), "area_m2": round(v, 6)}
            for k, v in sorted(pairs.items(), key=lambda kv: -kv[1])]


def _shape_mass(shape, rho, bbox, n):
    """Mass of a single shape by grid-sampling its own SDF — validates volume()."""
    (x0, x1), (y0, y1), (z0, z1) = bbox
    dx, dy, dz = (x1 - x0) / n, (y1 - y0) / n, (z1 - z0) / n
    cell = dx * dy * dz
    M = 0.0
    for iz in range(n):
        z = z0 + (iz + 0.5) * dz
        for iy in range(n):
            y = y0 + (iy + 0.5) * dy
            for ix in range(n):
                if shape.surface_distance(x0 + (ix + 0.5) * dx, y, z) < 0.0:
                    M += rho * cell
    return M


def _compile_parts(spec, resolution: int, tolerance: float) -> Construct:
    """Compose primitive parts → validated matter.

    Disjoint parts: mass / CoM are the exact analytic sum (Σ ρ·shape.volume());
    the self-check grid-integrates each shape against its own analytic volume.
    Overlapping parts (detected via intersecting bounding spheres): the SDF
    integrator (union, last-material-wins) is canonical — Σ ρ·V would double-count
    the overlap — and the self-check is a two-resolution stability test. Inertia
    always comes from the integrator.
    """
    if not spec.parts:
        raise ValueError(f"spec '{spec.name}' has no parts to compile")

    centers = _resolve_positions(spec.parts)   # honour attach: parts mate at interfaces
    by_name = {p.name: p for p in spec.parts}

    # normalise parts into build items: solids/carves first, then expand fluid
    # fills into liquid (+ gas) that flood the named cavity bottom-up.
    items = []
    for part in spec.parts:
        if getattr(part, "fill", None):
            continue                              # fills expanded after the solids
        carve = getattr(part, "op", "add") == "subtract"
        shp = _shape_from(part, centers[part.name])
        he = _half_extent(part)
        euler = getattr(part, "euler_deg", (0.0, 0.0, 0.0))
        if any(euler):
            R = _rotation(euler)
            shp = _Rotated(shp, R)
            he = _rotated_half_extent(he, R)
        cite = "carved cavity" if carve else part.density.cite()
        conform_to = getattr(part, "conform", None)
        if conform_to and not carve and conform_to in by_name:
            tgt = by_name[conform_to]
            tshape = _shape_from(tgt, centers[tgt.name])
            teuler = getattr(tgt, "euler_deg", (0.0, 0.0, 0.0))
            if any(teuler):
                tshape = _Rotated(tshape, _rotation(teuler))
            c = shp.center
            aabb = ((c[0] - he[0], c[0] + he[0]), (c[1] - he[1], c[1] + he[1]),
                    (c[2] - he[2], c[2] + he[2]))
            shp = _make_conform(shp, tshape, aabb)        # base − target: yield the overlap
            cite = f"{part.density.cite()} (conforms to {conform_to})"
        items.append(dict(name=part.name, shape=shp, material=part.material,
                          rho=0.0 if carve else part.density.value,
                          op="subtract" if carve else "add", he=he, cite=cite))
    for part in spec.parts:
        if getattr(part, "fill", None):
            items += _expand_fill(part, by_name, centers)

    stack = _LayerStack()
    density = {}
    name_to_material = {}
    layers = []
    info = []                                # (shape, rho, mass, half_extent)
    analytic_mass = 0.0
    Sx = Sy = Sz = 0.0                        # mass-weighted first moments
    xs, ys, zs = [], [], []
    for it in items:             # compose IN ORDER: solids → carves → fills (last wins)
        shp, he = it["shape"], it["he"]
        stack.add(shp, it["name"], it["op"])
        vol = shp.volume()
        rho = it["rho"]
        density[it["name"]] = rho
        name_to_material[it["name"]] = it["material"]
        m = rho * vol
        analytic_mass += m
        cx, cy, cz = shp.center
        Sx += m * cx; Sy += m * cy; Sz += m * cz
        xs += [cx - he[0], cx + he[0]]; ys += [cy - he[1], cy + he[1]]
        zs += [cz - he[2], cz + he[2]]
        info.append((shp, rho, m, he))
        layers.append(Layer(it["name"], it["material"], rho, vol, m, it["cite"]))

    sub_parts = [it for it in items if it["op"] == "subtract"]
    conformed = any(getattr(p, "conform", None) and getattr(p, "op", "add") != "subtract"
                    and p.conform in by_name for p in spec.parts)

    if analytic_mass <= 0.0:
        raise ValueError(f"spec '{spec.name}' compiled to zero mass")
    composed = ComposedSDF(stack)
    com_ana = (Sx / analytic_mass, Sy / analytic_mass, Sz / analytic_mass)
    pad = 0.02 * max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1e-9)
    bbox = ((min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad),
            (min(zs) - pad, max(zs) + pad))
    M_num, com_num, inertia, _vol = _integrate(composed, density, bbox, resolution)

    # overlap test on tight AABBs (touching faces => not overlapping)
    boxes = []
    for shp, _r, _m, he in info:
        c = shp.center
        boxes.append((c[0] - he[0], c[0] + he[0], c[1] - he[1], c[1] + he[1],
                      c[2] - he[2], c[2] + he[2]))
    overlapping = False
    for i in range(len(boxes)):
        ax0, ax1, ay0, ay1, az0, az1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            bx0, bx1, by0, by1, bz0, bz1 = boxes[j]
            if (ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1
                    and az0 < bz1 and bz0 < az1):
                overlapping = True
                break
        if overlapping:
            break

    if not overlapping and not sub_parts:
        worst = 0.0
        for shp, rho, m, he in info:
            c = shp.center
            pb = ((c[0] - he[0], c[0] + he[0]), (c[1] - he[1], c[1] + he[1]),
                  (c[2] - he[2], c[2] + he[2]))
            if m > 0:
                worst = max(worst, abs(_shape_mass(shp, rho, pb, resolution) - m) / m)
        passed = worst <= tolerance
        mass, com = analytic_mass, com_ana
        note = (f"{len(info)} disjoint primitive(s): each grid integration matches "
                f"its analytic volume (worst {worst*100:.1f}%) at {resolution}³.")
        validation = {"passed": passed, "mode": "disjoint",
                      "mass_analytic_total_kg": analytic_mass,
                      "mass_integrator_kg": M_num, "mass_residual": worst,
                      "resolution": resolution, "note": note}
    else:
        mode = "hollow" if sub_parts else ("conforming" if conformed else "overlapping")
        nhi = int(resolution * 1.5)
        M_hi, com_hi, inertia, _ = _integrate(composed, density, bbox, nhi)
        stab = abs(M_num - M_hi) / M_hi if M_hi > 0 else 1.0
        passed = stab <= tolerance
        mass, com = M_hi, com_hi
        note = (f"{len(info)} primitive(s) [{mode}]: mass from the SDF integrator "
                f"({M_hi*1000:.1f} g) vs Σρ·V solids {analytic_mass*1000:.1f} g; "
                f"convergence {stab*100:.1f}% ({resolution}³→{nhi}³).")
        validation = {"passed": passed, "mode": mode,
                      "mass_analytic_sum_kg": analytic_mass,
                      "mass_integrator_kg": M_hi, "mass_residual": stab,
                      "resolution": nhi, "note": note}

    attached = {p.name for p in spec.parts}
    validation["joints"] = [
        {"between": [p.name, p.attach["to"]], "at": p.attach.get("their", "top")}
        for p in spec.parts
        if getattr(p, "attach", None) and p.attach.get("to") in attached]
    validation["interfaces"] = _interfaces(composed, name_to_material, bbox,
                                            min(resolution, 48))

    return Construct(
        name=spec.name, composed=composed, density_by_label=density,
        layers=layers, mass_kg=mass, com_m=com, inertia_kgm2=inertia,
        bbox=bbox, validation=validation, identified=spec.identified,
        source=spec.note())


def _compile_vessel(spec, resolution: int, tolerance: float) -> Construct:
    """Fit nested capped cylinders to a layered-vessel spec, then self-check."""
    R_out = spec.dim("outer_radius_m")
    H     = spec.dim("height_m")
    glaze = spec.dim("glaze_m")
    wall  = spec.dim("wall_m")
    base  = spec.dim("base_m")
    fill  = spec.dim("fill_fraction")

    R_cer = R_out - glaze            # ceramic outer radius
    R_in  = R_cer - wall             # interior (cavity) radius
    h_fill = fill * (H - base)       # water column height
    z_water_top = base + h_fill

    Lg, Lc, Lw = _layer(spec, "glaze"), _layer(spec, "ceramic"), _layer(spec, "water")
    density = {"glaze": Lg.density.value, "ceramic": Lc.density.value,
               "water": Lw.density.value, "air": 0.0}

    # ── assemble the layered SDF (outer→inner; ComposedSDF: last-added wins) ──
    stack = _LayerStack()
    stack.add(_cyl(R_out, 0.0, H), "glaze",   "add")      # outer skin ring
    stack.add(_cyl(R_cer, 0.0, H), "ceramic", "add")      # body
    stack.add(_cyl(R_in,  base, H), "air",    "add")      # carve the bore (auto-subtract)
    stack.add(_cyl(R_in,  base, z_water_top), "water", "add")  # fill the bottom
    composed = ComposedSDF(stack)

    # ── closed-form volumes (exact) ─────────────────────────────────────
    pi = math.pi
    v_glaze   = pi * (R_out**2 - R_cer**2) * H
    v_ceramic = pi * R_cer**2 * H - pi * R_in**2 * (H - base)
    v_water   = pi * R_in**2 * h_fill
    # centroids (on axis)
    z_glaze = H / 2.0
    z_water = base + h_fill / 2.0
    v_solid_cyl, z_solid = pi * R_cer**2 * H, H / 2.0
    v_bore, z_bore = pi * R_in**2 * (H - base), base + (H - base) / 2.0
    z_ceramic = (v_solid_cyl * z_solid - v_bore * z_bore) / v_ceramic

    layers = [
        Layer("glaze", Lg.material, density["glaze"], v_glaze,
              density["glaze"] * v_glaze, Lg.density.cite()),
        Layer("ceramic", Lc.material, density["ceramic"], v_ceramic,
              density["ceramic"] * v_ceramic, Lc.density.cite()),
        Layer("water", Lw.material, density["water"], v_water,
              density["water"] * v_water, Lw.density.cite()),
    ]
    mass_ana = sum(L.mass_kg for L in layers)
    z_com_ana = sum(L.mass_kg * z for L, z in
                    zip(layers, (z_glaze, z_ceramic, z_water))) / mass_ana

    # ── independent SDF integration + cross-check ───────────────────────
    # The integrator resolves ceramic+water; the 0.3 mm glaze is sub-cell at any
    # practical grid, so it's computed analytically and excluded from the numeric
    # cross-check (compared apples-to-apples against ceramic+water).
    bbox = ((-R_out, R_out), (-R_out, R_out), (0.0, H))
    M_num, com_num, inertia, _vol = _integrate(composed, density, bbox, resolution)
    m_resolvable = layers[1].mass_kg + layers[2].mass_kg
    z_resolvable = (layers[1].mass_kg * z_ceramic
                    + layers[2].mass_kg * z_water) / m_resolvable
    mass_res = abs(M_num - m_resolvable) / m_resolvable
    zcom_res = abs(com_num[2] - z_resolvable)
    passed = mass_res <= tolerance and zcom_res <= 0.002    # 2 mm CoM tolerance

    validation = {
        "passed": passed,
        "mass_analytic_total_kg": mass_ana,
        "mass_resolvable_kg": m_resolvable,
        "mass_integrator_kg": M_num,
        "mass_residual": mass_res,
        "zcom_resolvable_m": z_resolvable,
        "zcom_integrator_m": com_num[2],
        "resolution": resolution,
        "note": (f"SDF integrator {M_num*1000:.1f} g vs closed-form ceramic+water "
                 f"{m_resolvable*1000:.1f} g ({mass_res*100:.1f}%); CoM "
                 f"{com_num[2]*1000:.2f} vs {z_resolvable*1000:.2f} mm at "
                 f"{resolution}³. Glaze ({v_glaze*1e6:.2f} cm³, sub-cell) is "
                 f"analytic; integrator inertia covers resolvable matter."),
    }

    return Construct(
        name=spec.name, composed=composed, density_by_label=density,
        layers=layers, mass_kg=mass_ana, com_m=(0.0, 0.0, z_com_ana),
        inertia_kgm2=inertia, bbox=bbox, validation=validation,
        identified=spec.identified, source=spec.note())
