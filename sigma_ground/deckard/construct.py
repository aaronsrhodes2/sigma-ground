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
def _shape_from(part):
    """Instantiate a kernel shape from a spec Part (sphere/cylinder/box/cone)."""
    d = {k: f.value for k, f in part.dims.items()}
    c = tuple(part.center_m)
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

    stack = _LayerStack()
    density = {}
    layers = []
    info = []                                # (shape, rho, mass, half_extent)
    analytic_mass = 0.0
    Sx = Sy = Sz = 0.0                        # mass-weighted first moments
    xs, ys, zs = [], [], []
    for part in spec.parts:
        shp = _shape_from(part)
        stack.add(shp, part.name, "add")
        rho = part.density.value
        density[part.name] = rho
        vol = shp.volume()
        m = rho * vol
        analytic_mass += m
        cx, cy, cz = shp.center
        Sx += m * cx; Sy += m * cy; Sz += m * cz
        he = _half_extent(part)
        xs += [cx - he[0], cx + he[0]]; ys += [cy - he[1], cy + he[1]]
        zs += [cz - he[2], cz + he[2]]
        info.append((shp, rho, m, he))
        layers.append(Layer(part.name, part.material, rho, vol, m, part.density.cite()))

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

    if not overlapping:
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
        nlo = max(8, resolution // 2)
        M_lo = _integrate(composed, density, bbox, nlo)[0]
        stab = abs(M_num - M_lo) / M_num if M_num > 0 else 1.0
        passed = stab <= tolerance
        mass, com = M_num, com_num
        note = (f"{len(info)} overlapping primitive(s): mass from the SDF integrator "
                f"(union {M_num*1000:.1f} g), not Σρ·V ({analytic_mass*1000:.1f} g); "
                f"2-resolution stability {stab*100:.1f}% ({resolution}³ vs {nlo}³).")
        validation = {"passed": passed, "mode": "overlapping",
                      "mass_analytic_sum_kg": analytic_mass,
                      "mass_integrator_kg": M_num, "mass_residual": stab,
                      "resolution": resolution, "note": note}

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
