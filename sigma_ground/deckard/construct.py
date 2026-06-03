"""Deckard's compile step — researched dimensions → layered SDF matter.

Fits the primitive kit (here: capped cylinders) to the researched dimensions,
assembles the layered CSG via the existing kernel (`sigma_ground.csg`,
`sigma_ground.shapes`), and integrates the result's physical properties — mass,
centre of mass, inertia tensor — by sampling `material_at` over the bounding
box with a per-material density.

The matter is *self-validating*: every layer's volume also has a closed-form
(cylinder/annulus) value, and the SDF integrator is cross-checked against it.
Agreement = the general integrator is trustworthy (the same discipline as
Materia's two-method energy check). For arbitrary constructs only the
integrator exists — validated here on a shape whose answer we know exactly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..shapes import Cylinder
from ..csg import ComposedSDF


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


def compile_vessel(spec, resolution: int = 64, tolerance: float = 0.05) -> Construct:
    """Compile a `layered_vessel` ItemSpec into validated Construct matter."""
    g = spec.geometry
    R_out = g["outer_radius_m"]
    H     = g["height_m"]
    glaze = g["glaze_m"]
    wall  = g["wall_m"]
    base  = g["base_m"]
    fill  = g["fill_fraction"]

    R_cer = R_out - glaze            # ceramic outer radius
    R_in  = R_cer - wall             # interior (cavity) radius
    h_fill = fill * (H - base)       # water column height
    z_water_top = base + h_fill

    rho = {L["name"]: L["density_kg_m3"] for L in
           zip_layer_labels(spec)}   # label → density
    density = {"glaze": rho["glaze"], "ceramic": rho["ceramic"],
               "water": rho["water"], "air": 0.0}

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
        Layer("glaze", spec.layers[0]["material"], density["glaze"], v_glaze,
              density["glaze"] * v_glaze, spec.layers[0]["source"]),
        Layer("ceramic", spec.layers[1]["material"], density["ceramic"], v_ceramic,
              density["ceramic"] * v_ceramic, spec.layers[1]["source"]),
        Layer("water", spec.layers[2]["material"], density["water"], v_water,
              density["water"] * v_water, spec.layers[2]["source"]),
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
        identified=spec.identified, source=spec.source)


def zip_layer_labels(spec):
    """Attach engine labels (glaze/ceramic/water) to the spec's ordered layers."""
    labels = ["glaze", "ceramic", "water"]
    out = []
    for label, L in zip(labels, spec.layers):
        out.append({"name": label, "density_kg_m3": L["density_kg_m3"]})
    return out
