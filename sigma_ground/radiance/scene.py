"""RadianceScene — what to draw: a field, its materials, and a light.

A scene is deliberately thin: a signed-distance field `sdf(Vec3) -> float`, a
`material_at(Vec3) -> material_key` lookup, one directional light, and a
background. Per the doctrine, the background is BLACK — empty space emits no
light, so anywhere there's no matter the render is dark, exactly as it would be.

Build one straight from a primitive (`from_shape`) or from any composed SDF
(`from_sdf`) — e.g. a CSG tree assembled with `sigma_ground.csg`.
"""
from __future__ import annotations

from ..dynamics.vec import Vec3


class RadianceScene:
    def __init__(self, sdf, material_at, light_dir: Vec3 = None,
                 light_color: Vec3 = None, ambient: float = 0.12,
                 background: Vec3 = None, max_dist: float = 50.0,
                 albedo=None, temperature_at=None, emissivity_of=None,
                 ao_rays: int = 0, ao_reach: float = None):
        self.sdf = sdf
        self.material_at = material_at
        # Optional label→Vec3 albedo (e.g. BAKED emergent colors from a
        # SceneSpec). If None, shading derives color from the material library.
        self.albedo = albedo
        # Optional thermal hooks (both None → identical pre-thermal output):
        #   temperature_at(Vec3) -> K   — the body/field temperature at a point
        #   emissivity_of(label) -> (r,g,b)  — Kirchhoff ε(λ)
        # shade() adds Planck × Kirchhoff incandescence when both are present.
        self.temperature_at = temperature_at
        self.emissivity_of = emissivity_of
        ld = light_dir if light_dir is not None else Vec3(-0.5, -1.0, -0.55)
        self.light_dir = ld.normalized()             # direction the light travels
        self.light_color = light_color if light_color is not None else Vec3(1, 1, 1)
        self.ambient = float(ambient)
        self.background = background if background is not None else Vec3(0, 0, 0)
        self.max_dist = float(max_dist)
        # Opt-in secondary-ray ambient occlusion (shade.ambient_occlusion):
        # 0 rays = off (default, output identical to pre-AO renders).
        # ao_reach None -> shade() defaults it to 0.1 * max_dist.
        self.ao_rays = int(ao_rays)
        self.ao_reach = float(ao_reach) if ao_reach is not None else None

    @classmethod
    def from_shape(cls, shape, material_key: str, **kw):
        """Scene of a single primitive (uses the shape's analytic SDF)."""
        def sdf(p):
            return shape.surface_distance(p.x, p.y, p.z)
        return cls(sdf, lambda p: material_key, **kw)

    @classmethod
    def from_sdf(cls, sdf, material, **kw):
        """Scene from any composed SDF. `material` is a key or a callable."""
        mat_at = material if callable(material) else (lambda p: material)
        return cls(sdf, mat_at, **kw)
