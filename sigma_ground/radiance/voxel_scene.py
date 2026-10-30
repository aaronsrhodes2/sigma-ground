"""
A multi-material voxel build as a staged scene: one Voxel leaf PER MATERIAL.

``construct_from_field`` → ``construct_to_scene`` ships a voxel construct as a
single Voxel leaf coloured by its most common material — right for a scanned
object that is mostly one substance, wrong for a built world where the floor is
granite, the tank glass and the pool mercury. Here each material in a
``VoxelField`` gets its own signed-distance grid (the cells of that label
only), so every leaf carries its own baked emergent colour, density and
mechanics, exactly as a primitive Deckard part would. The union of the leaves
is the build.

Air is never a leaf: label 0 is empty space, so it is invisible by
construction, and anything else can occupy it.

    field ──per label──► Voxel leaves + _bake_material ──► SceneSpec
          ──theaters.stage──► plain / room / void / deep_space
          ──scene_from_spec──► the Python path-tracer ──► PNG

The Python renderer shades diffusely (shade.py: transmission is its v2), so a
clear dielectric such as glass draws as its pale emergent colour, not as a
window. The web viewer's path tracer reads the same SceneSpec.
"""
from __future__ import annotations

import math

from .scene_export import _bake_material, _bounding_sphere, _default_lighting, _shape_to_dict, _suggest_camera


def scene_from_voxel_field(field, name: str = "voxel build") -> dict:
    """Serialize a ``VoxelField`` into a SceneSpec with one Voxel leaf per material."""
    import numpy as np
    import scipy.ndimage as ndi
    from ..kernel.voxel import Voxel

    pitch = float(field.voxel_size)
    dby = field.density_by_label or {}
    leaves, materials = [], {}
    for k, mat in enumerate(field.materials):
        if k == 0:
            continue                                  # air: empty space, never drawn
        mask = field.label_grid == k
        cells = int(mask.sum())
        if not cells:
            continue
        sdf = (ndi.distance_transform_edt(~mask) - ndi.distance_transform_edt(mask)) * pitch
        shape = Voxel(sdf.astype(np.float32), pitch, center=field.center,
                      volume_m3=cells * pitch ** 3, source=f"{mat}: {cells} cells")
        leaf = {"op": "add", "material": mat, "shape": _shape_to_dict(shape)}
        bs = _bounding_sphere(shape)
        if bs is not None:
            leaf["bounds"] = {"center": list(bs[0]), "radius": bs[1]}
        leaves.append(leaf)
        materials[mat] = _bake_material(mat, dby.get(mat))

    nx, ny, nz = field.label_grid.shape
    cx, cy, cz = field.center
    bbox = [[cx - nx * pitch / 2, cx + nx * pitch / 2], [cy - ny * pitch / 2, cy + ny * pitch / 2],
            [cz - nz * pitch / 2, cz + nz * pitch / 2]]
    cam = _suggest_camera(bbox)
    lighting = _default_lighting(cam["up"])
    return {
        "name": name, "csg_leaves": leaves, "materials": materials,
        "physics": {"mass_kg": field.mass_kg, "com_m": list(field.com_m),
                    "inertia_kgm2": list(field.inertia_kgm2)},
        "bbox": bbox, "camera": cam, "lights": lighting["lights"], "ambient": lighting["ambient"],
        "identified": True, "source": field.notes,
    }


def render_voxel_field(field, path: str, *, theater: str = "plain", name: str = "voxel build",
                       width: int = 320, height: int = 240,
                       azimuth_deg: float = 35.0, elevation_deg: float = 35.0) -> str:
    """Stage a ``VoxelField`` in a theater and render it to a PNG. Returns the path.

    The camera aims where the theater's own camera does, at the theater's orbit
    radius; ``azimuth_deg`` / ``elevation_deg`` place it (z-up, elevation from
    the horizon). The default 35° looks down into open containers.
    """
    from ..kernel.vec import Vec3
    from .camera import Camera
    from .render import render_to_png
    from .scene_export import scene_from_spec
    from .theaters import stage

    staged = stage(scene_from_voxel_field(field, name), theater)
    cam = staged["camera"]
    target = Vec3(*cam["target"])
    r, a, e = cam["orbit_radius"], math.radians(azimuth_deg), math.radians(elevation_deg)
    eye = target + Vec3(r * math.cos(e) * math.sin(a), -r * math.cos(e) * math.cos(a), r * math.sin(e))
    camera = Camera(eye, target, up=Vec3(*cam["up"]), fov_deg=cam["fov_deg"], width=width, height=height)
    return render_to_png(scene_from_spec(staged), camera, path)


__all__ = ["scene_from_voxel_field", "render_voxel_field"]
