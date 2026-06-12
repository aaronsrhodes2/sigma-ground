"""Radiance — the renderer.

Turns shaped matter into pixels by sphere-tracing the same analytic SDF the
physics uses, shading from first-principles material color, black wherever
there is no matter. Pure Python, zero external dependencies (stdlib `zlib` for
PNG). It draws only the engine's numbers — never a faked frame.

Layer position (see misc/MATERIA_STACK.md):
    Deckard (shape) → field (energy) → Materia (movement) → **Radiance** (render)

Quick start:
    from sigma_ground.shapes import Sphere
    from sigma_ground.radiance import RadianceScene, Camera, orbit_eye, render_to_png
    from sigma_ground.dynamics.vec import Vec3
    scene = RadianceScene.from_shape(Sphere(0.06), "copper")
    cam = Camera(orbit_eye(Vec3(0,0,0), 0.25, 35), Vec3(0,0,0))
    render_to_png(scene, cam, "copper.png")
"""
from __future__ import annotations

from .scene import RadianceScene
from .camera import Camera, orbit_eye
from .raymarch import march, surface_normal
from .shade import shade, material_albedo
from .render import render, render_to_png, render_turntable, render_ascii
from .image import write_png
from .scene_export import (construct_to_scene, scene_spec_to_sdf,
                           scene_from_spec)
from .trajectory import (record_fall, record_object_fall, record_descent,
                         record_horizontal_run)

__all__ = [
    "construct_to_scene",
    "scene_spec_to_sdf",
    "scene_from_spec",
    "record_fall",
    "record_object_fall",
    "record_descent",
    "record_horizontal_run",
    "RadianceScene",
    "Camera",
    "orbit_eye",
    "march",
    "surface_normal",
    "shade",
    "material_albedo",
    "render",
    "render_to_png",
    "render_turntable",
    "render_ascii",
    "write_png",
]
