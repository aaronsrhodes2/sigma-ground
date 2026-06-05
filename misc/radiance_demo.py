"""Radiance demo — sphere-trace the SDF, shade from emergent material color.

Prints emergent metal colors, an ASCII preview (so you see something in the
terminal immediately), then writes a real PNG of a CSG construct: a copper
sphere with a cylinder bored through it — geometry the physics layer would
also weigh and collide, drawn by marching the same field.
"""
import os
import sys
import time

sys.path.insert(0, r"D:\Aaron\development\sigma-ground")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sigma_ground.shapes import Sphere, Cylinder
from sigma_ground.csg import sdf_subtract
from sigma_ground.dynamics.vec import Vec3
from sigma_ground.radiance import (RadianceScene, Camera, orbit_eye,
                                   render_to_png, render_ascii, material_albedo)

print("=" * 64)
print("RADIANCE — emergent color, sphere-traced, zero-dependency")
print("=" * 64)

print("\nEmergent metal color (derived from the Drude/Fresnel response):")
for mk in ("copper", "gold", "iron", "silver", "lead"):
    print(f"  {mk:8s} {material_albedo(mk).to_hex()}")

# ── ASCII preview: a copper sphere ──────────────────────────────────────
sphere = Sphere(0.06)
scene = RadianceScene.from_shape(sphere, "copper")
cam_ascii = Camera(orbit_eye(Vec3(0, 0, 0), 0.26, 35.0, 20.0), Vec3(0, 0, 0),
                   fov_deg=42.0, width=64, height=28)
print("\nASCII preview — copper sphere:\n")
print(render_ascii(scene, cam_ascii))

# ── PNG: a CSG construct (copper sphere with a bore) ────────────────────
bore = Cylinder(0.028, 0.4)                        # axis along z
def bored(p):
    return sdf_subtract(sphere.surface_distance(p.x, p.y, p.z),
                        bore.surface_distance(p.x, p.y, p.z))

scene2 = RadianceScene.from_sdf(bored, "copper")
cam = Camera(orbit_eye(Vec3(0, 0, 0), 0.26, 40.0, 22.0), Vec3(0, 0, 0),
             fov_deg=42.0, width=200, height=150)

out = os.path.join(r"D:\Aaron\development\sigma-ground\misc\renders",
                   "radiance_bored_copper_sphere.png")
print(f"\nRendering 200×150 CSG construct → PNG …")
t0 = time.time()
path = render_to_png(scene2, cam, out)
print(f"  wrote {path}  ({time.time() - t0:.1f}s)")
print("  (turntable orbit available via radiance.render_turntable)")
