"""Render — march every pixel, shade the hits, write the image.

This is the loop that turns a RadianceScene + Camera into pixels. It is pure
Python (slow, but exact and zero-dependency): each pixel casts one ray, sphere-
traces the field, and shades the hit from the material's emergent color. Empty
rays return the black background.

Because the engine is deterministic, the same scene + camera always produce the
same image — so a turntable (walk around it) is a deterministic pre-compute we
render once and replay at wall-clock. Real-time *playback*, not a real-time
solver.
"""
from __future__ import annotations

import os

from .raymarch import march, surface_normal
from .shade import shade
from .image import write_png


def render(scene, camera) -> bytearray:
    """Return width*height*3 RGB bytes for one camera view."""
    buf = bytearray(camera.width * camera.height * 3)
    i = 0
    for py in range(camera.height):
        for px in range(camera.width):
            origin, direction = camera.ray(px, py)
            t = march(scene.sdf, origin, direction, scene.max_dist)
            if t is None:
                color = scene.background
            else:
                hit = origin + direction * t
                n = surface_normal(scene.sdf, hit)
                color = shade(scene, hit, n, -direction)
            r, g, b = color.to_rgb()
            buf[i] = r
            buf[i + 1] = g
            buf[i + 2] = b
            i += 3
    return buf


def render_to_png(scene, camera, path: str) -> str:
    """Render one view and write it as a PNG. Returns the path."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    write_png(path, camera.width, camera.height, bytes(render(scene, camera)))
    return path


def render_turntable(scene, target, radius, path_prefix, n_frames=24,
                     width=200, height=150, fov_deg=40.0, elevation_deg=18.0):
    """Render an orbit around `target` — one frame per azimuth step."""
    from .camera import Camera, orbit_eye
    paths = []
    for k in range(n_frames):
        az = 360.0 * k / n_frames
        cam = Camera(orbit_eye(target, radius, az, elevation_deg), target,
                     fov_deg=fov_deg, width=width, height=height)
        paths.append(render_to_png(scene, cam, f"{path_prefix}_{k:03d}.png"))
    return paths


_ASCII = " .:-=+*#%@"


def render_ascii(scene, camera) -> str:
    """A terminal preview — luminance to ASCII, for quick eyeballing."""
    rows = []
    for py in range(camera.height):
        line = []
        for px in range(camera.width):
            origin, direction = camera.ray(px, py)
            t = march(scene.sdf, origin, direction, scene.max_dist)
            if t is None:
                line.append(" ")
            else:
                hit = origin + direction * t
                n = surface_normal(scene.sdf, hit)
                c = shade(scene, hit, n, -direction)
                lum = 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
                line.append(_ASCII[min(len(_ASCII) - 1, int(lum * len(_ASCII)))])
        rows.append("".join(line))
    return "\n".join(rows)
