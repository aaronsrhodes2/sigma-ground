"""Camera — generates the primary ray through each pixel, and orbits the scene.

A pinhole camera: an eye point, a look direction, and a field of view. For each
pixel we build the ray that leaves the eye through that pixel. `orbit_eye` places
the eye on a circle around a target so a turntable (walk around it) is just a
sweep of azimuth — the deterministic pre-compute we replay at wall-clock.
"""
from __future__ import annotations

import math

from ..dynamics.vec import Vec3


class Camera:
    """Pinhole camera. World convention: x=right, y=up, z=toward-viewer."""

    def __init__(self, eye: Vec3, target: Vec3, up: Vec3 = None,
                 fov_deg: float = 40.0, width: int = 200, height: int = 150):
        self.eye = eye
        self.width = int(width)
        self.height = int(height)
        up = up if up is not None else Vec3(0, 1, 0)
        self.forward = (target - eye).normalized()
        self.right = self.forward.cross(up).normalized()
        self.up = self.right.cross(self.forward).normalized()
        self.aspect = self.width / self.height
        self.tan_half = math.tan(math.radians(fov_deg) * 0.5)

    def ray(self, px: int, py: int):
        """Return (origin, direction) for the ray through pixel-center (px, py)."""
        sx = (2.0 * (px + 0.5) / self.width - 1.0) * self.aspect * self.tan_half
        sy = (1.0 - 2.0 * (py + 0.5) / self.height) * self.tan_half
        direction = (self.forward + self.right * sx + self.up * sy).normalized()
        return self.eye, direction


def orbit_eye(target: Vec3, radius: float, azimuth_deg: float,
              elevation_deg: float = 18.0) -> Vec3:
    """Eye position on a circle of given radius around `target`."""
    a = math.radians(azimuth_deg)
    e = math.radians(elevation_deg)
    return target + Vec3(radius * math.cos(e) * math.sin(a),
                         radius * math.sin(e),
                         radius * math.cos(e) * math.cos(a))
