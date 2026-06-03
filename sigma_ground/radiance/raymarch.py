"""Sphere tracing — the heart of an SDF renderer.

Given a signed-distance field `sdf(Vec3) -> float` (negative inside, the
sigma-ground convention), a ray is advanced by exactly the distance to the
nearest surface each step. That distance is a guaranteed-safe step (the surface
can be no closer), so the ray never tunnels through anything. When the distance
drops below `eps`, we've hit the surface.

No polygons, no triangles — the same analytic field the physics uses for mass
and collision is what we draw.
"""
from __future__ import annotations

from ..dynamics.vec import Vec3


def march(sdf, origin: Vec3, direction: Vec3,
          max_dist: float = 50.0, max_steps: int = 96, eps: float = 1e-4):
    """Advance a ray until it hits the surface. Returns distance t, or None."""
    t = 0.0
    for _ in range(max_steps):
        d = sdf(origin + direction * t)
        if d < eps:
            return t
        t += d if d > eps else eps      # step by the safe distance
        if t > max_dist:
            return None
    return None


def surface_normal(sdf, p: Vec3, h: float = 1e-4) -> Vec3:
    """Outward normal = ∇(sdf), by central differences. The field's gradient."""
    return Vec3(
        sdf(p + Vec3(h, 0, 0)) - sdf(p - Vec3(h, 0, 0)),
        sdf(p + Vec3(0, h, 0)) - sdf(p - Vec3(0, h, 0)),
        sdf(p + Vec3(0, 0, h)) - sdf(p - Vec3(0, 0, h)),
    ).normalized()
