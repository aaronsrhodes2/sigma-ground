"""Secondary-ray ambient occlusion gates -- Teardown Phase 4 (part 2).
The verified real mechanism behind Teardown's lighting (research finding:
no global illumination, just a few cosine-weighted rays whose hit distance
sets darkening), expressed with this renderer's own march() as the
secondary-ray engine. Opt-in via RadianceScene(ao_rays=N): 0 rays must
leave every existing render byte-identical.
"""
import math

import pytest

from sigma_ground.dynamics.vec import Vec3
from sigma_ground.radiance.scene import RadianceScene
from sigma_ground.radiance.shade import shade, ambient_occlusion


def _floor_sdf(p):
    return p.y                                     # solid half-space y < 0


def _corner(p):
    """Inside corner: solid where y<0 (floor) OR x<0 (wall). SDF of a
    union is the min of the two half-space SDFs."""
    return min(p.y, p.x)


def test_flat_plane_hemisphere_is_fully_open():
    """A point on an infinite floor sees an empty hemisphere: AO == 1
    exactly (every secondary ray escapes to the reach limit)."""
    p = Vec3(0.0, 0.0, 0.0)
    n = Vec3(0.0, 1.0, 0.0)
    ao = ambient_occlusion(_floor_sdf, p, n, rays=16, reach=1.0)
    assert ao == pytest.approx(1.0)


def test_inside_corner_converges_to_the_closed_form_half():
    """The plan's closed-form corner case, stated correctly: the analytic
    limit for a point ON the wall-floor corner line is AO = 0.5 (half the
    hemisphere blocked at zero distance). At a FINITE standoff the
    distance-weighted visibility (hit at t -> visibility t/reach) pulls
    the value above 0.5 — measured 0.66 at 20mm standoff, 0.55 at 2mm,
    0.53 at 0.5mm with reach=1.0 — so the honest gate is CONVERGENCE
    toward 0.5 from above as the standoff shrinks, plus the near-limit
    value landing close to the closed form."""
    n = Vec3(0.0, 1.0, 0.0)

    def corner_union(q):
        return min(q.y, q.x)                       # solid: y<0 OR x<0

    ao_far = ambient_occlusion(corner_union, Vec3(0.02, 0, 0), n,
                               rays=64, reach=1.0)
    ao_mid = ambient_occlusion(corner_union, Vec3(0.002, 0, 0), n,
                               rays=64, reach=1.0)
    ao_near = ambient_occlusion(corner_union, Vec3(0.0005, 0, 0), n,
                                rays=64, reach=1.0)
    assert ao_far > ao_mid > ao_near               # monotone toward the limit
    assert ao_near == pytest.approx(0.5, abs=0.1)  # near the closed form
    assert ao_near >= 0.5 - 1e-6                   # from ABOVE, never below


def test_ao_is_deterministic():
    p = Vec3(0.01, 0.0, 0.0)
    n = Vec3(0.0, 1.0, 0.0)
    a1 = ambient_occlusion(_corner, p, n, rays=32, reach=0.5)
    a2 = ambient_occlusion(_corner, p, n, rays=32, reach=0.5)
    assert a1 == a2


def test_shade_with_zero_ao_rays_is_identical_to_before():
    """The opt-in contract: ao_rays=0 (the default) must not change a
    single output value -- every existing render stays reproducible."""
    scene_off = RadianceScene.from_sdf(_floor_sdf, "iron")
    scene_default = RadianceScene.from_sdf(_floor_sdf, "iron")
    assert scene_default.ao_rays == 0
    p, n, v = Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)
    c_off = shade(scene_off, p, n, v)
    c_default = shade(scene_default, p, n, v)
    assert (c_off.x, c_off.y, c_off.z) == (c_default.x, c_default.y, c_default.z)


def test_shade_ao_darkens_corner_but_not_open_floor():
    """End-to-end through shade(): with AO on, an open floor point is
    unchanged (AO=1) while a corner point loses part of its AMBIENT term
    only -- the directional term must be untouched (occlusion stands in
    for blocked indirect light, not for shadowing the light source)."""
    v = Vec3(0.0, 0.0, 1.0)
    n = Vec3(0.0, 1.0, 0.0)

    flat_on = RadianceScene.from_sdf(_floor_sdf, "iron", ao_rays=16)
    flat_off = RadianceScene.from_sdf(_floor_sdf, "iron")
    p_open = Vec3(0.0, 0.0, 0.0)
    c_on = shade(flat_on, p_open, n, v)
    c_off = shade(flat_off, p_open, n, v)
    assert c_on.x == pytest.approx(c_off.x, abs=1e-12)

    corner_on = RadianceScene.from_sdf(_corner, "iron", ao_rays=32)
    corner_off = RadianceScene.from_sdf(_corner, "iron")
    p_corner = Vec3(0.02, 0.0, 0.0)
    d_on = shade(corner_on, p_corner, n, v)
    d_off = shade(corner_off, p_corner, n, v)
    assert d_on.x < d_off.x                        # darker with AO
    # the loss is bounded by the full ambient term: removing ALL ambient
    # is the theoretical floor of what AO may take away
    max_loss = corner_off.ambient
    assert d_off.x - d_on.x <= max_loss * 1.0 + 1e-9
