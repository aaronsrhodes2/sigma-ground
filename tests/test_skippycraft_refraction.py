"""The refraction construction of the SkippyCraft mod (stage/skippycraft/mod/.../Refraction.java), mirrored here with sigma-ground's own
optics (the mod reads the same numbers from the record's table): an object under a flat surface appears at depth d/n for a near-vertical
view, and shallower-looking less so at oblique angles. A check on the geometry, not a second implementation to ship."""
import math

import pytest

from sigma_ground.field.interface import optics


def apparent_depth(n, h, depth, x):
    """Surface at 0, camera ``h`` above, vertex ``depth`` below and ``x`` to the side: the depth of the vertex's apparent position."""
    lo, hi = 0.0, x
    for _ in range(60):
        u = (lo + hi) / 2
        _, cos_t = optics.fresnel_interface(1.0, n, h / math.hypot(h, u))
        land = u + depth * math.sqrt(1 - cos_t ** 2) / cos_t
        lo, hi = (lo, u) if land > x else (u, hi)
    u = (lo + hi) / 2
    cs = math.hypot(h, u)
    return cs * (x / u) * h / cs - h


@pytest.mark.parametrize("material", ["water", "glass"])
def test_a_near_vertical_view_sees_an_object_at_depth_over_n(material):
    n = optics.cauchy_n({"water": "water", "glass": "crown_glass"}[material], optics.LAMBDA_G)
    assert apparent_depth(n, 10.0, 3.0, 0.05) == pytest.approx(3.0 / n, rel=2e-3)


def test_an_oblique_view_sees_it_shallower_still():
    n = optics.cauchy_n("water", optics.LAMBDA_G)
    depths = [apparent_depth(n, 10.0, 3.0, x) for x in (0.05, 1.0, 3.0, 6.0)]
    assert all(a > b for a, b in zip(depths, depths[1:])) and depths[-1] < 3.0 / n
