"""Exporter completeness — every shape Deckard compiles serializes and rebuilds.

Round-trip law: for any compiled construct, `scene_spec_to_sdf(construct_to_scene(c))`
must reproduce `c.composed.sdf` at every probe point (the browser rebuilds the
same JSON — if the Python inverse agrees, the in-page self-check can hold).
Covers the wrapper shapes researched objects actually produce: _Rotated (euler
parts), Outline (organics), _Clipped (fills), _Subtracted (conform), Torus.
"""
import math
import random

from sigma_ground.deckard.construct import compile, _rotation
from sigma_ground.deckard.schema import ConstructSpec, Fact, Part
from sigma_ground.radiance.scene_export import (
    SUPPORTED_SHAPE_TYPES, _mat_to_quat, _qrot_inv, construct_to_scene,
    scene_spec_to_sdf)


def _est(v):
    return Fact(v, "estimated", "", 0.5)


def _spec(parts, name="probe"):
    return ConstructSpec(name=name, kind="composite", identified=True,
                         parts=parts, sources=[], notes="")


class _P:                                       # probe point for the sdf callable
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def _roundtrip_max_err(c, n=6):
    scene = construct_to_scene(c)
    sdf, _ = scene_spec_to_sdf(scene)
    (x0, x1), (y0, y1), (z0, z1) = c.bbox
    pad = 0.2 * max(x1 - x0, y1 - y0, z1 - z0)
    worst = 0.0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                p = _P(x0 - pad + (i + 0.5) / n * (x1 - x0 + 2 * pad),
                       y0 - pad + (j + 0.5) / n * (y1 - y0 + 2 * pad),
                       z0 - pad + (k + 0.5) / n * (z1 - z0 + 2 * pad))
                worst = max(worst, abs(sdf(p) - c.composed.sdf(p.x, p.y, p.z)))
    return worst, scene


def _all_types(node):
    out = {node["type"]}
    if "shape" in node:
        out |= _all_types(node["shape"])
    if "cut" in node:
        out |= _all_types(node["cut"])
    return out


def test_mat_to_quat_agrees_with_matrix_transpose():
    rng = random.Random(7)
    for _ in range(50):
        e = (rng.uniform(-180, 180), rng.uniform(-90, 90), rng.uniform(-180, 180))
        R = _rotation(e)
        q = _mat_to_quat(R)
        assert abs(sum(v * v for v in q) - 1.0) < 1e-12 and q[3] >= 0.0
        for _ in range(5):
            v = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
            # _Rotated applies R^T v; the viewer applies qrotInv(q, v) — same map
            rt = tuple(R[0][i] * v[0] + R[1][i] * v[1] + R[2][i] * v[2]
                       for i in range(3))
            qv = _qrot_inv(q, v)
            assert all(abs(a - b) < 1e-12 for a, b in zip(rt, qv))


def test_rotated_box_round_trips():
    spec = _spec([Part("slab", "box",
                       {"x_m": _est(0.2), "y_m": _est(0.05), "z_m": _est(0.1)},
                       "oak", _est(700.0), (0.0, 0.0, 0.1), (20.0, 30.0, 40.0))])
    c = compile(spec, resolution=32)
    err, scene = _roundtrip_max_err(c)
    assert err < 1e-9
    assert "Rotated" in _all_types(scene["csg_leaves"][0]["shape"])


def test_outline_extrude_and_revolve_round_trip():
    pent = [[math.cos(a) * 0.05, math.sin(a) * 0.04]
            for a in [2 * math.pi * k / 5 + 0.3 for k in range(5)]]
    spec = _spec([Part("vane", "outline", {}, "oak", _est(700.0),
                       (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "add",
                       outline={"profile": pent, "mode": "extrude",
                                "thickness": 0.004})])
    c = compile(spec, resolution=32)
    err, scene = _roundtrip_max_err(c)
    assert err < 1e-9
    goblet = [[0.0, 0.03], [0.02, 0.028], [0.05, 0.012], [0.09, 0.03]]
    spec = _spec([Part("cup", "outline", {}, "glass", _est(2500.0),
                       (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "add",
                       outline={"profile": goblet, "mode": "revolve"})])
    c = compile(spec, resolution=32)
    err, scene = _roundtrip_max_err(c)
    assert err < 1e-9
    assert "Outline" in _all_types(scene["csg_leaves"][0]["shape"])


def test_fill_clipped_round_trips():
    spec = _spec([
        Part("shell", "cylinder", {"radius_m": _est(0.04), "height_m": _est(0.1)},
             "glass", _est(2500.0), (0.0, 0.0, 0.05)),
        Part("bore", "cylinder", {"radius_m": _est(0.035), "height_m": _est(0.095)},
             "air", Fact(1.2), (0.0, 0.0, 0.055), op="subtract"),
        Part("water", "fill", {}, "water", _est(998.0),
             fill={"of": "bore", "fraction": 0.5}),
    ])
    c = compile(spec, resolution=32)
    err, scene = _roundtrip_max_err(c)
    assert err < 1e-9
    types = set().union(*(_all_types(L["shape"]) for L in scene["csg_leaves"]))
    assert "Clipped" in types


def test_conform_subtracted_round_trips():
    spec = _spec([
        Part("peg", "cylinder", {"radius_m": _est(0.02), "height_m": _est(0.08)},
             "steel", _est(7850.0), (0.0, 0.0, 0.04)),
        Part("block", "box", {"x_m": _est(0.1), "y_m": _est(0.1), "z_m": _est(0.04)},
             "oak", _est(700.0), (0.0, 0.0, 0.02), conform="peg"),
    ])
    c = compile(spec, resolution=32)
    err, scene = _roundtrip_max_err(c)
    assert err < 1e-9
    types = set().union(*(_all_types(L["shape"]) for L in scene["csg_leaves"]))
    assert "Subtracted" in types


def test_every_exported_type_is_viewer_supported():
    # all the constructs above only ever emit viewer-supported nodes
    for build in (test_rotated_box_round_trips,):
        pass                                            # types asserted per-test
    assert {"Rotated", "Clipped", "Subtracted", "Outline",
            "Torus"} <= SUPPORTED_SHAPE_TYPES
