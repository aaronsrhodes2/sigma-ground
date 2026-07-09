"""The Python renderer's incandescence — the twin of viewer.js's GLSL.

Planck × Kirchhoff at (650, 550, 450) nm with the Draper cutoff; the tone-map
is a flagged exposure choice. A literal-parity test greps viewer.js for the
same constants so the two renderers cannot drift apart silently (the same
discipline scene_export.py applies to SUPPORTED_SHAPE_TYPES).
"""
import pathlib

import pytest

from sigma_ground.dynamics.vec import Vec3
from sigma_ground.radiance.shade import incandescence, shade
from sigma_ground.radiance.scene_export import scene_from_spec

_VIEWER = pathlib.Path(
    r"D:\Aaron\development\sigma-ground\sigma_ground\radiance\web\viewer.js")


def test_draper_cutoff_is_exactly_zero():
    dark = incandescence(699.9, (1.0, 1.0, 1.0))
    assert (dark.x, dark.y, dark.z) == (0.0, 0.0, 0.0)   # exact zero, not merely dim
    glow = incandescence(1000.0, (1.0, 1.0, 1.0))
    assert glow.x > 0.0                                # above Draper: it glows


def test_planck_hue_ordering():
    """Hotter = bluer: red dominates at 1000 K; the g/r and b/r ratios RISE
    with temperature (the Planck spectrum shifting, not a colour table)."""
    lo = incandescence(1000.0, (1.0, 1.0, 1.0))
    hi = incandescence(2500.0, (1.0, 1.0, 1.0))
    assert lo.x > lo.y > lo.z                          # 1000 K: deep red-orange
    assert hi.y / hi.x > lo.y / lo.x                   # hotter → greener
    assert hi.z / hi.x > lo.z / lo.x                   # hotter → bluer


def test_emissivity_scales_the_glow():
    full = incandescence(1500.0, (1.0, 1.0, 1.0))
    half = incandescence(1500.0, (0.5, 0.5, 0.5))
    assert half.x < full.x                             # ε<1 emits less (Kirchhoff)


def test_constants_match_viewer_js():
    """The drift guard: the exact literals the GLSL bakes must appear in
    viewer.js. If someone retunes one renderer, this fails until the twin
    follows."""
    js = _VIEWER.read_text(encoding="utf-8")
    for lit in ("2400000000.0",       # EMISSION_SCALE
                "700.0",              # Draper cutoff
                "1.4388e-2",          # c2 = hc/kB
                "*1.7"):              # the flagged exposure gain
        assert lit in js, f"viewer.js lost the shared constant {lit!r}"


def _sphere_spec(T_k=None, physics_env=None):
    leaf = {"op": "add", "material": "iron",
            "shape": {"type": "Sphere", "center": [0, 0, 0], "radius": 0.05}}
    if T_k is not None:
        leaf["temperature_k"] = T_k
    spec = {"csg_leaves": [leaf],
            "materials": {"iron": {"color_rgb": [0.9, 0.9, 0.9]}}}
    if physics_env:
        spec["physics_env"] = physics_env
    return spec


def _shade_surface(spec):
    scene = scene_from_spec(spec)
    p = Vec3(0.05, 0.0, 0.0)                          # a point ON the sphere
    n = Vec3(1.0, 0.0, 0.0)
    return shade(scene, p, n, Vec3(-1.0, 0.0, 0.0))


def test_scene_from_spec_wires_the_thermal_hooks():
    cold = _shade_surface(_sphere_spec(T_k=293.15))   # hooks active, below Draper
    hot = _shade_surface(_sphere_spec(T_k=1700.0))    # a hot copper-ball still
    none = _shade_surface(_sphere_spec())             # no temperature anywhere
    assert cold.x == pytest.approx(none.x, abs=1e-12)  # cold ≡ pre-thermal output
    assert hot.x > cold.x                              # the glow ADDS light
    # env datum flows through: a deep-space scene's leafless default is 2.725 K
    scene = scene_from_spec(_sphere_spec(T_k=900.0,
                                         physics_env={"temperature_k": 2.725}))
    assert scene.temperature_at(Vec3(2.0, 2.0, 2.0)) == pytest.approx(2.725)
    assert scene.temperature_at(Vec3(0.0, 0.0, 0.0)) == pytest.approx(900.0)


def test_bake_then_shade_end_to_end():
    """A frame-baked scene renders the interpolated temperature — the Python
    ground truth for any scrub position."""
    from sigma_ground.radiance.trajectory import bake_frame_temperatures
    frames = [{"t_sim": 0.0, "bodies": [{"pos": [0, 0, 0], "quat": [0, 0, 0, 1],
                                         "temperature_k": 288.15}]},
              {"t_sim": 1.0, "bodies": [{"pos": [0, 0, 0], "quat": [0, 0, 0, 1],
                                         "temperature_k": 1711.85}]}]
    spec = _sphere_spec()
    spec["csg_leaves"][0]["body"] = 0
    spec["bodies"] = [{"pivot": [0, 0, 0], "label": "iron"}]
    bundle = {"scene": spec, "trajectory": {"frames": frames}}
    baked = bake_frame_temperatures(bundle, 0.5)
    assert baked["csg_leaves"][0]["temperature_k"] == pytest.approx(1000.0)
    scene = scene_from_spec(baked)
    assert scene.temperature_at(Vec3(0, 0, 0)) == pytest.approx(1000.0)
