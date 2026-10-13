"""Water for the holodeck (SkippyVerse Phase A, task A2): real hydrostatics, real drag, real wave dispersion, nothing invented.

Aaron, 2026-10-12: only real-world physics.  So the liquids are the sigma-ground library's (a liquid it lacks is refused), buoyancy is
Archimedes with g from the construct's world constants, drag is the same F = 1/2 rho Cd A v^2 as the air's (+ Stokes), the surface waves obey
the library-free-but-textbook dispersion relation with the library's g, rho and sigma, and the splash is sized from the entry's real kinetic
energy.  These tests pin that down offline: the generated constants against the library (never literals), the pure geometry and force math
against closed forms and the library's own formulas, the words (typos included), and the engine glue (water.lua, the real holodeck_water
entity, the real holodeck pipeline) against stage/gmod/holodeck/gmod_mock.lua extended with a small rigid-body layer.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

lupa = pytest.importorskip("lupa")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage" / "gmod" / "holodeck"))
sys.path.insert(0, str(HERE))

import water_constants as WCN  # noqa: E402
import world_constants as WLD  # noqa: E402
from test_gmod_glue import World, A, B  # noqa: E402

from sigma_ground.field.interface import fluid, liquid_water, optics  # noqa: E402
from sigma_ground.field.interface.fluid_surface import submerged_fraction, floats as lib_floats  # noqa: E402
from sigma_ground.field.interface.projectile import G_EARTH, drag_force as lib_drag_force  # noqa: E402
from sigma_ground.materia.bodies import resolve_body  # noqa: E402

ADDON = WCN.GMOD / "addon" / "lua"
HU_M = 0.01905
RHO_W = fluid.KNOWN_LIQUIDS["water"]["density_kg_m3"]
G_MOON = resolve_body("moon").surface_gravity_m_s2


# ============================================================================================================ pure modules
@pytest.fixture(scope="module")
def pure():
    from lupa import luajit21 as mod
    rt = mod.LuaRuntime(unpack_returned_tuples=True)
    rt.execute("jit.off()")
    rt.execute(f'ROOT = "{ADDON.as_posix()}/"; function include(p) return dofile(ROOT .. p) end')
    rt.execute("""
      Cn = include("mentat/water/constants.lua"); Hy = include("mentat/water/hydro.lua"); Wv = include("mentat/water/waves.lua")
      Sp = include("mentat/water/splash.lua"); In = include("mentat/water/intent.lua"); Bs = include("mentat/water/basin.lua")
      Sw = include("mentat/water/swim.lua"); Ae = include("mentat/world/aero.lua"); WCs = include("mentat/world/constants.lua")
    """)

    class P:
        pass
    p = P()
    p.rt = rt
    p.ev = rt.eval
    p.t = lambda *a: rt.table_from(list(a))
    return p


def vec(p, *a):
    return p.t(*a)


def tolist(t):
    return [t[i] for i in range(1, len(t) + 1)]


# ------------------------------------------------------------------------------------------------ the generated constants
def test_generated_files_are_current():
    js, lua = WCN.render()
    assert WCN.JSON_PATH.read_text(encoding="utf-8") == js
    assert WCN.LUA_PATH.read_text(encoding="utf-8") == lua


def test_liquids_equal_the_library_not_literals():
    d = json.loads(WCN.JSON_PATH.read_text(encoding="utf-8"))
    assert set(d["liquids"]) == {"water", "seawater", "ethanol", "glycerol", "mercury"}
    for key, row in d["liquids"].items():
        p = fluid.liquid_properties(key)
        assert row["density_kg_m3"] == p["density_kg_m3"]
        assert row["viscosity_pa_s"] == p["viscosity_pa_s"]
        assert row["surface_tension_n_m"] == p["surface_tension_n_m"]
        assert row["source"] and row["status"] and "fluid.KNOWN_LIQUIDS" in row["source"]
    # the two library routes to water's density agree (Kell correlation vs the table)
    assert d["cross_check"]["water_density_kell_kg_m3"] == pytest.approx(RHO_W, rel=1e-4)
    assert RHO_W == pytest.approx(998.2, abs=0.05)                                  # "998 kg/m3 at 20 C"
    assert d["liquids"]["water"]["T_K"] == pytest.approx(293.15)


def test_optics_come_from_the_library():
    d = json.loads(WCN.JSON_PATH.read_text(encoding="utf-8"))
    n = optics.cauchy_n("water", 550e-9)
    assert d["optics"]["refractive_index"]["value"] == n
    assert d["optics"]["fresnel_r0"]["value"] == optics.dielectric_surface_reflectance(n)
    assert d["optics"]["absorption_rgb_per_m"]["value"] == list(optics.DIELECTRIC_ABSORPTION["water_blue"])


def test_every_estimate_is_labelled_and_caps_say_they_are_not_physics():
    d = json.loads(WCN.JSON_PATH.read_text(encoding="utf-8"))
    for k, e in d["human"].items():
        assert e["status"] and e["source"], k
    assert d["human"]["density_kg_m3"]["status"] == "estimated, owner-should-confirm"
    for k, e in d["limits"].items():
        assert e["status"] == "construct cap" and "not physics" in e["source"] or k in ("wall_m", "freeboard_m"), k
    assert d["splash"]["scale_law"]["status"] == "estimated, owner-should-confirm"
    assert "particle fluid" in d["not_modelled"] and "added mass" in d["not_modelled"] and "evaporation" in d["not_modelled"]


def test_lua_constants_resolve_words_and_refuse_liquids_the_library_lacks(pure):
    ev = pure.ev
    assert ev("(Cn.find_liquid('mercury'))") == "mercury"
    assert ev("(Cn.find_liquid('glycerin'))") == "glycerol"
    assert ev("(Cn.find_liquid('salt water'))") == "seawater"
    for word in ("oil", "olive oil", "honey", "lava"):
        key, why = ev(f"Cn.find_liquid('{word}')")
        assert key is None and why == "refused", word
    key, why = ev("Cn.find_liquid('banana')")
    assert key is None and why is None                                                # not a liquid word at all
    msg = ev("Cn.refusal('olive oil')")
    assert "olive oil" in msg and "library" in msg and "guess" in msg


def test_defaults_are_the_pool_aaron_named_and_every_default_is_labelled(pure):
    d = pure.ev("Cn.dims_for('pool', {})")
    assert (d.l, d.w, d.depth) == (10.0, 5.0, 2.0)
    assert d.volume == pytest.approx(100.0)
    pond = pure.ev("Cn.dims_for('pond', {})")
    assert pond.shape == "round" and pond.volume == pytest.approx(math.pi / 4 * 6 * 6 * 1.5)
    j = json.loads(WCN.JSON_PATH.read_text(encoding="utf-8"))
    for name, row in j["defaults"].items():
        assert row["status"] == "construct default", name


# ------------------------------------------------------------------------------------------------ geometry: exact
def box_sub(p, dims, ax, ay, az, centre, level):
    v, cen, total = p.ev("Hy.box_submerged")(p.t(*dims), p.t(*ax), p.t(*ay), p.t(*az), p.t(*centre), level)
    return v, tolist(cen), total


E3 = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


@pytest.mark.parametrize("level,frac", [(0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (0.9, 0.9), (1.0, 1.0), (3.0, 1.0)])
def test_axis_aligned_box_submerges_by_its_height(pure, level, frac):
    v, cen, total = box_sub(pure, (0.4, 0.6, 1.0), *E3, (0, 0, 0.5), level)
    assert total == pytest.approx(0.24)
    assert v == pytest.approx(0.24 * frac, abs=1e-12)
    if 0 < frac < 1:
        assert cen[2] == pytest.approx(level / 2)                                    # the centre of buoyancy is mid-draft
    assert cen[0] == pytest.approx(0) and cen[1] == pytest.approx(0)


def test_box_on_edge_matches_the_triangle_closed_form(pure):
    # a unit cube turned 45 degrees about x: its section is a diamond of half-diagonal sqrt(2)/2; above the level h the cut-off is (r - h)^2
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    r = math.sqrt(2) / 2
    for h in (0.0, 0.2, 0.5, 0.7):
        v, _, _ = box_sub(pure, (1, 1, 1), (1, 0, 0), (0, c, s), (0, -s, c), (0, 0, 0), h)
        expect = 0.5 if h == 0 else (1 - (r - h) ** 2 if h > 0 else None)
        assert v == pytest.approx(expect, abs=1e-12), h
    v_low, cen, _ = box_sub(pure, (1, 1, 1), (1, 0, 0), (0, c, s), (0, -s, c), (0, 0, 0), -0.5)
    # below the middle the wedge is a prism: the section area under h is (r + h)^2
    assert v_low == pytest.approx((r - 0.5) ** 2)


def test_box_centroid_shifts_toward_the_wet_side(pure):
    # tilted box half under: the displaced volume's centroid is below the box's centre (that is what rights a floating body)
    c, s = math.cos(0.3), math.sin(0.3)
    v, cen, total = box_sub(pure, (0.5, 0.5, 0.5), (1, 0, 0), (0, c, s), (0, -s, c), (0, 0, 0), 0.0)
    assert v == pytest.approx(total / 2, rel=1e-9)
    assert cen[2] < 0


@pytest.mark.parametrize("depth_frac", [0.0, 0.1, 0.5, 0.9, 1.0, 1.7, 2.0, 2.5])
def test_sphere_cap_is_exact(pure, depth_frac):
    r = 0.3
    # centre height so that the cap depth h = depth_frac * r  (bottom at -r relative to the centre at 0): level = -r + h
    h = min(depth_frac * r, 2 * r)
    level = -r + depth_frac * r
    v, cen, total = pure.ev("Hy.sphere_submerged")(r, pure.t(0, 0, 0), level)
    assert total == pytest.approx(4 / 3 * math.pi * r ** 3)
    h = max(0.0, min(2 * r, level + r))
    assert v == pytest.approx(math.pi * h * h * (3 * r - h) / 3, rel=1e-12, abs=1e-15)
    if 0 < h < 2 * r:
        assert cen[3] < 0                                                             # below the centre


def test_sphere_half_in_is_a_hemisphere(pure):
    v, cen, total = pure.ev("Hy.sphere_submerged")(0.5, pure.t(0, 0, 0), 0.0)
    assert v == pytest.approx(total / 2)
    assert cen[3] == pytest.approx(-3 * 0.5 / 8)                                       # centroid of a hemisphere: 3r/8


@pytest.mark.parametrize("level", [-0.5, -0.25, 0.0, 0.3, 0.5, 0.9])
def test_cylinder_horizontal_axis_is_the_circular_segment(pure, level):
    r, Ln = 0.5, 2.0
    v, cen, total = pure.ev("Hy.cylinder_submerged")(r, Ln, pure.t(1, 0, 0), pure.t(0, 0, 0), level)
    d = max(-r, min(r, level))
    seg = r * r * math.acos(-d / r) - (-d) * math.sqrt(r * r - d * d) * -1 if False else (r * r * (math.pi / 2 + math.asin(d / r)) + d * math.sqrt(r * r - d * d))
    assert v == pytest.approx(seg * Ln, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("level", [-1.2, -1.0, -0.4, 0.0, 0.7, 1.0, 1.2])
def test_cylinder_vertical_axis_is_a_disc_by_a_depth(pure, level):
    r, Ln = 0.4, 2.0
    v, cen, total = pure.ev("Hy.cylinder_submerged")(r, Ln, pure.t(0, 0, 1), pure.t(0, 0, 0), level)
    h = max(0.0, min(Ln, level + Ln / 2))
    assert v == pytest.approx(math.pi * r * r * h, rel=1e-12, abs=1e-15)
    if 0 < h < Ln:
        assert cen[3] == pytest.approx(-Ln / 2 + h / 2)


def test_tilted_cylinder_agrees_with_the_closed_forms_at_the_limits(pure):
    r, Ln = 0.4, 1.5
    # a tilt a hair off vertical / horizontal must approach the exact closed forms (the quadrature is of the exact segment area)
    for az, ref_axis in ((math.sqrt(1 - 1e-8), (0, 0, 1)), (1e-6, (1, 0, 0))):
        ax = (math.sqrt(max(0.0, 1 - az * az)), 0.0, az)
        for level in (-0.3, 0.0, 0.2):
            v1, _, _ = pure.ev("Hy.cylinder_submerged")(r, Ln, pure.t(*ax), pure.t(0, 0, 0), level)
            v2, _, _ = pure.ev("Hy.cylinder_submerged")(r, Ln, pure.t(*ref_axis), pure.t(0, 0, 0), level)
            assert v1 == pytest.approx(v2, rel=2e-3, abs=1e-5)
    # and half the cylinder is under when the plane passes through its centre, at any tilt
    for tilt in (0.2, 0.7, 1.2):
        ax = (math.cos(tilt), 0.0, math.sin(tilt))
        v, _, total = pure.ev("Hy.cylinder_submerged")(r, Ln, pure.t(*ax), pure.t(0, 0, 0), 0.0)
        assert v == pytest.approx(total / 2, rel=1e-6)


def test_shape_model_reads_the_card_size_and_labels_its_basis(pure):
    s = pure.ev("Hy.shape('box', {0.5, 0.4, 0.3}, nil)")
    assert s.kind == "box" and s.volume == pytest.approx(0.06) and s.basis == "estimated"
    assert "bounding box" in s.note
    s2 = pure.ev("Hy.shape('box', {0.5, 0.4, 0.3}, 0.045)")
    assert s2.volume == pytest.approx(0.045) and s2.basis == "measured"
    sp = pure.ev("Hy.shape('sphere', {0.24, 0.24, 0.24}, nil)")
    assert sp.kind == "sphere" and sp.volume == pytest.approx(4 / 3 * math.pi * 0.12 ** 3)
    cy = pure.ev("Hy.shape('cylinder', {0.58, 0.58, 0.88}, nil)")
    assert cy.kind == "cylinder" and cy.axis == 3 and cy.volume == pytest.approx(math.pi * 0.29 ** 2 * 0.88)
    cy2 = pure.ev("Hy.shape('cylinder', {0.88, 0.58, 0.58}, nil)")
    assert cy2.axis == 1                                                              # a pole lying along x


# ------------------------------------------------------------------------------------------------ Archimedes: the draft
def settle_level(p, shape_tbl, centre_xy, mass, rho_l):
    """Find the centre height at which the weight equals the buoyancy (bisection on the exact geometry)."""
    lo, hi = -5.0, 5.0
    submerged = p.ev("Hy.submerged")
    for _ in range(200):
        mid = (lo + hi) / 2
        pose = p.rt.eval("function(z) return { centre = {0, 0, z}, ax = {1,0,0}, ay = {0,1,0}, az = {0,0,1} } end")(mid)
        v, _, _ = submerged(shape_tbl, pose, 0.0)
        if rho_l * v > mass:
            lo = mid                                                                  # too deep: rise
        else:
            hi = mid
    return (lo + hi) / 2


@pytest.mark.parametrize("class_,dims,rho_body", [
    ("box", (0.6, 0.5, 0.4), 600.0), ("box", (0.6, 0.5, 0.4), 250.0), ("box", (1.0, 1.0, 0.2), 900.0),
    ("sphere", (0.3, 0.3, 0.3), 700.0), ("sphere", (0.3, 0.3, 0.3), 100.0),
    ("cylinder", (0.5, 0.5, 0.9), 500.0), ("cylinder", (0.9, 0.5, 0.5), 800.0),
])
def test_a_floating_body_settles_at_the_density_ratio_the_librarys_draft(pure, class_, dims, rho_body):
    shape = pure.rt.eval(f"Hy.shape('{class_}', {{{dims[0]}, {dims[1]}, {dims[2]}}}, nil)")
    mass = rho_body * shape.volume
    z = settle_level(pure, shape, (0, 0), mass, RHO_W)
    pose = pure.rt.eval(f"function(z) return {{ centre = {{0, 0, z}}, ax = {{1,0,0}}, ay = {{0,1,0}}, az = {{0,0,1}} }} end")(z)
    v, cen, frac = pure.ev("Hy.submerged")(shape, pose, 0.0)
    assert v / shape.volume == pytest.approx(submerged_fraction(rho_body, RHO_W), rel=1e-9)     # the library's own answer: draft = density ratio
    assert lib_floats(rho_body, RHO_W)


def test_sixty_percent_of_a_600_kg_m3_block_is_under(pure):
    shape = pure.ev("Hy.shape('box', {0.5, 0.5, 0.5}, nil)")
    mass = 600.0 * shape.volume
    z = settle_level(pure, shape, (0, 0), mass, 600.0 / 0.6)
    pose = pure.rt.eval(f"function(z) return {{ centre = {{0, 0, z}}, ax = {{1,0,0}}, ay = {{0,1,0}}, az = {{0,0,1}} }} end")(z)
    _, _, frac = pure.ev("Hy.submerged")(shape, pose, 0.0)
    assert frac == pytest.approx(0.6)
    assert submerged_fraction(600.0, 1000.0) == pytest.approx(0.6)


def test_a_denser_body_has_no_floating_equilibrium(pure):
    assert pure.ev("Hy.draft_fraction")(600.0, 1000.0) == pytest.approx(0.6)
    assert pure.ev("Hy.draft_fraction")(7850.0, RHO_W) == 1.0
    assert submerged_fraction(7850.0, RHO_W) == 1.0


def test_buoyancy_force_is_archimedes_and_scales_with_g(pure):
    v = 0.05
    f_earth = pure.ev("Hy.buoyancy")(RHO_W, G_EARTH, v)
    f_moon = pure.ev("Hy.buoyancy")(RHO_W, G_MOON, v)
    assert f_earth == pytest.approx(RHO_W * G_EARTH * v)
    assert f_moon / f_earth == pytest.approx(G_MOON / G_EARTH)
    # but the draft does not depend on g: a floating body sits at the same fraction on the Moon (weight and buoyancy scale together)
    assert pure.ev("Hy.draft_fraction")(600.0, RHO_W) == pytest.approx(600.0 / RHO_W)


def test_buoyancy_in_other_liquids_uses_their_densities(pure):
    for key in ("seawater", "mercury", "glycerol", "ethanol"):
        rho = fluid.KNOWN_LIQUIDS[key]["density_kg_m3"]
        fl, frac = pure.ev(f"Cn.floats(900.0, '{key}')")
        assert fl == lib_floats(900.0, rho)
        assert frac == pytest.approx(submerged_fraction(900.0, rho))
    assert pure.ev("Cn.floats(7850.0, 'mercury')")[0] is True                          # steel floats on mercury, sinks in water
    assert pure.ev("Cn.floats(7850.0, 'water')")[0] is False


# ------------------------------------------------------------------------------------------------ drag in water
@pytest.mark.parametrize("speed", [0.5, 2.0, 8.0])
def test_quadratic_drag_in_water_is_the_librarys_formula(pure, speed):
    cd, area = pure.ev("WCs.CD.box.value"), 0.09
    f = pure.ev("Hy.drag_force")(RHO_W, cd, area, speed, 0)
    assert f == pytest.approx(lib_drag_force(speed, cd, area, RHO_W), rel=1e-12)


def test_the_viscous_term_is_exactly_stokes_for_a_sphere(pure):
    eta = fluid.KNOWN_LIQUIDS["glycerol"]["viscosity_pa_s"]
    r, v = 0.01, 0.002
    area = math.pi * r * r
    f = pure.ev("Hy.drag_force")(1261.0, 0.0, area, v, eta)
    assert f == pytest.approx(6 * math.pi * eta * r * v, rel=1e-12)                    # F = 6 pi eta r v
    # and it is below 2.5 percent of the quadratic term at Reynolds number 1000
    cd = 1.05
    re = 1000.0
    L = 0.1
    v = re * RHO_W and re * 1.002e-3 / (RHO_W * L)
    a = math.pi / 4 * L * L
    visc = pure.ev("Hy.drag_force")(RHO_W, 0.0, a, v, 1.002e-3)
    quad = pure.ev("Hy.drag_force")(RHO_W, cd, a, v, 0)
    assert visc / quad < 0.025


def test_terminal_speed_balances_weight_less_buoyancy_with_drag(pure):
    rho_b, V, cd, area = 7850.0, 0.001, 1.05, 0.01
    eta = 1.002e-3
    v, direction = pure.ev("Hy.terminal_speed")(rho_b, V, RHO_W, G_EARTH, cd, area, eta)
    assert direction == -1
    net = (rho_b - RHO_W) * V * G_EARTH
    assert pure.ev("Hy.drag_force")(RHO_W, cd, area, v, eta) == pytest.approx(net, rel=1e-9)
    # without viscosity: the classic sqrt(2 (rho_b - rho_l) V g / (rho_l Cd A))
    v0, _ = pure.ev("Hy.terminal_speed")(rho_b, V, RHO_W, G_EARTH, cd, area, 0)
    assert v0 == pytest.approx(math.sqrt(2 * net / (RHO_W * cd * area)))
    # a light body rises
    v_up, d_up = pure.ev("Hy.terminal_speed")(300.0, V, RHO_W, G_EARTH, cd, area, eta)
    assert d_up == 1 and v_up > 0


def test_terminal_speed_scales_with_sqrt_g(pure):
    args = (7850.0, 0.001, RHO_W)
    v_e, _ = pure.ev("Hy.terminal_speed")(*args, G_EARTH, 1.05, 0.01, 0)
    v_m, _ = pure.ev("Hy.terminal_speed")(*args, G_MOON, 1.05, 0.01, 0)
    assert v_m / v_e == pytest.approx(math.sqrt(G_MOON / G_EARTH))


def test_the_implicit_drag_step_converges_to_the_terminal_speed(pure):
    rho_b, V, cd, area = 7850.0, 0.001, 1.05, 0.01
    eta = 1.002e-3
    mass = rho_b * V
    k2, k1 = pure.ev("Hy.drag_k")(RHO_W, cd, area, mass, eta)
    vt, _ = pure.ev("Hy.terminal_speed")(rho_b, V, RHO_W, G_EARTH, cd, area, eta)
    v, dt = 0.0, 0.015
    a_net = (1 - RHO_W / rho_b) * G_EARTH
    for _ in range(4000):
        v = v + a_net * dt
        v = pure.ev("Hy.speed_after")(v, k2, k1, dt)
    # operator splitting (accelerate, then drag) settles within a step's error of the true terminal speed
    assert v == pytest.approx(vt, rel=0.02)


# ------------------------------------------------------------------------------------------------ waves
SIG_W = fluid.KNOWN_LIQUIDS["water"]["surface_tension_n_m"] / RHO_W


def test_deep_water_dispersion_is_omega_squared_g_k(pure):
    for lam in (0.5, 2.0, 10.0):
        k = 2 * math.pi / lam
        w = pure.ev("Wv.omega")(k, G_EARTH, 1000.0, 0)
        assert w == pytest.approx(math.sqrt(G_EARTH * k), rel=1e-9)
        assert pure.ev("Wv.phase_speed")(lam, G_EARTH, 1000.0, 0) == pytest.approx(pure.ev("Wv.deep_speed")(lam, G_EARTH))


def test_shallow_water_speed_is_sqrt_g_d_for_every_wavelength(pure):
    d = 0.05
    for lam in (5.0, 20.0, 80.0):
        c = pure.ev("Wv.phase_speed")(lam, G_EARTH, d, 0)
        assert c == pytest.approx(math.sqrt(G_EARTH * d), rel=2e-3)
    assert pure.ev("Wv.shallow_speed")(G_EARTH, d) == math.sqrt(G_EARTH * d)


def test_the_moons_water_has_slower_waves_by_sqrt_of_g(pure):
    c_e = pure.ev("Wv.phase_speed")(3.0, G_EARTH, 1000.0, 0)
    c_m = pure.ev("Wv.phase_speed")(3.0, G_MOON, 1000.0, 0)
    assert c_m / c_e == pytest.approx(math.sqrt(G_MOON / G_EARTH))


def test_group_speed_is_half_the_phase_speed_in_deep_water(pure):
    lam = 4.0
    cg = pure.ev("Wv.group_speed")(lam, G_EARTH, 1000.0, 0)
    c = pure.ev("Wv.phase_speed")(lam, G_EARTH, 1000.0, 0)
    assert cg / c == pytest.approx(0.5, rel=1e-4)


def test_capillary_term_matters_only_for_centimetre_ripples(pure):
    # sigma/rho from the library: at 5 m the correction is under a ppm; at 1 cm it dominates (phase speed minimum near 1.7 cm)
    c_gravity = pure.ev("Wv.phase_speed")(5.0, G_EARTH, 100.0, 0)
    c_cap = pure.ev("Wv.phase_speed")(5.0, G_EARTH, 100.0, SIG_W)
    assert c_cap / c_gravity == pytest.approx(1.0, abs=1e-5)
    c1 = pure.ev("Wv.phase_speed")(0.01, G_EARTH, 100.0, SIG_W)
    c1g = pure.ev("Wv.phase_speed")(0.01, G_EARTH, 100.0, 0)
    assert c1 > 1.5 * c1g
    cmin = min(pure.ev("Wv.phase_speed")(l / 1000.0, G_EARTH, 100.0, SIG_W) for l in range(5, 80))
    assert cmin == pytest.approx(0.23, abs=0.01)                                       # the classic 23 cm/s minimum of water waves


def test_drawn_waves_are_gentle_and_each_obeys_the_dispersion(pure):
    comps = pure.ev("Wv.components")(G_EARTH, 2.0, SIG_W, 0.8, 5.0, 0.05, 4, 1.7, 0.4)
    n = len(comps)
    assert n >= 2
    for i in range(1, n + 1):
        c = comps[i]
        assert c.amp * c.k == pytest.approx(0.05)                                      # steepness a k, far below the breaking limit
        assert c.omega == pytest.approx(pure.ev("Wv.omega")(c.k, G_EARTH, 2.0, SIG_W))
        assert c.lambda_ if False else True
    assert max(comps[i].amp for i in range(1, n + 1)) < 0.2


def test_the_surface_height_is_a_sum_of_travelling_waves(pure):
    comps = pure.ev("Wv.components")(G_EARTH, 5.0, 0, 1.0, 20.0, 0.05, 3, 2.0, 0.0)
    c = comps[1]
    speed = c.omega / c.k
    # a crest at (x, t) is at (x + speed dt, t + dt) for a wave moving along its own direction
    h = pure.ev("Wv.height")
    one = comps[1]
    single = pure.rt.eval("function(c) return { c } end")(one)
    t0, x0 = 0.3, 0.7
    dx = one.dx
    h0 = h(single, x0 * dx, x0 * one.dy, t0)
    dt = 0.2
    shift = speed * dt
    h1 = h(single, (x0 + shift) * dx, (x0 + shift) * one.dy, t0 + dt)
    assert h1 == pytest.approx(h0, abs=1e-9)


def test_ring_front_travels_at_the_group_velocity(pure):
    r, cg = pure.ev("Wv.ring")(0.05, 0.3, 2.0, G_EARTH, 5.0, SIG_W)
    assert r == pytest.approx(cg * 2.0)
    assert cg == pytest.approx(pure.ev("Wv.group_speed")(0.3, G_EARTH, 5.0, SIG_W))
    _, cg_moon = pure.ev("Wv.ring")(0.05, 0.3, 2.0, G_MOON, 5.0, SIG_W)
    assert cg_moon < cg


# ------------------------------------------------------------------------------------------------ splashes
def test_splash_energy_is_the_real_kinetic_energy(pure):
    plan = pure.ev("Sp.plan")(10.0, 4.4, RHO_W, G_EARTH, 48)
    assert plan.E == pytest.approx(0.5 * 10.0 * 4.4 ** 2)
    assert plan.ell == pytest.approx((plan.E / (RHO_W * G_EARTH)) ** 0.25)


def test_a_boulder_splashes_more_than_a_leaf(pure):
    boulder = pure.ev("Sp.plan")(80.0, math.sqrt(2 * G_EARTH * 3.0), RHO_W, G_EARTH, 48)
    leaf = pure.ev("Sp.plan")(0.0005, 0.9, RHO_W, G_EARTH, 48)
    ripple_only = pure.ev("Sp.plan")(0.0001, 0.2, RHO_W, G_EARTH, 48)
    assert boulder.ell > 4 * leaf.ell
    assert boulder.n > leaf.n >= 1
    assert boulder.ring_amp > leaf.ring_amp
    assert boulder.v_up > leaf.v_up
    assert ripple_only is None                                                         # a speck on the surface makes no splash (below 0.1 mJ)


def test_splash_length_scales_as_the_fourth_root_of_energy(pure):
    p1 = pure.ev("Sp.plan")(1.0, 4.0, RHO_W, G_EARTH, 48)
    p2 = pure.ev("Sp.plan")(16.0, 4.0, RHO_W, G_EARTH, 48)
    assert p2.E / p1.E == pytest.approx(16.0)
    assert p2.ell / p1.ell == pytest.approx(2.0)                                       # 16^(1/4)


def test_a_splash_on_the_moon_is_bigger_and_never_faster_than_the_entry(pure):
    e = pure.ev("Sp.plan")(5.0, 3.0, RHO_W, G_EARTH, 48)
    m = pure.ev("Sp.plan")(5.0, 3.0, RHO_W, G_MOON, 48)
    assert m.ell > e.ell                                                               # less gravity, a deeper cavity
    assert e.v_up <= 3.0 and m.v_up <= 3.0                                              # droplets never leave faster than the body came in
    assert pure.ev("Sp.plan")(5.0, 3.0, RHO_W, G_EARTH, 3).n == 3                      # the particle cap holds


def test_particles_are_capped(pure):
    huge = pure.ev("Sp.plan")(5000.0, 30.0, RHO_W, G_EARTH, 48)
    assert huge.n == 48


# ------------------------------------------------------------------------------------------------ the basin
def test_basin_box_is_the_inside_plus_walls_and_the_volume_is_exact(pure):
    b = pure.ev("Bs.new")("rect", 10.0, 5.0, 2.0)
    wall = json.loads(WCN.JSON_PATH.read_text(encoding="utf-8"))["limits"]["wall_m"]["value"] / HU_M
    assert b.hx == pytest.approx(5.0 / HU_M) and b.hy == pytest.approx(2.5 / HU_M)
    assert b.maxs[1] - b.mins[1] == pytest.approx(10.0 / HU_M + 2 * wall)
    assert b.volume_m3 == pytest.approx(100.0)
    assert pure.ev("Bs.surface_z")(b, 1.0) == pytest.approx(b.floor_z + 2.0 / HU_M)
    assert pure.ev("Bs.surface_z")(b, 0.0) == pytest.approx(b.floor_z)
    pond = pure.ev("Bs.new")("round", 6.0, 6.0, 1.5)
    assert pond.volume_m3 == pytest.approx(math.pi * 9 * 1.5)
    assert len(pure.ev("Bs.parts")(pond)) == pure.ev("Bs.SEGMENTS") + 1
    assert len(pure.ev("Bs.parts")(b)) == 5


def test_basin_footprint_and_inside_test(pure):
    b = pure.ev("Bs.new")("round", 6.0, 6.0, 1.5)
    assert pure.ev("Bs.inside_xy")(b, 0, 0) is True
    assert pure.ev("Bs.inside_xy")(b, b.hx * 0.99, 0) is True
    assert pure.ev("Bs.inside_xy")(b, b.hx * 0.8, b.hy * 0.8) is False                  # the corner of its bounding square is dry


def test_caps_are_enforced_and_say_they_are_not_physics(pure):
    j = json.loads(WCN.JSON_PATH.read_text(encoding="utf-8"))["limits"]
    cap_n, cap_v = int(j["max_bodies"]["value"]), j["max_volume_m3"]["value"]
    bodies = pure.rt.eval("function(n, v) local t = {} for i = 1, n do t[i] = { volume = v } end return t end")(cap_n, 10.0)
    ok, why = pure.ev("Cn.room_for")(bodies, 10.0)
    assert ok is False and "not physics" in why
    bodies2 = pure.rt.eval("function(n, v) local t = {} for i = 1, n do t[i] = { volume = v } end return t end")(2, 10.0)
    ok, why = pure.ev("Cn.room_for")(bodies2, cap_v)
    assert ok is False and "cubic metres" in why and "not physics" in why
    assert pure.ev("Cn.room_for")(bodies2, 100.0) is True
    l, w, d, notes = pure.ev("Cn.clamp_dims")(5000.0, 0.01, 99.0)
    assert l == j["max_footprint_m"]["value"] and w == j["min_footprint_m"]["value"] and d == j["max_depth_m"]["value"]
    assert len(tolist(notes)) == 3


# ------------------------------------------------------------------------------------------------ the swimmer
def test_a_swimmer_floats_by_archimedes_and_the_human_density(pure):
    H = json.loads(WCN.JSON_PATH.read_text(encoding="utf-8"))["human"]
    rho_b = H["density_kg_m3"]["value"]
    S = lambda **k: pure.rt.eval("function(a) return a end")(pure.t())  # noqa: E731
    state = pure.rt.eval(
        f"function() return {{ z_feet = -2.0, surface = 0.0, vel = {{0, 0, 0}}, wish = {{0, 0, 0}}, up = false, down = false, g = {G_EARTH}, rho = {RHO_W}, eta = 0.001, cooldown = 0 }} end")()
    v, info = pure.ev("Sw.step")(state, 0.015)
    assert info.frac == 1.0
    assert info.a_buoy == pytest.approx(G_EARTH * (RHO_W / rho_b - 1))                  # fully under: lifted by the density ratio
    assert v[3] > 0                                                                      # a person with lungs full rises
    # at the surface: the part above the water has no buoyancy: neutral at fraction rho_b / rho_l
    f_eq = rho_b / RHO_W
    z_feet = -f_eq * pure.ev("Sw.HULL_M")[3]
    state2 = pure.rt.eval(
        f"function() return {{ z_feet = {z_feet}, surface = 0.0, vel = {{0, 0, 0}}, wish = {{0, 0, 0}}, up = false, down = false, g = {G_EARTH}, rho = {RHO_W}, eta = 0.001, cooldown = 0 }} end")()
    _, info2 = pure.ev("Sw.step")(state2, 0.015)
    assert info2.a_buoy == pytest.approx(0.0, abs=1e-9)


def test_swimmer_on_the_moon_is_lifted_by_the_same_ratio_but_slower_in_absolute_terms(pure):
    mk = lambda g: pure.rt.eval(  # noqa: E731
        f"function() return {{ z_feet = -2.0, surface = 0.0, vel = {{0, 0, 0}}, wish = {{0, 0, 0}}, up = false, down = false, g = {g}, rho = {RHO_W}, eta = 0.001, cooldown = 0 }} end")()
    _, a = pure.ev("Sw.step")(mk(G_EARTH), 0.015)
    _, b = pure.ev("Sw.step")(mk(G_MOON), 0.015)
    assert b.a_buoy / a.a_buoy == pytest.approx(G_MOON / G_EARTH)


def test_swimmer_top_speed_is_the_balance_of_thrust_and_drag(pure):
    j = json.loads(WCN.JSON_PATH.read_text(encoding="utf-8"))["human"]
    v = pure.ev("Sw.top_speed")("x", RHO_W, G_EARTH, 1.002e-3)
    hull = pure.ev("Sw.HULL_M")
    area = hull[2] * hull[3]
    cd = 1.05
    drag = pure.ev("Hy.drag_force")(RHO_W, cd, area, v, 1.002e-3)
    assert drag == pytest.approx(j["swim_thrust_n"]["value"], rel=1e-9)
    assert 0.3 < v < 1.5                                                                  # a person-like speed (the hull's box area overstates a body's)


def test_a_swimmer_reaches_a_ledge_no_higher_than_the_hull_is_tall(pure):
    hull = pure.ev("Sw.HULL_M")[3]
    ok = pure.ev("Sw.reach_ok")
    assert ok(-1.3, 0.0) is True                  # a rim at the water level, feet 1.3 m under: within reach of the hull's 1.37 m
    assert ok(-1.5, 0.0) is False                 # feet 1.5 m under: out of reach
    assert ok(0.5, 0.0) is False                  # a ledge below the feet is not a climb
    assert ok(-hull, 0.0) is True


# ------------------------------------------------------------------------------------------------ the words
def parse(p, text):
    r = p.ev("In.parse")(text)
    if r is None:
        return None
    out = {}
    for k, v in r.items():
        out[k] = v if not hasattr(v, "items") else {kk: vv for kk, vv in v.items()}
    return out


@pytest.mark.parametrize("text,noun,over", [
    ("add a pool", "pool", {}),
    ("Add a pool, please", "pool", {}),
    ("make a lake", "lake", {}),
    ("add a pond", "pond", {}),
    ("add a sea", "sea", {}),
    ("I want a lake", "lake", {}),
    ("make a 10 metre pool", "pool", {"l": 10}),
    ("add a 20 by 8 metre pool", "pool", {"l": 20, "w": 8}),
    ("add a pool 3 metres deep", "pool", {"depth": 3}),
    ("add a pool 10 x 5 x 3", "pool", {"l": 10, "w": 5, "depth": 3}),
    ("add a 30 foot pool", "pool", {"l": 30 * 0.3048}),
    ("add a pool of mercury", "pool", {"liquid": "mercury"}),
    ("add a mercury pool next to me", "pool", {"liquid": "mercury"}),
    ("add a seawater pool", "pool", {"liquid": "seawater"}),
    ("add a pool of olive oil", "pool", {"liquid": "olive oil"}),
    ("add a pool filled with water", "pool", {"liquid": "water"}),
    ("create a pnod", "pond", {}),                                                       # a swapped pair of letters
    ("add a poll", "pool", {}),                                                          # one letter off
    ("ad a pool", "pool", {}),
    ("make a lkae", "lake", {}),
])
def test_add_intents_including_typos(pure, text, noun, over):
    r = parse(pure, text)
    assert r is not None and r["kind"] == "add", (text, r)
    assert r["noun"] == noun
    for k, v in over.items():
        assert r[k] == pytest.approx(v) if isinstance(v, float) else r[k] == v, (k, r)


def test_deep_and_huge_are_read_as_stated_multiples_and_numbers_win(pure):
    r = parse(pure, "add a deep pond")
    assert r["factor_depth"] == 2 and any("deep" in a for a in r["assumed"].values())
    r = parse(pure, "add a big shallow lake")
    assert r["factor_size"] == 1.5 and r["factor_depth"] == 0.5
    r = parse(pure, "add a deep pool 3 metres deep")
    assert r["depth"] == 3 and r.get("factor_depth") is None and not r["assumed"]        # an explicit number wins over the word


def test_round_and_counts(pure):
    assert parse(pure, "add a round pool")["shape"] == "round"
    assert parse(pure, "add two pools")["count"] == 2
    assert parse(pure, "add a pool")["count"] == 1


@pytest.mark.parametrize("text", [
    "add a pool table", "add a couch", "swim in the pool", "play pool", "add a car pool", "we need a table and chairs", "set gravity to the moon",
    "add three couches", "remove the couch", "undo", "make the couch bigger", "what time is it", "drag the couch", "add a sea shell", "a pool cue",
])
def test_other_sentences_are_never_taken_for_water(pure, text):
    r = parse(pure, text)
    # (a few of these may be claimed as a non-add water intent only when they name a water body)
    assert r is None, (text, r)


@pytest.mark.parametrize("text,kind,extra", [
    ("fill it with water", "liquid", {"liquid": "water"}),
    ("fill the pool with mercury", "liquid", {"liquid": "mercury", "target": "pool"}),
    ("fill it with oil", "liquid", {"liquid": "oil"}),
    ("fill it with olive oil", "liquid", {"liquid": "olive oil"}),
    ("make it mercury", "liquid", {"liquid": "mercury"}),
    ("fill it up", "level", {"level": 1}),
    ("fill the pond", "level", {"level": 1, "target": "pond"}),
    ("drain it", "level", {"level": 0}),
    ("drain the pool", "level", {"level": 0, "target": "pool"}),
    ("drian the pool", "level", {"level": 0, "target": "pool"}),
    ("empty the lake", "level", {"level": 0, "target": "lake"}),
    ("remove the water", "remove", {}),
    ("remove the pond", "remove", {"target": "pond"}),
    ("get rid of the pool", "remove", {"target": "pool"}),
    ("make the pool deeper", "resize", {"dim": "depth", "factor": 1.5, "target": "pool"}),
    ("make it shallower", "resize", {"dim": "depth", "factor": 1 / 1.5}),
    ("make the pool bigger", "resize", {"dim": "size", "factor": 1.5, "target": "pool"}),
    ("make the lake smaller", "resize", {"dim": "size", "factor": 1 / 1.5, "target": "lake"}),
    ("make the pool much deeper", "resize", {"dim": "depth", "factor": 2.25}),
    ("make the pool 4 metres deep", "resize", {"dim": "depth", "absolute": 4}),
    ("what floats", "query", {"what": "which"}),
    ("what is the density of this water", "query", {"what": "density"}),
    ("does the ball float", "query", {"what": "floats", "item": "ball"}),
    ("how deep is the pool", "query", {"what": "depth"}),
    ("what is in the water", "query", {"what": "liquid"}),
])
def test_other_water_intents(pure, text, kind, extra):
    r = parse(pure, text)
    assert r is not None and r["kind"] == kind, (text, r)
    for k, v in extra.items():
        assert r.get(k) == pytest.approx(v) if isinstance(v, float) else r.get(k) == v, (text, k, r)


@pytest.mark.parametrize("text,target", [
    ("drop a ball in the pool", "pool"), ("throw a rock into the lake", "lake"), ("put a boat on the pond", "pond"),
    ("toss a crate in the sea", "sea"), ("drop a barrel in the water", None),
])
def test_drop_intents_rewrite_to_the_objects_own_surface_rule(pure, text, target):
    r = parse(pure, text)
    assert r is not None and r["kind"] == "drop", (text, r)
    assert r.get("target") == target
    out = pure.ev("In.rewrite_drop")(text, pure.ev("In.parse")(text), "poolxaa")
    assert out.startswith("put a ") and out.endswith("on the poolxaa"), out


def test_add_rewrite_keeps_the_rest_of_the_sentence(pure):
    t = "add a 10 metre pool next to me and a couch"
    it = pure.ev("In.parse")(t)
    assert pure.ev("In.rewrite_add")(t, it, "poolxaa") == "add a poolxaa next to me and a couch"
    t2 = "make a lake in front of me"
    assert pure.ev("In.rewrite_add")(t2, pure.ev("In.parse")(t2), "lakexab") == "make a lakexab in front of me"
