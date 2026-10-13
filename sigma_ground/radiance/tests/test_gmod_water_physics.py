"""Water for the holodeck, bodies in the water (SkippyVerse Phase A, task A2): the glue applies the real laws each physics tick.

The REAL water.lua against a small rigid-body layer (gmod_mock.lua + test_gmod_water_glue.ENGINE: gravity, velocity steps, a floor, sleeping).
Every number asserted is derived from the sigma-ground library (the liquid's density and viscosity, g) and the closed forms, never typed.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "stage" / "gmod" / "holodeck"))

from test_gmod_water_glue import (  # noqa: E402,F401
    build, pool_at, run, L, body, vol, surf_m, floor_m, z_of, vz_of, asleep, entries, n_entries, water_ent, HU_M, RHO_W, G_MOON, G_EARTH, A, B,
    fluid, submerged_fraction, say_text,
)
from sigma_ground.field.interface.projectile import drag_force  # noqa: E402

ETA_W = fluid.KNOWN_LIQUIDS["water"]["viscosity_pa_s"]
CD_BOX = 1.05


def centre_xy(w):
    v = vol(w)
    return v.x, v.y


def drop(w, dims_m, mass, height_above_surface_m=0.4, model="models/props_junk/wood_crate001a.mdl", dx_m=0.0):
    x, y = centre_xy(w)
    return body(w, dims_m, mass, x=x + dx_m / HU_M, y=y, z_m=surf_m(w) + height_above_surface_m, model=model)


def depth_under(w, e):
    """How far the body's bottom is below the surface (m)."""
    return surf_m(w) - z_of(w, e)


# ============================================================================================================ floating
@pytest.mark.parametrize("rho_body", [600.0, 250.0, 900.0])
def test_a_floating_block_settles_at_the_draft_the_density_ratio_gives_and_sleeps(rho_body):
    w = build()
    pool_at(w)
    side = 0.5
    e = drop(w, (side, side, side), rho_body * side ** 3)
    run(w, 25)
    expect = submerged_fraction(rho_body, RHO_W) * side                                  # the library's draft = density ratio
    assert depth_under(w, e) == pytest.approx(expect, rel=0.03)
    assert asleep(w, e)                                                                    # it settled and the engine can sleep it
    assert abs(vz_of(w, e)) < 0.02


def test_a_600_density_block_floats_with_sixty_percent_under_in_any_liquid_by_its_density():
    w = build()
    pool_at(w, "add a pool of seawater")
    rho_sea = fluid.KNOWN_LIQUIDS["seawater"]["density_kg_m3"]
    side = 0.4
    e = drop(w, (side, side, side), 0.6 * rho_sea * side ** 3)                            # a block of 60 percent of the liquid's density
    run(w, 25)
    assert depth_under(w, e) / side == pytest.approx(0.6, rel=0.03)


def test_the_same_block_rides_lower_in_water_than_in_seawater():
    drafts = {}
    for liq, text in (("water", "add a pool"), ("seawater", "add a seawater pool")):
        w = build()
        pool_at(w, text)
        e = drop(w, (0.5, 0.5, 0.5), 500.0 * 0.125)
        run(w, 25)
        drafts[liq] = depth_under(w, e)
    assert drafts["water"] > drafts["seawater"]
    assert drafts["water"] / drafts["seawater"] == pytest.approx(fluid.KNOWN_LIQUIDS["seawater"]["density_kg_m3"] / RHO_W, rel=0.04)


def test_steel_floats_on_mercury_and_sinks_in_water():
    steel = 7850.0
    w = build()
    pool_at(w, "add a pool of mercury")
    e = drop(w, (0.3, 0.3, 0.3), steel * 0.027)
    run(w, 30)
    assert depth_under(w, e) == pytest.approx(submerged_fraction(steel, fluid.KNOWN_LIQUIDS["mercury"]["density_kg_m3"]) * 0.3, rel=0.04)
    w2 = build()
    pool_at(w2)
    e2 = drop(w2, (0.3, 0.3, 0.3), steel * 0.027)
    run(w2, 10)
    assert z_of(w2, e2) == pytest.approx(floor_m(w2), abs=0.02)                            # on the bottom


def test_a_standing_cylinder_floats_by_its_exact_geometry():
    w = build()
    pool_at(w)
    e = drop(w, (0.5, 0.5, 0.9), 500.0 * math.pi * 0.25 ** 2 * 0.9, model="models/props_c17/oildrum001.mdl")
    run(w, 25)
    # a barrel standing up: the submerged fraction of its height is exactly the density ratio
    assert depth_under(w, e) / 0.9 == pytest.approx(500.0 / RHO_W, rel=0.04)


def test_a_sphere_floats_by_its_exact_cap():
    w = build()
    pool_at(w)
    r = 0.15
    e = drop(w, (2 * r, 2 * r, 2 * r), 500.0 * 4 / 3 * math.pi * r ** 3, model="models/props_phx/misc/ball.mdl")
    run(w, 25)
    h = depth_under(w, e)                                                                   # depth of the bottom below the surface = cap height
    vol_cap = math.pi * h * h * (3 * r - h) / 3
    assert vol_cap / (4 / 3 * math.pi * r ** 3) == pytest.approx(500.0 / RHO_W, rel=0.04)


def test_settled_floaters_stay_asleep_and_are_not_woken_again():
    w = build()
    pool_at(w)
    e = drop(w, (0.5, 0.5, 0.5), 300.0 * 0.125)
    run(w, 25)
    assert asleep(w, e)
    wakes = L(w, "function(e) return e.phys.wakes end")(e)
    run(w, 20)
    assert asleep(w, e)
    assert L(w, "function(e) return e.phys.wakes end")(e) == wakes


# ============================================================================================================ sinking
def solve_terminal(net, area):
    lo, hi = 0.0, 50.0
    for _ in range(100):
        mid = (lo + hi) / 2
        f = drag_force(mid, CD_BOX, area, RHO_W) + 12 * ETA_W * area * mid / math.sqrt(4 * area / math.pi)
        lo, hi = (mid, hi) if f < net else (lo, mid)
    return (lo + hi) / 2


def test_a_steel_block_sinks_at_the_terminal_speed_the_formula_gives():
    w = build()
    pool_at(w, "add a pool 20 metres deep")
    side = 0.2
    V = side ** 3
    e = drop(w, (side, side, side), 7850.0 * V, height_above_surface_m=0.0)
    run(w, 2.0)
    assert z_of(w, e) < surf_m(w) - 0.5                                                    # it is going down
    speeds = []
    for _ in range(20):
        run(w, 0.015)
        speeds.append(-vz_of(w, e))
    assert z_of(w, e) > floor_m(w) + 1.0                                                   # (still falling freely, not on the bottom)
    measured = sum(speeds[-10:]) / 10
    net = (7850.0 - RHO_W) * V * G_EARTH
    expect = solve_terminal(net, side * side)                                              # weight less buoyancy = quadratic + Stokes drag
    assert measured == pytest.approx(expect, rel=0.04)
    hy = L(w, "MentatWater.hydro")
    assert hy.terminal_speed(7850.0, V, RHO_W, G_EARTH, CD_BOX, side * side, ETA_W)[0] == pytest.approx(expect, rel=1e-6)


def test_a_light_thing_released_underwater_rises_and_floats():
    w = build()
    pool_at(w)
    x, y = centre_xy(w)
    e = body(w, (0.3, 0.3, 0.3), 200.0 * 0.027, x=x, y=y, z_m=floor_m(w), asleep=True)     # on the bottom, asleep, lighter than the water
    run(w, 15)
    assert depth_under(w, e) == pytest.approx(200.0 / RHO_W * 0.3, rel=0.05)                # it woke, rose, and floats at its draft
    assert L(w, "MentatWater.counters.woken") >= 1


def test_a_heavy_resting_body_is_left_asleep():
    w = build()
    pool_at(w)
    x, y = centre_xy(w)
    e = body(w, (0.3, 0.3, 0.3), 7850.0 * 0.027, x=x, y=y, z_m=floor_m(w), asleep=True)
    run(w, 5)
    assert asleep(w, e) and L(w, "MentatWater.counters.woken") == 0
    assert z_of(w, e) == pytest.approx(floor_m(w), abs=1e-6)


def test_water_made_under_a_resting_light_object_acts_on_it():
    # a pool is drained, a light block rests on its floor, the pool is filled again: the block must rise
    w = build()
    p = pool_at(w)
    w.ask(p, "drain it")
    x, y = centre_xy(w)
    e = body(w, (0.4, 0.4, 0.4), 150.0 * 0.064, x=x, y=y, z_m=floor_m(w), asleep=True)
    run(w, 3)
    assert asleep(w, e)                                                                     # nothing lifts it in an empty basin
    w.ask(p, "fill it up")
    run(w, 15)
    assert depth_under(w, e) == pytest.approx(150.0 / RHO_W * 0.4, rel=0.06)


# ============================================================================================================ g decides
def one_tick_dv(g, mass):
    w = build(g=g)
    pool_at(w)
    x, y = centre_xy(w)
    e = body(w, (0.5, 0.5, 0.5), mass, x=x, y=y, z_m=floor_m(w) + 0.8)                      # fully under, at rest
    L(w, "function() physenv.g = Vector(0, 0, 0) end")()                                     # (so the engine adds no gravity of its own to the reading)
    run(w, 0.015)
    return vz_of(w, e)


def test_buoyancy_is_proportional_to_g_the_moons_water_lifts_less():
    m = 100.0                                                                                  # displaces 0.125 m3: 125 kg of water: net up
    dv_e = one_tick_dv(G_EARTH, m)
    dv_m = one_tick_dv(G_MOON, m)
    def after_drag(g):
        dv = RHO_W * g * 0.125 / m * 0.015                                                    # buoyancy minus nothing: the engine's gravity was switched off
        k2 = 0.5 * RHO_W * CD_BOX * 0.25 / m
        k1 = 12 * ETA_W * 0.25 / math.sqrt(4 * 0.25 / math.pi) / m
        return dv / (1 + (k2 * dv + k1) * 0.015)                                              # then the water's drag on the speed it just gained
    assert dv_e == pytest.approx(after_drag(G_EARTH), rel=1e-6)
    assert dv_m == pytest.approx(after_drag(G_MOON), rel=1e-6)
    assert dv_m / dv_e == pytest.approx(G_MOON / G_EARTH, rel=0.01)


def test_draft_does_not_depend_on_g():
    drafts = {}
    for name, g in (("earth", G_EARTH), ("moon", G_MOON)):
        w = build(g=g)
        pool_at(w)
        e = drop(w, (0.5, 0.5, 0.5), 600.0 * 0.125, height_above_surface_m=0.0)
        run(w, 60)
        drafts[name] = depth_under(w, e)
    assert drafts["moon"] == pytest.approx(0.6 * 0.5, rel=0.05)
    assert drafts["earth"] == pytest.approx(0.6 * 0.5, rel=0.04)


def test_sinking_is_slower_on_the_moon_by_sqrt_g():
    speeds = {}
    for name, g in (("earth", G_EARTH), ("moon", G_MOON)):
        w = build(g=g)
        pool_at(w, "add a pool 20 metres deep")
        e = drop(w, (0.2, 0.2, 0.2), 7850.0 * 0.008, height_above_surface_m=0.0)
        run(w, 2.0 if name == "earth" else 6.0)                                              # (the Moon takes longer to get up to speed)
        sp = []
        for _ in range(20):
            run(w, 0.015)
            sp.append(-vz_of(w, e))
        assert z_of(w, e) > floor_m(w) + 0.5
        speeds[name] = sum(sp) / len(sp)
    assert speeds["moon"] / speeds["earth"] == pytest.approx(math.sqrt(G_MOON / G_EARTH), rel=0.06)


# ============================================================================================================ drag, torque, air
def test_the_water_drag_is_the_quadratic_law_with_the_same_cd_and_area_as_the_air():
    w = build(g=1e-12)
    pool_at(w)
    x, y = centre_xy(w)
    side, v0 = 0.3, 3.0
    m = RHO_W * side ** 3                                                                      # exactly the displaced mass: buoyancy = weight
    e = body(w, (side, side, side), m, x=x, y=y, z_m=floor_m(w) + 1.0)
    L(w, "function(e) e.phys.vel = Vector(%f, 0, 0) end" % (v0 / HU_M))(e)
    L(w, "function() physenv.g = Vector(0, 0, 0) end")()
    run(w, 0.015)
    v1 = L(w, "function(e) return e.phys.vel.x end")(e) * HU_M
    k2 = 0.5 * RHO_W * CD_BOX * side * side / m
    k1 = 12 * ETA_W * side * side / math.sqrt(4 * side * side / math.pi) / m
    assert v1 == pytest.approx(v0 / (1 + (k2 * v0 + k1) * 0.015), rel=1e-5)


def test_the_air_drag_of_the_world_stands_aside_under_water():
    w = build()
    pool_at(w)
    x, y = centre_xy(w)
    e = body(w, (0.3, 0.3, 0.3), 27.0, x=x, y=y, z_m=floor_m(w) + 1.0)
    run(w, 0.1)
    assert L(w, "function(e) return MentatWater.recs[e].wet end")(e) is True
    rec = L(w, "function(e) return MentatWorld.tracked[e] end")(e)
    assert rec is not None
    assert L(w, "function(e, r) return MentatWorld.drag_step(e, r, 0.015) end")(e, rec) is None            # wet: left to the water's drag
    out = body(w, (0.3, 0.3, 0.3), 27.0, x=x + 3000, y=y + 3000, z_m=5.0, vz=-5.0)
    run(w, 0.05)
    rec2 = L(w, "function(e) return MentatWorld.tracked[e] end")(out)
    assert L(w, "function(e, r) return MentatWorld.drag_step(e, r, 0.015) end")(out, rec2) is not None     # dry: the air's drag still applies


def roll_torque(rel_density, dims, roll_deg):
    """Float a block of this relative density, tilt it about its length, give it one tick, and return (omega . righting direction)."""
    w = build()
    pool_at(w)
    x, y = centre_xy(w)
    a, b, c = dims
    draft = rel_density * c
    e = body(w, dims, rel_density * RHO_W * a * b * c, x=x, y=y, z_m=surf_m(w) - draft)
    L(w, "function(e) e.GetAngles = function(s) return Angle(0, 0, %f) end end" % roll_deg)(e)
    run(w, 0.02)
    dw = L(w, "function(e) return { e.phys.w.x, e.phys.w.y, e.phys.w.z } end")(e)
    wv = [dw[1], dw[2], dw[3]]
    u = L(w, "function(e) local u = e:GetAngles():Up() return { u.x, u.y, u.z } end")(e)
    ux, uy, uz = u[1], u[2], u[3]
    righting = [uy, -ux, 0.0]                                                                  # up x z: the rotation that carries the body's up toward the world's
    return sum(p * q for p, q in zip(wv, righting))


def test_a_wide_plank_tilted_is_given_a_righting_torque():
    # a plank is stable flat: the centre of buoyancy moves toward the side that goes under, and the couple turns it back
    assert roll_torque(0.5, (1.0, 1.0, 0.2), 10.0) > 0
    assert roll_torque(0.5, (1.0, 1.0, 0.2), -10.0) > 0


def test_a_cube_of_relative_density_one_half_is_unstable_face_up():
    # the known result (a cube of relative density between 0.21 and 0.79 does not float face up): the same torque law finds it
    assert roll_torque(0.5, (0.6, 0.6, 0.6), 10.0) < 0
    assert roll_torque(0.1, (0.6, 0.6, 0.6), 10.0) > 0                                          # a very light cube is stable


# ============================================================================================================ budgets
def test_the_per_tick_cap_holds_and_every_body_is_still_served():
    w = build()
    pool_at(w, "add a 20 by 20 metre pool")
    x, y = centre_xy(w)
    L(w, "function() GetConVar('mentat_water_max_tick'):SetValue(5) end")()
    bodies = [body(w, (0.2, 0.2, 0.2), 4.0, x=x + i * 20, y=y, z_m=surf_m(w) - 0.5) for i in range(20)]
    c0 = L(w, "MentatWater.counters.processed")
    run(w, 0.015)
    assert L(w, "MentatWater.counters.processed") - c0 <= 5
    run(w, 0.015 * 3)
    for e in bodies:                                                                          # 4 ticks x 5 = 20: every body had its turn
        assert L(w, "function(e) return MentatWater.recs[e] ~= nil end")(e)


def test_without_water_the_tick_costs_nothing():
    w = build()
    for i in range(30):
        body(w, (0.2, 0.2, 0.2), 4.0, x=i * 30, y=0, z_m=1.0)
    run(w, 1.0)
    assert L(w, "MentatWater.counters.processed") == 0 and L(w, "MentatWater.counters.ticks") == 0


def test_bodies_outside_the_water_are_not_touched():
    w = build()
    pool_at(w)
    x, y = centre_xy(w)
    far = body(w, (0.3, 0.3, 0.3), 27.0, x=x + 3000, y=y + 3000, z_m=3.0)
    run(w, 0.3)
    assert L(w, "function(e) return MentatWater.recs[e].wet end")(far) is False


def test_the_switch_turns_the_water_off():
    w = build()
    pool_at(w)
    L(w, "function() GetConVar('mentat_water_enabled'):SetValue(0) end")()
    e = drop(w, (0.5, 0.5, 0.5), 60.0, height_above_surface_m=0.0)
    run(w, 3)
    assert z_of(w, e) == pytest.approx(floor_m(w), abs=0.02)                                  # it fell through as if there were no water


# ============================================================================================================ splashes
def splashes(w):
    return L(w, "(function() local o = {} for _, n in ipairs(MOCK.nets) do if n.name == 'mentat_water_splash' then o[#o + 1] = n end end return o end)()")


def test_entry_sends_a_splash_with_the_real_energy():
    w = build()
    pool_at(w)
    L(w, "function() MOCK.nets = {} end")()
    mass = 20.0
    drop(w, (0.3, 0.3, 0.3), mass, height_above_surface_m=2.0)
    run(w, 1.5)
    msgs = splashes(w)
    assert len(msgs) >= 1
    f = msgs[1].floats
    x, y, surf, m, speed = f[1], f[2], f[3], f[4], f[5]
    assert m == pytest.approx(mass)
    assert speed == pytest.approx(math.sqrt(2 * G_EARTH * 2.0), rel=0.05)                      # it fell 2 m
    assert 0.5 * m * speed ** 2 == pytest.approx(mass * G_EARTH * 2.0, rel=0.10)               # E = m g h of the fall


def test_a_bigger_drop_makes_a_bigger_splash_message():
    energies = []
    for h in (0.3, 3.0):
        w = build()
        pool_at(w)
        L(w, "function() MOCK.nets = {} end")()
        drop(w, (0.3, 0.3, 0.3), 30.0, height_above_surface_m=h)
        run(w, 1.5)
        f = splashes(w)[1].floats
        energies.append(0.5 * f[4] * f[5] ** 2)
    assert energies[1] > 5 * energies[0]


def test_splash_messages_are_rate_limited():
    w = build()
    pool_at(w, "add a 20 by 20 metre pool")
    L(w, "function() MOCK.nets = {} end")()
    x, y = centre_xy(w)
    for i in range(60):
        body(w, (0.2, 0.2, 0.2), 4.0, x=x + i * 10, y=y, z_m=surf_m(w) + 1.0)
    run(w, 0.8)
    assert len(splashes(w)) <= 24                                                              # a cap per second, not physics


# ============================================================================================================ the readback
def test_what_floats_says_which_by_density_with_the_bases():
    w = build()
    p = pool_at(w)
    drop(w, (0.5, 0.5, 0.5), 60.0, height_above_surface_m=0.1)
    drop(w, (0.2, 0.2, 0.2), 62.8, height_above_surface_m=0.1, dx_m=2.0)
    run(w, 10)
    w.ask(p, "what floats")
    t = say_text(w)
    assert "float" in t and "sink" in t and "density" in t
    assert "kg/m3" in t and "mass" in t and "volume" in t                                         # the lines carry the comparison and the bases


def test_the_density_question_quotes_the_library():
    w = build()
    p = pool_at(w)
    w.ask(p, "what is the density of this water")
    t = say_text(w)
    assert f"{RHO_W:.3g}" in t and "library" in t


# ============================================================================================================ a person in the water
SWIM_PRELUDE = r"""
IN_JUMP, IN_DUCK, IN_FORWARD = 2, 4, 8
MASK_PLAYERSOLID = 1
FLOOR_Z = 0
function util.TraceHull(t)
  local lowest = t.endpos.z + t.mins.z
  if lowest < FLOOR_Z then
    local frac = (t.start.z + t.mins.z - FLOOR_Z) / math.max(1e-9, (t.start.z - t.endpos.z))
    return { Hit = true, HitPos = Vector(t.start.x + (t.endpos.x - t.start.x) * frac, t.start.y + (t.endpos.y - t.start.y) * frac, FLOOR_Z - t.mins.z),
             HitNormal = Vector(0, 0, 1), Fraction = frac }
  end
  return { Hit = false, Fraction = 1 }
end
function SWIMMER(z_feet_hu, x, y)
  local p = MOCK.newplayer("swimmer", {})
  p.InVehicle = function() return false end
  p.GetHull = function() return Vector(-16, -16, 0), Vector(16, 16, 72) end
  p.GetMoveType = function() return MOVETYPE_WALK end
  p.mvpos, p.mvvel = Vector(x, y, z_feet_hu), Vector(0, 0, 0)
  p.keys, p.fwd = {}, 0
  return p
end
function MV(p)
  return {
    GetOrigin = function() return p.mvpos end, SetOrigin = function(_, v) p.mvpos = v end,
    GetVelocity = function() return p.mvvel end, SetVelocity = function(_, v) p.mvvel = v end,
    KeyDown = function(_, k) return p.keys[k] == true end, GetMoveAngles = function() return Angle(0, 0, 0) end,
    GetForwardSpeed = function() return p.fwd end, GetSideSpeed = function() return 0 end, GetMaxClientSpeed = function() return 200 end,
  }
end
function STEP(p)
  MOCK.advance(0.015)
  return MentatWater.swim_move.move(p, MV(p))
end
"""


def swimmer(w, depth_ft_hu=None):
    w.rt.execute(SWIM_PRELUDE)
    L(w, "MentatWater.swim_move") if False else None
    w.rt.execute('MentatWater.swim_move = dofile(MOCK.ROOT .. "/mentat/water/swim_move.lua")')
    v = vol(w)
    w.rt.execute(f"FLOOR_Z = {v.floor}")
    return w.rt.eval(f"SWIMMER({v.floor + 5}, {v.x}, {v.y})")


def swim_ticks(w, p, n):
    w.rt.eval("function(p, n) for _ = 1, n do STEP(p) end end")(p, n)


def feet_m(w, p):
    return L(w, "function(p) return p.mvpos.z end")(p) * HU_M


def test_a_person_in_the_water_floats_at_the_density_ratio_with_the_head_at_the_surface():
    import json as _json
    from test_gmod_water_glue import WCN
    human = _json.loads(WCN.JSON_PATH.read_text(encoding="utf-8"))["human"]
    w = build()
    pool_at(w, "add a pool 3 metres deep")
    p = swimmer(w)
    swim_ticks(w, p, 3600)                                                                  # 54 s of doing nothing (a person rises at about 0.2 m/s from the bottom)
    f_expected = human["density_kg_m3"]["value"] / RHO_W
    hull = 72 * HU_M
    assert (surf_m(w) - feet_m(w, p)) == pytest.approx(f_expected * hull, rel=0.03)         # Archimedes on the immersed hull
    assert abs(L(w, "function(p) return p.mvvel.z end")(p)) * HU_M < 0.02
    assert surf_m(w) - feet_m(w, p) < hull                                                    # the top of the head is at or above the surface


def test_a_person_swims_at_the_speed_thrust_and_drag_balance_gives():
    import json as _json
    from test_gmod_water_glue import WCN
    human = _json.loads(WCN.JSON_PATH.read_text(encoding="utf-8"))["human"]
    w = build()
    pool_at(w, "add a 40 by 40 metre pool 3 metres deep")
    p = swimmer(w)
    swim_ticks(w, p, 600)                                                                   # float up first
    L(w, "function(p) p.fwd = 200 end")(p)
    swim_ticks(w, p, 500)
    vx = L(w, "function(p) return p.mvvel.x end")(p) * HU_M
    thrust, area = human["swim_thrust_n"]["value"], 32 * HU_M * 72 * HU_M
    lo, hi = 0.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        f = 0.5 * RHO_W * CD_BOX * area * mid ** 2 + 12 * ETA_W * area * mid / math.sqrt(4 * area / math.pi)
        lo, hi = (mid, hi) if f < thrust else (lo, mid)
    assert vx == pytest.approx((lo + hi) / 2, rel=0.08)                                       # (the immersed fraction trims the area a little)


def test_jumping_at_an_edge_pulls_a_swimmer_onto_it_and_in_open_water_it_only_treads():
    w = build()
    pool_at(w, "add a pool 3 metres deep")
    v = vol(w)
    # open water: holding jump treads water (thrust up) and never leaves it
    p = swimmer(w)
    swim_ticks(w, p, 600)
    L(w, "function(p) p.keys[IN_JUMP] = true end")(p)
    swim_ticks(w, p, 300)
    head_above = feet_m(w, p) + 72 * HU_M - surf_m(w)
    assert 0.05 < head_above < 0.5                                                              # shoulders out: treading, held by thrust against the weight
    assert feet_m(w, p) < surf_m(w)                                                              # still in the water
    # an edge: the pool's wall one step ahead, its top 5 cm above the water
    w.rt.execute(f"WALL = {{ x = {v.x + v.b.hx} - 60 * 0 , top = {v.surf} + 2.6 }}")
    w.rt.execute("""
      local base = util.TraceHull
      function util.TraceHull(t)
        local tr = base(t)
        if tr.Hit then return tr end
        if WALL and t.endpos.x + t.maxs.x > WALL.x and t.endpos.z + t.mins.z < WALL.top then return { Hit = true, StartSolid = false, HitPos = t.start, HitNormal = Vector(-1, 0, 0), Fraction = 0 } end
        return { Hit = false, StartSolid = false, Fraction = 1 }
      end
      function util.TraceLine(t)
        if WALL and t.start.x > WALL.x - 1 and t.start.x < WALL.x + 400 then return { Hit = true, StartSolid = false, HitPos = Vector(t.start.x, t.start.y, WALL.top) } end
        return { Hit = false }
      end
    """)
    L(w, "function(p, x) p.mvpos = Vector(x, p.mvpos.y, p.mvpos.z) end")(p, v.x + v.b.hx - 20)
    swim_ticks(w, p, 200)
    z_after = L(w, "function(p) return p.mvpos.z end")(p)
    assert z_after >= v.surf + 2.6                                                              # standing on the rim
    assert L(w, "function(p) return p.mvpos.x end")(p) > v.x + v.b.hx - 20                       # and a step further on


def test_no_swimming_in_a_drained_basin_noclip_or_a_vehicle():
    w = build()
    p0 = pool_at(w, "add a pool 3 metres deep")
    p = swimmer(w)
    swim_ticks(w, p, 30)
    z_in = L(w, "function(p) return p.mvpos.z end")(p)
    w.ask(p0, "drain it")
    w.rt.execute("function PHYS_OFF() end")
    L(w, "function(p) p.mvvel = Vector(0, 0, -100) end")(p)
    r = L(w, "function(p) return STEP(p) end")(p)
    assert r is None                                                                          # nothing to swim in: the engine's own movement
    w.ask(p0, "fill it up")
    L(w, "function(p) p.GetMoveType = function() return 8 end end")(p)                         # noclip (the avatar bot)
    assert L(w, "function(p) return STEP(p) end")(p) is None
    L(w, "function(p) p.GetMoveType = function() return MOVETYPE_WALK end; p.InVehicle = function() return true end end")(p)
    assert L(w, "function(p) return STEP(p) end")(p) is None
