"""Water for the holodeck, how it looks (SkippyVerse Phase A, task A2): the client drawing run offline against a recording mesh/render mock.

The REAL render.lua, fx.lua and the REAL holodeck_water client entity, with the engine's mesh, render, cam and surface calls replaced by a
recorder that also CHECKS the contracts the engine enforces (a mesh declares as many primitives as it emits).  What is pinned: the optics
(Fresnel with the library's refractive index, Beer-Lambert with the library's absorption, total internal reflection at the critical angle
from Snell's law), the ripple and droplet laws (group velocity of the real dispersion relation, ballistics under the construct's g), the
triangle budget, and that every draw path runs.  The CPU cost of a frame's vertex work is MEASURED here (LuaJIT) and reported by the live run.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import pytest

lupa = pytest.importorskip("lupa")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage" / "gmod" / "holodeck"))
sys.path.insert(0, str(HERE))

import water_constants as WCN  # noqa: E402
from test_gmod_glue import World  # noqa: E402

from sigma_ground.field.interface import fluid, optics  # noqa: E402
from sigma_ground.field.interface.projectile import G_EARTH  # noqa: E402
from sigma_ground.materia.bodies import resolve_body  # noqa: E402

HU_M = 0.01905
G_MOON = resolve_body("moon").surface_gravity_m_s2
N_W = optics.cauchy_n("water", 550e-9)
R0 = optics.dielectric_surface_reflectance(N_W)
ABS = optics.DIELECTRIC_ABSORPTION["water_blue"]
LIQ = fluid.KNOWN_LIQUIDS["water"]
SIG_W = LIQ["surface_tension_n_m"] / LIQ["density_kg_m3"]

PRELUDE = r"""
-- a recording stand-in for the engine's drawing calls; it enforces the primitive counts
MATERIAL_TRIANGLES, MATERIAL_QUADS = 1, 2
REC = { meshes = 0, tris = 0, quads = 0, verts = 0, sprites = 0, draws = 0, bad = {}, mats = {}, fog = nil }
local cur
local function check_end()
  if not cur then return end
  local per = cur.prim == MATERIAL_TRIANGLES and 3 or 4
  if cur.verts ~= cur.count * per then REC.bad[#REC.bad + 1] = string.format("declared %d prims (%d verts) but emitted %d", cur.count, cur.count * per, cur.verts) end
  if cur.prim == MATERIAL_TRIANGLES then REC.tris = REC.tris + cur.count else REC.quads = REC.quads + cur.count end
  REC.verts = REC.verts + cur.verts
  cur = nil
end
mesh = {}
function mesh.Begin(a, b, c)
  if type(a) == "number" then cur = { prim = a, count = b, verts = 0 } else cur = { prim = b, count = c, verts = 0, imesh = a } end
end
function mesh.Position(v) end
function mesh.Color() end
function mesh.Normal() end
function mesh.TexCoord() end
function mesh.AdvanceVertex() cur.verts = cur.verts + 1 end
function mesh.End() check_end() end
function Mesh() REC.meshes = REC.meshes + 1; return { Draw = function() REC.draws = REC.draws + 1 end, Destroy = function() end } end
function CreateMaterial(name, shader, params) REC.mats[name] = params; return { name = name } end
render = render or {}
function render.SetMaterial() end
function render.SetColorMaterial() end
function render.DrawSprite() REC.sprites = REC.sprites + 1 end
function render.DrawWireframeBox() end
function render.DrawBox() end
function render.FogMode(m) REC.fog = REC.fog or {}; REC.fog.mode = m end
function render.FogStart(v) REC.fog.start = v end
function render.FogEnd(v) REC.fog["end"] = v end
function render.FogMaxDensity(v) REC.fog.density = v end
function render.FogColor(r, g, b) REC.fog.color = { r, g, b } end
MATERIAL_FOG_LINEAR = 1
cam = { PushModelMatrix = function() end, PopModelMatrix = function() end }
surface = { SetDrawColor = function(r, g, b, a) REC.tint = { r, g, b, a } end, DrawRect = function() REC.tint_rect = true end }
function ScrW() return 1920 end
function ScrH() return 1080 end
function EyePos() return REC.eye or Vector(0, 0, 100) end
function Entity(i) return REC.ent_by_index and REC.ent_by_index[i] or MOCK.NULL end
SysTime = function() return os.clock() end
"""


@pytest.fixture()
def cw():
    w = World(server=False)
    w.rt.execute(PRELUDE)
    w.rt.execute("""
      R = dofile(MOCK.ROOT .. "/mentat/water/render.lua")
      Fx = dofile(MOCK.ROOT .. "/mentat/water/fx.lua")
      Bs = dofile(MOCK.ROOT .. "/mentat/water/basin.lua")
      Wv = dofile(MOCK.ROOT .. "/mentat/water/waves.lua")
      Sp = dofile(MOCK.ROOT .. "/mentat/water/splash.lua")
    """)
    return w


def L(w, expr):
    return w.rt.eval(expr)


# ============================================================================================================ the optics
def test_fresnel_uses_the_librarys_refractive_index(cw):
    assert L(cw, "R.n_water") == N_W
    assert L(cw, "R.r0") == pytest.approx(R0)
    f = cw.rt.eval("R.fresnel")
    assert f(1.0) == pytest.approx(R0)                                         # looking straight down: clear
    assert f(0.0) == pytest.approx(1.0)                                         # grazing: a mirror
    # Schlick tracks the exact Fresnel equations within a few percent over the range that matters (unpolarised)
    def exact(cos_i):
        sin_t = math.sqrt(1 - cos_i ** 2) / N_W
        cos_t = math.sqrt(1 - sin_t ** 2)
        rs = ((cos_i - N_W * cos_t) / (cos_i + N_W * cos_t)) ** 2
        rp = ((N_W * cos_i - cos_t) / (N_W * cos_i + cos_t)) ** 2
        return (rs + rp) / 2
    for c in (0.9, 0.6, 0.4, 0.2, 0.1):
        assert f(c) == pytest.approx(exact(c), abs=0.07)
    vals = [f(c / 20) for c in range(20, -1, -1)]
    assert vals == sorted(vals)                                                 # more reflective as the angle grows


def test_beer_lambert_uses_the_librarys_absorption(cw):
    t = cw.rt.eval("R.transmit")
    r, g, b = t(3.0)
    assert (r, g, b) == pytest.approx((math.exp(-ABS[0] * 3), math.exp(-ABS[1] * 3), math.exp(-ABS[2] * 3)))
    assert r < g < b                                                            # water takes the red first: it goes blue
    d = cw.rt.eval("R.tile_colour")
    shallow, deep = d(0.5, 1.0), d(6.0, 1.0)
    assert shallow[0] > deep[0]                                                 # darker with depth
    dry = d(0.0, 1.0)
    tile = json.loads(WCN.JSON_PATH.read_text(encoding="utf-8"))["visual"]["tile_rgb"]
    assert tuple(dry) == pytest.approx(tuple(tile))
    deep_ratio = (deep[0] / deep[2]) / (dry[0] / dry[2])
    assert deep_ratio < 1.0                                                     # the deep floor is bluer than its tile


def test_looking_down_is_clear_and_grazing_is_a_mirror(cw):
    f = cw.rt.eval("R.surface_above")
    sun = cw.rt.eval("{ 0.0, 0.0, 1.0 }")
    down = f(0, 0, 0, 0, 1.0, 2.0, cw.rt.table_from([0.9, 0.0, 0.1]))             # eye straight above a flat patch
    graze = f(0, 0, 0.995, 0, 0.0999, 2.0, cw.rt.table_from([0.9, 0.0, 0.1]))
    assert down[3] < 0.2 and graze[3] > 0.55 and graze[3] > 3 * down[3]
    assert down[3] >= R0 * 0.99


def test_total_internal_reflection_starts_at_the_critical_angle_snells_law_gives(cw):
    f = cw.rt.eval("R.surface_below")
    crit = math.asin(1 / N_W)                                                  # from the library's n
    inside = f(0, 0, math.sin(crit - 0.05), 0, math.cos(crit - 0.05))
    outside = f(0, 0, math.sin(crit + 0.05), 0, math.cos(crit + 0.05))
    assert inside[3] < 0.7 and outside[3] > 0.7
    assert math.degrees(crit) == pytest.approx(48.2, abs=0.5)                   # the familiar window of about 49 degrees (n = 1.342 at 550 nm in the library)


def test_visible_distance_follows_absorption_plus_the_labelled_visual_scattering(cw):
    d = L(cw, "R.visible_distance_hu()")
    turb = L(cw, "R.TURBIDITY_PER_M")
    assert d == pytest.approx(3 / (ABS[1] + turb) / HU_M)
    assert 400 < d < 2000                                                       # a few tens of metres at most (in HU)


# ============================================================================================================ waves, ripples, droplets
def fx_state(cw):
    return cw.rt.eval("Fx.new()")


def test_a_ring_front_moves_at_the_group_velocity_of_the_real_dispersion_relation(cw):
    plan = cw.rt.eval("Sp.plan(10.0, 4.0, %f, %f, 48)" % (LIQ["density_kg_m3"], G_EARTH))
    st = fx_state(cw)
    cw.rt.eval("Fx.add_ripple")(st, 0.0, 0.0, plan, 0.0, G_EARTH, 2.0, SIG_W)
    r = st.ripples[1]
    cg = cw.rt.eval("Wv.group_speed")(plan.ring_lambda, G_EARTH, 2.0, SIG_W)
    assert r.cg == pytest.approx(cg)
    h = cw.rt.eval("Fx.height")
    # the packet peaks near the front r = cg t: the height there beats the height well inside it
    t = 1.0
    front = cg * t
    peak = max(abs(h(st, front + dx * plan.ring_lambda / 10, 0.0, t)) for dx in range(-10, 11))
    inside = abs(h(st, front * 0.3, 0.0, t))
    assert peak > 1.5 * inside
    assert abs(h(st, front + 6 * plan.ring_lambda, 0.0, t)) < 1e-3 * peak              # and nothing far beyond the front
    # nothing before the impact
    assert h(st, 0.0, 0.0, -1.0) == 0.0


def test_ripples_spread_slower_on_the_moon(cw):
    plan = cw.rt.eval("Sp.plan(10.0, 4.0, %f, %f, 48)" % (LIQ["density_kg_m3"], G_EARTH))
    st = fx_state(cw)
    cw.rt.eval("Fx.add_ripple")(st, 0, 0, plan, 0, G_EARTH, 2.0, SIG_W)
    cw.rt.eval("Fx.add_ripple")(st, 0, 0, plan, 0, G_MOON, 2.0, SIG_W)
    assert st.ripples[2].cg < st.ripples[1].cg


def test_droplets_fly_real_ballistics_and_land_when_the_flight_time_says(cw):
    plan = cw.rt.eval("Sp.plan(10.0, 4.0, %f, %f, 48)" % (LIQ["density_kg_m3"], G_EARTH))
    st = fx_state(cw)
    cw.rt.eval("Fx.add_splash")(st, 0, 0, plan, 0.0, G_EARTH, 2.0, SIG_W)
    n0 = len(st.drops)
    assert n0 == plan.n and n0 > 0
    flights = sorted(2 * st.drops[i].vz / G_EARTH for i in range(1, n0 + 1))            # up and back to the surface, from the real ballistics
    upd = cw.rt.eval("Fx.update")
    dt, t = 0.002, 0.0
    for check in (flights[0] * 0.5, (flights[0] + flights[-1]) / 2, flights[-1] * 1.05):
        while t < check:
            upd(st, dt, G_EARTH, t)
            t += dt
        expect_up = sum(1 for f in flights if f > t)
        assert abs(len(st.drops) - expect_up) <= 1                                       # each drop lands when its own flight time says
    assert len(st.drops) == 0
    # on the Moon the same drop stays up about six times longer
    st2 = fx_state(cw)
    cw.rt.eval("Fx.add_splash")(st2, 0, 0, plan, 0.0, G_MOON, 2.0, SIG_W)
    assert 2 * st2.drops[1].vz / G_MOON == pytest.approx(2 * st2.drops[1].vz / G_EARTH * G_EARTH / G_MOON)


def test_the_drop_and_ring_budgets_hold(cw):
    plan = cw.rt.eval("Sp.plan(5000.0, 30.0, %f, %f, 48)" % (LIQ["density_kg_m3"], G_EARTH))
    st = fx_state(cw)
    for i in range(100):
        cw.rt.eval("Fx.add_splash")(st, i, 0, plan, 0.0, G_EARTH, 2.0, SIG_W)
    assert len(st.drops) <= L(cw, "Fx.MAX_DROPS") and len(st.ripples) <= L(cw, "Fx.MAX_RIPPLES")


def test_old_ripples_are_dropped(cw):
    plan = cw.rt.eval("Sp.plan(10.0, 4.0, %f, %f, 48)" % (LIQ["density_kg_m3"], G_EARTH))
    st = fx_state(cw)
    cw.rt.eval("Fx.add_ripple")(st, 0, 0, plan, 0.0, G_EARTH, 2.0, SIG_W)
    cw.rt.eval("Fx.update")(st, 0.01, G_EARTH, 100.0)
    assert len(st.ripples) == 0


# ============================================================================================================ the basin and the grid
def test_basin_triangles_are_darker_and_bluer_with_depth(cw):
    b = cw.rt.eval("Bs.new('rect', 10.0, 5.0, 2.0)")
    tris = cw.rt.eval("R.basin_triangles")(b, 1.0)
    n = len(tris)
    assert n > 100
    floor_col, wall_top, wall_bottom = None, None, None
    for i in range(1, n + 1):
        t = tris[i]
        zs = [t[1][3], t[2][3], t[3][3]]
        if max(zs) == min(zs) == b.floor_z:
            floor_col = t[4]
        if min(zs) == pytest.approx(b.floor_z) and max(zs) > b.floor_z and max(zs) < b.floor_z + 0.2 * b.depth:
            wall_bottom = t[4]
    assert floor_col is not None
    dry = cw.rt.eval("R.tile_colour")(0.0, 1.0)
    assert floor_col[1] < dry[1] and floor_col[1] < dry[2] + 1                    # the wet floor is darker than a dry tile
    assert floor_col[3] > floor_col[1] * 0.9 or floor_col[2] > floor_col[1]       # and bluer: red is absorbed first
    tris0 = cw.rt.eval("R.basin_triangles")(b, 0.0)
    assert len(tris0) == n                                                         # drained: same shape, no tint


def test_the_grid_window_caps_the_cells_and_follows_the_eye_on_a_big_body(cw):
    cap = L(cw, "R.CELLS_MAX")
    hx = 75.0 / HU_M                                                               # a 150 m sea
    x0, x1, y0, y1, nx, ny = cw.rt.eval("R.grid_window")(hx, hx, 10 / HU_M, -20 / HU_M)
    assert nx <= cap and ny <= cap
    assert (x1 - x0) * HU_M == pytest.approx(L(cw, "R.WINDOW_M"))
    assert x0 <= 10 / HU_M <= x1 and y0 <= -20 / HU_M <= y1                        # the camera is inside the gridded window
    assert -hx <= x0 and x1 <= hx
    px0, px1, _, _, pnx, pny = cw.rt.eval("R.grid_window")(5.0 / HU_M, 2.5 / HU_M, 0, 0)
    assert px0 == -5.0 / HU_M and px1 == 5.0 / HU_M                                # a pool is gridded whole


# ============================================================================================================ the entity, drawn
def make_entity(cw, shape="rect", l=10.0, w=5.0, d=2.0, fill=1.0, liquid="water"):
    cw.rt.execute('MOCK.load_entity("holodeck_water", "client")')
    cw.rt.execute("MOCK.me = MOCK.me or MOCK.newplayer('viewer', {})")
    cw.rt.execute("""
      local ENT = MOCK.entity_classes["holodeck_water"].ENT
      ENT.SetRenderBounds = function() end
      ENT.SetNextClientThink = function() end
      ENT.GetWorldTransformMatrix = function() return {} end
      ENT.WorldToLocal = function(s, p) return Vector(p.x - s.pos.x, p.y - s.pos.y, p.z - s.pos.z) end
      ENT.GetColor = function() return { r = 255, g = 255, b = 255, a = 255 } end
    """)
    return cw.rt.eval(f"""(function()
      local e = ents.Create("holodeck_water")
      e:SetFound({{ name = "pool", water = {{ shape = "{shape}", l_m = {l}, w_m = {w}, depth_m = {d}, liquid = "{liquid}", fill = {fill} }} }})
      e.pos = Vector(0, 0, 0)
      e:SetBornAt(-100)
      e:Initialize()
      return e
    end)()""")


@pytest.mark.parametrize("shape,l,w,d", [("rect", 10.0, 5.0, 2.0), ("round", 6.0, 6.0, 1.5), ("rect", 150.0, 150.0, 5.0), ("rect", 3.0, 3.0, 0.5)])
def test_a_body_draws_within_the_triangle_budget_and_every_mesh_is_well_formed(cw, shape, l, w, d):
    e = make_entity(cw, shape, l, w, d)
    cw.rt.execute("REC.eye = Vector(30, 20, 200)")
    cw.rt.eval("function(e) e:DrawTranslucent() end")(e)
    assert list(L(cw, "REC.bad").values()) == []                                  # declared primitive counts matched what was emitted
    tris = L(cw, "Mentat.WaterRender.stats.tris")
    cap = L(cw, "R.CELLS_MAX")
    basin = cw.rt.eval("function(e) return Mentat.WaterRender.state_of(e).basin_tris end")(e)
    assert 0 < tris - basin <= 2 * cap * cap + 4 * 2 + 400                         # the surface grid + flat remainder + foam, per frame
    assert L(cw, "REC.draws") >= 1


def test_the_basin_mesh_is_rebuilt_only_when_the_basin_changes(cw):
    e = make_entity(cw)
    cw.rt.execute("REC.eye = Vector(30, 20, 200)")
    draw = cw.rt.eval("function(e) e:DrawTranslucent() end")
    draw(e)
    m1 = L(cw, "REC.meshes")
    draw(e); draw(e)
    assert L(cw, "REC.meshes") == m1
    cw.rt.eval("function(e) e:SetBasin('rect', 10, 5, 3) end")(e)
    draw(e)
    assert L(cw, "REC.meshes") == m1 + 1


def test_the_camera_under_the_surface_gets_a_tint_and_fog_from_the_library_optics(cw):
    cw.rt.execute('dofile(MOCK.ROOT .. "/mentat/water_client.lua")')
    e = make_entity(cw)
    cw.rt.execute("REC.eye = Vector(0, 0, 50)")                                    # inside the basin, below the surface
    cw.rt.eval("function(e) e:DrawTranslucent() end")(e)
    cw.rt.execute("MOCK.me = MOCK.newplayer('x', {})")
    cw.rt.execute("hook.Run('HUDPaintBackground')")
    assert L(cw, "REC.tint") is not None and L(cw, "REC.tint_rect") is True
    cw.rt.execute("hook.Run('SetupWorldFog')")
    fog = L(cw, "REC.fog")
    assert fog is not None and fog["end"] == pytest.approx(L(cw, "R.visible_distance_hu()"))
    # above the water: no tint, no fog
    cw.rt.execute("REC.tint = nil; REC.fog = nil; REC.eye = Vector(0, 0, 400)")
    cw.rt.execute("hook.Run('HUDPaintBackground')")
    cw.rt.execute("hook.Run('SetupWorldFog')")
    assert L(cw, "REC.tint") is None and L(cw, "REC.fog") is None


def test_server_splash_messages_become_rings_and_droplets_scaled_by_the_energy(cw):
    cw.rt.execute('dofile(MOCK.ROOT .. "/mentat/water_client.lua")')
    e = make_entity(cw)
    cw.rt.execute("REC.ent_by_index = {}")
    cw.rt.eval("function(e) REC.ent_by_index[e:EntIndex()] = e end")(e)
    # feed the receiver by hand: a 10 kg body at 4 m/s, then a 0.5 g leaf at 1 m/s
    cw.rt.execute("""
      local function msg(idx, kind, x, y, z, mass, speed)
        local q = { idx, kind }
        local f = { x, y, z, mass, speed }
        net.ReadUInt = function() return table.remove(q, 1) end
        net.ReadFloat = function() return table.remove(f, 1) end
        MOCK.receivers["mentat_water_splash"]()
      end
      MSG = msg
    """)
    idx = cw.rt.eval("function(e) return e:EntIndex() end")(e)
    cw.rt.eval("function(i) MSG(i, 1, 0, 0, 0, 10.0, 4.0) end")(idx)
    st = cw.rt.eval("function(e) return Mentat.WaterRender.state_of(e) end")(e)
    big = len(st.fx.drops), len(st.fx.ripples)
    cw.rt.eval("function(i) MSG(i, 1, 100, 0, 0, 0.0005, 1.0) end")(idx)
    assert len(st.fx.ripples) == 2 and len(st.fx.drops) > big[0]
    assert st.fx.ripples[1].amp > st.fx.ripples[2].amp                             # the big one is the bigger ring
    cw.rt.eval("function(i) MSG(i, 3, 0, 0, 0, 10.0, 0.3) end")(idx)                 # bobbing: a ring and no spray
    before = len(st.fx.drops)
    cw.rt.eval("function(i) MSG(i, 3, 0, 0, 0, 10.0, 0.3) end")(idx)
    assert len(st.fx.drops) == before


def test_in_free_fall_the_surface_is_still_and_nothing_breaks(cw):
    e = make_entity(cw)
    cw.rt.execute("REC.eye = Vector(30, 20, 200); SetGlobalFloat = SetGlobalFloat or function() end")
    cw.rt.execute("MOCK.globals = { mentat_world_g_m_s2 = 0 }")
    cw.rt.execute("GetGlobalFloat = function(k, d) return (MOCK.globals or {})[k] or d end")
    cw.rt.eval("function(e) e:DrawTranslucent() end")(e)
    assert list(L(cw, "REC.bad").values()) == []


def test_the_frame_cost_is_reported_and_is_small(cw):
    e = make_entity(cw, "rect", 10.0, 5.0, 2.0)
    cw.rt.execute("REC.eye = Vector(30, 20, 200)")
    draw = cw.rt.eval("function(e) e:DrawTranslucent() end")
    for _ in range(5):
        draw(e)                                                                        # warm up the JIT
    t0 = time.perf_counter()
    n = 120
    for _ in range(n):
        draw(e)
    ms = (time.perf_counter() - t0) * 1000 / n
    print(f"\nwater surface + basin CPU work: {ms:.3f} ms per frame (LuaJIT, mock engine calls)")
    assert ms < 6.0                                                                    # (the mock's own calls dominate; the live run measures the real cost)
