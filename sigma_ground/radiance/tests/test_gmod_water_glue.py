"""Water for the holodeck, the engine glue (SkippyVerse Phase A, task A2): the REAL water.lua, the REAL holodeck_water entity and the REAL
holodeck pipeline (parser, planner, ledger, undo, limits) against stage/gmod/holodeck/gmod_mock.lua extended with a small rigid-body layer
(gravity, velocity steps, a floor, sleeping).  The pure laws are pinned in test_gmod_water.py; here the question is whether the glue applies
them: that a floating block settles at the density-ratio draft and sleeps, a steel block sinks at the terminal speed the formula gives, g
decides (the Moon's water lifts less), a basin made under a resting body wakes it, the per-tick budget holds, and the commands behave.
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
from test_gmod_glue import World, A, B  # noqa: E402

from sigma_ground.field.interface import fluid  # noqa: E402
from sigma_ground.field.interface.fluid_surface import submerged_fraction  # noqa: E402
from sigma_ground.field.interface.projectile import G_EARTH  # noqa: E402
from sigma_ground.materia.bodies import resolve_body  # noqa: E402

HU_M = 0.01905
RHO_W = fluid.KNOWN_LIQUIDS["water"]["density_kg_m3"]
G_MOON = resolve_body("moon").surface_gravity_m_s2
LIM = json.loads(WCN.JSON_PATH.read_text(encoding="utf-8"))["limits"]

PRELUDE = r"""
-- a small rigid-body layer over gmod_mock.lua: gravity, velocity steps, a floor (the world's, or a basin's), sleeping
physenv = { g = Vector(0, 0, -600), SetGravity = function(v) physenv.g = v end, GetGravity = function() return physenv.g end }
CMDS = {}
function RunConsoleCommand(c, ...) CMDS[#CMDS + 1] = { c, ... } end
engine.TickInterval = function() return 0.015 end
function FrameTime() return 0.015 end
function net.WriteFloat(v) MOCK.cur.floats = MOCK.cur.floats or {}; MOCK.cur.floats[#MOCK.cur.floats + 1] = v end
function SetGlobalFloat(k, v) MOCK.globals = MOCK.globals or {}; MOCK.globals[k] = v end
function GetGlobalFloat(k, d) if MOCK.globals and MOCK.globals[k] ~= nil then return MOCK.globals[k] end return d end
function SetGlobalString(k, v) end
function SetGlobalBool(k, v) end
local function patch_cv(cv)
  if not cv.SetFloat then
    function cv:SetFloat(v) self:SetValue(v) end
    function cv:SetBool(v) self:SetValue(v and 1 or 0) end
    function cv:SetString(v) self:SetValue(v) end
  end
  return cv
end
local cc = CreateConVar
function CreateConVar(...) return patch_cv(cc(...)) end

local P = {}
P.__index = P
function P:IsValid() return true end
function P:GetMass() return self.mass or 1 end
function P:SetMass(m) self.mass = m end
function P:GetVelocity() return Vector(self.vel.x, self.vel.y, self.vel.z) end
function P:SetVelocity(v) self.vel = v end
function P:AddVelocity(v) self.vel = self.vel + v; self.adds = self.adds + 1 end
function P:IsAsleep() return self.asleep end
function P:Sleep() self.asleep = true; self.sleeps = (self.sleeps or 0) + 1 end
function P:Wake() self.asleep = false; self.wakes = self.wakes + 1; self.still = 0 end
function P:EnableMotion(b) self.motion = b end
function P:IsMotionEnabled() return self.motion end
function P:EnableDrag(b) self.drag = b end
function P:GetAngleVelocity() return Vector(self.w.x, self.w.y, self.w.z) end
function P:AddAngleVelocity(v) self.w = self.w + v; self.wadds = self.wadds + 1 end
function P:SetAngleVelocity(v) self.w = v end
function P:SetDamping() end
local orig_create = ents.Create
function ents.Create(class)
  local e = orig_create(class)
  if e == MOCK.NULL then return e end
  e.phys = setmetatable({ vel = Vector(0, 0, 0), w = Vector(0, 0, 0), asleep = false, motion = true, drag = true, wakes = 0, adds = 0, wadds = 0, mass = nil, still = 0 }, P)
  e.OBBMins = function(s) return s.obb_mins or Vector(-10, -10, 0) end
  e.OBBMaxs = function(s) return s.obb_maxs or Vector(10, 10, 20) end
  e.OBBCenter = function(s) local a, b = s:OBBMins(), s:OBBMaxs(); return Vector((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2) end
  e.LocalToWorld = function(s, p) return Vector(s.pos.x + p.x, s.pos.y + p.y, s.pos.z + p.z) end
  e.WorldToLocal = function(s, p) return Vector(p.x - s.pos.x, p.y - s.pos.y, p.z - s.pos.z) end
  e.GetModel = function(s) return s.model or "" end
  hook.Run("OnEntityCreated", e)
  return e
end
-- the basin entity's class: the REAL entity files, with the physics calls stubbed
MOCK.load_entity("holodeck_water", "server")
local W_ENT = MOCK.entity_classes["holodeck_water"].ENT
W_ENT.PhysicsInitMultiConvex = function(self, parts) self.parts = parts; return true end
W_ENT.PhysicsDestroy = function(self) self.parts = nil end
W_ENT.EnableCustomCollisions = function() end

ENGINE = { t = 0, ticks = 0 }
local function floor_at(x, y)
  local list = MentatWater and MentatWater.volumes.list() or {}
  for _, r in ipairs(list) do
    if MentatWater.volumes.inside_xy(r, x, y, 0) then return r.floor end
  end
  return 0
end
--- one physics tick: our Tick hook first (as the engine runs hooks before simulating), then the rigid bodies
function ENGINE.step()
  local dt = engine.TickInterval()
  MOCK.advance(dt)
  hook.Run("Tick")
  local gz = physenv.g.z
  for _, e in ipairs(MOCK.ents) do
    local p = e.phys
    if e.valid and p and e.simulate and p.motion and not p.asleep then
      p.vel.z = p.vel.z + gz * dt
      e.pos = e.pos + p.vel * dt
      local f = floor_at(e.pos.x, e.pos.y)
      if e.pos.z < f then
        e.pos.z = f
        if p.vel.z < 0 then p.vel.z = 0 end
      end
      local sp = p.vel:Length()
      if sp < 1.0 and e.pos.z <= f + 0.01 then p.still = p.still + dt else p.still = 0 end
      if p.still > 0.5 then p.asleep = true end             -- the engine sleeps what rests
    end
    if e.valid and e.simulate then e.maxz = math.max(e.maxz or -1e9, e.pos.z) end
  end
  ENGINE.ticks = ENGINE.ticks + 1
end
function ENGINE.run(seconds) local n = math.floor(seconds / engine.TickInterval() + 0.5); for _ = 1, n do ENGINE.step() end end

--- a loose body: size in HU (centred on x, y; its origin at the bottom), mass in kg, velocity in HU/s
function ENGINE.body(spec)
  local e = ents.Create("prop_physics")
  e:SetModel(spec.model or "models/props_junk/wood_crate001a.mdl")
  e.obb_mins = Vector(-spec.dims[1] / 2, -spec.dims[2] / 2, 0); e.obb_maxs = Vector(spec.dims[1] / 2, spec.dims[2] / 2, spec.dims[3])
  e:Spawn()
  e.pos = Vector(spec.x or 0, spec.y or 0, spec.z or 0)
  e.phys.mass = spec.mass
  e.phys.vel = Vector(spec.vx or 0, spec.vy or 0, spec.vz or 0)
  e.simulate = true
  if spec.asleep then e.phys.asleep = true end
  return e
end
"""


def build(g=None):
    w = World()
    w.rt.execute(PRELUDE)
    w.rt.execute('dofile(MOCK.ROOT .. "/mentat/world.lua")')
    w.rt.execute('dofile(MOCK.ROOT .. "/mentat/water.lua")')
    w.advance(0.01)
    if g is not None:
        w.rt.execute(f"SetGlobalFloat('mentat_world_g_m_s2', {g}); physenv.g = Vector(0, 0, -{g / HU_M})")
    return w


def L(w, expr):
    return w.rt.eval(expr)


def run(w, seconds):
    w.rt.eval("function(s) ENGINE.run(s) end")(seconds)


def pool_at(w, text="add a pool", player=None):
    p = player or w.player(A, admin=True)
    w.ask(p, text)
    w.advance(9)                      # the props materialise, then are released
    return p


def entries(w):
    return L(w, "(function() local o = {} for i, it in ipairs(MentatWater.entries()) do o[i] = it end return o end)()")


def n_entries(w):
    return L(w, "#MentatWater.entries()")


def water_ent(w, i=1):
    return L(w, f"MentatWater.entries()[{i}].ent")


def say_text(w):
    return " ".join(L(w, "(function() local o = {} for _, n in ipairs(MOCK.nets) do if n.name == 'holodeck_say' then o[#o+1] = tostring(n.writes[1]) end end return o end)()").values())


def body(w, dims_m, mass, x=0.0, y=0.0, z_m=0.0, vz=0.0, model="models/props_junk/wood_crate001a.mdl", asleep=False):
    d = [v / HU_M for v in dims_m]
    spec = w.rt.table_from({"dims": w.rt.table_from(d), "mass": mass, "x": x, "y": y, "z": z_m / HU_M, "vz": vz / HU_M, "model": model, "asleep": asleep})
    return w.rt.eval("function(spec) return ENGINE.body(spec) end")(spec)


def vol(w):
    return L(w, "MentatWater.volumes.list()[1]")


def surf_m(w):
    return L(w, "MentatWater.volumes.list()[1].surf") * HU_M


def floor_m(w):
    return L(w, "MentatWater.volumes.list()[1].floor") * HU_M


def z_of(w, e):
    return L(w, "function(e) return e.pos.z end")(e) * HU_M


def vz_of(w, e):
    return L(w, "function(e) return e.phys.vel.z end")(e) * HU_M


def asleep(w, e):
    return L(w, "function(e) return e.phys.asleep end")(e)


# ============================================================================================================ the ledger
def test_add_a_pool_makes_a_ledger_object_with_the_default_size():
    w = build()
    pool_at(w)
    assert n_entries(w) == 1 and w.count(A) == 1
    it = entries(w)[1]
    assert it.water.liquid == "water" and it.water.shape == "rect"
    assert (it.water.l_m, it.water.w_m, it.water.depth_m) == (10.0, 5.0, 2.0)                  # Aaron's default: 5 m x 10 m x 2 m
    ent = water_ent(w)
    assert L(w, "function(e) return e:GetClass() end")(ent) == "holodeck_water"
    assert L(w, "function(e) return #e.parts end")(ent) == 5                                    # floor and four walls
    assert L(w, "function(e) return e.HolodeckStatic end")(ent) is True
    assert L(w, "function(e) return e.phys.motion end")(ent) is False                           # a basin is not turned into a loose prop
    assert L(w, "function(e) return e:GetLiquid() end")(ent) == "water"
    assert L(w, "function(e) return e:GetFill() end")(ent) == 1.0


@pytest.mark.parametrize("text,shape,l,wd,d,liq", [
    ("add a pond", "round", 6.0, 6.0, 1.5, "water"),
    ("make a lake", "rect", 50.0, 50.0, 3.0, "water"),
    ("add a 20 by 8 metre pool", "rect", 20.0, 8.0, 2.0, "water"),
    ("add a pool 3 metres deep", "rect", 10.0, 5.0, 3.0, "water"),
    ("add a deep pond", "round", 6.0, 6.0, 3.0, "water"),
    ("add a pool of mercury", "rect", 10.0, 5.0, 2.0, "mercury"),
    ("add a seawater pool", "rect", 10.0, 5.0, 2.0, "seawater"),
])
def test_sizes_and_liquids_come_from_the_words(text, shape, l, wd, d, liq):
    w = build()
    pool_at(w, text)
    it = entries(w)[1]
    assert (it.water.shape, it.water.l_m, it.water.w_m, it.water.depth_m, it.water.liquid) == (shape, l, wd, d, liq)


def test_a_liquid_the_library_lacks_is_refused_and_nothing_is_made():
    w = build()
    p = w.player(A, admin=True)
    w.ask(p, "add a pool of olive oil")
    assert n_entries(w) == 0 and w.count() == 0
    text = say_text(w)
    assert "olive oil" in text and "library" in text


def test_filling_with_oil_is_refused_and_water_stays_water():
    w = build()
    p = pool_at(w)
    w.ask(p, "fill it with oil")
    assert L(w, "function(e) return e:GetLiquid() end")(water_ent(w)) == "water"
    assert "oil" in say_text(w)
    w.ask(p, "fill the pool with mercury")
    assert L(w, "function(e) return e:GetLiquid() end")(water_ent(w)) == "mercury"
    assert entries(w)[1].water.liquid == "mercury"


def test_drain_and_refill():
    w = build()
    p = pool_at(w)
    ent = water_ent(w)
    w.ask(p, "drain it")
    assert L(w, "function(e) return e:GetFill() end")(ent) == 0.0
    assert L(w, "(function() local v = MentatWater.volumes.list()[1] return v.surf - v.floor end)()") == 0
    w.ask(p, "fill it up")
    assert L(w, "function(e) return e:GetFill() end")(ent) == 1.0


def test_deeper_and_bigger_rebuild_the_basin_and_the_ledger_entry():
    w = build()
    p = pool_at(w)
    ent = water_ent(w)
    w.ask(p, "make the pool deeper")
    assert L(w, "function(e) return e:GetDepthM() end")(ent) == pytest.approx(3.0)
    assert entries(w)[1].water.depth_m == pytest.approx(3.0)
    reg_h = L(w, "Holodeck.registry[Holodeck.order[1]].h")
    assert reg_h == pytest.approx((0.2 + 3.0 + 0.05) / HU_M, rel=1e-6)                         # the planner's box follows
    w.ask(p, "make the pool bigger")
    assert L(w, "function(e) return e:GetLengthM() end")(ent) == pytest.approx(15.0)
    assert w.count(A) == 1


def test_undo_and_remove_work_like_any_ledger_object():
    w = build()
    p = pool_at(w)
    assert w.count(A) == 1
    w.ask(p, "remove the pool")
    w.advance(2)
    assert w.count(A) == 0 and n_entries(w) == 0
    w.ask(p, "undo")
    w.advance(9)
    assert w.count(A) == 1 and n_entries(w) == 1
    w.ask(p, "undo")
    assert w.count(A) == 0


def test_undo_brings_back_the_changed_pool_not_the_first_one():
    w = build()
    p = pool_at(w)
    w.ask(p, "make the pool deeper")
    w.ask(p, "fill it with mercury")
    w.ask(p, "remove the pool")
    w.advance(2)
    w.ask(p, "undo")
    w.advance(9)
    it = entries(w)[1]
    assert it.water.depth_m == pytest.approx(3.0) and it.water.liquid == "mercury"


def test_the_body_count_cap_holds_and_says_it_is_not_physics():
    w = build()
    p = w.player(A, admin=True)
    cap = int(LIM["max_bodies"]["value"])
    for _ in range(cap):
        w.ask(p, "add a 3 metre pond")
        w.advance(1)
    w.advance(9)
    assert n_entries(w) == cap and w.count(A) == cap
    w.ask(p, "add a 3 metre pond")
    w.advance(9)
    assert n_entries(w) == cap
    assert "not physics" in say_text(w)


def test_total_volume_cap():
    w = build()
    p = w.player(A, admin=True)
    w.ask(p, "add a 150 by 150 metre sea 5 metres deep")                                     # 112,500 m3: under the cap
    w.advance(9)
    assert n_entries(w) == 1
    w.ask(p, "add a 100 by 100 metre lake 5 metres deep")                                    # 50,000 more: over it
    w.advance(9)
    assert n_entries(w) == 1
    assert "cubic metres" in say_text(w)


def test_a_player_cannot_change_someone_elses_water():
    w = build()
    a = pool_at(w)
    b = w.player(B)
    w.ask(b, "drain it")
    assert L(w, "function(e) return e:GetFill() end")(water_ent(w)) == 1.0
    w.ask(a, "drain it")
    assert L(w, "function(e) return e:GetFill() end")(water_ent(w)) == 0.0


def test_typos_still_reach_the_water():
    w = build()
    p = w.player(A, admin=True)
    w.ask(p, "ad a pnod")
    w.advance(9)
    assert n_entries(w) == 1 and entries(w)[1].water.shape == "round"
    w.ask(p, "drian the pond")
    assert L(w, "function(e) return e:GetFill() end")(water_ent(w)) == 0.0


def test_object_sentences_still_go_to_the_object_parser():
    w = build()
    p = w.player(A, admin=True)
    w.ask(p, "add three couches")
    assert w.count(A) == 3
    w.ask(p, "add a pool table")
    assert n_entries(w) == 0


def test_a_ball_can_be_dropped_in_the_pool_it_is_placed_over_the_water():
    w = build()
    p = pool_at(w)
    before = w.count(A)
    w.ask(p, "drop a crate in the pool")
    w.advance(9)
    assert w.count(A) > before
    ball_z = L(w, "(function() for _, id in ipairs(Holodeck.order) do local e = Holodeck.registry[id] if not e.variant.water then return e.z end end end)()")
    water_top = L(w, "(function() local e = Holodeck.registry[Holodeck.order[1]] return e.z + e.h end)()")
    assert ball_z >= water_top - 1                                                             # at or above the rim: it will fall into the water


def test_it_sentences_reach_the_water_only_when_it_was_just_talked_about():
    w = build()
    p = w.player(A, admin=True)
    w.ask(p, "add a couch")
    w.ask(p, "make it bigger")                                    # about the couch: the object parser's, with no water in the room
    assert n_entries(w) == 0
    pool_at(w, player=p)
    w.ask(p, "add a couch")
    assert n_entries(w) == 1
    # a pool exists and the player has just spoken of it: "drain it" is about the pool
    w.ask(p, "drain the pool")
    w.ask(p, "fill it up")
    assert L(w, "function(e) return e:GetFill() end")(water_ent(w)) == 1.0
    # two minutes later "make it bigger" is no longer taken for the water (the player may mean the couch)
    w.advance(200)
    depth_before = L(w, "function(e) return e:GetLengthM() end")(water_ent(w))
    w.ask(p, "make it bigger")
    assert L(w, "function(e) return e:GetLengthM() end")(water_ent(w)) == depth_before


def test_a_tank_is_an_object_not_water():
    w = build()
    p = w.player(A, admin=True)
    w.ask(p, "add a tank")
    assert n_entries(w) == 0
