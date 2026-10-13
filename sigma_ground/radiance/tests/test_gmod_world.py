"""The construct's physics (SkippyVerse Phase A, task A1): real-world gravity and real aerodynamic drag, nothing else.

Aaron, 2026-10-12: only real-world physics.  So gravity is the library's Earth value (or a real place's: Moon, Mars, Jupiter, free
fall), air drag is the real F = 1/2 rho Cd A v^2 per object, and there is no multiplier and no universal damping number.  These tests
pin that down offline: the generated constants against the sigma-ground library (never literals), the pure drag math against the
library's own formulas, the words, and the engine glue (world.lua) against stage/gmod/holodeck/gmod_mock.lua extended with a small
physics layer.
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

import world_constants as WC  # noqa: E402
from test_gmod_glue import World, A, B  # noqa: E402

from sigma_ground.field.interface import atmosphere as atm  # noqa: E402
from sigma_ground.field.interface.projectile import G_EARTH, drag_force, terminal_velocity  # noqa: E402
from sigma_ground.dynamics.mechanisms.wind import C_N_FLAT_PLATE  # noqa: E402
from sigma_ground.materia.bodies import resolve_body  # noqa: E402

ADDON = WC.GMOD / "addon" / "lua"
HU_M = 0.01905

PRELUDE = r"""
-- a small physics layer over gmod_mock.lua: what world.lua needs from the engine
physenv = { g = Vector(0, 0, -600), SetGravity = function(v) physenv.g = v end, GetGravity = function() return physenv.g end }
CMDS = {}
function RunConsoleCommand(c, ...) CMDS[#CMDS + 1] = { c, ... } end
engine.TickInterval = function() return 0.015 end
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
local orig_create = ents.Create
local P = {}
P.__index = P
function ents.Create(class)
  local e = orig_create(class)
  if e == MOCK.NULL then return e end
  local old = e.phys
  e.phys = setmetatable({ vel = Vector(0, 0, 0), asleep = false, motion = true, drag = true, wakes = 0, adds = 0, mass = old.mass }, P)
  e.OBBMins = function(s) return s.obb_mins or Vector(-10, -10, 0) end
  e.OBBMaxs = function(s) return s.obb_maxs or Vector(10, 10, 20) end
  e.WorldToLocal = function(s, p) return Vector(p.x - s.pos.x, p.y - s.pos.y, p.z - s.pos.z) end
  e.GetModel = function(s) return s.model or "" end
  if not e.IsPlayer then e.IsPlayer = function() return false end end
  hook.Run("OnEntityCreated", e)
  return e
end
function P:IsValid() return true end
function P:GetMass() return self.mass or 1 end
function P:SetMass(m) self.mass = m end
function P:GetVelocity() return Vector(self.vel.x, self.vel.y, self.vel.z) end
function P:SetVelocity(v) self.vel = v end
function P:AddVelocity(v) self.vel = self.vel + v; self.adds = self.adds + 1 end
function P:IsAsleep() return self.asleep end
function P:Sleep() self.asleep = true end
function P:Wake() self.asleep = false; self.wakes = self.wakes + 1 end
function P:EnableMotion(b) self.motion = b end
function P:IsMotionEnabled() return self.motion end
function P:EnableDrag(b) self.drag = b end
function P:IsDragEnabled() return self.drag end
function P:SetDamping(l, a) self.damp = { l, a } end
function P:GetDamping() local d = self.damp or { 0, 0 }; return d[1], d[2] end
function P:EnableMotion(b) self.motion = b end
"""


def world(cvars=None, load_world=True):
    w = World(cvars=cvars)
    w.rt.execute(PRELUDE)
    if load_world:
        w.rt.execute('dofile(MOCK.ROOT .. "/mentat/world.lua")')
        w.advance(0.01)
    return w


def L(w, expr):
    return w.rt.eval(expr)


def prop(w, mass=10.0, vel=(0, 0, 0), dims_hu=(20, 20, 20), model="models/props_junk/wood_crate001a.mdl", asleep=False):
    return w.rt.eval(
        f"""(function()
          local e = ents.Create("prop_physics")
          e:SetModel("{model}")
          e.obb_mins = Vector({-dims_hu[0] / 2}, {-dims_hu[1] / 2}, 0); e.obb_maxs = Vector({dims_hu[0] / 2}, {dims_hu[1] / 2}, {dims_hu[2]})
          e:Spawn()
          e.phys.mass = {mass}; e.phys.vel = Vector({vel[0]}, {vel[1]}, {vel[2]}); e.phys.asleep = {str(asleep).lower()}
          return e
        end)()""")


# ============================================================================================================ the generated constants
def test_generated_files_are_current():
    js, lua = WC.render()
    assert WC.JSON_PATH.read_text(encoding="utf-8") == js
    assert WC.LUA_PATH.read_text(encoding="utf-8") == lua


def test_lua_table_is_valid_and_has_provenance_on_every_entry():
    w = world(load_world=False)
    w.rt.execute('Cn = dofile(MOCK.ROOT .. "/mentat/world/constants.lua")')
    res = L(w, "(function() local ok, err = Cn.validate_data() return tostring(ok) .. ':' .. tostring(err) end)()")
    assert res == "true:nil", res
    d = json.loads(WC.JSON_PATH.read_text(encoding="utf-8"))
    for group in ("engine", "entries"):
        for name, e in d[group].items():
            assert e["source"] and e["status"] and "units" in e and "value" in e, name
    for name, p in d["presets"].items():
        assert p["source"] and p["status"], name
    for name, c in d["cd"].items():
        assert c["source"] and c["status"] and c["value"] > 0, name


def test_presets_equal_the_library_not_literals():
    d = json.loads(WC.JSON_PATH.read_text(encoding="utf-8"))
    P = d["presets"]
    assert P["earth"]["gravity_m_s2"] == G_EARTH
    for name in ("moon", "mars", "jupiter"):
        assert P[name]["gravity_m_s2"] == resolve_body(name).surface_gravity_m_s2
    assert P["zero"]["gravity_m_s2"] == 0.0
    assert set(P) == {"earth", "moon", "mars", "jupiter", "zero"}                 # nothing arbitrary
    for name, p in P.items():                                                       # HU conversion is the stage scale
        assert p["gravity_hu_s2"] == pytest.approx(p["gravity_m_s2"] / HU_M, rel=1e-12)


def test_default_is_real_earth_and_no_gameplay_numbers_remain():
    d = json.loads(WC.JSON_PATH.read_text(encoding="utf-8"))
    assert d["default_preset"] == "earth"
    text = WC.JSON_PATH.read_text(encoding="utf-8")
    assert "specified" not in text and "SetDamping" not in text
    assert not any(k in d["entries"] for k in ("drag", "gravity_mult"))            # no multiplier, no blanket coefficient
    assert "F = 1/2 rho Cd A v^2" in d["entries"]["drag_law"]["value"]


def test_hu_scale_is_the_repos_own_constant():
    w = world(load_world=False)
    w.rt.execute('U = dofile(MOCK.ROOT .. "/mentat/units.lua")')
    assert 1 / L(w, "U.HU_PER_M") == pytest.approx(HU_M, rel=1e-12)
    d = json.loads(WC.JSON_PATH.read_text(encoding="utf-8"))
    assert d["hu_m"] == pytest.approx(1 / L(w, "U.HU_PER_M"), rel=1e-12)
    import fidelity
    assert fidelity.HU_M == d["hu_m"]


def test_engine_default_gravity_is_stronger_than_real_by_the_libraries_numbers():
    d = json.loads(WC.JSON_PATH.read_text(encoding="utf-8"))
    engine_m_s2 = d["engine_gravity_hu_s2"] * d["hu_m"]
    assert d["engine"]["gravity_m_s2"]["value"] == pytest.approx(engine_m_s2)
    ratio = engine_m_s2 / G_EARTH
    assert ratio > 1.1                                                              # the engine's 600 HU/s2 is about 17 % too strong
    assert "%.4g times" % ratio in d["engine"]["gravity_m_s2"]["source"]


def test_air_density_and_flat_plate_come_from_the_library():
    d = json.loads(WC.JSON_PATH.read_text(encoding="utf-8"))
    assert d["entries"]["air_density"]["value"] == atm.air_density(atm._T_STANDARD, atm._P_STANDARD)
    assert d["entries"]["air_density"]["value"] == pytest.approx(1.225, rel=1e-3)
    assert d["cd"]["flat_plate"]["value"] == C_N_FLAT_PLATE
    for k in ("sphere", "box", "cylinder"):                                         # not in the library: labelled, with a citation
        assert d["cd"][k]["status"] == "estimated, owner-should-confirm"
        assert any(a in d["cd"][k]["source"] for a in ("White", "Hoerner"))


# ============================================================================================================ the drag math
def aero(w=None):
    w = w or world(load_world=False)
    w.rt.execute('Ae = dofile(MOCK.ROOT .. "/mentat/world/aero.lua"); Cn = dofile(MOCK.ROOT .. "/mentat/world/constants.lua")')
    return w


@pytest.mark.parametrize("speed", [1.0, 5.0, 20.0, 50.0])
def test_drag_force_matches_the_library_for_a_sphere_and_a_cube(speed):
    w = aero()
    rho = L(w, "Cn.AIR_DENSITY")
    # sphere of 0.2 m diameter
    cd_s = L(w, "Cn.CD.sphere.value")
    area_s = L(w, "Ae.area('sphere', {0.2, 0.2, 0.2}, {1, 0, 0})")
    assert area_s == pytest.approx(math.pi / 4 * 0.2 ** 2)
    assert L(w, f"Ae.force({rho}, {cd_s}, {area_s}, {speed})") == pytest.approx(drag_force(speed, cd_s, area_s, rho), rel=1e-12)
    # cube of 0.3 m, face on
    cd_b = L(w, "Cn.CD.box.value")
    area_b = L(w, "Ae.area('box', {0.3, 0.3, 0.3}, {0, 0, 1})")
    assert area_b == pytest.approx(0.3 * 0.3)
    assert L(w, f"Ae.force({rho}, {cd_b}, {area_b}, {speed})") == pytest.approx(drag_force(speed, cd_b, area_b, rho), rel=1e-12)


def test_box_shadow_along_a_diagonal_is_exact():
    w = aero()
    s = 1 / math.sqrt(3)
    a = L(w, f"Ae.area('box', {{0.3, 0.3, 0.3}}, {{{s}, {s}, {s}}})")
    assert a == pytest.approx(3 * s * 0.09)                                          # a cube seen corner-on casts sqrt(3) * face area


def test_speed_after_is_the_exact_drag_only_solution_and_never_reverses():
    w = aero()
    k = 0.02
    v0 = 40.0
    one = L(w, f"Ae.speed_after({v0}, {k}, 1.5)")
    assert one == pytest.approx(v0 / (1 + k * v0 * 1.5))
    v = v0
    for _ in range(100):
        v = L(w, f"Ae.speed_after({v}, {k}, 0.015)")
    assert v == pytest.approx(one, rel=1e-9)                                         # 100 ticks of 15 ms == one step of 1.5 s
    assert L(w, f"Ae.speed_after(1e6, 1e6, 1)") > 0


def test_terminal_speed_matches_the_library():
    w = aero()
    rho = L(w, "Cn.AIR_DENSITY")
    m, cd, area = 0.004, 1.28, 0.01
    assert L(w, f"Ae.terminal({m}, {G_EARTH}, {rho}, {cd}, {area})") == pytest.approx(terminal_velocity(m, cd, area, rho, G_EARTH), rel=1e-12)


@pytest.mark.parametrize("name,dims,expect", [
    ("basketball", (0.24, 0.24, 0.24), "sphere"),
    ("models/props_c17/oildrum001.mdl", (0.58, 0.58, 0.88), "cylinder"),
    ("models/props_junk/wood_crate001a.mdl", (0.58, 0.58, 0.5), "box"),
    ("barrel", (0.58, 0.58, 0.88), "cylinder"),
    ("crate", (0.5, 0.5, 0.5), "box"),
    ("sheet of paper", (0.3, 0.2, 0.001), "flat_plate"),
    ("ball of string", (0.1, 0.1, 0.1), "sphere"),
    ("ball pit slab", (1.0, 1.0, 0.05), "flat_plate"),
])
def test_shape_classes(name, dims, expect):
    w = aero()
    got = L(w, f"Ae.shape_class('{name}', {{{dims[0]}, {dims[1]}, {dims[2]}}})")
    assert got == expect


# ============================================================================================================ the words
INTENT = [
    ("set gravity to the moon", [("gravity_preset", "moon")]),
    ("moon gravity", [("gravity_preset", "moon")]),
    ("Skippy, set gravity to mars", [("gravity_preset", "mars")]),
    ("set the gravity to jupiter", [("gravity_preset", "jupiter")]),
    ("zero gravity", [("gravity_preset", "zero")]),
    ("no gravity", [("gravity_preset", "zero")]),
    ("turn gravity off", [("gravity_preset", "zero")]),
    ("normal gravity", [("gravity_preset", "earth")]),
    ("earth gravity", [("gravity_preset", "earth")]),
    ("reset gravity", [("gravity_preset", "earth")]),
    ("real gravity", [("gravity_preset", "earth")]),
    ("set gravity to the moom", [("gravity_preset", "moon")]),                       # typos
    ("set gravty to the moon", [("gravity_preset", "moon")]),
    ("gravitty to mars", [("gravity_preset", "mars")]),
    ("zero gravity and earth gravity", [("gravity_preset", "zero"), ("gravity_preset", "earth")]),
    ("half gravity", [("not_real", "gravity")]),                                      # gameplay knobs are refused, not applied
    ("double gravity", [("not_real", "gravity")]),
    ("set gravity to 3", [("not_real", "gravity")]),
    ("more air drag", [("not_real", "drag")]),
    ("no air drag", [("not_real", "drag")]),
    ("less air resistance", [("not_real", "drag")]),
    ("what are the world's physics", [("query", "world")]),
    ("what are the worlds physics", [("query", "world")]),
    ("what is the gravity", [("query", "gravity")]),
    ("how strong is gravity", [("query", "gravity")]),
    ("what is the air drag", [("query", "drag")]),
    ("tell me the world physics", [("query", "world")]),
]
NOT_OURS = ["add a couch", "add three chairs and a table", "drag the couch to the window", "move the lamp", "gravity gun",
            "give me a gravity gun", "add a couch and zero gravity", "undo", "clear everything", "add some gravy", "make it a beach",
            "hello", "", "remove the table", "put the lamp on the moon"]


@pytest.mark.parametrize("text,expect", INTENT)
def test_intents(text, expect):
    w = aero()
    w.rt.execute('In = dofile(MOCK.ROOT .. "/mentat/world/intent.lua")')
    got = L(w, "function(t) local r = In.parse(t); if not r then return '' end; local o = {}; for _, it in ipairs(r) do o[#o+1] = it.kind .. ':' .. (it.name or it.what or '') end; return table.concat(o, ',') end")(text)
    assert got == ",".join(f"{k}:{n}" for k, n in expect)


@pytest.mark.parametrize("text", NOT_OURS)
def test_object_requests_are_never_taken_for_physics(text):
    w = aero()
    w.rt.execute('In = dofile(MOCK.ROOT .. "/mentat/world/intent.lua")')
    assert L(w, "function(t) return In.parse(t) == nil end")(text), text


# ============================================================================================================ the engine glue
def gravity_hu(w):
    return -L(w, "physenv.g.z")


def say_texts(w):
    return [m for m in L(w, "function() local o = {} for _, n in ipairs(MOCK.nets) do if n.name == 'holodeck_say' then o[#o+1] = n.writes[1] end end return o end")().values()]


def test_gravity_is_the_librarys_earth_in_engine_units_at_start():
    w = world()
    assert gravity_hu(w) == pytest.approx(G_EARTH / HU_M, rel=1e-9)
    cmds = [(c[1], c[2]) for c in L(w, "CMDS").values()]
    sv = [float(v) for k, v in cmds if k == "sv_gravity"]
    assert sv and sv[-1] == pytest.approx(G_EARTH / HU_M, rel=1e-6)                 # players follow the same number
    assert L(w, "GetGlobalFloat('mentat_world_g_m_s2')") == G_EARTH
    assert 500 < G_EARTH / HU_M < 530 < 600                                          # a little under the engine's 600


@pytest.mark.parametrize("phrase,body", [("set gravity to the moon", "moon"), ("mars gravity", "mars"), ("jupiter gravity", "jupiter")])
def test_a_real_place_sets_the_librarys_value(phrase, body):
    w = world()
    admin = w.player(A, admin=True)
    w.ask(admin, phrase)
    assert gravity_hu(w) == pytest.approx(resolve_body(body).surface_gravity_m_s2 / HU_M, rel=1e-9)
    assert L(w, "GetGlobalString('mentat_world_preset')") == body
    assert any("m/s2" in t for t in say_texts(w))                                    # Skippy announces it
    assert any(body in line.lower() or resolve_body(body).label.lower() in line.lower() for line in L(w, "MentatWorld.log").values())


def test_zero_gravity_and_back_to_earth():
    w = world()
    admin = w.player(A, admin=True)
    w.ask(admin, "zero gravity")
    assert gravity_hu(w) == 0
    w.ask(admin, "normal gravity")
    assert gravity_hu(w) == pytest.approx(G_EARTH / HU_M, rel=1e-9)


def test_typos_are_tolerated_in_the_game_too():
    w = world()
    admin = w.player(A, admin=True)
    w.ask(admin, "set gravty to the moom")
    assert gravity_hu(w) == pytest.approx(resolve_body("moon").surface_gravity_m_s2 / HU_M, rel=1e-9)


def test_gameplay_knobs_change_nothing_and_skippy_says_why():
    w = world()
    admin = w.player(A, admin=True)
    for phrase in ("half gravity", "double gravity", "more air drag", "no air drag"):
        w.ask(admin, phrase)
        assert gravity_hu(w) == pytest.approx(G_EARTH / HU_M, rel=1e-9)
    assert any("real" in t.lower() for t in say_texts(w))


def test_admin_restriction():
    w = world()
    nobody = w.player(A)
    w.ask(nobody, "set gravity to the moon")
    assert gravity_hu(w) == pytest.approx(G_EARTH / HU_M, rel=1e-9)                  # refused on a dedicated server
    host = w.player(B, host=True)
    w.ask(host, "set gravity to the moon")
    assert L(w, "GetGlobalString('mentat_world_preset')") == "moon"                  # the listen-server host may
    w2 = world(cvars={"mentat_world_admin_only": "0"})
    w2.ask(w2.player(A), "set gravity to mars")
    assert L(w2, "GetGlobalString('mentat_world_preset')") == "mars"
    w3 = world()
    w3.rt.execute("game.SinglePlayer = function() return true end")
    w3.ask(w3.player(A), "set gravity to jupiter")
    assert L(w3, "GetGlobalString('mentat_world_preset')") == "jupiter"             # single player may


def test_anyone_can_ask_what_the_physics_are_and_the_readback_is_honest():
    w = world()
    p = w.player(A)
    r = w.ask(p, "what are the world's physics")
    text = " ".join(L(w, "function() local o = {} for _, n in ipairs(MOCK.nets) do if n.name == 'holodeck_say' then o[#o+1] = n.writes[1] end end return o end")().values())
    lines = list(L(w, "MentatWorld.constants.readback(MentatWorld.state())").values())
    full = " ".join(lines)
    for needle in ("9.81", "m/s2", "HU/s2", "library", "estimated, owner-should-confirm", "G_EARTH", "Thermodynamics", "1/2 rho Cd A v"):
        assert needle in full, needle
    assert "Earth" in text and "m/s2" in text
    assert all(len(l) <= 240 for l in lines)


def test_commands_are_logged():
    w = world()
    w.ask(w.player(A, admin=True), "set gravity to mars")
    log = list(L(w, "MentatWorld.log").values())
    assert any("set gravity to mars" in l for l in log)


# ----- the objects
def test_new_props_lose_the_engines_drag_and_are_tracked_but_players_are_not():
    w = world()
    e = prop(w)
    w.advance(0.01)
    assert L(w, "function(e) return e.phys.drag end")(e) is False                    # Source's own air drag off: ours is the only one
    assert L(w, "function(e) return MentatWorld.tracked[e] ~= nil end")(e) is True
    assert L(w, "function(e) return e.phys.damp end")(e) is None                    # and no damping number is invented
    p = w.player(A)
    assert L(w, "function(p) return MentatWorld.is_free_object(p) end")(p) is False
    bot = L(w, "(function() local b = MOCK.newplayer('bot', {}); b.bot = true; return b end)()")
    w.advance(3.5)
    assert L(w, "function(b) return MentatWorld.tracked[b] == nil end")(bot) is True


def test_props_that_existed_before_the_world_loaded_are_picked_up():
    w = world(load_world=False)
    e = prop(w)
    w.rt.execute('dofile(MOCK.ROOT .. "/mentat/world.lua")')
    w.advance(0.1)
    assert L(w, "function(e) return MentatWorld.tracked[e] ~= nil and e.phys.drag == false end")(e)


def test_holodeck_props_are_tracked_and_the_object_parser_still_works():
    w = world()
    p = w.player(A, admin=True)
    w.ask(p, "add three couches")
    assert w.count(A) == 3
    w.advance(8)
    n = L(w, "(function() local n = 0 for e, rec in pairs(MentatWorld.tracked) do if e.HolodeckId then n = n + 1 end end return n end)()")
    assert n == 3


def test_a_tick_applies_the_real_force_to_a_moving_prop():
    w = world()
    speed, mass, side = 20.0, 12.0, 0.4
    e = prop(w, mass=mass, vel=(speed / HU_M, 0, 0), dims_hu=(side / HU_M,) * 3, model="models/props_junk/wood_crate001a.mdl")
    w.advance(0.01)
    w.rt.execute("MentatWorld.on_tick()")
    v = L(w, "function(e) return e.phys.vel.x end")(e) * HU_M
    rho, cd = atm.air_density(atm._T_STANDARD, atm._P_STANDARD), 1.05
    area = side * side                                                              # moving along its own x axis: one face
    k = drag_force(speed, cd, area, rho) / (mass * speed ** 2)                      # the library's force, per mass per speed squared
    assert v == pytest.approx(speed / (1 + k * speed * 0.015), rel=1e-9)
    assert v < speed


def test_a_light_thing_slows_much_more_than_a_heavy_one():
    w = world()
    light = prop(w, mass=0.005, vel=(0, 0, -15 / HU_M), dims_hu=(0.3 / HU_M, 0.2 / HU_M, 0.001 / HU_M))
    heavy = prop(w, mass=50.0, vel=(0, 0, -15 / HU_M), dims_hu=(0.3 / HU_M, 0.2 / HU_M, 0.001 / HU_M))
    w.advance(0.01)
    for _ in range(10):
        w.rt.execute("MentatWorld.on_tick()")
    vl = -L(w, "function(e) return e.phys.vel.z end")(light) * HU_M
    vh = -L(w, "function(e) return e.phys.vel.z end")(heavy) * HU_M
    assert 15 - vl > 100 * (15 - vh) * 0.9                                           # drag per unit mass: 10000 x, so the light one is far slower


def test_sleeping_bodies_are_never_touched_or_woken_by_the_drag_or_the_sweep():
    w = world()
    e = prop(w, vel=(5, 0, 0), asleep=True)
    w.advance(0.01)
    for _ in range(5):
        w.rt.execute("MentatWorld.on_tick()")
    w.advance(10)                                                                    # sweeps run
    assert L(w, "function(e) return e.phys.asleep and e.phys.adds == 0 and e.phys.wakes == 0 and e.phys.vel.x == 5 end")(e)


def test_resting_and_frozen_bodies_get_no_velocity_push():
    w = world()
    rest = prop(w)                                                                   # zero velocity, awake
    held = prop(w, vel=(500, 0, 0))
    L(w, "function(e) e.phys.motion = false end")(held)                              # frozen while it materialises
    w.advance(0.01)
    w.rt.execute("MentatWorld.on_tick()")
    assert L(w, "function(e) return e.phys.adds end")(rest) == 0
    assert L(w, "function(e) return e.phys.adds end")(held) == 0


def test_gravity_change_wakes_loose_things_but_a_repeat_does_not():
    w = world()
    admin = w.player(A, admin=True)
    e = prop(w, asleep=True)
    w.advance(0.01)
    w.ask(admin, "set gravity to the moon")
    assert L(w, "function(e) return e.phys.wakes end")(e) == 1
    L(w, "function(e) e.phys.asleep = true end")(e)
    w.ask(admin, "set gravity to the moon")
    assert L(w, "function(e) return e.phys.wakes end")(e) == 1


def test_turning_the_construct_off_gives_the_engine_back_its_own_gravity_and_drag():
    w = world()
    e = prop(w)
    w.advance(0.01)
    w.rt.execute("GetConVar('mentat_world_enabled'):SetValue(0)")
    w.advance(0.1)
    assert gravity_hu(w) == 600
    assert L(w, "function(e) return e.phys.drag end")(e) is True


def test_engine_drag_switch_leaves_the_engines_own_drag_on():
    w = world(cvars={"mentat_world_engine_drag": "1"})
    e = prop(w)
    w.advance(0.01)
    assert L(w, "function(e) return e.phys.drag end")(e) is True


def test_world_drop_measure_finds_the_first_impact_of_a_bouncing_ball():
    """A ball is kicked back up before it ever comes near its final resting height; the drop measurement must still find the first impact."""
    import world_drop
    dt, g, hu = 0.015, 514.78, 157.0
    z0, zr = 0.0, -hu
    s, v, z = [], 0.0, z0
    n = 0
    while True:
        n += 1
        v -= g * dt
        z += v * dt
        if z <= zr + 1.0:                       # touches down between ticks (inside the skin the tolerance misses), bounces
            s.append([n, zr + 0.9, 0.2 * -v])
            break
        s.append([n, z, v])
    for k in range(40):                          # the bounce and the slow settle
        n += 1
        s.append([n, zr + 1.0 + 3 * k * 0.1, 0.0])
    s[-1][1] = zr
    t, vi, h = world_drop.measure({"samples": s, "dt": dt, "z_start": z0, "z_rest": zr})
    assert t == pytest.approx(math.sqrt(2 * hu / g), rel=0.03)
    assert vi == pytest.approx(math.sqrt(2 * g * hu) * world_drop.HU_M, rel=0.03)
