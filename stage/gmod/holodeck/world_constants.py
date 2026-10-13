"""The construct's physics constants (SkippyVerse Phase A, task A1): one table, every value with its source.

    python stage/gmod/holodeck/world_constants.py            # regenerate both files below
    python stage/gmod/holodeck/world_constants.py --check    # exit 1 if they are stale

Writes
  stage/gmod/world_constants.json                                     the contract the Python side reads
  stage/gmod/addon/lua/mentat/world/constants_data.lua                the same table, for the game (generated: do not edit)

Real-world physics only (Aaron, 2026-10-12): gravity is Earth's (the library's G_EARTH) unless a real place is asked for (moon, mars,
jupiter, free fall), and air drag is the real F = 1/2 rho Cd A v^2 per object.  PLATINUM section 21: no physical number is typed here;
every one is read from the sigma-ground library at generation time (each entry's `source` names the function).  The only typed
numbers are the engine's own constants (Source's default gravity of 600 HU/s^2, its length unit, its air density: labelled `engine
default`) and the textbook drag coefficients (labelled `estimated, owner-should-confirm`, with citations).  Nothing here claims
thermodynamics, weather or fluids: the game does not model them.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GMOD = HERE.parent
JSON_PATH = GMOD / "world_constants.json"
LUA_PATH = GMOD / "addon" / "lua" / "mentat" / "world" / "constants_data.lua"
sys.path.insert(0, str(GMOD.parent.parent))          # the repo root, so `import sigma_ground` works from anywhere

VERSION = 2

# --- engine facts (Source / Garry's Mod); read back live by stage/gmod/addon/lua/mentat/tools/world_drop.lua
ENGINE_GRAVITY_HU_S2 = 600.0        # sv_gravity default in Garry's Mod
HU_M = 0.01905                      # 1 HU = 0.75 in; the stage scale (units.lua HU_PER_M = 1 / 0.01905; the tests pin them equal)
ENGINE_AIR_DENSITY = 2.0            # physenv.GetAirDensity() default (engine units; matters only while the engine's own drag is on)

CD_ESTIMATED = "estimated, owner-should-confirm"


def _body_g(name: str) -> tuple[float, str, str]:
    from sigma_ground.materia.bodies import resolve_body
    b = resolve_body(name)
    if b.key != name:
        raise RuntimeError(f"library has no body {name!r}")
    return (b.surface_gravity_m_s2, b.label,
            f"sigma_ground.materia.bodies.resolve_body('{name}').surface_gravity_m_s2 = G*M/R^2 "
            "(G from sigma_ground.field.constants; M and R are the library's IAU reference values)")


def _cd_table() -> dict:
    # The library has no Cd table (only C_N_FLAT_PLATE in dynamics.mechanisms.wind), so the handful of standard textbook values are
    # listed here, labelled for the owner to confirm.  Each depends on Reynolds number; these are the usual values for hand-to-table
    # sized objects moving through air at walking to falling speeds.
    from sigma_ground.dynamics.mechanisms.wind import C_N_FLAT_PLATE
    return {
        "sphere": {"value": 0.47, "status": CD_ESTIMATED,
                   "source": "White, Fluid Mechanics, Table 7.2 / Fig. 7.16 (smooth sphere, Re 1e3-2e5); Hoerner, Fluid-Dynamic Drag (1965) ch. 3"},
        "box": {"value": 1.05, "status": CD_ESTIMATED,
                "source": "Hoerner, Fluid-Dynamic Drag (1965) ch. 3 (cube, face on, Re > 1e4); used for every boxy object"},
        "flat_plate": {"value": C_N_FLAT_PLATE, "status": "library (flagged in the library as a standard value)",
                       "source": "sigma_ground.dynamics.mechanisms.wind.C_N_FLAT_PLATE (flat plate normal to the flow); used when the thinnest "
                                 "side is under a tenth of the next"},
        "cylinder": {"value": 1.20, "status": CD_ESTIMATED,
                     "source": "White, Fluid Mechanics, Table 7.2 (circular cylinder across the flow, L/D 5-20, Re 1e4-1e5); Hoerner ch. 3. "
                               "Projected area is the bounding box's, which overstates a cylinder's by 4/pi"},
    }


def build() -> dict:
    from sigma_ground.field.interface.projectile import G_EARTH, RHO_AIR
    from sigma_ground.field.interface import atmosphere as atm
    rho = atm.air_density(atm._T_STANDARD, atm._P_STANDARD)        # ISA sea level: 288.15 K, 101325 Pa
    engine_g_m_s2 = ENGINE_GRAVITY_HU_S2 * HU_M                    # the engine's own gravity in m/s^2 (a unit conversion only)

    def preset(label, g, source, status="library"):
        return {"label": label, "gravity_m_s2": g, "gravity_hu_s2": g / HU_M, "gravity_mult": g / engine_g_m_s2,
                "source": source, "status": status}

    presets = {
        "earth": preset("Earth", G_EARTH,
                        "sigma_ground.field.interface.projectile.G_EARTH = 9.80665 m/s^2, standard gravity (NIST); this is the default"),
        "zero": preset("zero gravity", 0.0, "free fall: zero by definition", "definition"),
    }
    for name in ("moon", "mars", "jupiter"):
        g, label, src = _body_g(name)
        presets[name] = preset(label, g, src)

    entries = {
        "air_density": {"value": rho, "units": "kg/m^3", "status": "library",
                        "source": "sigma_ground.field.interface.atmosphere.air_density(T=288.15 K, P=101325 Pa): ISA sea level, ideal gas "
                                  "(the library's own standard-atmosphere constants); cross-check projectile.RHO_AIR = %g" % RHO_AIR},
        "drag_law": {"value": "F = 1/2 rho Cd A v^2, opposing the velocity", "units": "", "status": "library",
                     "source": "sigma_ground.field.interface.projectile.drag_force (Prandtl 1904); applied to every free object each physics "
                               "tick, A = projected area of the object's real bounding box along its velocity, mass = the object's real mass"},
        "drag_coefficient": {"value": "by shape class", "units": "", "status": CD_ESTIMATED,
                             "source": "the cd table (sphere, box, flat plate, cylinder); there is no universal damping number"},
        "engine_air_drag": {"value": False, "units": "bool", "status": "construct decision",
                            "source": "Source's own air drag (PhysObj:EnableDrag, physenv air density) is switched off on these objects so the "
                                      "real force above is the only air drag; measured against it in the README 'The construct's physics'"},
        "time_scale": {"value": 1.0, "units": "x real time", "status": "engine default",
                       "source": "phys_timescale default; not changed unless the server owner sets mentat_world_time_scale"},
    }
    engine = {
        "gravity_hu_s2": {"value": ENGINE_GRAVITY_HU_S2, "units": "HU/s^2", "status": "engine default",
                          "source": "Garry's Mod sv_gravity default (read back live by the world_drop tool)"},
        "hu_m": {"value": HU_M, "units": "m per HU", "status": "engine default",
                 "source": "1 HU = 0.75 in = 0.01905 m (Source); the stage scale; same constant as mentat/units.lua HU_PER_M"},
        "gravity_m_s2": {"value": engine_g_m_s2, "units": "m/s^2", "status": "derived",
                         "source": "engine gravity * hu_m; %.4g times the library's Earth value (G_EARTH): the engine's default is stronger than real"
                                   % (engine_g_m_s2 / G_EARTH)},
        "air_density": {"value": ENGINE_AIR_DENSITY, "units": "engine units", "status": "engine default",
                        "source": "physenv.GetAirDensity() default; not changed"},
        "surfaces": {"value": "engine", "units": "", "status": "engine default",
                     "source": "friction and restitution come from the engine's surface properties (util.GetSurfaceData); Lua can only "
                               "assign an existing surface, and physenv.AddSurfaceData honours one new surface a session, so none is set"},
        "players": {"value": "engine", "units": "", "status": "engine default",
                    "source": "walking speed and jump are game-feel tuning, not physics: left as the engine has them; only gravity follows sv_gravity"},
    }
    limits = {"time_scale": [0.25, 2.0]}
    return {"version": VERSION, "hu_m": HU_M, "engine_gravity_hu_s2": ENGINE_GRAVITY_HU_S2, "default_preset": "earth",
            "engine": engine, "entries": entries, "presets": presets, "limits": limits, "cd": _cd_table(),
            "not_modelled": ["thermodynamics", "weather", "fluids", "electromagnetism", "chemistry"]}


def _lua(v, ind=0) -> str:
    pad = "  " * (ind + 1)
    if v is None:
        return "nil"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(float(v)) if isinstance(v, float) else str(v)
    if isinstance(v, str):
        if not v.isascii():
            raise ValueError("non-ASCII in a constants string (Lua 5.1 has no \\u escape): " + v)
        return json.dumps(v)
    if isinstance(v, list):
        return "{ " + ", ".join(_lua(x, ind) for x in v) + " }"
    items = [(k, x) for k, x in sorted(v.items()) if x is not None]
    body = "".join(f"{pad}{k if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', k) else '[' + json.dumps(k) + ']'} = {_lua(x, ind + 1)},\n"
                   for k, x in items)
    return "{\n" + body + "  " * ind + "}"


def render() -> tuple[str, str]:
    data = build()
    js = json.dumps(data, indent=1, sort_keys=True) + "\n"
    lua = ("-- GENERATED by stage/gmod/holodeck/world_constants.py from the sigma-ground library: DO NOT EDIT BY HAND.\n"
           "-- The construct's physics constants; every entry carries its units, source and status.\n"
           "return " + _lua(data) + "\n")
    return js, lua


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    js, lua = render()
    if a.check:
        ok = (JSON_PATH.exists() and LUA_PATH.exists() and JSON_PATH.read_text(encoding="utf-8") == js
              and LUA_PATH.read_text(encoding="utf-8") == lua)
        print("world constants are current" if ok else "world constants are STALE: run world_constants.py")
        return 0 if ok else 1
    LUA_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(js, encoding="utf-8", newline="\n")
    LUA_PATH.write_text(lua, encoding="utf-8", newline="\n")
    print(f"wrote {JSON_PATH.name} and {LUA_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
