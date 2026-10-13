"""The construct's water (SkippyVerse Phase A, task A2): one table, every value with its source.

    python stage/gmod/holodeck/water_constants.py            # regenerate both files below
    python stage/gmod/holodeck/water_constants.py --check    # exit 1 if they are stale

Writes
  stage/gmod/water_constants.json                                   the contract the Python side reads
  stage/gmod/addon/lua/mentat/water/data.lua                        the same table, for the game (generated: do not edit)

Real-world physics only (Aaron, 2026-10-12).  PLATINUM section 21: no physical number is typed here; each is read from the sigma-ground
library at generation time and carries its `source`.  What the library does NOT have is refused honestly in the game (olive oil, honey,
lava...), never guessed.  The only typed numbers are labelled `estimated, owner-should-confirm` (the human swimmer) or `construct cap`
(limits, which are not physics) or `visual` (drawing choices, which are not physics).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GMOD = HERE.parent
JSON_PATH = GMOD / "water_constants.json"
LUA_PATH = GMOD / "addon" / "lua" / "mentat" / "water" / "data.lua"
sys.path.insert(0, str(GMOD.parent.parent))
sys.path.insert(0, str(HERE))

from world_constants import _lua, HU_M  # noqa: E402  (the same Lua writer and the same stage scale as the world constants)

VERSION = 1

# --- words -> library liquid keys.  A word the library has data for maps to its key; the REFUSED words are liquids a player may
# ask for that the library has no data for (they are refused in the game with the reason, never guessed).
LIQUID_WORDS = {
    "water": ["water", "h2o", "freshwater"],
    "seawater": ["seawater", "saltwater", "brine", "salt water", "sea water", "salty"],
    "ethanol": ["ethanol", "alcohol", "vodka"],
    "glycerol": ["glycerol", "glycerin", "glycerine"],
    "mercury": ["mercury", "quicksilver"],
}
REFUSED_WORDS = ["oil", "olive oil", "honey", "milk", "lava", "blood", "petrol", "gasoline", "acid", "beer", "wine", "juice", "syrup",
                 "slime", "tar", "paint", "cola", "soda", "coffee", "tea", "soup", "diesel", "kerosene", "nitrogen", "helium", "ice"]

LABELS = {"water": "water", "seawater": "seawater", "ethanol": "ethanol", "glycerol": "glycerol", "mercury": "mercury"}

# --- drawing-only choices (not physics): tile colour of the basin
VISUAL = {
    "tile_rgb": [205, 235, 240],
    "sky_rgb": [150, 190, 235],
    "foam_rgb": [235, 245, 250],
}

HUMAN_STATUS = "estimated, owner-should-confirm"


def build() -> dict:
    from sigma_ground.field.interface import fluid, liquid_water, optics
    from sigma_ground.field.interface.projectile import G_EARTH

    liquids = {}
    for key, words in LIQUID_WORDS.items():
        p = fluid.liquid_properties(key)
        liquids[key] = {
            "label": LABELS[key],
            "words": words,
            "density_kg_m3": p["density_kg_m3"],
            "viscosity_pa_s": p["viscosity_pa_s"],
            "surface_tension_n_m": p["surface_tension_n_m"],
            "T_K": p["T_K"],
            "status": "library (MEASURED)",
            "source": (f"sigma_ground.field.interface.fluid.KNOWN_LIQUIDS['{key}'] via liquid_properties('{key}') at {p['T_K']:.2f} K, 1 atm: "
                       "density, dynamic viscosity (Arrhenius table; water IAPWS 2008, seawater Sharqawy 2010, ethanol/glycerol CRC, mercury "
                       "Filled et al. 1954), surface tension"),
        }
    # cross-check for the report: the library's own Kell correlation for water at the same temperature
    water_kell = liquid_water.water_density(liquids["water"]["T_K"])

    n_water = optics.cauchy_n("water", 550e-9)
    r0 = optics.dielectric_surface_reflectance(n_water)
    absorb = list(optics.DIELECTRIC_ABSORPTION["water_blue"])

    data = {
        "version": VERSION,
        "hu_m": HU_M,
        "liquids": liquids,
        "liquid_order": ["water", "seawater", "ethanol", "glycerol", "mercury"],
        "refused_words": REFUSED_WORDS,
        "refused_reason": ("the sigma-ground library has no density and viscosity for it, and I do not guess a liquid's properties; "
                           "I have water, seawater, ethanol, glycerol and mercury"),
        "cross_check": {
            "water_density_kell_kg_m3": water_kell,
            "source": "sigma_ground.field.interface.liquid_water.water_density(T) (Kell 1975) against KNOWN_LIQUIDS['water']",
        },
        "g_earth": {"value": G_EARTH, "units": "m/s^2", "status": "library",
                    "source": "sigma_ground.field.interface.projectile.G_EARTH (the construct's own g comes from mentat/world/constants.lua)"},
        "optics": {
            "refractive_index": {"value": n_water, "status": "library (MEASURED)",
                                 "source": "sigma_ground.field.interface.optics.cauchy_n('water', 550e-9) (Cauchy fit, Schott/Palik)"},
            "fresnel_r0": {"value": r0, "status": "library (FIRST_PRINCIPLES)",
                           "source": "sigma_ground.field.interface.optics.dielectric_surface_reflectance(n): ((n-1)/(n+1))^2, normal incidence"},
            "absorption_rgb_per_m": {"value": absorb, "status": "library (MEASURED)",
                                     "source": "sigma_ground.field.interface.optics.DIELECTRIC_ABSORPTION['water_blue'] (m^-1 at R, G, B)"},
        },
        "human": {
            "density_kg_m3": {"value": 985.0, "status": HUMAN_STATUS,
                              "source": "the library has no body density; whole-body density of a person with partly inflated lungs is about "
                                        "0.98-1.01 g/cm3 (Brozek/Siri body-composition literature); 985 is chosen so a person floats with the head "
                                        "just clear, which is the common experience"},
            "mass_kg": {"value": 80.0, "status": HUMAN_STATUS,
                        "source": "an adult; the engine's player mass is not used (swimmers are not rigid bodies here)"},
            "swim_thrust_n": {"value": 150.0, "status": HUMAN_STATUS,
                              "source": "sustained propulsive force of a recreational swimmer (order 100-200 N); with the hull's drag this gives about "
                                        "0.6 m/s horizontal, which is slow for a swimmer: the hull's box area overstates a body's"},
            "hull_hu": {"value": [32.0, 32.0, 72.0], "status": "engine default",
                        "source": "Source's standing player hull: 32 x 32 x 72 HU (0.61 x 0.61 x 1.37 m)"},
            "reach": {"value": "the hull's height", "status": "engine default",
                      "source": "a swimmer pulls themselves out at an edge when its top is no higher than their hands can reach overhead: "
                                "the standing hull is 72 HU = 1.37 m; the climb itself is an affordance, not a force model"},
        },
        "waves": {
            "steepness": {"value": 0.05, "status": "visual",
                          "source": "wave steepness a*k of each drawn component; the drawn waves are gentle: far below the Stokes limit "
                                    "a*k = 0.44 (H/lambda = 0.142, Michell 1893), so linear theory (the dispersion relation) is valid"},
            "dispersion": {"value": "omega^2 = (g k + sigma k^3 / rho) tanh(k d)", "status": "library (FIRST_PRINCIPLES)",
                           "source": "linear capillary-gravity waves (Kelvin 1871): deep water omega^2 = g k, shallow water c = sqrt(g d); "
                                     "g from the construct's world constants, sigma and rho from the liquid's own row"},
        },
        "radiation": {
            "heave_damping": {"value": "b = 1/2 rho omega k A_w^2 exp(-2 k T)", "status": "estimated, owner-should-confirm",
                              "source": "wave-radiation damping of a floating body moving up and down, in the small-body limit (derived from the radiated "
                                        "power of a point source at the surface, Newman 1977 ch. 6; omega the body's own heave frequency "
                                        "sqrt(rho g A_w / m), k = omega^2 / g, A_w its waterplane area, T its draft). Without it a floater would ring "
                                        "for half a minute; with it, about three swings. Not in the library; the factor 1/2 should be confirmed. "
                                        "Heave only, deep-water radiation: a pool's walls reflect the real waves, this does not"},
        },
        "splash": {
            "scale_law": {"value": "l = (E / (rho g))^(1/4)", "status": "estimated, owner-should-confirm",
                          "source": "gravity-regime cavity scaling for an impact of kinetic energy E in a liquid (crater radius ~ (E/rho g)^(1/4); "
                                    "Engel 1966, Gault); the prefactor 1 is an estimate. It sizes the DRAWN splash and ripple only: no fluid is simulated"},
            "max_particles": {"value": 48, "status": "construct cap", "source": "drawing budget, not physics"},
        },
        "limits": {
            "max_bodies": {"value": 6, "status": "construct cap", "source": "water bodies at once on a server; a cap, not physics"},
            "max_volume_m3": {"value": 150000.0, "status": "construct cap", "source": "total water volume at once; a cap, not physics"},
            "max_footprint_m": {"value": 200.0, "status": "construct cap", "source": "longest side or diameter; a cap, not physics (the engine's world is about 16000 HU = 300 m across)"},
            "min_footprint_m": {"value": 0.5, "status": "construct cap", "source": "shortest side or diameter; a cap, not physics"},
            "max_depth_m": {"value": 20.0, "status": "construct cap", "source": "depth; a cap, not physics"},
            "min_depth_m": {"value": 0.1, "status": "construct cap", "source": "depth; a cap, not physics"},
            "max_bodies_per_tick": {"value": 48, "status": "construct cap", "source": "objects processed per physics tick across all water; a cap, not physics"},
            "wall_m": {"value": 0.2, "status": "construct cap", "source": "thickness of the basin's walls and floor (collision shell)"},
            "freeboard_m": {"value": 0.05, "status": "construct cap", "source": "the wall stands this far above the water so it does not spill"},
        },
        "defaults": {
            "pool": {"shape": "rect", "w_m": 5.0, "l_m": 10.0, "depth_m": 2.0, "status": "construct default",
                     "source": "Aaron's default for a pool: 5 m x 10 m x 2 m"},
            "pond": {"shape": "round", "diameter_m": 6.0, "depth_m": 1.5, "status": "construct default", "source": "a garden pond; a default, not physics"},
            "tank": {"shape": "round", "diameter_m": 3.0, "depth_m": 2.0, "status": "construct default", "source": "a round tank; a default, not physics"},
            "lake": {"shape": "rect", "w_m": 50.0, "l_m": 50.0, "depth_m": 3.0, "status": "construct default",
                     "source": "a lake-sized basin; a default, not physics. It is a basin on the floor, not a shore"},
            "sea": {"shape": "rect", "w_m": 150.0, "l_m": 150.0, "depth_m": 5.0, "status": "construct default",
                    "source": "a sea-sized basin; a default, not physics. It is a basin on the floor, not an ocean"},
        },
        "visual": VISUAL,
        "not_modelled": ["thermodynamics", "evaporation", "freezing", "particle fluid", "added mass", "surface tension on bodies",
                         "waves from wind", "currents", "drowning"],
    }
    return data


def render() -> tuple[str, str]:
    data = build()
    js = json.dumps(data, indent=1, sort_keys=True, ensure_ascii=True) + "\n"
    lua = ("-- GENERATED by stage/gmod/holodeck/water_constants.py from the sigma-ground library: DO NOT EDIT BY HAND.\n"
           "-- The construct's water; every entry carries its source and status.\n"
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
        print("water constants are current" if ok else "water constants are STALE: run water_constants.py")
        return 0 if ok else 1
    LUA_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(js, encoding="utf-8", newline="\n")
    LUA_PATH.write_text(lua, encoding="utf-8", newline="\n")
    print(f"wrote {JSON_PATH.name} and {LUA_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
