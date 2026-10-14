"""Router v3.1 recipes: the parameters the model has never been shown (outside math).

tools/tool_census.py coverage found 99 of 545 parameters with under 10 training examples. The math tools
are in router_math.py. These recipes cover most of the rest where a natural question exists:

  keys     the SECOND material of a two-material tool (friction, wear, thermal contact, acoustic
           interface, wetting, natural convection, alloy resistivity, thermoelectric legs) and the
           isotope of the activity tool: the model must turn "mild steel" into the key "steel_mild"
  colligative   `van_t_hoff_i` of the three colligative tools (the question names the salt or the factor)
  numeric  `measured_mass_u` of the binding energy, `image_side` of the lens magnification, and the
           mass-and-radius forms of the orbital tools (`central_mass_kg`, `from_radius_m`, `to_radius_m`)

Same rules as every recipe: each plan is run on the real server; a result with empty fields (a tool that
could not do the job, such as a key it does not know) is dropped; the benchmark guard drops benchmark
instances. Not covered, on purpose: interface tools (`playground_*`, `request_clarification`, `simulate`,
`parse_quantity`, `list_constants`), the five guard-blocked no-input tools, and the niche quantum-algorithm
and sigma-model parameters. misc/ROUTER_V31_PLAN.md lists what is left.

    python -m sigma_ground.mcp.benchmark.router_recipes_v31 --per-recipe 60 --out local-cache/router_params_v31
"""
from __future__ import annotations

from sigma_ground.field.constants import AU_M, M_SUN_KG
from sigma_ground.field.interface.friction import MATERIALS as _MATERIALS
from sigma_ground.mcp.benchmark import router_recipes as RR
from sigma_ground.mcp.benchmark.router_recipes import C, K, Q, S, U, single
from sigma_ground.mcp.benchmark.router_recipes import KG_PLANET, KG_SAT

V31_START = len(RR.RECIPES)

NATURAL = {"steel_mild": "mild steel", "plastic_abs": "ABS plastic", "ceramic_alumina": "alumina ceramic", "carbon_fiber": "carbon fiber",
           "depleted_uranium": "depleted uranium", "water_ice": "ice", "wood_oak": "oak", "aluminum": "aluminum"}


def natural(key: str) -> str:
    return NATURAL.get(key, key.replace("_", " "))


def keys(names):
    return C(tuple((natural(k), k) for k in names))


MATS = sorted(_MATERIALS)
HARD = [m for m in MATS if m not in ("bone", "concrete", "granite", "water_ice", "wood_oak", "rubber", "kevlar", "carbon_fiber", "plastic_abs")]
METALS = ["aluminum", "copper", "iron", "gold", "silver", "nickel", "steel_mild", "titanium", "lead", "tungsten", "zinc", "platinum"]
NEWTON = U(1, 5000, ("N", 1), ("kN", 1e3))
PRESSURE = U(1e4, 5e8, ("Pa", 1), ("kPa", 1e3), ("MPa", 1e6))
TEMP_HOT = U(320, 1200, ("K", 1))
TEMP_LOW = U(250, 310, ("K", 1))
LENGTH_SMALL = U(0.005, 2, ("m", 1), ("cm", 1e-2))

# ── two materials ────────────────────────────────────────────────────────────────────────

single("friction_pair", "keys", {"a": keys(MATS), "b": keys(MATS), "F": Q(NEWTON)}, [
    "How much sliding friction is there between {a} and {b} pressed together with {F}?",
    "What is the friction when {a} slides on {b} under a {F} load?",
    "Find the friction force of {a} against {b} with a normal force of {F}.",
    "{a} rubbing on {b}, pressed with {F}: what friction do we get?",
], "friction_analysis", {"material_key_1": S("a"), "material_key_2": S("b"), "normal_force_n": S("F")}, per=70)

single("friction_pair_default_load", "keys", {"a": keys(MATS), "b": keys(MATS)}, [
    "What is the friction between {a} and {b}?",
    "How does {a} slide against {b}? Give the sliding friction.",
    "Dry friction of {a} on {b}.",
], "friction_analysis", {"material_key_1": S("a"), "material_key_2": S("b")}, per=40)

single("wear_pair", "keys", {"a": keys(MATS), "b": keys(HARD), "F": Q(NEWTON), "d": Q(U(1, 5e4, ("m", 1), ("km", 1e3)))}, [
    "A {a} pin slides {d} against {b} under a {F} load. How much does it wear?",
    "Sliding wear of {a} on a {b} counterface: {F} normal load, {d} sliding distance.",
    "How much material does {a} lose rubbing against {b} for {d} with {F} pressing?",
], "wear_analysis", {"material_key": S("a"), "counter_material": S("b"), "normal_force_n": S("F"), "sliding_distance_m": S("d")}, per=60)

single("thermal_contact_pair", "keys", {"a": keys(METALS), "b": keys(METALS), "P": Q(PRESSURE)}, [
    "How well does heat cross the joint where {a} is pressed against {b} at {P}?",
    "Thermal contact conductance between {a} and {b} under {P} of pressure?",
    "Two metal blocks, {a} and {b}, are clamped together at {P}. What is the joint conductance?",
], "thermal_contact_analysis", {"material_1": S("a"), "material_2": S("b"), "pressure_pa": S("P")}, per=60)

single("acoustic_pair", "keys", {"a": keys(MATS), "b": keys(MATS), "t": Q(U(0.5, 60, ("degrees", 1), log=False))}, [
    "Sound goes from {a} into {b} at {t} incidence. How much is reflected and transmitted?",
    "What happens to a sound wave crossing from {a} to {b} at an angle of {t}?",
    "Acoustic reflection and transmission at a {a}/{b} boundary, incidence angle {t}.",
], "acoustic_interface_analysis", {"material_key_1": S("a"), "material_key_2": S("b"), "incidence_angle_deg": S("t")}, per=60)

SOLIDS = C((("glass", "glass"), ("gold", "gold"), ("paraffin wax", "paraffin"), ("PMMA", "pmma"), ("polyethylene", "polyethylene"), ("PTFE", "ptfe"),
            ("silicon", "silicon"), ("steel", "steel")))
LIQUIDS = C((("water", "water"), ("ethanol", "ethanol"), ("glycerol", "glycerol"), ("ethylene glycol", "ethylene_glycol"), ("diiodomethane", "diiodomethane")))
single("wetting_pair", "keys", {"s": SOLIDS, "l": LIQUIDS}, [
    "Does {l} wet {s}? What is the contact angle?",
    "What contact angle does a drop of {l} make on {s}?",
    "Wetting of {s} by {l}: contact angle and work of adhesion?",
    "A drop of {l} sits on {s}. How well does it spread?",
], "wetting_analysis", {"solid_key": S("s"), "liquid_key": S("l")}, per=40)

GASES = C((("methane", "CH4"), ("carbon monoxide", "CO"), ("carbon dioxide", "CO2"), ("water vapor", "H2O"), ("nitrogen", "N2"), ("oxygen", "O2")))
single("convection_gases", "keys", {"hot": Q(TEMP_HOT), "amb": Q(TEMP_LOW), "L": Q(LENGTH_SMALL), "g1": GASES, "g2": GASES}, [
    "Natural convection in {g1} next to a surface at {hot}, ambient {amb}, over a length of {L}. Also the diffusivity of {g1} in {g2}.",
    "A {L} surface at {hot} heats {g1} at {amb}. What convection do we get, and how fast does {g2} diffuse through {g1}?",
    "Buoyancy-driven flow of {g1}: hot wall {hot}, room {amb}, length {L}; binary diffusion with {g2}.",
], "natural_convection_analysis", {"hot_temperature_k": S("hot"), "ambient_temperature_k": S("amb"), "length_m": S("L"), "gas_key": S("g1"), "gas_key_2": S("g2")}, per=40)

ISOTOPES = C((("uranium-238", "U238"), ("radium-226", "Ra226"), ("polonium-210", "Po210"), ("carbon-14", "C14"), ("cobalt-60", "Co60"), ("potassium-40", "K40"),
              ("free neutrons", "free_neutron")))
single("radioactivity_isotope", "keys", {"iso": ISOTOPES, "N": Q(U(1e6, 1e25, ("atoms", 1)))}, [
    "What is the activity of {N} of {iso}?",
    "How radioactive are {N} of {iso}? Give the activity in becquerel and curies.",
    "A sample holds {N} of {iso}. What is its decay activity?",
    "Radioactivity of {N} {iso}?",
], "radioactivity_analysis", {"isotope_key": S("iso"), "n_atoms": S("N")}, per=50)

ALLOY = C((("copper", "copper"), ("nickel", "nickel"), ("gold", "gold"), ("silver", "silver"), ("zinc", "zinc"), ("aluminum", "aluminum"), ("iron", "iron")))
single("alloy_pair", "keys", {"a": ALLOY, "b": ALLOY, "f": Q(U(0.02, 0.9, ("", 1), log=False))}, [
    "What is the residual resistivity of a {a}-{b} alloy with a {b} fraction of {f}?",
    "Nordheim residual resistivity for {a} with {f} of {b} mixed in?",
    "A binary alloy of {a} and {b}, {f} {b}. What extra resistivity does the mixing add?",
], "alloy_resistivity_analysis", {"metal_a": S("a"), "metal_b": S("b"), "fraction_b": S("f")}, per=40)

# ── colligative properties: the van 't Hoff factor ───────────────────────────────────────

SALT = C((("sodium chloride", 2), ("potassium chloride", 2), ("calcium chloride", 3), ("magnesium sulfate", 2), ("aluminum chloride", 4),
          ("sodium sulfate", 3), ("potassium nitrate", 2), ("iron(III) chloride", 4)))
FACTOR = C(tuple((str(x), x) for x in (1.5, 2, 2.5, 3, 4)))
MOLAL = U(0.01, 5, ("mol/kg", 1), ("molal", 1))
MOLAR = U(0.01, 3, ("mol/L", 1), ("M", 1))
COLLIGATIVE = (
    ("boiling_point_elevation", "m", "molality_mol_kg", MOLAL,
     ["How much does dissolving {c} of {salt} raise the boiling point of water?", "Boiling-point elevation for a {c} {salt} solution.",
      "By how many kelvin does {c} of {salt} elevate the boiling point of water?"],
     ["What is the boiling-point elevation of a {c} solution with a van 't Hoff factor of {i}?",
      "A solution of {c} with i = {i}: how much is its boiling point raised?"]),
    ("freezing_point_depression", "m", "molality_mol_kg", MOLAL,
     ["How far does {c} of {salt} lower the freezing point of water?", "Freezing-point depression of a {c} {salt} solution.",
      "A {c} solution of {salt}: by how much is the freezing point depressed?"],
     ["What is the freezing-point depression of a {c} solution with a van 't Hoff factor of {i}?",
      "A solution of {c} with i = {i}: how far is its freezing point lowered?"]),
    ("osmotic_pressure", "c", "molarity_mol_l", MOLAR,
     ["What osmotic pressure does {c} of {salt} produce?", "Osmotic pressure of a {c} {salt} solution.", "A {c} solution of {salt}: how large is the osmotic pressure?"],
     ["What is the osmotic pressure of a {c} solution with a van 't Hoff factor of {i}?", "A solution of {c} with i = {i}: what osmotic pressure does it give?"]),
)
for tool, _slot, param, unit, by_salt, by_factor in COLLIGATIVE:
    single(f"colligative_{tool}", "colligative", {"c": Q(unit), "salt": SALT}, by_salt, tool,
           {param: S("c"), "van_t_hoff_i": S("salt")}, per=45)
    single(f"colligative_{tool}_explicit_i", "colligative", {"c": Q(unit), "i": FACTOR}, by_factor, tool,
           {param: S("c"), "van_t_hoff_i": S("i")}, per=30)

# ── numeric extras ───────────────────────────────────────────────────────────────────────

NUCLIDE = C((("helium-4 (measured mass 4.0026 u)", (2, 2, 4.0026)), ("carbon-12 (measured mass 12.0000 u)", (6, 6, 12.0)),
             ("oxygen-16 (measured mass 15.9949 u)", (8, 8, 15.9949)), ("iron-56 (measured mass 55.9349 u)", (26, 30, 55.9349)),
             ("uranium-235 (measured mass 235.0439 u)", (92, 143, 235.0439)), ("deuterium (measured mass 2.0141 u)", (1, 1, 2.0141)),
             ("lithium-7 (measured mass 7.0160 u)", (3, 4, 7.0160)), ("neon-20 (measured mass 19.9924 u)", (10, 10, 19.9924)),
             ("helium-4 (measured mass 4.00260 u)", (2, 2, 4.0026)), ("calcium-40 (measured mass 39.9626 u)", (20, 20, 39.9626))))
single("binding_measured_mass", "numeric", {"nuc": NUCLIDE}, [
    "What is the binding energy of {nuc}?",
    "Find the nuclear binding energy of {nuc}.",
    "How much energy binds the nucleus of {nuc}? Use the measured mass given.",
], "nuclear_binding_energy", {"protons": K("nuc", 0), "neutrons": K("nuc", 1), "measured_mass_u": K("nuc", 2)}, per=24)

SIDE = C((("the image forms behind the lens", "behind"), ("the image is real", "real"), ("the image is virtual", "virtual"),
          ("the image is on the same side of the lens as the object", "front")))
single("lens_image_side", "numeric", {"o": Q(U(0.02, 2, ("m", 1), ("cm", 1e-2))), "i": Q(U(0.02, 2, ("m", 1), ("cm", 1e-2))), "side": SIDE}, [
    "An object is {o} from a lens and the image is {i} away; {side}. What is the magnification?",
    "Lens magnification: object distance {o}, image distance {i}, {side}.",
    "{side}; the object is {o} away and the image is {i} from the lens. How magnified is it?",
], "lens_magnification", {"object_distance_m": S("o"), "image_distance_m": S("i"), "image_side": S("side")}, per=60)

CENTRAL = U(0.1 * M_SUN_KG, 30 * M_SUN_KG, ("solar masses", M_SUN_KG))
AU_RANGE = U(0.05, 60, ("AU", AU_M), ("million km", 1e9))
single("orbital_period_central_mass", "numeric", {"M": Q(CENTRAL), "a": Q(AU_RANGE)}, [
    "A planet orbits a star of {M} at {a}. What is its orbital period?",
    "Kepler's third law for a body {a} from an object of {M}: how long is its orbit?",
    "How long does it take to circle a {M} star at a distance of {a}?",
], "orbital_period", {"semimajor_axis_m": S("a"), "central_mass_kg": S("M")}, per=60)

R_ORBIT = U(7e6, 5e9, ("km", 1e3), ("m", 1))
single("raise_orbit_radii", "numeric", {"m": Q(KG_SAT), "M": Q(KG_PLANET), "r1": Q(U(7e6, 2e7, ("km", 1e3))), "r2": Q(U(2.1e7, 8e7, ("km", 1e3)))}, [
    "How much energy does it take to lift a {m} satellite from a {r1} orbit radius to a {r2} radius around a body of {M}?",
    "Raise a {m} spacecraft from radius {r1} to radius {r2} about a {M} planet: energy needed?",
    "A {M} world; a {m} probe goes from {r1} to {r2} from its centre. What gravitational energy must be supplied?",
], "orbital_raise_energy", {"mass_kg": S("m"), "central_mass_kg": S("M"), "from_radius_m": S("r1"), "to_radius_m": S("r2")}, per=60)

# ── the materials database as empirical properties ──────────────────────────────────────
# The database (list_materials: densities, refractive indices, melting and boiling points, Young's moduli, band
# gaps) is the source of measured material properties. A plan REFERENCES it: look the property up, pass it on
# ("$1"). The model never writes a refractive index, a melting point or a density from memory.

def db_chain(id, slots, templates, steps, per=40):
    return RR.add(id, "db", slots, templates, steps, {}, per)


def pair(*options):
    return C(tuple((f"{a_text} {joiner} {b_text}", (a, b)) for a_text, joiner, b_text, a, b in options))


MEDIA_PAIRS = pair(("water", "into", "air", "water", "air"), ("glass", "into", "air", "glass", "air"), ("diamond", "into", "air", "diamond", "air"),
                   ("ice", "into", "air", "ice", "air"), ("glass", "into", "water", "glass", "water"), ("sapphire", "into", "air", "sapphire", "air"),
                   ("quartz", "into", "air", "quartz", "air"), ("flint glass", "into", "air", "glass_flint", "air"), ("diamond", "into", "water", "diamond", "water"),
                   ("acrylic (PMMA)", "into", "air", "pmma", "air"), ("polystyrene", "into", "air", "polystyrene", "air"), ("sapphire", "into", "water", "sapphire", "water"),
                   ("silicon", "into", "air", "silicon", "air"), ("ethanol", "into", "air", "ethanol", "air"), ("quartz", "into", "water", "quartz", "water"))
REFRACT = [("refractive_index", {"material": RR.K("pair", 0)}), ("refractive_index", {"material": RR.K("pair", 1)})]

db_chain("db_critical_angle", {"pair": MEDIA_PAIRS}, [
    "Light travels from {pair}. Beyond what angle is it totally internally reflected?",
    "What is the critical angle for light going from {pair}?",
    "At what angle does total internal reflection begin for light passing from {pair}?",
    "Find the critical angle at the boundary when light goes from {pair}.",
    "Total internal reflection: light from {pair}. What is the critical angle?",
], REFRACT + [("critical_angle_for_tir", {"n_dense": RR.R(1), "n_rare": RR.R(2)})], per=60)

db_chain("db_snell_between_media", {"pair": MEDIA_PAIRS, "t": Q(U(1, 45, ("degrees", 1), log=False))}, [
    "Light passes from {pair} at an incident angle of {t}. What is the refraction angle?",
    "A ray goes from {pair}, hitting the surface at {t}. At what angle does it continue?",
    "Snell's law: from {pair}, angle of incidence {t}. Find the refracted angle.",
], REFRACT + [("snells_law_refraction_angle", {"n1": RR.R(1), "n2": RR.R(2), "incident_angle_deg": S("t")})], per=60)

MEDIUM = C(tuple((natural(k), k) for k in ("water", "glass", "diamond", "ice", "sapphire", "quartz", "pmma", "polystyrene", "ethanol")))
db_chain("db_wavelength_in_medium", {"med": MEDIUM, "f": Q(U(1e12, 1e16, ("Hz", 1), ("THz", 1e12)))}, [
    "Light of frequency {f} enters {med}. What is its wavelength there?",
    "What is the wavelength inside {med} of a {f} wave?",
    "A {f} light wave travels through {med}. Find its wavelength in the medium.",
], [("refractive_index", {"material": S("med")}),
    ("em_wave_wavelength", {"frequency_hz": S("f"), "refractive_index": RR.R(1)})], per=50)

HOT = C(tuple((natural(k), k) for k in ("aluminum", "copper", "gold", "iron", "lead", "silver", "tungsten", "platinum", "silicon")))
db_chain("db_peak_at_melting_point", {"m": HOT}, [
    "At what wavelength does {m} radiate most strongly at its melting point?",
    "Peak blackbody wavelength of {m} glowing at its melting temperature?",
    "A piece of {m} is heated to its melting point. Where does its thermal emission peak?",
], [("melting_point", {"material": S("m")}), ("blackbody_peak_wavelength", {"temperature_k": RR.R(1)})], per=40)

BOILER = C(tuple((natural(k), k) for k in ("water", "ethanol", "nitrogen", "oxygen", "mercury", "iron", "copper", "lead", "helium", "hydrogen", "argon")))
db_chain("db_thermal_energy_at_boiling_point", {"m": BOILER}, [
    "What is the average thermal energy per molecule of {m} at its boiling point?",
    "How much thermal energy does each molecule have in {m} at its boiling point?",
    "Mean thermal energy per molecule when {m} boils?",
], [("boiling_point", {"material": S("m")}), ("thermal_energy_per_molecule", {"temperature_k": RR.R(1)})], per=40)

ENGINE = pair(("water", "(hot) and", "nitrogen (cold)", "water", "nitrogen"), ("water", "(hot) and", "oxygen (cold)", "water", "oxygen"),
              ("iron", "(hot) and", "water (cold)", "iron", "water"), ("copper", "(hot) and", "water (cold)", "copper", "water"),
              ("lead", "(hot) and", "water (cold)", "lead", "water"), ("mercury", "(hot) and", "nitrogen (cold)", "mercury", "nitrogen"),
              ("ethanol", "(hot) and", "nitrogen (cold)", "ethanol", "nitrogen"), ("gold", "(hot) and", "mercury (cold)", "gold", "mercury"))
db_chain("db_carnot_between_boiling_points", {"pair": ENGINE}, [
    "What is the maximum efficiency of a heat engine running between the boiling points of {pair}?",
    "Carnot efficiency of an engine whose hot side is at the boiling point of {pair}.",
    "A heat engine uses the boiling points of {pair} as its reservoirs. What is the Carnot limit?",
], [("boiling_point", {"material": RR.K("pair", 0)}), ("boiling_point", {"material": RR.K("pair", 1)}),
    ("carnot_efficiency", {"t_hot_k": RR.R(1), "t_cold_k": RR.R(2)})], per=40)

OBJ = C(tuple((natural(k), k) for k in ("iron", "aluminum", "copper", "lead", "gold", "glass", "concrete", "rubber", "wood_oak", "water_ice", "titanium")))
FLUID = C(tuple((natural(k), k) for k in ("water", "ethanol", "mercury", "honey", "milk", "oil_olive", "gasoline")))
db_chain("db_float_in_fluid", {"obj": OBJ, "fluid": FLUID}, [
    "Will a block of {obj} float in {fluid}?",
    "Does {obj} sink or float when placed in {fluid}? How deep does it sit?",
    "Buoyancy of {obj} in {fluid}: floats or sinks?",
], [("density", {"material": S("fluid")}), ("buoyancy_analysis", {"material_key": S("obj"), "fluid_density_kg_m3": RR.R(1)})], per=50)

V31_RECIPES = RR.RECIPES[V31_START:]


def main() -> int:
    import argparse
    import asyncio
    from pathlib import Path
    from sigma_ground.mcp.benchmark import router_dataset as D
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=D.REPO / "local-cache" / "router_params_v31")
    p.add_argument("--per-recipe", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-guard", action="store_true")
    p.add_argument("--server-mem-gb", type=float, default=6)
    a = p.parse_args()
    a.only = ",".join(r.id for r in V31_RECIPES)
    return asyncio.run(RR._amain(a))


if __name__ == "__main__":
    import sys
    sys.exit(main())
