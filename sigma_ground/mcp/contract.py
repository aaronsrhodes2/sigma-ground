"""Mentat's tool contract: what every tool promises its caller, enforced in one place.

Audit 2026-10-13 (misc/MENTAT_AUDIT_2026-10-13.md, F4/F9/F11): no tool accepted a quantity with units ("54 km/h" was a
pydantic error), 471 of 545 parameters had no description, and a failure came back as pydantic prose, as ``value: null``
with the reason in ``notes``, or as an exception. This module is the contract that closes those, applied to all ~230
tools at once by :func:`install` (called from ``server.main()``):

* **Quantities with units.** A numeric parameter whose spec carries a unit accepts a number (in the parameter's own unit,
  as before) or a string with units: ``"54 km/h"``, ``"150 cm"``, ``"27 degC"``, ``"0.5 c"``. It is converted with pint;
  the conversion is echoed in ``converted``. A string of the wrong dimension is refused, naming what was expected.
* **Documented parameters.** Every parameter gets a description in the tool's schema (unit, meaning, bounds, an example),
  and the tool's description gets an Args section and an example call.
* **Uniform errors.** Whatever goes wrong comes back as a normal result with ``value: null`` and
  ``error: {code, message, param?, expected?, got?, example?}``: ``missing_argument``, ``bad_type``, ``bad_quantity``,
  ``wrong_dimension``, ``out_of_range`` (definitional impossibilities only: T < 0 K, an emissivity of 1.2),
  ``unknown_key`` (a table lookup that missed), ``declined`` (the tool's own refusal). Never a stack trace.
* **No silent ignoring.** An argument the tool does not have is reported in ``ignored_arguments`` (with a "did you mean").
* **Bounds** use the four words of GOLDEN_RULES Rule 2 (``field/bounds.py``): only BEYOND is refused here; the validity
  edges of a model stay the model's business.

Parameter specs live in ``_SPEC_TABLE`` below, one line each, reviewed like data. Nothing numeric in it is a measured
value: units, meanings, definitional bounds, and examples only. Renames (``material_key_1`` -> ``material_a`` ...) are
listed in ``misc/CONTRACT_RENAMES.md`` and are NOT applied: the router's training labels use today's names.
"""
from __future__ import annotations

import difflib
import functools
import inspect
import json
import math
import re
from dataclasses import dataclass
from typing import Annotated, Any, Optional, Union

from pydantic import Field

from sigma_ground.field.bounds import Bounds, Regime, classify, describe
from sigma_ground.mcp import param_names


# ── the specs ──────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Spec:
    unit: str | None               # pint unit the parameter is expressed in; None = dimensionless / not a physical quantity
    meaning: str
    example: str = ""
    bounds: Bounds | None = None
    enforce: bool = False          # refuse a BEYOND value (definitional impossibilities only)

    @property
    def one_example(self) -> str:
        """The first example when the table lists alternatives ("300 K, or 27 degC")."""
        return self.example.split(", or ")[0].split(",")[0].strip()


def _bounds(text: str) -> tuple[Bounds | None, bool]:
    text = text.strip()
    if not text:
        return None, False
    enforce = text.endswith("!")
    text = text.rstrip("!").strip()
    if text.startswith("[") and text.endswith("]"):
        lo, hi = (float(x) for x in text[1:-1].split(","))
        return Bounds(hard_lo=lo, hard_hi=hi), enforce
    if text.startswith(">="):
        return Bounds(hard_lo=float(text[2:])), enforce
    if text.startswith(">"):
        return Bounds(hard_lo=float(text[1:]), lo_open=True), enforce
    raise ValueError(f"bad bounds {text!r}")


# name | unit | meaning | example | bounds   (unit "-" = none; bounds: ">0", ">=0", "[0,1]"; a trailing "!" = refuse when violated)
_SPEC_TABLE = """
L_meters | m | length scale L of the holographic dark-energy model (default: the Hubble radius) | 4.4e26 |
altitude_km | km | altitude above the surface | 10 |
area_m2 | m^2 | area | 0.5 | >0
barrier_width_nm | nm | width of the potential barrier | 1 |
box_size_nm | nm | side of the confining box | 1 |
core_thickness_m | m | core thickness | 8e-6 |
design_wavelength_m | m | design (reference) wavelength | 550 nm |
distance_m | m | distance | 100 |
focal_length_m | m | focal length (negative for a diverging lens) | 0.1 |
from_altitude_km | km | starting altitude | 400 |
from_radius_m | m | starting orbital radius | 6.8e6 |
grating_spacing_m | m | spacing between grating lines | 1e-6 |
height_m | m | height | 10 |
image_distance_m | m | distance from the lens to the image | 0.3 |
length_m | m | length | 2 |
loop_length_m | m | length of the loop | 0.1 |
object_distance_m | m | distance from the lens to the object | 0.5 |
orbital_radius_m | m | orbital radius (from the centre of the central body) | 4.2e7 |
planet_radius_km | km | planet radius | 6371 |
r_a_pm | pm | radius of atom A | 120 |
r_b_pm | pm | radius of atom B | 100 |
r_inner_m | m | inner radius | 0.01 |
r_outer_m | m | outer radius | 0.02 |
radius_m | m | radius | 0.1 | >0
rest_length_m | m | length measured in the object's rest frame | 100 |
roughness_m | m | surface roughness | 1e-6 |
screen_distance_m | m | distance from the slits to the screen | 1 |
semimajor_axis_au | au | semi-major axis | 1 |
semimajor_axis_m | m | semi-major axis | 1.496e11 |
separation_m | m | separation between the two bodies or charges | 1e-9 | >0
size_m | m | characteristic size | 0.1 |
slit_separation_m | m | distance between the slits | 0.1 mm |
slit_width_m | m | slit width | 0.02 mm |
sliding_distance_m | m | distance slid | 100 |
sphere_inner_m | m | inner sphere radius | 0.01 |
sphere_outer_m | m | outer sphere radius | 0.02 |
thickness_m | m | thickness | 1e-3 |
to_altitude_km | km | final altitude | 35786 |
to_radius_m | m | final orbital radius | 4.2e7 |
wavelength_m | m | wavelength | 550 nm | >0
width_m | m | width | 0.1 |
central_mass_kg | kg | mass of the central body | 1 solar_mass | >0
ion_mass_kg | kg | ion mass | 6.6e-27 | >0
mass1_kg | kg | first mass | 1000 | >0
mass2_kg | kg | second mass | 1000 | >0
mass_kg | kg | mass | 0.145 | >0
molecular_mass_kg | kg | mass of one molecule | 4.8e-26 | >0
rest_mass_kg | kg | rest mass | 9.109e-31 | >0
m_a_amu | u | mass of nucleus A | 12 |
m_b_amu | u | mass of nucleus B | 16 |
measured_mass_u | u | measured atomic (or nuclear) mass | 4.0026 |
rest_time_s | s | time interval measured in the rest frame | 1 |
tau_s | s | time constant or lifetime | 1e-3 | >0
time_s | s | time | 3600 |
step_s | s | integration time step of an N-body run | 1080 | >0
initial_speed_m_s | m/s | initial speed | 20 |
initial_velocity_m_s | m/s | initial velocity | 20 |
perp_velocity_m_s | m/s | velocity component perpendicular to the line of sight | 1e4 |
speed_m_s | m/s | speed | 15 |
u_m_s | m/s | speed of the second frame or object | 0.5 c |
v_m_s | m/s | speed | 0.5 c |
velocity1_m_s | m/s | velocity of the first body | 2 |
velocity2_m_s | m/s | velocity of the second body | -1 |
velocity_m_s | m/s | speed | 54 km/h |
wind_speed_m_s | m/s | wind speed | 10 |
acceleration_m_s2 | m/s^2 | acceleration | 1e-10 |
g_m_s2 | m/s^2 | gravitational acceleration | 9.80665 | >0
angle_deg | degree | angle | 30 |
angle_to_los_deg | degree | angle between the velocity and the line of sight | 60 |
incidence_angle_deg | degree | angle of incidence, from the normal | 30 |
incident_angle_deg | degree | angle of incidence, from the normal | 30 |
launch_angle_deg | degree | launch angle above the horizontal | 45 |
angular_velocity_rad_s | rad/s | angular velocity | 10 |
ambient_temperature_k | K | ambient temperature | 293.15 | >=0!
cold_temperature_k | K | cold-reservoir temperature | 300 | >=0!
critical_temp_k | K | critical temperature | 9.2 | >=0!
electron_temperature_k | K | electron temperature | 1e6 | >=0!
hot_temperature_k | K | hot-reservoir temperature | 600 | >=0!
t_cold_k | K | cold-reservoir temperature | 300 | >=0!
t_hot_k | K | hot-reservoir temperature | 600 | >=0!
t_kelvin | K | temperature in kelvin | 300 | >=0!
temperature_k | K | absolute temperature | 300 K, or 27 degC | >=0!
t_celsius | degC | temperature in degrees Celsius | 27 |
activation_energy_eV | eV | activation energy | 0.8 |
barrier_height_eV | eV | barrier height | 1 |
energy_eV | eV | energy | 13.6 |
energy_ev | eV | energy | 13.6 |
kinetic_energy_eV | eV | kinetic energy | 100 |
particle_energy_eV | eV | particle energy | 0.5 |
spin_orbit_constant_ev | eV | spin-orbit coupling constant | 0.01 |
energy_j | J | energy | 1e6 |
energy_joules | J | energy | 4.184e9 |
luminosity_watts | W | luminosity | 3.828e26 | >=0
power_w | W | power | 100 | >=0
intensity_w_m2 | W/m^2 | intensity | 1000 | >=0
pressure_pa | Pa | pressure | 1 atm |
bulk_modulus1_pa | Pa | bulk modulus of phase 1 | 160 GPa |
bulk_modulus2_pa | Pa | bulk modulus of phase 2 | 76 GPa |
e1_pa | Pa | Young's modulus of material 1 | 200 GPa |
e2_pa | Pa | Young's modulus of material 2 | 70 GPa |
yield1_pa | Pa | yield strength of material 1 | 250 MPa |
yield2_pa | Pa | yield strength of material 2 | 100 MPa |
force_n | N | force | 10 |
normal_force_n | N | normal (perpendicular) force | 100 | >=0
charge1_c | C | first charge | 1e-6 |
charge2_c | C | second charge | -1e-6 |
charge_c | C | charge | 1.602e-19 |
current_a | A | current | 2 |
saturation_current_a | A | diode saturation current | 1e-12 |
voltage_v | V | voltage | 12 |
applied_voltage_v | V | applied voltage | 0.5 |
resistance_ohm | ohm | resistance | 100 |
capacitance_f | F | capacitance | 10 uF |
inductance_h | H | inductance | 10 mH |
e_field_v_m | V/m | electric field | 1e5 |
b_field_t | T | magnetic flux density | 0.5 |
b_tesla | T | magnetic flux density | 1 |
magnetic_field_t | T | magnetic flux density | 1 |
fluid_density_kg_m3 | kg/m^3 | density of the fluid | 1000 | >0
acceptor_density_m3 | 1/m^3 | acceptor doping density | 1e23 | >=0
donor_density_m3 | 1/m^3 | donor doping density | 1e23 | >=0
electron_density_m3 | 1/m^3 | electron number density | 1e19 | >=0
number_density_m3 | 1/m^3 | number density | 2.5e25 | >=0
volume_m3 | m^3 | volume | 0.001 | >0
diffusivity_m2_s | m^2/s | diffusivity | 1e-9 |
viscosity_pa_s | Pa*s | dynamic viscosity | 1e-3 |
n2_m2_w | m^2/W | nonlinear refractive index n2 | 2.6e-20 |
frequency_hz | Hz | frequency | 2.4 GHz | >0
molality_mol_kg | mol/kg | molality | 0.5 | >=0
concentration_mol_l | mol/L | concentration | 0.1 | >=0
molarity_mol_l | mol/L | molarity | 0.1 | >=0
acid_concentration_M | mol/L | acid concentration | 0.1 | >=0
anion_concentration_M | mol/L | anion concentration | 0.1 | >=0
base_concentration_M | mol/L | base concentration | 0.1 | >=0
cation_concentration_M | mol/L | cation concentration | 0.1 | >=0
concentration_M | mol/L | concentration | 0.1 | >=0
initial_concentration_M | mol/L | initial concentration | 0.1 | >=0
second_concentration_M | mol/L | second concentration | 0.1 | >=0
acid_volume_mL | mL | volume of acid | 25 | >=0
base_volume_mL | mL | volume of base added | 20 | >=0
final_volume_mL | mL | final volume | 100 | >0
initial_volume_mL | mL | initial volume | 10 | >0
second_volume_mL | mL | second volume | 10 | >=0
n_moles | mol | amount of substance | 1 | >=0
emissivity | - | emissivity (0 = perfect mirror, 1 = black body) | 0.9 | [0,1]!
restitution | - | coefficient of restitution | 0.8 | [0,1]!
fraction1 | - | volume fraction of phase 1 | 0.3 | [0,1]!
fraction_b | - | fraction of component B | 0.3 | [0,1]!
friction_coefficient | - | coefficient of friction | 0.5 | >=0
drag_coefficient | - | drag coefficient | 0.47 | >=0
refractive_index | - | refractive index | 1.5 | >0
n1 | - | refractive index of the first medium | 1.0 | >0
n2 | - | refractive index of the second medium | 1.5 | >0
n_core | - | refractive index of the core | 1.48 | >0
n_clad | - | refractive index of the cladding | 1.46 | >0
n_dense | - | refractive index of the denser medium | 1.5 | >0
n_rare | - | refractive index of the rarer medium | 1.0 | >0
order | - | order (an integer) | 2 |
percent | - | percentage | 2 |
van_t_hoff_i | - | van 't Hoff factor i (particles per formula unit) | 2 | >0
degrees_of_freedom | - | degrees of freedom of a molecule (3, 5 or 6) | 3 | >0
gamma | - | heat-capacity ratio Cp/Cv (a pure number) | 1.4 | >0
applied_stress | Pa | applied stress | 100 MPa |
applied_voltage | V | applied bias voltage | 0.5 |
acceptor_density | - | acceptor doping density, per m^3 | 1e22 |
donor_density | - | donor doping density, per m^3 | 1e22 |
asperity_slope | - | mean absolute asperity slope m of the rough surfaces (the Cooper-Mikic-Yovanovich m) | 0.1 | >0
atomic_number | - | atomic number Z (an integer) | 26 | >0
c_squared | - | holographic dark-energy parameter c^2 (default: the DESI Union3 fit) | 0.41 |
current_density | - | current density of the Tafel calculation (same unit as exchange_current_density) | 10 |
exchange_current_density | - | exchange current density (same unit as current_density) | 0.001 | >0
d_electron_count | - | number of d electrons | 5 | >=0
density1 | kg/m^3 | mass density of body 1 | 7850 | >0
density2 | kg/m^3 | mass density of body 2 | 7850 | >0
nu1 | - | Poisson's ratio of body 1 | 0.3 |
nu2 | - | Poisson's ratio of body 2 | 0.3 |
dielectric_constant | - | relative permittivity eps_r (1 = vacuum) | 1 | >0
epsilon_r | - | relative permittivity eps_r (1 = vacuum) | 1 | >0
dislocation_density | - | dislocation density rho, per m^2 | 1e14 | >=0
electron_domains | - | steric number: bonded atoms plus lone pairs around the central atom (an integer) | 4 | >0
lone_pairs | - | number of lone pairs on the central atom (an integer) | 0 | >=0
electrons | - | electrons transferred per ion, n in m = M I t / (n F) (an integer) | 2 | >0
molar_mass_kg | kg/mol | molar mass | 0.0635 | >0
molar_mass_kg_per_mol | kg/mol | molar mass | 0.029 | >0
lambda_anion | - | limiting molar conductivity of the anion, S cm^2/mol | 76.3 |
lambda_cation | - | limiting molar conductivity of the cation, S cm^2/mol | 73.5 |
grover_n_qubits | - | number of qubits in the Grover search (an integer) | 3 | >0
qaoa_n_nodes | - | number of nodes of the Max-Cut graph (an integer) | 3 | >0
initial_brightness | - | initial brightness I0 (arbitrary units) | 1 | >=0
initial_strain | - | initial strain (dimensionless) | 0.001 |
plastic_strain | - | plastic strain (dimensionless) | 0.05 | >=0
intensity_max | - | maximum fringe intensity (any unit, the same as intensity_min) | 1 | >=0
intensity_min | - | minimum fringe intensity (any unit, the same as intensity_max) | 0.2 | >=0
n0 | - | linear refractive index of the medium | 1.45 | >0
n_high | - | refractive index of the high-index layer | 2.35 | >0
n_low | - | refractive index of the low-index layer | 1.46 | >0
n_pairs | - | number of high/low layer pairs (an integer) | 10 | >0
n_atoms | - | number of radioactive atoms N | 6.022e23 | >=0
n_initial | - | principal quantum number of the initial level (an integer) | 3 | >0
n_final | - | principal quantum number of the final level (an integer) | 2 | >0
n | - | principal quantum number (an integer) | 2 | >0
protons | - | number of protons (an integer) | 26 | >=0
neutrons | - | number of neutrons (an integer) | 30 | >=0
oxidation_state | - | oxidation state of the magnetic ion (an integer) | 3 |
ph | - | pH of the solution | 7 |
polarizability_fm2 | F*m^2 | molecular polarizability | 1.6e-40 |
prefactor | - | pre-exponential (collision) factor A of the Arrhenius law, per second for a first-order reaction | 1e13 | >0
rate_constant | - | first-order rate constant k, per second | 0.01 | >0
target_rate | - | rate constant k to reach with the Arrhenius law, per second | 0.1 | >0
ratio_base_over_acid | - | concentration ratio [base]/[acid] | 1 | >0
reaction_quotient | - | reaction quotient Q (dimensionless) | 1 | >0
rho_de_J_m3 | J/m^3 | observed dark-energy density | 6e-10 | >0
t_cold | K | temperature of the cold junction | 300 | >=0!
t_hot | K | temperature of the hot junction | 400 | >=0!
tnt_amount | - | amount of TNT equivalent, in the tool's `unit` ("ton", "kt" or "MT") | 1 | >=0
total_angular_momentum_j | - | total angular-momentum quantum number j (an integer or half-integer) | 1.5 | >=0
total_j | - | total angular-momentum quantum number j (an integer or half-integer) | 1.5 | >=0
""".strip()


def _load_specs() -> dict[str, Spec]:
    specs: dict[str, Spec] = {}
    for line in _SPEC_TABLE.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 4:
            raise ValueError(f"bad spec line: {line!r}")
        name, unit, meaning, example = cells[:4]
        bounds, enforce = _bounds(cells[4] if len(cells) > 4 else "")
        specs[name] = Spec(None if unit == "-" else unit, meaning, example, bounds, enforce)
    return specs


SPECS: dict[str, Spec] = _load_specs()

# meanings for the non-numeric parameter families (a regex on the parameter name -> text)
_TEXT_RULES: list[tuple[str, str]] = [
    (r"^(expression|scalar_field)$", 'a math expression written as in a textbook: x^2, x², 2x, sqrt(x) or √x, pi or π, e^x, |x|, sin x, (x+1)(x-1)'),
    (r"^when$", 'a date: an ISO date or date-time in UTC such as "2031-06-15", "JD 2462000.5" (TDB), or "now"'),
    (r"^equation$", 'an equation "lhs = rhs", or an expression taken as = 0; for an ODE use y, y\', y\'\''),
    (r"^(variable|func|t_var|s_var|x_var|k_var)$", "the name of a variable or function (a letter such as x or t); optional when the expression has exactly one free symbol"),
    (r"^variables$", 'comma-separated variable names, e.g. "x,y,z"'),
    (r"^(lower|upper|point)$", 'a number or expression; infinity may be written oo, inf, infinity or ∞'),
    (r"^direction$", 'one of "+", "-" or "+-" (two-sided)'),
    (r"^convention$", 'which convention the result uses; the default is the previous behaviour. fourier_transform: "ordinary" (e^(-2πikx)), "angular" (e^(-ikx)) or "unitary" (angular with 1/√(2π)); series_expansion: "exclusive" (terms below x^order) or "inclusive" (through x^order)'),
    (r"^(matrix|matrix_a|matrix_b)$", 'a matrix as a nested list, e.g. "[[1,2],[3,4]]"'),
    (r"^vector_b$", 'a vector as a list, e.g. "[1,2]"'),
    (r"^vector_field$", 'comma-separated components, e.g. "x*y, y*z, z*x"'),
    (r"^(from_units|to_units|unit)$", 'a unit name pint understands, e.g. "km/h", "eV", "solar_mass"'),
    (r"^quantity_string$", 'a number with units, e.g. "5.6 light_year"'),
    (r"^element_symbol$", 'chemical element symbol, e.g. "Fe"'),
    (r"^(name|key|category|contains|scope|mode|dataset|structure|structure_name|particle|qubit_type|direction|image_side)$",
     "a name from the tool's own list of known entries; an unknown one returns the list of what exists"),
    (r"(^|_)(key|key_1|key_2|material|material_1|material_2|material_a|material_b|material_n|material_p|semiconductor|mat_a|mat_b|mat_n|mat_p|metal_a|metal_b|anode|cathode|fuel|isotope|atom_a|atom_b|"
     r"central_body|body_name|other_body|star_name|counter_material|environment)$",
     "a name from the tool's own table (materials: see list_materials; bodies: list_bodies); an unknown name returns the list of known ones"),
    (r"^(question|scenario|reason|verb|object|handle|options|params|bond_angles_deg|bond_dipoles_debye|pka_list|qaoa_edges|grover_marked_item|simon_hidden_string)$",
     "see the tool's description"),
]


def meaning_of(name: str) -> str:
    if name in SPECS:
        return SPECS[name].meaning
    for pat, text in _TEXT_RULES:
        if re.search(pat, name):
            return text
    return ""


# ── quantities ─────────────────────────────────────────────────────────────────────────────────────

class ContractError(Exception):
    def __init__(self, code: str, message: str, param: str | None = None, expected: str | None = None, got: Any = None, example: Any = None):
        super().__init__(message)
        self.code, self.message, self.param, self.expected, self.got, self.example = code, message, param, expected, got, example

    def payload(self) -> dict:
        out = {"code": self.code, "message": self.message}
        for key in ("param", "expected", "got", "example"):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key)
        return out


_NUM = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*?)\s*$")
_UNIT_FIX = (("°C", "degC"), ("°F", "degF"), ("℃", "degC"), ("Ω", "ohm"), ("μ", "µ"), ("per cent", "percent"))

# Units people write as two words, which pint reads as a product ("solar masses" -> solar * masses): one unit each, singular or plural.
_PHRASES = tuple((re.compile(rf"\b{re.escape(words)}\b", re.I), unit) for words, unit in (
    ("solar masses", "solar_mass"), ("solar mass", "solar_mass"), ("solar radii", "solar_radius"), ("solar radius", "solar_radius"),
    ("solar luminosities", "solar_luminosity"), ("solar luminosity", "solar_luminosity"),
    ("earth masses", "earth_mass"), ("earth mass", "earth_mass"), ("earth radii", "earth_radius"), ("earth radius", "earth_radius"),
    ("jupiter masses", "jupiter_mass"), ("jupiter mass", "jupiter_mass"), ("jupiter radii", "jupiter_radius"), ("jupiter radius", "jupiter_radius"),
    ("light years", "light_year"), ("light year", "light_year"),
    ("astronomical units", "astronomical_unit"), ("astronomical unit", "astronomical_unit"),
    ("nautical miles", "nautical_mile"), ("nautical mile", "nautical_mile"),
))


def normalize_unit_text(unit_text: str) -> str:
    """The unit as pint should read it: °C/°F/Ω/µ spelled for pint, and the two-word units ("solar masses", "light years") joined into one."""
    for a, b in _UNIT_FIX:
        unit_text = unit_text.replace(a, b)
    for pattern, unit in _PHRASES:
        unit_text = pattern.sub(unit, unit_text)
    return unit_text


def _registry():
    from sigma_ground.mcp.tools.units import _get_registry
    reg = _get_registry()
    if reg is None:
        raise ContractError("unit_support_missing", "unit conversion needs pint, which is not installed")
    return reg


def to_si(text: str, name: str, spec: Spec) -> tuple[float, str | None]:
    """A string for a numeric parameter -> (value in the parameter's unit, a note if a conversion happened)."""
    m = _NUM.match(text)
    if not m:
        raise ContractError("bad_quantity", f"{name}: '{text}' is not a number with optional units",
                            param=name, expected=_expected(name, spec), got=text, example=spec.example or None)
    number, unit_text = float(m.group(1)), m.group(2)
    if not unit_text:
        return number, None
    if spec.unit is None:
        raise ContractError("bad_quantity", f"{name} is a pure number; '{text}' carries units", param=name, expected="a plain number", got=text)
    unit_text = normalize_unit_text(unit_text)
    reg = _registry()
    import pint
    try:
        quantity = reg.Quantity(number, unit_text)
        value = quantity.to(spec.unit).magnitude
    except pint.DimensionalityError as e:
        raise ContractError("wrong_dimension", f"{name}: '{text}' is {_dimension_name(quantity)}, but {name} is {_dimension_phrase(spec)}",
                            param=name, expected=_expected(name, spec), got=text, example=spec.example or None) from e
    except Exception as e:  # noqa: BLE001 — pint raises several things for a unit it does not know
        raise ContractError("bad_quantity", f"{name}: could not read the unit in '{text}' ({type(e).__name__})",
                            param=name, expected=_expected(name, spec), got=text, example=spec.example or None) from e
    return float(value), (None if quantity.units == reg.Unit(spec.unit) else f"{text} = {value:.10g} {spec.unit}")


def _dimension_name(q) -> str:
    return "a " + str(q.dimensionality).replace("[", "").replace("]", "").replace("**", "^") if str(q.dimensionality) != "dimensionless" else "a pure number"


def _dimension_phrase(spec: Spec) -> str:
    return f"a quantity in {spec.unit}" if spec.unit else "a pure number"


def _expected(name: str, spec: Spec) -> str:
    bits = [f"a number in {spec.unit}" if spec.unit else "a number"]
    if spec.unit:
        bits.append(f'or a string with units, e.g. "{spec.example}"' if spec.example and any(c.isalpha() for c in spec.example) else "or a string with units, e.g. \"1 km\"")
    return " ".join(bits)


# ── wrapping a tool ────────────────────────────────────────────────────────────────────────────────

def _describe_param(name: str, ann: Any, default: Any) -> str:
    spec = SPECS.get(name)
    parts = []
    meaning = meaning_of(name)
    if meaning:
        parts.append(meaning)
    if spec:
        if spec.unit:
            parts.append(f"unit {spec.unit}; also accepts a string with units")
        if spec.bounds is not None:
            parts.append(describe(spec.bounds))
        if spec.example:
            parts.append(f"e.g. {spec.example}")
    return "; ".join(p for p in parts if p)


def _is_float(ann: Any) -> bool:
    if ann is float:
        return True
    args = getattr(ann, "__args__", ())
    return bool(args) and float in args and all(a in (float, type(None)) for a in args)


def _unit_params(sig: inspect.Signature) -> set[str]:
    return {n for n, p in sig.parameters.items() if n in SPECS and SPECS[n].unit and _is_float(p.annotation)}


def _args_block(fn_name: str, sig: inspect.Signature, doc: str, has_units: bool) -> str:
    lines = []
    for n, p in sig.parameters.items():
        d = _describe_param(n, p.annotation, p.default)
        if d:
            req = "" if p.default is not inspect.Parameter.empty else " (required)"
            lines.append(f"  {n}{req}: {d}")
    out = doc.rstrip()
    if lines and "Args:" not in doc:
        out += "\n\nArgs:\n" + "\n".join(lines)
    required = [n for n, p in sig.parameters.items() if p.default is inspect.Parameter.empty]
    if required and all(n in SPECS and SPECS[n].example for n in required) and "Example" not in doc and "e.g." not in doc:
        call = ", ".join(f"{n}={SPECS[n].one_example!r}" if any(c.isalpha() for c in SPECS[n].one_example) else f"{n}={SPECS[n].one_example}" for n in required)
        out += f"\n\nExample: {fn_name}({call})"
    if has_units:
        out +="\n\nUnits: a numeric parameter with a unit accepts a number in that unit or a string with units (\"54 km/h\", \"27 degC\"); the conversion is echoed in `converted`, and when the units are not SI the result's `display` gives the answer in the units you used with the SI value in parentheses (\"1 yard (0.9144 m)\"); `value` is always SI."
    out += "\n\nOn failure the result has value null and an `error` {code, message, param, expected, example}."
    return out


ALIASES: dict[str, dict[str, str]] = {}      # tool -> {old parameter name: canonical name}, filled as tools are wrapped (mcp/param_names.py is the map)


def translate_aliases(tool: str, arguments: dict) -> dict:
    """A call's arguments with the old parameter names of `tool` replaced by the canonical ones; if both are given the canonical one wins."""
    renamed = ALIASES.get(tool)
    if not renamed:
        return arguments
    out: dict = {}
    for k, v in arguments.items():
        new = renamed.get(k)
        if new is not None and new in arguments:
            continue
        out[new or k] = v
    return out


def wrap_function(fn, name: str | None = None):
    """The contract around one tool function: canonical parameter names (old ones still accepted), quantities, bounds, documentation, uniform results."""
    try:
        raw_sig = inspect.signature(fn, eval_str=True)
    except Exception:  # noqa: BLE001 — leave a function we cannot introspect alone
        return fn
    tool_name = name or fn.__name__
    simple = not any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in raw_sig.parameters.values())
    renamed = param_names.rename_map(tool_name, raw_sig.parameters) if simple else {}
    back = param_names.inverse(renamed)
    if renamed:
        ALIASES[tool_name] = renamed
    sig = raw_sig.replace(parameters=[p.replace(name=renamed.get(n, n)) for n, p in raw_sig.parameters.items()])
    doc_text = fn.__doc__ or fn.__name__
    for old, new in renamed.items():
        doc_text = re.sub(rf"\b{re.escape(old)}\b", new, doc_text)
    unit_params = _unit_params(sig)
    params = []
    for n, p in sig.parameters.items():
        ann = p.annotation if p.annotation is not inspect.Parameter.empty else Any
        desc = _describe_param(n, ann, p.default)
        if n in unit_params:
            optional = type(None) in getattr(ann, "__args__", ())
            ann = Optional[Union[float, str]] if optional else Union[float, str]
        if desc:
            ann = Annotated[ann, Field(description=desc)]
        params.append(p.replace(annotation=ann))
    new_sig = sig.replace(parameters=params)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        bound = new_sig.bind(*args, **kwargs)
        converted: dict[str, str] = {}
        user_units: list[str] = []
        for n, v in list(bound.arguments.items()):
            spec = SPECS.get(n)
            if spec is None or v is None:
                continue
            if n in unit_params and isinstance(v, str):
                unit_text = _unit_text(v)
                if unit_text:
                    user_units.append(unit_text)
                v, note = to_si(v, n, spec)
                bound.arguments[n] = v
                if note:
                    converted[n] = note
            if spec.enforce and spec.bounds is not None and isinstance(v, (int, float)) and not isinstance(v, bool):
                regime, why = classify(float(v), spec.bounds)
                if regime is Regime.BEYOND:
                    raise ContractError("out_of_range", f"{n} = {v:g} is impossible: it {why}", param=n,
                                        expected=f"{n} {describe(spec.bounds)}", got=v, example=spec.example or None)
        result = fn(*bound.args, **{back.get(k, k): v for k, v in bound.kwargs.items()})
        return _finish(result, converted, user_units, renamed)

    wrapper.__signature__ = new_sig
    wrapper.__doc__ = _args_block(fn.__name__, new_sig, doc_text, bool(unit_params))
    return wrapper


_UNKNOWN_KEY = re.compile(r"not in (the )?(lookup )?table|unknown (key|material|body|element|isotope)|no such|not found|not a known|no data for", re.I)


def _unit_text(text: str) -> str:
    """The unit part of a quantity string ("54 km/h" -> "km/h"), spelled as to_si reads it; "" for a bare number."""
    m = _NUM.match(text)
    return normalize_unit_text(m.group(2)) if m else ""


def _finish(result: Any, converted: dict, user_units: Any = (), renamed: dict | None = None) -> Any:
    """Uniform result shape: `error` whenever value is null; `converted` when a quantity string was converted; `display` (the answer in the
    caller's own measurement system, SI in parentheses) when a quantity was given in non-SI units; `inputs` echoed under the canonical names."""
    if not isinstance(result, dict):
        return result
    if renamed and isinstance(result.get("inputs"), dict):
        result["inputs"] = {renamed.get(k, k): v for k, v in result["inputs"].items()}
    if converted:
        result["converted"] = converted
    if user_units and result.get("value") is not None:
        from sigma_ground.mcp.unit_display import display_answer
        shown = display_answer(result.get("value"), str(result.get("units") or ""), user_units)
        if shown:
            result["display"] = shown
    if result.get("value") is None and not result.get("error") and "value" in result:
        msg = str(result.get("notes") or result.get("source") or "the tool declined")
        result["error"] = {"code": "unknown_key" if _UNKNOWN_KEY.search(msg) else "declined", "message": msg}
    if isinstance(result.get("value"), str) and "value_float" not in result:
        f = _symbolic_number(result["value"])
        if f is not None:
            result["value_float"] = f      # "1/3", "pi", "-1 + E": the exact answer stays `value`; this is its decimal
    return result


def _symbolic_number(text: str) -> float | None:
    """The decimal value of a symbolic answer that is a number ("1/3" -> 0.333..., "pi" -> 3.14159...); None for anything with a free symbol."""
    if not text or len(text) > 200 or not any(ch.isdigit() or ch.isalpha() for ch in text):
        return None
    try:
        import sympy as sp
        v = sp.sympify(text)
        if getattr(v, "free_symbols", None) == set() and v.is_real:
            f = float(v.evalf())
            return f if math.isfinite(f) else None
    except Exception:  # noqa: BLE001 — not a number, nothing to add
        return None
    return None


# ── errors that happen before the tool runs, and the server hook ───────────────────────────────────

def error_result(tool: str, arguments: dict, err: ContractError) -> dict:
    return {"value": None, "units": "", "source": "mentat tool contract", "uncertainty": None, "formula": None,
            "provenance_tag": None, "notes": err.message, "inputs": arguments, "interpreted_as": None,
            "error": err.payload(), "assumed": None}


def _example_call(tool: str, schema: dict) -> dict | None:
    ex = {}
    for n in schema.get("required", []):
        spec = SPECS.get(n)
        if not spec or not spec.example:
            return None
        ex[n] = spec.one_example if any(c.isalpha() for c in spec.one_example) else float(spec.one_example)
    return {tool: ex}


def from_validation_error(tool: str, arguments: dict, exc: BaseException, schema: dict) -> ContractError:
    """pydantic's complaint about the arguments -> a ContractError that says what to send instead."""
    errors = getattr(exc, "errors", lambda: [])()
    if not errors:
        return ContractError("invalid_arguments", f"{tool}: could not use the arguments ({str(exc)[:200]})")
    e = errors[0]
    loc = ".".join(str(x) for x in e.get("loc", ()) if x != "arguments")
    kind = e.get("type", "")
    if kind == "missing":
        spec = SPECS.get(loc)
        return ContractError("missing_argument", f"{tool} needs '{loc}'" + (f" ({meaning_of(loc)})" if meaning_of(loc) else ""),
                             param=loc, expected=_expected(loc, spec) if spec else meaning_of(loc) or "a value",
                             example=_example_call(tool, schema))
    got = e.get("input")
    spec = SPECS.get(loc)
    return ContractError("bad_type", f"{tool}: '{loc}' = {got!r} is not valid ({e.get('msg', kind)})", param=loc,
                         expected=_expected(loc, spec) if spec else meaning_of(loc) or e.get("msg"), got=got,
                         example=spec.example if spec and spec.example else _example_call(tool, schema))


def install(server) -> None:
    """Apply the contract to every tool registered on `server` afterwards, and make its errors uniform."""
    from mcp.server.fastmcp.exceptions import ToolError
    from mcp.types import TextContent

    original_tool = server.tool

    def tool(*a, **k):
        deco = original_tool(*a, **k)
        registered = k.get("name") or (a[0] if a and isinstance(a[0], str) else None)
        return lambda fn: deco(wrap_function(fn, registered))
    server.tool = tool

    original_call = server.call_tool

    async def call_tool(name, arguments):
        arguments = translate_aliases(name, arguments or {})
        schema, props = {}, set()
        try:
            t = server._tool_manager.get_tool(name)
            schema = t.parameters if t is not None else {}
            props = set(schema.get("properties", {}))
        except Exception:  # noqa: BLE001
            pass
        try:
            out = await original_call(name, arguments)
        except ToolError as e:
            cause = e.__cause__
            err = cause if isinstance(cause, ContractError) else from_validation_error(name, arguments, cause or e, schema)
            if err.code == "missing_argument":
                solved = await solve_for_missing(name, arguments, err.param, props, original_call)
                if solved is not None:
                    return [TextContent(type="text", text=json.dumps(solved, ensure_ascii=False, default=str))], {"result": solved}
            payload = error_result(name, arguments, err)
            return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, default=str))], {"result": payload}
        extra = [k for k in arguments if props and k not in props]
        if extra:
            out = _disclose_ignored(out, extra, props)
        return out
    server.call_tool = call_tool
    try:   # the protocol handler was bound to the original method when the server was built: bind it to ours
        server._mcp_server.call_tool(validate_input=False)(call_tool)
    except Exception:  # noqa: BLE001 — SDK internals changed: the wrapped tools still work, errors are just not uniform on stdio
        pass


def _payload_of(out) -> dict | None:
    """The result dict inside what server.call_tool returns."""
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        p = out[1].get("result", out[1]) if set(out[1]) == {"result"} else out[1]
        return p if isinstance(p, dict) else None
    return None


_SOLVE_DECADES = range(-18, 19)


async def solve_for_missing(tool: str, arguments: dict, missing: str | None, props: set, call) -> dict | None:
    """A formula tool asked backwards: "what plate area gives 1 uF?" arrives as parallel_plate_capacitance(capacitance_f=1e-6,
    separation_m=1e-3) with the required `area_m2` missing and the tool's own OUTPUT given under a name the tool has no parameter for.
    When exactly one required parameter is missing and exactly one unknown argument is given, and that argument's unit has the
    dimension of the tool's result, the tool is solved numerically for the missing parameter (positive quantities, bracketed over
    decades, bisection to 1e-12 relative) and the forward call at the answer is checked and disclosed. Anything else -> None, and
    the missing_argument error stands. No formula is written by hand here: the tool itself is the function being inverted.
    (misc/STATIC_INTENTS_PLAN.md group D, 2026-10-30.)"""
    if not missing or not props or missing in arguments:
        return None
    extra = [k for k in arguments if k not in props]
    if len(extra) != 1:
        return None
    target_name = extra[0]
    target_spec, missing_spec = SPECS.get(target_name), SPECS.get(missing)
    if target_spec is None or target_spec.unit is None or missing_spec is None:
        return None
    try:
        target = float(to_si(arguments[target_name], target_name, target_spec)[0]) if isinstance(arguments[target_name], str) else float(arguments[target_name])
    except (TypeError, ValueError, ContractError):
        return None
    known = {k: v for k, v in arguments.items() if k != target_name}

    async def forward(p: float) -> dict | None:
        try:
            got = _payload_of(await call(tool, {**known, missing: p}))
        except Exception:  # noqa: BLE001 — a trial value the tool refuses is just not a bracket point
            return None
        return got if got and isinstance(got.get("value"), (int, float)) and not isinstance(got.get("value"), bool) else None

    # the unknown argument must be the tool's result: same dimension as the forward value's units
    probe = None
    for k in (0, -3, 3, -6, 6):
        probe = await forward(10.0 ** k)
        if probe is not None:
            break
    if probe is None:
        return None
    try:
        reg = _registry()
        if reg(target_spec.unit).dimensionality != reg(str(probe.get("units") or "dimensionless")).dimensionality:
            return None
    except Exception:  # noqa: BLE001 — a unit pint cannot read: do not guess
        return None

    # bracket: the first adjacent pair of decades where value - target changes sign
    prev = None
    lo = hi = None
    for k in _SOLVE_DECADES:
        p = 10.0 ** k
        got = await forward(p)
        if got is None:
            prev = None
            continue
        g = float(got["value"]) - target
        if g == 0.0:
            lo = hi = p
            break
        if prev is not None and (prev[1] < 0) != (g < 0):
            lo, hi = prev[0], p
            break
        prev = (p, g)
    if lo is None:
        return None
    if lo != hi:
        glo = float((await forward(lo))["value"]) - target
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            got = await forward(mid)
            if got is None:
                return None
            gm = float(got["value"]) - target
            if (gm < 0) == (glo < 0):
                lo, glo = mid, gm
            else:
                hi = mid
            if hi / lo - 1.0 < 1e-12:
                break
        p = math.sqrt(lo * hi)
    else:
        p = lo
    check = await forward(p)
    if check is None:
        return None
    rel = abs(float(check["value"]) - target) / abs(target) if target else abs(float(check["value"]))
    if rel > 1e-6:
        return None
    units = missing_spec.unit or ""
    return {
        "value": p, "units": units,
        "source": f"{check.get('source')}; solved numerically for {missing}",
        "uncertainty": None, "provenance_tag": check.get("provenance_tag"),
        "formula": f"{check.get('formula') or tool}, solved for {missing}",
        "inputs": {**known, target_name: target, "solved_for": missing},
        "interpreted_as": f"{tool} solved for {missing} given {target_name} = {target:g}",
        "notes": " ".join(x for x in (
            f"The tool was asked backwards: {missing} ({meaning_of(missing) or missing}) is the unknown and {target_name} = {target:g} "
            f"{target_spec.unit} is the given result. Solved numerically; check: {tool}({missing} = {p:.6g} {units}) returns "
            f"{float(check['value']):.6g} {check.get('units') or ''} (relative difference {rel:.1e}).",
            str(check.get("notes") or "")) if x),
        "assumed": check.get("assumed"), "error": None, "display": None, "converted": None,
    }


def _disclose_ignored(out, extra: list, props: set):
    """A tool received arguments it does not have: say so in the result instead of dropping them silently."""
    note = "; ".join(f"ignored unknown argument '{k}'" + (f" (did you mean '{m[0]}'?)" if (m := difflib.get_close_matches(k, props, n=1)) else "")
                     for k in extra)
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        payload = out[1].get("result", out[1]) if set(out[1]) == {"result"} else out[1]
        if isinstance(payload, dict):
            payload["ignored_arguments"] = extra
            payload["notes"] = "; ".join(x for x in (payload.get("notes"), note) if x)
            from mcp.types import TextContent
            text = json.dumps(payload, ensure_ascii=False, default=str)
            return [TextContent(type="text", text=text)], ({"result": payload} if set(out[1]) == {"result"} else payload)
    return out
