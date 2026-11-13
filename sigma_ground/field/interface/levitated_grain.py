"""
A grain of osmium, resting and then floating, and what a photon does to it: the numbers behind the "lights up" stage.

Resting     Weight W = m g presses on the floor through a contact patch (Hertz 1882): for a sphere of radius R on a flat of
            combined modulus E*, the patch has radius a = (3 W R / 4 E*)^(1/3), the peak pressure is 3 W / (2 pi a^2) and the sphere
            and floor approach by a^2 / R. The floor's atoms are displaced differently for each place the grain could sit: the floor
            records where it is (see the cat-box gap "contact records the lump").
Floating    Osmium is a type-I superconductor below T_c = 0.66 K (``superconductivity``): it expels a magnetic field up to the
            thermodynamic critical field B_c, so a field pressing on its underside carries a weight up to B_c^2 / (2 mu_0) per unit
            projected area, no contact. For a sphere that bounds the radius: (4/3) pi R^3 rho g <= (B_c^2 / 2 mu_0) pi R^2, so
            R <= 3 B_c^2 / (8 mu_0 rho g). A 1 cm lump is far above it: only a grain can float.
A photon    carries h / lambda of momentum (a recoil of p / m) and the share of its energy the metal absorbs (a temperature rise
            of that over m c). Neither is anywhere near visible: what a photon changes is what is recorded, not the grain.
Flight      light crosses a distance d in d / c.

Origin tags:
  - Hertz contact: FIRST_PRINCIPLES (linear elasticity, frictionless, small contact)
  - Critical field: FIRST_PRINCIPLES BCS condensation energy (``superconductivity.thermodynamic_critical_field``; the free-electron
    density of states makes it a LOWER bound for a d-band metal like osmium: the real B_c is about 7 mT against ~4.7 mT here)
  - Osmium's modulus, Poisson ratio, specific heat and density: MEASURED (CRC Handbook: 570 GPa, 0.25, 130 J/(kg K), 22590 kg/m3)
  - The floor's modulus and Poisson ratio (granite): EMPIRICAL-INPUT, typical values
  - The grain's size, 20 micrometres: EXHIBIT choice below the floating limit

Not in it (GRAIN_GAPS): Hertz is for smooth spheres (a real grain's surface is rough: contact is at asperities, with a smaller true area); the
magnet's field is taken as uniform up to B_c under the grain (a real levitation needs a shaped field, so the true limit is lower); the
lattice of the floor is a continuum; adhesion (van der Waals) at this size is as large as the weight and is not included.
"""

import math

from ..constants import C, H_PLANCK
from . import superconductivity as _sc

MU_0 = 4.0e-7 * math.pi          # exact before 2019, 1.2566e-6 to 10 digits after
G = 9.80665                      # MEASURED: standard gravity
OSMIUM_DENSITY = 22590.0         # MEASURED: CRC Handbook
OSMIUM_E_PA = 570.0e9            # MEASURED: CRC Handbook, Young's modulus
OSMIUM_NU = 0.25                 # MEASURED: CRC Handbook, Poisson ratio
OSMIUM_CP = 130.0                # MEASURED: CRC Handbook, specific heat (J / kg K)
FLOOR_E_PA = 50.0e9              # EMPIRICAL-INPUT: granite
FLOOR_NU = 0.25                  # EMPIRICAL-INPUT: granite
GRAIN_RADIUS_M = 20.0e-6         # EXHIBIT choice: a speck, below the floating limit
GRAIN_SEPARATION_M = 100.0e-6    # EXHIBIT choice: the trigger moves it five radii sideways, or not
ABSORPTIVITY = 0.05              # EMPIRICAL-INPUT: a polished metal in the visible (the same as the emissivity used for the lump)

GRAIN_GAPS = (
    "Hertz contact assumes a smooth sphere on a smooth flat: a real grain touches at asperities, over a smaller true area, and adhesion at this size is as large as its weight.",
    "The levitation limit takes a uniform field up to the critical field under the whole grain; a real trap needs a shaped field, so the true limit is smaller.",
    "The critical field is the free-electron BCS value, a lower bound (~4.7 mT against ~7 mT measured), so the limit is conservative.",
    "The floor is a continuum with granite's modulus; its lattice's record of the grain is not computed (it is assumed to be immediate).",
)


def mass_kg(radius_m: float = GRAIN_RADIUS_M) -> float:
    return OSMIUM_DENSITY * 4.0 / 3.0 * math.pi * radius_m ** 3


def weight_n(radius_m: float = GRAIN_RADIUS_M) -> float:
    return mass_kg(radius_m) * G


def combined_modulus(e1: float = OSMIUM_E_PA, nu1: float = OSMIUM_NU, e2: float = FLOOR_E_PA, nu2: float = FLOOR_NU) -> float:
    """E* = 1 / ((1 - nu1^2) / E1 + (1 - nu2^2) / E2), the Hertz modulus of two bodies in contact."""
    return 1.0 / ((1.0 - nu1 ** 2) / e1 + (1.0 - nu2 ** 2) / e2)


def hertz_contact(radius_m: float = GRAIN_RADIUS_M) -> dict:
    """A sphere resting on a flat under its own weight: the contact patch's radius, the peak and mean pressure, and the approach (m, Pa)."""
    w = weight_n(radius_m)
    a = (3.0 * w * radius_m / (4.0 * combined_modulus())) ** (1.0 / 3.0)
    return {"weight_n": w, "patch_radius_m": a, "peak_pressure_pa": 3.0 * w / (2.0 * math.pi * a ** 2),
            "mean_pressure_pa": w / (math.pi * a ** 2), "approach_m": a ** 2 / radius_m}


def critical_field_t(temperature_k: float = 0.0) -> float:
    """Osmium's thermodynamic critical field B_c = mu_0 H_c (tesla), from the BCS condensation energy."""
    o = _sc.SUPERCONDUCTORS["osmium"]
    return MU_0 * _sc.thermodynamic_critical_field(o["n_e_m3"], o["T_c_K"], temperature_k)


def field_pressure_limit_pa(temperature_k: float = 0.0) -> float:
    """The most a magnetic field can push on a superconductor's surface before it enters the metal: B_c^2 / (2 mu_0)."""
    return critical_field_t(temperature_k) ** 2 / (2.0 * MU_0)


def levitation_limit_radius_m(temperature_k: float = 0.0) -> float:
    """The largest osmium sphere the field pressure can carry: R <= 3 B_c^2 / (8 mu_0 rho g), an upper bound (a uniform field under it)."""
    return 3.0 * critical_field_t(temperature_k) ** 2 / (8.0 * MU_0 * OSMIUM_DENSITY * G)


def can_float(radius_m: float, temperature_k: float = 0.0) -> bool:
    return radius_m <= levitation_limit_radius_m(temperature_k)


def photon_effects(wavelength_m: float, radius_m: float = GRAIN_RADIUS_M) -> dict:
    """What one photon that bounces off the grain does to it: its momentum kick, the velocity that gives, the energy absorbed, the temperature rise."""
    m = mass_kg(radius_m)
    e = H_PLANCK * C / wavelength_m
    p = H_PLANCK / wavelength_m
    return {"photon_energy_j": e, "momentum_kg_m_s": p, "recoil_m_s": p / m, "absorbed_j": ABSORPTIVITY * e,
            "temperature_rise_k": ABSORPTIVITY * e / (m * OSMIUM_CP)}


def flight_time_s(distance_m: float) -> float:
    """Light crosses ``distance_m`` in distance / c."""
    return distance_m / C
