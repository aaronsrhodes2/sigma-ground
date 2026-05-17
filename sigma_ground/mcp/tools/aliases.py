"""Centralized alias maps so tools accept symbol / nickname / formal name.

The benchmark surfaced that Qwen often phrases a question using the
informal name a physicist would actually use ("Sirius", "the Dog Star",
"G", "Newton's constant", "Plank's h-bar") while the tool's canonical
key expects a specific form ("sirius_a", "gravitational_constant").

Each alias dict is just {informal_form -> canonical_key}. All keys
should be lowercased and underscore-spaced. The `resolve()` helper
applies the same normalization to the input.

Adding aliases:
  - Stay conservative -- only add genuinely common names, not every
    possible misspelling. The benchmark + real usage will reveal what's
    actually needed.
  - Prefer canonical names that are already in use (don't rename keys
    in the data tables; just point aliases at them).
"""

from __future__ import annotations


def _norm(s: str) -> str:
    """Lowercase, trim, collapse separators. Returns space-separated form.

    All alias-dict keys go through this on lookup, AND the dicts below
    are normalized at module import (see `_normalize_dict()`), so you
    can spell keys naturally in this file ('k_B', 'h-bar', 'α CMa')
    without worrying about matching the lookup form.
    """
    return (
        s.strip()
         .lower()
         .replace("'", "")
         .replace("’", "")
         .replace("-", " ")
         .replace("_", " ")
    )


def _normalize_dict(d: dict[str, str]) -> dict[str, str]:
    """Apply _norm() to every key so lookup matches regardless of style."""
    return {_norm(k): v for k, v in d.items()}


def resolve(name: str, alias_map: dict[str, str]) -> str:
    """Return the canonical key for `name`, or the normalized `name` if no alias.

    The caller can then look up its data dict with the result. Returning
    the normalized (underscore-joined) name on a miss means lookup
    tables that use e.g. lowercase keys ('earth', 'mars') still resolve
    directly.
    """
    norm = _norm(name)
    return alias_map.get(norm, norm.replace(" ", "_"))


# ============================================================
# CONSTANTS — symbol / nickname / formal name -> canonical sigma_ground
# or scipy.constants name
# ============================================================
# Canonical names match `sigma_ground.field.constants` UPPER_CASE keys
# OR `scipy.constants` module attribute names (lowercase). The
# constants.py lookup tries both, so either side of this map works.
CONSTANT_ALIASES: dict[str, str] = {
    # Alias values are LOWERCASE forms that, when uppercased by
    # _lookup_sigma_ground, match a constant in sigma_ground.field.constants
    # (the primary source). They also tend to match scipy.constants module
    # attributes for the same physical constant, so the scipy fallback works.

    # Gravitational constant (G, sigma_ground.field.constants.G)
    "big g": "g",
    "newton g": "g",
    "newton constant": "g",
    "newtons constant": "g",
    "newton's constant": "g",
    "newtons gravitational constant": "g",
    "newtonian constant of gravitation": "g",
    "gravitational constant": "g",
    # Standard gravity (g at Earth's surface) - no sigma_ground entry;
    # scipy.constants has `g`. Resolve to lowercase so lookup_scipy finds it.
    "little g": "g_standard",
    "g earth": "g_standard",
    "earth gravity": "g_standard",
    "surface gravity of earth": "g_standard",
    "gravitational acceleration at sea level": "g_standard",
    "free fall acceleration": "g_standard",
    "standard gravity": "g_standard",
    "9.8": "g_standard",
    "9.81": "g_standard",
    "9.80665": "g_standard",
    # Speed of light (sigma_ground.C)
    "c": "c",
    "c_0": "c",
    "c0": "c",
    "lightspeed": "c",
    "celeritas": "c",
    "speed of light": "c",
    "speed of light in vacuum": "c",
    "speed of light in a vacuum": "c",
    # Planck constant (sigma_ground.H_PLANCK)
    "h": "h_planck",
    "planck constant": "h_planck",
    "plancks constant": "h_planck",
    "planck's constant": "h_planck",
    "planck h": "h_planck",
    # Reduced Planck h-bar (sigma_ground.HBAR)
    "hbar": "hbar",
    "h bar": "hbar",
    "h-bar": "hbar",
    "ℏ": "hbar",
    "h slash": "hbar",
    "planck over 2 pi": "hbar",
    "reduced planck": "hbar",
    "reduced planck constant": "hbar",
    # Boltzmann (sigma_ground.K_B)
    "k": "k_b",
    "k_b": "k_b",
    "kb": "k_b",
    "boltzmann": "k_b",
    "boltzmann constant": "k_b",
    "boltzmanns k": "k_b",
    "boltzmann's k": "k_b",
    "boltzmanns constant": "k_b",
    # Avogadro (sigma_ground.N_AVOGADRO)
    "n_a": "n_avogadro",
    "na": "n_avogadro",
    "avogadro": "n_avogadro",
    "avogadros number": "n_avogadro",
    "avogadros constant": "n_avogadro",
    "molar number": "n_avogadro",
    # Gas constant (sigma_ground.R_GAS)
    "r": "r_gas",
    "molar gas constant": "r_gas",
    "ideal gas constant": "r_gas",
    "universal gas constant": "r_gas",
    "gas constant": "r_gas",
    # Stefan-Boltzmann (sigma_ground.STEFAN_BOLTZMANN)
    "sigma": "stefan_boltzmann",
    "σ": "stefan_boltzmann",
    "stefan boltzmann": "stefan_boltzmann",
    "stefan-boltzmann constant": "stefan_boltzmann",
    "stefan boltzmann constant": "stefan_boltzmann",
    "stefans constant": "stefan_boltzmann",
    "stefans sigma": "stefan_boltzmann",
    # Fine structure (sigma_ground.ALPHA)
    "alpha": "alpha",
    "α": "alpha",
    "alpha fs": "alpha",
    "alpha_em": "alpha",
    "fine structure": "alpha",
    "fine structure constant": "alpha",
    "1/137": "alpha",
    # Elementary charge (sigma_ground.E_CHARGE)
    "e": "e_charge",
    "elementary charge": "e_charge",
    "electron charge magnitude": "e_charge",
    "proton charge": "e_charge",
    "fundamental charge": "e_charge",
    # Electron rest mass (sigma_ground.M_ELECTRON_KG)
    "m_e": "m_electron_kg",
    "me": "m_electron_kg",
    "electron mass": "m_electron_kg",
    "electron rest mass": "m_electron_kg",
    # Proton / neutron rest mass (use scipy fallback; sigma_ground has
    # MeV variants only)
    "m_p": "proton_mass",
    "mp": "proton_mass",
    "proton rest mass": "proton_mass",
    "m_n": "neutron_mass",
    "mn": "neutron_mass",
    "neutron rest mass": "neutron_mass",
    # Vacuum permittivity (sigma_ground.EPS_0)
    "epsilon_0": "eps_0",
    "epsilon0": "eps_0",
    "ε_0": "eps_0",
    "electric constant": "eps_0",
    "vacuum permittivity": "eps_0",
    "permittivity of free space": "eps_0",
    # Vacuum permeability (sigma_ground.MU_0)
    "mu_0": "mu_0",
    "mu0": "mu_0",
    "μ_0": "mu_0",
    "magnetic constant": "mu_0",
    "vacuum permeability": "mu_0",
    "permeability of free space": "mu_0",
    # Bohr radius (sigma_ground.BOHR_RADIUS)
    "a_0": "bohr_radius",
    "a0": "bohr_radius",
    "bohr": "bohr_radius",
    # Rydberg (scipy.constants `Rydberg`)
    "r_inf": "rydberg",
    "r_infinity": "rydberg",
    "rydberg constant": "rydberg",
    # Hubble (sigma_ground.H0)
    "h_0": "h0",
    "h0": "h0",
    "hubble": "h0",
    "hubble constant": "h0",
    "hubbles constant": "h0",
    "hubble rate": "h0",
    # SSBM-specific (eta, sigma, phi, xi)
    "eta": "eta",
    "η": "eta",
    "sigma here": "sigma_here",
    "phi": "phi",
    "φ": "phi",
    "xi": "xi",
    "ξ": "xi",
    "gamma": "gamma",
    "γ": "gamma",
    # Astronomical units
    "au": "au_m",
    "astronomical unit": "au_m",
    "ev_to_j": "ev_to_j",
    "year_s": "year_s",
    "year in seconds": "year_s",
    # Solar mass / luminosity
    "m_sun": "m_sun_kg",
    "solar mass": "m_sun_kg",
    "l_sun": "l_sun_w",
    "solar luminosity": "l_sun_w",
}


# ============================================================
# STARS — common name / Bayer designation / nickname
# -> canonical key in astronomy._NAMED_STARS
# ============================================================
STAR_ALIASES: dict[str, str] = {
    # Sirius (αCMa, Dog Star)
    "sirius": "sirius_a",
    "α cma": "sirius_a",
    "alpha cma": "sirius_a",
    "alpha canis majoris": "sirius_a",
    "α canis majoris": "sirius_a",
    "dog star": "sirius_a",
    "the dog star": "sirius_a",
    # Alpha Centauri (Rigil Kentaurus)
    "alpha centauri": "alpha_centauri_a",
    "α centauri": "alpha_centauri_a",
    "α cen": "alpha_centauri_a",
    "alpha cen": "alpha_centauri_a",
    "rigil kentaurus": "alpha_centauri_a",
    "toliman": "alpha_centauri_a",
    # Betelgeuse (αOri, "Beetlejuice")
    "α ori": "betelgeuse",
    "alpha orionis": "betelgeuse",
    "α orionis": "betelgeuse",
    "beetlejuice": "betelgeuse",  # phonetic
    # Rigel (β Ori)
    "β ori": "rigel",
    "beta orionis": "rigel",
    "β orionis": "rigel",
    # Vega (α Lyr)
    "α lyr": "vega",
    "alpha lyrae": "vega",
    "α lyrae": "vega",
    # Polaris (α UMi, North Star)
    "north star": "polaris",
    "pole star": "polaris",
    "α umi": "polaris",
    "alpha ursae minoris": "polaris",
    "α ursae minoris": "polaris",
    # Proxima Centauri
    "proxima": "proxima_centauri",
    "proxima cen": "proxima_centauri",
    "α cen c": "proxima_centauri",
    "alpha centauri c": "proxima_centauri",
    "nearest star": "proxima_centauri",
}


# ============================================================
# SOLAR SYSTEM BODIES — formal / nickname / mythological name
# -> canonical key in astronomy._SOLAR_SYSTEM_BODIES
# ============================================================
BODY_ALIASES: dict[str, str] = {
    # Sun
    "sol": "sun",
    "helios": "sun",
    "the sun": "sun",
    "our star": "sun",
    # Mercury (no common alternate name)
    # Venus (Morning Star, Evening Star, Phosphorus, Hesperus)
    "morning star": "venus",
    "evening star": "venus",
    "phosphorus": "venus",
    "hesperus": "venus",
    # Earth
    "terra": "earth",
    "gaia": "earth",
    "the earth": "earth",
    "our planet": "earth",
    "planet earth": "earth",
    "world": "earth",
    # Moon (Luna)
    "luna": "moon",
    "the moon": "moon",
    "earths moon": "moon",
    # Mars (Red Planet, Ares)
    "ares": "mars",
    "red planet": "mars",
    "the red planet": "mars",
    # Jupiter (Zeus, Jove)
    "zeus": "jupiter",
    "jove": "jupiter",
    # Saturn (Cronus, Kronos)
    "cronus": "saturn",
    "kronos": "saturn",
    # Uranus
    "ouranos": "uranus",
    # Neptune (Poseidon)
    "poseidon": "neptune",
    # Pluto (Hades; dwarf planet)
    "hades": "pluto",
    "dwarf planet pluto": "pluto",
}


# ============================================================
# ELEMENTS — symbol / name / atomic number / Latin / alchemical
# -> canonical CHEMICAL SYMBOL ("H", "Fe", "Hg")
# ============================================================
# atomic.py keys its data tables by symbol (Capitalized). This dict
# maps every other form a user might type to that symbol.
ELEMENT_ALIASES: dict[str, str] = {
    # Hydrogen (H, Z=1)
    "h": "H",
    "hydrogen": "H",
    "1": "H",
    "z=1": "H",
    "protium": "H",
    # Helium (He, Z=2)
    "he": "He",
    "helium": "He",
    "2": "He",
    "z=2": "He",
    # Lithium (Li, Z=3)
    "li": "Li",
    "lithium": "Li",
    "3": "Li",
    # Beryllium (Be, Z=4)
    "be": "Be",
    "beryllium": "Be",
    "4": "Be",
    # Boron (B, Z=5)
    "boron": "B",
    "5": "B",
    # Carbon (C, Z=6)
    "carbon": "C",
    "6": "C",
    "z=6": "C",
    # Nitrogen (N, Z=7)
    "nitrogen": "N",
    "7": "N",
    "z=7": "N",
    # Oxygen (O, Z=8)
    "oxygen": "O",
    "8": "O",
    "z=8": "O",
    # Fluorine (F, Z=9)
    "f": "F",
    "fluorine": "F",
    "9": "F",
    # Neon (Ne, Z=10)
    "ne": "Ne",
    "neon": "Ne",
    "10": "Ne",
    # Sodium (Na, Natrium, Z=11)
    "na": "Na",
    "sodium": "Na",
    "natrium": "Na",
    "11": "Na",
    # Magnesium (Mg, Z=12)
    "mg": "Mg",
    "magnesium": "Mg",
    "12": "Mg",
    # Aluminum (Al, Z=13)
    "al": "Al",
    "aluminum": "Al",
    "aluminium": "Al",
    "13": "Al",
    # Silicon (Si, Z=14)
    "si": "Si",
    "silicon": "Si",
    "14": "Si",
    # Phosphorus (P, Z=15)
    "phosphorus": "P",
    "15": "P",
    # Sulfur (S, Z=16)
    "sulfur": "S",
    "sulphur": "S",
    "16": "S",
    # Chlorine (Cl, Z=17)
    "cl": "Cl",
    "chlorine": "Cl",
    "17": "Cl",
    # Argon (Ar, Z=18)
    "ar": "Ar",
    "argon": "Ar",
    "18": "Ar",
    # Potassium (K, Kalium, Z=19)
    "potassium": "K",
    "kalium": "K",
    "19": "K",
    # Calcium (Ca, Z=20)
    "ca": "Ca",
    "calcium": "Ca",
    "20": "Ca",
    # Iron (Fe, Ferrum, Z=26)
    "fe": "Fe",
    "iron": "Fe",
    "ferrum": "Fe",
    "26": "Fe",
    "z=26": "Fe",
    # Copper (Cu, Cuprum, Z=29)
    "cu": "Cu",
    "copper": "Cu",
    "cuprum": "Cu",
    "29": "Cu",
    # Silver (Ag, Argentum, Z=47)
    "ag": "Ag",
    "silver": "Ag",
    "argentum": "Ag",
    "47": "Ag",
    # Gold (Au, Aurum, Z=79)
    "au": "Au",
    "gold": "Au",
    "aurum": "Au",
    "79": "Au",
    # Mercury element (Hg, Hydrargyrum, Z=80) -- distinct from planet
    "hg": "Hg",
    "mercury_element": "Hg",
    "hydrargyrum": "Hg",
    "quicksilver": "Hg",
    "80": "Hg",
    # Lead (Pb, Plumbum, Z=82)
    "pb": "Pb",
    "lead": "Pb",
    "plumbum": "Pb",
    "82": "Pb",
    # Uranium (U, Z=92)
    "u": "U",
    "uranium": "U",
    "92": "U",
}


# ============================================================
# MATERIALS — formal / common / chemical formula
# -> canonical lowercase material name (matches materials.py table)
# ============================================================
MATERIAL_ALIASES: dict[str, str] = {
    # Water
    "h2o": "water",
    "h₂o": "water",
    "dihydrogen monoxide": "water",
    "liquid water": "water",
    # Air
    "atmospheric air": "air",
    "earth atmosphere": "air",
    # Diamond
    "c diamond": "diamond",
    "diamond carbon": "diamond",
    # Common metals already keyed by element name in materials.py;
    # add aliases that differ from the element table
    "stainless": "steel",
    "stainless steel": "steel",
    "mild steel": "steel",
    # Glass
    "silica glass": "glass",
    "fused silica": "glass",
    "window glass": "glass",
    # Ice
    "h2o ice": "ice",
    "frozen water": "ice",
    "water ice": "ice",
}


# Normalize all dict keys so 'k_B' / 'k-B' / 'k B' all match.
CONSTANT_ALIASES = _normalize_dict(CONSTANT_ALIASES)
STAR_ALIASES = _normalize_dict(STAR_ALIASES)
BODY_ALIASES = _normalize_dict(BODY_ALIASES)
ELEMENT_ALIASES = _normalize_dict(ELEMENT_ALIASES)
MATERIAL_ALIASES = _normalize_dict(MATERIAL_ALIASES)
