"""Material properties: densities, melting/boiling points, refractive indices.

Curated lookup tables for common reference values. Sources cited per
table; CRC Handbook of Chemistry and Physics 100th ed. (2019) is the
primary reference unless otherwise noted.

For element-level data (atomic mass, isotopes, electron configs) we
defer to the periodictable package which is already installed.
"""

from __future__ import annotations

from sigma_ground.mcp.provenance import ToolResult


# Density at STP (T=273.15 K, P=100 kPa unless noted) in kg/m^3.
# CRC Handbook 100th ed. (2019), Section 4 (Properties of the Elements
# and Inorganic Compounds) and Section 6 (Fluid Properties).
_DENSITIES_KG_PER_M3: dict[str, tuple[float, str]] = {
    "water":      (999.972, "CRC Handbook (water at 4 C, 1 atm)"),
    "water_room": (997.05,  "CRC Handbook (water at 25 C, 1 atm)"),
    "ice":        (916.7,   "CRC Handbook (ice at 0 C)"),
    "seawater":   (1025.0,  "CRC Handbook (seawater, 35 g/kg salinity, 25 C)"),
    "air":        (1.2041,  "CRC Handbook (dry air at 20 C, 101.325 kPa)"),
    "air_stp":    (1.2754,  "CRC Handbook (dry air at 0 C, 100 kPa)"),
    "helium":     (0.17847, "CRC Handbook (He gas at 0 C, 100 kPa)"),
    "hydrogen":   (0.08988, "CRC Handbook (H2 gas at 0 C, 100 kPa)"),
    "iron":       (7874.0,  "CRC Handbook (iron at 20 C)"),
    "copper":     (8960.0,  "CRC Handbook (copper at 20 C)"),
    "aluminum":   (2700.0,  "CRC Handbook (aluminum at 20 C)"),
    "gold":       (19300.0, "CRC Handbook (gold at 20 C)"),
    "silver":     (10490.0, "CRC Handbook (silver at 20 C)"),
    "lead":       (11340.0, "CRC Handbook (lead at 20 C)"),
    "platinum":   (21450.0, "CRC Handbook (platinum at 20 C)"),
    "tungsten":   (19300.0, "CRC Handbook (tungsten at 20 C)"),
    "mercury":    (13534.0, "CRC Handbook (mercury liquid at 20 C)"),
    "uranium":    (19050.0, "CRC Handbook (uranium-238 at 20 C)"),
    "osmium":     (22590.0, "CRC Handbook (osmium, densest natural element)"),
    "glass":      (2500.0,  "CRC Handbook (soda-lime glass nominal)"),
    "concrete":   (2400.0,  "ACI 318 (normal-weight concrete)"),
    "wood_oak":   (770.0,   "CRC Handbook (white oak nominal)"),
    "wood_pine":  (530.0,   "CRC Handbook (Eastern white pine nominal)"),
    "rubber":     (1100.0,  "CRC Handbook (vulcanized rubber)"),
    "plastic":    (1100.0,  "CRC Handbook (typical thermoplastic, nominal)"),
    "diamond":    (3515.0,  "CRC Handbook (diamond at 20 C)"),
    "graphite":   (2266.0,  "CRC Handbook (graphite at 20 C)"),
    "silicon":    (2329.0,  "CRC Handbook (Si crystal at 20 C)"),
    "germanium":  (5323.0,  "CRC Handbook (Ge crystal at 20 C)"),
    "salt_nacl":  (2160.0,  "CRC Handbook (NaCl crystal at 20 C)"),
    "ethanol":    (789.0,   "CRC Handbook (ethanol at 20 C)"),
    "gasoline":   (737.0,   "ASTM (gasoline nominal at 15 C)"),
    "oil_olive":  (915.0,   "CRC Handbook (olive oil at 20 C)"),
    "honey":      (1420.0,  "USDA (honey at 20 C)"),
    "milk":       (1030.0,  "USDA (whole cow's milk at 20 C)"),
}


# Refractive indices at sodium D-line (589.3 nm), 20 C, 1 atm unless noted.
# CRC Handbook 100th ed., Section 10.
_REFRACTIVE_INDICES: dict[str, tuple[float, str]] = {
    "vacuum":     (1.0,      "exact by definition"),
    "air":        (1.000293, "CRC Handbook (dry air at 0 C, sea level, 589 nm)"),
    "water":      (1.333,    "CRC Handbook (water at 20 C, 589 nm)"),
    "ice":        (1.31,     "CRC Handbook (ice at 0 C, visible)"),
    "ethanol":    (1.361,    "CRC Handbook (ethanol at 20 C)"),
    "glass":      (1.50,     "CRC Handbook (typical crown glass)"),
    "glass_flint":(1.62,     "CRC Handbook (flint glass)"),
    "quartz":     (1.46,     "CRC Handbook (fused quartz/silica)"),
    "diamond":    (2.42,     "CRC Handbook (diamond at 589 nm)"),
    "silicon":    (3.42,     "CRC Handbook (Si at 1.55 um, telecom)"),
    "polystyrene":(1.59,     "CRC Handbook (polystyrene)"),
    "pmma":       (1.49,     "CRC Handbook (PMMA/acrylic)"),
    "sapphire":   (1.77,     "CRC Handbook (sapphire/corundum, ordinary ray)"),
}


# Melting points in kelvin at 1 atm. CRC Handbook 100th ed.
_MELTING_POINTS_K: dict[str, tuple[float, str]] = {
    "water":      (273.15,  "SI definition (triple point + offset)"),
    "ice":        (273.15,  "SI definition"),
    "ethanol":    (159.05,  "CRC Handbook"),
    "iron":       (1811.15, "CRC Handbook"),
    "copper":     (1357.77, "CRC Handbook"),
    "aluminum":   (933.47,  "CRC Handbook"),
    "gold":       (1337.33, "CRC Handbook"),
    "silver":     (1234.93, "CRC Handbook"),
    "lead":       (600.61,  "CRC Handbook"),
    "tungsten":   (3695.0,  "CRC Handbook (highest of all metals)"),
    "mercury":    (234.32,  "CRC Handbook"),
    "platinum":   (2041.4,  "CRC Handbook"),
    "diamond":    (4300.0,  "CRC Handbook (sublimation/decomposition at 1 atm)"),
    "silicon":    (1687.0,  "CRC Handbook"),
    "salt_nacl":  (1074.0,  "CRC Handbook"),
    "helium":     (0.95,    "CRC Handbook (He at 25 atm; no solid at 1 atm)"),
    "nitrogen":   (63.15,   "CRC Handbook"),
    "oxygen":     (54.36,   "CRC Handbook"),
    "carbon":     (3823.0,  "CRC Handbook (graphite at 1 atm sublimes)"),
}


# Boiling points in kelvin at 1 atm. CRC Handbook 100th ed.
_BOILING_POINTS_K: dict[str, tuple[float, str]] = {
    "water":      (373.15,  "SI definition at 1 atm"),
    "ethanol":    (351.39,  "CRC Handbook"),
    "iron":       (3134.0,  "CRC Handbook"),
    "copper":     (2835.0,  "CRC Handbook"),
    "aluminum":   (2792.0,  "CRC Handbook"),
    "gold":       (3129.0,  "CRC Handbook"),
    "silver":     (2435.0,  "CRC Handbook"),
    "lead":       (2022.0,  "CRC Handbook"),
    "mercury":    (629.88,  "CRC Handbook"),
    "helium":     (4.222,   "CRC Handbook"),
    "hydrogen":   (20.28,   "CRC Handbook"),
    "nitrogen":   (77.355,  "CRC Handbook"),
    "oxygen":     (90.188,  "CRC Handbook"),
    "neon":       (27.07,   "CRC Handbook"),
    "argon":      (87.30,   "CRC Handbook"),
    "co2":        (194.65,  "CRC Handbook (sublimation point at 1 atm)"),
}


# Young's modulus in Pa. CRC Handbook 100th ed. + ASM Metals Handbook.
_YOUNG_MODULUS_PA: dict[str, tuple[float, str]] = {
    "steel":      (200e9,   "ASM Handbook (low-C steel nominal)"),
    "iron":       (211e9,   "CRC Handbook"),
    "copper":     (110e9,   "CRC Handbook"),
    "aluminum":   (69e9,    "CRC Handbook"),
    "tungsten":   (411e9,   "ASM Handbook"),
    "diamond":    (1050e9,  "CRC Handbook"),
    "graphite":   (4.8e9,   "CRC Handbook (along c-axis)"),
    "glass":      (70e9,    "CRC Handbook (soda-lime nominal)"),
    "concrete":   (30e9,    "ACI 318 (compressive)"),
    "wood_oak":   (12e9,    "wood handbook (oak parallel to grain)"),
    "rubber":     (0.05e9,  "CRC Handbook (vulcanized natural rubber)"),
    "bone":       (15e9,    "Currey (cortical bone, dry)"),
}


# Semiconductor band gaps in eV at 300 K. CRC Handbook 100th ed.
_BAND_GAPS_EV: dict[str, tuple[float, str]] = {
    "silicon":    (1.12,    "CRC Handbook (Si at 300 K, indirect)"),
    "germanium":  (0.66,    "CRC Handbook (Ge at 300 K, indirect)"),
    "gaas":       (1.43,    "CRC Handbook (GaAs at 300 K, direct)"),
    "diamond":    (5.47,    "CRC Handbook (diamond)"),
    "sic":        (3.26,    "CRC Handbook (4H-SiC, ~3.0-3.3)"),
    "inp":        (1.34,    "CRC Handbook (InP at 300 K)"),
    "gan":        (3.39,    "CRC Handbook (GaN at 300 K, direct)"),
    "cdte":       (1.49,    "CRC Handbook (CdTe at 300 K)"),
    "ge_quantum_dot": (1.10, "approximate, depends on dot size"),
}


def _generic_lookup(table: dict, key: str, units: str,
                      kind: str) -> ToolResult:
    """Internal helper for table-based lookups.

    Accepts material name, chemical formula ('H2O' -> water), and
    common nicknames ('stainless' -> steel). See aliases.MATERIAL_ALIASES.
    Element symbols (Fe -> iron) also resolve here via ELEMENT_ALIASES
    so 'Fe' / 'iron' / 'ferrum' all hit the iron entry.
    """
    from sigma_ground.mcp.tools.aliases import (MATERIAL_ALIASES,
                                                    ELEMENT_ALIASES, resolve)
    # Try material aliases first; fall back to element aliases for
    # element-symbol inputs (Fe, Au, etc.).
    key_norm = resolve(key, MATERIAL_ALIASES)
    if key_norm not in table:
        # Element-symbol form (Fe -> iron); MATERIAL_ALIASES doesn't
        # know about element symbols, so try the element map and then
        # lowercase the element name.
        elem_sym = resolve(key, ELEMENT_ALIASES)
        if elem_sym != resolve(key, MATERIAL_ALIASES):  # element-lookup hit
            # Convert "Fe" -> "iron" via the values we just resolved against
            for alias, sym in ELEMENT_ALIASES.items():
                if sym == elem_sym and alias.isalpha() and len(alias) > 2:
                    if alias in table:
                        key_norm = alias
                        break
    if key_norm not in table:
        candidates = ", ".join(sorted(list(table.keys())[:20]))
        return ToolResult(
            value=None, source=f"sigma-ground ({kind} table)",
            notes=(f"'{key}' not in lookup table. Some available: "
                    f"{candidates}{'...' if len(table) > 20 else ''}. "
                    f"Total: {len(table)} entries."),
            inputs={"material": key},
        )
    value, src = table[key_norm]
    return ToolResult(
        value=value,
        units=units,
        source=f"sigma-ground via {src}",
        provenance_tag="VERIFIED",
        inputs={"material": key},
    )


def density(material: str) -> ToolResult:
    """Density of common material in kg/m^3.

    Examples: 'water', 'copper', 'air', 'gold', 'wood_oak', 'diamond'.
    """
    return _generic_lookup(_DENSITIES_KG_PER_M3, material, "kg/m^3", "density")


def refractive_index(material: str) -> ToolResult:
    """Refractive index at 589 nm (sodium D-line), 20 C unless noted.

    Examples: 'water' (1.333), 'glass' (1.50), 'diamond' (2.42).
    """
    return _generic_lookup(_REFRACTIVE_INDICES, material, "dimensionless",
                              "refractive index")


def melting_point(material: str) -> ToolResult:
    """Melting point at 1 atm in kelvin.

    Use the temperature conversion tools to get Celsius.
    """
    return _generic_lookup(_MELTING_POINTS_K, material, "K", "melting point")


def boiling_point(material: str) -> ToolResult:
    """Boiling point at 1 atm in kelvin."""
    return _generic_lookup(_BOILING_POINTS_K, material, "K", "boiling point")


def youngs_modulus(material: str) -> ToolResult:
    """Young's modulus (elastic modulus) in Pa."""
    return _generic_lookup(_YOUNG_MODULUS_PA, material, "Pa", "Young's modulus")


def band_gap_ev(material: str) -> ToolResult:
    """Semiconductor band gap at 300 K in eV.

    Examples: 'silicon' (1.12), 'germanium' (0.66), 'gaas' (1.43),
    'diamond' (5.47, insulator).
    """
    return _generic_lookup(_BAND_GAPS_EV, material, "eV", "band gap")


def list_materials() -> ToolResult:
    """List every material in the lookup tables.

    Returns a dict mapping property name -> list of available materials.
    Useful for the LLM to know what 'density of X' queries can succeed.
    """
    return ToolResult(
        value={
            "densities":         sorted(_DENSITIES_KG_PER_M3.keys()),
            "refractive_indices": sorted(_REFRACTIVE_INDICES.keys()),
            "melting_points":    sorted(_MELTING_POINTS_K.keys()),
            "boiling_points":    sorted(_BOILING_POINTS_K.keys()),
            "youngs_moduli":     sorted(_YOUNG_MODULUS_PA.keys()),
            "band_gaps":         sorted(_BAND_GAPS_EV.keys()),
        },
        source="sigma-ground (curated material tables)",
        notes=("All values from CRC Handbook 100th ed. unless otherwise "
                "noted. For element-level data (atomic mass, isotopes, "
                "electron configs) use the atomic / periodictable tools."),
    )


def element_atomic_data(element_symbol: str) -> ToolResult:
    """Atomic data for an element via periodictable: Z, name, atomic mass.

    Parameters
    ----------
    element_symbol : str
        Element symbol (e.g. 'H', 'He', 'Fe', 'Au') or full name.
    """
    try:
        import periodictable
    except ImportError:
        return ToolResult(value=None, source="periodictable not installed",
                           notes="pip install sigma-ground[mcp]",
                           inputs={"element_symbol": element_symbol})
    try:
        # periodictable indexes both by symbol and by name.
        elem = periodictable.elements.symbol(element_symbol)
    except (ValueError, AttributeError):
        try:
            elem = periodictable.elements.name(element_symbol.lower())
        except (ValueError, AttributeError):
            return ToolResult(
                value=None, source="periodictable lookup failed",
                notes=f"Element '{element_symbol}' not found",
                inputs={"element_symbol": element_symbol},
            )
    return ToolResult(
        value={
            "name":             elem.name,
            "symbol":           elem.symbol,
            "atomic_number":    elem.number,
            "atomic_mass_amu":  elem.mass,
            "covalent_radius_angstrom": getattr(elem, "covalent_radius", None),
        },
        source="sigma-ground via periodictable (NIST atomic data)",
        inputs={"element_symbol": element_symbol},
        notes=("Atomic mass in unified atomic mass units (amu). For natural "
                "abundance weighted; use isotope-specific lookups for "
                "pure isotopes."),
    )
