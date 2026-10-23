"""The one pint unit registry behind Mentat's unit tools.

``convert_units`` / ``parse_quantity`` (mcp/tools/units.py) and Materia's question
reader (materia/quantities.py: "100 mph" in a question) use this same registry, so
a unit in a question is converted exactly as the public conversion tool converts it.
Kernel tier: importable from every layer. None without pint (a bare core install).
"""
from __future__ import annotations

# Singleton registry, lazily constructed. Extending pint's defaults with
# astrophysics units that aren't in the default registry but are commonly
# requested. Magnitudes match the values in sigma_ground.field.constants.
_REGISTRY = None


def get_registry():
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    try:
        import pint
    except ImportError:
        return None
    ureg = pint.UnitRegistry()
    # Add astronomy / astrophysics units missing from the default registry.
    # Pint already has: light_year, parsec, astronomical_unit ("au"), atomic_mass_unit, eV.
    # We add solar_*, jupiter_*, earth_*. Values from NASA fact sheets.
    from sigma_ground.field.constants import M_SUN_KG
    extra_defs = [
        f"solar_mass = {M_SUN_KG!r} * kilogram = M_sun",   # GM☉/G, field.constants
        "solar_radius = 6.957e8 * meter = R_sun",
        "solar_luminosity = 3.828e26 * watt = L_sun",
        "jupiter_mass = 1.89813e27 * kilogram = M_jup",
        "jupiter_radius = 7.1492e7 * meter = R_jup",
        "earth_mass = 5.97219e24 * kilogram = M_earth",
        "earth_radius = 6.3781e6 * meter = R_earth",
    ]
    for defn in extra_defs:
        try:
            ureg.define(defn)
        except (pint.errors.RedefinitionError, pint.errors.DefinitionSyntaxError):
            pass  # already defined or unparseable; skip
    _REGISTRY = ureg
    return ureg
