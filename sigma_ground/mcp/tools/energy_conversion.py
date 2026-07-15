"""Energy conversion and mass-energy equivalence helpers.

The "Sun's mass-energy rate" failure case in our earlier benchmark
(naive Qwen said 'millionth of a gram per second' when it should have
been ~4 million tonnes/second) is exactly what this module prevents:
ground every E↔m calculation in the tool, not the LLM's head.
"""

from __future__ import annotations

from sigma_ground.mcp.provenance import ToolResult


def mass_to_energy(mass_kg: float) -> ToolResult:
    """E = m c^2. Convert mass (rest mass) to energy in joules.

    Parameters
    ----------
    mass_kg : float
        Rest mass in kilograms. For atomic-scale masses use the
        convert_units tool first ("1 amu to kg" etc.).

    Returns
    -------
    ToolResult with energy in joules.
    """
    if mass_kg < 0:
        return ToolResult(value=None, source="invalid input",
                           notes="mass_kg must be non-negative",
                           inputs={"mass_kg": mass_kg})
    from sigma_ground.field.constants import C
    E = mass_kg * C * C
    return ToolResult(
        value=E,
        units="J",
        source="sigma-ground (Einstein mass-energy equivalence)",
        formula="E = m c^2",
        inputs={"mass_kg": mass_kg},
        notes=("This is the REST energy. Total energy at finite v is "
                "gamma m c^2 (see relativity.relativistic_energy)."),
    )


def energy_to_mass(energy_j: float) -> ToolResult:
    """m = E / c^2. Convert energy to equivalent rest mass.

    For an energy E in joules, the rest mass that would contain that
    much rest energy. Used for things like 'how much mass does a star
    convert per second?' (Solar luminosity = 3.828e26 W / c^2 = ~4.3e9 kg/s).
    """
    if energy_j < 0:
        return ToolResult(value=None, source="invalid input",
                           notes="energy_j must be non-negative",
                           inputs={"energy_j": energy_j})
    from sigma_ground.field.constants import C
    m = energy_j / (C * C)
    return ToolResult(
        value=m,
        units="kg",
        source="sigma-ground (Einstein mass-energy equivalence)",
        formula="m = E / c^2",
        inputs={"energy_j": energy_j},
        notes=("Inverse of E = m c^2. For 1 joule, this is ~1.11e-17 kg "
                "(tiny). For typical stellar luminosities (1e26-1e27 W) "
                "the rate is millions of tonnes per second."),
    )


def luminosity_to_mass_conversion_rate(luminosity_watts: float) -> ToolResult:
    """Mass converted to energy per second for a given radiated power.

    dm/dt = L / c^2. For the Sun (L_sun = 3.828e26 W), this is 4.26e9 kg/s
    = ~4.26 million tonnes per second.
    """
    if luminosity_watts < 0:
        return ToolResult(value=None, source="invalid input",
                           notes="luminosity_watts must be non-negative",
                           inputs={"luminosity_watts": luminosity_watts})
    from sigma_ground.field.constants import C
    dm_dt = luminosity_watts / (C * C)
    return ToolResult(
        value=dm_dt,
        units="kg/s",
        source="sigma-ground (mass-energy conservation)",
        formula="dm/dt = L / c^2",
        inputs={"luminosity_watts": luminosity_watts},
        notes=(f"Equivalent rate: {dm_dt * 1000:.3e} g/s, "
                f"{dm_dt / 1000:.3e} tonnes/s. For the Sun "
                f"(L=3.828e26 W) this is 4.26 million tonnes/s."),
    )


def joules_to_eV(energy_joules: float) -> ToolResult:
    """Convert joules to electronvolts. 1 eV = 1.602176634e-19 J (SI exact)."""
    from sigma_ground.field.constants import E_CHARGE
    eV = energy_joules / E_CHARGE
    return ToolResult(
        value=eV,
        units="eV",
        source="sigma-ground via SI definition of eV",
        formula="E_eV = E_J / e",
        inputs={"energy_joules": energy_joules},
        notes="1 eV = 1.602176634e-19 J (defined exactly via SI 2019).",
    )


def eV_to_joules(energy_eV: float) -> ToolResult:
    """Convert electronvolts to joules."""
    from sigma_ground.field.constants import E_CHARGE
    J = energy_eV * E_CHARGE
    return ToolResult(
        value=J,
        units="J",
        source="sigma-ground via SI definition of eV",
        formula="E_J = E_eV * e",
        inputs={"energy_eV": energy_eV},
    )


def joules_to_TNT(energy_joules: float, unit: str = "ton") -> ToolResult:
    """Convert joules to TNT equivalent.

    Standard: 1 ton TNT = 4.184e9 J (defined; IUPAC). 1 megaton = 1e6 ton.

    Parameters
    ----------
    energy_joules : float
        Energy in joules.
    unit : str
        "ton", "kiloton" (= "kt"), or "megaton" (= "MT"). Default "ton".
    """
    TON_TNT_J = 4.184e9  # IUPAC convention, exact by definition
    scale = {
        "ton": 1.0,
        "kiloton": 1e3,
        "kt": 1e3,
        "megaton": 1e6,
        "MT": 1e6,
    }
    factor = scale.get(unit)
    if factor is None:
        return ToolResult(
            value=None, source="invalid input",
            notes=f"unit must be one of {sorted(scale.keys())}",
            inputs={"energy_joules": energy_joules, "unit": unit},
        )
    tnt = energy_joules / (TON_TNT_J * factor)
    return ToolResult(
        value=tnt,
        units=unit + "_TNT" if not unit.endswith("TNT") else unit,
        source="sigma-ground via IUPAC TNT-equivalence (1 ton = 4.184e9 J)",
        formula="E_TNT = E_J / (4.184e9 * scale)",
        inputs={"energy_joules": energy_joules, "unit": unit},
        notes=("1 ton TNT = 4.184e9 J by IUPAC convention. Real TNT "
                "energy density is ~4.6e6 J/kg; the 4.184e9 J/ton "
                "value rounds for convention."),
    )


def tnt_to_joules(tnt_amount: float, unit: str = "ton") -> ToolResult:
    """Convert a TNT-equivalent amount to joules -- the inverse of
    joules_to_TNT. Standard: 1 ton TNT = 4.184e9 J (defined; IUPAC).

    Parameters
    ----------
    tnt_amount : float
        Amount of TNT equivalent (in the given unit).
    unit : str
        "ton", "kiloton" (= "kt"), or "megaton" (= "MT"). Default "ton".
    """
    TON_TNT_J = 4.184e9  # IUPAC convention, exact by definition
    scale = {
        "ton": 1.0,
        "kiloton": 1e3,
        "kt": 1e3,
        "megaton": 1e6,
        "MT": 1e6,
    }
    factor = scale.get(unit)
    if factor is None:
        return ToolResult(
            value=None, source="invalid input",
            notes=f"unit must be one of {sorted(scale.keys())}",
            inputs={"tnt_amount": tnt_amount, "unit": unit},
        )
    energy_j = tnt_amount * TON_TNT_J * factor
    return ToolResult(
        value=energy_j,
        units="J",
        source="sigma-ground via IUPAC TNT-equivalence (1 ton = 4.184e9 J)",
        formula="E_J = E_TNT * 4.184e9 * scale",
        inputs={"tnt_amount": tnt_amount, "unit": unit},
        notes=("1 ton TNT = 4.184e9 J by IUPAC convention. Real TNT "
                "energy density is ~4.6e6 J/kg; the 4.184e9 J/ton "
                "value rounds for convention."),
    )
