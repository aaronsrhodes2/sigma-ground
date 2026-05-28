"""Thermodynamics / statmech pre-classifier.

Thermodynamics questions in the corpus all use a small set of formulas:
  - Ideal gas pressure / volume      -> ideal_gas_pressure, ideal_gas_volume
  - Wien's law (peak wavelength)     -> blackbody_peak_wavelength
  - Stefan-Boltzmann total power     -> blackbody_total_power
  - Carnot efficiency                -> carnot_efficiency
  - Equipartition thermal energy     -> thermal_energy_per_molecule
  - Maxwell-Boltzmann most probable  -> maxwell_boltzmann_most_probable_speed
  - Boiling / melting point lookup   -> boiling_point / melting_point

Qwen 7b reliably picks unrelated tools (hydrogen_like_energy_level for
thermal energy, rydberg_hydrogen_wavelength for Wien's-law, etc.).
The classifier dispatches the right tool and extracts the relevant
numbers from the question text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ThermoMatch:
    tool: str
    args: dict
    rationale: str


def _extract_temperature_K(question: str) -> float | None:
    """Extract a temperature in Kelvin from the question text.

    Tries: 'X K' / 'X kelvin' (direct), 'X Celsius' / 'X C' (+273.15),
    'room temperature' (300 K convention used in the corpus).
    """
    # Direct K
    m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][\-+]?[0-9]+)?)\s*"
                    r"K(?:\s+star|\b|\s+blackbody)",
                    question, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # "at T = X K" or "at X K"
    m = re.search(r"\bat\s+(?:T\s*=\s*)?([\-+]?[0-9]+(?:\.[0-9]+)?)\s*K\b",
                    question, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # Celsius
    m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*(?:Celsius|°C|deg\s*C)",
                    question, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1)) + 273.15
        except ValueError:
            pass
    # Room temperature -> 300 K (corpus convention)
    if re.search(r"\broom\s+temperature\b", question, re.IGNORECASE):
        return 300.0
    return None


def _extract_volume_m3(question: str) -> float | None:
    """Extract a volume in m^3."""
    # "X liter[s]" or "X L"
    m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*liter|"
                    r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*L\b",
                    question, re.IGNORECASE)
    if m:
        n = m.group(1) or m.group(2)
        try:
            return float(n) / 1000.0
        except ValueError:
            pass
    # "X m^3"
    m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*m\^?3\b",
                    question, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _extract_moles(question: str) -> float | None:
    m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*mole",
                    question, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _extract_pressure_Pa(question: str) -> float | None:
    """Extract pressure in Pa. 'atmospheric pressure' -> 101325 Pa."""
    if re.search(r"\batmospheric\s+pressure\b|\bat\s+1\s+atm\b|\bone\s+atmosphere\b",
                  question, re.IGNORECASE):
        return 101325.0
    m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*Pa\b",
                    question, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _extract_two_temperatures_K(question: str) -> tuple[float, float] | None:
    """Extract T_hot, T_cold for Carnot efficiency questions."""
    # "between X K and Y K"
    m = re.search(r"\bbetween\s+([\-+]?[0-9]+(?:\.[0-9]+)?)\s*K\s+and\s+"
                    r"([\-+]?[0-9]+(?:\.[0-9]+)?)\s*K",
                    question, re.IGNORECASE)
    if m:
        try:
            a, b = float(m.group(1)), float(m.group(2))
            return max(a, b), min(a, b)
        except ValueError:
            pass
    return None


def _extract_particle_mass_kg(question: str) -> float | None:
    """Extract a particle/molecule mass in kg from common phrasings.

    Handles 'mass of O2 ~5.3e-26 kg' and similar.
    """
    m = re.search(r"\bmass\s+of\s+\S+\s+(?:is|~|=|≈)?\s*"
                    r"([\-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][\-+]?[0-9]+)?)\s*kg",
                    question, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # Just 'X kg' in context
    m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][\-+]?[0-9]+)?)\s*kg",
                    question, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _extract_material_for_boiling_melting(question: str) -> str | None:
    """Pick out a material name for boiling/melting questions."""
    m = re.search(
        r"\b(?:boiling|melting)\s+(?:point|temperature)\s+of\s+"
        r"(?:liquid\s+)?(\w+)",
        question, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # "X melts at what" or "X freezes at"
    m = re.search(r"\b(\w+)\s+(?:melts|freezes|boils)\b", question, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None


def classify_for_thermo(question: str) -> ThermoMatch | None:
    q = question

    # ── Ideal gas: pressure ───────────────────────────────
    if re.search(r"\b(pressure|exerts)\b.*\bideal\s+gas\b|"
                  r"\bideal\s+gas.*\bpressure\b",
                  q, re.IGNORECASE | re.DOTALL):
        T = _extract_temperature_K(q)
        V = _extract_volume_m3(q)
        n = _extract_moles(q)
        if T is not None and V is not None and n is not None:
            return ThermoMatch(
                tool="ideal_gas_pressure",
                args={"n_moles": n, "temperature_k": T, "volume_m3": V},
                rationale=f"PV=nRT, n={n}, T={T}K, V={V}m^3",
            )

    # ── Ideal gas: volume ─────────────────────────────────
    if re.search(r"\bvolume\b.*\bgas\b|\bgas.*\b(?:take\s+up|occup)",
                  q, re.IGNORECASE | re.DOTALL):
        T = _extract_temperature_K(q)
        P = _extract_pressure_Pa(q)
        n = _extract_moles(q)
        if T is not None and P is not None and n is not None:
            return ThermoMatch(
                tool="ideal_gas_volume",
                args={"n_moles": n, "temperature_k": T, "pressure_pa": P},
                rationale=f"V=nRT/P, n={n}, T={T}K, P={P}Pa",
            )

    # ── Wien's law: peak emission ─────────────────────────
    if re.search(r"\bpeak\s+(?:emission|wavelength|emit)|"
                  r"\bwavelength.*\bpeak\s+emission|"
                  r"\bWien", q, re.IGNORECASE):
        T = _extract_temperature_K(q)
        if T is not None:
            return ThermoMatch(
                tool="blackbody_peak_wavelength",
                args={"temperature_k": T},
                rationale=f"lambda_max = b/T, T={T}K",
            )

    # ── Stefan-Boltzmann total power ──────────────────────
    if re.search(r"\bpower\s+per\s+square\s+meter|"
                  r"\bblackbody\s+(?:total\s+)?(?:power|radiation)|"
                  r"\bStefan", q, re.IGNORECASE):
        T = _extract_temperature_K(q)
        if T is not None:
            return ThermoMatch(
                tool="blackbody_total_power",
                args={"temperature_k": T, "area_m2": 1.0},
                rationale=f"P=sigma T^4 A, T={T}K, A=1m^2",
            )

    # ── Carnot efficiency ─────────────────────────────────
    if re.search(r"\bCarnot|\b(?:maximum|max)\s+(?:possible\s+)?efficiency",
                  q, re.IGNORECASE):
        temps = _extract_two_temperatures_K(q)
        if temps is not None:
            T_hot, T_cold = temps
            return ThermoMatch(
                tool="carnot_efficiency",
                args={"t_hot_k": T_hot, "t_cold_k": T_cold},
                rationale=f"eta = 1 - T_c/T_h, T_h={T_hot}K, T_c={T_cold}K",
            )

    # ── Equipartition: average thermal energy ─────────────
    if re.search(r"\b(?:average\s+)?thermal\s+energy|"
                  r"\bequipartition|\benergy\s+per\s+molecule",
                  q, re.IGNORECASE):
        T = _extract_temperature_K(q)
        if T is not None:
            return ThermoMatch(
                tool="thermal_energy_per_molecule",
                args={"temperature_k": T},
                rationale=f"<E> = (3/2) k_B T, T={T}K",
            )

    # ── Maxwell-Boltzmann most probable speed ─────────────
    if re.search(r"\bmost\s+probable\s+speed|"
                  r"\bMaxwell.Boltzmann|\bv_?mp\b",
                  q, re.IGNORECASE):
        T = _extract_temperature_K(q)
        mass_kg = _extract_particle_mass_kg(q)
        if T is not None and mass_kg is not None:
            return ThermoMatch(
                tool="maxwell_boltzmann_most_probable_speed",
                args={"molecular_mass_kg": mass_kg, "temperature_k": T},
                rationale=f"v_mp = sqrt(2 k T / m), m={mass_kg}kg, T={T}K",
            )

    # ── Melting / boiling point lookup ────────────────────
    if re.search(r"\bboiling\s+point", q, re.IGNORECASE):
        mat = _extract_material_for_boiling_melting(q)
        if mat:
            return ThermoMatch(
                tool="boiling_point",
                args={"material": mat},
                rationale=f"boiling point of {mat}",
            )
    if re.search(r"\bmelting\s+point|\bmelts\s+at\b|\bfreezes\s+at\b",
                  q, re.IGNORECASE):
        mat = _extract_material_for_boiling_melting(q)
        if mat:
            return ThermoMatch(
                tool="melting_point",
                args={"material": mat},
                rationale=f"melting point of {mat}",
            )

    return None


def execute_thermo_match(match: ThermoMatch) -> tuple[object, str, str]:
    """Run the dispatched tool. Returns (value, units, answer_text)."""
    from sigma_ground.mcp.tools import thermodynamics as t_thermo
    from sigma_ground.mcp.tools import materials as t_mat
    try:
        if match.tool == "ideal_gas_pressure":
            r = t_thermo.ideal_gas_pressure(**match.args)
        elif match.tool == "ideal_gas_volume":
            r = t_thermo.ideal_gas_volume(**match.args)
        elif match.tool == "blackbody_peak_wavelength":
            r = t_thermo.blackbody_peak_wavelength(**match.args)
        elif match.tool == "blackbody_total_power":
            r = t_thermo.blackbody_total_power(**match.args)
        elif match.tool == "carnot_efficiency":
            r = t_thermo.carnot_efficiency(**match.args)
        elif match.tool == "thermal_energy_per_molecule":
            r = t_thermo.thermal_energy_per_molecule(**match.args)
        elif match.tool == "maxwell_boltzmann_most_probable_speed":
            r = t_thermo.maxwell_boltzmann_most_probable_speed(**match.args)
        elif match.tool == "boiling_point":
            r = t_mat.boiling_point(**match.args)
        elif match.tool == "melting_point":
            r = t_mat.melting_point(**match.args)
        else:
            return None, "", ""
    except Exception as e:
        return None, "", f"<thermo_classifier ERROR: {e}>"
    val = r.value if hasattr(r, "value") else None
    units = r.units if hasattr(r, "units") else ""
    answer_text = (
        f"ANSWER: {val} {units}\n\n"
        f"Computed via thermo_classifier: tool={match.tool} ({match.rationale})"
    )
    return val, units, answer_text
