"""Modern physics (special relativity + early QM) pre-classifier.

Modern-physics questions in the corpus follow a tight set of formulas:
  - Lorentz factor gamma = 1/sqrt(1-v^2/c^2)
  - Time dilation Delta_t = gamma * Delta_tau
  - Length contraction L = L_0 / gamma
  - E = mc^2 (mass-energy equivalence)
  - L = (dm/dt) c^2 (luminosity -> mass conversion rate)
  - E = h c / lambda (photon energy from wavelength)
  - Einstein velocity addition u' = (u + v) / (1 + uv/c^2)
  - de Broglie lambda = h / (m v)
  - Doppler factor sqrt((1+beta)/(1-beta)) (non-relativistic: 1 + v/c)
  - TNT energy: 1 MT = 4.184e15 J

Qwen 7b picks unrelated tools (thin_lens_image_distance for time
dilation, rydberg for velocity addition, electrical_power for E=mc^2).
This classifier dispatches the right tool with correct args.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Speed of light
_C = 299792458.0


@dataclass
class ModernMatch:
    tool: str
    args: dict
    rationale: str
    result_override: float | None = None  # for computed values not from a tool
    result_units: str = ""


def _extract_velocity_fraction_of_c(question: str) -> float | None:
    """Parse 'X c' / 'X times c' / 'X times the speed of light' -> X."""
    # "0.9 c" / "0.9c" / "0.9 of c"
    m = re.search(r"\b([\-+]?[0-9]*\.?[0-9]+)\s*(?:times\s+(?:the\s+)?speed\s+of\s+light|times\s+c|\s*c\b|c\s+of\s+light)",
                    question, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _extract_proper_time_years(question: str) -> float | None:
    """Parse 'X years (his/her/proper time)' for time-dilation."""
    m = re.search(
        r"\bfor\s+([\-+]?[0-9]+(?:\.[0-9]+)?)\s*years?\s*"
        r"\(?(?:his|her|its|proper)\s*time\)?",
        question, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _extract_length_m(question: str) -> float | None:
    """Extract a length in meters for length-contraction questions."""
    m = re.search(r"\bA\s+([\-+]?[0-9]+(?:\.[0-9]+)?)\s*meter\s+rod",
                    question, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _extract_mass_kg(question: str) -> float | None:
    """Extract a mass in kg from 'X kg of mass' or 'rest mass X kg'."""
    m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][\-+]?[0-9]+)?)\s*kg",
                    question, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _extract_wavelength_m(question: str) -> float | None:
    """Extract a wavelength in meters from 'X nm' / 'X micrometer' etc."""
    m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*(nm|nanometer|nanometers)\b",
                    question, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1)) * 1e-9
        except ValueError:
            pass
    m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*(um|micrometer|micrometers|microns?)\b",
                    question, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1)) * 1e-6
        except ValueError:
            pass
    m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][\-+]?[0-9]+)?)\s*m(?!s)\b",
                    question, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _extract_luminosity_w(question: str) -> float | None:
    """Extract a luminosity in watts from 'X watts'."""
    m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][\-+]?[0-9]+)?)\s*watts?\b",
                    question, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _extract_velocity_ms(question: str) -> float | None:
    """Extract a velocity in m/s from 'X m/s' (NOT 'X c')."""
    m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][\-+]?[0-9]+)?)\s*m\s*/\s*s\b",
                    question, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _extract_velocity_kms_or_c(question: str) -> float | None:
    """For Doppler: 'X km/s' or 'X c' -> velocity in m/s."""
    m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*km\s*/\s*s\b",
                    question, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1)) * 1000.0
        except ValueError:
            pass
    frac = _extract_velocity_fraction_of_c(question)
    if frac is not None:
        return frac * _C
    return None


def classify_for_modern(question: str) -> ModernMatch | None:
    q = question

    # ── Lorentz factor ────────────────────────────────────
    if re.search(r"\b(?:gamma|lorentz)\s+factor\b", q, re.IGNORECASE):
        v_frac = _extract_velocity_fraction_of_c(q)
        if v_frac is not None:
            return ModernMatch(
                tool="lorentz_factor",
                args={"velocity_m_s": v_frac * _C},
                rationale=f"gamma at v={v_frac}c",
            )

    # ── Relativistic time dilation ────────────────────────
    if re.search(r"\b(?:time\s+passes|time\s+dilation|how\s+much\s+time\b)",
                  q, re.IGNORECASE):
        v_frac = _extract_velocity_fraction_of_c(q)
        years = _extract_proper_time_years(q)
        if v_frac is not None and years is not None:
            return ModernMatch(
                tool="relativistic_time_dilation",
                args={"rest_time_s": years * 365.25 * 86400,
                        "velocity_m_s": v_frac * _C},
                rationale=f"delta_t = gamma * tau, tau={years}yr, v={v_frac}c",
                result_units="s",
            )

    # ── Length contraction ────────────────────────────────
    if re.search(r"\b(?:length\s+contraction|how\s+long\s+is\s+it|long\s+is\s+it\s+as\s+seen)",
                  q, re.IGNORECASE):
        v_frac = _extract_velocity_fraction_of_c(q)
        L0 = _extract_length_m(q)
        if v_frac is not None and L0 is not None:
            return ModernMatch(
                tool="relativistic_length_contraction",
                args={"rest_length_m": L0, "velocity_m_s": v_frac * _C},
                rationale=f"L = L_0 / gamma, L_0={L0}m, v={v_frac}c",
            )

    # ── Mass to energy (E = mc^2) ─────────────────────────
    if re.search(r"\bE\s*=\s*m\s*c\^?2|energy\s+(?:contained\s+in|in)\s+\d+(?:\.\d+)?\s*kg|"
                  r"\brest\s+(?:mass.energy|energy)\s+in\s+(?:eV|MeV|GeV|J)",
                  q, re.IGNORECASE):
        mass_kg = _extract_mass_kg(q)
        if mass_kg is not None:
            return ModernMatch(
                tool="mass_to_energy",
                args={"mass_kg": mass_kg},
                rationale=f"E = m c^2, m={mass_kg}kg",
            )

    # ── Luminosity -> mass conversion rate ────────────────
    if re.search(r"\b(?:how\s+much\s+mass.*?convert|mass.*?convert.*?per\s+(?:second|year))",
                  q, re.IGNORECASE | re.DOTALL):
        L = _extract_luminosity_w(q)
        if L is not None:
            per_year = bool(re.search(r"\bper\s+year\b|\bevery\s+year\b",
                                          q, re.IGNORECASE))
            return ModernMatch(
                tool="luminosity_to_mass_conversion_rate",
                args={"luminosity_watts": L, "_per_year": per_year},
                rationale=f"dm/dt = L/c^2, L={L}W"
                            + (" (then * year_s)" if per_year else ""),
            )

    # ── Photon energy from wavelength ─────────────────────
    if re.search(r"\bphoton.*?carries|photon\s+energy.*?wavelength|"
                  r"\bphoton\s+of\s+\d", q, re.IGNORECASE | re.DOTALL):
        wl = _extract_wavelength_m(q)
        if wl is not None:
            return ModernMatch(
                tool="photon_energy_from_wavelength",
                args={"wavelength_m": wl},
                rationale=f"E = hc/lambda, lambda={wl}m",
                result_units="J",  # tool returns J; eV conversion handled by scorer
            )

    # ── Relativistic velocity addition ────────────────────
    if re.search(r"\b(?:relativistic\s+velocity\s+addition|"
                  r"add\s+\d.*?c\s+to\s+\d.*?c|Einstein\s+style)",
                  q, re.IGNORECASE | re.DOTALL):
        # Extract TWO velocity fractions
        fracs = re.findall(r"\b([\-+]?[0-9]*\.?[0-9]+)\s*(?:c\b|times\s+(?:the\s+speed\s+of\s+light|c))",
                              q, re.IGNORECASE)
        if len(fracs) >= 2:
            try:
                u = float(fracs[0]) * _C
                v = float(fracs[1]) * _C
                return ModernMatch(
                    tool="relativistic_velocity_addition",
                    args={"u_m_s": u, "v_m_s": v},
                    rationale=f"u'=(u+v)/(1+uv/c^2), u={fracs[0]}c, v={fracs[1]}c",
                )
            except ValueError:
                pass

    # ── 1 megaton nuclear -> joules ───────────────────────
    if re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*[\-]?megaton(?:s)?",
                  q, re.IGNORECASE) and \
       re.search(r"\b(?:nuclear|tnt|joule)", q, re.IGNORECASE):
        m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*[\-]?megaton",
                       q, re.IGNORECASE)
        try:
            n = float(m.group(1))
            return ModernMatch(
                tool="joules_to_TNT",
                args={"energy_joules": n * 4.184e15, "unit": "megaton"},
                rationale=f"{n} MT * 4.184e15 J/MT",
                result_override=n * 4.184e15,
                result_units="J",
            )
        except (ValueError, AttributeError):
            pass

    # ── de Broglie wavelength ─────────────────────────────
    if re.search(r"\bde\s+Broglie\s+wavelength", q, re.IGNORECASE):
        v_m_s = _extract_velocity_ms(q)
        if v_m_s is not None:
            # Mass priority: explicit "(X kg)" in question, then
            # particle-keyword default (electron / proton).
            m_kg_match = re.search(
                r"\(\s*([\-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][\-+]?[0-9]+)?)\s*kg\s*\)",
                q, re.IGNORECASE)
            if m_kg_match:
                try:
                    mass_kg = float(m_kg_match.group(1))
                except ValueError:
                    mass_kg = 9.1093837015e-31
            elif re.search(r"\bproton\b", q, re.IGNORECASE):
                mass_kg = 1.67262192e-27
            elif re.search(r"\belectron\b", q, re.IGNORECASE):
                mass_kg = 9.1093837015e-31
            else:
                # Don't dispatch -- defer to LLM if mass is unclear
                return None
            return ModernMatch(
                tool="de_broglie_wavelength",
                args={"mass_kg": mass_kg, "velocity_m_s": v_m_s},
                rationale=f"lambda = h/(m v), m={mass_kg}kg, v={v_m_s}m/s",
            )

    # ── Doppler shift factor ──────────────────────────────
    if re.search(r"\b(?:Doppler\s+shift|recedes\s+(?:from|at)|wavelength\s+(?:is\s+)?(?:shifted|redshift|blueshift))",
                  q, re.IGNORECASE):
        v = _extract_velocity_kms_or_c(q)
        if v is not None:
            return ModernMatch(
                tool="doppler_shift_factor",
                args={"velocity_m_s": v},
                rationale=f"sqrt((1+beta)/(1-beta)), v={v}m/s",
            )

    return None


def execute_modern_match(match: ModernMatch) -> tuple[object, str, str]:
    """Run the dispatched tool. Returns (value, units, answer_text)."""
    # If the classifier already computed the result (no tool call), return it
    if match.result_override is not None:
        return (match.result_override, match.result_units,
                f"ANSWER: {match.result_override} {match.result_units}\n\n"
                f"Computed via modern_classifier: tool={match.tool} ({match.rationale})")

    from sigma_ground.mcp.tools import relativity as t_rel
    from sigma_ground.mcp.tools import energy_conversion as t_econv
    from sigma_ground.mcp.tools import atomic as t_atom
    try:
        if match.tool == "lorentz_factor":
            r = t_rel.lorentz_factor(**match.args)
        elif match.tool == "relativistic_time_dilation":
            r = t_rel.relativistic_time_dilation(**match.args)
        elif match.tool == "relativistic_length_contraction":
            r = t_rel.relativistic_length_contraction(**match.args)
        elif match.tool == "mass_to_energy":
            r = t_econv.mass_to_energy(**match.args)
        elif match.tool == "luminosity_to_mass_conversion_rate":
            per_year = match.args.pop("_per_year", False)
            r = t_econv.luminosity_to_mass_conversion_rate(**match.args)
            # If question asked "per year", convert kg/s -> kg/year
            if per_year and r.value is not None:
                year_s = 365.25 * 86400.0
                r.value = r.value * year_s
                r.units = "kg"
        elif match.tool == "photon_energy_from_wavelength":
            r = t_atom.photon_energy_from_wavelength(**match.args)
        elif match.tool == "relativistic_velocity_addition":
            r = t_rel.relativistic_velocity_addition(**match.args)
        elif match.tool == "de_broglie_wavelength":
            r = t_atom.de_broglie_wavelength(**match.args)
        elif match.tool == "doppler_shift_factor":
            r = t_rel.doppler_shift_factor(**match.args)
        else:
            return None, "", ""
    except Exception as e:
        return None, "", f"<modern_classifier ERROR: {e}>"
    val = r.value if hasattr(r, "value") else None
    units = r.units if hasattr(r, "units") else ""
    if match.result_units and not units:
        units = match.result_units
    answer_text = (
        f"ANSWER: {val} {units}\n\n"
        f"Computed via modern_classifier: tool={match.tool} ({match.rationale})"
    )
    return val, units, answer_text
