"""Waves/optics pre-classifier.

Targets the failing optics questions:
  - Snell's law refraction angle
  - Total internal reflection critical angle
  - Diffraction grating first-order angle
  - Speed of sound in air at temperature

Qwen 7b picks single_slit_first_minimum_angle and rydberg for these.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import asin, sin, degrees, sqrt, pi


@dataclass
class OpticsMatch:
    tool: str
    args: dict
    rationale: str
    result_override: float | None = None
    result_units: str = ""


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def classify_for_optics(question: str) -> OpticsMatch | None:
    q = question

    # ── Snell's law refraction ────────────────────────────
    # "Light hits X (n=N) from air at theta degrees from vertical"
    if re.search(r"\bbends\s+to\b|\b(?:refraction|refracts?\b)|\bSnell|"
                  r"\bat\s+\d+(?:\.\d+)?\s+degrees\s+from\s+vertical",
                  q, re.IGNORECASE):
        # Extract n_to (e.g. "n=1.333", "water (n=1.333)")
        m_n = re.search(r"\bn\s*=\s*([\-+]?[0-9]+(?:\.[0-9]+)?)", q, re.IGNORECASE)
        # Extract incident angle
        m_a = re.search(r"\bat\s+([\-+]?[0-9]+(?:\.[0-9]+)?)\s*degrees?\b",
                          q, re.IGNORECASE)
        if m_n and m_a:
            n_to = _num(m_n.group(1))
            theta_i = _num(m_a.group(1))
            if n_to is not None and theta_i is not None:
                # Default n_from = 1 (air)
                return OpticsMatch(
                    tool="snells_law_refraction_angle",
                    args={"n1": 1.0, "n2": n_to,
                            "incident_angle_deg": theta_i},
                    rationale=f"Snell: n1 sin(θ1) = n2 sin(θ2), "
                                  f"n1=1, n2={n_to}, θ1={theta_i}°",
                )

    # ── Total internal reflection critical angle ─────────
    # "looking up from underwater" / "critical angle"
    if re.search(r"\bcritical\s+angle\b|"
                  r"\btotal\s+internal\s+reflection\b|"
                  r"\b(?:from\s+)?underwater.*?(?:stop|escape)|"
                  r"\blight.*?stop\s+(?:being\s+able\s+to\s+)?escape",
                  q, re.IGNORECASE | re.DOTALL):
        # Determine n_from / n_to
        # Default: water (n=1.333) -> air (n=1.0)
        n_from = 1.333
        n_to = 1.0
        # Optional explicit n values
        m_nw = re.search(r"\bwater\b.*?\(?\s*n\s*=\s*([\-+]?[0-9]+(?:\.[0-9]+)?)\)?",
                          q, re.IGNORECASE)
        if m_nw:
            n_from = _num(m_nw.group(1)) or 1.333
        crit_deg = degrees(asin(n_to / n_from))
        return OpticsMatch(
            tool="critical_angle_for_tir",
            args={"n_from": n_from, "n_to": n_to},
            rationale=f"asin(n2/n1), n1={n_from}, n2={n_to}",
            result_override=crit_deg,
            result_units="deg",
        )

    # ── Diffraction grating angle ────────────────────────
    if re.search(r"\bdiffraction\s+grating\b.*?(?:lines|grooves)\s+per\s+(?:mm|millimeter|m\b)|"
                  r"\bgrating.*?lines.*?nm\s+light",
                  q, re.IGNORECASE | re.DOTALL):
        # Extract lines/mm and wavelength
        m_lines = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s+lines?\s+per\s+(mm|millimeter|m)",
                               q, re.IGNORECASE)
        m_wl = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*nm\b",
                            q, re.IGNORECASE)
        if m_lines and m_wl:
            lines = _num(m_lines.group(1))
            unit = m_lines.group(2).lower()
            wl_nm = _num(m_wl.group(1))
            if lines is not None and wl_nm is not None:
                # Spacing d in meters
                if unit in ("mm", "millimeter"):
                    d = 1.0 / (lines * 1000.0)  # lines per mm -> meters
                else:
                    d = 1.0 / lines  # lines per meter
                wl_m = wl_nm * 1e-9
                # First-order: d sin(theta) = m lambda, m=1
                ratio = wl_m / d
                if -1 <= ratio <= 1:
                    theta = degrees(asin(ratio))
                    return OpticsMatch(
                        tool="diffraction_grating_angle",
                        args={"spacing_m": d, "wavelength_m": wl_m, "order": 1},
                        rationale=f"d sin(θ) = m λ, d={d:.3e}m, λ={wl_m:.3e}m",
                        result_override=theta,
                        result_units="deg",
                    )

    # ── Speed of sound in air ─────────────────────────────
    if re.search(r"\bspeed\s+of\s+sound\s+in\s+air\b", q, re.IGNORECASE):
        m_T = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*(?:degrees?\s+)?Celsius\b|"
                          r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*°C\b",
                          q, re.IGNORECASE)
        T_celsius = None
        if m_T:
            T_celsius = _num(m_T.group(1) or m_T.group(2))
        if T_celsius is None:
            T_celsius = 20.0  # sensible default
        # v = 331.4 + 0.6 * T_C  (good empirical approximation)
        v = 331.4 + 0.6 * T_celsius
        return OpticsMatch(
            tool="speed_of_sound_in_ideal_gas",
            args={"temperature_c": T_celsius},
            rationale=f"v ≈ 331.4 + 0.6 T, T={T_celsius}°C",
            result_override=v,
            result_units="m/s",
        )

    return None


def execute_optics_match(match: OpticsMatch) -> tuple[object, str, str]:
    if match.result_override is not None:
        return (match.result_override, match.result_units,
                f"ANSWER: {match.result_override} {match.result_units}\n\n"
                f"Computed via optics_classifier: {match.tool} ({match.rationale})")

    from sigma_ground.mcp.tools import optics as t_opt
    try:
        if match.tool == "snells_law_refraction_angle":
            r = t_opt.snells_law_refraction_angle(**match.args)
        else:
            return None, "", ""
    except Exception as e:
        return None, "", f"<optics_classifier ERROR: {e}>"
    val = r.value if hasattr(r, "value") else None
    units = r.units if hasattr(r, "units") else ""
    answer_text = (
        f"ANSWER: {val} {units}\n\n"
        f"Computed via optics_classifier: {match.tool} ({match.rationale})"
    )
    return val, units, answer_text
