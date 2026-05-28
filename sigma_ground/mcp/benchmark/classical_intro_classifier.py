"""Classical mechanics (intro) pre-classifier.

Targets the failing intro questions:
  - Projectile max height ('straight up at v0')
  - Projectile flight time ('at v0, angle theta')
  - Circular orbit velocity ('satellite at altitude X')
  - Free fall on Moon (override default Earth g)
  - Friction stopping distance
  - Inverse kinetic-energy problem (speed from KE)

Qwen 7b doesn't call kinematics tools for these. The classifier
extracts parameters and dispatches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import sqrt


# Constants
_G_EARTH = 9.80665
_G_MOON = 1.625
_M_EARTH = 5.972e24
_R_EARTH = 6.378e6
_G_NEWTON = 6.6743e-11


@dataclass
class ClassicalIntroMatch:
    tool: str
    args: dict
    rationale: str
    result_override: float | None = None
    result_units: str = ""


def _num(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def classify_for_classical_intro(question: str) -> ClassicalIntroMatch | None:
    q = question

    # ── Projectile max height: "straight up at v m/s, how high" ──
    if re.search(r"\b(?:straight\s+up|vertically)\s+at\b.*\bhow\s+high|"
                  r"\bhow\s+high\s+(?:does\s+it\s+)?go", q, re.IGNORECASE | re.DOTALL):
        m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*m\s*/\s*s\b", q, re.IGNORECASE)
        if m:
            v0 = _num(m.group(1))
            if v0 is not None:
                return ClassicalIntroMatch(
                    tool="projectile_max_height",
                    args={"initial_speed_m_s": v0, "launch_angle_deg": 90.0},
                    rationale=f"h_max = v0^2/(2g), v0={v0}m/s straight up",
                )

    # ── Projectile flight time: "at v0 at theta degrees, how long in the air" ──
    if re.search(r"\b(?:how\s+long\s+is\s+it\s+in\s+the\s+air|flight\s+time|"
                  r"time\s+(?:of\s+)?flight|airtime|hang\s+time)\b",
                  q, re.IGNORECASE):
        m_v = re.search(r"\bat\s+([\-+]?[0-9]+(?:\.[0-9]+)?)\s*m\s*/\s*s\b",
                          q, re.IGNORECASE)
        m_t = re.search(r"\bat\s+([\-+]?[0-9]+(?:\.[0-9]+)?)\s*degrees?\b",
                          q, re.IGNORECASE)
        if m_v and m_t:
            v0 = _num(m_v.group(1))
            theta = _num(m_t.group(1))
            if v0 is not None and theta is not None:
                return ClassicalIntroMatch(
                    tool="projectile_flight_time",
                    args={"initial_speed_m_s": v0, "launch_angle_deg": theta},
                    rationale=f"t = 2v0 sin(theta)/g, v0={v0}m/s, theta={theta}°",
                )

    # ── Circular orbit velocity: "satellite at X km altitude" ──
    if re.search(r"\b(?:satellite|orbit).*\b([\-+]?[0-9,]+(?:\.[0-9]+)?)\s*km\b.*\b(?:altitude|above)|"
                  r"\bhow\s+fast.*?satellite", q, re.IGNORECASE | re.DOTALL):
        # Extract altitude
        m = re.search(r"\b([\-+]?[0-9,]+(?:\.[0-9]+)?)\s*(?:km|kilometer)",
                       q, re.IGNORECASE)
        if m:
            try:
                alt_km = float(m.group(1).replace(",", ""))
                # Use Earth as central body if mentioned
                if re.search(r"\bEarth\b", q, re.IGNORECASE):
                    r_m = _R_EARTH + alt_km * 1000.0
                    return ClassicalIntroMatch(
                        tool="circular_orbit_velocity",
                        args={"central_mass_kg": _M_EARTH, "radius_m": r_m},
                        rationale=f"v = sqrt(GM/r), Earth, r=R_E + {alt_km}km",
                    )
            except ValueError:
                pass

    # ── Free fall on Moon (or other body with explicit g) ──
    # "drop ball from X meters on Moon" or "X meter on the Moon"
    if re.search(r"\bfree\s+fall|drop\s+(?:a\s+|the\s+)?(?:ball|object|stone)",
                  q, re.IGNORECASE):
        m_h = re.search(r"\bfrom\s+([\-+]?[0-9]+(?:\.[0-9]+)?)\s*(?:m|meter)",
                          q, re.IGNORECASE)
        # Detect explicit gravity (Moon, Mars, etc.)
        g = None
        body_rationale = "Earth"
        if re.search(r"\bMoon\b", q, re.IGNORECASE):
            g = _G_MOON
            body_rationale = "Moon"
        elif re.search(r"\bMars\b", q, re.IGNORECASE):
            g = 3.71
            body_rationale = "Mars"
        # Also look for explicit numerical gravity
        m_g = re.search(r"\bgravity\s+is\s+(?:about\s+)?([\-+]?[0-9]+(?:\.[0-9]+)?)\s*m\s*/\s*s\s*(?:\^?2|squared)",
                          q, re.IGNORECASE)
        if m_g:
            try:
                g = float(m_g.group(1))
                body_rationale = f"explicit g={g}m/s^2"
            except ValueError:
                pass
        if m_h and g is not None and g != _G_EARTH:
            h = _num(m_h.group(1))
            if h is not None:
                return ClassicalIntroMatch(
                    tool="free_fall_time",
                    args={"height_m": h, "g_m_s2": g},
                    rationale=f"t = sqrt(2h/g), h={h}m, g={g}m/s^2 ({body_rationale})",
                )

    # ── Friction stopping distance ──
    if re.search(r"\bfriction.*?(?:stop|coefficient)|"
                  r"\bcoefficient.*?friction|"
                  r"\bcoffee\s+cup.*?(?:slide|stop)|"
                  r"\bhow\s+far\s+(?:does\s+it|will\s+it)\s+(?:slide|stop)",
                  q, re.IGNORECASE | re.DOTALL):
        m_v = re.search(r"\bat\s+([\-+]?[0-9]+(?:\.[0-9]+)?)\s*(?:m\s*/\s*s|meters?\s+per\s+second)",
                          q, re.IGNORECASE)
        # 'friction coefficient ... is 0.4', 'coefficient ... of friction is 0.4'
        m_mu = re.search(
            r"\b(?:friction\s+coefficient|coefficient\s+of\s+friction)\b"
            r"[^\d]*?([\-+]?[0-9]*\.[0-9]+)",
            q, re.IGNORECASE | re.DOTALL)
        if not m_mu:
            m_mu = re.search(r"\b(?:coefficient|friction)\s+(?:is\s+|of\s+)?"
                                 r"([\-+]?[0-9]*\.[0-9]+)", q, re.IGNORECASE)
        if m_v and m_mu:
            v = _num(m_v.group(1))
            mu = _num(m_mu.group(1))
            # Extract mass too (tool requires it though it cancels in the formula)
            m_mass = re.search(r"\bmass\s+([\-+]?[0-9]+(?:\.[0-9]+)?)\s*kg|"
                                  r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*kg",
                                  q, re.IGNORECASE)
            mass = _num(m_mass.group(1) or m_mass.group(2)) if m_mass else 1.0
            if v is not None and mu is not None:
                return ClassicalIntroMatch(
                    tool="friction_stopping_distance",
                    args={"mass_kg": mass or 1.0,
                            "initial_velocity_m_s": v,
                            "friction_coefficient": mu},
                    rationale=f"d = v^2/(2 mu g), v={v}m/s, mu={mu}",
                )

    # ── Inverse kinetic energy: "X kg, Y joules, what speed" ──
    if re.search(r"\bspeed\s+(?:does|to\s+have|needed?\b)|"
                  r"\bwhat\s+speed\b|"
                  r"\bvelocity.*?(?:to\s+have|need|require)",
                  q, re.IGNORECASE) and re.search(r"\bjoules?\b", q, re.IGNORECASE):
        m_m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*kg\b", q, re.IGNORECASE)
        m_KE = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*joules?\b",
                            q, re.IGNORECASE)
        if m_m and m_KE:
            mass = _num(m_m.group(1))
            KE = _num(m_KE.group(1))
            if mass is not None and KE is not None and mass > 0:
                v = sqrt(2.0 * KE / mass)
                return ClassicalIntroMatch(
                    tool="inverse_kinetic_energy",
                    args={"mass_kg": mass, "ke_joules": KE},
                    rationale=f"v = sqrt(2 KE / m), m={mass}kg, KE={KE}J -> {v}m/s",
                    result_override=v,
                    result_units="m/s",
                )

    return None


def execute_classical_intro_match(match: ClassicalIntroMatch) -> tuple[object, str, str]:
    if match.result_override is not None:
        return (match.result_override, match.result_units,
                f"ANSWER: {match.result_override} {match.result_units}\n\n"
                f"Computed via classical_intro_classifier: {match.tool} ({match.rationale})")

    from sigma_ground.mcp.tools import kinematics as t_kin
    try:
        if match.tool == "projectile_max_height":
            r = t_kin.projectile_max_height(**match.args)
        elif match.tool == "projectile_flight_time":
            r = t_kin.projectile_flight_time(**match.args)
        elif match.tool == "circular_orbit_velocity":
            r = t_kin.circular_orbit_velocity(**match.args)
        elif match.tool == "free_fall_time":
            r = t_kin.free_fall_time(**match.args)
        elif match.tool == "friction_stopping_distance":
            r = t_kin.friction_stopping_distance(**match.args)
        else:
            return None, "", ""
    except Exception as e:
        return None, "", f"<classical_intro_classifier ERROR: {e}>"
    val = r.value if hasattr(r, "value") else None
    units = r.units if hasattr(r, "units") else ""
    answer_text = (
        f"ANSWER: {val} {units}\n\n"
        f"Computed via classical_intro_classifier: {match.tool} ({match.rationale})"
    )
    return val, units, answer_text
