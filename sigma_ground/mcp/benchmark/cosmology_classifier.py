"""Cosmology question pre-classifier.

The 8 cosmology questions in the corpus follow a tight set of patterns:
  - "Hubble radius" / "Hubble sphere"   -> hubble_radius()
  - "Hubble time" / "age of universe"   -> age_of_universe()  (with unit pivot)
  - "critical density"                  -> critical_density()
  - "MOND regime" / "Newtonian regime"  -> mond_regime_classifier(a)
  - "MOND a_0" / "MOND acceleration"    -> mond_a0_constant()
  - "what is eta" / "eta value"         -> eta_value_report()

Qwen 7b picks rydberg_hydrogen_wavelength for the Newtonian-regime
question and element_atomic_data for eta. The classifier dispatches
the right tool from question keywords.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class CosmologyMatch:
    tool: str
    arg: float | None     # for mond_regime_classifier (the acceleration)
    rationale: str
    matched_pattern: str


def classify_for_cosmology(question: str) -> CosmologyMatch | None:
    q = question

    # MOND regime classifier needs an acceleration in m/s^2 to dispatch.
    # Match phrases like "1e-10 m/s squared" or "9.8 m/s squared".
    mond_regime = re.search(
        r"(?:\bis\s+that|which\s+regime|\b(?:Newtonian|MOND)\s+regime)",
        q, re.IGNORECASE)
    if mond_regime:
        # Extract a number followed by m/s^2 or m/s squared
        m = re.search(
            r"\b([\-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][\-+]?[0-9]+)?)\s*"
            r"(?:m\s*/\s*s\s*(?:\^?2|squared)|m\.?s\^?-2)\b",
            q, re.IGNORECASE)
        if m:
            try:
                a = float(m.group(1))
                return CosmologyMatch(
                    tool="mond_regime_classifier", arg=a,
                    rationale=f"acceleration {a} m/s^2",
                    matched_pattern="mond_regime_with_acceleration",
                )
            except ValueError:
                pass

    # MOND a_0 constant
    if re.search(
        r"\bMOND\s+(?:a_?0|acceleration\s+(?:constant|scale))\b|"
        r"\bMilgrom['s]+\s+constant\b",
        q, re.IGNORECASE,
    ):
        return CosmologyMatch(
            tool="mond_a0_constant", arg=None,
            rationale="MOND characteristic acceleration",
            matched_pattern="mond_a0",
        )

    # Hubble radius
    if re.search(
        r"\bHubble\s+(?:radius|sphere|length|distance)\b",
        q, re.IGNORECASE,
    ):
        return CosmologyMatch(
            tool="hubble_radius", arg=None,
            rationale="Hubble radius c/H_0",
            matched_pattern="hubble_radius",
        )

    # Hubble time / age of universe
    if re.search(
        r"\bHubble\s+(?:time|age)\b|"
        r"\bage\s+of\s+(?:the\s+)?universe\b|"
        r"\bhow\s+old\s+is\s+(?:the\s+)?universe\b",
        q, re.IGNORECASE,
    ):
        return CosmologyMatch(
            tool="age_of_universe", arg=None,
            rationale="true flat-LambdaCDM age (Planck 2018)",
            matched_pattern="age_of_universe",
        )

    # Critical density
    if re.search(
        r"\bcritical\s+density\b|\brho_?crit\b|\bOmega\s*=\s*1\b",
        q, re.IGNORECASE,
    ):
        return CosmologyMatch(
            tool="critical_density", arg=None,
            rationale="rho_crit = 3 H^2 / (8 pi G)",
            matched_pattern="critical_density",
        )

    # eta value report
    if re.search(
        r"\bwhat(?:'s|\s+is)?\s+eta\b|"
        r"\beta\s+value\b|"
        r"\beta\s+in\s+(?:the\s+)?sigma.?ground\b|"
        r"\bsigma.?ground.*\beta\b",
        q, re.IGNORECASE,
    ):
        return CosmologyMatch(
            tool="eta_value_report", arg=None,
            rationale="SSBM eta parameter",
            matched_pattern="eta_value",
        )

    return None


def execute_cosmology_match(match: CosmologyMatch) -> tuple[object, str, str]:
    """Run the dispatched tool. Returns (value, units, answer_text)."""
    from sigma_ground.mcp.tools import cosmology as t_cos
    try:
        if match.tool == "hubble_radius":
            r = t_cos.hubble_radius()
        elif match.tool == "age_of_universe":
            r = t_cos.age_of_universe()
        elif match.tool == "critical_density":
            r = t_cos.critical_density()
        elif match.tool == "mond_regime_classifier":
            if match.arg is None:
                return None, "", ""
            r = t_cos.mond_regime_classifier(match.arg)
        elif match.tool == "mond_a0_constant":
            r = t_cos.mond_a0_constant()
        elif match.tool == "eta_value_report":
            r = t_cos.eta_value_report()
        else:
            return None, "", ""
    except Exception as e:
        return None, "", f"<cosmology_classifier ERROR: {e}>"
    val = r.value if hasattr(r, "value") else None
    units = r.units if hasattr(r, "units") else ""
    answer_text = (
        f"ANSWER: {val} {units}\n\n"
        f"Computed via cosmology_classifier: tool={match.tool} ({match.rationale})"
    )
    return val, units, answer_text
