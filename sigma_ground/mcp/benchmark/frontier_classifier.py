"""Router for frontier black-hole-thermodynamics / holography questions.

Same architecture as the other classifiers: regex → (tool, slots) →
deterministic dispatch. The frontier tools are reached only here, kept off
the LLM's flat list. Handles the fringe questions about Bekenstein-Hawking
entropy, the bubble-pop thread count, the baryon-vs-horizon crossover, and
gravitational binding energy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_M_SUN = 1.98892e30


@dataclass
class FrontierMatch:
    tool: str
    args: dict
    rationale: str
    field: str | None = None   # pick a scalar from a dict-valued result


def _num(s):
    try:
        return float(str(s).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _extract_mass_kg(q: str) -> float | None:
    m = re.search(r"\b([\-+]?[0-9]*\.?[0-9]+)\s*solar\s+mass", q, re.IGNORECASE)
    if m:
        return _num(m.group(1)) * _M_SUN
    if re.search(r"\bsolar[-\s]mass\b|\bof\s+the\s+sun\b|\b1\s+M_?sun\b", q, re.IGNORECASE):
        return _M_SUN
    # other body masses (allow "Earth-mass" / "Earth mass" / "mass of the Earth")
    if re.search(r"\bearth[-\s]mass\b|\bmass\s+of\s+the\s+earth\b", q, re.IGNORECASE):
        return 5.972e24
    if re.search(r"\bjupiter[-\s]mass\b|\bmass\s+of\s+jupiter\b", q, re.IGNORECASE):
        return 1.898e27
    if re.search(r"\bmoon[-\s]mass\b|\blunar\s+mass\b|\bmass\s+of\s+the\s+moon\b", q, re.IGNORECASE):
        return 7.342e22
    m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][\-+]?[0-9]+)?)\s*kg", q, re.IGNORECASE)
    if m:
        return _num(m.group(1))
    return None


def _extract_radius_m(q: str) -> float | None:
    # Planck length/bubble (allow "Planck-length" hyphen or space)
    if re.search(r"\bplanck[\s\-](length|radius|scale|bubble)\b", q, re.IGNORECASE):
        import sigma_ground.field.constants as C
        return C.L_PLANCK
    # number, optional hyphen/space, unit (handles "1-kilometer", "1 km", "1km")
    for pat, scale in [(r"\b([\-+]?[0-9.]+)\s*[-\s]?\s*(?:fm|femtomet)", 1e-15),
                         (r"\b([\-+]?[0-9.]+)\s*[-\s]?\s*(?:nm|nanomet)", 1e-9),
                         (r"\b([\-+]?[0-9.]+)\s*[-\s]?\s*(?:angstrom|Å)", 1e-10),
                         (r"\b([\-+]?[0-9.]+)\s*[-\s]?\s*(?:km|kilomet)", 1e3),
                         (r"\b([\-+]?[0-9.]+(?:[eE][\-+]?[0-9]+)?)\s*[-\s]?\s*m(?:et(?:er|re)s?)?\b", 1.0)]:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            v = _num(m.group(1))
            if v is not None:
                return v * scale
    return None


def classify_for_frontier(question: str) -> FrontierMatch | None:
    q = question

    # ── entanglement as a channel (no-signaling guardrail) ──
    # Catches "can entangled particles communicate faster than light?" and
    # routes to the canonical NO before the LLM can mis-answer. The tool
    # tailors its verdict from the full question (communication→NO,
    # key→YES/QKD, CHSH→2√2), so we just pass the question as scenario.
    if re.search(r"\bentangl\w+|\bepr\b|\bbell['’]?s?\s+(test|pair|state|"
                  r"inequalit|theorem|parameter|value)|\bchsh\b|\btsirelson\b",
                  q, re.IGNORECASE):
        is_chsh = bool(re.search(r"\b(chsh|tsirelson|bell\s+(value|inequality|"
                                   r"parameter)|maximum\s+(value|violation)|"
                                   r"correlation\s+bound)\b", q, re.IGNORECASE))
        is_comm = bool(re.search(r"\b(communicat|send|transmit|signal|mess\w+|"
                                   r"faster[\s-]than[\s-]light|ftl|instantaneous|"
                                   r"information|talk)\b", q, re.IGNORECASE))
        is_key = bool(re.search(r"\b(key|qkd|secret|crypto\w*|encrypt\w*|secure)\b",
                                  q, re.IGNORECASE))
        if is_chsh:
            return FrontierMatch("entanglement_channel", {"scenario": q},
                                   "Tsirelson bound 2√2", field="primary")
        if is_comm or is_key:
            return FrontierMatch("entanglement_channel", {"scenario": q},
                                   "no-communication theorem", field="verdict")

    # ── entanglements to pop a bubble ──
    if re.search(r"\b(pop|saturat\w+|collaps\w+).{0,40}bubble|"
                  r"\bbubble.{0,40}(pop|saturat|collaps)|"
                  r"\bentanglement\w*\s+to\s+(pop|fill|saturate)|"
                  r"\bthreads?\s+to\s+(pop|saturate|fill)", q, re.IGNORECASE):
        R = _extract_radius_m(q)
        if R is not None:
            return FrontierMatch("entanglements_to_pop_bubble",
                                   {"radius_m": R}, f"N=πR²/L_p², R={R:g} m")

    # ── holographic matching mass (no args) ──
    if re.search(r"\b(matching|crossover)\s+mass|"
                  r"\bmass\s+where\s+(baryon|quark).{0,40}(equal|match|disc)|"
                  r"\b(at\s+what|what)\s+(black[\s\-]hole\s+)?mass\b.{0,50}baryon"
                  r".{0,40}(equal|match|same)|"
                  r"\bbaryon\w*.{0,20}(equal|equals|match|same).{0,30}(disc|horizon|pixel)",
                  q, re.IGNORECASE):
        return FrontierMatch("holographic_matching_mass", {},
                               "M=ħc/(4πGm_p)")

    # ── baryon vs disc (needs a mass) ──
    if re.search(r"\b(baryon|quark)\w*\s+(vs|versus|compared|relative).{0,20}(disc|horizon|pixel)|"
                  r"\b(does|do)\s+the\s+(baryon|matter|quark).{0,40}(overflow|fit|exceed|room)|"
                  r"\b(overflow|room\s+to\s+spare)\b.{0,30}horizon|"
                  r"\bregime\b.{0,30}(black hole|horizon)", q, re.IGNORECASE):
        M = _extract_mass_kg(q)
        if M is not None:
            # qualitative ("does it overflow / have room?") → the regime string;
            # quantitative ("ratio of ...") → the numeric ratio.
            qualitative = bool(re.search(r"\b(overflow|fit|exceed|room|regime|"
                                          r"have\s+room|enough\s+room)\b", q, re.IGNORECASE))
            field = "regime" if qualitative else "ratio_disc_over_baryons"
            return FrontierMatch("baryon_vs_disc", {"mass_kg": M},
                                   "baryon count vs horizon pixels", field=field)

    # ── Bekenstein-Hawking entropy / thread count ──
    if re.search(r"\b(bekenstein|hawking)\s+entropy|"
                  r"\bentropy\s+of\s+a?\s*black\s+hole|"
                  r"\b(how\s+many|number\s+of)\s+(threads|pixels|bits|entanglement)"
                  r".{0,30}(horizon|black hole)|"
                  r"\bholographic\s+(thread|pixel|information)\s+(count|capacity)",
                  q, re.IGNORECASE):
        M = _extract_mass_kg(q)
        if M is not None:
            return FrontierMatch("bekenstein_hawking_entropy", {"mass_kg": M},
                                   "S=A/4L_p²", field="entropy_k_B")

    # ── gravitational binding energy of a sphere ──
    if re.search(r"\b(gravitational\s+)?binding\s+energy\b.{0,40}(sphere|star|planet|uniform)|"
                  r"\benergy\s+to\s+(assemble|disperse)\b", q, re.IGNORECASE):
        M = _extract_mass_kg(q)
        R = _extract_radius_m(q)
        if M is not None and R is not None:
            return FrontierMatch("gravitational_binding_energy",
                                   {"mass_kg": M, "radius_m": R}, "U=(3/5)GM²/R")

    # ── Unruh temperature ──
    if re.search(r"\bunruh\s+temperature\b", q, re.IGNORECASE):
        m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][\-+]?[0-9]+)?)\s*m\s*/\s*s\s*(?:\^?2|squared)",
                        q, re.IGNORECASE)
        if m:
            return FrontierMatch("unruh_temperature",
                                   {"acceleration_m_s2": _num(m.group(1))},
                                   "T=ħa/(2πck_B)")

    return None


def execute_frontier_match(match: FrontierMatch) -> tuple[object, str, str]:
    from sigma_ground.mcp.tools import frontier as t
    fn = getattr(t, match.tool, None)
    if fn is None:
        return None, "", ""
    try:
        r = fn(**match.args)
    except Exception as e:
        return None, "", f"<frontier_classifier ERROR: {e}>"
    val = r.value
    units = r.units or ""
    if isinstance(val, dict) and match.field:
        val = val.get(match.field)
    answer = (f"ANSWER: {val} {units}\n\n"
              f"Computed via frontier_classifier: {match.tool} ({match.rationale})")
    return val, units, answer


FRONTIER_HIDDEN = {
    "bekenstein_hawking_entropy", "entanglements_to_pop_bubble",
    "holographic_matching_mass", "baryon_vs_disc",
    "gravitational_binding_energy", "unruh_temperature",
    "entanglement_channel",
}
