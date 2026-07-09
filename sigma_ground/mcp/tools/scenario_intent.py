"""Mentat-side scenario-intent enrichment — the Mentat→Materia seam (Q4).

Mentat used to hand the Materia spine only a bare scenario string, and the
playground dropped the object identity entirely (it was tacked on as an
after-note, so Materia never knew *which* object). The spine's
translator/front_door can do far better when it knows WHY the object exists:
a tip-over test needs accurate CoM + friction; a still render needs accurate
optics. Same object, different physics emphasis.

This module is Mentat's side of that contract. It:
  • detects a COARSE intent from the request text, and
  • assembles a forward-compatible ScenarioManifest that travels with the call.

It does NOT own the manifest — the orchestration spine (manifest.py /
scenarios.py / translator.py / front_door.py, in the Materia track) is
authoritative and is not touched here. The EMPHASIS map below is a PROVISIONAL
fallback so the "why" shows up in answers today; the instant the spine attaches
an authoritative emphasis to its spec/result, ``read_emphasis()`` prefers it and
this table is ignored.
"""
from __future__ import annotations

from typing import Any

# Coarse intent taxonomy (a small mirror of the spine's verb space). Keyword →
# canonical intent. Order of CHECK (below) is most-specific-first so e.g.
# "tip the full cup over" reads as tip, not fill.
_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tip":     ("tip over", "tip it over", "topple", "tip-over", "tipover",
                "knock over", "fall over", "overturn", "tipping", "stability",
                "how stable", "wobble"),
    "shatter": ("shatter", "smash", "fracture", "crack", "shatters",
                "break when", "breaks when", "drop and break"),
    "fill":    ("fill", "filled", "fill with", "fill it", "filled with",
                "fill the"),
    "pour":    ("pour", "pour out", "spill", "decant", "empty it"),
    "float":   ("float", "sink", "buoyan", "submerg", "half-sunk", "half sunk",
                "floats", "will it float", "waterline"),
    "slide":   ("slide", "slides", "skid", "down a ramp", "down an incline",
                "on an incline", "slip"),
    "collide": ("collide", "collision", "crash into", "hit each other",
                "strike each other", "smash into"),
    "still":   ("render", "still render", "what does it look like",
                "how does it look", "photoreal", "appearance", "picture of",
                "image of", "show me a picture", "what color"),
    "drop":    ("drop", "dropped", "drops", "free fall", "free-fall",
                "falls from", "fall from", "let go", "release from",
                "throw it down"),
}

# PROVISIONAL emphasis: intent → which properties each subsystem must compute to
# HIGH accuracy (and, implicitly, which it may approximate). Authoritative copy
# belongs in the spine's manifest.py; this is only a stopgap (see module doc).
_PROVISIONAL_EMPHASIS: dict[str, dict[str, Any]] = {
    "tip":     {"physics": ["center_of_mass", "friction_static",
                            "friction_kinetic", "base_support_polygon"],
                "shape": ["base", "legs", "footprint"], "render": "low"},
    "drop":    {"physics": ["mass", "restitution", "impact_velocity",
                            "contact_stiffness"],
                "shape": ["overall", "contact_face"], "render": "low"},
    "shatter": {"physics": ["fracture_toughness", "youngs_modulus",
                            "stress_concentration", "brittleness"],
                "shape": ["thickness", "seams", "weak_points"], "render": "low"},
    "fill":    {"physics": ["fluid_density", "fluid_viscosity",
                            "surface_tension", "container_volume",
                            "free_surface"],
                "shape": ["interior_volume", "rim"], "render": "medium"},
    "pour":    {"physics": ["fluid_density", "fluid_viscosity",
                            "surface_tension", "flow_rate"],
                "shape": ["spout", "rim", "interior_volume"], "render": "medium"},
    "float":   {"physics": ["density", "fluid_density", "buoyancy", "waterline"],
                "shape": ["displaced_volume", "hull"], "render": "medium"},
    "slide":   {"physics": ["friction_kinetic", "normal_force", "incline_angle"],
                "shape": ["contact_face"], "render": "low"},
    "collide": {"physics": ["mass", "restitution", "momentum",
                            "relative_velocity"],
                "shape": ["contact_faces"], "render": "low"},
    # render-only: optics (albedo, IOR, roughness, microfacet); dynamics ignored.
    "still":   {"physics": [], "shape": ["full_surface_detail"],
                "render": "high"},
}
_DEFAULT_EMPHASIS: dict[str, Any] = {
    "physics": ["mass", "dimensions"], "shape": ["overall"], "render": "medium"}

# Check order: specific intents before the catch-all "drop"/"still".
_CHECK_ORDER = ("tip", "shatter", "fill", "pour", "float", "slide", "collide",
                "still", "drop")


def detect_intent(text: str) -> str | None:
    """Coarse intent from the request text (keyword match; None if unclear)."""
    t = (text or "").lower()
    # "tip" needs co-occurrence, not a fixed bigram: "tip the chair over",
    # "knock it over" have words between the verb and the particle.
    if (("tip" in t and "over" in t) or ("knock" in t and "over" in t)
            or "topple" in t or "overturn" in t or "fall over" in t
            or "falls over" in t or "tipping" in t or "wobble" in t
            or "how stable" in t or "stability" in t):
        return "tip"
    for intent in _CHECK_ORDER:
        if intent == "tip":
            continue
        if any(kw in t for kw in _INTENT_KEYWORDS[intent]):
            return intent
    return None


def emphasis_for(intent: str | None) -> dict[str, Any]:
    """Provisional per-subsystem emphasis for an intent (spine overrides this)."""
    return dict(_PROVISIONAL_EMPHASIS.get(intent or "", _DEFAULT_EMPHASIS))


def build_manifest(scenario: str, object_handle: str | None = None,
                   intent: str | None = None) -> dict[str, Any]:
    """Assemble the forward-compatible ScenarioManifest Mentat hands the spine.

    Carries the object identity (a Deckard handle — shape stays Deckard's), the
    coarse intent, the FULL request text (so the spine's translator has
    everything), and a provisional emphasis. Once the spine accepts a structured
    intake, this dict is exactly what gets forwarded; until then it rides in the
    ToolResult so the "why" is visible.
    """
    intent = intent or detect_intent(scenario)
    return {
        "object": object_handle,
        "intent": intent,
        "intent_text": scenario,
        "emphasis": emphasis_for(intent),
        "emphasis_source": "mentat_provisional",
    }


def read_emphasis(*objs: Any) -> dict[str, Any] | None:
    """Forward-compatible: if the spine ever attaches an authoritative
    ``emphasis`` (or a ``manifest`` carrying one) to its spec/result, return it.
    Returns None until the spine emits one, so callers fall back to provisional.
    """
    for obj in objs:
        if obj is None:
            continue
        emph = getattr(obj, "emphasis", None)
        if isinstance(emph, dict):
            return emph
        man = getattr(obj, "manifest", None)
        if isinstance(man, dict) and isinstance(man.get("emphasis"), dict):
            return man["emphasis"]
    return None


def why_sentence(manifest: dict[str, Any]) -> str:
    """One-line 'why the physics was computed this way', for ToolResult notes."""
    intent = manifest.get("intent")
    if not intent:
        return ""
    emph = manifest.get("emphasis", {}) or {}
    phys = ", ".join(emph.get("physics", []) or [])
    obj = manifest.get("object")
    obj_s = f" for the {obj}" if obj else ""
    if phys:
        return f"computed {phys}{obj_s} because the intent is '{intent}'."
    return (f"emphasis on optics (render={emph.get('render', 'high')}){obj_s} "
            f"because the intent is '{intent}'.")
