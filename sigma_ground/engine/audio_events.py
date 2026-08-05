"""audio_events — server-side audio-event annotation (engine phase E4).

The SERVER computes every physics number an audio event carries; the
browser client (radiance/web/audio.js) is a DUMB synthesizer that owns
only disclosed presentation Choices (distance law, full-scale mapping,
volume). StackSim records raw impact/bounce events in a recording-only
buffer (push_off.py's ``sim.events`` — physics never reads it); the
WorldHost drains that buffer at emit time and this module turns each raw
event into the wire dict that rides the broadcast frame.

Events are PRESENTATION derived from physics: they never enter the
command log, _full_state_floats, or any save — saves replay
hash-identical and deterministically regenerate the same events.

PANEL VERDICT carried here (E4 design panel, disclosed): impact/bounce
events are SIM-SIDE RECORDING-ONLY appends in push_off.py, because the
impact speed vimp is computed and CONSUMED inside one substep (vz becomes
e*vimp, then possibly 0.0 via the settle arm) — at any step boundary vimp
is gone, and reconstructing it host-side would either divide by the
restitution Choice (float-noise amplification) or re-derive ballistic
kinematics: a second physics implementation, forbidden. SLIDE audio is
DEFERRED (disclosed): sliding is a PERSISTING state (support set and
|vrel| > EPS_V, visible at any step boundary), so host-side observation
IS its honest mechanism — but continuous-loop synthesis is not cheap to
do honestly, so slide events ship in a later arc via boundary observation
of the support/velocity state, never via a sim-side second emission path.

Tier note: engine (tier 3) imports field.interface.acoustics (tier 1) —
downward, legal. Construct is DUCK-TYPED (.layers[].material/.mass_kg,
.bbox) — no deckard import.
"""
from __future__ import annotations

import math

from ..field.interface import acoustics
from ..field.interface.surface import MATERIALS as _SURFACE_MATERIALS

# ── Disclosed Choices (each with a reason carried into provenance) ────────

# Deckard vocabulary -> field.interface.surface.MATERIALS key. A Choice:
# the two vocabularies are different by design (Deckard names what a thing
# IS, MATERIALS names lab-characterized substances). CROSS-REFERENCE:
# deckard/sources/local.py::_ALIASES — the two tables move together (this
# map is that one plus the synthesis-relevant extras). A material that is
# ALREADY a MATERIALS key maps to itself (identity arm in _mapped_key —
# e.g. the table's dominant layer is literally "wood_oak"). Unmapped
# vocabulary (e.g. "leavened dough", "frosting") gets synth "thud": NO
# frequency claim — the same honest vocabulary-gap disclosure push_off's
# e_choice makes.
MATERIAL_MAP = {
    "stoneware": "ceramic_alumina", "earthenware": "ceramic_alumina",
    "ceramic": "ceramic_alumina", "porcelain": "ceramic_alumina",
    "glaze": "glass", "glaze (glassy)": "glass", "glass": "glass",
    "oak": "wood_oak", "wood": "wood_oak",
    "steel": "steel_mild", "mild steel": "steel_mild",
    "iron": "iron",
    "aluminium": "aluminum", "aluminum": "aluminum",
    "concrete": "concrete",
}

# Quality factor per MATERIALS key — a Choice: no internal-friction dataset
# exists on disk and acoustics.py ships no Q function; literature-ballpark
# per material family, for a SUPPORTED tabletop body (contact-damped — not
# the bell route's free-hanging tau).
Q_BY_KEY = {
    "ceramic_alumina": 300.0, "glass": 400.0, "wood_oak": 60.0,
    "steel_mild": 900.0, "iron": 700.0, "aluminum": 800.0,
    "concrete": 40.0,
}
Q_DEFAULT = 60.0                    # a mapped key with no Q row — modest ring

# Acoustic radiation efficiency of a hard impact — a Choice: literature
# range 1e-5..1e-3 for hard-body impacts, geometric mean taken; no on-disk
# dataset to derive it from.
ETA_ACOUSTIC = 1e-4

# Impacts slower than this are sub-audible policy floor — applied at
# ANNOTATE time (policy stays out of the integrator; the sim records
# everything).
MIN_IMPACT_SPEED = 0.05             # m/s

# Flood honesty: at most this many events ride one emitted frame; the
# excess is dropped WITH a count (frame key "events_dropped").
MAX_EVENTS_PER_EMIT = 32

# The staging slab bodies land on ("floor" in raw events) — used only in
# the materials disclosure; the struck BODY's voice drives synthesis (one
# voice per event, a disclosed Choice).
FLOOR_PROFILE = {"material": "floor (staging slab)", "mapped_key": "concrete"}

CHOICES = {
    "material_map": {
        "value": "deckard vocabulary -> MATERIALS key (see MATERIAL_MAP)",
        "reason": "no measured per-body acoustic dataset; Deckard names map "
                  "onto lab-characterized MATERIALS keys (cross-reference "
                  "deckard/sources/local.py::_ALIASES — the tables move "
                  "together); unmapped vocabulary gets an honest no-pitch "
                  "'thud', never a fake frequency"},
    "q_by_key": {
        "value": dict(Q_BY_KEY),
        "reason": "no internal-friction dataset on disk and acoustics.py "
                  "ships no Q function; literature-ballpark per material "
                  "for a contact-damped supported body"},
    "eta_acoustic": {
        "value": ETA_ACOUSTIC,
        "reason": "hard-impact acoustic radiation efficiency, literature "
                  "range 1e-5..1e-3, geometric mean; no on-disk dataset"},
    "min_impact_speed_m_s": {
        "value": MIN_IMPACT_SPEED,
        "reason": "sub-audible policy floor applied at annotate time — the "
                  "sim records everything, policy stays out of the "
                  "integrator"},
    "max_events_per_emit": {
        "value": MAX_EVENTS_PER_EMIT,
        "reason": "flood honesty: excess events in one emit window are "
                  "dropped WITH a count (frame key events_dropped), never "
                  "silently"},
    "one_voice_per_event": {
        "value": "the struck BODY's material voices the event",
        "reason": "a real impact excites both surfaces; synthesizing one "
                  "voice is a disclosed simplification (the surface "
                  "material is still disclosed in the event)"},
    "size_from_bbox": {
        "value": "max horizontal bbox extent",
        "reason": "the circumference-setting dimension of a flat-landing "
                  "body drives ring_frequency's diameter argument"},
}


def _mapped_key(material: str) -> "str | None":
    """MATERIAL_MAP arm first, then the identity arm for materials that
    are already MATERIALS keys (e.g. 'wood_oak'); None = honest gap."""
    key = MATERIAL_MAP.get(material)
    if key is not None:
        return key
    if material in _SURFACE_MATERIALS:
        return material
    return None


def build_profiles(built) -> dict:
    """{body label: synthesis profile} from a [(name, Construct)] list.

    Duck-typed: needs .layers[].material/.mass_kg, .bbox, .mass_kg only.
    Per body: material = dominant layer BY MASS (a Deduction, disclosed);
    size_m = max horizontal bbox extent (disclosed SIZE Choice);
    frequency from acoustics.ring_frequency (FIRST_PRINCIPLES v_L/(pi*d))
    when the material maps; honest "thud" (no pitch claim) otherwise.
    """
    profiles = {}
    for name, c in built:
        layers = list(getattr(c, "layers", []) or [])
        if layers:
            material = max(layers, key=lambda L: L.mass_kg).material
        else:
            material = "unknown"
        (x0, x1), (y0, y1), _zz = c.bbox
        size_m = max(x1 - x0, y1 - y0)
        key = _mapped_key(material)
        if key is not None and size_m > 0:
            freq = acoustics.ring_frequency(key, size_m)
            q = Q_BY_KEY.get(key, Q_DEFAULT)
            prof = {
                "material": material, "mapped_key": key,
                "size_m": size_m, "mass_kg": float(c.mass_kg),
                "synth": "ring", "frequency_hz": freq, "q": q,
                "provenance": (
                    f"ring_frequency({key!r}, d={size_m:.3f} m) — "
                    f"FIRST_PRINCIPLES v_L/(π·d); Q and efficiency are "
                    f"disclosed Choices"),
            }
        else:
            prof = {
                "material": material, "mapped_key": None,
                "size_m": size_m, "mass_kg": float(c.mass_kg),
                "synth": "thud", "frequency_hz": None, "q": None,
                "provenance": (
                    f"unmapped material {material!r} — no pitch claim; "
                    f"generic damped noise burst (honest fallback)"),
            }
        profiles[name] = prof
    return profiles


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and math.isfinite(x)


def annotate(raw: dict, names, masses, profiles, step: int) -> "dict | None":
    """One raw sim event -> the wire event dict, or None (dropped).

    PURE: same input, same output — no state, no clock. Drops (None):
    * impact_speed below MIN_IMPACT_SPEED (sub-audible policy floor);
    * any non-finite numeric field (guard — a physics NaN would already
      have failed the hash gates loudly; this keeps frames strict-JSON).

    energy_j = 0.5 * m * vimp**2 is a Deduction (server-side mass);
    acoustic_energy_j applies the ETA_ACOUSTIC Choice. ``step`` is the
    EMIT-boundary step (raw events carry the exact sim time ``t``; frames
    only leave at boundaries — disclosed).
    """
    try:
        vimp = raw["impact_speed_m_s"]
        body_i = raw["body"]
        label = names[body_i]
        surface = raw["surface"]
        x, z, t = raw["x"], raw["z"], raw["t"]
        e = raw.get("e")
    except (KeyError, IndexError, TypeError):
        return None
    if not (_finite(vimp) and _finite(x) and _finite(z) and _finite(t)):
        return None
    if vimp < MIN_IMPACT_SPEED:
        return None
    mass = (masses or {}).get(label)
    if not _finite(mass):
        return None
    energy = 0.5 * mass * vimp ** 2
    if not _finite(energy):
        return None
    surf_label = surface if isinstance(surface, str) else names[surface]
    prof = (profiles or {}).get(label)
    if prof is None:
        # a bare host (no profiles) — honest all-thud fallback
        prof = {"material": "unknown", "mapped_key": None, "synth": "thud",
                "frequency_hz": None, "q": None,
                "provenance": "no audio profiles on this host — no pitch "
                              "claim (honest fallback)"}
    if surf_label == "floor":
        surf_material = FLOOR_PROFILE["material"]
    else:
        sp = (profiles or {}).get(surf_label)
        surf_material = sp["material"] if sp else "unknown"
    return {
        "kind": raw.get("kind", "impact"),
        "body": label, "surface": surf_label,
        "materials": {"body": prof["material"], "surface": surf_material},
        "impact_speed_m_s": vimp,
        "restitution_e": e if _finite(e) else None,
        "energy_j": energy,                       # 0.5*m*v² — Deduction
        "acoustic_energy_j": ETA_ACOUSTIC * energy,
        "position": [x, 0.0, z],
        "step": int(step), "t": t,
        "synth": {"kind": prof["synth"],
                  "frequency_hz": prof["frequency_hz"],
                  "q": prof["q"],
                  "provenance": prof["provenance"]},
    }


def audio_disclosure(profiles) -> dict:
    """The ready-message audio block: per-body synthesis profiles plus
    every tuning Choice with its reason — full disclosure up front."""
    return {"profiles": {k: dict(v) for k, v in (profiles or {}).items()},
            "choices": {k: dict(v) for k, v in CHOICES.items()}}


__all__ = ["MATERIAL_MAP", "Q_BY_KEY", "ETA_ACOUSTIC", "MIN_IMPACT_SPEED",
           "MAX_EVENTS_PER_EMIT", "FLOOR_PROFILE", "CHOICES",
           "build_profiles", "annotate", "audio_disclosure"]
