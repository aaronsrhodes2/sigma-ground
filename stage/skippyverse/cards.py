"""Object card v1 (misc/SKIPPYVERSE_OBJECTS_DESIGN.md section 4): schema, validation, the LICENCE GATE, and the converter to the
holodeck planner's footprint fields.

Units: metres and kilograms in the card; the planner works in Hammer units (1 HU = 0.01905 m). +Z up, front faces +X by the card's
own convention, origin on the floor under the footprint centre. `w` is the extent across the front, `d` along the front (the same
meaning as stage/gmod/holodeck/build_catalog.py).

Nothing here computes physics. Mass numbers come from the existing fidelity layer or are labelled `guessed`.
"""
from __future__ import annotations

import math
import re

CARD_V = 1
HU_M = 0.01905                                    # one Hammer unit in metres (stage/gmod/holodeck/build_catalog.py)

BASIS_ORDER = ["guessed", "estimated", "sourced", "measured"]       # weakest first
BASIS = frozenset(BASIS_ORDER)
CLASSES = frozenset({"curated", "found", "generated", "deckard", "placeholder"})
FRONT_SOURCES = frozenset({"curated-by-eye", "heuristic", "unknown"})
MOBILITY = frozenset({"static", "dynamic"})
COLLISION_MODES = frozenset({"box", "compound", "hull", "none"})

# --- the licence gate ---------------------------------------------------------------------------------------------------------
ALLOWED_LICENCES = frozenset({"CC0", "CC-BY-3.0", "CC-BY-4.0", "PD-USGOV"})
# Anything whose source text matches these is never servable, whatever licence string it claims (design section 10).
NEVER_SERVABLE = re.compile(
    r"valve|half[- ]?life|counter[- ]?strike|\bhl2\b|\bcss\b|source engine|steam subscriber|shapenet|partnet|sketchfab|"
    r"turbosquid|thingiverse|printables|\bfab\b|quaternius", re.I)

_NC = re.compile(r"\bnc\b|non[- ]?commercial", re.I)
_SA = re.compile(r"\bsa\b|share[- ]?alike", re.I)
_ND = re.compile(r"\bnd\b|no[- ]?deriv", re.I)
_CC0 = re.compile(r"^(cc[- ]?0(?:[- ]?1\.0)?|cc[- ]?zero|creative commons[- ]+public domain|public domain \(cc0\)|cc0 ?1\.0 ?universal)$", re.I)
_CCBY = re.compile(r"^(?:cc|creative commons)[- ]?(?:by|attribution)[- ]?(\d(?:\.\d)?)$", re.I)
_USGOV = re.compile(r"^(pd[- ]?usgov|us[- ]?gov(?:ernment)?(?: public domain)?|public[- ]domain[- ]us[- ]gov(?:ernment)?|nasa)$", re.I)


def normalize_licence(lic: str | None) -> str:
    """Canonical licence token: CC0, CC-BY-3.0, CC-BY-4.0, PD-USGOV, CC-NC, CC-SA, CC-ND, or the stripped input (unrecognised)."""
    s = (lic or "").strip()
    if not s:
        return "UNKNOWN"
    if _NC.search(s):
        return "CC-NC"
    if _SA.search(s):
        return "CC-SA"
    if _ND.search(s):
        return "CC-ND"
    if _CC0.match(s):
        return "CC0"
    m = _CCBY.match(s)
    if m:
        ver = m.group(1)
        return "CC-BY-" + (ver if "." in ver else ver + ".0")
    if _USGOV.match(s):
        return "PD-USGOV"
    return s


def licence_allowed(lic: str | None) -> bool:
    return normalize_licence(lic) in ALLOWED_LICENCES


def blocked_reason(card: dict) -> str | None:
    """Why this card's content can never be served, regardless of its claimed licence; None if nothing blocks it."""
    prov = card.get("provenance") or {}
    for field in ("source", "license", "url", "author"):
        v = prov.get(field)
        if v and NEVER_SERVABLE.search(str(v)):
            return f"provenance.{field} matches a never-servable source ({v!r})"
    return None


def compute_redistributable(card: dict) -> tuple[bool, str]:
    """(may this card's data/asset be sent to friends' browsers, reason). The only place the gate is decided."""
    prov = card.get("provenance") or {}
    lic = normalize_licence(prov.get("license"))
    why = blocked_reason(card)
    if why:
        return False, why
    if lic not in ALLOWED_LICENCES:
        return False, f"licence {lic!r} is not in the allowed set {sorted(ALLOWED_LICENCES)}"
    return True, "licence allowed"


# --- basis helpers ------------------------------------------------------------------------------------------------------------
def basis_rank(b: str) -> int:
    return BASIS_ORDER.index(b)


def weakest_basis(*bases: str) -> str:
    """A derived number is never stronger than its weakest input."""
    return min(bases, key=basis_rank)


# --- construction -------------------------------------------------------------------------------------------------------------
def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "x"


def make_card(*, id: str, name: str, kind: str | None = None, aliases=(), category: str = "misc", size=None, front=None,
              symmetric: bool = True, top=None, mass=None, mobility: str | None = None, collision=None, visual=None,
              provenance=None, seat: bool = False, wall: str = "free", approx_names=(), tags=(), flags=(), say: str | None = None) -> dict:
    """Assemble a card and fill the gate fields (`redistributable`, `say`, `mobility`). `size` = {w,d,h,basis,source} in metres or None."""
    card = {
        "card_v": CARD_V, "id": id, "name": name, "kind": kind or name, "aliases": list(dict.fromkeys(aliases)),
        "approx_names": list(dict.fromkeys(approx_names)), "category": category, "tags": list(tags),
        "size_m": size, "front": front or {"yaw_deg": 0.0, "source": "unknown"}, "symmetric": bool(symmetric), "top": top,
        "seat": bool(seat), "wall": wall, "mass": mass,
        "collision": collision or {"mode": "box"},
        "visual": visual or {"class": "placeholder", "style": "box", "blocky": True},
        "provenance": provenance or {}, "flags": list(flags),
    }
    if size is None and "size_unknown" not in card["flags"]:
        card["flags"].append("size_unknown")
    if mobility is None:
        mobility = "dynamic"
        if size and (max(size["w"], size["d"], size["h"]) > 8.0 or (mass and mass.get("kg", 0) > 2000)):
            mobility = "static"                                   # design section 7: over 8 m or 2,000 kg is static
    card["mobility"] = mobility
    ok, why = compute_redistributable(card)
    card["redistributable"] = ok
    card["redistributable_reason"] = why
    if not ok and "not_redistributable" not in card["flags"]:
        card["flags"].append("not_redistributable")
    card["say"] = say if say is not None else honesty_clause(card)
    return card


def prior_mass(size: dict) -> dict:
    """The design's fallback mass prior, 41.9 x V^0.72 (V = bounding-box volume in m^3), ALWAYS labelled `guessed`
    (misc/SKIPPYVERSE_OBJECTS_DESIGN.md section 4 and 9: rms error about x2.2, so low/high = kg / 2.2 and kg * 2.2). It is a stated
    prior, not a physical computation, and never stronger than a guess."""
    v = size["w"] * size["d"] * size["h"]
    kg = 41.9 * v ** 0.72
    return {"kg": kg, "low": kg / 2.2, "high": kg * 2.2, "basis": "guessed", "derived_from_size": True,
            "source": "prior 41.9*V^0.72 (V = bbox volume m^3) from misc/SKIPPYVERSE_OBJECTS_DESIGN.md; rms about x2.2"}


# --- validation ---------------------------------------------------------------------------------------------------------------
def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_card(card: dict) -> list[str]:
    """Return a list of problems (empty = valid). Includes the licence gate: redistributable=True must be earned."""
    p: list[str] = []
    if not isinstance(card, dict):
        return ["card is not an object"]
    if card.get("card_v") != CARD_V:
        p.append(f"card_v must be {CARD_V}")
    for f in ("id", "name", "kind", "category"):
        if not isinstance(card.get(f), str) or not card[f].strip():
            p.append(f"{f} must be a non-empty string")
    if not isinstance(card.get("aliases"), list) or not all(isinstance(a, str) for a in card.get("aliases", [])):
        p.append("aliases must be a list of strings")
    flags = card.get("flags") or []
    size = card.get("size_m")
    if size is None:
        if "size_unknown" not in flags:
            p.append("size_m is null but the card is not flagged size_unknown")
    else:
        for k in ("w", "d", "h"):
            if not _num(size.get(k)) or size[k] <= 0:
                p.append(f"size_m.{k} must be a positive finite number (metres)")
        if size.get("basis") not in BASIS:
            p.append(f"size_m.basis must be one of {sorted(BASIS)}")
        if not size.get("source"):
            p.append("size_m.source is required")
    fr = card.get("front") or {}
    if not _num(fr.get("yaw_deg")):
        p.append("front.yaw_deg must be a number")
    if fr.get("source") not in FRONT_SOURCES:
        p.append(f"front.source must be one of {sorted(FRONT_SOURCES)}")
    if fr.get("source") == "unknown" and not card.get("symmetric"):
        p.append("front.source is unknown, so the card must be symmetric (design: unknown front is treated as symmetric)")
    if not isinstance(card.get("symmetric"), bool):
        p.append("symmetric must be a bool")
    mass = card.get("mass")
    if mass is not None:
        if not _num(mass.get("kg")) or mass["kg"] <= 0:
            p.append("mass.kg must be positive")
        if mass.get("basis") not in BASIS:
            p.append("mass.basis invalid")
        lo, hi = mass.get("low"), mass.get("high")
        if lo is not None and hi is not None and _num(mass.get("kg")) and not (lo <= mass["kg"] <= hi):
            p.append("mass.kg outside its own low..high")
        if (mass.get("derived_from_size") and size is not None and mass.get("basis") in BASIS and size.get("basis") in BASIS
                and basis_rank(mass["basis"]) > basis_rank(size["basis"])):
            p.append("mass.basis is stronger than the size it was derived from")
    if card.get("mobility") not in MOBILITY:
        p.append("mobility invalid")
    if (card.get("collision") or {}).get("mode") not in COLLISION_MODES:
        p.append("collision.mode invalid")
    vis = card.get("visual") or {}
    if vis.get("class") not in CLASSES:
        p.append(f"visual.class must be one of {sorted(CLASSES)}")
    prov = card.get("provenance") or {}
    if not prov.get("source"):
        p.append("provenance.source is required")
    if not prov.get("license"):
        p.append("provenance.license is required")
    # --- the licence gate
    ok, why = compute_redistributable(card)
    if card.get("redistributable") is True and not ok:
        p.append(f"LICENCE GATE: redistributable=true but {why}")
    if card.get("redistributable") is True and vis.get("class") == "found":
        if not prov.get("author"):
            p.append("LICENCE GATE: a servable found asset needs provenance.author")
        if not prov.get("url"):
            p.append("LICENCE GATE: a servable found asset needs provenance.url")
    if card.get("redistributable") is False and "not_redistributable" not in flags:
        p.append("redistributable=false must be flagged not_redistributable")
    return p


def assert_valid(card: dict) -> dict:
    probs = validate_card(card)
    if probs:
        raise ValueError(f"invalid card {card.get('id')!r}: " + "; ".join(probs))
    return card


# --- what Skippy says ---------------------------------------------------------------------------------------------------------
def basis_phrase(what: str, block: dict | None) -> str:
    if not block:
        return f"{what} unknown"
    b = block.get("basis")
    return {"measured": f"{what} measured", "sourced": f"{what} from a published source",
            "estimated": f"{what} estimated", "guessed": f"{what} is a guess"}.get(b, f"{what} unlabelled")


def honesty_clause(card: dict) -> str:
    """Generated only from card fields (class, style, blocky, basis, licence). Never from a model's opinion."""
    vis = card.get("visual") or {}
    cls = vis.get("class")
    prov = card.get("provenance") or {}
    size, mass = card.get("size_m"), card.get("mass")
    parts: list[str] = []
    if cls == "curated":
        parts.append("a real model from my game library")
    elif cls == "found":
        style = vis.get("style") or "model"
        lic = normalize_licence(prov.get("license"))
        by = f" by {prov['author']}" if prov.get("author") else ""
        parts.append(f"a {style} model{by}, {lic}")
    elif cls == "generated":
        parts.append("an AI-made guess from one picture")
    elif cls == "deckard":
        parts.append("built from simple parts" + (", blocky" if vis.get("blocky") else ""))
    else:
        parts.append("only a box, no model")
    if size is None:
        parts.append("size unknown")
    elif size.get("basis") != "measured" or cls != "curated":
        parts.append(basis_phrase("size", size))
    if mass and mass.get("basis") == "guessed":
        parts.append("weight is a guess")
    elif mass is None:
        parts.append("weight unknown")
    s = "; ".join(parts)
    return s[0].upper() + s[1:] + "."


# --- converter to the holodeck planner ---------------------------------------------------------------------------------------
def m_to_hu(m: float) -> float:
    return m / HU_M


def to_planner_footprint(card: dict) -> dict:
    """A card -> one variant record in the shape stage/gmod/holodeck/data/catalog.json uses (Hammer units), ready for a later
    addKind. Raises ValueError for a card that cannot be placed (no size). `model` is the card id (an asset id, not a .mdl path)
    unless the card carries a local model path."""
    size = card.get("size_m")
    if not size:
        raise ValueError(f"card {card.get('id')!r} has no size, so the planner cannot place it")
    vis = card.get("visual") or {}
    frame = vis.get("model_frame_hu")
    w, d, h = (m_to_hu(size[k]) for k in ("w", "d", "h"))
    front = float(card["front"]["yaw_deg"]) % 360.0
    if frame:
        mins, maxs = list(frame["mins"]), list(frame["maxs"])
        cx, cy = frame["cx"], frame["cy"]
    else:
        ex, ey = (d, w) if int(round(front)) % 180 == 0 else (w, d)
        mins, maxs = [-ex / 2, -ey / 2, 0.0], [ex / 2, ey / 2, h]
        cx = cy = 0.0
    top = None
    t = card.get("top")
    if t:
        top = {k: m_to_hu(t[k]) for k in ("z", "x0", "x1", "y0", "y1")}
    mass = card.get("mass")
    return {
        "model": vis.get("local_model") or card["id"], "front": front, "symmetric": bool(card.get("symmetric")),
        "front_source": card["front"]["source"], "top": top, "w": w, "d": d, "h": h, "mins": mins, "maxs": maxs, "cx": cx, "cy": cy,
        "mass": mass["kg"] if mass else None, "mass_status": mass["basis"] if mass else None,
        "material": "default", "skins": 1,
    }


def to_planner_kind(cards: list[dict]) -> dict:
    """Cards of ONE kind -> a catalog.json kind record (names, approx, category, seat, surface, wall, variants)."""
    if not cards:
        raise ValueError("no cards")
    first = cards[0]
    names = list(dict.fromkeys([first["kind"]] + [a for c in cards for a in c["aliases"]]))
    approx = list(dict.fromkeys(a for c in cards for a in c.get("approx_names", [])))
    return {
        "kind": first["kind"], "names": names, "approx": approx, "category": first["category"],
        "seat": bool(first.get("seat")), "surface": any(c.get("top") for c in cards), "wall": first.get("wall", "free"),
        "variants": [to_planner_footprint(c) for c in cards if c.get("size_m")],
    }


def strip_for_web(card: dict) -> dict:
    """The copy of a card that may leave this machine: local model paths and other game-only fields are removed from every card;
    a card that is not redistributable also loses everything about the asset itself (it keeps only name, size and mass facts)."""
    c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in card.items()}
    vis = dict(c.get("visual") or {})
    vis.pop("local_model", None)
    vis.pop("local_asset", None)
    vis.pop("model_frame_hu", None)
    if not c.get("redistributable"):
        vis.pop("lods", None)
        c["provenance"] = {k: v for k, v in (c.get("provenance") or {}).items() if k in ("source", "license")}
    c["visual"] = vis
    return c
