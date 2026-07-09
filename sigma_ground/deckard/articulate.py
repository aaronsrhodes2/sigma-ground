"""The A-graph — parts as bodies, joints as typed interface labels.

A ``Construct``'s parts already touch (the voxel interface scan measures every
contact patch); this module gives each touching pair its KINEMATIC MEANING:

    weld       no relative motion (the default — today's rigid behavior)
    revolute   rotation about one axis (a hinge, a pivot, a journal)
    prismatic  sliding along one axis (a drawer, a piston)
    free       no permanent bond (resting contact only; b=None grounds to world)

Acquisition precedence (the shape-oracle doctrine — real data first, honest
inference second, silence never):

  1. a DISTILLED annotation file (``inventory/data/articulations/<anno>.json``,
     written from PartNet-Mobility when the dataset is present) — conf 0.95;
  2. deterministic inference: every contacting pair → WELD at conf 0.8
     (flagged ``default_weld`` — exactly the rigid lump we shipped yesterday,
     now stated per-pair), plus a small NAME-PAIR rule table that PROPOSES a
     revolute/prismatic at conf 0.4–0.5 with the axis SELECTED from the
     recorded interface geometry (never invented).

Motion gating is the consumer's job: ``effective_joints(art, min_confidence)``
downgrades low-confidence movers to weld (flagged), so nothing swings on a
guess unless the user's intent explicitly lowers the bar.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field


@dataclass
class ConstructPart:
    part_id: int
    name: str
    material: str
    labels: tuple                # voxel label ids belonging to this part
    mass_kg: float
    com_m: tuple
    inertia_kgm2: tuple          # about the part's own CoM, construct rest axes
    volume_m3: float
    bbox_m: tuple = None
    flags: tuple = ()
    source: str = ""


@dataclass
class Joint:
    a: int                       # part id
    b: int | None                # part id, or None = the world (grounded/free)
    type: str                    # weld | revolute | prismatic | free
    axis_origin: tuple | None = None   # m, construct rest frame
    axis_dir: tuple | None = None      # unit
    limits: tuple | None = None        # (lo, hi): rad (revolute) | m (prismatic)
    confidence: float = 0.8
    source: str = ""
    flags: tuple = ()
    contact_area_m2: float = 0.0


@dataclass
class Articulation:
    parts: list = field(default_factory=list)     # [ConstructPart]
    joints: list = field(default_factory=list)    # [Joint]
    source: str = ""
    notes: str = ""
    mobility: dict = None        # {"available": bool, "model_id": str|None, "note": str}

    def to_dict(self) -> dict:
        return {
            "parts": [{**p.__dict__, "labels": list(p.labels),
                       "com_m": list(p.com_m),
                       "inertia_kgm2": list(p.inertia_kgm2),
                       "bbox_m": (list(p.bbox_m) if p.bbox_m else None),
                       "flags": list(p.flags)} for p in self.parts],
            "joints": [{**j.__dict__,
                        "axis_origin": (list(j.axis_origin) if j.axis_origin else None),
                        "axis_dir": (list(j.axis_dir) if j.axis_dir else None),
                        "limits": (list(j.limits) if j.limits else None),
                        "flags": list(j.flags)} for j in self.joints],
            "source": self.source, "notes": self.notes, "mobility": self.mobility,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Articulation":
        parts = [ConstructPart(
            part_id=p["part_id"], name=p["name"], material=p["material"],
            labels=tuple(p["labels"]), mass_kg=p["mass_kg"],
            com_m=tuple(p["com_m"]), inertia_kgm2=tuple(p["inertia_kgm2"]),
            volume_m3=p["volume_m3"],
            bbox_m=(tuple(p["bbox_m"]) if p.get("bbox_m") else None),
            flags=tuple(p.get("flags") or ()), source=p.get("source", ""))
            for p in d.get("parts", [])]
        joints = [Joint(
            a=j["a"], b=j["b"], type=j["type"],
            axis_origin=(tuple(j["axis_origin"]) if j.get("axis_origin") else None),
            axis_dir=(tuple(j["axis_dir"]) if j.get("axis_dir") else None),
            limits=(tuple(j["limits"]) if j.get("limits") else None),
            confidence=j.get("confidence", 0.8), source=j.get("source", ""),
            flags=tuple(j.get("flags") or ()),
            contact_area_m2=j.get("contact_area_m2", 0.0))
            for j in d.get("joints", [])]
        return cls(parts=parts, joints=joints, source=d.get("source", ""),
                   notes=d.get("notes", ""), mobility=d.get("mobility"))


def effective_joints(art: Articulation, min_confidence: float = 0.75) -> list:
    """The joints a consumer should MOVE on: low-confidence movers are
    downgraded to weld (flagged ``downgraded_low_confidence``) so nothing
    articulates on a guess. Explicit user intent may pass a lower threshold."""
    out = []
    for j in art.joints:
        if j.type in ("revolute", "prismatic") and j.confidence < min_confidence:
            out.append(Joint(a=j.a, b=j.b, type="weld",
                             confidence=j.confidence, source=j.source,
                             flags=tuple(j.flags) + ("downgraded_low_confidence",),
                             contact_area_m2=j.contact_area_m2))
        else:
            out.append(j)
    return out


# name-pair cues → a PROPOSED joint type at low confidence. The axis is
# SELECTED from the measured interface geometry (elongated patch → its
# principal direction is the hinge line; round patch → its normal-ish
# principal is a pivot axis guess) — never a hand-invented vector.
_RULE_TABLE = (
    ({"wheel", "caster"}, "revolute", 0.5),
    ({"blade"}, "revolute", 0.45),                    # blade-on-blade: scissors
    ({"door", "lid", "cover"}, "revolute", 0.5),
    ({"drawer", "slider"}, "prismatic", 0.5),
    ({"hand", "pendulum"}, "revolute", 0.4),          # clock hands / pendulums
)


def _rule_for(name_a: str, name_b: str):
    base_a = name_a.split("_")[0]
    base_b = name_b.split("_")[0]
    for keys, jtype, conf in _RULE_TABLE:
        if base_a in keys or base_b in keys or (base_a == base_b and base_a in keys):
            return jtype, conf
    return None


def infer_joints(parts: list, part_interfaces: dict, *, source_note="") -> Articulation:
    """Deterministic tier: every contacting pair → WELD conf 0.8 (today's
    rigid behavior, per-pair and flagged); name-pair rules may PROPOSE a
    mover at conf 0.4–0.5 with axis geometry from the interface scan."""
    cparts = [ConstructPart(
        part_id=p["part_id"], name=p["name"], material=p["material"],
        labels=tuple(p["labels"]), mass_kg=p["mass_kg"], com_m=tuple(p["com_m"]),
        inertia_kgm2=tuple(p["inertia_kgm2"]), volume_m3=p["volume_m3"],
        flags=tuple(p["flags"]), source="voxel field") for p in (parts or [])]
    by_id = {p.part_id: p for p in cparts}
    joints = []
    for (pa, pb), g in sorted((part_interfaces or {}).items()):
        rule = _rule_for(by_id[pa].name, by_id[pb].name) if pa in by_id and pb in by_id else None
        if rule:
            jtype, conf = rule
            joints.append(Joint(
                a=pa, b=pb, type=jtype,
                axis_origin=tuple(g["centroid_m"]),
                axis_dir=tuple(g["principal_dir"]),
                limits=None, confidence=conf,
                source="name-pair rule + interface geometry (axis SELECTED, "
                       "not invented)",
                flags=("rule_hypothesis", "axis_estimated"),
                contact_area_m2=g["area_m2"]))
        else:
            joints.append(Joint(
                a=pa, b=pb, type="weld", confidence=0.8,
                source="deterministic default: contacting parts weld "
                       "(the pre-articulation rigid behavior, stated per pair)",
                flags=("default_weld",),
                contact_area_m2=g["area_m2"]))
    return Articulation(parts=cparts, joints=joints,
                        source=source_note or "deterministic inference",
                        mobility={"available": False, "model_id": None,
                                  "note": "no mobility annotation consulted"})


_ART_DIR = (pathlib.Path(__file__).resolve().parents[1]
            / "inventory" / "data" / "articulations")


def _load_distilled(anno_id) -> dict | None:
    p = _ART_DIR / f"{anno_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def acquire(anno_id, parts, part_interfaces, *, source_note="") -> Articulation:
    """The precedence chain: distilled mobility annotations (conf 0.95) when a
    file exists for this model, else deterministic inference. Name-keyed joints
    in the file resolve to part ids by exact name; unresolved names are
    dropped WITH a flagged note (never guessed)."""
    art = infer_joints(parts, part_interfaces, source_note=source_note)
    d = _load_distilled(anno_id) if anno_id else None
    if not d:
        return art
    by_name = {p.name: p.part_id for p in art.parts}
    resolved, dropped = [], []
    for j in d.get("joints", []):
        a = by_name.get(j.get("a"))
        b = by_name.get(j.get("b")) if j.get("b") is not None else None
        if a is None or (j.get("b") is not None and b is None):
            dropped.append(f"{j.get('a')}-{j.get('b')}")
            continue
        area = 0.0
        for (pa, pb), g in (part_interfaces or {}).items():
            if {pa, pb} == {a, b}:
                area = g["area_m2"]
        resolved.append(Joint(
            a=a, b=b, type=j["type"],
            axis_origin=(tuple(j["axis_origin"]) if j.get("axis_origin") else None),
            axis_dir=(tuple(j["axis_dir"]) if j.get("axis_dir") else None),
            limits=(tuple(j["limits"]) if j.get("limits") else None),
            confidence=j.get("confidence", 0.95),
            source=d.get("source", "distilled mobility annotation"),
            flags=tuple(j.get("flags") or ("partnet_mobility",)),
            contact_area_m2=area))
    if resolved:
        # annotated pairs REPLACE their inferred welds; everything else keeps
        # its inferred (weld) edge, so the graph stays connected and honest
        keyed = {tuple(sorted((j.a, j.b if j.b is not None else -1))): j
                 for j in resolved}
        kept = [j for j in art.joints
                if tuple(sorted((j.a, j.b if j.b is not None else -1))) not in keyed]
        art.joints = list(resolved) + kept
        art.source = d.get("source", art.source)
        art.mobility = {"available": True, "model_id": d.get("model_id"),
                        "note": f"{len(resolved)} annotated joints"
                                + (f"; dropped unresolved: {dropped}" if dropped else "")}
    if dropped and not resolved:
        art.notes = f"annotation file present but no joints resolved: {dropped}"
    return art


__all__ = ["ConstructPart", "Joint", "Articulation", "effective_joints",
           "infer_joints", "acquire"]
