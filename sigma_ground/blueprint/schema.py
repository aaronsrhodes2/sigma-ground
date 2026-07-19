"""Blueprint Spec — a cited build sheet for a MECHANISM (gear train, spring,
escapement), sibling to ``deckard.schema.ConstructSpec`` (a cited build sheet
for an object's SHAPE). Same discipline, extended: every value is a
``BlueprintFact`` — a value with its (source, license, confidence) AND a
mandatory verbatim ``quote`` + ``locator``, so a number is never presented
as real unless the exact sentence/table row it came from is attached.

Deckard's ``Fact`` doesn't carry a quote because most of its sources are
structured APIs (Wikidata claims, NIST tables) where the field IS the
citation. Blueprint sources are unstructured prose (patent claims, textbook
paragraphs) — the quote is the only thing that lets a reviewer verify the
number wasn't invented.

``MechanismSpec`` mirrors ``ConstructSpec``'s markdown round-trip
(``emit_markdown`` / ``parse_markdown``, fenced ```json payload) so a demo
rebuild reads committed, reviewed text — never a live extraction call at
build time.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

ESTIMATED = "estimated"   # BlueprintFact.source sentinel: a guess, not a citation


@dataclass
class BlueprintFact:
    """A single sourced value with its provenance AND its verbatim quote.

    quote:   the exact sentence/table-row/claim-text the value was read
             from — mandatory for anything not ``estimated`` (enforced by
             validate.py, not the dataclass itself, so a value CAN be
             constructed mid-pipeline before its quote is attached).
    locator: where in the source ("US1,234,567 claim 3", "Kelly p.18",
             "KHK module table m=1.5 row") — human-checkable, not a URL.
    """
    value: Any
    source: str = ESTIMATED
    license: str = ""
    confidence: float = 0.5
    quote: str = ""
    locator: str = ""

    @property
    def estimated(self) -> bool:
        return (not self.source) or self.source == ESTIMATED

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "source": self.source,
            "license": self.license,
            "confidence": self.confidence,
            "quote": self.quote,
            "locator": self.locator,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "BlueprintFact":
        if not isinstance(d, dict):
            return cls(value=d)                # a bare value => an estimate
        return cls(
            value=d.get("value"),
            source=d.get("source", ESTIMATED),
            license=d.get("license", ""),
            confidence=float(d.get("confidence", 0.5)),
            quote=d.get("quote", ""),
            locator=d.get("locator", ""),
        )

    def cite(self) -> str:
        if self.estimated:
            return f"{self.value}  [estimated]"
        lic = f", {self.license}" if self.license else ""
        loc = f" — {self.locator}" if self.locator else ""
        return f"{self.value} ({self.source}{lic}{loc}; conf {self.confidence:.2f})"


def _fact_or_none(d):
    return None if d is None else BlueprintFact.from_dict(d)


def _dict_or_none(f):
    return None if f is None else f.to_dict()


@dataclass
class GearSpec:
    """One wheel or pinion in a going train. Module/pressure-angle/tooth-form
    are OPTIONAL — a real source (like a horology textbook's wheel-work
    chapter) often states tooth/leaf counts without stating physical module;
    inventing a plausible module to fill the gap is exactly the fabrication
    this schema exists to prevent. Downstream geometry (Phase 4) simply
    cannot be generated until a module source is also cited — that is a
    real, honest gap, not a missing feature."""
    name: str                                   # "center_wheel", "escape_pinion"
    teeth: BlueprintFact                        # tooth count (wheel) or leaf count (pinion)
    is_pinion: bool = False
    module_mm: BlueprintFact | None = None
    pressure_angle_deg: BlueprintFact | None = None
    tooth_form: BlueprintFact | None = None      # "involute" | "cycloidal" | unstated
    addendum_coeff: BlueprintFact | None = None  # x module; ISO default 1.0 ONLY if EXTRACTED as such
    dedendum_coeff: BlueprintFact | None = None  # x module; ISO default 1.25 ONLY if EXTRACTED as such
    face_width_mm: BlueprintFact | None = None
    material: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "teeth": self.teeth.to_dict(),
            "is_pinion": self.is_pinion,
            "module_mm": _dict_or_none(self.module_mm),
            "pressure_angle_deg": _dict_or_none(self.pressure_angle_deg),
            "tooth_form": _dict_or_none(self.tooth_form),
            "addendum_coeff": _dict_or_none(self.addendum_coeff),
            "dedendum_coeff": _dict_or_none(self.dedendum_coeff),
            "face_width_mm": _dict_or_none(self.face_width_mm),
            "material": self.material,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GearSpec":
        return cls(
            name=d["name"],
            teeth=BlueprintFact.from_dict(d["teeth"]),
            is_pinion=bool(d.get("is_pinion", False)),
            module_mm=_fact_or_none(d.get("module_mm")),
            pressure_angle_deg=_fact_or_none(d.get("pressure_angle_deg")),
            tooth_form=_fact_or_none(d.get("tooth_form")),
            addendum_coeff=_fact_or_none(d.get("addendum_coeff")),
            dedendum_coeff=_fact_or_none(d.get("dedendum_coeff")),
            face_width_mm=_fact_or_none(d.get("face_width_mm")),
            material=d.get("material", ""),
        )


@dataclass
class MeshPair:
    """A wheel driving a pinion (a.teeth engages b.teeth). ``turns_ratio`` is
    the DERIVED a/b ratio (turns of b per turn of a) — computed, not cited,
    but tagged so validate.py can cross-check it against any independently
    stated period/turns-per-hour the source ALSO gives (the strongest check
    available when module/center-distance data is absent)."""
    a: str                                       # GearSpec.name (the wheel)
    b: str                                       # GearSpec.name (the pinion it drives)
    center_distance_mm: BlueprintFact | None = None

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b,
                "center_distance_mm": _dict_or_none(self.center_distance_mm)}

    @classmethod
    def from_dict(cls, d: dict) -> "MeshPair":
        return cls(a=d["a"], b=d["b"],
                  center_distance_mm=_fact_or_none(d.get("center_distance_mm")))


@dataclass
class SpringSpec:
    kind: str = "mainspring"
    torque_curve: list = field(default_factory=list)   # [BlueprintFact] sampled points, never fitted
    free_length_mm: BlueprintFact | None = None
    strip_thickness_mm: BlueprintFact | None = None
    strip_width_mm: BlueprintFact | None = None
    material: str = "spring steel"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "torque_curve": [f.to_dict() for f in self.torque_curve],
            "free_length_mm": _dict_or_none(self.free_length_mm),
            "strip_thickness_mm": _dict_or_none(self.strip_thickness_mm),
            "strip_width_mm": _dict_or_none(self.strip_width_mm),
            "material": self.material,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SpringSpec":
        return cls(
            kind=d.get("kind", "mainspring"),
            torque_curve=[BlueprintFact.from_dict(x) for x in d.get("torque_curve", [])],
            free_length_mm=_fact_or_none(d.get("free_length_mm")),
            strip_thickness_mm=_fact_or_none(d.get("strip_thickness_mm")),
            strip_width_mm=_fact_or_none(d.get("strip_width_mm")),
            material=d.get("material", "spring steel"),
        )


@dataclass
class EscapementSpec:
    kind: str = ""                               # "verge" | "anchor" | "lever" | "cylinder" | ...
    escape_wheel_teeth: BlueprintFact | None = None
    beats_per_hour: BlueprintFact | None = None   # vibrations/impulses per hour, if stated
    lift_angle_deg: BlueprintFact | None = None
    drop_angle_deg: BlueprintFact | None = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "escape_wheel_teeth": _dict_or_none(self.escape_wheel_teeth),
            "beats_per_hour": _dict_or_none(self.beats_per_hour),
            "lift_angle_deg": _dict_or_none(self.lift_angle_deg),
            "drop_angle_deg": _dict_or_none(self.drop_angle_deg),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EscapementSpec":
        return cls(
            kind=d.get("kind", ""),
            escape_wheel_teeth=_fact_or_none(d.get("escape_wheel_teeth")),
            beats_per_hour=_fact_or_none(d.get("beats_per_hour")),
            lift_angle_deg=_fact_or_none(d.get("lift_angle_deg")),
            drop_angle_deg=_fact_or_none(d.get("drop_angle_deg")),
        )


@dataclass
class MechanismSpec:
    """A researched mechanism: cited gear train + optional spring/escapement,
    the mechanism analogue of ``deckard.schema.ConstructSpec``."""
    name: str
    identified: bool = True
    gears: list = field(default_factory=list)      # [GearSpec]
    meshes: list = field(default_factory=list)      # [MeshPair]
    spring: SpringSpec | None = None
    escapement: EscapementSpec | None = None
    sources: list = field(default_factory=list)     # [{"name","license","url","locator"}]
    notes: str = ""

    def gear(self, name: str) -> GearSpec:
        for g in self.gears:
            if g.name == name:
                return g
        raise KeyError(f"gear '{name}' not in mechanism '{self.name}'")

    def note(self) -> str:
        tag = "" if self.identified else "  [UNIDENTIFIED — best-guess proportions]"
        src = ", ".join(s.get("name", "?") for s in self.sources) or "no sources"
        return f"{self.name} (mechanism) — {src}{tag}"

    def to_payload(self) -> dict:
        return {
            "name": self.name,
            "identified": self.identified,
            "gears": [g.to_dict() for g in self.gears],
            "meshes": [m.to_dict() for m in self.meshes],
            "spring": _dict_or_none(self.spring),
            "escapement": _dict_or_none(self.escapement),
            "sources": list(self.sources),
            "notes": self.notes,
        }

    @classmethod
    def from_payload(cls, d: dict) -> "MechanismSpec":
        return cls(
            name=d["name"],
            identified=bool(d.get("identified", True)),
            gears=[GearSpec.from_dict(x) for x in d.get("gears", [])],
            meshes=[MeshPair.from_dict(x) for x in d.get("meshes", [])],
            spring=(SpringSpec.from_dict(d["spring"]) if d.get("spring") else None),
            escapement=(EscapementSpec.from_dict(d["escapement"])
                       if d.get("escapement") else None),
            sources=list(d.get("sources", [])),
            notes=d.get("notes", ""),
        )


# ── markdown ↔ spec ─────────────────────────────────────────────────────────
# Same delimiter convention as deckard.schema: the fenced ```json block is
# canonical; the prose above it is human-facing provenance.
_JSON_BLOCK = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)


def emit_markdown(spec: MechanismSpec) -> str:
    flag = "" if spec.identified else "  ⚠ UNIDENTIFIED — best-guess proportions"
    out: list[str] = [f"# Blueprint Spec — {spec.name}{flag}", ""]
    out += [f"- **identified**: {spec.identified}", ""]

    out += ["## Sources", ""]
    if spec.sources:
        for s in spec.sources:
            line = f"- {s.get('name', '?')} — {s.get('license', '?')}"
            if s.get("url"):
                line += f" — {s['url']}"
            if s.get("locator"):
                line += f" ({s['locator']})"
            out.append(line)
    else:
        out.append("- (none cited)")
    out.append("")

    if spec.gears:
        out += ["## Gears", ""]
        for g in spec.gears:
            kind = "pinion (leaves)" if g.is_pinion else "wheel (teeth)"
            out.append(f"- **{g.name}** — {kind}: {g.teeth.cite()}")
            if g.teeth.quote:
                out.append(f'    - quote: "{g.teeth.quote}"')
            for label, f in (("module", g.module_mm), ("pressure angle", g.pressure_angle_deg),
                             ("tooth form", g.tooth_form)):
                if f is not None:
                    out.append(f"    - {label}: {f.cite()}")
                    if f.quote:
                        out.append(f'        - quote: "{f.quote}"')
        out.append("")

    if spec.meshes:
        out += ["## Meshes", ""]
        for m in spec.meshes:
            out.append(f"- **{m.a}** → **{m.b}**"
                       + (f" — center distance: {m.center_distance_mm.cite()}"
                          if m.center_distance_mm else ""))
        out.append("")

    if spec.spring:
        out += ["## Spring", "", f"- kind: {spec.spring.kind}",
               f"- material: {spec.spring.material}"]
        if spec.spring.torque_curve:
            out.append("- torque curve (sampled points, not fitted):")
            for pt in spec.spring.torque_curve:
                out.append(f"    - {pt.cite()}")
        out.append("")

    if spec.escapement:
        e = spec.escapement
        out += ["## Escapement", "", f"- kind: {e.kind}"]
        if e.escape_wheel_teeth:
            out.append(f"- escape wheel teeth: {e.escape_wheel_teeth.cite()}")
            if e.escape_wheel_teeth.quote:
                out.append(f'    - quote: "{e.escape_wheel_teeth.quote}"')
        if e.beats_per_hour:
            out.append(f"- beats/hour: {e.beats_per_hour.cite()}")
            if e.beats_per_hour.quote:
                out.append(f'    - quote: "{e.beats_per_hour.quote}"')
        out.append("")

    if spec.notes:
        out += ["## Notes", "", spec.notes, ""]

    payload = json.dumps(spec.to_payload(), indent=2, ensure_ascii=False)
    out += ["## Canonical payload", "",
            "<!-- consumers read this block; keep the prose above in sync. -->",
            "```json", payload, "```", ""]
    return "\n".join(out)


def parse_markdown(md: str) -> MechanismSpec:
    m = _JSON_BLOCK.search(md)
    if not m:
        raise ValueError("no ```json payload found in Blueprint-Spec markdown")
    return MechanismSpec.from_payload(json.loads(m.group(1)))


__all__ = [
    "ESTIMATED", "BlueprintFact", "GearSpec", "MeshPair", "SpringSpec",
    "EscapementSpec", "MechanismSpec", "emit_markdown", "parse_markdown",
]
