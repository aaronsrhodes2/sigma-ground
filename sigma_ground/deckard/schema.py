"""Deckard's Construct Spec — a researched, cited build sheet for one object.

The research step turns a *name* into a ConstructSpec: a human-readable
provenance document (markdown) that carries a canonical ```json payload the
deterministic compiler reads. Every dimension and density is a Fact — a value
with its (source, license, confidence) — so a guess is never presented as a
measurement:

  * DB / measured values are cited (source + license);
  * LLM / heuristic guesses are flagged ``[estimated]``;
  * unknown objects fall back to a default and set ``identified = False``.

Two shape representations:
  * ``layered_vessel`` geometry (``geometry`` dims + ``layers``) — the cup kit;
  * a general primitive kit (``parts`` — spheres/cylinders/boxes/cones) for
    solid and composite objects.

``compile()`` (construct.py) consumes whichever is present.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

ESTIMATED = "estimated"   # Fact.source sentinel: a guess, not a measurement


@dataclass
class Fact:
    """A single researched value with its provenance."""
    value: Any
    source: str = ESTIMATED
    license: str = ""
    confidence: float = 0.5

    @property
    def estimated(self) -> bool:
        return (not self.source) or self.source == ESTIMATED

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "source": self.source,
            "license": self.license,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "Fact":
        if not isinstance(d, dict):
            return cls(value=d)               # a bare value => an estimate
        return cls(
            value=d.get("value"),
            source=d.get("source", ESTIMATED),
            license=d.get("license", ""),
            confidence=float(d.get("confidence", 0.5)),
        )

    def cite(self) -> str:
        if self.estimated:
            return f"{self.value}  [estimated]"
        lic = f", {self.license}" if self.license else ""
        return f"{self.value} ({self.source}{lic}; conf {self.confidence:.2f})"


@dataclass
class SpecLayer:
    """One material shell of a layered vessel, ordered outer→inner."""
    name: str
    material: str
    density: Fact                       # kg/m3
    thickness: Fact                     # m — physical thickness of this shell
    interfaces: list = field(default_factory=list)   # neighbours: air|vacuum|<material>

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "material": self.material,
            "density_kg_m3": self.density.to_dict(),
            "thickness_m": self.thickness.to_dict(),
            "interfaces": list(self.interfaces),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SpecLayer":
        return cls(
            name=d["name"],
            material=d.get("material", d["name"]),
            density=Fact.from_dict(d.get("density_kg_m3")),
            thickness=Fact.from_dict(d.get("thickness_m")),
            interfaces=list(d.get("interfaces", [])),
        )


@dataclass
class Part:
    """One primitive solid: a kernel shape + its material + pose.

    A general-kit member (sphere/cylinder/box/cone) for solid and composite
    objects, as opposed to the layered-vessel representation. ``dims`` are the
    shape's parameters as Facts (e.g. sphere ``{radius_m}``; cylinder
    ``{radius_m, height_m}``; box ``{x_m, y_m, z_m}``).

    Every Part still carries exactly ONE material/color — that invariant is
    never broken. ``scatter`` does not add a second material to a Part; it
    tells the compiler to multiply this ONE Part's shape/size into many
    single-material siblings (see the convention doc below).
    """
    name: str
    shape: str                          # sphere | cylinder | cone | box | torus | ellipsoid
    dims: dict                          # dim_name -> Fact
    material: str
    density: Fact                       # kg/m3
    center_m: tuple = (0.0, 0.0, 0.0)
    euler_deg: tuple = (0.0, 0.0, 0.0)  # rotation about the part centre (Rz·Ry·Rx, degrees)
    op: str = "add"                     # "add" | "subtract" (carve a cavity / hollow)
    attach: dict | None = None          # {"to": <part>, "my": <anchor>, "their": <anchor>}
    fill: dict | None = None            # fluid fill: {"of": <cavity>, "fraction": 0..1, "gas": <material>}
    conform: str | None = None          # name of the solid this part yields to (carved to mate)
    conform_region: str | None = None   # "top": keep only the coat ABOVE the host's
                                        # equator — a dipped doughnut is iced on top,
                                        # fried dough underneath (Captain, 2026-07-30)
    outline: dict | None = None         # swept profile: {"profile":[[u,v],...],"mode":"extrude|revolve","thickness":t}
    scatter: dict | None = None         # multiply this part across a host's surface — see below

    # ── the SCATTER convention ────────────────────────────────────────────
    # Rationale: a Part is (and stays) exactly one shape + one material. Some
    # real objects (sprinkles on a doughnut, rivets on a hull, gravel on a
    # path) are dozens-to-hundreds of small pieces that are the SAME rough
    # size/shape but DIFFERENT materials, resting on another part's surface.
    # Modelling each one as a hand-positioned Part would be unreadable and
    # non-reusable. Instead: THIS Part is a TEMPLATE — its own shape/dims are
    # "one representative instance" — and ``scatter`` is instructions for
    # multiplying that template across a host part's surface at seeded
    # (reproducible) random positions, one material drawn per instance from a
    # palette. The template Part itself never becomes geometry; only its
    # expanded instances do (construct.py: ``_expand_scatter``).
    #
    #   scatter = {
    #       "on":            <host part name>,        # required
    #       "count":         <int N>,                  # required
    #       "materials":     [<name>, ...],             # palette; default [self.material]
    #       "seed":          <int>,                     # RNG seed, default 0 (deterministic)
    #       "region":        "top" | "all",              # default "top" (gravity-dusted look)
    #       "region_up":     [x, y, z],                  # "up" for region="top", default +z
    #       "embed_frac":    0.0..1.0,                   # how deep into the host surface the
    #                                                     # instance's centre sits; 0 = resting
    #                                                     # entirely outside (tangent), 1 =
    #                                                     # centred ON the surface. Default 0.35.
    #       "size_jitter":   0.0..1.0,                   # +/- fractional per-instance size
    #                                                     # variation about the template dims;
    #                                                     # default 0.0 (uniform size).
    #   }
    #
    # Same generality as ``conform``/``attach``: any Part on any host shape,
    # since it only ever calls the host's ``surface_distance``/kernel Shape
    # interface that every primitive already implements — no per-shape-family
    # special-casing (a scatter over a torus, a box, a sphere all "just work").
    #
    # Density note: every expanded instance uses the TEMPLATE Part's OWN
    # ``density`` Fact, regardless of which ``materials`` label it draws — the
    # palette is for identity/cosmetic (colour) variation of one physically-
    # similar substance (e.g. sugar candy dyed different colours), not a
    # per-label density re-lookup. If two scattered pieces are meant to be
    # PHYSICALLY different materials (different density), give them separate
    # scatter groups (two Parts, each with its own ``scatter``), not one
    # palette — this keeps "one Fact = one physical claim" honest.

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "shape": self.shape,
            "dims": {k: f.to_dict() for k, f in self.dims.items()},
            "material": self.material,
            "density_kg_m3": self.density.to_dict(),
            "center_m": list(self.center_m),
            "euler_deg": list(self.euler_deg),
            "op": self.op,
        }
        if self.attach:
            d["attach"] = dict(self.attach)
        if self.fill:
            d["fill"] = dict(self.fill)
        if self.conform:
            d["conform"] = self.conform
        if self.conform_region:
            d["conform_region"] = self.conform_region
        if self.outline:
            d["outline"] = {
                "profile": [[float(u), float(v)] for u, v in self.outline.get("profile", [])],
                "mode": self.outline.get("mode", "extrude"),
                "thickness": self.outline.get("thickness"),
            }
        if self.scatter:
            d["scatter"] = dict(self.scatter)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Part":
        return cls(
            name=d["name"],
            shape=d["shape"],
            dims={k: Fact.from_dict(v) for k, v in d.get("dims", {}).items()},
            material=d.get("material", d["name"]),
            density=Fact.from_dict(d.get("density_kg_m3")),
            center_m=tuple(d.get("center_m", (0.0, 0.0, 0.0))),
            euler_deg=tuple(d.get("euler_deg", (0.0, 0.0, 0.0))),
            op=d.get("op", "add"),
            attach=d.get("attach"),
            fill=d.get("fill"),
            conform=d.get("conform"),
            conform_region=d.get("conform_region"),
            outline=d.get("outline"),
            scatter=d.get("scatter"),
        )


@dataclass
class ConstructSpec:
    """A researched object: cited geometry + materials + provenance.

    Generalises ItemSpec. A ``layered_vessel`` carries ``geometry`` (dim → Fact)
    and ``layers`` (SpecLayers outer→inner); a solid/composite object carries
    ``parts`` (primitive kit). ``identified`` is False for a flagged best-guess.
    """
    name: str
    kind: str = "layered_vessel"
    identified: bool = True
    geometry: dict = field(default_factory=dict)    # dim_name -> Fact
    layers: list = field(default_factory=list)       # [SpecLayer]
    parts: list = field(default_factory=list)        # [Part] (general primitive kit)
    sources: list = field(default_factory=list)      # [{"name","license","url"}]
    notes: str = ""

    # -- convenience for the compiler (plain values) --------------------------
    def dim(self, key: str, default: float | None = None) -> float:
        """Plain numeric value of a geometry dimension (raises if missing)."""
        f = self.geometry.get(key)
        if f is None:
            if default is None:
                raise KeyError(
                    f"geometry dimension '{key}' missing from spec '{self.name}'")
            return default
        return f.value

    def note(self) -> str:
        tag = "" if self.identified else "  [UNIDENTIFIED — best-guess proportions]"
        src = ", ".join(s.get("name", "?") for s in self.sources) or "no sources"
        return f"{self.name} ({self.kind}) — {src}{tag}"

    # -- (de)serialisation ----------------------------------------------------
    def to_payload(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "identified": self.identified,
            "geometry": {k: f.to_dict() for k, f in self.geometry.items()},
            "layers": [L.to_dict() for L in self.layers],
            "parts": [p.to_dict() for p in self.parts],
            "sources": list(self.sources),
            "notes": self.notes,
        }

    @classmethod
    def from_payload(cls, d: dict) -> "ConstructSpec":
        return cls(
            name=d["name"],
            kind=d.get("kind", "layered_vessel"),
            identified=bool(d.get("identified", True)),
            geometry={k: Fact.from_dict(v) for k, v in d.get("geometry", {}).items()},
            layers=[SpecLayer.from_dict(x) for x in d.get("layers", [])],
            parts=[Part.from_dict(x) for x in d.get("parts", [])],
            sources=list(d.get("sources", [])),
            notes=d.get("notes", ""),
        )


# ── markdown ↔ spec ─────────────────────────────────────────────────────────
# The fenced ```json block is the delimiter (handles nested braces).
_JSON_BLOCK = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)


def emit_markdown(spec: ConstructSpec) -> str:
    """Render a Construct-Spec as a cited markdown doc: prose + canonical json.

    The prose is human-facing provenance; the embedded ```json block is the
    canonical payload that ``parse_markdown`` / ``compile()`` read back.
    """
    flag = "" if spec.identified else "  ⚠ UNIDENTIFIED — best-guess proportions"
    out: list[str] = [f"# Construct Spec — {spec.name}{flag}", ""]
    out += [f"- **kind**: {spec.kind}",
            f"- **identified**: {spec.identified}", ""]

    out += ["## Sources", ""]
    if spec.sources:
        for s in spec.sources:
            line = f"- {s.get('name', '?')} — {s.get('license', '?')}"
            if s.get("url"):
                line += f" — {s['url']}"
            out.append(line)
    else:
        out.append("- (none cited)")
    out.append("")

    if spec.geometry:
        out += ["## Geometry", ""]
        for key, f in spec.geometry.items():
            out.append(f"- **{key}**: {f.cite()}")
        out.append("")

    if spec.layers:
        out += ["## Layers (outer→inner)", ""]
        for L in spec.layers:
            out.append(f"- **{L.name}** — {L.material}")
            out.append(f"    - density: {L.density.cite()} kg/m³")
            out.append(f"    - thickness: {L.thickness.cite()} m")
            if L.interfaces:
                out.append(f"    - interfaces: {', '.join(L.interfaces)}")
        out.append("")

    if spec.parts:
        out += ["## Parts (primitives)", ""]
        for p in spec.parts:
            out.append(f"- **{p.name}** — {p.shape} ({p.material}) @ {tuple(p.center_m)}")
            if p.dims:
                out.append(f"    - dims: {', '.join(f'{k}={f.cite()}' for k, f in p.dims.items())}")
            out.append(f"    - density: {p.density.cite()} kg/m³")
            if p.fill:
                out.append(f"    - fills: {p.fill.get('of')} to "
                           f"{p.fill.get('fraction', 1.0)} (gas on top: {p.fill.get('gas', 'air')})")
            if p.conform:
                out.append(f"    - conforms to: {p.conform} (yields the overlap; carved to mate)")
            if p.scatter:
                sc = p.scatter
                out.append(f"    - scattered: {sc.get('count', '?')}x on {sc.get('on', '?')} "
                           f"(materials: {', '.join(sc.get('materials') or [p.material])}; "
                           f"seed={sc.get('seed', 0)}; region={sc.get('region', 'top')})")
        out.append("")

    if spec.notes:
        out += ["## Notes", "", spec.notes, ""]

    payload = json.dumps(spec.to_payload(), indent=2, ensure_ascii=False)
    out += ["## Canonical payload", "",
            "<!-- compile() reads this block; keep the prose above in sync. -->",
            "```json", payload, "```", ""]
    return "\n".join(out)


def parse_markdown(md: str) -> ConstructSpec:
    """Recover a ConstructSpec from Construct-Spec markdown (its json block)."""
    m = _JSON_BLOCK.search(md)
    if not m:
        raise ValueError("no ```json payload found in Construct-Spec markdown")
    return ConstructSpec.from_payload(json.loads(m.group(1)))


__all__ = [
    "ESTIMATED", "Fact", "SpecLayer", "Part", "ConstructSpec",
    "emit_markdown", "parse_markdown",
]
