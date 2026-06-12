"""Deckard's Researcher — synthesise a cited ConstructSpec for an unknown name.

Fully LOCAL: the research LLM is qwen via ollama ``/api/chat`` (Gemini dropped —
no cloud, no API key). The prompt is GROUNDED with a free, cached Wikipedia
extract of the object (its real typical proportions + what it is made of), so the
model anchors the shape it proposes in web text rather than pure recall. The LLM
proposes the object's *shape* — either a hollow ``layered_vessel`` or a
``composite`` of solid primitives — plus material names; Deckard GROUNDS each
density in our own data / Wikidata (``sources``) and flags LLM-proposed dimensions
as ``[estimated]``. Offline or on any failure it returns None, so ``research()``
falls back to a flagged best-guess — never a fake.

Facts-first, facts-and-primitives only: no images, no meshes — the web + the
model supply proportions, our data supplies attributable densities.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

from .schema import ConstructSpec, Fact, SpecLayer, Part
from . import sources as _sources

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("DECKARD_OLLAMA_MODEL", "qwen2.5:7b")

_SYS = (
    "You are Deckard, a shape researcher for a physics compiler. Given an "
    "everyday object NAME, output ONLY JSON (no prose), all lengths in SI metres. "
    "Choose ONE form:\n"
    "A) a HOLLOW VESSEL (cup, mug, glass, bowl, bottle):\n"
    '{"kind":"layered_vessel","geometry":{"outer_radius_m":<r>,"height_m":<h>,'
    '"wall_m":<t>,"glaze_m":<skin>,"base_m":<floor>,"fill_fraction":<0..1>},'
    '"layers":[{"name":"glaze","material":"<outer skin>"},'
    '{"name":"ceramic","material":"<body>"},'
    '{"name":"water","material":"<liquid fill>"}]}\n'
    "B) a SOLID or COMPOUND object (rod, ball, die, hammer, dumbbell, ring):\n"
    '{"kind":"composite","parts":[{"name":"<part>","shape":'
    '"sphere|cylinder|box|cone|torus|ellipsoid","dims":{...},'
    '"material":"<material>","center_m":[x,y,z],"euler_deg":[rx,ry,rz]}]}\n'
    "   dims: sphere {radius_m}; cylinder/cone {radius_m,height_m}; box "
    "{x_m,y_m,z_m}; torus {major_radius_m,minor_radius_m}; ellipsoid "
    "{rx_m,ry_m,rz_m}. Cylinders/cones point along +z; euler_deg rotates a part "
    "(e.g. [0,90,0] lays a cylinder along x). One part for a simple solid; several "
    "with center_m offsets for a compound object (hammer = vertical handle "
    "cylinder + head cylinder rotated horizontal across the top). Set "
    'op:"subtract" on a part to carve a cavity, and a later op:"add" part to '
    "fill it (a pipe = outer cylinder + inner cylinder op:subtract). Parts compose "
    "IN ORDER — list solids first, then carves, then fills. To FILL a hollow with "
    "fluid, prefer a FILL part over a hand-sized solid: a part with shape:\"fill\" "
    'and fill:{"of":"<cavity part>","fraction":<0..1>,"gas":"<gas, default air>"} '
    "floods that carved cavity — the liquid sinks to the bottom, the gas settles on "
    "top (gravity). A half-full water bottle = body + interior op:subtract + "
    '{"name":"water","shape":"fill","material":"liquid water",'
    '"fill":{"of":"interior","fraction":0.5,"gas":"air"}}. '
    "Two SOLIDS mate only where their surfaces are exactly congruent (a flat end "
    "on a flat face). To mate a part onto a CURVED solid, set conform:\"<other "
    'part>\" and position this part overlapping it: the part yields the overlap '
    "(carved to the other's surface) so they share a real interface, never a clip "
    "(a stud on a ball = ball sphere + stud cylinder pushed into it with "
    'conform:"ball"). To JOIN parts, prefer attach over '
    'center_m: set attach:{"to":"<part>","my":"<anchor>","their":"<anchor>"} so '
    "the anchors meet exactly (no overlap, no gap). Anchors: top, bottom (any "
    "shape); +x,-x,+y,-y (box/sphere). E.g. hammer = head + handle "
    "attach:{to:head,my:top,their:bottom}.\n"
    "For an ORGANIC shape that is not a simple solid (feather, leaf, petal, "
    "blade), use a part with shape:\"outline\" and dims {length_m,thickness_m} "
    "plus a material — Deckard supplies the researched 2D outline (you do NOT "
    "draw it); it becomes a thin extruded sheet. E.g. a feather = a vane "
    'shape:"outline" (length_m~0.1, thickness_m~0.0006, keratin) + a thin '
    "cylinder shaft laid along it.\n"
    "Use realistic typical dimensions and a real material name (steel, glass, "
    'aluminium, oak, stoneware, ...). If you cannot, output {"kind":"unknown"}.'
)


def _qwen(name: str) -> str | None:
    """Ask local qwen via ollama /api/chat; return raw text or None."""
    body = json.dumps({
        "model": OLLAMA_MODEL, "stream": False, "format": "json",
        "options": {"temperature": 0},
        "messages": [{"role": "system", "content": _SYS},
                     {"role": "user", "content": name}],
    }).encode()
    try:
        req = urllib.request.Request(OLLAMA_URL + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["message"]["content"]
    except Exception:
        return None


def _ask(name: str) -> str | None:
    """The research LLM: LOCAL qwen via ollama (Gemini dropped — fully local)."""
    return _qwen(name)


def _augment_with_web(name: str) -> str:
    """Prepend a free Wikipedia extract (the object's real typical dimensions and
    what it is made of) to the research prompt, so the model GROUNDS the shape in
    web text rather than pure recall. Degrades silently to the bare name when
    offline or when there's no article — research still runs."""
    try:
        extract = _sources.wikipedia_summary(name)
    except Exception:
        extract = None
    if not extract:
        return name
    return (f"{name}\n\nReference (Wikipedia — use it for this object's typical "
            f"real dimensions and what it is made of):\n{extract}")


_JSON = re.compile(r"\{.*\}", re.DOTALL)
_VESSEL_DIMS = ("outer_radius_m", "height_m", "wall_m", "glaze_m", "base_m", "fill_fraction")
_SHAPE_DIMS = {"sphere": ["radius_m"], "cylinder": ["radius_m", "height_m"],
               "cone": ["radius_m", "height_m"], "box": ["x_m", "y_m", "z_m"],
               "torus": ["major_radius_m", "minor_radius_m"],
               "ellipsoid": ["rx_m", "ry_m", "rz_m"]}


def _density(material: str) -> Fact:
    """Grounded density Fact (local first, then Wikidata), else flagged [estimated]."""
    f = _sources.density_of(material, allow_web=True)
    return f if f is not None else Fact(1000.0, "estimated", "", 0.2)


def _cite_source(dens: Fact, sources: list, seen: set) -> None:
    if not dens.estimated and dens.source not in seen:
        sources.append({"name": dens.source, "license": dens.license})
        seen.add(dens.source)


_HUMAN_SCALE_MAX_M = 100.0   # above this an object isn't human-scale (hallucination guard)


def _ground_dims(name, shape, dims, sources, seen, allow_web):
    """Replace the model's dimension ESTIMATES with CITED dimensions where the
    object is a known standard (a tennis ball's radius, A4's sides, an AA cell).
    Mutates ``dims`` in place and records the source; leaves estimates otherwise.
    Local table first; Wikidata length properties only in production (allow_web).
    """
    grounded = _sources.dimensions_of(name, shape, allow_web=allow_web)
    if not grounded:
        return
    for k, fact in grounded.items():
        if k in dims:
            dims[k] = fact
            _cite_source(fact, sources, seen)


def _scale_to_typical_size(name, parts, sources, seen, allow_web):
    """Plausibility-scaling: the model gets proportions right but absolute SIZE
    wrong (a 90 kg "toaster"). If the object has NO grounded dimension but a
    typical OVERALL size is known, uniformly scale the whole construct so its
    longest extent matches that size — proportions kept, only the scale fixed.
    The scaled dims are re-provenanced to the size source (no longer pure guesses).
    """
    if any(not f.estimated for p in parts for f in p.dims.values()):
        return                                  # already has a grounded dim -> trust it
    got = _sources.typical_size_of(name, allow_web=allow_web)
    if not got:
        return
    target, src, lic = got
    from .construct import _half_extent
    xs, ys, zs = [], [], []
    for p in parts:
        if (p.shape or "").lower() == "fill":
            continue
        try:
            hx, hy, hz = _half_extent(p)
        except Exception:
            continue
        cx, cy, cz = p.center_m
        xs += [cx - hx, cx + hx]; ys += [cy - hy, cy + hy]; zs += [cz - hz, cz + hz]
    if not xs:
        return
    ext = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    if ext <= 0.0:
        return
    factor = target / ext
    if 0.7 < factor < 1.43:                     # already about the right size -> leave it
        return
    note = f"scaled to {src}"
    for p in parts:
        p.center_m = tuple(c * factor for c in p.center_m)
        for k, f in list(p.dims.items()):
            p.dims[k] = Fact(round(f.value * factor, 6), note, lic, 0.45)
        if p.outline:
            p.outline = dict(p.outline)
            p.outline["profile"] = [[u * factor, v * factor]
                                    for u, v in p.outline.get("profile", [])]
            if p.outline.get("thickness"):
                p.outline["thickness"] = p.outline["thickness"] * factor
    sources.append({"name": f"{src} — overall size (construct scaled to it)", "license": lic})


# Object classes that are mostly AIR (a thin shell), not solid material.
_HOLLOW_WORDS = {
    "toaster", "microwave", "oven", "stove", "refrigerator", "fridge", "freezer",
    "dishwasher", "washer", "washing", "dryer", "kettle", "blender", "fryer",
    "cooker", "box", "bin", "basket", "bucket", "crate", "container", "tank",
    "barrel", "drum", "bottle", "jar", "jug", "flask", "can", "carton", "bag",
    "backpack", "suitcase", "briefcase", "case", "cabinet", "drawer", "dresser",
    "wardrobe", "cupboard", "chest", "locker", "bookcase", "television", "tv",
    "monitor", "computer", "laptop", "console", "radio", "speaker", "printer",
    "router", "camera", "microwave", "mug", "cup", "bowl", "pot", "pan", "kettle",
    "vase", "teapot", "helmet", "fan", "heater", "fridge", "guitar", "violin",
}
# ...unless clearly SOLID.
_SOLID_WORDS = {
    "ball", "brick", "block", "dumbbell", "weight", "anvil", "hammer", "mallet",
    "rock", "stone", "bar", "rod", "ingot", "bullet", "coin", "dice", "die",
    "marble", "bead", "nail", "screw", "bolt", "pencil", "crayon", "sword",
    "blade", "knife", "axe", "key", "wrench", "spanner", "chisel", "magnet",
}


def _is_hollow(name: str) -> bool:
    w = {x for x in re.split(r"[^a-z]+", name.lower()) if x}
    if w & _SOLID_WORDS:
        return False
    return bool(w & _HOLLOW_WORDS)


def _hollow_pass(name, parts, sources):
    """Hollow-class objects (appliances, containers, furniture, electronics) are
    mostly air, but the model fills them solid -> absurd masses (a 300 kg
    "microwave"). Give each BULKY part an EFFECTIVE bulk density (its material
    density x the shell fraction for a thin wall), so its mass is a shell's mass,
    not a solid block's.

    We reduce the density rather than CARVE a true cavity on purpose: a 1-2 mm
    wall on a 0.3 m box falls between integration cells, so a carved shell's mass
    collapses to a resolution-dependent 0-14 kg that never passes the self-check.
    A solid body at the reduced density integrates exactly and gives a stable
    mass; the Fact is re-provenanced as an estimate, so the self-audit still
    reports density as not-grounded. Conservative: known hollow classes only,
    box/cylinder/sphere parts above 10% of the total volume.
    """
    if not _is_hollow(name):
        return
    import math
    from .construct import _shape_from
    info = []
    for p in parts:
        if p.op == "add" and (p.shape or "").lower() in ("box", "cylinder", "sphere"):
            try:
                info.append((p, _shape_from(p).volume()))
            except Exception:
                info.append((p, 0.0))
        else:
            info.append((p, 0.0))
    total = sum(v for _, v in info) or 1.0
    touched = 0
    for p, vol in info:
        if vol < 0.10 * total:                    # only the bulky parts
            continue
        d = {k: f.value for k, f in p.dims.items()}
        if not d:
            continue
        w = max(0.0008, min(0.0025, 0.01 * min(d.values())))   # thin wall ~1-2.5 mm
        shape = p.shape.lower()
        if shape == "box" and all(d[k] > 2.2 * w for k in ("x_m", "y_m", "z_m")):
            outer = d["x_m"] * d["y_m"] * d["z_m"]
            inner = (d["x_m"] - 2 * w) * (d["y_m"] - 2 * w) * (d["z_m"] - 2 * w)
        elif shape == "cylinder" and d["radius_m"] > 1.5 * w and d["height_m"] > 2.2 * w:
            outer = math.pi * d["radius_m"] ** 2 * d["height_m"]
            inner = math.pi * (d["radius_m"] - w) ** 2 * (d["height_m"] - 2 * w)
        elif shape == "sphere" and d["radius_m"] > 1.5 * w:
            outer = d["radius_m"] ** 3
            inner = (d["radius_m"] - w) ** 3
        else:
            continue
        frac = 1.0 - inner / outer if outer > 0 else 1.0
        if not (0.0 < frac < 0.9):                # not really hollow (thin/solid) -> leave it
            continue
        # An effective bulk density is a heuristic, NOT a citation -> flag it estimated
        # (so the audit reports density as not-grounded); material name stays on p.material.
        p.density = Fact(round(p.density.value * frac, 3), confidence=0.4)
        touched += 1
    if touched:
        sources.append({"name": "modeled hollow - bulk density from a thin-wall shell estimate "
                                "(appliances/containers are mostly air)", "license": ""})


def _build_vessel_spec(name: str, data: dict, model: str) -> ConstructSpec | None:
    g = data.get("geometry") or {}
    geometry: dict = {}
    for k in _VESSEL_DIMS:
        v = g.get(k)
        if not isinstance(v, (int, float)):
            return None
        v = float(v)
        if k == "fill_fraction":
            if not 0.0 <= v <= 1.0:
                return None
        elif k == "glaze_m":
            if v < 0.0:
                return None
        elif v <= 0.0:
            return None
        geometry[k] = Fact(v, "estimated", "", 0.5)   # LLM proportions => [estimated]

    R_in = (geometry["outer_radius_m"].value
            - geometry["glaze_m"].value - geometry["wall_m"].value)
    if R_in <= 0.0 or geometry["base_m"].value >= geometry["height_m"].value:
        return None

    by_name = {(L.get("name") or "").lower(): (L.get("material") or "")
               for L in data.get("layers", [])}
    defaults = {"glaze": "glaze (glassy)", "ceramic": "stoneware", "water": "liquid water"}
    thick = {
        "glaze": geometry["glaze_m"].value,
        "ceramic": geometry["wall_m"].value,
        "water": geometry["fill_fraction"].value
                 * (geometry["height_m"].value - geometry["base_m"].value),
    }
    interfaces = {"glaze": ["air", "ceramic"],
                  "ceramic": ["glaze", "air", "water"],
                  "water": ["ceramic", "air"]}

    sources = [{"name": f"{model} — researched proportions (estimates)", "license": ""}]
    seen: set = set()
    layers = []
    for layer_name in ("glaze", "ceramic", "water"):
        material = by_name.get(layer_name) or defaults[layer_name]
        dens = _density(material)
        _cite_source(dens, sources, seen)
        layers.append(SpecLayer(layer_name, material, dens,
                                Fact(thick[layer_name], "estimated", "", 0.4),
                                interfaces[layer_name]))

    return ConstructSpec(
        name=name, kind="layered_vessel", identified=True,
        geometry=geometry, layers=layers, sources=sources,
        notes=str(data.get("notes", "")) or f"Researched by {model}.",
    )


def _build_parts_spec(name: str, data: dict, model: str,
                      allow_web: bool = False) -> ConstructSpec | None:
    raw_parts = data.get("parts") or []
    if not raw_parts:
        return None
    sources = [{"name": f"{model} — researched proportions (estimates)", "license": ""}]
    seen: set = set()
    parts = []
    # Tolerant: a single unparseable part is SKIPPED, not fatal — a good
    # decomposition isn't thrown away over one odd part. (Only an all-bad spec
    # returns None, so research() can fall back.)
    def _vec3(v, default=(0.0, 0.0, 0.0)):
        try:
            t = tuple(float(x) for x in v)[:3]
            return t if len(t) == 3 else default
        except Exception:
            return default

    for i, p in enumerate(raw_parts):
        di = p.get("dims") or {}
        fill = p.get("fill") if isinstance(p.get("fill"), dict) else None
        if fill or (p.get("shape") or "").lower() == "fill":
            of = (fill or {}).get("of") or di.get("of")   # qwen sometimes puts it in dims
            if not of:
                continue                                   # a fill needs a cavity — skip it
            src = fill or di
            try:
                frac = min(1.0, max(0.0, float(src.get("fraction", 1.0))))
            except Exception:
                frac = 1.0
            gas = str(src.get("gas", "air"))
            material = p.get("material") or "liquid water"
            dens = _density(material)
            _cite_source(dens, sources, seen)
            _cite_source(_density(gas), sources, seen)
            parts.append(Part(p.get("name") or f"part{i}", "fill", {}, material, dens,
                              (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "add", None,
                              {"of": str(of), "fraction": frac, "gas": gas}))
            continue
        if (p.get("shape") or "").lower() == "outline":
            got = _sources.outline_of(name)               # Deckard supplies the researched outline
            try:
                length = float(di.get("length_m") or di.get("size_m") or 0.0)
                thick = float(di.get("thickness_m") or 0.001)
            except Exception:
                length = thick = 0.0
            if not got or not (0.0 < length <= _HUMAN_SCALE_MAX_M and thick > 0.0):
                continue                                   # no researched outline / bad size — skip it
            prof_unit, osrc, olic = got
            profile = [[u * length, v * length] for u, v in prof_unit]   # scale unit -> metres
            material = p.get("material") or "unknown"
            dens = _density(material)
            _cite_source(dens, sources, seen)
            if osrc and osrc not in seen:
                sources.append({"name": osrc, "license": olic})
                seen.add(osrc)
            parts.append(Part(p.get("name") or f"part{i}", "outline", {}, material, dens,
                              _vec3(p.get("center_m")), (0.0, 0.0, 0.0), "add", None,
                              outline={"profile": profile, "mode": "extrude", "thickness": thick}))
            continue
        shape = (p.get("shape") or "").lower()
        if shape not in _SHAPE_DIMS:
            continue                                       # unknown shape — skip it
        dims, bad = {}, False
        for k in _SHAPE_DIMS[shape]:
            v = di.get(k)
            if not isinstance(v, (int, float)) or v <= 0.0:
                bad = True
                break
            dims[k] = Fact(float(v), "estimated", "", 0.5)   # LLM => [estimated]
        if bad:
            continue                                       # missing/bad dims — skip it
        _ground_dims(name, shape, dims, sources, seen, allow_web)   # cite real dims if standard
        if any(f.value > _HUMAN_SCALE_MAX_M for f in dims.values()):
            continue                                       # not a human-scale object — skip it
        material = p.get("material") or "unknown"
        dens = _density(material)
        _cite_source(dens, sources, seen)
        op = "subtract" if str(p.get("op", "add")).lower() == "subtract" else "add"
        attach = p.get("attach") if isinstance(p.get("attach"), dict) else None
        conform = p.get("conform") if isinstance(p.get("conform"), str) else None
        parts.append(Part(p.get("name") or f"part{i}", shape, dims, material, dens,
                          _vec3(p.get("center_m")), _vec3(p.get("euler_deg")),
                          op, attach, conform=conform))

    if not parts:
        return None                                        # nothing usable — let research() fall back

    _scale_to_typical_size(name, parts, sources, seen, allow_web)   # fix absolute scale if known
    _hollow_pass(name, parts, sources)            # shell-model hollow-class objects (not solid)

    comp = _sources.composition_of(name)          # cite the documented part decomposition
    if comp and len(parts) > 1 and comp[1] and comp[1] not in seen:
        sources.append({"name": comp[1], "license": comp[2]})
        seen.add(comp[1])

    return ConstructSpec(
        name=name, kind="composite", identified=True, parts=parts, sources=sources,
        notes=str(data.get("notes", "")) or f"Researched by {model}.",
    )


def _spec_from_raw(name: str, raw, model: str, allow_web: bool) -> ConstructSpec | None:
    """Parse one raw LLM reply into a ConstructSpec, or None on bad/unknown output."""
    if not raw:
        return None
    m = _JSON.search(raw)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    kind = data.get("kind")
    if kind == "layered_vessel":
        return _build_vessel_spec(name, data, model)
    if kind == "composite" or data.get("parts"):
        return _build_parts_spec(name, data, model, allow_web)
    return None


def research_spec(name: str, *, ask=None, model: str = OLLAMA_MODEL) -> ConstructSpec | None:
    """Synthesise a cited ConstructSpec for ``name`` via the local LLM, or None.

    ``ask`` (name -> raw LLM text or None) is injectable for testing; in
    production it is local qwen and the prompt is GROUNDED with a free Wikipedia
    extract + the object's known parts. A flaky/empty reply is RETRIED once in
    production (the local 7B model occasionally returns malformed JSON or an
    empty body), so a nameable object is less likely to be dropped over a
    transient miss. Returns None on persistent no-LLM / bad output / unknown so
    ``research()`` can fall back to a flagged best-guess.
    """
    if ask is None:                       # production: local qwen, web- + structure-grounded
        q = _augment_with_web(name)
        ch = _sources.composition.hint(name)      # anchor the decomposition in known parts
        if ch:
            q = f"{q}\n\n{ch}"
        mh = _sources.shapenetsem.hint(name)      # ground the material choice (Sem ratios)
        if mh:
            q = f"{q}\n{mh}"
        runner, query, allow_web, attempts = _ask, q, True, 2     # one retry on a flaky reply
    else:                                 # injected (tests): bare name, no network, deterministic
        runner, query, allow_web, attempts = ask, name, False, 1
    for _ in range(attempts):
        spec = _spec_from_raw(name, runner(query), model, allow_web)
        if spec is not None:
            return spec
    return None


__all__ = ["research_spec", "OLLAMA_MODEL"]
