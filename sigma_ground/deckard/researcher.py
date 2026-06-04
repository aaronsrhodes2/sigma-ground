"""Deckard's Researcher — synthesise a cited ConstructSpec for an unknown name.

Primary: Gemini-free (``google-generativeai`` with the dev-root ``.env`` key);
fallback: local qwen via ollama ``/api/chat``. (The env-load + call are inlined
rather than imported from the mcp layer, which sits far above deckard in the
role tiers.) The LLM proposes the object's *shape* — either a hollow
``layered_vessel`` or a ``composite`` of solid primitives — plus material names;
Deckard GROUNDS each density in our own data (``sources.local``) and flags
LLM-proposed dimensions as ``[estimated]``. Offline or on any failure it returns
None, so ``research()`` falls back to a flagged best-guess — never a fake.

Facts-first, facts-and-primitives only: no images, no meshes — the LLM supplies
proportions, our data supplies attributable densities.
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
GEMINI_MODEL = os.environ.get("DECKARD_GEMINI_MODEL", "gemini-2.5-flash")

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
    '"material":"<material>","center_m":[x,y,z]}]}\n'
    "   dims: sphere {radius_m}; cylinder/cone {radius_m,height_m}; box "
    "{x_m,y_m,z_m}; torus {major_radius_m,minor_radius_m}; ellipsoid "
    "{rx_m,ry_m,rz_m}. One part for a simple solid; several with center_m "
    "offsets for a compound object (hammer = handle cylinder + head box).\n"
    "Use realistic typical dimensions and a real material name (steel, glass, "
    'aluminium, oak, stoneware, ...). If you cannot, output {"kind":"unknown"}.'
)


def _load_dev_env() -> None:
    """Best-effort: load KEY=VALUE from the nearest ancestor .env (no override).

    Mirrors the dev-root .env convention without importing the mcp layer (which
    would invert the role tiers — deckard sits well below mcp).
    """
    import pathlib
    cwd = pathlib.Path.cwd().resolve()
    pkg = pathlib.Path(__file__).resolve()
    seen: set = set()
    for d in [cwd, *cwd.parents, *pkg.parents]:
        p = d / ".env"
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        except Exception:
            pass


def _gemini(name: str) -> str | None:
    """Ask Gemini-free; return raw text or None (no key / error / offline)."""
    key = os.environ.get("GEMINI_FREE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        _load_dev_env()
        key = os.environ.get("GEMINI_FREE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=_SYS)
        resp = model.generate_content(name)
        return (getattr(resp, "text", "") or "") or None
    except Exception:
        return None


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
    """Default LLM: Gemini-free primary, local qwen fallback."""
    return _gemini(name) or _qwen(name)


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


def _build_parts_spec(name: str, data: dict, model: str) -> ConstructSpec | None:
    raw_parts = data.get("parts") or []
    if not raw_parts:
        return None
    sources = [{"name": f"{model} — researched proportions (estimates)", "license": ""}]
    seen: set = set()
    parts = []
    for i, p in enumerate(raw_parts):
        shape = (p.get("shape") or "").lower()
        if shape not in _SHAPE_DIMS:
            return None
        dims_in = p.get("dims") or {}
        dims = {}
        for k in _SHAPE_DIMS[shape]:
            v = dims_in.get(k)
            if not isinstance(v, (int, float)) or v <= 0.0:
                return None
            dims[k] = Fact(float(v), "estimated", "", 0.5)   # LLM => [estimated]
        material = p.get("material") or "unknown"
        dens = _density(material)
        _cite_source(dens, sources, seen)
        center = p.get("center_m", (0.0, 0.0, 0.0))
        try:
            center = tuple(float(x) for x in center)[:3]
            if len(center) != 3:
                center = (0.0, 0.0, 0.0)
        except Exception:
            center = (0.0, 0.0, 0.0)
        parts.append(Part(p.get("name") or f"part{i}", shape, dims, material, dens, center))

    return ConstructSpec(
        name=name, kind="composite", identified=True, parts=parts, sources=sources,
        notes=str(data.get("notes", "")) or f"Researched by {model}.",
    )


def research_spec(name: str, *, ask=None, model: str = GEMINI_MODEL) -> ConstructSpec | None:
    """Synthesise a cited ConstructSpec for ``name`` via the LLM, or None.

    ``ask`` (name -> raw LLM text or None) is injectable for testing; it defaults
    to Gemini-free → qwen. Dispatches on the proposed kind (vessel or composite).
    Returns None on no-LLM / bad output / unknown so research() can fall back to a
    flagged best-guess.
    """
    raw = (ask or _ask)(name)
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
        return _build_parts_spec(name, data, model)
    return None


__all__ = ["research_spec", "GEMINI_MODEL", "OLLAMA_MODEL"]
