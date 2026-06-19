"""Radiance ↔ Deckard — the SHAPE seam.

Radiance never invents a silhouette. When a simulation needs a real object — a
coffee cup, an anvil, a wine glass — it asks **Deckard** to compile that *name*
into grounded matter (a multi-part Construct: real geometry, mass, centre-of-mass,
and per-part substances), then bakes the Construct into a SceneSpec whose every
colour is *derived* from each substance's optical constants
(:func:`scene_export.construct_to_scene`).

This is the render-side counterpart to :mod:`materia.deckard_bridge`, which asks
Deckard only for the motion handful (mass / inertia / footprint). Here we want
the GEOMETRY *and* the material optics, ready to render — Deckard already solves
the fill/combine (cavity-fill, layering); Radiance's job is the render
integration, not re-solving the physics.

Infallibility (the Golden Goal — *physics to screen, nothing in between*): if
Deckard can't GROUND the shape, this returns ``None`` and the caller refuses. We
never fabricate a stand-in shape for an object we couldn't research — an honest
refusal beats a confident fake.
"""
from __future__ import annotations

# Compiling a novel object can take the LLM research path (~60 s) the first time;
# a repeat ask for the same (name, hint) should be instant. In-process, ephemeral.
_CACHE: dict[tuple, dict | None] = {}


def request_shape_scene(name: str, material_hint: str | None = None, *,
                        allow_llm: bool = True, resolution: int = 64) -> dict | None:
    """Ask Deckard to compile ``name`` into grounded matter, then bake it into a
    render-ready SceneSpec (``csg_leaves`` + per-part ``materials`` with derived
    optics + ``physics`` + ``camera``/``lights``).

    Returns the SceneSpec ``dict`` when Deckard GROUNDED the shape, or ``None``
    when it could not — the caller then refuses rather than render a fabricated
    silhouette.

    ``material_hint`` (e.g. ``"wooden"``, ``"cast iron"``) biases Deckard's frame
    material when the name alone is ambiguous; it SELECTS a cited substance, it
    does not invent one. ``allow_llm=False`` forces the deterministic path
    (catalogue hit, else a flagged best-guess → refused here) for offline callers
    and tests.
    """
    key = ((name or "").strip().lower(), (material_hint or "").strip().lower(),
           bool(allow_llm), int(resolution))
    if key in _CACHE:
        return _CACHE[key]

    scene: dict | None = None
    try:
        from sigma_ground import deckard
        from .scene_export import construct_to_scene
        # identify() does not thread material_hint, so replicate its two beats
        # when a hint is supplied: research the (hint-biased) spec → compile to
        # matter. Without a hint, identify() is the canonical one-call path.
        if material_hint:
            spec = deckard.research(name, allow_llm=allow_llm, material_hint=material_hint)
            construct = deckard.compile(spec, resolution=resolution)
        else:
            construct = deckard.identify(name, resolution=resolution, allow_llm=allow_llm)
        # Only render what Deckard actually grounded (Construct.identified) — a
        # flagged best-guess is a refusal here, not a silhouette to fake.
        if construct is not None and getattr(construct, "identified", False):
            scene = construct_to_scene(construct)        # bakes per-part optics
    except Exception:
        scene = None

    _CACHE[key] = scene
    return scene
