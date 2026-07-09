"""Mentat's front door — the single text input.

ONE entry, three destinations. A sentence is classified and routed to the right
subsystem, all on local models:

  • ASK      → the physics/math Q&A switchboard (the semantic interpreter, then
               Materia) — a grounded value or an honest refusal, never a fake.
  • SIMULATE → Materia compiles the sentence to a verb chain and runs it
               (right-or-refuse). If the simulation grounded a real OBJECT (via
               the Deckard shape seam), it can be RENDERED — so we offer it.
  • RENDER   → Radiance turns that grounded simulation into a watchable 3D fall
               (play button) and SAVES it for replay.

Tier: this lives at tier 4 (``mcp``) because it is the only layer allowed to
import the Q&A switchboard (mcp) AND Materia AND Radiance together — see
``tests/test_layering.py``. Materia never imports Radiance; the render hand-off
happens HERE, from the render-handle Materia hands up.

Conversational state: a simulation that can be rendered stashes a render-handle on
the ``Session``; the user's next "yes" fires the render. The single text input is
the whole protocol — no modal, no second channel.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class Session:
    """The thread of one conversation. State kept: the pending render offer
    (``render_handle``, consumed by the next "yes") and the most recent
    renderable (``last_renderable``, sticky — so "render that" still works after
    intervening turns)."""
    render_handle: dict | None = None
    last_renderable: dict | None = None
    last_intent: str | None = None


# Cue words for the deterministic classifier. Materia's own router is the real
# simulate detector (if it can compile the sentence, it's a simulation); these
# only disambiguate an explicit render and an affirmation.
_RENDER_CUES = ("render", "draw ", "show me a picture", "show me a render",
                "paint ", "visualize", "visualise")
_SIM_CUES = ("simulate", "drop ", "throw", "fall", "falls", "what happens",
             "watch it", "see what it looks like", "hits the ground",
             "terminal velocity", "how fast", "how hot")
_AFFIRM = {"yes", "y", "yeah", "yep", "yup", "ok", "okay", "sure", "do it",
           "render it", "go ahead", "please do", "yes please", "play it"}
# Effects the simulator can't model yet — name them honestly instead of silently
# substituting an impact-speed fall (a "shatter" request must not look answered).
_UNMODELED_INTENTS = ("shatter", "fracture", "smash", "crack", "explode",
                      "bounce", "splash", "spill", "melt", "burn")
_RENDER_REF_WORDS = (" that", " it", " this", " last", " previous", " again",
                     "that sim", "the sim", "the fall", "the drop", "the run")

_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "radiance", "web", "data"))
_VIEWER = "http://127.0.0.1:8765"
_OFFER = ("Do you want Radiance to render this as a watchable 3D fall? "
          "Reply 'yes'.")
# Kind-aware offers — a thermal sim promises what its render actually shows.
_OFFERS = {
    "sphere_thermal": ("Do you want Radiance to render this as a watchable fall "
                       "with live body temperature — it glows where the physics "
                       "says it glows? Reply 'yes'."),
    "conduction_field": ("Do you want Radiance to render the heat flowing "
                         "between them (a per-cell temperature field)? "
                         "Reply 'yes'."),
}


def _offer_for(handle: dict | None) -> str:
    return _OFFERS.get((handle or {}).get("kind"), _OFFER)


def _envelope(intent, *, text="", value=None, can_render=False,
              render_handle=None, saved=None, source=""):
    return {"intent": intent, "text": text, "value": value,
            "can_render": can_render, "render_handle": render_handle,
            "saved": saved, "source": source}


def _is_affirmative(text: str) -> bool:
    low = text.strip().lower().rstrip("!. ")
    return (low in _AFFIRM
            or low.split(" ", 1)[0] in {"yes", "yeah", "yep", "yup", "ok",
                                        "okay", "sure"})


def _has(text, cues):
    low = text.lower()
    return any(c in low for c in cues)


def _is_render_reference(text: str) -> bool:
    """A 'render that / render it / show the last one' back-reference: a render
    cue pointing at the PREVIOUS renderable, with no new object to compile."""
    low = text.strip().lower()
    if not _has(low, ("render", "draw", "show", "play", "watch", "replay")):
        return False
    return (any(w in low for w in _RENDER_REF_WORDS)
            or low.rstrip("!. ") in ("render", "replay", "play", "show me",
                                     "render please", "render the simulation"))


def dispatch(text: str, *, use_llm: bool = True,
             session: Session | None = None, mode: str | None = None) -> dict:
    """Classify one sentence and route it. Returns an envelope dict (keys: intent,
    text, value, can_render, render_handle, saved, source). Pass the same
    ``session`` across turns so a "yes" renders the last simulation.

    ``mode`` FORCES the lane — no classifier guess: "ask" | "simulate" | "render"
    (None/"auto" → classify, the old behavior). RENDER runs the sim AND builds the
    watchable scene in one shot — that is the path that yields a runnable .json."""
    session = session or Session()
    text = (text or "").strip()
    if not text:
        return _envelope("clarify",
                         text="Say something to ask, simulate, or render.")
    # Guard against pasted output / runaway input — a huge blob is never a real
    # request and mis-routes badly (a long paste once landed on a quantum demo).
    if len(text) > 800 or any(m in text for m in ("━━", "[routed →", "self-check ✓")):
        return _envelope("clarify",
            text=("That looks like pasted output — give me a short request, e.g. "
                  "“drop a brick from 3 m” or “what's the escape velocity of Mars?”"))
    mode = (mode or "").strip().lower() or None
    if mode == "auto":
        mode = None

    # 1. A bare "yes" confirms the pending render offer — regardless of which lane
    #    flag was prepended. Replying to "Reply 'yes'", users routinely type
    #    "/simulate yes"; "yes" is never a real ask/sim query. (The bug this fixes:
    #    "/simulate yes" used to re-route "yes" → the model hallucinated the demo
    #    example instead of rendering the sim just computed.) ASK is the exception.
    if session.render_handle and _is_affirmative(text) and mode != "ask":
        handle = session.render_handle
        session.render_handle = None
        env = _render_from_handle(handle)
        session.last_intent = env["intent"]
        return env

    # 1b. "render that / render it / show the last one" → re-render the most recent
    #     renderable, even after intervening turns cleared the live offer.
    if session.last_renderable and _is_render_reference(text) \
            and mode in (None, "render"):
        env = _render_from_handle(session.last_renderable)
        session.last_intent = env["intent"]
        return env

    # Explicit ASK — force the Q&A switchboard; never attempt a sim.
    if mode == "ask":
        env = _ask(text)
        session.last_intent = env["intent"]
        return env

    # 2. Simulation — an OBJECT IN MOTION. Materia's own router compiles it
    #    (deterministic first; the qwen residual when the lane is forced to
    #    sim/render, or the sentence is sim-cued). A runnable spec is a SIMULATION
    #    only when it concerns a physical object/motion — a domain *report* verb
    #    that happens to be routable is a fact and belongs to ASK (auto mode).
    from sigma_ground import materia
    from sigma_ground.materia import translator as _t
    forced = mode in ("simulate", "render")
    spec = materia.translate(text, use_qwen=False)
    if not spec.is_runnable() and use_llm and (forced or _has(text, _SIM_CUES)):
        spec = materia.translate(text, use_qwen=True)
    verbs = {st.verb for st in spec.steps}
    # Fix A — a drop with no grounded object must not silently default to a
    # stand-in (an anvil). Name the user's word honestly and refuse the drop.
    if "drop_object" in verbs and not any(
            st.params.get("object_name") for st in spec.steps
            if st.verb == "drop_object"):
        cand = _t._candidate_object(text.lower())
        msg = (f"I don't have a {cand!r} shape — give me an object I can ground "
               f"(e.g. an anvil, a brick, a feather)." if cand else
               "Name an object I can ground to drop (e.g. an anvil, a brick, "
               "a feather).")
        env = _envelope("simulate", text=msg, can_render=False, source=spec.source)
        session.last_intent = env["intent"]
        return env
    # Forced sim/render skips the object-context gate (the user already said it's
    # a sim); auto requires object context before it calls something a simulation.
    # contact_conduction is objects-in-contact by definition (a hot block ON a
    # cold slab) — like drop_object, the verb itself IS the object context.
    is_sim = spec.is_runnable() and (forced or "drop_object" in verbs
                                     or "contact_conduction" in verbs
                                     or _t._has_object_context(text.lower()))
    if is_sim:
        env = _run_simulation(text, spec, use_llm=use_llm, session=session)
        # RENDER mode (or an explicit "render"/"draw" cue) → build the watchable
        # scene right now, one shot — no "reply yes" round-trip.
        if env.get("render_handle") and (mode == "render" or _has(text, _RENDER_CUES)):
            handle = env["render_handle"]
            session.render_handle = None
            env = _render_from_handle(handle)
        elif mode == "render" and not env.get("render_handle") \
                and env.get("intent") == "simulate":
            env = {**env, "text": (env.get("text") or "")
                   + "\n\n(Nothing to render here — this one's a number, not a "
                     "moving object.)"}
        session.last_intent = env["intent"]
        return env

    # A forced sim/render that wouldn't compile → refuse honestly, don't fall to ASK.
    if forced:
        msg = (getattr(spec, "note", "") or
               "I couldn't compile that into a simulation. Name an object and a "
               "height — e.g. 'drop a feather from 8 feet'.")
        env = _envelope("simulate", text=msg, can_render=False, source=spec.source)
        session.last_intent = env["intent"]
        return env

    # 3. Auto mode, not a sim → the Q&A switchboard.
    env = _ask(text)
    session.last_intent = env["intent"]
    return env


def _run_simulation(text, spec, *, use_llm, session) -> dict:
    """Run a compiled Materia spec under the infallibility gate, narrate it, and —
    if it grounded a real object — stash a render-handle and offer the render."""
    from sigma_ground import materia
    from sigma_ground.materia import groundedness as _g, translator as _t

    # Route-consistency gate for the LLM residual (deterministic routes are
    # already reliable and skip it): an uncertain route → refuse, never guess.
    if use_llm and spec.source == "qwen" and not _t._route_consistent(
            text, use_qwen=use_llm):
        v = _g.Verdict(False, _g.CROSS_CHECK_FAILED,
                       "router not self-consistent — uncertain which simulation")
        return _envelope("clarify", text=_g.refuse_text(v), source=spec.source)

    results = materia.run_spec(spec)
    verdict = _g.gate_results(results)
    if not verdict.grounded:                      # right-or-refuse
        return _envelope("simulate", text=_g.refuse_text(verdict),
                         can_render=False, source=spec.source)

    routes = "; ".join(
        f"{s.verb}(" + ", ".join(f"{k}={v}" for k, v in s.params.items()) + ")"
        for s in spec.steps)
    body = "\n\n".join(r.render() for r in results)
    narration = f"[routed → {routes}  via {spec.source}]\n\n" + body

    # Did the sim ground a real OBJECT (the Deckard shape seam)? Then it's
    # renderable — offer it and remember how.
    handle = next((r.outputs.get("render_handle") for r in results
                   if r.outputs.get("can_render")
                   and r.outputs.get("render_handle")), None)
    # Honest caveat: if the user asked for an effect we don't model yet (shatter,
    # bounce, …), say so — don't let the impact-speed fall masquerade as the answer.
    unmodeled = next((w for w in _UNMODELED_INTENTS if w in text.lower()), None)
    if unmodeled:
        narration += (f"\n\n(Note — I can't model **{unmodeled}** yet: this is the "
                      f"fall and impact speed; any render shows the drop, not the "
                      f"{unmodeled}.)")
    if handle:
        session.render_handle = handle
        session.last_renderable = handle      # sticky → enables "render that" later
        narration += "\n\n" + _offer_for(handle)
    return _envelope("simulate", text=narration, can_render=bool(handle),
                     render_handle=handle, source=spec.source)


def _render_from_handle(handle: dict) -> dict:
    """Turn a Materia render-handle into a watchable, saved Radiance fall.

    Two kinds: a generic SPHERE renders natively from its material + radius
    (radiance.record_fall — no Deckard); a NAMED object has Deckard compile its
    real shape, then record_object_fall drops it. Either way we never fake a shape.
    """
    kind = handle.get("kind")
    if kind == "conduction_field":           # per-cell heat flow between solids
        from sigma_ground.radiance import record_thermal_field
        bundle = record_thermal_field(handle)
        return _announce_render(handle.get("label", "contact conduction"), bundle)
    if kind == "sphere_thermal":             # drag-heated fall: frames carry T(t)
        from sigma_ground.radiance import record_fall_thermal
        label = handle.get("label", "sphere")
        bundle = record_fall_thermal(
            handle.get("material_key", "iron"),
            handle.get("radius_m", 0.05),
            handle.get("start_altitude_m", 30_000.0),
            body_fraction=handle.get("body_fraction", 1.0),
            T0=handle.get("T0", 288.15),
            expected_delta_T_K=handle.get("expected_delta_T_K"),
            windward_field=True)             # the flagship: leading-face glow
        return _announce_render("heating " + label, bundle)
    if kind == "sphere":
        from sigma_ground.radiance import record_fall
        label = handle.get("label", "sphere")
        bundle = record_fall(handle.get("material_key", "iron"),
                             radius_m=handle.get("radius_m", 0.05),
                             start_altitude_m=handle.get("start_altitude_m", 1000.0))
        bundle["kind"] = "trajectory"        # record_fall omits it; the viewer needs it
        return _announce_render("falling " + label, bundle)
    if kind == "launch_arc":                 # whole arc: up, apex, down, bounce
        from sigma_ground.radiance import record_fall
        label = handle.get("label", "launched ball")
        r = handle.get("radius_m", 0.05)
        bundle = record_fall(handle.get("material_key", "steel_mild"),
                             radius_m=r, start_altitude_m=r,
                             v0_m_s=handle.get("launch_speed_m_s", 30.0),
                             dt_max=0.004, frame_dt=0.04)
        bundle["kind"] = "trajectory"
        return _announce_render(label, bundle)
    if kind == "descent":                    # the same drag body the verb integrated
        from sigma_ground.radiance import record_descent
        bundle = record_descent(
            payload_mass_kg=handle.get("payload_mass_kg", 118.0),
            drag_area_m2=handle.get("drag_area_m2", 0.28),
            cd=handle.get("cd", 0.70),
            start_altitude_m=handle.get("start_altitude_m", 35_000.0))
        return _announce_render(handle.get("label", "high-altitude descent"), bundle)
    if kind == "horizontal":                 # transonic slug along +x
        from sigma_ground.radiance import record_horizontal_run
        bundle = record_horizontal_run(
            mass_kg=handle.get("mass_kg", 0.02),
            diameter_m=handle.get("diameter_m", 0.01),
            launch_mach=handle.get("launch_mach", 2.5))
        return _announce_render(handle.get("label", "supersonic projectile"), bundle)

    from sigma_ground import deckard
    from sigma_ground.radiance import record_object_fall
    name = handle.get("object_name", "object")
    try:
        construct = deckard.identify(name)        # catalogue hit = instant
    except Exception:
        construct = None
    if construct is None or not getattr(construct, "identified", False):
        return _envelope("render", source="deckard",
                         text=f"Couldn't ground the shape of {name!r} to render.")
    bundle = record_object_fall(construct,
                                handle.get("start_altitude_m", 2.4384),
                                cd=handle.get("cd", 1.0))
    return _announce_render("falling " + name, bundle)


def _announce_render(title: str, bundle: dict) -> dict:
    """Save a render bundle to the viewer's data dir; return the render envelope."""
    from sigma_ground.deckard import catalog
    slug = catalog.slugify(title)
    path = _save_bundle(slug, bundle)
    url = f"{_VIEWER}/?scene={slug}"
    tr = bundle["trajectory"]
    span = (f"ground in {tr['natural_timescale_s']:.2f} s"
            if any(f.get("bodies") for f in tr["frames"])
            else f"spanning {tr['natural_timescale_s']:.2f} s")   # static solids, field-only playback
    text = (f"Rendered “{title}” — {len(tr['frames'])} frames, {span}; "
            f"saved for replay. Serve with "
            f"`python -m sigma_ground.radiance.web.serve`, then open {url} and "
            f"press ▶.")
    v = tr.get("validation") or {}
    if bundle.get("thermal") and v.get("delta_T_final_K") is not None:
        try:
            from sigma_ground.field.interface.thermal import is_visibly_glowing
            frames = bundle["trajectory"]["frames"]
            peak = max(b.get("temperature_k", 0.0)
                       for f in frames for b in f["bodies"])
            glow = ("glows visibly" if is_visibly_glowing(peak)
                    else "stays below the visible-glow (Draper) threshold")
            text += (f"\nBody temperature rides the frames: peak {peak:.0f} K — "
                     f"{glow} [{v.get('body_fraction_flag', 'f flagged')}]; "
                     f"recorder ΔT cross-check residual "
                     f"{(v.get('thermal_residual') or 0.0) * 100:.2f}%.")
        except Exception:
            pass                              # the announce line never blocks a render
    return _envelope("render", text=text, can_render=False,
                     saved={"slug": slug, "path": path, "url": url, "title": title},
                     source="radiance")


def _save_bundle(slug: str, bundle: dict) -> str:
    os.makedirs(_DATA_DIR, exist_ok=True)
    path = os.path.join(_DATA_DIR, slug + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=1)
    return path


def _ask(text: str) -> dict:
    """A physics/math question → the Q&A switchboard: the semantic interpreter
    first (grounded, conservative — None when unsure), then Materia. Honest
    refusal over fabrication."""
    try:
        from sigma_ground.mcp.benchmark import interpreter_demo
        a = interpreter_demo.semantic_answer(text)
    except Exception:
        a = None
    if a:
        # semantic_answer returns a benchmark-record dict (display string in
        # `answer_text`, number in `extracted_value`). The envelope contract is
        # text=str, value=number — unpack it so clients never see a raw dict.
        if isinstance(a, dict):
            txt = a.get("answer_text") or str(a)
            val = a.get("extracted_value")
        else:
            txt, val = str(a), None
        if txt and txt.strip():
            return _envelope("ask", text=txt, value=val, source="semantic-interpreter")
    from sigma_ground import materia
    a2 = materia.answer(text, use_qwen=False)     # clarifies if it isn't a sim
    return _envelope("ask", text=a2, value=None, source="materia")


__all__ = ["dispatch", "Session"]
