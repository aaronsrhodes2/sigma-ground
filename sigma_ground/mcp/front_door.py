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
import re
from dataclasses import dataclass


@dataclass
class Session:
    """The thread of one conversation. State kept: the pending render offer
    (``render_handle``, consumed by the next "yes"), the most recent
    renderable (``last_renderable``, sticky — so "render that" still works after
    intervening turns), and a pending clarification (``pending_clarification``,
    consumed by the next turn's reply — the same "remember something across a
    turn boundary" pattern as render_handle, just for "which material?"
    instead of "render it?")."""
    render_handle: dict | None = None
    last_renderable: dict | None = None
    last_intent: str | None = None
    pending_clarification: dict | None = None


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

# Cue phrases for an OBJECT/SCENE build request — "show me a coffee cup",
# "build a rocking chair", "I want a simulation of a doughnut, with frosting
# and rainbow sprinkles" — routed to the Deckard SHAPE seam
# (radiance.ask_deckard.request_shape_scene), never Materia's physics verbs.
# Same convention as _RENDER_CUES/_SIM_CUES just above: a plain lowercase
# substring, checked with _has(). Checked only in dispatch() step 2b, AFTER
# Materia's own verb router has had first refusal on the sentence — so an
# existing physics phrasing that already compiles (a fall, a bell strike, a
# pliers actuation…) is never pre-empted; see the note at that call site.
_SCENE_CUES = (
    "i want a simulation of", "i want a model of", "i want a scene of",
    "i want a render of", "simulation of",
    "show me", "build me", "build a", "build an",
    "render me", "render a", "render an",
    "make me", "generate a", "generate an", "generate me",
    "create a", "create an", "create me",
    "what does", "what would",
    "visualize a", "visualize an", "visualise a", "visualise an",
)
# The object NAME ends at the first clause boundary — ", with" / ", which" /
# ". I want" / or the sentence just ends. Deliberately simple (no parser): cut
# the name off at the first of these, then drop a trailing "look(s) like"
# ("what does X look like") and a leading article.
_SCENE_BOUNDARY = re.compile(
    r",\s*(?:with|which|that|sitting|resting|standing|placed|showing)\b"
    r"|\.\s*(?:i\ want|it\ should|please)\b"
    r"|[.!?]",
    re.IGNORECASE)
_SCENE_TRAILING_FILLER = re.compile(r"\s*(?:looks?\s+like)\s*$", re.IGNORECASE)
_SCENE_LEADING_ARTICLE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)

# A multi-OBJECT stack: "X sitting on Y, which is on top of Z" — each clause
# names one SUPPORT object underneath the one before it. First clause is
# introduced by a stacking verb ("sitting/resting/standing/placed on");
# every subsequent clause is chained by ", which/that is (flat) on (top of) …".
# Deliberately simple (no parser), same discipline as _SCENE_BOUNDARY —
# only recognizes THIS specific "A sitting on B, which is on C" shape, not
# arbitrary prepositional nesting.
_STACK_FIRST_RE = re.compile(
    r"\b(?:sitting|resting|standing|placed|sits|rests|stands)\s+on\s+"
    r"(?:top\s+of\s+)?(?:a|an|the)?\s*([^,.!?]+)", re.IGNORECASE)
_STACK_CHAIN_RE = re.compile(
    r",\s*(?:which|that)\s+is\s+(?:flat\s+)?(?:on\s+top\s+of|on)\s+"
    r"(?:a|an|the)?\s*([^,.!?]+)", re.IGNORECASE)


def _extract_scene_stack(text: str) -> list[str] | None:
    """A bottom-to-top list of object names for a multi-object stack request
    ("a doughnut … sitting on a plate, which is flat on top of a 1700s
    table"), or None if the text doesn't match this specific shape.

    The TOP of the stack is extracted from the text BEFORE the stacking verb
    (not the whole sentence) — _extract_scene_name's own boundary only cuts
    at a COMMA-prefixed "sitting"/"resting"/etc. ("...Doughnut, with icing...
    sitting on..." — the Captain's actual phrasing), so a comma-less variant
    ("a mug sitting on a table") would otherwise swallow the stacking verb
    into the base name. Slicing at the verb's start sidesteps that instead
    of tightening the comma rule (which stays correct for its own job:
    cutting at ", with"/", which"/etc. in front-loaded decoration clauses).
    Each _STACK_*_RE match names one SUPPORT underneath the top, in sentence
    order (nearest support first) — reversed here so the return value is
    bottom-to-top, matching radiance.ask_deckard.compose_scene's contract."""
    m = _STACK_FIRST_RE.search(text)
    if m is None:
        return None
    top = _extract_scene_name(text[:m.start()])
    if top is None:
        return None
    supports = [m.group(1).strip(" .,!?")]
    rest = text[m.end():]
    while True:
        m2 = _STACK_CHAIN_RE.search(rest)
        if m2 is None:
            break
        supports.append(m2.group(1).strip(" .,!?"))
        rest = rest[m2.end():]
    return list(reversed(supports)) + [top]

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


def _extract_scene_name(text: str) -> str | None:
    """The object NAME in a scene-build request: everything after the matched
    _SCENE_CUES trigger phrase, cut at the first clause boundary (", with" /
    ", which" / ". I want" / a sentence end) and any trailing "look(s) like".
    Simple and testable on purpose — not a full NLP parser. Returns None when
    no cue matched (the caller then falls through to the Q&A switchboard, same
    as before this route existed)."""
    low = text.lower()
    best_idx, best_len = None, 0
    for cue in _SCENE_CUES:
        idx = low.find(cue)
        if idx != -1 and (best_idx is None or idx < best_idx
                          or (idx == best_idx and len(cue) > best_len)):
            best_idx, best_len = idx, len(cue)
    if best_idx is None:
        return None
    tail = text[best_idx + best_len:]
    m = _SCENE_BOUNDARY.search(tail)
    piece = tail[:m.start()] if m else tail
    piece = _SCENE_TRAILING_FILLER.sub("", piece)
    piece = piece.strip(" .,!?")
    piece = _SCENE_LEADING_ARTICLE.sub("", piece).strip()
    return piece or None


def _render_scene(name: str, *, use_llm: bool) -> dict:
    """Ask Deckard to compile ``name`` into grounded matter and stage a
    watchable render of it — the Radiance⇄Deckard shape seam
    (ask_deckard.request_shape_scene), invoked directly. Unlike a Materia
    simulation this never runs physics: a "build/show me X" request wants to
    SEE the object, not watch it fall, so it renders in one shot (no "reply
    yes" round-trip). Infallibility: a name Deckard can't ground is an honest
    refusal, never a fabricated silhouette (ask_deckard's own contract)."""
    from sigma_ground.radiance.ask_deckard import request_shape_scene
    from sigma_ground.radiance.theaters import stage
    scene = request_shape_scene(name, allow_llm=use_llm)
    if scene is None:
        text = (f"I couldn't ground the shape of {name!r} — Deckard can only "
                f"build objects it can research and cite. Try a more common "
                f"name, or describe it differently.")
        return _envelope("simulate", text=text, can_render=False, source="deckard")
    staged = stage(scene, "room")
    return _announce_scene_render(name, staged)


def _render_scene_stack(items: list[str], *, use_llm: bool) -> dict:
    """Ask Deckard to compile and stack MULTIPLE independently-researched
    objects into one scene (radiance.ask_deckard.compose_scene) — the
    multi-object counterpart of _render_scene. ``items`` is bottom-to-top
    (base first). Same infallibility contract: any object in the chain that
    doesn't ground refuses the WHOLE stack rather than silently dropping it."""
    from sigma_ground.radiance.ask_deckard import compose_scene
    from sigma_ground.radiance.theaters import stage
    scene = compose_scene(items, allow_llm=use_llm)
    if scene is None:
        chain = " -> ".join(items)
        text = (f"I couldn't ground the full stack ({chain}) — at least one "
                f"of these objects didn't research/compile cleanly. Deckard "
                f"refuses the whole scene rather than silently dropping the "
                f"piece it couldn't ground.")
        return _envelope("simulate", text=text, can_render=False, source="deckard")
    staged = stage(scene, "room")
    title = " on ".join(reversed(items))
    return _announce_scene_render(title, staged)


def _announce_scene_render(name: str, scene: dict) -> dict:
    """Save a staged SceneSpec (a bare scene dict, no trajectory — the same
    on-disk shape as the curated obj_*/theater_*.json gallery entries) to the
    viewer's data dir; return the render envelope."""
    from sigma_ground.deckard import catalog
    slug = "scene_" + catalog.slugify(name)
    path = _save_bundle(slug, scene)
    url = f"{_VIEWER}/?scene={slug}"
    n_parts = len(scene.get("csg_leaves", []))
    text = (f"Built the scene for “{name}” — {n_parts} part"
            f"{'s' if n_parts != 1 else ''}, staged and lit; saved for "
            f"viewing. Serve with `python -m sigma_ground.radiance.web.serve`, "
            f"then open {url}.")
    return _envelope("render", text=text, can_render=False,
                     saved={"slug": slug, "path": path, "url": url, "title": name},
                     source="radiance")


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

    # 1d. A push/shove follow-up against the stack the conversation just built:
    #     "now push it off the table", "push the plate off". Requires the
    #     memory from the scene-stack route above — the verb alone matches no
    #     Materia trigger, so before that memory existed this fell through to
    #     the canned physics clarify (confirmed live before building this).
    if session.last_renderable \
            and session.last_renderable.get("kind") == "scene_stack" \
            and _match_applied_kind(text) and mode != "ask":
        akind = _match_applied_kind(text)
        if akind not in _IMPLEMENTED_KINDS:
            pretty = {"impulse_twist": "an impulse-with-twist (tip/spin)",
                      "force": "a sustained force (press/drag)",
                      "tilt": "a tilt (re-orienting the scene vs gravity)",
                      "hold": "a hold (an artificial pin)",
                      "source": "a heat source"}.get(akind, akind)
            session.last_intent = "clarify"
            return {"intent": "clarify", "source": "mentat",
                    "can_render": False, "saved": None, "value": None,
                    "text": (f"\u2753 that's {pretty} \u2014 a defined member "
                             f"of the closed artificial-force set (Golden "
                             f"Rule 7), but its solver hasn't landed yet. "
                             f"Today I can apply an IMPULSE: push, shove, "
                             f"knock or slide an object off its support.")}
        items = session.last_renderable["items"]
        idx = _resolve_push_target(text, items)
        if idx is not None:
            env = _render_push(items, idx, use_llm=use_llm,
                               push_speed=_parse_push_speed(text))
            session.last_intent = env["intent"]
            return env
        # push verb + a remembered stack, but the target didn't resolve (or
        # names the base): an honest, specific clarify beats the generic one
        pushable = ", ".join(items[1:]) or "nothing"
        session.last_intent = "clarify"
        return {"intent": "clarify", "source": "mentat", "can_render": False,
                "saved": None, "value": None,
                "text": (f"❓ I can push {pushable} off the {items[0]} — the "
                         f"{items[0]} is the base of the stack, so pushing IT "
                         f"is a different simulation (tipping) I don't claim "
                         f"yet. Which one should I push?")}

    # 1c. A pending clarification ("which material?") — this turn's reply answers
    #     it. Concatenate with the ORIGINAL question and re-dispatch: whatever
    #     extraction failed the first time (_extract_material, etc.) gets another
    #     pass over the combined text, so no per-slot patching logic is needed
    #     here — it reuses the exact same routing/extraction path as a fresh
    #     question. Same "remember across a turn" shape as 1/1b above, for a
    #     targeted follow-up instead of a render offer.
    if session.pending_clarification and mode != "ask":
        pending = session.pending_clarification
        session.pending_clarification = None
        combined = f"{pending['question']} {text.strip()}"
        return dispatch(combined, use_llm=use_llm, session=session, mode=mode)

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
    # actuate_hand_tool/strike_bell are the same: a named pliers/bell/tongs IS
    # the object, even though it's not in the generic falling-object noun list.
    is_sim = spec.is_runnable() and (forced or "drop_object" in verbs
                                     or "contact_conduction" in verbs
                                     or "actuate_hand_tool" in verbs
                                     or "strike_bell" in verbs
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
        if getattr(spec, "missing_slot", ""):
            session.pending_clarification = {"question": text}
        env = _envelope("simulate", text=msg, can_render=False, source=spec.source)
        session.last_intent = env["intent"]
        return env

    # 2b. Not a physics simulation — is it an OBJECT/SCENE build request?
    #     ("show me a coffee cup", "build a rocking chair", "I want a
    #     simulation of a doughnut, with frosting and rainbow sprinkles…").
    #     Checked ONLY here — after Materia's own verb router (steps 1-2 above)
    #     has had first refusal on the sentence, and only in auto mode (an
    #     explicit mode="simulate"/"render" already refused honestly just
    #     above and keeps that contract unchanged) — so an existing physics
    #     phrasing that already compiles (a fall, a bell strike, a pliers
    #     actuation…) is NEVER pre-empted by this route; it only catches the
    #     residual Materia declines.
    if mode is None:
        stack = _extract_scene_stack(text)
        if stack:
            env = _render_scene_stack(stack, use_llm=use_llm)
            session.last_intent = env["intent"]
            if env.get("saved"):
                # the conversation now REMEMBERS what it built — the missing
                # half of "now push it off the table" (tested live 2026-07-30:
                # this was None after a successful stack render, so the
                # follow-up had nothing to bind "it" against)
                session.last_renderable = {"kind": "scene_stack",
                                           "items": list(stack)}
            return env
        name = _extract_scene_name(text)
        if name:
            env = _render_scene(name, use_llm=use_llm)
            session.last_intent = env["intent"]
            return env

    # 3. Auto mode, not a sim → the Q&A switchboard.
    env = _ask(text, session=session)
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


# verb -> artificial-force kind (dynamics.mechanisms.applied.KINDS — the
# CLOSED set; Golden Rule 7's whole vocabulary). One table, not one handler
# per verb: a new phrasing is a new ROW; a new KIND is a doctrine amendment.
# Kinds whose solver hasn't landed refuse honestly, naming themselves.
_VERB_KINDS = (
    # order = specificity: the generic impulse verbs go LAST, so "tilt the
    # table until the plate slides" reads as a TILT (the commanded action),
    # not an impulse (the hoped-for consequence).
    (re.compile(r"\btilt(?:s|ed)?\b", re.I), "tilt"),
    (re.compile(r"\b(hold|pin|freeze)\b.*\bin place\b", re.I), "hold"),
    (re.compile(r"\b(heat|torch|warm)\s+(it|the)\b", re.I), "source"),
    (re.compile(r"\b(tip|topple|flip)(?:s|ped)?\b", re.I), "impulse_twist"),
    (re.compile(r"\b(spin|twist|rotate)(?:s|ted|d)?\b", re.I), "impulse_twist"),
    (re.compile(r"\b(press|squeeze|drag)(?:es|ed|s)?\b", re.I), "force"),
    (re.compile(r"\b(push|shove|knock|slide)(?:es|ed|s)?\b", re.I), "impulse"),
)
_IMPLEMENTED_KINDS = {"impulse"}         # the stack solver's current reach
_PUSH_RE = _VERB_KINDS[-1][0]            # back-compat alias: the impulse verbs


def _match_applied_kind(text: str):
    for rx, kind in _VERB_KINDS:
        if rx.search(text):
            return kind
    return None


def _resolve_push_target(text: str, items: list) -> int | None:
    """Which stacked object is being pushed? A named match wins ("push the
    PLATE off"); a bare pronoun ("push it off") takes the object directly on
    the base — the natural referent of a stack. The base itself can't be
    pushed (index 0 is what everything stands on); pushing it is a different
    simulation (tipping the table) this route honestly doesn't claim."""
    low = text.lower()
    for i in range(len(items) - 1, 0, -1):            # topmost named match wins
        words = [w for w in re.split(r"[^a-z]+", items[i].lower()) if len(w) > 2]
        if any(re.search(r"\b" + re.escape(w) + r"\b", low) for w in words):
            return i
    if re.search(r"\bit\b", low) and len(items) >= 2:
        return 1
    return None


# how hard is "push"? Verbal intensities map onto the ONE artificial force in
# the scene. Golden Rule 7 (Captain, 2026-07-30): only artificial forces are
# adjustable DURING a simulation — matter is fact. So the exposed knob (these
# words, and the viewer's variant slider) is the push speed, never a mu.
_PUSH_SPEEDS = (1.2, 1.7, 2.2, 3.0)                  # the variant sweep, m/s
_PUSH_WORDS = (("gently", 1.2), ("softly", 1.2), ("lightly", 1.2),
               ("slowly", 1.2), ("nudge", 1.2), ("barely", 1.2),
               ("firmly", 1.7),
               ("very hard", 3.0), ("as hard", 3.0), ("harder", 3.0),
               ("hard", 3.0), ("shove", 3.0), ("violently", 3.0))
_PUSH_AT_RE = re.compile(r"\bat\s+(\d+(?:\.\d+)?)\s*m\s*/\s*s\b", re.I)


def _parse_push_speed(text: str):
    m = _PUSH_AT_RE.search(text)
    if m:
        return float(m.group(1))
    low = text.lower()
    for word, v in _PUSH_WORDS:
        if word in low:
            return v
    return None                                       # the recorder's Choice default


def _render_push(items: list, pushed_index: int, *, use_llm: bool,
                 push_speed=None) -> dict:
    """The follow-up simulation: re-research the SAME stack (same citations),
    push one object, record what friction decides. Refusal contract matches
    the stack route — an ungroundable item refuses the whole follow-up.

    Also records the push-speed VARIANT SWEEP: the same matter re-simulated at
    each _PUSH_SPEEDS value (research once via prebuilt, push many times), so
    the viewer's variant slider walks the one legitimate during-sim knob.
    Each sibling is a full recorded bundle — frozen runs, never live physics
    in the renderer."""
    from sigma_ground.deckard import catalog
    from sigma_ground.deckard.scene import build_stack
    from sigma_ground.radiance.push_off import record_push_off
    try:
        prebuilt = build_stack(items, allow_llm=use_llm)
    except ValueError as e:
        return {"intent": "simulate", "source": "deckard", "can_render": False,
                "saved": None, "value": None,
                "text": f"⚠ couldn't run the push: {e}"}
    title = f"{items[pushed_index]} pushed off the {items[pushed_index - 1]}"
    base_slug = catalog.slugify(title)
    chosen = push_speed if push_speed is not None else 2.2
    speeds = sorted(set(_PUSH_SPEEDS) | {chosen})
    variants = [{"slug": (base_slug if v == chosen
                          else f"{base_slug}_v{int(round(v * 10)):02d}"),
                 "label": f"push {v:g} m/s"} for v in speeds]
    chosen_bundle = None
    for v in speeds:
        bundle = record_push_off(items, pushed_index, push_speed_m_s=v,
                                 prebuilt=prebuilt)
        bundle["scene"]["variants"] = variants
        bundle["scene"]["variant_current"] = (
            base_slug if v == chosen else f"{base_slug}_v{int(round(v * 10)):02d}")
        if v == chosen:
            chosen_bundle = bundle                    # saved by _announce_render
        else:
            _save_bundle(f"{base_slug}_v{int(round(v * 10)):02d}", bundle)
    env = _announce_render(title, chosen_bundle)
    outcome = chosen_bundle["trajectory"]["validation"]["final_support"]
    ends = "; ".join(f"{k} → {v}" for k, v in outcome.items())
    env["text"] += (f" Outcome at {chosen:g} m/s: {ends}. The viewer's variant "
                    f"slider re-runs the SAME matter at "
                    f"{', '.join(f'{s:g}' for s in speeds)} m/s — the push is "
                    f"the one artificial force here, so it is the one knob "
                    f"(matter stays fact).")
    return env


def _render_from_handle(handle: dict) -> dict:
    """Turn a Materia render-handle into a watchable, saved Radiance fall.

    Two kinds: a generic SPHERE renders natively from its material + radius
    (radiance.record_fall — no Deckard); a NAMED object has Deckard compile its
    real shape, then record_object_fall drops it. Either way we never fake a shape.
    """
    kind = handle.get("kind")
    if kind == "scene_stack":                # re-render the remembered stack
        return _render_scene_stack(handle["items"], use_llm=True)
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
    if kind == "hand_tool_actuation":         # a DISCOVERED pivot, opened/closed
        from sigma_ground.radiance.trajectory import record_hand_tool_actuation
        tool = handle.get("tool", "pliers")
        bundle = record_hand_tool_actuation(tool, n_cycles=handle.get("n_cycles", 3.0))
        return _announce_render(f"{tool} opening and closing", bundle)
    if kind == "bell_strike":                # a hanging bell struck by a stone
        from sigma_ground.radiance.trajectory import record_bell_strike
        bundle = record_bell_strike(
            bell_material=handle.get("bell_material", "iron"),
            bell_diameter_m=handle.get("bell_diameter_m", 0.15),
            stone_mass_kg=handle.get("stone_mass_kg", 0.2),
            observer_distance_m=handle.get("observer_distance_m", 50.0))
        return _announce_render("bell struck by a stone", bundle)

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
    ac = bundle.get("acoustics")
    if ac and bundle.get("audio_path"):
        struck = v.get("struck")
        if struck:
            text += (f"\nStruck at t={v.get('struck_at_s', 0.0):.3f} s "
                     f"(v_impact={v.get('v_impact_m_s', 0.0):.2f} m/s, "
                     f"{v.get('dissipated_energy_j', 0.0):.3g} J dissipated "
                     f"into deformation/heat/sound — Johnson-Thornton "
                     f"energy partition, not raw KE) — an initial "
                     f"~{ac.get('impact_click_frequency_hz', 0.0):.0f} Hz "
                     f"Hertzian click, then rings at "
                     f"{ac['ring_frequency_hz']:.0f} Hz, decaying over "
                     f"~{ac['decay_tau_s']:.1f} s; reaches an observer in "
                     f"{ac['arrival_time_s']:.2f} s. Audio saved to "
                     f"{bundle['audio_path']}")
        else:
            text += "\n(No contact registered — the stone missed the bell.)"
    return _envelope("render", text=text, can_render=False,
                     saved={"slug": slug, "path": path, "url": url, "title": title},
                     source="radiance")


def _save_bundle(slug: str, bundle: dict) -> str:
    os.makedirs(_DATA_DIR, exist_ok=True)
    path = os.path.join(_DATA_DIR, slug + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=1)
    return path


def _ask(text: str, *, session: Session | None = None) -> dict:
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
    # A separate, cheap, side-effect-free translate() call (deterministic
    # path only) purely to recover the structured missing_slot metadata that
    # materia.answer()'s plain-string return discards -- lets a targeted
    # clarify set session.pending_clarification for next-turn resume even
    # though the display text comes from answer()'s own formatting.
    if session is not None:
        spec = materia.translate(text, use_qwen=False)
        if spec.missing_slot:
            session.pending_clarification = {"question": text}
    a2 = materia.answer(text, use_qwen=False)     # clarifies if it isn't a sim
    return _envelope("ask", text=a2, value=None, source="materia")


__all__ = ["dispatch", "Session"]
