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
    """The thread of one conversation. The only state the front door keeps is the
    last renderable simulation, so a bare "yes" can render it."""
    render_handle: dict | None = None
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

_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "radiance", "web", "data"))
_VIEWER = "http://127.0.0.1:8765"
_OFFER = ("Do you want Radiance to render this as a watchable 3D fall? "
          "Reply 'yes'.")


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


def dispatch(text: str, *, use_llm: bool = True,
             session: Session | None = None) -> dict:
    """Classify one sentence and route it. Returns an envelope dict (keys: intent,
    text, value, can_render, render_handle, saved, source). Pass the same
    ``session`` across turns so a "yes" renders the last simulation."""
    session = session or Session()
    text = (text or "").strip()
    if not text:
        return _envelope("clarify",
                         text="Say something to ask, simulate, or render.")

    # 1. A "yes" with a pending renderable simulation → render it now.
    if session.render_handle and _is_affirmative(text):
        handle = session.render_handle
        session.render_handle = None
        env = _render_from_handle(handle)
        session.last_intent = env["intent"]
        return env

    # 2. Simulation — an OBJECT IN MOTION (the North Star's dividing line). Let
    #    Materia's own router compile it (deterministic first; the qwen residual
    #    only if the sentence is sim-cued, to keep ASK fast). A runnable spec is a
    #    SIMULATION only when it concerns a physical object/motion — a domain
    #    *report* verb (e.g. "the speed of light") that happens to be routable is
    #    a fact, and belongs to ASK.
    from sigma_ground import materia
    from sigma_ground.materia import translator as _t
    spec = materia.translate(text, use_qwen=False)
    if not spec.is_runnable() and use_llm and _has(text, _SIM_CUES):
        spec = materia.translate(text, use_qwen=True)
    verbs = {st.verb for st in spec.steps}
    is_sim = spec.is_runnable() and (
        "drop_object" in verbs or _t._has_object_context(text.lower()))
    if is_sim:
        env = _run_simulation(text, spec, use_llm=use_llm, session=session)
        # An EXPLICIT "render"/"draw" of a renderable sim skips the offer and
        # renders straight away (the user already said they want to see it).
        if env.get("render_handle") and _has(text, _RENDER_CUES):
            handle = env["render_handle"]
            session.render_handle = None
            env = _render_from_handle(handle)
        session.last_intent = env["intent"]
        return env

    # 3. Otherwise it's a question → the Q&A switchboard.
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
    if handle:
        session.render_handle = handle
        narration += "\n\n" + _OFFER
    return _envelope("simulate", text=narration, can_render=bool(handle),
                     render_handle=handle, source=spec.source)


def _render_from_handle(handle: dict) -> dict:
    """Turn a Materia render-handle into a watchable, saved Radiance fall."""
    from sigma_ground import deckard
    from sigma_ground.radiance import record_object_fall
    from sigma_ground.deckard import catalog

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
    slug = catalog.slugify("falling " + name)
    path = _save_bundle(slug, bundle)
    url = f"{_VIEWER}/?scene={slug}"
    title = f"falling {name}"
    tr = bundle["trajectory"]
    text = (f"Rendered the {name} fall — its real shape descending to the floor "
            f"({len(tr['frames'])} frames, ground in "
            f"{tr['natural_timescale_s']:.2f} s). Saved as “{title}”. "
            f"Serve with `python -m sigma_ground.radiance.web.serve`, then open "
            f"{url} and press ▶.")
    return _envelope("render", text=text, can_render=False,
                     saved={"slug": slug, "path": path, "url": url,
                            "title": title},
                     source="radiance.record_object_fall")


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
        return _envelope("ask", text=a, value=a, source="semantic-interpreter")
    from sigma_ground import materia
    a2 = materia.answer(text, use_qwen=False)     # clarifies if it isn't a sim
    return _envelope("ask", text=a2, value=None, source="materia")


__all__ = ["dispatch", "Session"]
