"""The front door's SCENE route — "build/show/render/simulate a scene of X"
reaching the Deckard shape seam (ask_deckard.request_shape_scene) instead of
falling through to Materia's physics clarify.

This is the fix for Patrick's challenge: "I want a simulation of a Doughnut,
with frosting and rainbow sprinkles. I want it sitting on a plate, which is
flat on top of a 1700s table." — a request front_door.dispatch() used to have
NO route for at all (it only understood physics-simulation verbs).

Three things are proven here:
  1. name extraction (_extract_scene_name) is correct on the trigger
     phrasings this task named, including the exact Captain prompt;
  2. the SCENE route reaches the shape pipeline end-to-end (a catalog object,
     deterministic/offline — fast — and the literal Captain prompt, which
     needs the LLM research path since "doughnut" isn't in Deckard's frozen
     catalog);
  3. existing physics-verb routing is UNCHANGED (no regression) — Materia's
     own router still gets first refusal on every sentence.
"""
import json
import os

from sigma_ground.mcp.front_door import (
    dispatch, dispatch_accepting_suggestions, Session, _extract_scene_name, _extract_scene_stack)
from sigma_ground.radiance.scene_export import scene_spec_to_sdf
from sigma_ground.dynamics.vec import Vec3

CAPTAIN_PROMPT = (
    "I want a simulation of a Doughnut, with frosting and rainbow sprinkles. "
    "I want it sitting on a plate, which is flat on top of a 1700s table.")


def test_extract_scene_stack_from_the_exact_captain_prompt():
    # bottom-to-top: table first, doughnut (the sentence's base object) last.
    assert _extract_scene_stack(CAPTAIN_PROMPT) == ["1700s table", "plate", "Doughnut"]


def test_extract_scene_stack_two_object_chain():
    assert _extract_scene_stack("show me a mug sitting on a table") == ["table", "mug"]


def test_extract_scene_stack_none_for_a_plain_single_object_request():
    assert _extract_scene_stack("show me a coffee cup") is None


def test_extract_scene_stack_none_for_a_physics_question():
    assert _extract_scene_stack("how fast does a steel ball fall from 10m") is None


# ── 1. name extraction (pure, fast, deterministic) ──────────────────────────
def test_extract_scene_name_from_the_exact_captain_prompt():
    assert _extract_scene_name(CAPTAIN_PROMPT) == "Doughnut"


def test_extract_scene_name_broad_trigger_phrasings():
    cases = {
        "show me a coffee cup": "coffee cup",
        "build a rocking chair": "rocking chair",
        "render a wine glass": "wine glass",
        "make me a golden anvil": "golden anvil",
        "what does a doughnut look like": "doughnut",
        "what does a doughnut look like?": "doughnut",
        "I want a scene of an old wooden ladder": "old wooden ladder",
        "create me a teapot, resting on a shelf": "teapot",
    }
    for text, expected in cases.items():
        assert _extract_scene_name(text) == expected, text


def test_extract_scene_name_returns_none_for_physics_and_fact_questions():
    # No scene cue in these — the caller must fall through to Materia/ask,
    # exactly as it did before this route existed.
    for text in ("how fast does a steel ball fall from 10m",
                 "what is the speed of light",
                 "does an iron sphere heat up falling from 30 km?"):
        assert _extract_scene_name(text) is None, text


def test_scene_stack_request_reaches_the_multi_object_pipeline_deterministic():
    """A simplified, offline (allow_llm=False) 2-object stack — the doughnut
    archetype and the "mug"/"table" PartNet exemplars are all deterministic,
    so this proves the stack route (not just the single-object route) end to
    end without needing the LLM research path."""
    env = dispatch("show me a Doughnut sitting on a table", use_llm=False,
                   session=Session())
    assert env["intent"] == "render"
    assert env["source"] == "radiance"
    assert env["saved"]["slug"] == "scene_doughnut_on_table"
    bundle = json.load(open(env["saved"]["path"], encoding="utf-8"))
    # dough+frosting+24 sprinkles (26) + the table's own real parts + room
    # dressing -- comfortably more leaves than either object alone.
    assert len(bundle["csg_leaves"]) > 26
    sdf, _ = scene_spec_to_sdf(bundle)
    assert isinstance(sdf(Vec3(0.0, 0.0, 0.0)), float)


# ── 2. the SCENE route reaches the shape pipeline (not the physics clarify) ─
def test_scene_request_reaches_shape_pipeline_deterministic():
    """A simplified, single-object version of the Captain's ask — a catalog
    object, so this runs fully offline/deterministic (allow_llm=False) and
    fast: no Materia physics clarify, a real saved, SDF-checkable scene."""
    env = dispatch("show me a coffee cup", use_llm=False, session=Session())
    assert env["intent"] == "render"
    assert env["source"] == "radiance"
    assert env["saved"] is not None
    assert env["saved"]["slug"] == "scene_coffee_cup"
    assert os.path.exists(env["saved"]["path"])

    bundle = json.load(open(env["saved"]["path"], encoding="utf-8"))
    # A bare, staged SceneSpec — the same on-disk shape as the curated
    # obj_*/theater_*.json gallery entries (no trajectory wrapper: this is a
    # static build, not a physics run).
    assert "csg_leaves" in bundle and len(bundle["csg_leaves"]) > 1
    assert bundle["identified"] is True
    sdf, material_at = scene_spec_to_sdf(bundle)
    assert isinstance(sdf(Vec3(0.0, 0.0, 0.0)), float)


def test_captain_doughnut_prompt_reaches_shape_pipeline_not_physics_clarify():
    """The exact, unmodified Captain/Patrick prompt, through the single text
    entry point. The sentence names a 3-object stack (doughnut on plate on
    1700s table) — _extract_scene_stack recognizes the "sitting on X, which
    is on Y" shape and routes to compose_scene, not the single-object path,
    so a full render covers all three; accept either that OR an honest
    Deckard refusal as a pass, but NEVER the old physics clarify message
    (the actual regression this task fixes)."""
    env = dispatch(CAPTAIN_PROMPT, use_llm=True, session=Session())
    assert env["intent"] in ("render", "simulate")
    assert env["source"] in ("radiance", "deckard")
    # The old failure mode: falling through to Materia's canned physics
    # clarify ("I can simulate a falling object's *impact speed*...").
    assert "impact speed" not in env["text"]
    assert "terminal velocity" not in env["text"].lower()
    if env["intent"] == "render":
        assert env["saved"]["slug"] == "scene_doughnut_on_plate_on_1700s_table"
        assert os.path.exists(env["saved"]["path"])
        bundle = json.load(open(env["saved"]["path"], encoding="utf-8"))
        # 3 real objects (doughnut w/ frosting+sprinkles, plate, table) plus
        # room dressing -- comfortably more than a single bare torus would be.
        assert len(bundle["csg_leaves"]) > 20
        sdf, _ = scene_spec_to_sdf(bundle)
        assert isinstance(sdf(Vec3(0.0, 0.0, 0.0)), float)
    else:
        # An honest "couldn't ground the shape" refusal from Deckard is still
        # a PASS for this test — it proves the route was reached.
        assert "doughnut" in env["text"].lower()


def test_scene_request_for_an_ungroundable_name_refuses_honestly():
    """A name Deckard can't research is an honest refusal, never a fabricated
    silhouette — same infallibility discipline as the physics side."""
    env = dispatch("build a nonexistent flibbertigibbet contraption",
                   use_llm=False, session=Session())
    assert env["intent"] == "simulate"          # not "render" — nothing was built
    assert env["can_render"] is False
    assert env["saved"] is None
    assert "couldn't ground" in env["text"].lower()


# ── 3. no regression — existing physics-verb routing is untouched ──────────
def test_existing_physics_routing_unaffected_by_the_new_scene_route():
    """A handful of prompts from the existing routing corpus / test_front_door
    suite, re-asserted here so the SCENE route's own test file proves the
    additive guarantee on its own (Materia's verb router still gets first
    refusal on every sentence — see dispatch() step 2b's comment)."""
    cases = [
        ("how fast does a steel ball fall from 10m", "simulate"),
        ("what is the speed of light", "ask"),
        ("put a hot iron cube on a cold copper slab", "simulate"),
        ("does an iron sphere heat up falling from 30 km?", "simulate"),
    ]
    for text, expected_intent in cases:
        env = dispatch_accepting_suggestions(text, use_llm=False, session=Session())
        assert env["intent"] == expected_intent, (text, env["intent"])


def test_pliers_and_bell_captain_prompts_still_simulate_not_scene():
    """These already-physics prompts start with "Simulate..." — the same verb
    family a scene request might use ("I want a simulation of..."). Confirm
    Materia's own router (hand-tool actuation / bell strike) still claims
    them before the new SCENE route ever runs."""
    s = Session()
    env = dispatch_accepting_suggestions("Simulate a pair of pliers that are being open and closed "
                                         "over time.", use_llm=False, session=s)
    assert env["intent"] == "simulate"
    assert s.render_handle and s.render_handle["kind"] == "hand_tool_actuation"

    env2 = dispatch_accepting_suggestions("Simulate a hanging bell being struck by a stone.",
                                          use_llm=False, session=Session())
    assert env2["intent"] == "simulate"
