"""The front door: one text input → ask / simulate / render, on local models.

The two-turn feather flow runs fully OFFLINE and DETERMINISTIC (use_llm=False):
the feather routes to the Deckard seam, drop_object grounds the *catalogued*
feather, the dispatcher offers a render, and a "yes" renders + saves a watchable
fall whose geometry round-trips to an SDF.
"""
import json
import os

from sigma_ground.mcp.front_door import dispatch, Session
from sigma_ground.radiance.scene_export import scene_spec_to_sdf
from sigma_ground.dynamics.vec import Vec3

FEATHER = ("drop a 4 inch long feather from 8 ft and see what it looks like as "
           "it falls to the floor.")


def test_simulate_then_render_two_turns():
    s = Session()
    # Turn 1 — Materia simulates and OFFERS a render (it grounded a real object).
    e1 = dispatch(FEATHER, use_llm=False, session=s)
    assert e1["intent"] == "simulate"
    assert e1["can_render"] is True
    assert s.render_handle and s.render_handle["object_name"] == "feather"
    assert "render" in e1["text"].lower()            # the offer line
    assert "feather" in e1["text"].lower()           # shown work cites the object

    # Turn 2 — "yes" → Radiance renders + SAVES it for replay.
    e2 = dispatch("yes", use_llm=False, session=s)
    assert e2["intent"] == "render"
    assert e2["saved"]["slug"] == "falling_feather"
    assert os.path.exists(e2["saved"]["path"])
    assert s.render_handle is None                   # consumed

    bundle = json.load(open(e2["saved"]["path"], encoding="utf-8"))
    assert bundle["kind"] == "trajectory"
    # the saved scene's geometry round-trips to an SDF (faithful, not faked)
    sdf, _ = scene_spec_to_sdf(bundle["scene"])
    assert isinstance(sdf(Vec3(0.0, 0.0, 0.0)), float)
    # the moving body is the real feather (cone+ellipsoid), never a sphere
    moving = {l["shape"]["type"] for l in bundle["scene"]["csg_leaves"]
              if l.get("body") == 0}
    assert moving and "Sphere" not in moving


def test_bare_yes_without_pending_is_not_a_render():
    # a "yes" with no pending simulation must NOT fabricate a render
    e = dispatch("yes", use_llm=False, session=Session())
    assert e["intent"] != "render"
    assert e["saved"] is None


def test_question_routes_to_ask():
    e = dispatch("what is the speed of light", use_llm=False, session=Session())
    assert e["intent"] == "ask"
    assert e["text"]                                 # a grounded value or honest clarify


def test_bare_ball_drop_simulates_and_renders_as_sphere():
    """A plain 'drop a ball from a height' (no 'how fast' cue, ambient words
    present) now SIMULATES and renders natively as a sphere via record_fall — the
    regression that routed it to atmospheric_profile / decline is fixed."""
    s = Session()
    e1 = dispatch("drop a 5 cm steel ball from 10 km onto a concrete floor "
                  "in standard atmosphere", use_llm=False, session=s)
    assert e1["intent"] == "simulate" and e1["can_render"] is True
    assert s.render_handle and s.render_handle["kind"] == "sphere"
    e2 = dispatch("yes", use_llm=False, session=s)
    assert e2["intent"] == "render"
    assert e2["saved"]["slug"].startswith("falling_")
    bundle = json.load(open(e2["saved"]["path"], encoding="utf-8"))
    assert bundle["kind"] == "trajectory"
    moving = {l["shape"]["type"] for l in bundle["scene"]["csg_leaves"]
              if l.get("body") == 0}
    assert moving == {"Sphere"}
