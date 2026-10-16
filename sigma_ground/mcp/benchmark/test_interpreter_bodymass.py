"""The semantic interpreter's body_mass dispatcher answers a question about a mass and nothing else.

Found 2026-10-16 by test_targets (astro_002): "What's the surface temperature of the Sun?" is near "the mass of the Sun" in embedding space (similarity 0.64,
exactly the threshold) and was answered 1.989e30 kg. semantic_answer runs before the LLM, so a wrong confident answer is worse than none: the dispatcher now
returns None unless the question has a mass word, and the question goes on to the LLM with its tools. These tests need no embedding server.
"""
import pytest

from sigma_ground.mcp.benchmark import interpreter_demo as I
from sigma_ground.field.constants import M_SUN_KG


@pytest.mark.parametrize("question", [
    "What's the surface temperature of the Sun?",
    "What is the surface temperature of the Sun?",
    "How hot is the surface of the Sun?",
    "What is the radius of Jupiter?",
    "How far is Mars from the Sun?",
    "What is the luminosity of the Sun?",
    "How strong is gravity on the Moon?",
    "How long is a year on Venus?",
])
def test_a_question_about_another_property_is_not_answered_with_a_mass(question):
    assert I._disp_bodymass(question) is None


@pytest.mark.parametrize("question,body_mass", [
    ("What is the mass of the Sun?", M_SUN_KG),
    ("How massive is Jupiter?", 1.898e27),
    ("How heavy is Mars?", 6.39e23),
    ("How much does the Moon weigh?", 7.342e22),
    ("What is the weight of the Earth?", 5.972e24),
    ("Earth mass in kg", 5.972e24),
])
def test_a_question_about_a_mass_is_still_answered(question, body_mass):
    r = I._disp_bodymass(question)
    assert r is not None and r.value == pytest.approx(body_mass) and r.units == "kg"


def test_a_mass_question_about_an_unknown_body_is_still_not_answered():
    assert I._disp_bodymass("What is the mass of Pluto?") is None


@pytest.mark.parametrize("question", [
    "What's the surface temperature of the Sun?",
    "What is the surface temperature of the Sun?",
    "How hot is the surface of the Sun?",
    "What is the effective temperature of the Sun?",
    "how hot is the sun",
])
def test_the_sun_surface_temperature_is_answered_from_the_tool(question):
    r = I._disp_sun_surface_temperature(question)
    assert r is not None and r.value == pytest.approx(5772.0) and r.units == "K" and r.inputs == {"body": "sun"}


@pytest.mark.parametrize("question", [
    "The Sun's surface temperature is about 5778 K. What wavelength is its peak emission?",     # gives the temperature, asks for another quantity (thermo_012)
    "Sirius is the brightest star in the night sky. What's its luminosity compared to the Sun?",  # astro_003
    "How hot is the core of the Sun?",
    "What is the temperature of the solar corona?",
    "How much hotter is the Sun's surface than the Earth's?",
    "What is the surface temperature of Mars?",                                                  # no surface temperature for the planets in the tool: the LLM path
    "What is the mass of the Sun?",
])
def test_the_sun_surface_temperature_is_not_given_for_other_questions(question):
    assert I._disp_sun_surface_temperature(question) is None


def test_the_sun_surface_temperature_intent_is_in_the_index():
    assert "sun_surface_temperature" in [t for t, *_ in I.INTENTS]


def test_the_semantic_answer_gives_up_so_the_llm_can_answer(monkeypatch):
    """Route the temperature question to body_mass at the threshold, as the embedding did: semantic_answer must return None, not a mass."""
    route = {"kind": "tool", "tool": "body_mass", "score": 0.64, "disp": I._disp_bodymass, "field": None,
             "normalized": "what is the surface temperature of the sun", "runner_up": ("orbital_period", 0.5)}
    monkeypatch.setattr(I, "route", lambda question, threshold=0.55: route)
    assert I.semantic_answer("What's the surface temperature of the Sun?") is None
    route["normalized"] = "what is the mass of the sun"
    a = I.semantic_answer("What is the mass of the Sun?")
    assert a is not None and a["semantic_router_hit"] == "body_mass" and a["extracted_value"] == pytest.approx(M_SUN_KG)
