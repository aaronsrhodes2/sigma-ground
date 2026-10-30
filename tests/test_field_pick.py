"""run_sigma_ground.pick_field: the one field of a record-valued tool result the question asked for, and how sure the pick is.

Background (misc/STATIC_INTENTS_PLAN.md, group A): 7 of the 18 physics misses of 2026-10-27 were a right record tool
(solar_system_body, element_atomic_data) whose whole record was presented when one field was asked for. The harness already
picked the field for scoring; this names the field and the confidence so the live /ask can present it, and ONLY when a phrase
in the question named it.
"""
from sigma_ground.mcp.benchmark.run_sigma_ground import _pick_dict_scalar, pick_field

MARS = {"mass_kg": 6.4171e23, "radius_m": 3389500.0, "semimajor_axis_au": 1.523679, "orbital_period_d": 686.971,
        "surface_g_ms2": 3.711}
GOLD = {"symbol": "Au", "atomic_number": 79, "atomic_mass_amu": 196.966570, "is_metal": True}


def test_a_phrase_in_the_question_names_the_field():
    assert pick_field(MARS, "What's the surface gravity of Mars?") == (3.711, "m/s^2", "surface_g_ms2", "cue")
    assert pick_field(MARS, "How massive is Jupiter?") == (6.4171e23, "kg", "mass_kg", "cue")
    assert pick_field(MARS, "What's the orbital period of Mars?") == (686.971, "day", "orbital_period_d", "cue")
    assert pick_field(GOLD, "What's the atomic number of gold?") == (79.0, "", "atomic_number", "cue")
    assert pick_field(GOLD, "Uranium atomic mass?")[2:] == ("atomic_mass_amu", "cue")


def test_without_a_phrase_the_pick_is_a_guess_and_says_so():
    value, units, field, how = pick_field(MARS, "Tell me about Mars.")
    assert how == "default" and field == "mass_kg" and value == 6.4171e23
    assert pick_field({"foo": 2.0, "bar": 3.0}, "anything")[2:] == ("foo", "first")


def test_a_boolean_is_never_the_answer():
    # is_metal is True; bool is an int in Python and must not be read as 1.0
    assert pick_field({"is_metal": True, "atomic_number": 79}, "how many protons does gold have")[2] == "atomic_number"
    assert pick_field({"is_metal": True}, "gold") == (None, "", None, "none")


def test_old_wrapper_keeps_its_shape():
    assert _pick_dict_scalar(MARS, "surface gravity of Mars") == (3.711, "m/s^2")
    assert _pick_dict_scalar({}, "x") == (None, "")
