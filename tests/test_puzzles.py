"""Puzzle tools and the puzzle cascade cue (the adversarial suite's puzzle and unit rows, 2026-10-30)."""
import pytest

from sigma_ground.mcp.benchmark.cosmology_classifier import classify_for_cosmology
from sigma_ground.mcp.benchmark.puzzle_classifier import classify_for_puzzle
from sigma_ground.mcp.tools.puzzles import birthday_collision, closing_time, monty_hall_switch


def test_monty_hall_birthday_and_closing_time_tools():
    assert monty_hall_switch().value == pytest.approx(2 / 3) and monty_hall_switch(4).value == pytest.approx(0.75)
    assert monty_hall_switch(2).value is None
    assert birthday_collision(23).value == pytest.approx(0.5073, abs=1e-4)
    assert birthday_collision(1).value == 0.0 and birthday_collision(366).value == 1.0
    r = closing_time(100e3, 60 / 3.6, 40 / 3.6)
    assert r.value == pytest.approx(3600.0) and r.units == "s"
    assert closing_time(100.0, 0.0, 0.0).value is None


def test_the_cue_answers_the_suites_five_rows():
    m = classify_for_puzzle("In the Monty Hall problem, you pick door 1. The host opens door 3 to reveal a goat. Should you switch to door 2, "
                            "and what's your probability of winning the car if you do?")
    assert m.tool == "monty_hall_switch" and m.result_override == pytest.approx(2 / 3)
    m = classify_for_puzzle("In a room of 23 people, what is the probability that at least two of them share a birthday?")
    assert m.tool == "birthday_collision" and m.result_override == pytest.approx(0.5073, abs=1e-4)
    m = classify_for_puzzle("Two trains, 100 km apart, approach each other on the same track. One moves at 60 km/h, the other at 40 km/h. "
                            "How long until they meet?")
    assert m.tool == "closing_time" and m.result_override == pytest.approx(1.0) and m.result_units == "h"
    m = classify_for_puzzle("How many heartbeats does an average human have in their lifetime? Assume 80 years and 70 beats per minute.")
    assert m.result_override == pytest.approx(70 * 60 * 24 * 365.25 * 80) and m.result_units == "heartbeats"
    m = classify_for_puzzle("If a nanosecond cost one US dollar, how many dollars would the age of the universe (13.787 billion years) cost?")
    assert m.result_override == pytest.approx(13.787e9 * 365.25 * 86400 * 1e9, rel=1e-6) and m.result_units == "USD"


def test_the_cue_stays_out_of_ordinary_questions():
    for q in ("What is the age of the universe?", "How fast is a train going at 60 km/h in m/s?", "what is the kinetic energy of love",
              "How many protons does gold have?", "Two trains leave the station at 60 km/h; how far apart are they after an hour?",
              "What's the probability of rolling two sixes?"):
        assert classify_for_puzzle(q) is None, q


def test_the_age_of_the_universe_cue_leaves_a_count_or_a_price_alone():
    # the cosmology cue answered the nanosecond-dollar question with the age in SECONDS: a silently wrong answer
    assert classify_for_cosmology("If a nanosecond cost one US dollar, how many dollars would the age of the universe cost?") is None
    assert classify_for_cosmology("What is the age of the universe?").tool == "age_of_universe"
    assert classify_for_cosmology("How old is the universe?").tool == "age_of_universe"
