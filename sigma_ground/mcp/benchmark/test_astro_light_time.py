"""astro_classifier: light travel time over a stated distance or from the Sun to the Earth (misc/STATIC_INTENTS_PLAN.md group C).

The two physics-suite misses mech_adv_004 (Sun -> Earth, 499 s) and astro_007 (1 parsec, 3.262 years) were right tools stopped one
step short (parsec converted to metres); a stated distance over c is one formula. Negative cases keep the rule narrow.
"""
from sigma_ground.mcp.benchmark.astro_classifier import classify_for_astro


def test_sun_to_earth_in_seconds():
    m = classify_for_astro("How long does it take for light from the Sun to reach Earth?")
    assert m is not None and m.tool == "light_travel_time" and m.result_units == "s"
    assert abs(m.result_override - 499.0) < 0.5


def test_a_stated_parsec_or_light_year_distance_answers_in_years():
    m = classify_for_astro("How long does it take light to travel 1 parsec?")
    assert m.tool == "light_travel_time" and m.result_units == "year" and abs(m.result_override - 3.2616) < 0.002
    m = classify_for_astro("how long would light take to cross 4.2 light-years")
    assert m.result_units == "year" and abs(m.result_override - 4.2) < 0.001


def test_shorter_distances_answer_in_seconds():
    m = classify_for_astro("How long does light take to travel 300000 km?")
    assert m.result_units == "s" and abs(m.result_override - 1.0007) < 0.001


def test_the_rule_stays_narrow():
    # a named star keeps the star rule (its own distance), a question without a distance or without light is not matched
    m = classify_for_astro("How long does light from Sirius take to reach us?")
    assert m is not None and m.tool == "light_travel_time_from_star"
    assert classify_for_astro("How long does it take a ball dropped from 10 m to fall?") is None
    assert classify_for_astro("how long does light take to reach the ground") is None
    assert classify_for_astro("what is the speed of light") is None
