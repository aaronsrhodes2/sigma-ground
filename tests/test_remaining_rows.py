"""The last suite rows made static (2026-10-30): y(t) from the question, GEO ground-relative speed, GPS clock altitude, Everest in bananas."""
import pytest

from sigma_ground.mcp.benchmark.astro_classifier import classify_for_astro
from sigma_ground.mcp.benchmark.param_aliases import enrich_from_question
from sigma_ground.mcp.benchmark.puzzle_classifier import classify_for_puzzle
from sigma_ground.mcp.tools.clocks import altitude_for_clock_drift, orbital_clock_drift
from sigma_ground.mcp.tools.landmarks import landmark_height
from sigma_ground.mcp.tools.mathx import solve_ode


def test_the_questions_y_of_t_fills_the_routers_plan():
    args, added = enrich_from_question("solve_ode", {"equation": "y' + 4*y"}, "Give me every function y(t) with y' + 4y = 0")
    assert args["func"] == "y(t)" and added == ["func=y(t) from the question"]
    assert solve_ode(**args).value == "Eq(y(t), C1*exp(-4*t))"
    # nothing to take: no f(v) in the question, or a function the equation does not use, or the router already named func
    assert enrich_from_question("solve_ode", {"equation": "y' + 4*y"}, "every function with y' + 4y = 0")[1] == []
    assert enrich_from_question("solve_ode", {"equation": "y' + 4*y"}, "with g(x) given, solve y' + 4y = 0")[1] == []
    assert enrich_from_question("solve_ode", {"equation": "y' + 4*y", "func": "y"}, "y(t) with y' + 4y = 0")[0]["func"] == "y"
    assert enrich_from_question("integrate_expr", {"expression": "x"}, "y(t)")[1] == []


def test_orbital_clock_drift_reproduces_gps():
    r = orbital_clock_drift(20200.0)
    assert r.value == pytest.approx(38.5, abs=0.3) and r.units == "us/day" and "+45.7" in r.notes and "-7.2" in r.notes
    assert orbital_clock_drift(3190.0).value == pytest.approx(0.0, abs=0.2)        # the cancellation altitude, r = 1.5 R
    assert orbital_clock_drift(400.0).value < 0                                     # the ISS clock runs slow
    h = altitude_for_clock_drift(38.0)
    assert 19.0e6 < h < 21.0e6 and orbital_clock_drift(h / 1e3).value == pytest.approx(38.0, abs=1e-6)
    assert altitude_for_clock_drift(1e6) is None


def test_geo_and_gps_cues():
    m = classify_for_astro("If I am in orbit 35,786 kilometers above Earth (geostationary altitude), how fast am I moving relative to the ground I'm orbiting over?")
    assert m.result_override == 0.0 and m.result_units == "m/s" and "geostationary" in m.rationale
    assert classify_for_astro("How fast does a geostationary satellite move?") is None          # the inertial speed is a real question
    m = classify_for_astro("At what altitude above Earth do GPS satellite clocks run faster by about 38 microseconds per day?")
    assert m.tool == "orbital_clock_drift" and m.result_units == "m" and 19.0e6 < m.result_override < 21.0e6
    assert classify_for_astro("How much faster do GPS clocks run per day?") is None              # no stated drift, nothing to invert


def test_landmarks_and_the_bananas_cue():
    assert landmark_height("Mount Everest").value == 8848.86 and landmark_height("everest").value == 8848.86
    assert landmark_height("the Eiffel Tower").value == 330.0 and landmark_height("Mariana Trench").units == "m"
    assert landmark_height("my house").value is None
    m = classify_for_puzzle("How many bananas tall is Mount Everest? Assume one banana is 18 centimeters long.")
    assert m.tool == "landmark_height" and m.result_override == pytest.approx(8848.86 / 0.18) and m.result_units == "bananas"
    assert classify_for_puzzle("How many bananas tall is my house? Assume one banana is 18 centimeters long.") is None
    assert classify_for_puzzle("How many bananas tall is Mount Everest?") is None                # no banana length stated
