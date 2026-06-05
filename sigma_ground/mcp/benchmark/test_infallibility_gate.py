"""Infallibility gate for the Q&A switchboard — an answer is grounded (tool +
a CONCRETE value) or it is refused. Closes the 'present-but-garbage value' gap
(a tool ran but returned a half-rendered LaTeX fragment / NaN), and provides the
route-consistency cross-check for the wrong-tool class.
"""
import math

from sigma_ground.mcp.benchmark import groundedness as G


def _rec(value, tool=True, turns=1):
    return {"turns": turns,
            "tool_calls": [{"name": "t"}] if tool else [],
            "extracted_value": value}


def test_real_tool_value_is_grounded():
    assert G.assess(_rec(938.27)).grounded


def test_garbage_value_is_refused():
    # a tool ran, but the "value" is a half-rendered LaTeX fragment → refuse
    assert not G.assess(_rec(r"\[ 1 + z = \frac")).grounded
    assert not G.assess(_rec(float("nan"))).grounded
    assert not G.assess(_rec(float("inf"))).grounded


def test_no_value_still_refused():
    assert not G.assess(_rec(None)).grounded            # tool ran, no value
    assert not G.assess(_rec(None, tool=False)).grounded  # no tool at all


def test_deterministic_path_grounded():
    assert G.assess(_rec(5.0, tool=False, turns=0)).grounded


def test_route_consistency():
    a = [{"function": {"name": "orbital_velocity"}}]
    b = [{"function": {"name": "orbital_velocity"}}]
    c = [{"function": {"name": "black_hole"}}]
    assert G.route_consistent(a, b)          # same tool → confident
    assert not G.route_consistent(a, c)      # diverged → refuse
    assert not G.route_consistent(a, [])     # one route found nothing
