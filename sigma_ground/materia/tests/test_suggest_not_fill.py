"""Mentat never fills in an input by itself; it suggests, and computes on a yes (the Captain, 2026-10-23).

The bug that started it: "a baseball at 45 degrees at 100 miles per hour" ran at 30 m/s, every time, because
projectile_motion declared no slots and its Python default won over the question.
"""
import math

import pytest

from sigma_ground.materia import inputs as mi
from sigma_ground.materia import translator as tr
from sigma_ground.materia.manifest import VERB_MANIFEST
from sigma_ground.mcp.tools.simulation import run_simulation, simulate

BASEBALL = ("If I throw a baseball at 45 degrees at 100 miles per hour, how far will it travel before it "
            "touches the ground on a flat surface?")
G = 9.80665          # the projectile module's g


def test_the_baseball_question_uses_the_speed_it_gives():
    r = simulate(BASEBALL, use_llm=False).to_dict()
    step, = r["inputs"]["understood"]
    assert step["verb"] == "projectile_motion"
    assert step["params"]["speed_m_s"] == pytest.approx(44.704, abs=1e-9)        # 100 mph, exactly
    assert step["params"]["angle_deg"] == 45.0
    assert r["value"] == pytest.approx(44.704 ** 2 / G, rel=1e-6) and r["units"] == "m"   # ≈ 203.8 m, no air
    assert 203 < r["value"] < 205 and r["clarification"] is None


@pytest.mark.parametrize("words, m_s", [
    ("100 miles per hour", 44.704), ("100 mph", 44.704), ("60 km/h", 60 / 3.6), ("60 kph", 60 / 3.6),
    ("30 ft/s", 9.144), ("20 m/s", 20.0), ("20 meters per second", 20.0), ("10 knots", 5.144444),
])
def test_a_speed_in_any_unit_is_converted_by_mentats_unit_tool(words, m_s):
    assert tr._extract_speed(f"how far does a ball thrown at {words} at 30 degrees go") == pytest.approx(m_s, rel=1e-6)


def test_60_kmh_reaches_the_answer():
    r = simulate("how far does a ball thrown at 60 km/h at 30 degrees travel", use_llm=False).to_dict()
    v = 60 / 3.6
    assert r["value"] == pytest.approx(v ** 2 * math.sin(math.radians(60)) / G, rel=1e-6)


def test_a_missing_input_gets_a_suggestion_and_no_number():
    r = simulate("range of a projectile launched at 45 degrees", use_llm=False).to_dict()
    assert r["value"] is None and "chain" not in r["inputs"]          # nothing was computed
    c = r["clarification"]
    assert c["needs"] == ["speed_m_s"]
    assert c["suggested"]["speed_m_s"]["value"] == 30.0 and c["suggested"]["speed_m_s"]["units"] == "m/s"
    assert c["suggested"]["speed_m_s"]["why"]
    assert c["sentence"].startswith("Your question doesn't give everything") and c["sentence"].endswith("Use that?")
    assert "30 m/s" in c["sentence"]
    assert c["follow_up"] == {"question": "range of a projectile launched at 45 degrees", "given": {"speed_m_s": 30.0}}
    assert r["error"]["code"] == "needs_input"


def test_yes_computes_with_exactly_the_suggested_values():
    first = simulate("range of a projectile launched at 45 degrees", use_llm=False).to_dict()
    f = first["clarification"]["follow_up"]
    second = simulate(f["question"], use_llm=False, given=f["given"]).to_dict()
    assert second["value"] == pytest.approx(30.0 ** 2 / G, rel=1e-6)
    assert second["inputs"]["understood"][0]["params"]["speed_m_s"] == 30.0


def test_given_may_be_a_quantity_and_a_bad_one_stays_missing():
    q = "range of a projectile launched at 45 degrees"
    r = simulate(q, use_llm=False, given={"speed_m_s": "100 mph"}).to_dict()
    assert r["value"] == pytest.approx(44.704 ** 2 / G, rel=1e-6)
    bad = simulate(q, use_llm=False, given={"speed_m_s": "100 kilograms"}).to_dict()
    assert bad["value"] is None and bad["clarification"]["not_used"]["speed_m_s"]


def test_no_material_or_size_is_assumed_for_a_drop():
    r = simulate("how fast does a ball dropped from 100 m hit the ground", use_llm=False).to_dict()
    assert r["value"] is None
    assert {"material_key", "radius_m", "T"} <= set(r["clarification"]["needs"])
    step, = r["inputs"]["understood"]
    assert "material_key" not in step["params"] and "radius_m" not in step["params"]


def test_a_unique_temperature_in_the_question_is_read_and_converted():
    p = tr._params_for("terminal_velocity_drop", "how fast does a 5 cm copper ball hit the ground from 10 km in 15 °C air")
    assert p["T"] == pytest.approx(288.15)


def test_the_models_numbers_are_kept_only_when_they_come_from_the_question():
    assert tr._qwen_number(BASEBALL, "projectile_motion", "speed_m_s", 44.7) == pytest.approx(44.704)
    assert tr._qwen_number(BASEBALL, "projectile_motion", "speed_m_s", 100) is None       # 100 mph is not 100 m/s
    assert tr._qwen_number(BASEBALL, "projectile_motion", "speed_m_s", 30) is None        # not in the question at all
    assert tr._qwen_number("relativistic energy at 0.9c", "relativistic_motion", "velocity_fraction_c", 0.9) == 0.9


def test_answer_and_run_simulation_suggest_too():
    assert tr.answer("range of a projectile launched at 45 degrees", use_qwen=False).startswith("❓ Your question")
    r = run_simulation("projectile_motion", {"angle_deg": 30}).to_dict()
    assert r["value"] is None and r["clarification"]["follow_up"] == {
        "verb": "projectile_motion", "params": {"angle_deg": 30, "speed_m_s": 30.0}}
    assert run_simulation("projectile_motion", {"angle_deg": 30, "speed_m_s": 10}).to_dict()["value"] is not None


def test_the_cockpit_takes_a_yes_to_a_suggestion():
    from sigma_ground.mcp.front_door import Session, dispatch
    s = Session()
    first = dispatch("range of a projectile launched at 45 degrees", use_llm=False, session=s, mode="simulate")
    assert first["intent"] == "clarify" and first["clarification"]["needs"] == ["speed_m_s"]
    second = dispatch("yes", use_llm=False, session=s)
    assert second["intent"] == "simulate" and "91.8" in second["text"]


def test_a_value_is_never_reported_with_an_input_nobody_gave():
    """Every example question in the manifest: either a suggestion, or every input the verb uses is in what
    Mentat understood from the question (or threaded from an earlier step)."""
    for verb, m in VERB_MANIFEST.items():
        for q in m.get("examples", [])[:2]:
            r = simulate(q, use_llm=False).to_dict()
            if r["value"] is None:
                continue
            for st in r["inputs"].get("understood", []):
                have = set(st["params"]) | set(st["bindings"])
                for name, meta in mi.verb_inputs(st["verb"]).items():
                    if meta["default"] is None:
                        continue
                    alts = [g for g in mi.ALTERNATIVES.get(st["verb"], []) if name in g]
                    assert name in have or (alts and have & set(alts[0])), (q, st["verb"], name)
