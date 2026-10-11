"""Router plan v2 in the harness: `tool_args`, multi-step `steps`, and the request formats.

The same contract as test_router_mode.py, with plans of up to three calls where a
later argument "$n" stands for call n's headline value.
"""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from sigma_ground.mcp.benchmark import groundedness as G
from sigma_ground.mcp.benchmark import interpreter_demo
from sigma_ground.mcp.benchmark import router_plan as RP
from sigma_ground.mcp.benchmark import run_sigma_ground as R

TOOLS = {
    "circular_orbit_velocity": lambda a: {
        "value": (3.986e14 * a["central_mass_kg"] / 5.972e24 / a["radius_m"]) ** 0.5,
        "units": "m/s", "source": "fake"},
    "kinetic_energy": lambda a: {"value": 0.5 * a["mass_kg"] * a["velocity_m_s"] ** 2,
                                 "units": "J", "source": "fake"},
    "joules_to_TNT": lambda a: {"value": a["energy_joules"] / 4.184e9,
                                "units": "ton_TNT", "source": "fake"},
    "element_atomic_data": lambda a: {"value": {"name": "gold", "atomic_number": 79},
                                      "units": "", "source": "fake"},
    "density": lambda a: {"value": None, "units": "", "source": "not in table"},
}
OLLAMA_TOOLS = [{"type": "function", "function": {"name": n, "description": "",
                                                  "parameters": {}}} for n in TOOLS]
STEPS = ('{"steps": [{"tool": "circular_orbit_velocity", "tool_args": '
         '{"central_mass_kg": 5.972e24, "radius_m": 4.2164e7}}, '
         '{"tool": "kinetic_energy", "tool_args": {"mass_kg": 1000, "velocity_m_s": "$1"}}]}')


class FakeSession:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, args):
        self.calls.append((name, dict(args)))
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(TOOLS[name](args)))])


def ollama(monkeypatch, replies):
    bodies, replies = [], list(replies)

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            bodies.append(json)
            reply = replies.pop(0)
            return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                                   json=lambda: {"message": {"content": reply}})

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setattr(interpreter_demo, "semantic_answer", lambda q: None)
    return bodies


def ask(question="A 1000 kg satellite orbits at 42164 km. What is its kinetic energy?",
        fmt="toolfirst", plan="v2", prompt=None):
    session = FakeSession()
    rec = asyncio.run(R._run_one_question(
        session, "http://fake/api/chat", "router-model", OLLAMA_TOOLS, question,
        prompt or R._router_system_prompt(OLLAMA_TOOLS, False, plan),
        router=True, router_format=fmt, router_plan=plan))
    return G.apply_gate(rec), session


def test_two_steps_run_in_order_and_the_reference_is_filled(monkeypatch):
    ollama(monkeypatch, [STEPS])
    rec, session = ask()
    assert [c[0] for c in session.calls] == ["circular_orbit_velocity", "kinetic_energy"]
    v = TOOLS["circular_orbit_velocity"](session.calls[0][1])["value"]
    assert session.calls[1][1] == {"mass_kg": 1000, "velocity_m_s": v}     # "$1" -> the number
    assert rec["extracted_value"] == pytest.approx(0.5 * 1000 * v ** 2)
    assert rec["extracted_units"] == "J" and rec["finalized_from_tool"] == "kinetic_energy"
    assert rec["chain_length"] == 2 and rec["finalize_reason"] == "router plan, 2 steps"
    assert [c["name"] for c in rec["tool_calls"]] == ["circular_orbit_velocity", "kinetic_energy"]
    assert rec["grounded"] and "circular_orbit_velocity" in rec["answer_text"]
    assert rec["router_plan_version"] == "v2" and rec["turns"] == 1


def test_a_single_call_reads_in_either_spelling_and_runs_once(monkeypatch):
    ollama(monkeypatch, ['{"tool": "element_atomic_data", "tool_args": {"element_symbol": "Au"}}',
                         '{"tool": "element_atomic_data", "args": {"element_symbol": "Au"}}'])
    for _ in range(2):
        rec, session = ask("What's the atomic number of gold?")
        assert session.calls == [("element_atomic_data", {"element_symbol": "Au"})]
        assert rec["chain_length"] == 1 and rec["finalize_reason"] == "router plan"


def test_request_formats(monkeypatch):
    bodies = ollama(monkeypatch, [STEPS] * 4)
    names = set(TOOLS) | {"request_clarification"}
    ask(fmt="toolfirst", plan="v2")
    ask(fmt="toolfirst", plan="v1")
    ask(fmt="free", plan="v2")
    ask(fmt="schema", plan="v2")
    toolfirst_v2, toolfirst_v1, free, schema = bodies
    assert toolfirst_v2["format"] == RP.router_format(names, steps=True)
    assert toolfirst_v1["format"] == RP.router_format(names, steps=False)
    assert RP.keys_in_training_order(toolfirst_v2["format"])
    assert "format" not in free
    assert not RP.keys_in_training_order(schema["format"])           # the v1 schema: args before tool
    assert toolfirst_v2["messages"][0]["content"] == RP.ROUTER_SYSTEM_PROMPT_V2
    assert R._router_system_prompt(OLLAMA_TOOLS, False, "v1") != RP.ROUTER_SYSTEM_PROMPT_V2


@pytest.mark.parametrize("reply,error", [
    # a reference to a step that has not run, or to itself
    ('{"steps": [{"tool": "kinetic_energy", "tool_args": {"mass_kg": 1, "velocity_m_s": "$2"}}, '
     '{"tool": "joules_to_TNT", "tool_args": {"energy_joules": "$1"}}]}', "bad reference"),
    ('{"steps": [{"tool": "kinetic_energy", "tool_args": {"mass_kg": 1, "velocity_m_s": "$1"}}, '
     '{"tool": "joules_to_TNT", "tool_args": {"energy_joules": 1}}]}', "bad reference"),
    # an unknown tool anywhere in the plan stops everything BEFORE any call runs
    ('{"steps": [{"tool": "kinetic_energy", "tool_args": {"mass_kg": 1, "velocity_m_s": 2}}, '
     '{"tool": "make_coffee", "tool_args": {}}]}', "unknown tool 'make_coffee'"),
    # a clarification cannot sit inside a plan
    ('{"steps": [{"tool": "request_clarification", "tool_args": {"variable": "x"}}, '
     '{"tool": "kinetic_energy", "tool_args": {}}]}', "request_clarification inside steps"),
    ('{"steps": "nope"}', "invalid steps"),
    ('{"steps": []}', "invalid steps"),
])
def test_bad_plans_are_recorded_and_run_nothing(monkeypatch, reply, error):
    ollama(monkeypatch, [reply])
    rec, session = ask()
    assert session.calls == [] and rec["router_error"].startswith(error)
    assert rec["extracted_value"] is None and not rec["grounded"]


def test_a_step_with_no_value_stops_the_plan_and_keeps_what_ran(monkeypatch):
    ollama(monkeypatch, ['{"steps": [{"tool": "density", "tool_args": {"material": "mars"}}, '
                         '{"tool": "kinetic_energy", "tool_args": {"mass_kg": 1, "velocity_m_s": "$1"}}]}'])
    rec, session = ask()
    assert [c[0] for c in session.calls] == ["density"]                  # step 2 never ran
    assert rec["router_error"] == "density returned no value"
    assert [c["name"] for c in rec["tool_calls"]] == ["density"]
    assert not rec["grounded"]


def test_a_reference_to_a_non_number_is_an_error(monkeypatch):
    ollama(monkeypatch, ['{"steps": [{"tool": "element_atomic_data", "tool_args": {"element_symbol": "Au"}}, '
                         '{"tool": "kinetic_energy", "tool_args": {"mass_kg": "$1", "velocity_m_s": 2}}]}'])
    rec, session = ask("What's the atomic number of gold?")
    # element_atomic_data's headline value is the atomic number (79), a number: this runs.
    assert [c[0] for c in session.calls] == ["element_atomic_data", "kinetic_energy"]
    assert session.calls[1][1]["mass_kg"] == 79


def test_a_decline_and_a_clarification_still_run_nothing(monkeypatch):
    ollama(monkeypatch, ['{"decline": "not physics"}',
                         '{"tool": "request_clarification", "tool_args": {"variable": "mass_kg", '
                         '"question": "What is the mass?"}}'])
    rec, session = ask("What's the atomic number of gold?")     # a question the cascade leaves to the model
    assert session.calls == [] and rec["router_decline"] == "not physics"
    rec, session = ask("What's the atomic number of gold?")
    assert session.calls == [] and rec["router_clarify"] == "mass_kg"
    assert rec["answer_text"].endswith("CLARIFY: What is the mass?")      # after the ungrounded flag
