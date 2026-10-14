"""Harness-side finalization (run_sigma_ground finalize="chain").

The LLM path used to run to max turns on nearly every question: the model
found the right tool, then re-called it (qwen2.5) or wandered into an
unrelated tool and looped on that (qwen3). Now the first successful call ends
the question unless the model extends a data-flow chain -- a different tool
consuming a value the chain produced. These tests drive the real
_run_one_question with a scripted model and a fake MCP session (no Ollama).
"""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from sigma_ground.mcp.benchmark import groundedness as G
from sigma_ground.mcp.benchmark import interpreter_demo
from sigma_ground.mcp.benchmark import run_sigma_ground as R

SUN = {"value": {"mass_kg": 1.98892e30, "radius_m": 6.957e8},
       "units": "", "source": "sun table"}
GOLD = {"value": {"name": "gold", "atomic_number": 79, "atomic_mass_amu": 196.97},
        "units": "", "source": "periodictable"}


def _tool(name, value, units="", source="lib", **extra):
    return {"value": value, "units": units, "source": source, **extra}


TOOLS = {
    "solar_system_body":   lambda a: SUN,
    "element_atomic_data": lambda a: GOLD,
    "density":             lambda a: _tool("density", 19300.0, "kg/m^3"),
    "schwarzschild_radius": lambda a: _tool(
        "schwarzschild_radius", 2 * 6.674e-11 * float(a["mass_kg"]) / 2.998e8 ** 2,
        "m", "gr_basics", provenance_tag="STANDARD"),
    "broken":              lambda a: {"error": "bad input"},
}


class _FakeSession:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, args):
        self.calls.append(name)
        text = json.dumps(TOOLS[name](args))
        return SimpleNamespace(content=[SimpleNamespace(text=text)])


def _call(name, **args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def _say(text):
    return {"role": "assistant", "content": text}


def _run(monkeypatch, question, script, finalize="chain"):
    """Run one question against a scripted model; returns (record, session)."""
    script = list(script)

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            msg = script.pop(0) if script else _say("ANSWER: [SOURCE: Fitted due to incompetence]")
            return SimpleNamespace(raise_for_status=lambda: None,
                                   json=lambda: {"message": msg})

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setattr(interpreter_demo, "semantic_answer", lambda q: None)
    session = _FakeSession()
    rec = asyncio.run(R._run_one_question(
        session, "http://fake", "fake-model", [], question, "sys",
        finalize=finalize))
    assert not any(k.endswith("_hit") for k in rec), (
        f"a deterministic classifier caught {question!r}; pick a question "
        f"that reaches the LLM path")
    return G.apply_gate(rec), session


# ── the chain-step predicate ───────────────────────────────────────────

def _step(name, payload):
    return {"name": name, "args": {}, "result_text": json.dumps(payload)}


def test_lookup_feeding_a_computation_is_a_chain_step():
    chain = [_step("solar_system_body", SUN)]
    # the model re-types 1.98892e30 at 4 sig figs -- still the same quantity
    assert R._is_chain_step("schwarzschild_radius", {"mass_kg": 1.989e30}, chain)


def test_recall_and_unrelated_calls_are_not_chain_steps():
    chain = [_step("element_atomic_data", GOLD)]
    assert not R._is_chain_step("element_atomic_data", {"element_symbol": "Au"}, chain)
    assert not R._is_chain_step("density", {"material": "gold"}, chain)
    # a different tool with a number that matches nothing the chain produced
    assert not R._is_chain_step("density", {"temperature_k": 300}, chain)
    # zero never links (too many tools default an argument to 0)
    zero = [_step("t", _tool("t", 0.0))]
    assert not R._is_chain_step("u", {"x": 0.0}, zero)


def test_declined_call_is_not_a_success_despite_echoed_inputs():
    # convert_units with an unknown unit: null value, but its inputs echo a
    # number under "value" that a regex extractor would happily return
    declined = {"value": None, "units": "", "source": "pint undefined unit",
                "inputs": {"value": 1.0, "from_units": "furlongz", "to_units": "m"}}
    assert R._tool_call_value(_step("convert_units", declined)) is None
    assert R._tool_call_value(_step("density", _tool("density", 19300.0))) == (19300.0, "")


# ── end-to-end through _run_one_question ───────────────────────────────

def test_recall_loop_ends_on_the_first_result(monkeypatch):
    # qwen2.5's failure: the right call, then the same call until max turns
    rec, session = _run(monkeypatch, "What's the atomic number of gold?", [
        _call("element_atomic_data", element_symbol="Au"),
        _call("element_atomic_data", element_symbol="Au"),
        _call("element_atomic_data", element_symbol="Au"),
    ])
    assert session.calls == ["element_atomic_data"]     # the re-call never ran
    assert rec["extracted_value"] == 79
    assert rec["turns"] == 2
    assert rec["finalized_from_tool"] == "element_atomic_data"
    assert rec["grounded"] and "harness-finalized" in rec["groundedness_reason"]
    assert rec["answer_text"].startswith("ANSWER: 79")
    assert "[SOURCE: periodictable]" in rec["answer_text"]


def test_wander_is_not_run_and_cannot_replace_the_answer(monkeypatch):
    # qwen3's failure: the right call, then an unrelated tool it loops on
    rec, session = _run(monkeypatch, "What's the atomic number of gold?", [
        _call("element_atomic_data", element_symbol="Au"),
        _call("density", material="gold"),
    ])
    assert session.calls == ["element_atomic_data"]
    assert rec["extracted_value"] == 79
    assert "unrelated call" in rec["finalize_reason"]


def test_lookup_then_compute_chain_reports_the_computation(monkeypatch):
    rec, session = _run(monkeypatch, "How big a black hole would the Sun make?", [
        _call("solar_system_body", body_name="sun"),
        _call("schwarzschild_radius", mass_kg=1.989e30),
        _say("ANSWER: about 3 km"),
    ])
    assert session.calls == ["solar_system_body", "schwarzschild_radius"]
    assert rec["chain_length"] == 2
    assert rec["extracted_value"] == pytest.approx(2953.6, rel=1e-3)
    assert rec["extracted_units"] == "m"
    assert "provenance STANDARD" in rec["answer_text"]
    assert "solar_system_body" in rec["answer_text"]    # the path is shown
    # the model's own prose is kept, but labelled and never used for the value
    assert "Model note (not used for the value): ANSWER: about 3 km" in rec["answer_text"]
    assert rec["grounded"]


def test_failed_first_call_leaves_the_model_free_to_retry(monkeypatch):
    rec, session = _run(monkeypatch, "What's the atomic number of gold?", [
        _call("broken", x=1),
        _call("element_atomic_data", element_symbol="Au"),
        _say("ANSWER: 79"),
    ])
    assert session.calls == ["broken", "element_atomic_data"]
    assert rec["extracted_value"] == 79 and rec["finalized_from_tool"]


def test_legacy_mode_keeps_the_old_loop(monkeypatch):
    rec, session = _run(monkeypatch, "What's the atomic number of gold?", [
        _call("element_atomic_data", element_symbol="Au"),
        _call("element_atomic_data", element_symbol="Au"),
        _say("ANSWER: 79"),
    ], finalize="model")
    assert session.calls == ["element_atomic_data", "element_atomic_data"]
    assert "finalized_from_tool" not in rec
    assert rec["grounded"] and rec["extracted_value"] == 79
