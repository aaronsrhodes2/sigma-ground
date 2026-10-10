"""Router mode (run_sigma_ground --router): one model call, one JSON plan.

After the deterministic cascade the model is asked ONCE, with the prompt the
router fine-tune is trained on, for a plan constrained to the real tool names.
The harness runs the plan at most once and reports the result itself. The last
test closes the loop with router_dataset.py: a model that reproduces its
training targets must get exactly the verified answers back through the real
MCP server.
"""
import asyncio
import json
import os
import sys
from types import SimpleNamespace

import httpx
import pytest

from sigma_ground.mcp.benchmark import groundedness as G
from sigma_ground.mcp.benchmark import interpreter_demo
from sigma_ground.mcp.benchmark import router_dataset as D
from sigma_ground.mcp.benchmark import run_sigma_ground as R

GOLD = {"value": {"name": "gold", "atomic_number": 79, "atomic_mass_amu": 196.97},
        "units": "", "source": "periodictable"}
TOOLS = {
    "element_atomic_data": lambda a: GOLD,
    "schwarzschild_radius": lambda a: {"value": 2 * 6.674e-11 * a["mass_kg"] / 2.998e8 ** 2,
                                       "units": "m", "source": "gr_basics"},
    "density": lambda a: {"value": None, "units": "", "source": "not in table",
                          "inputs": {"value": 1.0}},
}
OLLAMA_TOOLS = [{"type": "function", "function": {"name": n, "description": "",
                                                  "parameters": {}}} for n in TOOLS]


class _FakeSession:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, args):
        self.calls.append((name, dict(args)))
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(TOOLS[name](args)))])


def _ollama(monkeypatch, replies):
    """Patch httpx with a scripted Ollama; returns the list of request bodies.
    A reply is the assistant's content string, or (status, None)."""
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
            status, content = reply if isinstance(reply, tuple) else (200, reply)
            def raise_for_status():
                if status >= 400:
                    raise httpx.HTTPStatusError("bad", request=None, response=None)
            return SimpleNamespace(status_code=status, raise_for_status=raise_for_status,
                                   json=lambda: {"message": {"content": content}})

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setattr(interpreter_demo, "semantic_answer", lambda q: None)
    return bodies


def _ask(question, session=None, tools=OLLAMA_TOOLS, real_params=None, prompt=None):
    session = session or _FakeSession()
    rec = asyncio.run(R._run_one_question(
        session, "http://fake/api/chat", "router-model", tools, question,
        prompt or R._router_system_prompt(tools, False),
        real_params_by_tool=real_params, router=True))
    assert not any(k.endswith("_hit") for k in rec), "a classifier caught the question"
    return G.apply_gate(rec), session


def test_plan_runs_once_and_mentat_reports(monkeypatch):
    bodies = _ollama(monkeypatch, ['{"tool": "element_atomic_data", "args": {"element_symbol": "Au"}}'])
    rec, session = _ask("What's the atomic number of gold?")
    assert session.calls == [("element_atomic_data", {"element_symbol": "Au"})]
    assert rec["extracted_value"] == 79 and rec["turns"] == 1
    assert rec["answer_text"].startswith("ANSWER: 79")
    assert rec["grounded"] and "harness-finalized" in rec["groundedness_reason"]
    # one request: the training prompt, no tool loop, the name pinned to the catalogue
    (body,) = bodies
    assert body["messages"][0]["content"] == D.ROUTER_SYSTEM_PROMPT
    assert "tools" not in body and body["options"]["temperature"] == 0
    enum = body["format"]["anyOf"][0]["properties"]["tool"]["enum"]
    assert set(enum) == set(TOOLS) | {"request_clarification"}


def test_an_ollama_without_schema_support_gets_json_mode(monkeypatch):
    bodies = _ollama(monkeypatch, [(400, None),
                                   '{"tool": "element_atomic_data", "args": {"element_symbol": "Au"}}'])
    rec, _ = _ask("What's the atomic number of gold?")
    assert [b["format"] if isinstance(b["format"], str) else "schema" for b in bodies] == \
        ["schema", "json"]
    assert rec["extracted_value"] == 79


def test_decline_and_clarification_run_nothing(monkeypatch):
    _ollama(monkeypatch, ['{"decline": "not a question about a physical system"}',
                          '{"tool": "request_clarification", "args": '
                          '{"variable": "mass_kg", "question": "What is the mass?"}}'])
    rec, session = _ask("Who wrote Pride and Prejudice?")
    assert session.calls == [] and rec["router_decline"]
    assert not rec["grounded"]           # a model's call, not a tool's: flagged
    rec, session = _ask("How small to be a black hole?")
    assert session.calls == []
    assert rec["router_clarify"] == "mass_kg" and "What is the mass?" in rec["answer_text"]


@pytest.mark.parametrize("reply,error", [
    ("I think it's 79.", "unparseable plan"),
    ('{"tool": "make_coffee", "args": {}}', "unknown tool 'make_coffee'"),
])
def test_bad_replies_are_recorded_not_raised(monkeypatch, reply, error):
    _ollama(monkeypatch, [reply])
    rec, session = _ask("What's the atomic number of gold?")
    assert session.calls == [] and rec["router_error"] == error
    assert not rec["grounded"]


def test_a_declined_tool_call_is_not_an_answer(monkeypatch):
    # density's null value echoes {"inputs": {"value": 1.0}}: not an answer
    _ollama(monkeypatch, ['{"tool": "density", "args": {"material": "unobtainium"}}'])
    rec, session = _ask("What's the atomic number of gold?")
    assert len(session.calls) == 1 and rec["extracted_value"] is None
    assert rec["router_error"] == "density returned no value" and not rec["grounded"]


def test_plan_in_prose_or_fence_and_aliased_args(monkeypatch):
    _ollama(monkeypatch, ['Sure:\n```json\n{"tool": "schwarzschild_radius", "args": {"mass": 1.989e30}}\n```'])
    rec, session = _ask("How small to be a black hole?",
                        real_params={"schwarzschild_radius": {"mass_kg"}})
    assert session.calls == [("schwarzschild_radius", {"mass_kg": 1.989e30})]
    assert rec["extracted_value"] == pytest.approx(2953.6, rel=1e-3)
    assert rec["tool_calls"][0]["renames_applied"] == ["mass->mass_kg"]


def test_index_is_opt_in():
    bare = R._router_system_prompt(OLLAMA_TOOLS, False)
    indexed = R._router_system_prompt(OLLAMA_TOOLS, True)
    assert bare == D.ROUTER_SYSTEM_PROMPT
    assert indexed.startswith(bare) and "schwarzschild_radius(" in indexed


# ── the contract with router_dataset.py, through the real server ─────────

def test_a_router_that_reproduces_its_training_targets_scores_exactly(monkeypatch):
    pytest.importorskip("mcp.server.fastmcp")
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    monkeypatch.setattr(interpreter_demo, "semantic_answer", lambda q: None)
    wanted = {"kinetic_energy", "density", "schwarzschild_radius", "em_wave_wavelength",
              "convert_units", "first_ionization_energy", "weak_acid_ph", "carnot_efficiency"}

    async def run():
        async with stdio_client(D.server_params(6)) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                mcp_tools = (await session.list_tools()).tools
                tools = R._ollama_tools(mcp_tools)
                real_params = R._build_real_params_map(tools)
                specs = [t for t in await D.visible_tools(session) if t.name in wanted]
                examples, _ = await D.build(session, specs, D.manifest_by_tool(),
                                            per_tool=4, seed=2, clarify_rate=1.0,
                                            guard=D.BenchmarkGuard.from_repo(),
                                            log=lambda _: None)
                # only what the cascade leaves to the model reaches the router
                examples = [e for e in examples if e.meta.get("classifier") == "silent"]
                prompt = R._router_system_prompt(tools, False)
                _ollama(monkeypatch, [e.to_record()["messages"][2]["content"] for e in examples])
                out = []
                for e in examples:
                    rec = await R._run_one_question(
                        session, "http://fake/api/chat", "router", tools, e.question,
                        prompt, real_params_by_tool=real_params, router=True)
                    out.append((e, rec))
                return out

    pairs = asyncio.run(run())
    kinds = {e.source for e, _ in pairs}
    assert {"template", "clarify", "decline"} <= kinds and len(pairs) > 20
    for e, rec in pairs:
        if e.source == "decline":
            assert rec["router_decline"], e.question
        elif e.plan["tool"] == "request_clarification":
            assert rec["router_clarify"] == e.meta["missing"], e.question
        else:
            assert rec["finalized_from_tool"] == e.plan["tool"], (e.question, rec.get("router_error"))
            assert rec["extracted_value"] == pytest.approx(e.meta["verified"]["value"], rel=1e-9)
