"""Tweaks: a switch must switch the whole tweak off, and only what it names.

The point of the registry is that a trained model is tested WITHOUT the rule that used to
stand in for it. These tests pin that: no classifier call escapes its gate, a hidden tool is
shown when its tweak is off, a made-up parameter is renamed only while the alias tweak is on
and an error while the strict check is on, and a typo in a tweak name is an error.
"""
import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from sigma_ground.mcp.benchmark import run_sigma_ground as H
from sigma_ground.mcp.benchmark import tweaks as T
from sigma_ground.mcp.benchmark.new_tools_classifier import HIDDEN_FROM_LLM

BENCH = Path(H.__file__).parent


@pytest.fixture(autouse=True)
def _clean():
    T.reset()
    yield
    T.reset()


def test_legacy_is_the_default_and_changes_nothing():
    assert T.snapshot() == {"profile": "legacy", "off": [], "on": []}
    assert T.label(T.snapshot()) == "legacy"
    assert all(T.on(n) for n, t in T.REGISTRY.items() if t.default)
    assert not T.on("strict_params")


def test_a_misspelled_name_is_an_error_not_a_silent_no_op():
    for bad in ("cascade.astr", "hidden_tool", "paramaliases"):
        with pytest.raises(ValueError):
            T.configure(off=[bad])
    with pytest.raises(ValueError):
        T.configure(profile="trainded")
    with pytest.raises(KeyError):
        T.on("nobody_registered_this")


def test_profiles_nest_and_only_remove_tweaks_that_can_be_replaced():
    T.configure("trained")
    t = set(T.snapshot()["off"])
    T.configure("model-only")
    m = set(T.snapshot()["off"])
    assert t < m and {f"cascade.{g}" for g in T.CASCADE_MODULES} <= m
    assert {"hidden_tools", "param_aliases", "body_name_chain"} <= t
    for name in m:                                       # nothing marked "stays" may be removed
        assert not T.REGISTRY[name].keep_reason, name
    assert T.snapshot()["on"] == ["strict_params"]


def test_every_tweak_a_profile_removes_declares_what_trains_its_replacement():
    """The pairing: tools_for() (tools/tweak_pairs.py) must find something to teach."""
    catalogue = set(T.server_catalogue())
    assert len(catalogue) > 200 and T.server_catalogue()["wear_analysis"].count("velocity_m_s") == 1
    for g in T.CASCADE_MODULES:
        assert T.emitted_tools(g, catalogue), f"cascade.{g} reaches no tool the data could teach"
    assert set(T.REGISTRY["hidden_tools"].covers) == set(HIDDEN_FROM_LLM)


def test_environment_configures_a_run(monkeypatch):
    monkeypatch.setenv("MENTAT_TWEAKS", "trained")
    monkeypatch.setenv("MENTAT_TWEAKS_OFF", "cascade.astro, cascade.optics")
    T.configure_from_env()
    s = T.snapshot()
    assert s["profile"] == "trained" and {"cascade.astro", "cascade.optics"} <= set(s["off"])
    assert not T.on("cascade.astro") and T.on("cascade.gr")


def test_every_cascade_call_in_the_harness_sits_behind_its_own_switch():
    src = (BENCH / "run_sigma_ground.py").read_text(encoding="utf-8")
    calls = re.findall(r"^\s+\w+ = classify_for_(\w+)\(question\)(.*)$", src, re.M)
    assert len(calls) == 14, [c[0] for c in calls]       # 14 since group_extreme (2026-10-30)
    for group, rest in calls:
        assert f'_tw.on("cascade.{group}")' in rest, f"classify_for_{group} runs ungated"
        assert f"cascade.{group}" in T.REGISTRY
    assert 'semantic_answer(question) if _tw.on("cascade.semantic")' in src
    assert set(T.CASCADE_MODULES) == {g for g, _ in calls} | {"semantic"}


def _mcp_tools(names):
    return [SimpleNamespace(name=n, description="", inputSchema={"type": "object", "properties": {}})
            for n in names]


def test_hidden_tools_are_shown_only_when_their_tweak_is_off():
    names = ["kinetic_energy", "orbital_velocity", "unruh_temperature", "convert_units"]
    shown = lambda: {t["function"]["name"] for t in H._ollama_tools(_mcp_tools(names))}  # noqa: E731
    assert shown() == {"kinetic_energy", "convert_units"}
    T.configure(off=["hidden_tools"])
    assert shown() == set(names)
    T.configure("trained")
    assert shown() == set(names)


def test_the_tool_index_drops_the_pattern_hints_with_their_tweak():
    tools = H._ollama_tools(_mcp_tools(["kinetic_energy"]))
    assert "PATTERN HINTS" in H._build_tool_index(tools)
    T.configure(off=["pattern_hints"])
    assert "PATTERN HINTS" not in H._build_tool_index(tools)


class _Http:
    def __init__(self, plan):
        self.plan = plan

    async def post(self, url, json=None, **kw):
        plan = self.plan
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: {"message": {"content": plan}})


class _Session:
    def __init__(self):
        self.called = []

    async def call_tool(self, name, args):
        self.called.append((name, dict(args)))
        text = json.dumps({"value": 42.0, "units": "J", "source": "fake"})
        return SimpleNamespace(content=[SimpleNamespace(text=text)])


def _route(plan, real_params):
    session = _Session()
    tools = [{"type": "function", "function": {"name": "free_fall_time", "description": "",
                                               "parameters": {"type": "object", "properties": {}}}}]
    rec = asyncio.run(H._route_once(session, _Http(json.dumps(plan)), "http://x", "m", "q?",
                                    "sys", tools, {"free_fall_time": real_params}, 0.0,
                                    "free", "v2"))
    return rec, session


MADE_UP = {"tool": "free_fall_time", "tool_args": {"height_m": 10.0, "gravity_ms2": 1.62}}


def test_a_made_up_parameter_is_renamed_while_the_alias_tweak_is_on():
    rec, session = _route(MADE_UP, {"height_m", "g_m_s2"})
    assert session.called == [("free_fall_time", {"height_m": 10.0, "g_m_s2": 1.62})]
    assert rec["tool_calls"][0]["renames_applied"] and "router_error" not in rec


def test_with_aliases_off_the_made_up_name_reaches_the_tool_and_is_reported():
    T.configure(off=["param_aliases"])
    rec, session = _route(MADE_UP, {"height_m", "g_m_s2"})
    assert session.called == [("free_fall_time", {"height_m": 10.0, "gravity_ms2": 1.62})]
    assert rec["tool_calls"][0]["unknown_params"] == ["gravity_ms2"]       # seen, not hidden
    assert not rec["tool_calls"][0]["renames_applied"]


def test_strict_params_turns_the_silent_default_into_an_error():
    T.configure("trained")
    rec, session = _route(MADE_UP, {"height_m", "g_m_s2"})
    assert session.called == []                                  # the tool is never run
    assert rec["router_error"].startswith("unknown parameter: free_fall_time(gravity_ms2")
    assert rec["tool_calls"][0]["unknown_params"] == ["gravity_ms2"]


def test_exact_parameter_names_pass_in_every_profile():
    for profile in ("legacy", "trained", "model-only"):
        T.configure(profile)
        rec, session = _route({"tool": "free_fall_time", "tool_args": {"height_m": 10.0}},
                              {"height_m", "g_m_s2"})
        assert session.called and rec["extracted_value"] == 42.0 and "router_error" not in rec


def test_the_answer_key_is_withheld_from_the_chain_loop_when_asked():
    src = (BENCH / "run_sigma_ground.py").read_text(encoding="utf-8")
    assert 'if not _tw.on("answer_key_hint"):\n        primary_tool_expected = None' in src
