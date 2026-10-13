"""Router plan v3: "$n.field" references, and a dataset built in that format.

A lookup tool (solar_system_body, named_star) returns several fields. In v2 the harness
chose which one a later call received, from the wording of the question; in v3 the plan says
("$1.mass_kg"), so the model is trained to do what a harness rule used to do.
"""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sigma_ground.mcp.benchmark import router_plan as RP
from sigma_ground.mcp.benchmark import run_sigma_ground as H

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))


def test_a_field_reference_reads_one_field_of_the_whole_result():
    raws = [{"mass_kg": 6.0e24, "radius_m": 6.4e6}]
    out = RP.resolve_args({"mass_kg": "$1.mass_kg", "radius_m": "$1.radius_m", "k": 3}, [1.0], raws)
    assert out == {"mass_kg": 6.0e24, "radius_m": 6.4e6, "k": 3}
    assert RP.refs_in({"a": "$2.radius_m", "b": "$1", "c": "plain"}) == [2, 1]


@pytest.mark.parametrize("args,values,raws", [
    ({"x": "$1.nothing_here"}, [1.0], [{"mass_kg": 1.0}]),     # no such field
    ({"x": "$1.mass_kg"}, [1.0], [5.0]),                        # a scalar has no fields
    ({"x": "$1.name"}, [1.0], [{"name": "earth"}]),             # a field that is not a number
    ({"x": "$2.mass_kg"}, [1.0], [{"mass_kg": 1.0}]),           # step 2 has not run
])
def test_a_bad_field_reference_is_an_error_not_a_guess(args, values, raws):
    with pytest.raises(RP.PlanError):
        RP.resolve_args(args, values, raws)


def test_the_v3_prompt_adds_exactly_the_field_sentence():
    v2, v3 = RP.system_prompt("v2"), RP.system_prompt("v3")
    assert v3 != v2 and "$1.<field>" in v3 and "$1.<field>" not in v2
    assert v3.replace(' When a call looks something up and returns several fields '
                      '(solar_system_body, named_star), "$1.<field>" is one of them, '
                      'e.g. "$1.mass_kg".', "") == v2
    assert RP.system_prompt("v1") not in (v2, v3)
    assert "v3" in RP.MULTISTEP and "v2" in RP.MULTISTEP and "v1" not in RP.MULTISTEP


def test_a_plan_with_field_references_is_a_valid_plan():
    plan = {"steps": [{"tool": "solar_system_body", "tool_args": {"body_name": "mars"}},
                      {"tool": "escape_velocity",
                       "tool_args": {"mass_kg": "$1.mass_kg", "radius_m": "$1.radius_m"}}]}
    c = RP.canonical(plan)
    RP.check_refs(c["steps"])
    with pytest.raises(RP.PlanError):
        RP.check_refs([{"tool": "a", "args": {"x": "$1.mass_kg"}}])       # refers to itself
    assert RP.keys_in_training_order(RP.router_format(["solar_system_body", "escape_velocity"]))


TOOLS = {
    "solar_system_body": lambda a: {"value": {"mass_kg": 6.4171e23, "radius_m": 3.3895e6},
                                    "units": "", "source": "fake"},
    "escape_velocity": lambda a: {"value": (2 * 6.674e-11 * a["mass_kg"] / a["radius_m"]) ** 0.5,
                                  "units": "m/s", "source": "fake"},
}


class _Session:
    def __init__(self):
        self.called = []

    async def call_tool(self, name, args):
        self.called.append((name, dict(args)))
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(TOOLS[name](args)))])


class _Http:
    def __init__(self, plan):
        self.plan = plan

    async def post(self, url, json=None, **kw):
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: {"message": {"content": self.plan}})


def test_the_harness_follows_a_field_chain_with_no_name_resolution_of_its_own():
    plan = json.dumps({"steps": [
        {"tool": "solar_system_body", "tool_args": {"body_name": "mars"}},
        {"tool": "escape_velocity", "tool_args": {"mass_kg": "$1.mass_kg", "radius_m": "$1.radius_m"}}]})
    tools = [{"type": "function", "function": {"name": n, "description": "", "parameters": {}}} for n in TOOLS]
    session = _Session()
    real = {"solar_system_body": {"body_name"}, "escape_velocity": {"mass_kg", "radius_m"}}
    rec = asyncio.run(H._route_once(session, _Http(plan), "http://x", "m", "escape velocity of Mars?",
                                    "sys", tools, real, 0.0, "free", "v3"))
    assert session.called[1] == ("escape_velocity", {"mass_kg": 6.4171e23, "radius_m": 3.3895e6})
    assert rec["finalized_from_tool"] == "escape_velocity" and rec["chain_length"] == 2
    assert 4900 < rec["extracted_value"] < 5100 and "router_error" not in rec


def test_a_missing_field_stops_the_plan_after_the_lookup():
    plan = json.dumps({"steps": [
        {"tool": "solar_system_body", "tool_args": {"body_name": "mars"}},
        {"tool": "escape_velocity", "tool_args": {"mass_kg": "$1.mass", "radius_m": "$1.radius_m"}}]})
    tools = [{"type": "function", "function": {"name": n, "description": "", "parameters": {}}} for n in TOOLS]
    session = _Session()
    rec = asyncio.run(H._route_once(session, _Http(plan), "http://x", "m", "q", "sys", tools, None,
                                    0.0, "free", "v3"))
    assert [c[0] for c in session.called] == ["solar_system_body"]
    assert rec["router_error"].startswith("bad reference")


def test_the_builder_writes_v3_records_and_keeps_the_rewrite_cache_aligned(tmp_path):
    import build_router_v2 as B
    v1 = tmp_path / "v1"
    v1.mkdir()
    old_plan = {"tool": "kinetic_energy", "args": {"mass_kg": 2.0, "velocity_m_s": 3.0}}
    for name in ("train", "val"):
        rec = {"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": f"v1 {name} question?"},
                            {"role": "assistant", "content": json.dumps(old_plan)}],
               "meta": {"group": f"v1:{name}", "source": "template"}}
        (v1 / f"{name}.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    rows = [
        {"question": "How hard is it to escape Mars?",
         "plan": RP.make_plan([{"tool": "solar_system_body", "args": {"body_name": "mars"}},
                               {"tool": "escape_velocity", "args": {"mass_kg": "$1.mass_kg", "radius_m": "$1.radius_m"}}]),
         "group": "lookup:1", "meta": {"source": "recipe", "family": "lookup", "verified": [{"tool": "x", "value": 1.0}]},
         "args_meta": []},
        {"question": "Which stocks should I buy?", "plan": {"decline": "not a question about a physical system"},
         "group": "decline:1", "meta": {"source": "decline", "family": "decline", "no_rewrite": True, "verified": []},
         "args_meta": []},
    ]
    (seeds / "seeds.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    asked = []

    def teacher(question, salt):
        asked.append(question)
        return "Tell me how hard it is to leave Mars?"

    out = tmp_path / "out"
    m = B.assemble(v1, [seeds], out, teacher=teacher, variants=1, guard=None, val_pct=0, plan="v3")
    assert m["format"] == "v3" and m["system_prompt"] == RP.ROUTER_SYSTEM_PROMPT_V3
    recs = [json.loads(line) for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert recs and all(r["messages"][0]["content"] == RP.ROUTER_SYSTEM_PROMPT_V3 for r in recs)
    assert any("$1.mass_kg" in r["messages"][2]["content"] for r in recs)
    assert asked == ["How hard is it to escape Mars?"]                  # the decline was not rewritten
    cache = [json.loads(line) for line in (out / "rewrites.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [c["i"] for c in cache] == [0, 1] and cache[1]["rewrites"] == []


def test_a_rewrite_cache_that_does_not_line_up_with_the_seeds_is_refused(tmp_path):
    """The teacher-rewrite cache is indexed by seed; a seed inserted before the cached ones
    would hang another question's rewrite on this plan. That must stop the build."""
    import build_router_v2 as B
    from sigma_ground.mcp.benchmark import router_dataset as D
    seed = {"question": "A 5 kg rock moves at 10 m/s. What is its kinetic energy?",
            "plan": {"tool": "kinetic_energy", "tool_args": {"mass_kg": 5.0, "velocity_m_s": 10.0}},
            "group": "g1", "meta": {"source": "recipe", "verified": [{"tool": "kinetic_energy", "value": 250.0}]},
            "args_meta": [{"name": "m", "value": 5.0, "text": "5 kg", "unit_text": "kg", "key_text": ""},
                          {"name": "v", "value": 10.0, "text": "10 m/s", "unit_text": "m/s", "key_text": ""}]}
    cache = tmp_path / "rewrites.jsonl"
    cache.write_text(json.dumps({"i": 0, "rewrites": ["A body of 7 kg goes 99 m/s. What is its energy?"]}) + "\n",
                     encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not line up"):
        B.teacher_pass([seed], lambda q, k: "unused", 1, cache, None, set(), log=lambda *a: None)
