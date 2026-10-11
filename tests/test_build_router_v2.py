"""tools/build_router_v2.py: v1 data converted to the v2 format, recipe seeds added,
teacher rewrites accepted only when they keep every number and unit."""
import importlib.util
import json
from pathlib import Path

import pytest

from sigma_ground.mcp.benchmark import router_dataset as D
from sigma_ground.mcp.benchmark import router_plan as RP

SPEC = importlib.util.spec_from_file_location(
    "build_router_v2", Path(__file__).resolve().parents[1] / "tools" / "build_router_v2.py")
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def v1(question, plan, group, source="template"):
    return {"messages": [{"role": "system", "content": D.ROUTER_SYSTEM_PROMPT},
                         {"role": "user", "content": question},
                         {"role": "assistant", "content": json.dumps(plan)}],
            "meta": {"group": group, "source": source, "tool": "x"}}


def write(dirpath: Path, name: str, rows: list):
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


SEED = {
    "question": "A 5 kg ball moves at 3 m/s. What is its kinetic energy?",
    "plan": {"tool": "kinetic_energy", "tool_args": {"mass_kg": 5.0, "velocity_m_s": 3.0}},
    "group": "recipe:demo:1:0",
    "meta": {"source": "recipe", "family": "chain", "recipe": "demo", "steps": 1, "classifier": "silent",
             "verified": [{"tool": "kinetic_energy", "value": 22.5, "units": "J"}]},
    "args_meta": [{"name": "m", "value": 5.0, "text": "5 kg", "unit_text": "kg", "key_text": ""},
                  {"name": "v", "value": 3.0, "text": "3 m/s", "unit_text": "m/s", "key_text": ""}],
}


@pytest.fixture(autouse=True)
def neutral_classifier(monkeypatch):
    monkeypatch.setattr(D, "classifier_verdict", lambda q, plan: "silent")


def test_a_v1_record_is_rewritten_in_v2():
    rec = v1("Wavelength 2.42 µm, slit 0.233 mm.", {"tool": "double_slit_fringe_spacing",
             "args": {"wavelength_m": 2.42e-06, "slit_separation_m": 0.000233, "screen_distance_m": 2.68}}, "g1")
    out = B.convert_v1(rec)
    assert out["messages"][0]["content"] == RP.ROUTER_SYSTEM_PROMPT_V2
    assert out["messages"][1]["content"] == rec["messages"][1]["content"]       # the question is untouched
    text = out["messages"][2]["content"]
    assert text == ('{"tool": "double_slit_fringe_spacing", "tool_args": {"wavelength_m": 2.42e-6, '
                    '"slit_separation_m": 0.000233, "screen_distance_m": 2.68}}')
    assert out["meta"]["group"] == "g1" and out["meta"]["origin"] == "v1"
    B.validate(out)


def test_clarifications_and_declines_convert():
    c = B.convert_v1(v1("q", {"tool": "request_clarification",
                              "args": {"variable": "mass_kg", "question": "What is the mass?"}}, "g2", "clarify"))
    assert json.loads(c["messages"][2]["content"]) == {
        "tool": "request_clarification", "tool_args": {"variable": "mass_kg", "question": "What is the mass?"}}
    d = B.convert_v1(v1("q", {"decline": "not physics"}, "g3", "decline"))
    assert json.loads(d["messages"][2]["content"]) == {"decline": "not physics"}


@pytest.mark.parametrize("mutate", [
    lambda r: r["messages"][0].update(content=D.ROUTER_SYSTEM_PROMPT),          # the old prompt
    lambda r: r["messages"][2].update(content='{"tool": "t", "tool_args": {"x": 2.4e-06}}'),   # padded exponent
    lambda r: r["messages"][2].update(content='{"tool": "t", "args": {"x": 1}}'),               # a v1 key
    lambda r: r["messages"][2].update(content='{"steps": [{"tool": "t", "tool_args": {"x": "$1"}}, '
                                              '{"tool": "u", "tool_args": {}}]}'),             # a forward reference
    lambda r: r["messages"][2].update(content="not json"),
])
def test_validate_refuses_what_the_model_must_not_learn(mutate):
    rec = B.record("q", {"tool": "t", "tool_args": {"x": 1.0}}, {"group": "g"})
    B.validate(rec)
    mutate(rec)
    with pytest.raises((AssertionError, ValueError, RP.PlanError)):
        B.validate(rec)


def build(tmp_path, v1_rows_train, v1_rows_val, seeds, **kw):
    write(tmp_path / "v1", "train.jsonl", v1_rows_train)
    write(tmp_path / "v1", "val.jsonl", v1_rows_val)
    write(tmp_path / "rec", "seeds.jsonl", seeds)
    return B.assemble(tmp_path / "v1", tmp_path / "rec", tmp_path / "out", **kw)


def test_assemble_keeps_the_v1_split_and_adds_recipe_seeds(tmp_path):
    tr = [v1("Mass of 2 kg.", {"tool": "momentum", "args": {"mass_kg": 2.0}}, "g1")]
    va = [v1("Mass of 3 kg.", {"tool": "momentum", "args": {"mass_kg": 3.0}}, "g2")]
    seeds = [SEED | {"group": f"recipe:demo:{i}:0", "question": f"A {i + 2} kg ball moves at 3 m/s. Energy?"}
             for i in range(40)]
    m = build(tmp_path, tr, va, seeds, log=lambda *a: None)
    train = B.read_jsonl(tmp_path / "out" / "train.jsonl")
    val = B.read_jsonl(tmp_path / "out" / "val.jsonl")
    assert train[0]["meta"]["group"] == "g1" and val[0]["meta"]["group"] == "g2"      # v1 split kept
    assert m["v1_records"] == {"train": 1, "val": 1} and m["recipe_seeds"] == 40
    assert len(train) + len(val) == 42 and len(val) >= 2
    assert {r["meta"]["group"] for r in train}.isdisjoint({r["meta"]["group"] for r in val})
    for r in train + val:
        B.validate(r)
    assert m["train"]["by_origin"]["v1"] == 1


def test_a_recipe_question_already_in_v1_is_not_added_twice(tmp_path):
    tr = [v1(SEED["question"], {"tool": "kinetic_energy", "args": {"mass_kg": 5.0, "velocity_m_s": 3.0}}, "g1")]
    m = build(tmp_path, tr, [], [SEED], log=lambda *a: None)
    assert m["recipe_seeds"] == 0


def test_teacher_rewrites_must_keep_numbers_and_units(tmp_path):
    replies = iter(["How much kinetic energy does a 5 kg ball moving at 3 m/s carry?",   # kept
                    "How much energy does a 5 kg ball moving at 4 m/s carry?",           # a number changed
                    "How much kinetic energy does a 5 g ball moving at 3 m/s carry?"])   # a unit changed
    m = build(tmp_path, [], [], [SEED], teacher=lambda q, k: next(replies), variants=3, log=lambda *a: None)
    rows = B.read_jsonl(tmp_path / "out" / "train.jsonl") + B.read_jsonl(tmp_path / "out" / "val.jsonl")
    rewrites = [r for r in rows if r["meta"]["source"] == "rewrite"]
    assert len(rewrites) == 1 and "5 kg ball moving at 3 m/s" in rewrites[0]["messages"][1]["content"]
    assert rewrites[0]["meta"]["group"] == SEED["group"]                    # stays with its seed
    assert rewrites[0]["messages"][2]["content"] == B.record("x", SEED["plan"], {})["messages"][2]["content"]
    assert m["rewrite_counts"] == {"attempts": 3, "failed the rewrite check": 2, "accepted": 1}


def test_a_duplicate_rewrite_and_a_resumed_pass(tmp_path):
    calls = []

    def teacher(q, k):
        calls.append(k)
        return "How much kinetic energy does a 5 kg ball moving at 3 m/s carry?"
    build(tmp_path, [], [], [SEED], teacher=teacher, variants=3, log=lambda *a: None)
    rows = B.read_jsonl(tmp_path / "out" / "train.jsonl") + B.read_jsonl(tmp_path / "out" / "val.jsonl")
    assert sum(r["meta"]["source"] == "rewrite" for r in rows) == 1          # the 2 repeats are duplicates
    assert len(calls) == 3
    # a second run reads rewrites.jsonl and does not call the teacher again
    build(tmp_path, [], [], [SEED], teacher=teacher, variants=3, log=lambda *a: None)
    assert len(calls) == 3


def test_a_benchmark_look_alike_rewrite_is_dropped(tmp_path):
    class Guard:
        def check(self, q, tool=None, value=None):
            return "text overlap" if "carry" in q else None
    m = build(tmp_path, [], [], [SEED], guard=Guard(), variants=1, log=lambda *a: None,
              teacher=lambda q, k: "How much kinetic energy does a 5 kg ball moving at 3 m/s carry?")
    assert m["rewrite_counts"]["benchmark overlap"] == 1 and "accepted" not in m["rewrite_counts"]
