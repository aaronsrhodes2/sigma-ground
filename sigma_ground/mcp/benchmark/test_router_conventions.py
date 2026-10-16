"""Tests for the convention seeds (router_conventions.py) and the builder's label-consistency filter (2026-10-15)."""
from __future__ import annotations

import importlib.util
import json
import random
import re
import sys
from pathlib import Path

import pytest

from sigma_ground.mcp.benchmark import router_conventions as RC
from sigma_ground.mcp.benchmark import router_plan as RP
from sigma_ground.mcp.tools import mathx

REPO = Path(__file__).resolve().parents[3]


def _shows(variable: str, question: str) -> bool:
    return re.search(rf"(?<![A-Za-z]){re.escape(variable)}(?![A-Za-z])", question) is not None


# ── the recipes ────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("holdout", [False, True])
def test_every_fourier_label_is_readable_from_its_question(holdout):
    rng = random.Random(7)
    for _ in range(400):
        p = RC.p_fourier(rng, holdout)
        assert p.args["convention"] in ("angular", "unitary", "ordinary")
        for key in ("x_var", "k_var"):
            if key in p.args:
                assert _shows(p.args[key], p.question), f"{key}={p.args[key]!r} is not in: {p.question}"
        if "x_var" not in p.args:
            assert "t" not in re.findall(r"\b[tx]\b", p.args["expression"]) and "x" in p.args["expression"]


@pytest.mark.parametrize("holdout", [False, True])
def test_every_series_label_matches_its_wording(holdout):
    rng = random.Random(8)
    seen = {"inclusive": 0, "exclusive": 0}
    for _ in range(400):
        p = RC.p_series(rng, holdout)
        conv = p.args["convention"]
        seen[conv] += 1
        assert str(p.args["order"]) in re.sub(r"[^\d]", " ", p.question).split(), p.question                # the order is stated
        text = p.question.lower()
        if conv == "inclusive":
            assert any(w in text for w in ("including", "through", "degree", "inclusive", "included", "includes")), p.question
        else:
            assert any(w in text for w in ("below", "not including", "o(", "stops before")), p.question
    assert all(v > 20 for v in seen.values())


def test_no_wording_is_shared_between_training_and_the_suite():
    for family in (RC.FOURIER, RC.SERIES):
        for conv, pools in family.items():
            assert not set(pools["train"]) & set(pools["holdout"]), conv


def test_the_oracle_accepts_the_right_convention_and_rejects_the_wrong_one():
    spec = lambda conv: {"kind": "fourier_conv", "expr": "exp(-2*x**2)", "xvar": "x", "kvar": "k", "conv": conv}
    for conv in ("ordinary", "angular", "unitary"):
        value = mathx.fourier_transform("exp(-2*x**2)", "x", "k", conv).value
        assert RC.check(spec(conv), value) is True
        for other in {"ordinary", "angular", "unitary"} - {conv}:
            assert RC.check(spec(other), value) is False, f"a {conv} result must not pass as {other}"


def test_the_series_oracle_agrees_with_both_readings():
    for conv, extra in (("exclusive", 0), ("inclusive", 1)):
        value = mathx.series_expansion("cos(x)", "x", "0", 4, conv).value
        assert RC.check({"kind": "series", "expr": "cos(x)", "var": "x", "point": "0", "order": 4 + extra}, value) is True


# ── the builder ────────────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def builder():
    sys.path.insert(0, str(REPO / "tools"))
    spec = importlib.util.spec_from_file_location("build_router_v2_conv_test", REPO / "tools" / "build_router_v2.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_router_v2_conv_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_a_label_naming_a_variable_the_question_never_shows_is_found(builder):
    f = builder.label_names_missing_variable
    plan = lambda **a: {"tool": "fourier_transform", "tool_args": {"expression": "exp(-t**2)", **a}}
    assert f("Fourier transform of e^(-t^2), as a function of w", plan(x_var="t", k_var="w")) is None
    assert f("Fourier transform of e^(-x^2)", plan(x_var="t")) == "x_var=t"
    assert f("Fourier transform of e^(-2x^2) in terms of f", plan(x_var="t", k_var="f")) == "x_var=t"
    assert f("Fourier transform of e^(-x^2)", plan()) is None                                                  # defaults name nothing
    assert f("anything", {"decline": "not physics"}) is None
    assert f("Fourier transform of e^(-t^2) in omega", {"steps": [{"tool": "fourier_transform", "args": {"k_var": "w"}}]}) == "k_var=w"


def test_a_filtered_seed_keeps_its_slot_so_the_rewrite_cache_still_lines_up(tmp_path, builder):
    v1 = tmp_path / "v1"
    v1.mkdir()
    for name in ("train", "val"):
        (v1 / f"{name}.jsonl").write_text("", encoding="utf-8")
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    base = {"meta": {"source": "recipe", "family": "math", "no_rewrite": True, "verified": []}, "args_meta": []}
    rows = [{**base, "question": "Fourier transform of e^(-2x^2) in terms of f", "group": "a",
             "plan": RP.make_plan([{"tool": "fourier_transform", "args": {"expression": "exp(-2*t**2)", "x_var": "t", "k_var": "f"}}])},      # inconsistent
            {**base, "question": "Fourier transform of e^(-t^2) in angular frequency, as a function of w", "group": "b",
             "plan": RP.make_plan([{"tool": "fourier_transform", "args": {"expression": "exp(-t**2)", "x_var": "t", "k_var": "w", "convention": "angular"}}])}]
    (seeds / "seeds.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    old_plan = builder.PLAN
    try:
        out = tmp_path / "out"
        m = builder.assemble(v1, [seeds], out, teacher=lambda q, s: "x", variants=1, guard=None, val_pct=0, plan="v4")
    finally:
        builder.PLAN = old_plan
    assert m["recipe_seeds"] == 2 and m["seeds_without_a_record_label_inconsistent"] == 1
    recs = [json.loads(l) for l in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 1 and "angular" in recs[0]["messages"][2]["content"]
    cache = [json.loads(l) for l in (out / "rewrites.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [c["i"] for c in cache] == [0, 1]                                                                # the dropped seed still holds index 0
    plan = json.loads(recs[0]["messages"][2]["content"])
    assert plan["tool_args"] == {"expression": "exp(-t**2)", "x_var": "t", "k_var": "w", "convention": "angular"}   # not unit-bearing: untouched by the v4 relabel
