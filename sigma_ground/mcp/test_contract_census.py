"""The Mentat tool contract, held to a ratchet.

Runs the cheap parts of tools/mentat_contract_census.py on every test run (about 5 s, in process, no network) and
compares the headline numbers with misc/contract_budget.json. A number may only go down; when it does, lower the
budget in the same commit (this test says so). New tools therefore cannot make the interface worse, and the
silently-wrong-notation count is pinned at zero.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BUDGET = json.loads((REPO / "misc" / "contract_budget.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def census():
    pytest.importorskip("mcp")
    spec = importlib.util.spec_from_file_location("mentat_contract_census", REPO / "tools" / "mentat_contract_census.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    server = mod.build_server()
    rows = mod.schema_census(server)
    errors = mod.error_probes(server, rows)
    notation = mod.notation_probe(server)
    quantity = {"p2_probed": 0, "p2_accepted": 0}
    return {"rows": rows, "notation": notation, "headline": mod.headline(rows, errors, notation, quantity)}


@pytest.mark.parametrize("key", [k for k in BUDGET if not k.startswith("_")])
def test_the_contract_does_not_get_worse(census, key):
    got, ceiling = census["headline"][key], BUDGET[key]
    assert got <= ceiling, f"{key}: {got} > budget {ceiling} (a change made the tool contract worse)"
    if got < ceiling:
        pytest.fail(f"{key} improved: {got} < budget {ceiling}. Lower it in misc/contract_budget.json (the ratchet only turns down).")


def test_every_notation_in_the_table_is_right_through_the_server(census):
    n = census["notation"]
    assert n["cases"] >= 100, n["cases"]
    assert n["wrong"] == 0 and n["refused"] == 0, n["not_right"]


def test_the_math_tools_take_no_required_variable(census):
    """Wolfram infers the variable from the expression; so do we (the result says which)."""
    by = {r["name"]: r for r in census["rows"]}
    for name in ("integrate_expr", "differentiate_expr", "solve_equation", "compute_limit", "series_expansion", "summation"):
        assert not any(p["name"] == "variable" and p["required"] for p in by[name]["params"]), name
