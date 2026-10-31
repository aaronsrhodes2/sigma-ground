"""The improvement-distillation rows kept for good (tools/distill_fold.py): each pinned row's plan is run through the tools in process and must give the oracle's answer.

A maths row's answer is router_math's independent sympy check, compared with the evaluator's own same_answer; a physics row's is the tool at the spec's arguments (the tool
against its past self, at 1e-3, the distillation scorer's tolerance). This pins the tools; asking the question itself of the deployed Mentat is the live suite `pinned`."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PINNED = json.loads((REPO / "sigma_ground/mcp/benchmark/distill_pinned.json").read_text(encoding="utf-8"))["pinned"]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def session():
    return _load(REPO / "tools" / "distill_day.py", "distill_day_for_pinned").LocalSession()


def run_plan(session, expected: dict):
    """The value of the last step; a string "$k" in an argument is the value of step k (1-based)."""
    plan = expected.get("plan") or [{"tool": expected["tool"], "args": expected["args"]}]
    values: list = []
    for step in plan:
        args = {k: (values[int(v[1:]) - 1] if isinstance(v, str) and v.startswith("$") and v[1:].isdigit() else v) for k, v in step["args"].items()}
        res = asyncio.run(session.call_tool(step["tool"], args))
        out = json.loads("\n".join(p.text for p in res.content))
        assert out.get("value") is not None, f"{step['tool']}({args}) returned {out.get('error') or out.get('notes')}"
        values.append(out["value"])
    return values[-1]


def test_the_pinned_file_is_what_the_day_files_give():
    r = subprocess.run([sys.executable, "-X", "utf8", str(REPO / "tools/distill_fold.py"), "--check"], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr


def test_something_is_pinned_and_every_row_has_an_oracle():
    assert PINNED
    for p in PINNED:
        assert p["expected"].get("result") is not None and p["expected"].get("oracle") in ("agrees", "tool"), p["id"]
        assert p["reason"] in ("new_combination", "fixed_miss"), p["id"]


@pytest.mark.parametrize("pin", PINNED, ids=[p["id"] for p in PINNED])
def test_the_tools_still_give_the_oracles_answer(session, pin):
    exp = pin["expected"]
    got = run_plan(session, exp)
    if exp["oracle"] == "tool":
        g, w = float(got), float(exp["result"])
        assert math.isclose(g, w, rel_tol=1e-3, abs_tol=1e-12), f"{pin['question']}: {g} != {w}"
    else:
        sys.path.insert(0, str(REPO / "tools"))
        import eval_math_suite as E
        assert E.same_answer(exp["result"], got), f"{pin['question']}: {got} != {exp['result']}"
