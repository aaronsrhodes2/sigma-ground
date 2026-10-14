"""Score a router on the held-out math suite, end to end.

The suite (local-cache/router_math_suite.json, from `router_math --suite`) is math the model was never
trained on: wordings the training templates do not use, plus hand-written questions. For each question the
model writes a plan; this runs it through the harness's own router path (`_route_once`, structured results on,
the `trained` tweak profile: hidden tools shown, strict parameters) against a real Mentat server, and scores:

  tool    the plan calls the expected tool
  args    every expected argument is there and MEANS the same (x^2 + 1 and x**2 + 1 agree; "pi / 2" and
          "pi/2" agree; a matrix by its entries), no matter how it is spelled
  answer  the answer the harness reads equals the oracle's (numbers to 1e-6, lists as sets, expressions by
          sympy equivalence)

Extra arguments the model adds are reported, not penalised. Results are broken down by tool and by kind
(templated held-out vs hand-written wild).

    python tools/eval_math_suite.py --model mentat-router-v3-4b --ollama-url http://127.0.0.1:11435 --plan v3 \
        --suite local-cache/router_math_suite.json --out local-cache/eval_math_v3_4b.json
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import collections
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import httpx  # noqa: E402
import sympy as sp  # noqa: E402

from sigma_ground.mcp.benchmark import router_plan as RP  # noqa: E402
from sigma_ground.mcp.benchmark import run_sigma_ground as H  # noqa: E402
from sigma_ground.mcp.benchmark import tweaks as T  # noqa: E402
from sigma_ground.mcp.benchmark.math_classifier import _try_evaluate  # noqa: E402


def sym(text):
    return sp.sympify(str(text).replace("^", "**"))


def same_expr(a, b) -> bool:
    try:
        if "=" in str(a) or "=" in str(b):
            def side(t):
                l, _, r = str(t).partition("=")
                return sym(l) - sym(r or "0")
            return sp.simplify(side(a) - side(b)) == 0 or sp.simplify(side(a) + side(b)) == 0
        return sp.simplify(sym(a) - sym(b)) == 0
    except Exception:
        return str(a).replace(" ", "") == str(b).replace(" ", "")


def same_arg(key: str, want, got) -> bool:
    if want == got:
        return True
    if key in ("matrix", "matrix_a", "matrix_b", "vector_b"):
        try:
            return ast.literal_eval(str(want)) == ast.literal_eval(str(got))
        except Exception:
            return False
    if key in ("variables", "direction", "variable", "func", "t_var", "s_var", "x_var", "k_var"):
        return str(want).replace(" ", "") == str(got).replace(" ", "")
    if key in ("vector_field",):
        a, b = [x.strip() for x in str(want).split(",")], [x.strip() for x in str(got).split(",")]
        return len(a) == len(b) and all(same_expr(x, y) for x, y in zip(a, b))
    if isinstance(want, (int, float)) and not isinstance(want, bool):
        try:
            return abs(float(got) - float(want)) < 1e-9
        except (TypeError, ValueError):
            return False
    return same_expr(want, got)


def same_answer(expected, got) -> bool:
    e = _try_evaluate(expected)
    g = _try_evaluate(got) if isinstance(got, (str, list)) else got
    if isinstance(e, bool) or e is None or g is None:
        return False
    if isinstance(e, (int, float)) and isinstance(g, (int, float)):
        return math.isclose(e, g, rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(e, list) and isinstance(g, list):
        if len(e) != len(g):
            return False
        left = list(g)
        for x in e:
            hit = next((i for i, y in enumerate(left) if same_answer(x, y)), None)
            if hit is None:
                return False
            left.pop(hit)
        return True
    try:
        if str(e).startswith("[[") or str(g).startswith("[["):
            return sp.Matrix(sp.sympify(str(e))) == sp.Matrix(sp.sympify(str(g)))
        return same_expr(e, g)
    except Exception:
        return False


async def run(a) -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    suite = json.loads(a.suite.read_text(encoding="utf-8"))["items"]
    if a.limit:
        suite = suite[: a.limit]
    T.configure("trained", on=["structured_results"])           # hidden tools shown, strict parameters, symbolic results read
    params = StdioServerParameters(command=sys.executable, args=["-m", "sigma_ground.mcp.server"], env=dict(__import__("os").environ))
    rows = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = H._ollama_tools((await session.list_tools()).tools)
            real = H._build_real_params_map(tools)
            prompt = H._router_system_prompt(tools, False, plan=a.plan)
            async with httpx.AsyncClient(timeout=180) as http:
                for n, item in enumerate(suite, 1):
                    exp = item["expected"]
                    t0 = time.time()
                    try:
                        rec = await H._route_once(session, http, a.ollama_url.rstrip("/") + "/api/chat", a.model, item["question"], prompt,
                                                  tools, real, t0, "free", a.plan)
                    except Exception as e:  # noqa: BLE001
                        rec = {"router_error": f"{type(e).__name__}: {e}"}
                    plan = {}
                    try:
                        c = RP.canonical(json.loads(rec.get("router_plan") or ""))
                        plan = c["steps"][0] if c and c["kind"] == "steps" else {}
                    except ValueError:
                        pass
                    tool_ok = plan.get("tool") == exp["tool"]
                    got_args = plan.get("args", {})
                    missing = [k for k in exp["args"] if k not in got_args]
                    wrong = [k for k in exp["args"] if k in got_args and not same_arg(k, exp["args"][k], got_args[k])]
                    extra = [k for k in got_args if k not in exp["args"]]
                    args_ok = tool_ok and not missing and not wrong
                    answer_ok = bool(not rec.get("router_error") and rec.get("extracted_value") is not None
                                     and exp.get("result") is not None and same_answer(exp["result"], rec["extracted_value"]))
                    rows.append({"id": item["id"], "kind": item["kind"], "recipe": item["recipe"], "tool_expected": exp["tool"], "tool": plan.get("tool"),
                                 "tool_ok": tool_ok, "args_ok": args_ok, "answer_ok": answer_ok, "missing": missing, "wrong": wrong, "extra": extra,
                                 "error": rec.get("router_error"), "question": item["question"], "plan": rec.get("router_plan")})
                    if n % 50 == 0:
                        print(f"  {n}/{len(suite)}  tool {sum(r['tool_ok'] for r in rows)}  args {sum(r['args_ok'] for r in rows)}  answer {sum(r['answer_ok'] for r in rows)}", flush=True)
    return {"model": a.model, "plan": a.plan, "n": len(rows), "rows": rows}


def summarize(res: dict) -> dict:
    rows = res["rows"]
    def rate(rs, key):
        return f"{sum(r[key] for r in rs)}/{len(rs)}"
    out = {"model": res["model"], "n": len(rows), "tool": rate(rows, "tool_ok"), "args": rate(rows, "args_ok"), "answer": rate(rows, "answer_ok"),
           "by_kind": {}, "by_tool": {}, "errors": dict(collections.Counter((r["error"] or "").split(":")[0][:40] for r in rows if r["error"]))}
    for kind in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == kind]
        out["by_kind"][kind] = {"n": len(rs), "tool": rate(rs, "tool_ok"), "args": rate(rs, "args_ok"), "answer": rate(rs, "answer_ok")}
    for tool in sorted({r["tool_expected"] for r in rows}):
        rs = [r for r in rows if r["tool_expected"] == tool]
        out["by_tool"][tool] = {"n": len(rs), "args": rate(rs, "args_ok"), "answer": rate(rs, "answer_ok")}
    return out


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11435")
    ap.add_argument("--plan", default="v3", choices=["v2", "v3"])
    ap.add_argument("--suite", type=Path, default=REPO / "local-cache" / "router_math_suite.json")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    res = asyncio.run(run(a))
    summ = summarize(res)
    print(f"\n{summ['model']}: tool {summ['tool']}, args {summ['args']}, answer {summ['answer']} of {summ['n']}")
    for kind, v in summ["by_kind"].items():
        print(f"  {kind:18s} n={v['n']:4d} tool {v['tool']:9s} args {v['args']:9s} answer {v['answer']}")
    print("  by tool (args / answer):")
    for tool, v in summ["by_tool"].items():
        print(f"    {tool:22s} n={v['n']:3d}  {v['args']:8s} {v['answer']}")
    if summ["errors"]:
        print("  errors:", summ["errors"])
    if a.out:
        a.out.write_text(json.dumps({"summary": summ, **res}, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main(sys.argv[1:]))
