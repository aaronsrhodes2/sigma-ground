"""Mentat contract census: how good is every tool's interface, measured in process (no stdio, no network).

    python tools/mentat_contract_census.py                 # prints the headline numbers, writes misc/contract_census.json

What it measures (audit 2026-10-13, misc/MENTAT_AUDIT_2026-10-13.md):
  schema     tools, parameters, required ones, numeric ones, unit-suffixed ones, descriptions with an Args section / an example
  errors     P0: a required parameter missing -> does the error name it?   P1: a non-number where a number is expected
  units      P2: every unit-bearing float parameter takes a quantity string ("54 km/h") in its schema
  notation   the math tools on the notation table of sigma_ground/mcp/tools/test_mathparse.py: right / wrong / refused

`sigma_ground/mcp/test_contract_census.py` runs the cheap parts on every test run and holds the numbers to the budget in
misc/contract_budget.json. When a number improves, LOWER the budget in the same commit: the ratchet only turns one way.
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures as cf
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

UNIT_SUFFIX = re.compile(r"_(m|cm|mm|km|nm|um|kg|g|s|ms|k|c|hz|ev|kev|mev|gev|j|w|pa|v|a|ohm|f|h|t|deg|rad|m_s|m_s2|m3|m2|kg_m3|mol|mol_l|mol_kg|"
                         r"j_kg_k|w_m_k|gpa|mpa|n|au|ly|pc|percent|bar|atm|l|kpa|w_m2|c_m2|s_m|kg_s|m3_s|pa_s)$")
QTY = {"_m_s": "54 km/h", "_m": "150 cm", "_kg": "500 g", "_s": "2 min", "_hz": "2.4 GHz", "_k": "300 K", "_pa": "1 atm",
       "_j": "1 kJ", "_ev": "1 keV", "_w": "1 kW", "_v": "1 kV"}
SKIP_CALL = {"simulate", "run_simulation"}
SKIP_PREFIX = ("playground_",)


def build_server():
    """FastMCP instance as sigma_ground.mcp.server.main() builds it, captured instead of run."""
    from mcp.server.fastmcp import FastMCP
    captured = {}
    orig_init, orig_run = FastMCP.__init__, FastMCP.run

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        captured["server"] = self
    FastMCP.__init__, FastMCP.run = init, (lambda self, *a, **k: None)
    try:
        import numpy  # noqa: F401  (server.main() imports it before starting threads)
        from sigma_ground.mcp import server as S
        S.main()
    finally:
        FastMCP.__init__, FastMCP.run = orig_init, orig_run
    return captured["server"]


def call(server, name, args, timeout=30):
    def run():
        return asyncio.run(server.call_tool(name, args))
    with cf.ThreadPoolExecutor(1) as ex:
        fut = ex.submit(run)
        try:
            r = fut.result(timeout=timeout)
        except cf.TimeoutError:
            return {"status": "timeout"}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "error": f"{type(e).__name__}: {e}"[:600]}
    payload = r[1] if isinstance(r, tuple) and isinstance(r[1], dict) else None
    if payload is None:
        try:
            payload = json.loads((r[0] if isinstance(r, tuple) else r)[0].text)
        except Exception:  # noqa: BLE001
            payload = None
    if isinstance(payload, dict) and set(payload) == {"result"}:
        payload = payload["result"]
    return {"status": "ok", "result": payload}


def schema_census(server):
    tools = asyncio.run(server.list_tools())
    rows = []
    for t in tools:
        schema = t.inputSchema or {}
        props, req = schema.get("properties", {}), set(schema.get("required", []))
        desc = t.description or ""
        params = []
        for pn, ps in props.items():
            typ = ps.get("type") or "|".join(x.get("type", "?") for x in ps.get("anyOf", []))
            params.append({"name": pn, "type": typ, "required": pn in req, "unit_suffix": bool(UNIT_SUFFIX.search(pn.lower())),
                           "described": bool(ps.get("description")) or bool(re.search(r"\b" + re.escape(pn) + r"\b", desc))})
        rows.append({"name": t.name, "params": params, "n_required": len(req), "desc_len": len(desc),
                     "has_args_section": "Args:" in desc or "Parameters" in desc, "has_example": "e.g." in desc.lower() or "example" in desc.lower()})
    return rows


def error_probes(server, rows):
    """P0 for every tool with a required parameter: does the error name the parameter?"""
    out = {"p0_probed": 0, "p0_names_param": 0}
    for r in rows:
        req = [p["name"] for p in r["params"] if p["required"]]
        if not req or r["name"] in SKIP_CALL or r["name"].startswith(SKIP_PREFIX):
            continue
        res = call(server, r["name"], {})
        out["p0_probed"] += 1
        err = res.get("error", "") if res["status"] == "error" else json.dumps(res.get("result"), default=str)
        out["p0_names_param"] += any(p in err for p in req)
    return out


def notation_probe(server):
    """The notation table through the server: differentiate_expr with the written text. Returns counts and the wrong/refused cases."""
    import sympy as sp
    from sigma_ground.mcp.tools.test_mathparse import NOTATION, _oracle
    x = sp.Symbol("x")
    right = wrong = refused = 0
    bad = []
    for written, declared, meant in NOTATION:
        names = [n for n in declared.split(",") if n]
        if names != ["x"]:
            continue                                                  # derivatives w.r.t. x of x-only expressions
        xr = sp.Symbol("x", real=True)                                # |x|, sign: the derivative meant is the real-variable one
        want = sp.diff(_oracle(meant, declared).subs(x, xr), xr).subs(xr, x)
        res = call(server, "differentiate_expr", {"expression": written, "variable": "x"})
        val = res.get("result", {}).get("value") if res["status"] == "ok" and isinstance(res.get("result"), dict) else None
        if val is None:
            refused += 1
            bad.append((written, "refused"))
            continue
        try:
            got = sp.sympify(str(val), locals={"x": x})
            ok = sp.simplify(got - want) == 0
        except Exception:  # noqa: BLE001
            ok = False
        if ok:
            right += 1
        else:
            wrong += 1
            bad.append((written, f"got {val}, want {want}"))
    return {"cases": right + wrong + refused, "right": right, "wrong": wrong, "refused": refused, "not_right": bad[:40]}


def quantity_probe(server, rows, examples=None):
    """P2, structural: every float parameter the contract gives a unit (mcp/contract.py SPECS) must take a quantity STRING in its schema.
    (The conversions themselves are tested in sigma_ground/mcp/test_contract.py.)"""
    from sigma_ground.mcp.contract import SPECS
    total = accepting = 0
    for r in rows:
        for p in r["params"]:
            spec = SPECS.get(p["name"])
            if spec and spec.unit and ("number" in p["type"]):
                total += 1
                accepting += "string" in p["type"]
    return {"unit_params_total": total, "unit_params_accepting_strings": accepting}


def headline(rows, errors, notation, quantity):
    params = [p for r in rows for p in r["params"]]
    return {
        "tools": len(rows), "params": len(params), "required_params": sum(p["required"] for p in params),
        "numeric_params": sum(("number" in p["type"] or "integer" in p["type"]) for p in params),
        "unit_suffixed_params": sum(p["unit_suffix"] for p in params),
        "undescribed_params": sum(not p["described"] for p in params),
        "tools_without_args_section": sum(bool(r["params"]) and not r["has_args_section"] for r in rows),
        "tools_without_example": sum(not r["has_example"] for r in rows),
        "p0_errors_not_naming_param": errors["p0_probed"] - errors["p0_names_param"],
        "notation_wrong": notation["wrong"], "notation_refused": notation["refused"],
        "unit_params_not_taking_a_quantity": quantity["unit_params_total"] - quantity["unit_params_accepting_strings"],
    }


def seed_examples():
    """tool -> first verified single-step plan args from the router seeds (optional: only if local-cache has them)."""
    ex = {}
    for d in ("router_recipes", "router_recipes_v3", "router_math_v31", "router_params_v31"):
        f = REPO / "local-cache" / d / "seeds.jsonl"
        if not f.exists():
            continue
        for line in f.open(encoding="utf-8"):
            seed = json.loads(line)
            plan = seed["plan"]
            for st in (plan["steps"] if "steps" in plan else [plan])[:1]:
                args = st.get("tool_args", {})
                if "tool" in st and not any(isinstance(v, str) and v.startswith("$") for v in args.values()):
                    ex.setdefault(st["tool"], args)
    return ex


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "misc" / "contract_census.json")
    a = ap.parse_args(argv)
    t0 = time.time()
    server = build_server()
    rows = schema_census(server)
    errors = error_probes(server, rows)
    notation = notation_probe(server)
    quantity = quantity_probe(server, rows)
    h = headline(rows, errors, notation, quantity)
    a.out.write_text(json.dumps({"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "headline": h, "notation_not_right": notation["not_right"],
                                 "tools": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(h, indent=1))
    print(f"wrote {a.out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main(sys.argv[1:]))
