"""Check the router's training plans against every tool's real signature, for ALL tools.

The benchmark has a handful of questions per tool, so a failure there says little about the
next question. This looks at the whole catalogue instead, from the data side:

    python tools/tool_census.py lint     --data local-cache/router_dataset_v3
    python tools/tool_census.py coverage --data local-cache/router_dataset_v3 [--tools solve_equation,...] [--min 10]
    python tools/tool_census.py signatures [--tools ...]
    python tools/tool_census.py shapes   --data local-cache/router_dataset_v3 [--tools ...]

lint      every step of every gold plan (train, val, and the seeds) is checked against the
          tool's function in server.py: unknown tool, unknown parameter, a REQUIRED parameter
          missing, an argument of the wrong type (a number for a `str` parameter, a string for
          a `float`). A plan the server would reject is a wrong label.
coverage  how many training examples use each parameter of each tool, in the plan, so a
          parameter the model has never been shown (an optional one such as `lower`, `upper`,
          `point`) is listed. Examples that ask a clarification instead are not counted.

shapes    runs ONE gold call per tool on a real Mentat server and classifies what comes back
          (number, numeric string, symbolic string, list, dict, no `value` at all, error) and
          whether the harness's router path can read it as an answer today. A shape the harness
          cannot read is a tool that will fail for any question, not just the benchmark's.

lint, coverage and signatures read server.py with ast and start no server; shapes starts one.
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sigma_ground.mcp.benchmark import router_plan as RP  # noqa: E402

SERVER = REPO / "sigma_ground" / "mcp" / "server.py"
REF = re.compile(r"^\$[1-9](\.\w+)?$")


def signatures() -> dict[str, list[dict]]:
    """tool -> [{name, ann, required}] for every @server.tool() function."""
    out: dict[str, list[dict]] = {}
    for node in ast.walk(ast.parse(SERVER.read_text(encoding="utf-8"))):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(ast.unparse(d).startswith("server.tool") for d in node.decorator_list):
            continue
        a = node.args
        pos = a.posonlyargs + a.args
        defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
        params = [{"name": x.arg, "ann": ast.unparse(x.annotation) if x.annotation else "",
                   "required": d is None} for x, d in zip(pos, defaults)]
        params += [{"name": x.arg, "ann": ast.unparse(x.annotation) if x.annotation else "",
                    "required": d is None} for x, d in zip(a.kwonlyargs, a.kw_defaults)]
        out[node.name] = params
    return out


def type_problem(value, ann: str) -> str | None:
    """Why `value` cannot be an argument annotated `ann` (pydantic v2 rules for JSON), or None."""
    if isinstance(value, str) and REF.match(value):
        return None                                       # "$1", "$1.mass_kg": resolved before the call
    branches = [b.strip() for b in re.split(r"\|", ann)] if ann else []
    if not branches or any(b in ("Any", "object") for b in branches):
        return None
    if value is None:
        return None if "None" in branches else "null for a required parameter"
    def ok(b: str) -> bool:
        if b == "float":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if b == "int":
            return (isinstance(value, int) and not isinstance(value, bool)) or (
                isinstance(value, float) and value.is_integer())
        if b == "str":
            return isinstance(value, str)
        if b == "bool":
            return isinstance(value, bool)
        if b.startswith("list"):
            return isinstance(value, list)
        if b.startswith("dict"):
            return isinstance(value, dict)
        return True                                       # an annotation this check does not model
    if any(ok(b) for b in branches if b != "None"):
        return None
    return f"{type(value).__name__} for `{ann}`"


def read_steps(path: Path):
    """(question, [step, ...]) for every record of a train/val jsonl or a seeds jsonl."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "messages" in r:
            q, plan = r["messages"][1]["content"], json.loads(r["messages"][-1]["content"])
        else:
            q, plan = r["question"], r["plan"]
        c = RP.canonical(plan)
        if c and c["kind"] == "steps":
            yield q, c["steps"]


def lint(files: list[Path], sigs: dict) -> dict:
    issues: collections.Counter = collections.Counter()
    examples: dict[str, list] = collections.defaultdict(list)
    by_tool: collections.Counter = collections.Counter()
    n_steps = 0
    for f in files:
        for q, steps in read_steps(f):
            for s in steps:
                tool, args = s["tool"], s["args"]
                if tool == "request_clarification":
                    continue
                n_steps += 1
                if tool not in sigs:
                    kinds = [("unknown tool", tool)]
                else:
                    sig = {p["name"]: p for p in sigs[tool]}
                    kinds = [("unknown parameter", f"{tool}.{k}") for k in args if k not in sig]
                    kinds += [("missing required", f"{tool}.{p['name']}") for p in sigs[tool]
                              if p["required"] and p["name"] not in args]
                    for k, v in args.items():
                        if k in sig and (why := type_problem(v, sig[k]["ann"])):
                            kinds.append(("wrong type", f"{tool}.{k}: {why}"))
                for kind, detail in kinds:
                    issues[kind] += 1
                    by_tool[tool] += 1
                    if len(examples[kind]) < 6 or detail not in {d for d, _ in examples[kind]}:
                        if len(examples[kind]) < 25:
                            examples[kind].append((detail, q[:70]))
    return {"steps": n_steps, "issues": dict(issues), "by_tool": by_tool, "examples": examples}


def coverage(files: list[Path], sigs: dict) -> dict:
    use: collections.Counter = collections.Counter()
    calls: collections.Counter = collections.Counter()
    for f in files:
        for _, steps in read_steps(f):
            for s in steps:
                if s["tool"] == "request_clarification":
                    continue
                calls[s["tool"]] += 1
                for k in s["args"]:
                    use[(s["tool"], k)] += 1
    return {"use": use, "calls": calls}


def classify_value(v) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        try:
            float(v)
            return "numeric string"
        except ValueError:
            return "symbolic string"
    if isinstance(v, list):
        if not v:
            return "empty list"
        kinds = {classify_value(x) for x in v}
        return "list of " + (kinds.pop() if len(kinds) == 1 else "mixed")
    if isinstance(v, dict):
        return "dict"
    return "null"


def gold_calls(files: list[Path]) -> dict[str, tuple[str, dict]]:
    """tool -> (question, args) of the first training example with only literal arguments."""
    out: dict[str, tuple[str, dict]] = {}
    for f in files:
        for q, steps in read_steps(f):
            for s in steps:
                if s["tool"] in out or s["tool"] == "request_clarification":
                    continue
                if any(isinstance(v, str) and REF.match(v) for v in s["args"].values()):
                    continue
                out[s["tool"]] = (q, s["args"])
    return out


async def census_shapes(calls: dict, want: list[str], timeout: float) -> list[dict]:
    import asyncio
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    from sigma_ground.mcp.benchmark import router_dataset as D
    from sigma_ground.mcp.benchmark import run_sigma_ground as H
    rows = []
    async with stdio_client(D.server_params(0)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for tool in want:
                if tool not in calls:
                    rows.append({"tool": tool, "shape": "no gold call in the data", "readable": None})
                    continue
                q, args = calls[tool]
                try:
                    res = await asyncio.wait_for(session.call_tool(tool, args), timeout)
                    text = "\n".join(p.text for p in (res.content or []) if getattr(p, "text", None))
                    failed = bool(getattr(res, "isError", False)) or text.startswith("Error executing tool")
                except Exception as e:  # noqa: BLE001
                    text, failed = f"<{type(e).__name__}>", True
                if failed:
                    rows.append({"tool": tool, "shape": "error", "readable": False, "detail": text[:140]})
                    continue
                try:
                    parsed = json.loads(text)
                except ValueError:
                    parsed = None
                if not isinstance(parsed, dict):
                    shape = "not a JSON object"
                elif "value" not in parsed:
                    shape = "no `value` key (" + ", ".join(list(parsed)[:4]) + ")"
                else:
                    shape = classify_value(parsed["value"])
                got = H._tool_call_value({"name": tool, "args": args, "result_text": text}, q)
                rows.append({"tool": tool, "shape": shape, "readable": got is not None, "question": q[:80],
                             "sample": text.replace("\n", " ")[:110]})
    return rows


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["lint", "coverage", "signatures", "shapes"])
    ap.add_argument("--data", type=Path, help="dataset dir (train.jsonl, val.jsonl)")
    ap.add_argument("--seeds", type=Path, action="append", default=[], help="extra seeds dir(s)")
    ap.add_argument("--tools", default="", help="comma list; default all")
    ap.add_argument("--min", type=int, default=10, help="coverage: flag parameters with fewer examples")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    sigs = signatures()
    want = [t for t in a.tools.split(",") if t] or sorted(sigs)
    if a.cmd == "signatures":
        for t in want:
            print(t + "(" + ", ".join(f"{p['name']}: {p['ann'] or '?'}" + ("" if p["required"] else " = ...")
                                      for p in sigs[t]) + ")")
        return 0
    if a.cmd == "shapes":
        import asyncio
        files = [a.data / n for n in ("train.jsonl", "val.jsonl")]
        rows = asyncio.run(census_shapes(gold_calls(files), want, 45.0))
        by_shape = collections.defaultdict(list)
        for r in rows:
            by_shape[r["shape"].split(" (")[0]].append(r)
        print(f"{len(rows)} tools; one gold call each")
        for shape, rs in sorted(by_shape.items(), key=lambda kv: -len(kv[1])):
            unread = [r["tool"] for r in rs if r["readable"] is False]
            print(f"  {len(rs):4d}  {shape:28s} the harness reads {sum(1 for r in rs if r['readable'])} of them"
                  + (f"; cannot read: {', '.join(unread[:12])}{' ...' if len(unread) > 12 else ''}" if unread else ""))
        if a.json:
            a.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        return 0
    files = [a.data / n for n in ("train.jsonl", "val.jsonl") if (a.data / n).exists()]
    files += [d / "seeds.jsonl" for d in a.seeds]
    if a.cmd == "lint":
        r = lint(files, sigs)
        print(f"checked {r['steps']} tool calls in {len(files)} files against {len(sigs)} signatures")
        print("issues:", r["issues"] or "none")
        for kind, ex in r["examples"].items():
            print(f"\n{kind}: {sum(1 for _ in ex)} distinct shown")
            for detail, q in ex[:12]:
                print(f"   {detail:60s} | {q}")
        if a.json:
            a.json.write_text(json.dumps({"steps": r["steps"], "issues": r["issues"],
                                          "by_tool": dict(r["by_tool"])}, indent=2), encoding="utf-8")
        return 1 if r["issues"] else 0
    c = coverage(files, sigs)
    rows = []
    for t in want:
        for p in sigs[t]:
            rows.append((t, p["name"], p["required"], c["use"][(t, p["name"])], c["calls"][t]))
    thin = [r for r in rows if r[3] < a.min]
    print(f"{len(rows)} parameters of {len(want)} tools; {len(thin)} have fewer than {a.min} training examples "
          f"({sum(1 for r in thin if r[3] == 0)} have none; {sum(1 for r in thin if r[2])} of the thin ones are required)")
    for t, p, req, n, calls in sorted(thin, key=lambda r: (r[3], r[0])):
        print(f"   {n:4d}  {t}.{p}{'  (required)' if req else ''}   [{calls} examples of the tool]")
    if a.json:
        a.json.write_text(json.dumps([{"tool": t, "param": p, "required": req, "examples": n, "tool_examples": calls}
                                      for t, p, req, n, calls in rows], indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main(sys.argv[1:]))
