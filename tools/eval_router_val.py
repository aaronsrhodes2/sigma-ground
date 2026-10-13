"""Score a router model on router_dataset.py's held-out split (val.jsonl).

For every record: one Ollama /api/chat call with the record's own system and
user messages, think=false, temperature 0, and the harness's JSON-schema
`format` (run_sigma_ground._router_format over the router's tool names), the
same request run_sigma_ground --router makes. The reply is parsed with the
harness's _parse_plan and compared to the gold plan:

  tool match   same tool (a decline counts as the tool "decline")
  exact plan   tool match, the same arg keys, numbers within 1e-6 relative,
               strings and booleans equal (a decline's reason string too)

Reports both, overall and by meta.source (template / rewrite / clarify /
decline), and lists failures. Nothing here runs a tool; the gold plans were
verified against the live server when the dataset was built.

    python tools/eval_router_val.py --model mentat-router \
        --data local-cache/router_dataset --out local-cache/router/eval_val_mentat_router.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sigma_ground.mcp.benchmark.run_sigma_ground import (  # noqa: E402
    _ollama_tools, _parse_plan, _router_format)
from sigma_ground.mcp.benchmark import router_plan as RP  # noqa: E402

REL_TOL = 1e-6


async def router_tool_names() -> list[str]:
    """The names run_sigma_ground --router constrains to: the visible tools
    plus request_clarification (see _route_once)."""
    import os
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    params = StdioServerParameters(command=sys.executable,
                                   args=["-m", "sigma_ground.mcp.server"],
                                   env=dict(os.environ))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = _ollama_tools((await session.list_tools()).tools)
    names = [t["function"]["name"] for t in tools]
    if "request_clarification" not in names:
        names.append("request_clarification")
    return names


def signature(plan: dict | None) -> tuple | None:
    """What a plan calls, in order: ("decline",) or ("tool", name, ...). None if it
    is not a plan. Both spellings (`args` / `tool_args`) and one call or several."""
    c = RP.canonical(plan)
    if c is None:
        return None
    if c["kind"] == "decline":
        return ("decline",)
    return tuple(s["tool"] for s in c["steps"])


def same_value(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return True
        if not (math.isfinite(a) and math.isfinite(b)):
            return False
        return abs(a - b) <= REL_TOL * max(abs(a), abs(b))
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_value(a[k], b[k]) for k in a)
    return a == b


def compare(pred: dict | None, gold: dict) -> tuple[bool, bool, str]:
    """(same tools in the same order, exact plan, why not exact). A plan is exact
    when every call has the same argument keys and values (numbers within 1e-6
    relative, strings and "$n" references equal)."""
    sp, sg = signature(pred), signature(gold)
    if sp is None:
        return False, False, "unparseable or no tool"
    if sp != sg:
        return False, False, f"tools {'>'.join(sp)} != {'>'.join(sg)}"
    cp, cg = RP.canonical(pred), RP.canonical(gold)
    if cg["kind"] == "decline":
        ok = same_value(cp.get("reason"), cg.get("reason"))
        return True, ok, "" if ok else "decline reason differs"
    for n, (a, b) in enumerate(zip(cp["steps"], cg["steps"]), start=1):
        at = f"step {n} " if len(cg["steps"]) > 1 else ""
        if b["tool"] == "request_clarification":
            # what matters is WHICH input is asked for, not how the question is worded
            a, b = ({"variable": x["args"].get("variable")} for x in (a, b))
            a, b = {"args": a}, {"args": b}
        if set(a["args"]) != set(b["args"]):
            return True, False, (f"{at}arg keys: missing {sorted(set(b['args']) - set(a['args']))}, "
                                 f"extra {sorted(set(a['args']) - set(b['args']))}")
        bad = [k for k in b["args"] if not same_value(a["args"][k], b["args"][k])]
        if bad:
            return True, False, f"{at}values: " + ", ".join(
                f"{k}={a['args'][k]!r} vs {b['args'][k]!r}" for k in bad)
    return True, True, ""


SMOKE_Q = "What is the Schwarzschild radius of 2 solar masses?"


def trained_prompt(system: str, user: str) -> str:
    """The text the router was trained on (Qwen3 chat template, non-thinking
    generation prompt; tools/train_router.py checks it against the HF template)."""
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def smoke(a) -> int:
    """1. The harness's exact request (schema `format`, think=false): it must
    be accepted, or the harness would silently drop to plain JSON mode.
    2. Template check, unconstrained: chat with think=false, chat without
    `think`, and the trained text sent raw must see the same number of prompt
    tokens and give the same reply. (Not under the schema: the chat and raw
    endpoints compile the schema into different grammars.)"""
    import httpx
    ROUTER_SYSTEM_PROMPT = RP.system_prompt(a.plan)
    names = asyncio.run(router_tool_names())
    fmt = (RP.router_format(names, steps=(a.plan in RP.MULTISTEP)) if a.format == "toolfirst"
           else _router_format(names))
    base = a.ollama_url.rstrip("/")
    opts = {"temperature": 0.0, "num_ctx": 4096}
    msgs = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": SMOKE_Q}]

    def n_tokens(resp: dict) -> int | None:
        if a.server_log:
            hits = re.findall(r"task\.n_tokens = (\d+)",
                              a.server_log.read_text(encoding="utf-8", errors="replace"))
            if hits:
                return int(hits[-1])
        return resp.get("prompt_eval_count")

    ok = True
    with httpx.Client(timeout=300) as http:
        r = http.post(base + "/api/chat", json={
            "model": a.model, "stream": False, "think": False, "format": fmt,
            "options": opts, "messages": msgs})
        print(f"harness request status {r.status_code}")
        if r.status_code != 200:
            print(r.text[:500])
            return 1
        chat = r.json()["message"]["content"]
        print(f"harness-shaped reply: {chat}")
        free = {**opts, "num_predict": 64}
        seen = {}
        for tag, path, body in [
                ("chat think=false", "/api/chat", {"think": False, "messages": msgs}),
                ("chat, no think", "/api/chat", {"messages": msgs}),
                ("raw trained text", "/api/generate",
                 {"raw": True, "prompt": trained_prompt(ROUTER_SYSTEM_PROMPT, SMOKE_Q)})]:
            rr = http.post(base + path, json={"model": a.model, "stream": False,
                                              "options": free, **body})
            rr.raise_for_status()
            js = rr.json()
            text = (js.get("message") or {}).get("content") if "message" in js else js["response"]
            seen[tag] = (n_tokens(js), text)
            print(f"{tag:18s} prompt tokens {seen[tag][0]}: {text!r}")
    if len(set(seen.values())) != 1:
        print("TEMPLATE MISMATCH: Ollama does not render the trained prompt")
        ok = False
    else:
        print("template OK: all three render the trained prompt")
    plan = _parse_plan(chat)
    if plan is None:
        print("reply is not a JSON plan")
        return 1
    c = RP.canonical(plan)
    first = c["steps"][0] if c and c["kind"] == "steps" else {"tool": None, "args": {}}
    m = first["args"].get("mass_kg")
    good = first["tool"] == "schwarzschild_radius" and isinstance(m, (int, float)) \
        and abs(m - 3.977e30) <= 1e-3 * 3.977e30
    print("smoke plan:", "as expected" if good else "NOT the expected schwarzschild_radius(mass_kg~3.977e30)")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--smoke", action="store_true",
                   help="one harness-shaped request plus the template check, instead of val")
    p.add_argument("--server-log", type=Path, default=None,
                   help="with --smoke: the Ollama server log, to compare prompt token counts")
    p.add_argument("--data", type=Path, default=REPO / "local-cache" / "router_dataset")
    p.add_argument("--split", default="val")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--out", type=Path, default=None,
                   help="JSON with the summary and every record's verdict")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--show", type=int, default=10, help="failures to print")
    p.add_argument("--format", choices=["schema", "json", "none", "toolfirst"], default="schema",
                   help="schema: the harness's v1 JSON-schema `format` (default, what "
                        "run_sigma_ground --router sends); json: Ollama's plain JSON mode; "
                        "none: free generation; toolfirst: a schema whose keys are already "
                        "alphabetical in the training order (tool < tool_args), with the "
                        "tool-name enum (router_plan.router_format)")
    p.add_argument("--plan", choices=["v1", "v2", "v3"], default="v1",
                   help="the plan format the model was trained on; with --format toolfirst, "
                        "v2 also allows `steps`")
    p.add_argument("--num-predict", type=int, default=1024,
                   help="cap on generated tokens per reply (plans are under 150 tokens); "
                        "a reply that hits the cap is a failure, counted in the summary")
    p.add_argument("--timeout", type=float, default=120,
                   help="seconds per request; a timeout is a failure, not a crash")
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")    # questions hold Ω, µ, °; a Windows console is cp1252
    if a.smoke:
        return smoke(a)
    if a.out is None:
        p.error("--out is required")

    import httpx
    recs = [json.loads(line) for line in
            (a.data / f"{a.split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if a.limit:
        recs = recs[:a.limit]
    names = asyncio.run(router_tool_names())
    fmt = (RP.router_format(names, steps=(a.plan in RP.MULTISTEP)) if a.format == "toolfirst"
           else _router_format(names))
    url = a.ollama_url.rstrip("/") + "/api/chat"
    rows = []
    t_start = time.time()
    with httpx.Client(timeout=a.timeout) as http:
        for i, rec in enumerate(recs):
            system, user, gold_msg = rec["messages"]
            gold = json.loads(gold_msg["content"])
            body = {"model": a.model, "stream": False, "think": False,
                    "messages": [system, user],
                    "options": {"temperature": 0.0, "num_predict": a.num_predict,
                                "num_ctx": 32768 if len(system["content"]) > 6000 else 4096}}
            if a.format in ("schema", "toolfirst"):
                body["format"] = fmt
            elif a.format == "json":
                body["format"] = "json"
            t0 = time.time()
            capped = timed_out = False
            try:
                r = http.post(url, json=body)
                r.raise_for_status()
                js = r.json()
                raw = (js.get("message") or {}).get("content") or ""
                capped = js.get("done_reason") == "length"
            except httpx.TimeoutException:
                raw, timed_out = "", True
            dt = time.time() - t0
            pred = None if (timed_out or capped) else _parse_plan(raw)
            tool_ok, exact, why = compare(pred, gold)
            if timed_out:
                why = f"timed out after {a.timeout:.0f} s"
            elif capped:
                why = f"hit the {a.num_predict}-token cap"
            meta = rec.get("meta") or {}
            rows.append({"i": i, "source": meta.get("source"), "group": meta.get("group"),
                         "family": meta.get("family"),
                         "question": user["content"], "gold": gold, "raw": raw,
                         "tool_ok": tool_ok, "exact": exact, "why": why,
                         "timed_out": timed_out, "capped": capped,
                         "elapsed_s": round(dt, 3)})
            if (i + 1) % 100 == 0:
                done = rows
                print(f"[{i + 1}/{len(recs)}] tool {sum(x['tool_ok'] for x in done)}"
                      f"  exact {sum(x['exact'] for x in done)}  "
                      f"({time.time() - t_start:.0f}s)", flush=True)

    def summ(rs: list[dict]) -> dict:
        n = len(rs)
        return {"n": n,
                "tool_acc": round(sum(r["tool_ok"] for r in rs) / n, 4) if n else None,
                "exact_acc": round(sum(r["exact"] for r in rs) / n, 4) if n else None,
                "tool_correct": sum(r["tool_ok"] for r in rs),
                "exact_correct": sum(r["exact"] for r in rs)}

    by_source = defaultdict(list)
    by_family = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
        if r.get("family"):
            by_family[r["family"]].append(r)
    fails = [r for r in rows if not r["exact"]]
    summary = {
        "model": a.model, "data": str(a.data), "split": a.split, "format": a.format, "plan": a.plan,
        "num_predict": a.num_predict, "timeout_s": a.timeout,
        "timed_out": sum(r["timed_out"] for r in rows), "hit_token_cap": sum(r["capped"] for r in rows),
        "tool_names": len(names), "overall": summ(rows),
        "by_source": {k: summ(v) for k, v in sorted(by_source.items())},
        "by_family": {k: summ(v) for k, v in sorted(by_family.items())},
        "failure_kinds": Counter("wrong tool" if r["why"].startswith("tool")
                                 else r["why"].split(":")[0] for r in fails).most_common(),
        "median_latency_s": round(statistics.median(r["elapsed_s"] for r in rows), 3),
        "wall_s": round(time.time() - t_start, 1),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1,
                                ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    show = random.Random(0).sample(fails, min(a.show, len(fails)))
    for r in show:
        print(f"\n[{r['source']}] {r['question']}\n  gold: {json.dumps(r['gold'])}"
              f"\n  got:  {r['raw']}\n  why:  {r['why']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
