"""Build and verify the hard validation set for the router.

sigma_ground/mcp/benchmark/router_hardval.json holds hand-written questions in a
different voice from the training templates, each with its gold plan (v2 format). This
script
  * resolves the `"solar:N"` macro (N solar masses, with the library's solar mass),
  * runs every gold plan against the real MCP server (a plan that returns no value is
    reported and left out),
  * checks each question against the benchmark guard (reported, not removed: this set is
    never trained on),
  * measures the trigram similarity of each question to every training question, and
    leaves out any at 0.5 or above, so the set stays out-of-template,
  * writes hard.jsonl (chat records, split name "hard") and report.json.

    python tools/build_router_hardval.py --train local-cache/router_dataset_v2/train.jsonl \
        --out local-cache/router_hardval
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sigma_ground.field.constants import M_SUN_KG  # noqa: E402
from sigma_ground.mcp.benchmark import router_dataset as D  # noqa: E402
from sigma_ground.mcp.benchmark import router_plan as RP  # noqa: E402

SPEC = REPO / "sigma_ground" / "mcp" / "benchmark" / "router_hardval.json"
SOLAR = re.compile(r"^solar:([0-9.]+)$")
MAX_SIMILARITY = 0.5


def resolve_macros(node):
    """"solar:7.3" -> 7.3 solar masses in kg, as router_dataset writes it."""
    if isinstance(node, str):
        m = SOLAR.match(node)
        return D._clean(float(m.group(1)) * M_SUN_KG) if m else node
    if isinstance(node, list):
        return [resolve_macros(x) for x in node]
    if isinstance(node, dict):
        return {k: resolve_macros(v) for k, v in node.items()}
    return node


def to_record(item: dict, plan: dict, verified: list) -> dict:
    return {"messages": [
        {"role": "system", "content": RP.ROUTER_SYSTEM_PROMPT_V2},
        {"role": "user", "content": item["question"]},
        {"role": "assistant", "content": RP.dumps_plan(plan)}],
        "meta": {"group": item["id"], "source": "hard", "family": item["family"],
                 "steps": len(RP.canonical(plan).get("steps", [])), "verified": verified}}


async def verify(session, plan: dict, question: str):
    """-> (verified results, '') or (None, reason). Clarifications and declines run nothing."""
    c = RP.canonical(plan)
    if c is None:
        return None, "not a plan"
    if c["kind"] == "decline" or c["steps"][0]["tool"] == "request_clarification":
        return [], ""
    try:
        RP.check_refs(c["steps"])
    except RP.PlanError as e:
        return None, str(e)
    values, out = [], []
    for s in c["steps"]:
        try:
            args = RP.resolve_args(s["args"], values)
        except RP.PlanError as e:
            return None, str(e)
        res = await D.execute(session, s["tool"], args, question, timeout_s=300)
        if res is None:
            return None, f"{s['tool']} returned no value for {json.dumps(args)[:90]}"
        values.append(res["value"])
        out.append({"tool": s["tool"], "value": res["value"], "units": res.get("units")})
    return out, ""


def similarity(question: str, train_grams: list) -> float:
    g = D._trigrams(question)
    best = 0.0
    for tg in train_grams:
        inter = len(g & tg)
        if inter:
            best = max(best, inter / len(g | tg))
    return best


async def build(spec: list, train_questions: list, guard, session) -> tuple[list, dict]:
    train_grams = [D._trigrams(q) for q in train_questions]
    keep, report = [], {"failed": [], "too_similar": [], "benchmark_overlap": []}
    sims = []
    for item in spec:
        plan = resolve_macros(item["plan"])
        verified, why = await verify(session, plan, item["question"])
        if verified is None:
            report["failed"].append({"id": item["id"], "why": why})
            continue
        sim = similarity(item["question"], train_grams) if train_grams else 0.0
        sims.append(sim)
        if sim >= MAX_SIMILARITY:
            report["too_similar"].append({"id": item["id"], "similarity": round(sim, 3),
                                          "question": item["question"]})
            continue
        if guard is not None and verified:
            hit = guard.check(item["question"], verified[-1]["tool"], verified[-1]["value"])
            if hit:
                report["benchmark_overlap"].append({"id": item["id"], "why": hit})
        keep.append(to_record(item, plan, verified))
    report["similarity"] = ({"max": round(max(sims), 3), "median": round(statistics.median(sims), 3),
                             "p90": round(sorted(sims)[int(0.9 * len(sims))], 3)} if sims else None)
    return keep, report


async def _amain(args) -> int:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    ids = [s["id"] for s in spec]
    assert len(ids) == len(set(ids)), "duplicate ids in the spec"
    train_q = []
    if args.train and Path(args.train).exists():
        train_q = [json.loads(line)["messages"][1]["content"]
                   for line in Path(args.train).read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"{len(spec)} questions; comparing with {len(train_q)} training questions")
    guard = None if args.no_guard else D.BenchmarkGuard.from_repo()
    async with stdio_client(D.server_params(args.server_mem_gb)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await D.execute(session, "convert_units", {"value": 1.0, "from_units": "eV", "to_units": "J"},
                            "", timeout_s=900)          # a fresh server's first call is slow on a busy CPU
            keep, report = await build(spec, train_q, guard, session)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "hard.jsonl").open("w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report["kept"] = len(keep)
    report["by_family"] = dict(Counter(r["meta"]["family"] for r in keep))
    report["calls_per_plan"] = dict(Counter(r["meta"]["steps"] for r in keep))
    (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"kept {len(keep)} of {len(spec)}; by family {report['by_family']}; similarity {report['similarity']}")
    for k in ("failed", "too_similar", "benchmark_overlap"):
        if report[k]:
            print(f"{k} ({len(report[k])}):")
            for x in report[k]:
                print("  ", json.dumps(x, ensure_ascii=False)[:200])
    return 1 if report["failed"] else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", type=Path, default=SPEC)
    p.add_argument("--train", type=Path, default=None, help="train.jsonl to measure similarity against")
    p.add_argument("--out", type=Path, default=REPO / "local-cache" / "router_hardval")
    p.add_argument("--no-guard", action="store_true")
    p.add_argument("--server-mem-gb", type=float, default=6)
    return asyncio.run(_amain(p.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
