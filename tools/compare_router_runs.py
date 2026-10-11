"""Side-by-side numbers for benchmark runs of the chain harness and router mode.

Scores each run with the benchmark's own scorer (score.score_run against
questions.json + ground_truth.json; nothing is re-implemented) and splits out
the model path, the questions the deterministic cascade did not answer:
records carrying `router_plan` in router mode, `turns > 0` in chain mode.

    python tools/compare_router_runs.py \
        chain=sigma_ground/mcp/benchmark/results/chain_qwen3_8b_run.json \
        stock=sigma_ground/mcp/benchmark/results/router_qwen3_8b_stock_run.json \
        tuned=sigma_ground/mcp/benchmark/results/router_mentat_router_run.json
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sigma_ground.mcp.benchmark.score import score_run  # noqa: E402

BENCH = REPO / "sigma_ground" / "mcp" / "benchmark"


def on_model_path(rec: dict) -> bool:
    return "router_plan" in rec or (rec.get("turns") or 0) > 0


def load(path: Path) -> dict:
    recs = {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))}
    rows = {r.id: r for r in score_run(path, BENCH / "questions.json",
                                       BENCH / "ground_truth.json")}
    mp = [i for i in recs if on_model_path(recs[i])]
    med = lambda xs: round(statistics.median(xs), 2) if xs else None  # noqa: E731
    return {
        "recs": recs, "rows": rows, "model_path": set(mp),
        "overall": sum(r.correct for r in rows.values()), "total": len(rows),
        "mp_n": len(mp),
        "mp_correct": sum(rows[i].correct for i in mp if i in rows),
        "mp_grounded": sum(bool(recs[i].get("grounded")) for i in mp),
        "declines": sum("router_decline" in recs[i] for i in mp),
        "clarifications": sum("router_clarify" in recs[i] for i in mp),
        "router_errors": sum("router_error" in recs[i] for i in mp),
        "median_s_all": med([r.get("elapsed_s") or 0 for r in recs.values()]),
        "median_s_mp": med([recs[i].get("elapsed_s") or 0 for i in mp]),
    }


def main(argv: list[str]) -> int:
    runs = {}
    for arg in argv:
        name, _, path = arg.partition("=")
        runs[name] = load(Path(path))
    cols = ["overall", "mp_n", "mp_correct", "mp_grounded", "declines",
            "clarifications", "router_errors", "median_s_all", "median_s_mp"]
    print("| run | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for name, r in runs.items():
        cells = [f"{r['overall']}/{r['total']}"] + [str(r[c]) for c in cols[1:]]
        print(f"| {name} | " + " | ".join(cells) + " |")
    names = list(runs)
    for a in names:
        for b in names:
            if a >= b:
                continue
            if any(n.startswith("tuned") for n in runs) \
                    and (a.startswith("tuned")) == (b.startswith("tuned")):
                continue    # with a fine-tune present, list only tuned-vs-baseline pairs
            ra, rb = runs[a]["rows"], runs[b]["rows"]
            only_a = sorted(i for i in ra if ra[i].correct and i in rb and not rb[i].correct)
            only_b = sorted(i for i in rb if rb[i].correct and i in ra and not ra[i].correct)
            print(f"\nright in {a}, wrong in {b} ({len(only_a)}): {', '.join(only_a)}")
            print(f"right in {b}, wrong in {a} ({len(only_b)}): {', '.join(only_b)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
