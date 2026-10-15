"""The no-tool baselines side by side (misc/BASELINES.md): the same 100 held-out math questions, the router + Mentat route for comparison.

    python tools/baselines_report.py --router local-cache/eval_math_v31_4b_on_master_tools3.json --out local-cache/baselines_report.json

Per model: the strict count (`eval_math_suite.same_answer`, unchanged) recomputed from the stored answers, then the hand-read verdicts for every miss
(`misc/baselines_adjudication.json`, key "model|id"): `equivalent` is added for the lenient number, `convention` is shown on its own, and incomplete / wrong /
no_answer are never added. A miss with no verdict is listed as UNREAD. Rows that errored (a timeout, a hung server) are counted as errors, not as wrong.

Rescoring: a row that kept the model's whole reply (`text`) is re-extracted with the current extractor. Older rows have only the extracted answer; those carry
the one known extractor bug, a `\\boxed{...}` inside an ANSWER line that came out as the literal text `boxed(...)`, and are unwrapped here before scoring.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

CACHE = REPO / "local-cache"
MODELS = [("P1-30B-A3B", ["direct_p1_100.json"]),
          ("Phi-4-mini-reasoning", ["direct_phi4-mini-reasoning_100.json"]),
          ("Qwen3-4B", ["direct_qwen3_4b_100.json"]),
          ("Qwen3-8B", ["direct_qwen3_8b_100_retry.json", "direct_qwen3_8b_100.json"])]       # the retry file (hung-server rows asked again) wins


def legacy_unwrap(extracted: str | None) -> str | None:
    """'boxed(3)' -> '3': what older result files kept of an ANSWER line written as \\boxed{3}."""
    if not extracted:
        return extracted
    m = re.fullmatch(r"boxed\((.*)\)", extracted.strip(), flags=re.S)
    if not m:
        return extracted
    from eval_direct_answer import tidy_answer
    return tidy_answer(m.group(1))


def rescored(row: dict) -> tuple[bool, str | None]:
    """(right, answer) from the stored row with the current extractor."""
    from eval_direct_answer import canonical, extract_answer
    from eval_math_suite import same_answer
    ans = extract_answer(row["text"]) if row.get("text") else legacy_unwrap(row.get("extracted"))
    return bool(ans) and bool(same_answer(row["expected"], canonical(ans))), ans


def model_rows(files: list[str]) -> tuple[Path | None, dict | None]:
    for f in files:
        if (CACHE / f).exists():
            return CACHE / f, json.loads((CACHE / f).read_text(encoding="utf-8"))
    return None, None


def summarize_model(name: str, data: dict, adjud: dict) -> dict:
    rows = []
    for r in data["rows"]:
        ok, ans = rescored(r)
        adj = adjud.get(f"{name}|{r['id']}")
        rows.append({"id": r["id"], "recipe": r["recipe"], "expected": r["expected"], "answer": ans, "right": ok, "recorded_right": bool(r["answer_ok"]),
                     "error": r.get("error"), "seconds": r.get("seconds"), "tokens": r.get("tokens"), "verdict": (adj or {}).get("verdict"),
                     "note": (adj or {}).get("note")})
    scored = [r for r in rows if not r["error"]]
    misses = [r for r in scored if not r["right"]]
    verdicts = Counter(r["verdict"] or "UNREAD" for r in misses)
    secs = [r["seconds"] for r in rows if r["seconds"]]
    toks = [r["tokens"] for r in rows if r["tokens"]]
    strict = sum(r["right"] for r in rows)
    return {"model": name, "n": len(rows), "errors": sum(bool(r["error"]) for r in rows), "strict_recorded": sum(r["recorded_right"] for r in rows),
            "strict": strict, "equivalent": verdicts["equivalent"], "lenient": strict + verdicts["equivalent"], "convention": verdicts["convention"],
            "lenient_plus_convention": strict + verdicts["equivalent"] + verdicts["convention"],
            "other_misses": {k: v for k, v in verdicts.items() if k not in ("equivalent", "convention")},
            "median_seconds": round(statistics.median(secs), 1) if secs else None, "median_tokens": int(statistics.median(toks)) if toks else None,
            "retried": sum(bool(r.get("retried")) for r in data["rows"]), "rows": rows}


def paired(a: dict, b_ok: dict[str, bool], test) -> dict:
    from wa_trickle_score import mcnemar_exact
    ids = [r["id"] for r in a["rows"] if r["id"] in b_ok and not r["error"]]
    mine = {r["id"]: test(r) for r in a["rows"]}
    both = sum(mine[i] and b_ok[i] for i in ids)
    only_a = sum(mine[i] and not b_ok[i] for i in ids)
    only_b = sum((not mine[i]) and b_ok[i] for i in ids)
    return {"n": len(ids), "both": both, "only_model": only_a, "only_router": only_b, "neither": len(ids) - both - only_a - only_b,
            "p_sign_test": round(mcnemar_exact(only_a, only_b), 4)}


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--router", type=Path, help="a router result file (tools/eval_math_suite.py) to pair against, by question id")
    ap.add_argument("--adjudications", type=Path, default=REPO / "misc" / "baselines_adjudication.json")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    adjud = json.loads(a.adjudications.read_text(encoding="utf-8")) if a.adjudications.exists() else {}
    router_ok = {r["id"]: bool(r["answer_ok"]) for r in json.loads(a.router.read_text(encoding="utf-8"))["rows"]} if a.router else {}
    out, lines = [], []
    lines.append("| model | strict (recorded) | strict now | + equivalent = lenient | + convention | errors | other misses | median s | median tokens |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, files in MODELS:
        path, data = model_rows(files)
        if data is None:
            lines.append(f"| {name} | not run |  |  |  |  |  |  |  |")
            continue
        s = summarize_model(name, data, adjud)
        out.append(s)
        n = s["n"]
        lines.append(f"| {name} | {s['strict_recorded']}/{n} | {s['strict']}/{n} | {s['lenient']}/{n} | {s['lenient_plus_convention']}/{n} | {s['errors']} | "
                     f"{s['other_misses']} | {s['median_seconds']} | {s['median_tokens']} |")
    if router_ok and out:
        n = sum(1 for r in out[0]["rows"] if r["id"] in router_ok)
        lines.append(f"| v3.1 4B router + Mentat | {sum(router_ok[r['id']] for r in out[0]['rows'] if r['id'] in router_ok)}/{n} |  |  |  |  |  |  |  |")
        lines += ["", "paired against the router on the same questions (two-sided exact sign test on the disagreements):", ""]
        for s in out:
            for label, test in (("strict", lambda r: r["right"]), ("lenient", lambda r: r["right"] or r["verdict"] == "equivalent")):
                p = paired(s, router_ok, test)
                s.setdefault("paired", {})[label] = p
                lines.append(f"* {s['model']}, {label}: both {p['both']}, only the model {p['only_model']}, only the router {p['only_router']}, neither {p['neither']} "
                             f"(n={p['n']}), p = {p['p_sign_test']}")
    unread = [(s["model"], r["id"]) for s in out for r in s["rows"] if not r["right"] and not r["error"] and not r["verdict"]]
    if unread:
        lines += ["", f"UNREAD misses (no hand-read verdict yet): {unread}"]
    print("\n".join(lines))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main(sys.argv[1:]))
