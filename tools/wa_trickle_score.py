"""Score the Wolfram|Alpha free-tier trickle on the held-out math suite.

SkippyTel's wa-trickle service (services/wa-trickle) sends each suite question to Wolfram|Alpha once, verbatim, and freezes the reply. This reads
those frozen replies (the service's /export, or a saved copy), pulls Wolfram's answer out of its pods, and scores it with the router's own scorer
(`eval_math_suite.same_answer`, unchanged) against the suite's oracle. The router + Mentat route is scored on the same questions from a router
result file, so the two are compared question by question.

    python tools/wa_trickle_score.py                                           # live /export, no router column
    python tools/wa_trickle_score.py --router local-cache/eval_math_v31_4b_on_master_tools3.json \
        --adjudications misc/wa_trickle_adjudication.json --out local-cache/wa_trickle_score.json --show-misses

What counts as Wolfram's answer, decided from the pod structure only (never from whether it matches the oracle): the first pod whose title is not
an input echo ("Input", "Input interpretation") and that has text; pod ids are not used (Wolfram's "Input" id can carry the Derivative pod). Then the
text is reduced to the value: only the first line; the right side of "lhs = rhs"; no "≈ decimal" tail, "+ constant" or "+ O(x^n)" (the oracle drops
those too); "x = a | x = b", "x = 2 and y = 1" and "±" become a list; a matrix drawn with "|" becomes [[...]]. Mentat's own parser (mathparse) reads
the result; a reply it refuses is `unparsed`, which is different from `wrong`.

Outcomes per question: right / wrong / unparsed (Wolfram's text could not be read as a result) / no_result (Wolfram returned no pods) /
unscored (the suite has no oracle answer for it: `result: null`). `strict` is the machine's count. A human reading of the disagreements goes in
the adjudication file ({id: {"verdict": "equivalent|convention|misread|wrong", "note": ...}}); "equivalent" (the same answer in another form or
convention) is added for the lenient count, the others are listed but never added. Strict and lenient are always printed side by side.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

DEFAULT_EXPORT = "http://127.0.0.1:3010/export"
INPUT_TITLES = {"input", "input interpretation", "input information"}
_ASSIGN = re.compile(r"^\s*[^\W\d]\w*(?:\([^()]*\))?\s*=\s*\S")          # "x = 3", "λ_1 = 3", "y(x) = ..."
_BARE_ASSIGN = re.compile(r"^\s*[^\W\d]\w*\s*=\s*\S")                    # a symbol on the left, not a function: solutions, not "y(x) = ..."
_DECIMAL = re.compile(r"^\s*[-+]?\d+\.\d+\s*$")
_EQUALS = re.compile(r"(?<![<>=!])=(?!=)")


# ── Wolfram's pods -> an answer ────────────────────────────────────────────────────────────────────

def answer_pod(pods: dict | None) -> dict | None:
    """A pod titled '...result...' if there is one ('Result', 'Results', 'Result in 3D Cartesian coordinates'); otherwise the first pod that is not an
    echo of the input. Both need text. (The stored pods carry no `primary` flag, so the answer pod is chosen by title.)"""
    usable = [p for p in (pods or {}).get("pods", []) or []
              if (p.get("title") or "").strip().lower() not in INPUT_TITLES and any((t or "").strip() for t in p.get("plaintext") or [])]
    return next((p for p in usable if "result" in (p.get("title") or "").lower()), usable[0] if usable else None)


def _split_top(text: str) -> list[str]:
    """Split on '=' that sits outside every bracket ('sum_(i=1)^100 i = 5050' -> two parts)."""
    parts, depth, last = [], 0, 0
    for m in re.finditer(r"[()\[\]{}]|" + _EQUALS.pattern, text):
        ch = m.group(0)
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0:
            parts.append(text[last:m.start()])
            last = m.end()
    parts.append(text[last:])
    return [p.strip() for p in parts]


def item_value(text: str) -> str:
    """One line of Wolfram plaintext -> the value it states."""
    s = text.split("\n")[0].split("≈")[0].strip()
    parts = _split_top(s)
    if len(parts) > 1:
        vals = parts[1:]
        s = ([v for v in vals if not _DECIMAL.match(v)] or vals)[0]
    s = re.sub(r"\s*\+\s*constant\s*$", "", s)
    s = re.sub(r"\s*\+\s*O\(.*\)\s*$", "", s)                           # "+ O(x^5)": the oracle drops the remainder too
    return s.strip()


def matrix_text(text: str) -> str | None:
    """'(-5 | 2\\n3 | -1)' -> '[[-5, 2], [3, -1]]'."""
    lines = [l.strip() for l in text.strip().split("\n")]
    if len(lines) < 2 or not all("|" in l for l in lines) or not lines[0].startswith("(") or not lines[-1].endswith(")"):
        return None
    lines[0], lines[-1] = lines[0][1:], lines[-1][:-1]
    return "[" + ", ".join("[" + ", ".join(c.strip() for c in l.split("|")) + "]" for l in lines) + "]"


def _solution_items(subs: list[str]) -> list[str]:
    """Several 'name = value' subpods are several solutions; otherwise the first subpod is the answer and the rest are alternate forms."""
    if len(subs) > 1 and all(_ASSIGN.match(s) for s in subs):
        return subs
    first = subs[0].split("\n")[0]
    parts = re.split(r"\s+and\s+", first)
    if len(parts) > 1 and all(_ASSIGN.match(p) for p in parts):
        return parts
    return [first]


def pod_answer(pod: dict) -> str:
    """The pod as one plain-text answer: a value, or '[a, b]' for several."""
    subs = [s for s in pod.get("plaintext") or [] if s and s.strip()]
    mat = matrix_text(subs[0])
    if mat:
        return mat
    items = _solution_items(subs)
    values: list[str] = []
    for item in items:
        v = item_value(item)
        values += [v.replace("±", "+"), v.replace("±", "-")] if "±" in v else [v]
    solution_set = all(_BARE_ASSIGN.match(it) for it in items)          # "x = e^2" is the solution set {e^2}, however many members it has
    return values[0] if len(values) == 1 and not solution_set else "[" + ", ".join(values) + "]"


def extract(wolfram: dict) -> dict:
    out = {"status": wolfram.get("status"), "pod_title": None, "raw": None, "answer": None}
    if out["status"] != "ok":
        return out
    pod = answer_pod(wolfram.get("pods"))
    if pod is None:
        out["status"] = "no_answer_pod"
        return out
    out.update(pod_title=pod.get("title"), raw=pod.get("plaintext"), answer=pod_answer(pod))
    return out


def to_canonical(answer: str) -> str | None:
    """Mentat's parser reads the answer into the plain sympy spelling the shared scorer reads; None when it refuses."""
    from eval_direct_answer import canonical_item
    from sigma_ground.mcp.tools.mathparse import parse
    try:
        return canonical_item(parse(answer).expr)
    except Exception:  # noqa: BLE001
        return None


# ── scoring ────────────────────────────────────────────────────────────────────────────────────────

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact sign test on the b + c questions where the two systems disagree."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load_export(src: str) -> list[dict]:
    if src.startswith("http"):
        with urllib.request.urlopen(src, timeout=60) as r:                # noqa: S310 (the Captain's own local service)
            text = r.read().decode("utf-8")
    else:
        text = Path(src).read_text(encoding="utf-8")
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def score_rows(export: list[dict], suite: str = "math329", router: dict | None = None, adjud: dict | None = None) -> dict:
    from eval_math_suite import same_answer
    router_by_id = {r["id"]: r for r in (router or {}).get("rows", [])}
    rows = []
    for q in export:
        if q["suite"] != suite or not q.get("wolfram"):
            continue
        x = extract(q["wolfram"])
        exp = q["expected"]
        canon = to_canonical(x["answer"]) if x["answer"] is not None else None
        if exp.get("result") is None:
            auto = "unscored"
        elif x["status"] != "ok":
            auto = "no_result"
        elif canon is None:
            auto = "unparsed"
        else:
            auto = "right" if same_answer(exp["result"], canon) else "wrong"
        rr = router_by_id.get(q["id"])
        adj = (adjud or {}).get(q["id"])
        rows.append({"id": q["id"], "kind": exp.get("kind"), "tool": exp.get("tool"), "question": q["question"], "expected": exp.get("result"),
                     "wolfram_status": x["status"], "pod_title": x["pod_title"], "wolfram_raw": x["raw"], "wolfram_answer": x["answer"],
                     "wolfram_canonical": canon, "auto": auto, "adjudication": adj,
                     "wording": "wild" if exp.get("kind") == "wild" else ("templated, Python syntax" if ("**" in q["question"] or "[[" in q["question"]) else "templated, plain"),
                     "router_answer_ok": None if rr is None else bool(rr["answer_ok"]),
                     "router_tool_ok": None if rr is None else bool(rr["tool_ok"]),
                     "mentat_ask_value": ((q.get("mentat") or {}).get("result") or {}).get("value")})
    return {"suite": suite, "rows": rows, "summary": summarize(rows, router is not None)}


def lenient(r: dict) -> bool:
    return r["auto"] == "right" or (r["auto"] in ("wrong", "unparsed") and (r["adjudication"] or {}).get("verdict") == "equivalent")


def summarize(rows: list[dict], with_router: bool) -> dict:
    scored = [r for r in rows if r["auto"] != "unscored"]
    s = {"asked": len(rows), "answered": sum(r["wolfram_status"] == "ok" for r in rows), "no_result": sum(r["wolfram_status"] != "ok" for r in rows),
         "scorable": len(scored), "unscored_no_oracle": len(rows) - len(scored), "outcomes": dict(Counter(r["auto"] for r in rows)),
         "wolfram_strict": sum(r["auto"] == "right" for r in scored), "wolfram_lenient": sum(lenient(r) for r in scored),
         "adjudicated": dict(Counter((r["adjudication"] or {}).get("verdict") for r in scored if r["adjudication"])),
         "deployed_ask_gave_a_value": sum(r["mentat_ask_value"] is not None for r in rows), "by_kind": {}}
    for kind in sorted({r["kind"] for r in scored}):
        ks = [r for r in scored if r["kind"] == kind]
        s["by_kind"][kind] = {"n": len(ks), "no_result": sum(r["wolfram_status"] != "ok" for r in ks), "strict": sum(r["auto"] == "right" for r in ks),
                              "lenient": sum(lenient(r) for r in ks), "router": sum(bool(r["router_answer_ok"]) for r in ks) if with_router else None}
    s["by_wording"] = {}
    for w in sorted({r["wording"] for r in scored}):
        ws = [r for r in scored if r["wording"] == w]
        s["by_wording"][w] = {"n": len(ws), "no_result": sum(r["wolfram_status"] != "ok" for r in ws), "strict": sum(r["auto"] == "right" for r in ws),
                              "lenient": sum(lenient(r) for r in ws), "router": sum(bool(r["router_answer_ok"]) for r in ws) if with_router else None}
    if with_router:
        for label, pair in (("", [r for r in scored if r["router_answer_ok"] is not None]),
                            ("_answered_only", [r for r in scored if r["router_answer_ok"] is not None and r["wolfram_status"] == "ok"])):
            for name, test in (("strict", lambda r: r["auto"] == "right"), ("lenient", lenient)):
                both = sum(test(r) and r["router_answer_ok"] for r in pair)
                only_w = sum(test(r) and not r["router_answer_ok"] for r in pair)
                only_r = sum((not test(r)) and r["router_answer_ok"] for r in pair)
                s[f"paired_{name}{label}"] = {"n": len(pair), "both": both, "only_wolfram": only_w, "only_router": only_r,
                                              "neither": len(pair) - both - only_w - only_r, "router_total": both + only_r, "wolfram_total": both + only_w,
                                              "p_sign_test": round(mcnemar_exact(only_w, only_r), 4)}
    return s


# ── report ─────────────────────────────────────────────────────────────────────────────────────────

def report(res: dict, show_misses: bool) -> str:
    s, rows = res["summary"], res["rows"]
    L = [f"Wolfram free-tier trickle, suite {res['suite']}: {s['asked']} questions asked; Wolfram answered {s['answered']}, returned no result for {s['no_result']}.",
         f"{s['scorable']} have an oracle answer; {s['unscored_no_oracle']} do not (suite `result: null`) and are not scored.",
         f"outcomes: {s['outcomes']}",
         f"Wolfram right on {s['wolfram_strict']}/{s['scorable']} strict (the shared scorer) and {s['wolfram_lenient']}/{s['scorable']} with the hand-read 'equivalent' cases added"
         + (f" ({s['adjudicated']})" if s["adjudicated"] else "") + ".",
         f"The deployed Mentat front door (/ask) returned a value for {s['deployed_ask_gave_a_value']} of {s['asked']}.", "", "by kind (scorable):"]
    for k, v in s["by_kind"].items():
        L.append(f"  {k:18s} n={v['n']:3d}  Wolfram no-result {v['no_result']:3d}  strict {v['strict']:3d}  lenient {v['lenient']:3d}"
                 + (f"  router {v['router']:3d}" if v["router"] is not None else ""))
    L += ["", "by wording (scorable):"]
    for k, v in s["by_wording"].items():
        L.append(f"  {k:24s} n={v['n']:3d}  Wolfram no-result {v['no_result']:3d}  strict {v['strict']:3d}  lenient {v['lenient']:3d}"
                 + (f"  router {v['router']:3d}" if v["router"] is not None else ""))
    for label, what in (("", "all scorable questions"), ("_answered_only", "only the questions Wolfram answered")):
        for name in ("strict", "lenient"):
            p = s.get(f"paired_{name}{label}")
            if p:
                L += ["", f"paired, {name}, {what} (n={p['n']}): both {p['both']}, only Wolfram {p['only_wolfram']}, only router {p['only_router']}, "
                          f"neither {p['neither']}  ->  Wolfram {p['wolfram_total']}, router {p['router_total']}; sign test p = {p['p_sign_test']}"]
    if show_misses:
        L += ["", "Wolfram not right (strict), one per question:"]
        for r in rows:
            if r["auto"] in ("wrong", "unparsed", "no_result", "unscored"):
                adj = r["adjudication"]
                L.append(f"{r['id']} [{r['kind']}/{r['tool']}] {r['auto']}" + (f" -> {adj['verdict']}: {adj['note']}" if adj else ""))
                L.append(f"    Q: {r['question']}")
                L.append(f"    oracle: {r['expected']}   wolfram[{r['pod_title']}]: {r['wolfram_answer']}")
    return "\n".join(L)


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", default=DEFAULT_EXPORT, help="the trickle's /export URL, or a saved copy of it")
    ap.add_argument("--suite", default="math329")
    ap.add_argument("--router", type=Path, help="a router result file from tools/eval_math_suite.py, joined by question id")
    ap.add_argument("--adjudications", type=Path, help="hand-read verdicts for disagreements: {id: {verdict, note}}")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--show-misses", action="store_true")
    a = ap.parse_args(argv)
    router = json.loads(a.router.read_text(encoding="utf-8")) if a.router else None
    adjud = json.loads(a.adjudications.read_text(encoding="utf-8")) if a.adjudications and a.adjudications.exists() else None
    res = score_rows(load_export(a.export), a.suite, router, adjud)
    print(report(res, a.show_misses))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main(sys.argv[1:]))
