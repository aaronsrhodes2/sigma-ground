"""Score the improvement-distillation suites (distill-YYYY-MM-DD) from the wa-trickle export, diagnose Mentat's misses from ITS OWN record,
and write the day files and the ledger (misc/DISTILLATION_BRANCH_PLAN.md).

    python -m sigma_ground.mcp.benchmark.distill_score                      # read the trickle at WA_TRICKLE_URL (default 127.0.0.1:3010)
    python -m sigma_ground.mcp.benchmark.distill_score --export rows.jsonl  # or a saved /export
    python -m sigma_ground.mcp.benchmark.distill_score --ledger-only        # rebuild misc/DISTILLATION_LEDGER.md from the day files

Verdicts on a row come from OUR oracle (the set's expected result: router_math's independent check for maths, the tool at the spec's
arguments for physics) compared with what Mentat answered. A miss is classified from Mentat's result alone:
  no_route     Mentat answered nothing: a decline, a clarification, no stage took the question
  bad_args     the tool was reached but refused the arguments (a contract error)
  tool_error   the tool ran and declined (unknown key, invalid input)
  wrong_tool   a value, from a tool other than the expected one, and not the expected value
  wrong_value  the expected tool, a different value
  unverified   no oracle (a wild question): not a miss, a row for a person to read
  unasked      Mentat has not been asked yet at the current ref
Wolfram's reply is recorded beside each row (wa_status, wa_agrees_with_us): a yardstick for the page, never a cause and never an expected
value. Wolfram's own text is not copied into the day files: only the status and the agreement flag.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import os
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RESULTS = HERE / "results" / "distill"
LEDGER = REPO / "misc" / "DISTILLATION_LEDGER.md"
CAUSES = ("no_route", "bad_args", "tool_error", "wrong_tool", "wrong_value")


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_export(src: str) -> list[dict]:
    if src.startswith("http"):
        with urllib.request.urlopen(src.rstrip("/") + "/export", timeout=120) as r:
            text = r.read().decode("utf-8", "replace")
    else:
        text = Path(src).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ── our verdict ────────────────────────────────────────────────────────────────────────────────────────────────────────

def _num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return None
    return None


def agrees(expected: dict, result: dict | None, same_answer) -> bool | None:
    """True/False against our oracle; None when there is no oracle or no answer to judge."""
    if expected.get("oracle") in (None, "none") or expected.get("result") is None:
        return None
    if not result or result.get("value") is None:
        return False
    got, want = result.get("value"), expected["result"]
    if expected.get("oracle") == "tool":                                   # physics: a number with units
        g, w = _num(got), _num(want)
        if g is None or w is None:
            return same_answer(want, got)
        if expected.get("units") and result.get("units") and str(result["units"]).replace(" ", "") != str(expected["units"]).replace(" ", ""):
            try:
                from sigma_ground.mcp.contract import _registry
                reg = _registry()
                g = reg.Quantity(g, str(result["units"])).to(str(expected["units"])).magnitude
            except Exception:  # noqa: BLE001 — units pint cannot relate: judged on the number alone, below
                pass
        return math.isclose(g, w, rel_tol=1e-3, abs_tol=1e-12)
    return bool(same_answer(want, got))


def cause(expected: dict, mentat: dict | None, verdict: bool | None) -> str | None:
    """Why Mentat missed, from its own record; None when it did not miss."""
    if verdict is None:
        return "unverified" if expected.get("oracle") in (None, "none") else ("unasked" if mentat is None else None)
    if verdict:
        return None
    if mentat is None:
        return "unasked"
    res = mentat.get("result") if mentat.get("ok") else None
    if not res:
        return "no_route"
    inputs = res.get("inputs") or {}
    err = res.get("error") or {}
    if res.get("value") is None:
        code = str(err.get("code") or "")
        if res.get("clarification") or inputs.get("route") in (None, "clarify", "decline") and not code:
            return "no_route"
        if code in ("missing_argument", "bad_type", "bad_quantity", "out_of_range", "invalid_arguments", "unknown_parameter"):
            return "bad_args"
        if code:
            return "tool_error"
        return "no_route"
    chain = inputs.get("chain") or []
    used = chain[-1].get("tool") if chain else None
    if expected.get("tool") and used and used != expected["tool"]:
        return "wrong_tool"
    return "wrong_value"


def wolfram_view(row: dict, expected: dict, same_answer, wa) -> tuple[str, bool | None]:
    """(status, agrees with OUR oracle or None): the yardstick, read from the frozen reply, never copied."""
    w = row.get("wolfram")
    if not w:
        return "unasked", None
    ex = wa.extract(w)
    if ex["status"] != "ok" or not ex.get("answer"):
        return "no_result", None
    if expected.get("oracle") in (None, "none") or expected.get("result") is None:
        return "answered", None
    canon = wa.to_canonical(ex["answer"])
    if canon is None:
        return "answered", None
    try:
        return "answered", bool(same_answer(expected["result"], canon))
    except Exception:  # noqa: BLE001
        return "answered", None


# ── a day ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

def score_suite(rows: list[dict], same_answer, wa) -> dict:
    out = []
    for r in rows:
        expected = r.get("expected") or {}
        mentat = r.get("mentat")
        res = (mentat or {}).get("result") if (mentat or {}).get("ok") else None
        verdict = agrees(expected, res, same_answer)
        why = cause(expected, mentat, verdict)
        wa_status, wa_ok = wolfram_view(r, expected, same_answer, wa)
        inputs = (res or {}).get("inputs") or {}
        chain = inputs.get("chain") or []
        out.append({"id": r["id"], "question": r["question"], "domain": expected.get("domain"), "kind": expected.get("kind"),
                    "expected_tool": expected.get("tool"), "oracle": expected.get("oracle"), "expected": expected.get("result"),
                    "mentat_value": (res or {}).get("value"), "mentat_units": (res or {}).get("units"), "mentat_tool": chain[-1].get("tool") if chain else None,
                    "route": inputs.get("route"), "router_miss": inputs.get("router_miss"), "error": ((res or {}).get("error") or {}).get("code") if res else None,
                    "mentat_ref": (mentat or {}).get("sigma_ground_ref"), "agrees": verdict, "cause": why,
                    "wa_status": wa_status, "wa_agrees_with_us": wa_ok})
    n = len(out)
    judged = [x for x in out if x["agrees"] is not None]
    summary = {"n": n, "judged": len(judged), "mentat_right": sum(1 for x in judged if x["agrees"]),
               "mentat_misses": Counter(x["cause"] for x in out if x["cause"] in CAUSES),
               "unverified": sum(1 for x in out if x["cause"] == "unverified"), "unasked": sum(1 for x in out if x["cause"] == "unasked"),
               "wolfram": {"answered": sum(1 for x in out if x["wa_status"] == "answered"), "no_result": sum(1 for x in out if x["wa_status"] == "no_result"),
                           "unasked": sum(1 for x in out if x["wa_status"] == "unasked"),
                           "agrees_with_us": sum(1 for x in out if x["wa_agrees_with_us"]), "disagrees_with_us": sum(1 for x in out if x["wa_agrees_with_us"] is False)},
               "mentat_ref": Counter(x["mentat_ref"] for x in out if x["mentat_ref"]).most_common(1)[0][0] if any(x["mentat_ref"] for x in out) else None,
               "by_domain": {d: {"n": sum(1 for x in out if x["domain"] == d), "right": sum(1 for x in out if x["domain"] == d and x["agrees"])}
                             for d in sorted({x["domain"] for x in out if x["domain"]})}}
    return {"summary": summary, "rows": out}


def write_days(rows: list[dict], out_dir: Path) -> list[Path]:
    same_answer = _tool("eval_math_suite").same_answer
    wa = _tool("wa_trickle_score")
    by_suite = defaultdict(list)
    for r in rows:
        if str(r.get("suite", "")).startswith("distill-"):
            by_suite[r["suite"]].append(r)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suite, srows in sorted(by_suite.items()):
        day = score_suite(srows, same_answer, wa)
        day["suite"], day["scored_at"] = suite, dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        p = out_dir / f"{suite}.json"
        p.write_text(json.dumps(day, indent=1, ensure_ascii=False, default=str), encoding="utf-8", newline="\n")
        written.append(p)
    return written


# ── the ledger ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def build_ledger(out_dir: Path, fixes_path: Path | None = None) -> str:
    days = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("distill-*.json")) if not p.name.endswith(".set.json")]
    fixes = json.loads(fixes_path.read_text(encoding="utf-8")) if fixes_path and fixes_path.is_file() else {}
    lines = ["# Distillation ledger: what Mentat missed on the daily new questions, and why",
             "",
             "Written by `sigma_ground/mcp/benchmark/distill_score.py` from the trickle's export; do not edit the tables by hand (the fix status column "
             "comes from `misc/distill_fixes.json`: cause|tool -> {status, commit}). Wolfram's column is a yardstick: the causes come from Mentat's own record.",
             "", "## Days", "",
             "| day | questions | judged | Mentat right | misses by cause | unverified | Wolfram answered / no result | Wolfram agrees / disagrees with our oracle | Mentat ref |",
             "|---|---|---|---|---|---|---|---|---|"]
    groups: dict[tuple, list] = defaultdict(list)
    for d in days:
        s = d["summary"]
        misses = ", ".join(f"{k} {v}" for k, v in sorted(s["mentat_misses"].items())) or "none"
        lines.append(f"| {d['suite'].removeprefix('distill-')} | {s['n']} | {s['judged']} | {s['mentat_right']} | {misses} | {s['unverified']} | "
                     f"{s['wolfram']['answered']} / {s['wolfram']['no_result']} | {s['wolfram']['agrees_with_us']} / {s['wolfram']['disagrees_with_us']} | "
                     f"{str(s.get('mentat_ref') or '')[:7]} |")
        for r in d["rows"]:
            if r["cause"] in CAUSES:
                groups[(r["cause"], r["expected_tool"] or "?")].append(r)
    lines += ["", "## Causes, grouped: the candidate fixes", "",
              "| cause | expected tool | rows | examples | status |", "|---|---|---|---|---|"]
    for (c, t), rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        ex = "; ".join(f"`{r['question'][:70]}`" + (f" (got {str(r['mentat_value'])[:20]} via {r['mentat_tool']})" if r["mentat_value"] is not None else
                                                     f" ({r['router_miss'] or r['error'] or r['route'] or 'no answer'})"[:60]) for r in rs[:2])
        st = fixes.get(f"{c}|{t}", {})
        lines.append(f"| {c} | {t} | {len(rs)} | {ex} | {st.get('status', 'open')}{(' ' + st['commit'][:7]) if st.get('commit') else ''} |")
    unverified = [r for d in days for r in d["rows"] if r["cause"] == "unverified"]
    if unverified:
        lines += ["", f"## Unverified ({len(unverified)}): wild questions with no oracle, for a person to read", ""]
        for r in unverified[:40]:
            lines.append(f"- `{r['question']}` -> Mentat {str(r['mentat_value'])[:30]} {r['mentat_units'] or ''} ({r['route'] or 'no route'}); Wolfram {r['wa_status']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", default=os.environ.get("WA_TRICKLE_URL", "http://127.0.0.1:3010"), help="the trickle URL or a saved export file")
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    ap.add_argument("--fixes", type=Path, default=REPO / "misc" / "distill_fixes.json")
    ap.add_argument("--ledger-only", action="store_true")
    a = ap.parse_args(argv)
    if not a.ledger_only:
        written = write_days(read_export(a.export), a.out)
        for p in written:
            s = json.loads(p.read_text(encoding="utf-8"))["summary"]
            print(f"{p.name}: {s['mentat_right']}/{s['judged']} right of {s['n']} (misses {dict(s['mentat_misses'])}, unverified {s['unverified']}, "
                  f"unasked {s['unasked']}); Wolfram answered {s['wolfram']['answered']}, agrees with us {s['wolfram']['agrees_with_us']}")
    a.ledger.write_text(build_ledger(a.out, a.fixes), encoding="utf-8", newline="\n")
    print(f"ledger -> {a.ledger}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
