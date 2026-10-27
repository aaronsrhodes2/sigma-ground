"""The LIVE Mentat, measured: the benchmark corpora through a deployed Mentat's POST /ask, written as harness run files.

Why this exists (misc/MENTAT_AUDIT_2026-10-27.md, findings 1, 2 and 13): the daily dashboard scored a stored run of the benchmark harness's own
cascade on this PC and said PASS for 13 days while the deployed Mentat answered 0 of the 329 maths questions. The product is what a user gets from
`/ask` (and the MCP `ask` on top of it); nothing else may stand in for it on the dashboard.

What it writes into `results/` (the daily job scores and shows them; `score.py` is unchanged):
  live_mentat_run.json               the 150 narrative physics questions, one harness record each
  adversarial_live_mentat_run.json   the adversarial set
  live_mentat_math.json              the 329 held-out maths questions, scored here with eval_math_suite.same_answer ({"summary", "rows"})
  live_mentat_meta.json              what was measured: the service's commit, its container start time before and after, counts, wall time
  live_history.jsonl                 one line per run (date, commit, counts), so the dashboard can say "below yesterday"

How an /ask reply becomes a harness record (the same mapping as the audit; `score.py` reads only these fields):
  extracted_value   result.value when it is a number, a string or a list (roots, a gradient, a matrix: the maths oracle compares lists as
                    sets and matrices by their entries); a dict (a body's whole record) -> None: the asked quantity was not given as the answer
  extracted_units   result.units
  answer_text       value, units, notes, the clarification sentence and any error text, joined (the deployed /ask returns no prose),
                    which is what the keyword scorer of a refusal question reads
  elapsed_s         the round trip

A live run must bracket itself: skippy-core recreates every container on each trunk push and on a repair (about a minute of 502s from the
model shim), and a run that overlaps one measures the outage. `/health` reports `since` (the container's start); it is read before and after,
rows that hit transport trouble are asked again once the service is back, and a run whose `since` still changed is marked `cut` in the meta
file and the history line, so the dashboard shows it instead of a false drop.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parents[2]
DEFAULT_URL = "http://skippy-core:7350"
SYSTEM = "live_mentat"
CALLER = "live_ask"
TRANSPORT_MARKS = ("502", "503", "504", "ConnectError", "RemoteDisconnected", "ConnectionReset", "WinError 100")


class Unreachable(Exception):
    """The service did not answer /health: nothing is measured (the daily job says so and goes on)."""


def _get(url: str, timeout: float = 15) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def service_state(base: str) -> dict:
    """{"since", "ready", "ref"}: the container's start time and the sigma-ground commit it serves."""
    try:
        h = _get(base.rstrip("/") + "/health")
    except urllib.error.HTTPError as e:                       # /health answers 503 with a body while not ready
        try:
            h = json.loads(e.read().decode())
        except ValueError:
            raise Unreachable(f"{base}/health: HTTP {e.code}") from e
    except Exception as e:  # noqa: BLE001
        raise Unreachable(f"{base}/health: {type(e).__name__}: {e}") from e
    try:
        ref = _get(base.rstrip("/") + "/version").get("sigma_ground_ref")
    except Exception:  # noqa: BLE001
        ref = None
    return {"since": h.get("since"), "ready": bool(h.get("ready")), "ref": ref}


def ask(base: str, question: str, timeout: float = 180) -> tuple[dict, str | None]:
    """(result, transport_error): Mentat's result dict (value None when it refused), or {} and the transport error's text."""
    body = json.dumps({"question": question}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/ask", data=body, headers={"content-type": "application/json", "X-Skippy-Caller": CALLER})
    try:
        reply = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
    except urllib.error.HTTPError as e:
        if e.code in (502, 503, 504):
            return {}, f"HTTP {e.code}"
        try:
            reply = json.loads(e.read().decode())                # 400/413/429: Mentat's own refusal, an answer
        except ValueError:
            return {}, f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return {}, f"{type(e).__name__}: {e}"
    result = reply.get("result")
    if not isinstance(result, dict):
        result = {"value": None, "error": reply.get("error") or reply.get("detail")}
    return result, None


def answer_text(result: dict) -> str:
    parts = [str(result.get("value")), str(result.get("units") or ""), str(result.get("notes") or "")]
    c = result.get("clarification")
    if isinstance(c, dict):
        parts.append(str(c.get("sentence") or ""))
    elif c:
        parts.append(str(c))
    e = result.get("error")
    if e:
        parts.append(json.dumps(e, default=str) if isinstance(e, dict) else str(e))
    return " ".join(p for p in parts if p)


def record(qid: str, result: dict, elapsed_s: float, transport_error: str | None, state: dict) -> dict:
    """One harness record, as score.py reads it, plus what the audit needs (route, the router's miss reason, the service's commit)."""
    v = result.get("value")
    ev = v if isinstance(v, (int, float, str, list)) and not isinstance(v, bool) else None
    inputs = result.get("inputs") if isinstance(result.get("inputs"), dict) else {}
    calls = [{"name": c.get("tool"), "args": c.get("args"), "result_text": json.dumps(c, default=str)} for c in (inputs.get("chain") or []) if isinstance(c, dict)]
    return {"id": qid, "system": SYSTEM, "model": result.get("interpreter"), "sigma_ground_ref": state.get("ref"), "mentat_since": state.get("since"),
            "answer_text": f"<ERROR transport: {transport_error}>" if transport_error else answer_text(result),
            "extracted_value": None if transport_error else ev, "extracted_units": "" if transport_error else (result.get("units") or ""),
            "elapsed_s": round(elapsed_s, 3), "tool_calls": calls, "route": inputs.get("route"), "router_miss": inputs.get("router_miss"),
            "transport_error": transport_error}


def hit_outage(rec: dict) -> bool:
    miss = json.dumps(rec.get("router_miss") or {})
    return bool(rec.get("transport_error")) or any(m in miss for m in TRANSPORT_MARKS)


def load_corpora() -> dict[str, list[dict]]:
    """{"physics": [...], "adversarial": [...], "math": [...]}: question dicts with at least "id" and "question"."""
    out = {"physics": json.loads((HERE / "questions.json").read_text(encoding="utf-8")),
           "adversarial": json.loads((HERE / "adversarial_questions.json").read_text(encoding="utf-8"))}
    math_path = REPO / "local-cache" / "router_math_suite.json"
    out["math"] = json.loads(math_path.read_text(encoding="utf-8"))["items"] if math_path.exists() else []
    return out


def run_corpus(base: str, items: list[dict], workers: int = 1, timeout: float = 180) -> tuple[list[dict], dict, dict]:
    """Ask every item; re-ask the rows that hit an outage once the service answers again. Returns (records, state_before, state_after)."""
    before = service_state(base)

    def one(q):
        t = time.time()
        result, err = ask(base, q["question"], timeout)
        return record(q["id"], result, time.time() - t, err, before)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        recs = list(ex.map(one, items))
    again = [i for i, r in enumerate(recs) if hit_outage(r)]
    if again:
        for _ in range(12):                                       # wait for the service to come back (a recreation takes about a minute)
            try:
                if service_state(base)["ready"]:
                    break
            except Unreachable:
                pass
            time.sleep(10)
        by_id = {q["id"]: q for q in items}
        for i in again:
            recs[i] = one(by_id[recs[i]["id"]])
            recs[i]["re_asked"] = True
    after = service_state(base)
    return recs, before, after


def score_math(rows: list[dict], items: list[dict]) -> dict:
    """The 329 scored with the router evaluator's own `same_answer` (unchanged): {"summary", "rows"}."""
    sys.path.insert(0, str(REPO / "tools"))
    import eval_math_suite as E  # noqa: E402
    by_id = {q["id"]: q for q in items}
    out = []
    for r in rows:
        exp = (by_id[r["id"]].get("expected") or {})
        v = r.get("extracted_value")
        try:
            ok = bool(v is not None and exp.get("result") is not None and E.same_answer(exp["result"], v))
        except Exception:  # noqa: BLE001
            ok = False
        out.append({**r, "answer_ok": ok, "tool_expected": exp.get("tool"), "kind": by_id[r["id"]].get("kind")})
    n_ok = sum(r["answer_ok"] for r in out)
    return {"summary": {"n": len(out), "answer_ok": n_ok, "answer": f"{n_ok}/{len(out)}",
                        "by_route": _by_route(out, "answer_ok")}, "rows": out}


def _by_route(rows: list[dict], ok_key: str) -> dict:
    agg: dict[str, list[int]] = {}
    for r in rows:
        a = agg.setdefault(str(r.get("route")), [0, 0])
        a[0] += bool(r.get(ok_key))
        a[1] += 1
    return {k: f"{a}/{b}" for k, (a, b) in agg.items()}


def run_all(base: str, results_dir: Path, suites=("physics", "adversarial", "math"), workers: int = 1, timeout: float = 180,
            retry_cut: bool = True) -> dict:
    """Ask the chosen corpora, write the run files and the meta file (the history line is the daily job's, after scoring). Returns the meta dict."""
    corpora = load_corpora()
    results_dir.mkdir(parents=True, exist_ok=True)
    meta = {"url": base, "system": SYSTEM, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "suites": {}}
    retried: list[str] = []
    t0 = time.time()
    first = service_state(base)
    meta["ref"] = first["ref"]
    meta["since_before"] = first["since"]
    cut = False
    for name in suites:
        items = corpora.get(name) or []
        if not items:
            meta["suites"][name] = {"n": 0, "note": "corpus not available on this machine"}
            continue
        recs, before, after = run_corpus(base, items, workers, timeout)
        suite_cut = before["since"] != after["since"] or any(hit_outage(r) for r in recs)
        if suite_cut and retry_cut:
            # the box recreates its stack on every trunk push; one more try usually lands between pushes (the first try is kept if the
            # second is cut as well: it is at least as complete)
            recs2, before2, after2 = run_corpus(base, items, workers, timeout)
            if before2["since"] == after2["since"] and not any(hit_outage(r) for r in recs2):
                recs, before, after, suite_cut = recs2, before2, after2, False
            retried.append(name)
        cut = cut or suite_cut
        if name == "math":
            scored = score_math(recs, items)
            (results_dir / "live_mentat_math.json").write_text(json.dumps(scored, indent=1, default=str), encoding="utf-8")
            meta["suites"][name] = {"n": len(recs), "ok": scored["summary"]["answer_ok"], "cut": suite_cut, "by_route": scored["summary"]["by_route"]}
        else:
            prefix = "adversarial_" if name == "adversarial" else ""
            (results_dir / f"{prefix}live_mentat_run.json").write_text(json.dumps(recs, indent=1, default=str), encoding="utf-8")
            meta["suites"][name] = {"n": len(recs), "answered": sum(r["extracted_value"] is not None for r in recs), "cut": suite_cut,
                                    "by_route": _by_route(recs, "extracted_value"), "re_asked": sum(bool(r.get("re_asked")) for r in recs)}
        meta["suites"][name]["since_before"], meta["suites"][name]["since_after"] = before["since"], after["since"]
    meta["since_after"] = service_state(base)["since"]
    meta["cut"] = cut
    meta["retried_suites"] = retried
    meta["wall_s"] = round(time.time() - t0, 1)
    (results_dir / "live_mentat_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return meta


def history_path(results_dir: Path) -> Path:
    return results_dir / "live_history.jsonl"


def append_history(results_dir: Path, meta: dict, scored_counts: dict) -> dict:
    """One line per run: {"at", "ref", "cut", "physics", "adversarial", "math"} with the SCORED counts (the daily job scores physics)."""
    line = {"at": meta.get("at"), "ref": meta.get("ref"), "cut": bool(meta.get("cut")), **scored_counts}
    with history_path(results_dir).open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    return line


def previous_history(results_dir: Path, before_at: str | None = None) -> dict | None:
    """The last history line written before `before_at` (the previous run), or None."""
    p = history_path(results_dir)
    if not p.exists():
        return None
    lines = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if before_at:
        lines = [l for l in lines if (l.get("at") or "") < before_at]
    return lines[-1] if lines else None
