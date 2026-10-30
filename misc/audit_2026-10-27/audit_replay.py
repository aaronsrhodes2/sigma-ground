"""Mentat audit 2026-10-27: the deployed /ask, and a local replay of the same code with one switch changed.

Parts (run in this order, each writes OUT_DIR/<part>.json and prints a summary line):
  http_physics   the 150 narrative physics questions through the DEPLOYED POST /ask (tunnel 127.0.0.1:17350)
  replay_*       SkippyTel mentat/app.py `_ask` imported here (sigma-ground = this checkout = the container's ref), the router model and the
                 chat model on skippy-core through the tunnel (127.0.0.1:17434); tools run on this PC.
                 variant D = tweak state as deployed (registry defaults); variant S = D + structured_results on (the proposed fix).

Scorers unchanged: physics with benchmark/score.py `_score_one` (the function score_run uses), math with tools/eval_math_suite.py `same_answer`.
Mapping from an /ask result to the scorer's record (disclosed in the audit): extracted_value = result.value when it is a number or a string
(a dict or list value -> None: the asked quantity was not given as the answer), extracted_units = result.units,
answer_text = value, units, notes, clarification and error text joined (the deployed /ask returns no prose answer).
"""
import asyncio
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path("D:/Aaron/development/sigma-ground")
OUT = Path(sys.argv[1])
PARTS = sys.argv[2].split(",")
OUT.mkdir(parents=True, exist_ok=True)
BASE = "http://127.0.0.1:17350"

os.environ.update({"MENTAT_USE_LLM": "1", "OLLAMA_URL": os.environ.get("OLLAMA_URL") or "http://127.0.0.1:17434", "MENTAT_ROUTER_MODEL": os.environ.get("MENTAT_ROUTER_MODEL") or "mentat-router-v31-4b",
                   "MENTAT_INSTANCE": "home", "SIGMA_DATASETS": "D:/datasets"})
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, "D:/Aaron/development/SkippyTel/mentat")

import eval_math_suite as E  # noqa: E402
from sigma_ground.mcp.benchmark import score as SC  # noqa: E402
from sigma_ground.mcp.benchmark import tweaks as T  # noqa: E402

BENCH = REPO / "sigma_ground" / "mcp" / "benchmark"
PHYS_Q = json.loads((BENCH / "questions.json").read_text(encoding="utf-8"))
PHYS_GT = json.loads((BENCH / "ground_truth.json").read_text(encoding="utf-8"))
ADV_Q = json.loads((BENCH / "adversarial_questions.json").read_text(encoding="utf-8"))
ADV_GT = json.loads((BENCH / "adversarial_ground_truth.json").read_text(encoding="utf-8"))
MATH = json.loads((REPO / "local-cache" / "router_math_suite.json").read_text(encoding="utf-8"))["items"]


def deployed_ref():
    try:
        return json.loads(urllib.request.urlopen(BASE + "/version", timeout=10).read().decode()).get("sigma_ground_ref")
    except Exception as e:  # noqa: BLE001
        return f"unreadable: {type(e).__name__}"


def text_of(res: dict) -> str:
    parts = [str(res.get("value")), str(res.get("units") or ""), str(res.get("notes") or "")]
    c = res.get("clarification")
    if c:
        parts.append(json.dumps(c, default=str))
    e = res.get("error")
    if e:
        parts.append(json.dumps(e, default=str))
    return " ".join(parts)


def record_of(res: dict, qid: str, seconds: float) -> dict:
    v = res.get("value")
    ev = v if isinstance(v, (int, float, str, list)) and not isinstance(v, bool) else None   # a list (roots) is an answer, as live_ask.py reads it
    return {"id": qid, "system": "mentat_ask", "answer_text": text_of(res), "extracted_value": ev,
            "extracted_units": res.get("units") or "", "elapsed_s": seconds}


def score_physics(q, res, seconds):
    rec = record_of(res, q["id"], seconds)
    truth = PHYS_GT.get(q["id"]) or ADV_GT.get(q["id"])
    ok, rel, notes = SC._score_one(rec, q, truth)
    return bool(ok), notes


def score_math(item, res):
    v = res.get("value")
    exp = item["expected"].get("result")
    try:
        return bool(v is not None and exp is not None and E.same_answer(exp, v))
    except Exception:  # noqa: BLE001
        return False


def qtext(q):
    return q.get("question") or q.get("text") or q.get("prompt")


def summarize(name, rows, extra=None):
    lat = sorted(r["seconds"] for r in rows) or [0]
    routes = {}
    for r in rows:
        routes.setdefault(str(r.get("route")), [0, 0])
        routes[str(r.get("route"))][0] += r["ok"]
        routes[str(r.get("route"))][1] += 1
    s = {"part": name, "n": len(rows), "ok": sum(r["ok"] for r in rows), "p50_s": lat[len(lat) // 2], "p90_s": lat[int(len(lat) * 0.9)],
         "by_route": {k: f"{a}/{b}" for k, (a, b) in routes.items()}, **(extra or {})}
    print(json.dumps(s), flush=True)
    return s


# ---------------------------------------------------------------- deployed /ask, physics
def http_physics():
    def one(q):
        t = time.time()
        body = json.dumps({"question": qtext(q)}).encode()
        req = urllib.request.Request(BASE + "/ask", data=body, headers={"content-type": "application/json", "X-Skippy-Caller": "audit-2026-10-27"})
        retries, reply = 0, None
        while reply is None:
            try:
                reply = json.loads(urllib.request.urlopen(req, timeout=180).read().decode())
            except urllib.error.HTTPError as e:      # Mentat's own refusal body (4xx) is an answer; 502/503 during a redeploy is not
                if e.code in (502, 503, 504) and retries < 6:
                    retries += 1
                    time.sleep(10)
                    continue
                try:
                    reply = json.loads(e.read().decode())
                except ValueError:
                    reply = {"ok": False, "exc": f"HTTP {e.code}"}
            except Exception as e:  # noqa: BLE001
                if retries < 6:
                    retries += 1
                    time.sleep(10)
                    continue
                reply = {"ok": False, "exc": f"{type(e).__name__}: {e}"}
        dt = time.time() - t
        res = reply.get("result") or {"value": None, "error": reply.get("error") or reply.get("exc")}
        ok, notes = score_physics(q, res, dt)
        inp = res.get("inputs") or {}
        return {"id": q["id"], "domain": q.get("domain"), "ok": ok, "notes": notes, "seconds": round(dt, 2), "retries": retries,
                "route": inp.get("route"), "interpreter": res.get("interpreter"), "interpreted_as": res.get("interpreted_as"),
                "value": res.get("value") if not isinstance(res.get("value"), (dict, list)) else f"<{type(res.get('value')).__name__}>",
                "units": res.get("units"), "question": qtext(q)}
    v0 = deployed_ref()
    with ThreadPoolExecutor(max_workers=1) as ex:
        rows = list(ex.map(one, PHYS_Q))
    extra = {"ref_before": v0, "ref_after": deployed_ref(), "rows_retried": sum(1 for r in rows if r["retries"])}
    s = summarize("http_physics", rows, extra)
    (OUT / "http_physics.json").write_text(json.dumps({"summary": s, "rows": rows}, indent=1, default=str), encoding="utf-8")


# ---------------------------------------------------------------- local replay of the deployed code
_A = None
LAST = {}


def app():
    global _A
    if _A is None:
        import app as A
        import router_stage as RS
        orig = RS._run

        async def wrapped(*a, **k):
            LAST.clear()
            try:
                rec = await orig(*a, **k)
            except BaseException as e:  # noqa: BLE001
                LAST["exc"] = f"{type(e).__name__}: {e}"[:300]
                raise
            LAST["rec"] = rec
            return rec
        RS._run = wrapped
        _A = A
    return _A


def replay(suite, variant, limit=0):
    import os as _os
    _os.environ["MENTAT_CASCADE"] = {"CB": "before", "CA": "after", "CBm": "before"}.get(variant, "")
    _os.environ["MENTAT_CASCADE_OFF"] = "math" if variant == "CBm" else ""      # CBm: before-mode with the maths stage off
    A = app()
    import cascade_stage as CS
    CS.MODE = _os.environ["MENTAT_CASCADE"]
    CS.OFF = ("math",) if variant == "CBm" else ()
    T.configure(None)
    if variant in ("S", "C0", "CB", "CA", "CBm"):
        T.configure(None, on=["structured_results"])
    state = {n: T.on(n) for n in sorted(T.REGISTRY)}
    items = {"physics": PHYS_Q, "adversarial": ADV_Q, "math": MATH}[suite]
    if limit:
        items = items[:limit]
    only = os.environ.get("REPLAY_IDS")                       # a second pass over the rows a transport failure spoiled (merge_replay.py)
    if only:
        keep = set(only.split(","))
        items = [q for q in items if q["id"] in keep]
    rows = []
    v0 = deployed_ref()
    for i, q in enumerate(items, 1):
        question = qtext(q) if suite != "math" else q["question"]
        t = time.time()
        try:
            res = A._ask(question, None)
        except Exception as e:  # noqa: BLE001
            res = {"value": None, "error": f"{type(e).__name__}: {e}"[:300]}
        dt = time.time() - t
        rec = LAST.get("rec") or {}
        ok = score_physics(q, res, dt)[0] if suite != "math" else score_math(q, res)
        inp = res.get("inputs") or {}
        rows.append({"id": q["id"], "ok": ok, "seconds": round(dt, 2), "route": inp.get("route"), "interpreter": res.get("interpreter"),
                     "interpreted_as": res.get("interpreted_as"), "router_plan": rec.get("router_plan"), "router_error": rec.get("router_error"),
                     "router_decline": rec.get("router_decline"), "router_clarify": rec.get("router_clarify"), "router_exc": LAST.get("exc"),
                     "value": res.get("value") if not isinstance(res.get("value"), (dict, list)) else f"<{type(res.get('value')).__name__}>",
                     "tool_expected": (q.get("expected") or {}).get("tool") if suite == "math" else None, "question": question})
        if i % 25 == 0:
            (OUT / f"replay_{suite}_{variant}.partial.json").write_text(json.dumps(rows, default=str), encoding="utf-8")
    errs = {}
    for r in rows:
        k = (r["router_exc"] or r["router_error"] or "")[:50]
        if k:
            errs[k] = errs.get(k, 0) + 1
    extra = {"tweaks_on": [n for n, on in state.items() if on], "ref_before": v0, "ref_after": deployed_ref(),
             "router_errors": dict(sorted(errs.items(), key=lambda kv: -kv[1])[:8])}
    s = summarize(f"replay_{suite}_{variant}", rows, extra)
    (OUT / f"replay_{suite}_{variant}.json").write_text(json.dumps({"summary": s, "rows": rows}, indent=1, default=str), encoding="utf-8")


for part in PARTS:
    if part == "http_physics":
        http_physics()
    elif part.startswith("replay_"):
        _, suite, variant, *lim = part.split("_")
        replay(suite, variant, int(lim[0]) if lim else 0)
print("DONE", flush=True)
