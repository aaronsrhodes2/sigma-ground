"""Send the held-out math suite (329) through a deployed Mentat's POST /ask and score with eval_math_suite.same_answer (unchanged scorer).

    python ask_math_suite.py http://127.0.0.1:17350 OUT.json [LIMIT] [WORKERS]
"""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "D:/Aaron/development/sigma-ground")
sys.path.insert(0, "D:/Aaron/development/sigma-ground/tools")
import eval_math_suite as E  # noqa: E402

base, out_path = sys.argv[1], Path(sys.argv[2])
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
workers = int(sys.argv[4]) if len(sys.argv) > 4 else 2
suite = json.loads(Path("D:/Aaron/development/sigma-ground/local-cache/router_math_suite.json").read_text(encoding="utf-8"))["items"]
if limit:
    suite = suite[:limit]


def one(item):
    t = time.time()
    req = urllib.request.Request(base + "/ask", data=json.dumps({"question": item["question"]}).encode(),
                                 headers={"content-type": "application/json", "X-Skippy-Caller": "paces-math"})
    retries = 0
    while True:
        try:
            reply = json.loads(urllib.request.urlopen(req, timeout=180).read().decode())
            break
        except Exception as e:  # noqa: BLE001  (transport trouble: a redeploy, a dropped tunnel; an HTTP 4xx/5xx body from Mentat is an answer)
            if retries < 5:
                retries += 1
                time.sleep(8)
                continue
            reply = {"ok": False, "exc": f"{type(e).__name__}: {e}"}
            break
    dt = time.time() - t
    res = reply.get("result") or {}
    value = res.get("value")
    inputs = res.get("inputs") or {}
    exp = item["expected"]
    try:
        ok = value is not None and exp.get("result") is not None and E.same_answer(exp["result"], value)
    except Exception:  # noqa: BLE001
        ok = False
    return {"id": item["id"], "kind": item["kind"], "tool_expected": exp["tool"], "answer_ok": bool(ok), "seconds": round(dt, 2),
            "route": inputs.get("route"), "interpreter": res.get("interpreter"), "value": value if isinstance(value, (str, int, float)) else str(value)[:200],
            "tool_used": [c.get("tool") for c in (inputs.get("chain") or [])], "question": item["question"], "expected": exp.get("result"),
            "exc": reply.get("exc"), "retries": retries}


def version():
    try:
        return json.loads(urllib.request.urlopen(base + "/version", timeout=10).read().decode()).get("sigma_ground_ref")
    except Exception as e:  # noqa: BLE001
        return f"unreadable: {e}"


v0 = version()
t0 = time.time()
with ThreadPoolExecutor(max_workers=workers) as ex:
    rows = list(ex.map(one, suite))
total = time.time() - t0
v1 = version()
doc = {"base": base, "n": len(rows), "wall_s": round(total, 1), "ref_before": v0, "ref_after": v1, "rows": rows}
out_path.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
lat = sorted(r["seconds"] for r in rows)
by_route = {}
for r in rows:
    by_route.setdefault(str(r["route"]), []).append(r["answer_ok"])
print(f"n={len(rows)} answer_ok={sum(r['answer_ok'] for r in rows)} wall={total:.0f}s  latency p50={lat[len(lat)//2]:.1f}s p90={lat[int(len(lat)*0.9)]:.1f}s max={lat[-1]:.1f}s")
print("ref before/after:", v0, v1, " rows needing retries:", sum(1 for r in rows if r["retries"]), " failed after retries:", sum(1 for r in rows if r["exc"]))
print("by route:", {k: f"{sum(v)}/{len(v)}" for k, v in by_route.items()})
