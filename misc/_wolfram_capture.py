"""Background Wolfram capture for the 150-Q benchmark.

The benchmark questions are conversational; Wolfram's NL returns "No Results"
on them verbatim (confirmed). So we translate each question -> a concise
Wolfram-style query with the LOCAL qwen model, fire it at the Wolfram Alpha
HTTP API, and extract the Result-pod value. Resumable: skips ids already in
the accumulator (results/wolfram_mcp_capture.json), so the 11 hand-captured
mech_intro values are preserved.

NOTE (truth-first): auto-translation adds noise that can only LOWER Wolfram's
apparent score (a bad translation -> Wolfram answers the wrong thing). So this
is a LOWER BOUND on Wolfram's true accuracy. The hand-crafted 34-case parity
sample (wolfram_parity_cases.py, 34/34) is the clean upper-bound anchor.
"""
import json, os, re, sys, time, urllib.request
REPO = r"D:\Aaron\development\sigma-ground-mentat"
sys.path.insert(0, REPO)
from sigma_ground.mcp.benchmark import load_env_from_dev_root
load_env_from_dev_root(verbose=False)
from sigma_ground.mcp.benchmark.run_wolfram import _query_wolfram_raw

# Extract the CANONICAL value from a Wolfram Result pod: take the first line,
# the segment after the last '|' (Wolfram formats "label | value unit"), and
# pull the leading number + trailing unit. Avoids grabbing a later conversion
# line (e.g. the 'inches' restatement of a picometer wavelength).
_NUM = re.compile(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:[×x*]\s*10\^([-+]?\d+))?')


def _extract_value(primary):
    if not primary:
        return None, ""
    line1 = primary.strip().splitlines()[0]
    seg = line1.rsplit("|", 1)[-1].strip()
    m = _NUM.search(seg)
    if not m:
        return None, ""
    val = float(m.group(1))
    if m.group(2):
        val *= 10 ** int(m.group(2))
    units = seg[m.end():].strip().split("(")[0].strip().rstrip(".,;")
    return val, units

KEY = os.environ["WOLFRAM_ALPHA_APP_ID"]
HERE = os.path.join(REPO, "sigma_ground", "mcp", "benchmark")
RES = os.path.join(HERE, "results")
ACC = os.path.join(RES, "wolfram_mcp_capture.json")

_TR_PROMPT = (
    "Rewrite this physics word problem as a SHORT Wolfram Alpha query. Rules: "
    "a concise noun phrase targeting the SINGLE numeric quantity asked; keep all "
    "given numbers and units; drop narrative/names; no question mark. "
    "Output ONLY the query, nothing else.\n\nProblem: ")


def translate(question):
    body = json.dumps({
        "model": "qwen2.5:7b",
        "messages": [{"role": "user", "content": _TR_PROMPT + question}],
        "stream": False, "options": {"temperature": 0},
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=180))
    q = (r.get("message", {}).get("content") or "").strip().strip('"').strip()
    return q.splitlines()[0].strip() if q else question


def main():
    corpus = {q["id"]: q for q in json.load(open(os.path.join(HERE, "questions.json")))}
    missing = json.load(open(os.path.join(RES, "_wolfram_missing_ids.json")))
    acc = json.load(open(ACC)) if os.path.exists(ACC) else {}
    todo = [q for q in missing if q not in acc]
    print(f"capture: {len(todo)} to do, {len(acc)} already done", flush=True)
    for i, qid in enumerate(todo):
        question = corpus[qid]["question"]
        try:
            cq = translate(question)
            rows = _query_wolfram_raw(KEY, cq, 30.0)
            primary = ""
            for title, pid, text in rows:
                if (pid and "Result" in pid) or title == "Result":
                    primary = text; break
            if not primary and rows:
                primary = rows[min(1, len(rows) - 1)][2]  # fall back to 2nd pod
            val, units = _extract_value(primary)
            acc[qid] = {"id": qid, "system": "wolfram", "extracted_value": val,
                        "extracted_units": units, "source": "wolfram_qwen_api",
                        "clean_query": cq, "result_pod": primary[:200]}
            print(f"[{i+1}/{len(todo)}] {qid}: q={cq[:45]!r} -> {val} {units}", flush=True)
        except Exception as e:
            acc[qid] = {"id": qid, "system": "wolfram", "extracted_value": None,
                        "extracted_units": "", "source": "wolfram_qwen_api",
                        "error": f"{type(e).__name__}: {e}"}
            print(f"[{i+1}/{len(todo)}] {qid}: ERROR {e}", flush=True)
        json.dump(acc, open(ACC, "w"), indent=1)
        time.sleep(1.0)
    print(f"DONE. captured {len(acc)} total.", flush=True)


if __name__ == "__main__":
    main()
