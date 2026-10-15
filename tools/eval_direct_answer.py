"""Ask a model the held-out math questions with NO tools, and score its final answer with the same scorer as the router evaluation.

This is the baseline question of the contender plan (misc/CONTENDER_PLAN.md, gate 4): how far does a model get on the math suite by itself
(P1-30B-A3B, Phi-4-mini-reasoning, stock Qwen3 thinking), against the router + Mentat route (tools/eval_math_suite.py: 194/329 answers for the
v3 4B)? The scorer is `eval_math_suite.same_answer` and the suite is the one the router was never trained on, unchanged.

    python tools/eval_direct_answer.py --model phi4-mini-reasoning --ollama-url http://127.0.0.1:11435 --out local-cache/direct_phi4.json
    python tools/eval_direct_answer.py --model p1 --ollama-url http://127.0.0.1:8081 --openai --stratified 100 --seed 7

The prompt asks for the final result on a last line "ANSWER: <expression>" (plain text, not LaTeX); the extractor also accepts \\boxed{...},
which reasoning models tend to write, and turns LaTeX into plain math before scoring. Tokens and seconds per question are recorded.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

PROMPT = ("Solve the following math problem. Think as long as you need, then give the final result on the LAST line in the form\n"
          "ANSWER: <result>\nwhere <result> is plain text math (use * for products, ^ for powers, sqrt(), pi, exp(), oo for infinity; "
          "a list of results as [a, b]; a matrix as [[a, b], [c, d]]). No LaTeX in that line.\n\nProblem: {q}")


# ── from a model's text to something the scorer can read ───────────────────────────────────────────

def _brace_group(s: str, i: int):
    """s[i] == '{' -> (inner text, index after the matching '}')."""
    depth, j = 0, i
    while j < len(s):
        depth += (s[j] == "{") - (s[j] == "}")
        j += 1
        if depth == 0:
            return s[i + 1:j - 1], j
    return s[i + 1:], len(s)


def latex_to_plain(s: str) -> str:
    """The LaTeX a reasoning model writes in a final answer -> plain math the parser reads."""
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\(left|right|displaystyle|,|;|!|quad|qquad)", " ", s)
    for _ in range(8):                                                   # nested \frac / \sqrt
        out, i, changed = [], 0, False
        while i < len(s):
            m = re.match(r"\\(?:d|t)?frac\s*\{", s[i:])
            if m:
                a, j = _brace_group(s, i + m.end() - 1)
                if j < len(s) and s[j] == "{":
                    b, j = _brace_group(s, j)
                    out.append(f"(({a})/({b}))")
                    i, changed = j, True
                    continue
            m = re.match(r"\\sqrt\s*\{", s[i:])
            if m:
                a, j = _brace_group(s, i + m.end() - 1)
                out.append(f"sqrt({a})")
                i, changed = j, True
                continue
            m = re.match(r"\\sqrt\s*\[(\d+)\]\s*\{", s[i:])
            if m:
                a, j = _brace_group(s, i + m.end() - 1)
                out.append(f"(({a})^(1/{m.group(1)}))")
                i, changed = j, True
                continue
            out.append(s[i])
            i += 1
        s = "".join(out)
        if not changed:
            break
    s = re.sub(r"\\(cdot|times)", "*", s)
    s = re.sub(r"\\(pi|infty)", lambda m: {"pi": "pi", "infty": "oo"}[m.group(1)], s)
    s = re.sub(r"\\(sin|cos|tan|exp|ln|log|arctan|arcsin|arccos|sinh|cosh|tanh)\b", r"\1", s)
    s = re.sub(r"\^\{([^{}]*)\}", r"^(\1)", s)
    s = re.sub(r"_\{([^{}]*)\}", r"_\1", s)
    s = s.replace("\\", "").replace("{", "(").replace("}", ")")
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"([(\[])\s+", r"\1", re.sub(r"\s+([)\]])", r"\1", s))


def tidy_answer(ans: str) -> str | None:
    """Plain-text answer -> the value alone: no 'x = ' or 'f(x) = ' in front, no '+ C', no closing full stop."""
    ans = re.sub(r"^\s*[A-Za-z_]\w*\s*(\([^)]*\))?\s*=\s*", "", ans)    # "x = 5" / "f(x) = ..." -> the value
    ans = re.sub(r"\s*\+\s*C\b\.?$", "", ans)                            # "+ C" on an antiderivative is not part of the result
    return ans.strip().rstrip(".") or None


def extract_answer(text: str) -> str | None:
    """The model's final answer as plain text, or None. Order: the last 'ANSWER:' line, else the last \\boxed{...}, else None."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)           # reasoning traces
    m = list(re.finditer(r"(?im)^\W*answer\s*[:=]\s*(.+?)\s*$", text))
    if m:
        ans = m[-1].group(1).strip().strip("*").strip()
        boxed_in_line = list(re.finditer(r"\\boxed\s*\{", ans))
        if boxed_in_line:                                                # "ANSWER: \boxed{3}" states 3
            ans, _ = _brace_group(ans, ans.index("{", boxed_in_line[-1].start()))
    else:
        boxed = [mm.start() for mm in re.finditer(r"\\boxed\s*\{", text)]
        if not boxed:
            return None
        ans, _ = _brace_group(text, text.index("{", boxed[-1]))
    return tidy_answer(latex_to_plain(ans) if "\\" in ans or "{" in ans else ans)


# ── asking a model ────────────────────────────────────────────────────────────────────────────────

def canonical(answer: str) -> str:
    """The answer in the plain sympy spelling the shared scorer reads (x^2, 2x, e^x (cos x + sin x) -> x**2, 2*x, ...), via Mentat's own parser.
    If the parser refuses it, the raw text goes to the scorer, which may still read it."""
    try:
        import sympy as sp
        from sigma_ground.mcp.tools.mathparse import parse
        e = parse(answer).expr
        if isinstance(e, (list, tuple)):
            return "[" + ", ".join(canonical_item(x) for x in e) + "]"
        return sp.sstr(e)
    except Exception:  # noqa: BLE001
        return answer


def canonical_item(x) -> str:
    import sympy as sp
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(canonical_item(y) for y in x) + "]"
    return sp.sstr(x)


def ask(http, url: str, model: str, question: str, openai: bool, num_predict: int, think: bool | None,
        temperature: float = 0.0, seed: int = 7, num_ctx: int = 16384) -> dict:
    prompt = PROMPT.format(q=question)
    t0 = time.time()
    if openai:                                       # llama.cpp server / any OpenAI-compatible endpoint
        r = http.post(url.rstrip("/") + "/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": prompt}],
                                                                      "max_tokens": num_predict, "temperature": temperature, "seed": seed})
        r.raise_for_status()
        j = r.json()
        return {"text": j["choices"][0]["message"].get("content") or "", "tokens": (j.get("usage") or {}).get("completion_tokens"),
                "seconds": time.time() - t0}
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict, "seed": seed, "num_ctx": num_ctx}}
    if think is not None:
        body["think"] = think
    r = http.post(url.rstrip("/") + "/api/chat", json=body)
    r.raise_for_status()
    j = r.json()
    msg = j.get("message", {})
    return {"text": (msg.get("content") or ""), "thinking": msg.get("thinking"), "tokens": j.get("eval_count"), "seconds": time.time() - t0}


def pick(items: list, n: int, seed: int) -> list:
    """A stratified sample: n questions spread over the recipes (tools), the same for every model given the same seed."""
    by = defaultdict(list)
    for it in items:
        by[it["recipe"]].append(it)
    rng = random.Random(seed)
    for v in by.values():
        rng.shuffle(v)
    out = []
    while len(out) < n and any(by.values()):
        for k in sorted(by):
            if by[k] and len(out) < n:
                out.append(by[k].pop())
    return out


def main(argv) -> int:
    import httpx
    from eval_math_suite import same_answer, summarize   # noqa: E402
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11435")
    ap.add_argument("--openai", action="store_true", help="the endpoint speaks the OpenAI chat API (llama.cpp server) instead of Ollama's")
    ap.add_argument("--suite", type=Path, default=REPO / "local-cache" / "router_math_suite.json")
    ap.add_argument("--stratified", type=int, default=0, help="a stratified sample of N questions instead of all")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-predict", type=int, default=6000)
    ap.add_argument("--think", choices=["on", "off", "default"], default="default")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy; reasoning models are usually run at 0.6 (greedy can loop)")
    ap.add_argument("--timeout", type=float, default=1800, help="seconds to wait for one answer")
    ap.add_argument("--retry-errors", type=Path, help="a result file from an earlier run of the same sample and settings: keep its answered rows, "
                    "ask again only the rows that errored (a hung server, a timeout); the rows asked again are marked `retried`")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    items = [i for i in json.loads(a.suite.read_text(encoding="utf-8"))["items"] if i["expected"].get("result") is not None]
    if a.stratified:
        items = pick(items, a.stratified, a.seed)
    if a.limit:
        items = items[: a.limit]
    think = {"on": True, "off": False, "default": None}[a.think]
    kept = {}
    if a.retry_errors:
        prior = json.loads(a.retry_errors.read_text(encoding="utf-8"))
        if (prior.get("stratified"), prior.get("seed")) != (a.stratified, a.seed):
            raise SystemExit(f"--retry-errors: {a.retry_errors} used a different sample (stratified {prior.get('stratified')}, seed {prior.get('seed')})")
        kept = {r["id"]: r for r in prior["rows"] if not r["error"]}
    rows = []
    with httpx.Client(timeout=a.timeout) as http:
        for n, it in enumerate(items, 1):
            if it["id"] in kept:                                          # answered in the earlier run: not asked again
                rows.append(kept[it["id"]])
                continue
            try:
                got = ask(http, a.ollama_url, a.model, it["question"], a.openai, a.num_predict, think, a.temperature, a.seed)
                err = None
            except Exception as e:  # noqa: BLE001
                got, err = {"text": "", "tokens": None, "seconds": None}, f"{type(e).__name__}: {e}"
            ans = extract_answer(got["text"])
            ok = bool(ans) and same_answer(it["expected"]["result"], canonical(ans))
            rows.append({"id": it["id"], "recipe": it["recipe"], "kind": it["kind"], "question": it["question"], "expected": it["expected"]["result"],
                         "extracted": ans, "answer_ok": bool(ok), "no_answer": ans is None, "tokens": got.get("tokens"), "seconds": got.get("seconds"),
                         "text": got["text"],                              # the model's whole reply, so an answer can be re-extracted without asking again
                         "error": err, "tool_expected": it["expected"]["tool"], "tool_ok": False, "args_ok": False,
                         **({"retried": True} if a.retry_errors else {})})
            if n % 20 == 0:
                print(f"  {n}/{len(items)}  right {sum(r['answer_ok'] for r in rows)}  no-answer {sum(r['no_answer'] for r in rows)}", flush=True)
    summ = summarize({"model": a.model, "rows": rows})
    secs = [r["seconds"] for r in rows if r["seconds"]]
    toks = [r["tokens"] for r in rows if r["tokens"]]
    print(f"\n{a.model}: answer {summ['answer']} of {summ['n']}; no final answer on {sum(r['no_answer'] for r in rows)}; "
          f"median {sorted(secs)[len(secs) // 2]:.0f} s, {sorted(toks)[len(toks) // 2] if toks else 0} tokens per question")
    for tool, v in summ["by_tool"].items():
        print(f"    {tool:22s} n={v['n']:3d}  {v['answer']}")
    if a.out:
        a.out.write_text(json.dumps({"summary": summ, "model": a.model, "stratified": a.stratified, "seed": a.seed, "rows": rows},
                                    indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main(sys.argv[1:]))
