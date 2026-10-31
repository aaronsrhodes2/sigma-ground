"""One day of the improvement-distillation branch: a NEW set of terse science questions, with OUR oracle where we have one, queued for the
Wolfram|Alpha trickle and for Mentat (misc/DISTILLATION_BRANCH_PLAN.md).

    python tools/distill_day.py                               # today's set, written to local-cache/distill/<date>.set.json (no network but the trickle's export)
    python tools/distill_day.py --enqueue                     # ... and queued in the trickle as suite distill-<date>
    python tools/distill_day.py --date 2026-11-01 --n 40 --rota math=20,physics=12,chemistry=5,astronomy=3
    python tools/distill_day.py --no-wild                     # skip the model-written questions (no Core chat model reachable)

Where the questions come from:
  maths      router_math's held-out generators (every maths tool, fresh numbers), re-rendered TERSELY (the wording Wolfram's free tier
             parses: "integrate x^2 sin x from 0 to 1", not "Over the interval ..."); the oracle is router_math's independent sympy check.
  physics    router_recipes (percent, convert, chain) at fresh numbers, shortest template; the oracle is the tool itself at the spec's
             arguments (labelled "tool": it checks Mentat's front door, not the physics).
  chemistry / astronomy   the atomic and astro recipes, plus questions written by Core's chat model ("wild", no oracle: unverified).
New means not in the trickle's database (normalised text) and not in the held-out suite. Wolfram's answers never come back into this
file or any prompt: they are a yardstick, read by benchmark/distill_score.py beside OUR oracle.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sigma_ground.mcp.benchmark import router_dataset as D  # noqa: E402
from sigma_ground.mcp.benchmark import router_math as RM  # noqa: E402
from sigma_ground.mcp.benchmark import router_recipes as RR  # noqa: E402

DEFAULT_ROTA = {"math": 20, "physics": 12, "chemistry": 5, "astronomy": 3}
FAMILY_DOMAIN = {"percent": "physics", "convert": "physics", "chain": "physics", "atomic": "chemistry", "astro": "astronomy"}
WILD_MODEL = os.environ.get("DISTILL_WILD_MODEL", "chat")
WILD_PROMPT = ("Write {n} short science questions in the field of {domain}, one per line, no numbering, each a single quantity to compute or "
               "look up, in the terse style a search engine accepts (for example 'boiling point of ethanol', 'molar mass of CaCO3', "
               "'escape velocity of Mars', 'distance to Proxima Centauri in light years'). Use standard notation and SI or common units, "
               "never programming syntax. Each question must have one definite answer. Do not number them, do not add anything else.")


# ── a tool session in this process (no subprocess, no network) ──────────────────────────────────────────────────────────

class _Res:
    def __init__(self, text: str):
        self.content = [type("T", (), {"text": text})()]
        self.isError = False


class LocalSession:
    """Looks like an MCP ClientSession to router_dataset.execute, but calls the server built in this process."""

    def __init__(self):
        spec = importlib.util.spec_from_file_location("mentat_contract_census", REPO / "tools" / "mentat_contract_census.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["mentat_contract_census"] = mod
        spec.loader.exec_module(mod)
        self.server = mod.build_server()

    async def call_tool(self, name: str, args: dict):
        out = await self.server.call_tool(name, args)
        if isinstance(out, tuple) and len(out) == 2:
            content = out[0]
            text = "\n".join(getattr(p, "text", "") for p in content)
        else:
            text = json.dumps(out, default=str)
        return _Res(text)


# ── terse wording for the maths tools (what Wolfram's free tier parses) ────────────────────────────────────────────────

def _m(expr) -> str:
    s = str(expr).replace("**", "^")
    return re.sub(r"\s+", " ", s).strip()


def _bound(b) -> str:
    return {"oo": "infinity", "-oo": "-infinity", "+oo": "infinity"}.get(str(b).strip(), str(b))


def terse_math(tool: str, args: dict) -> str | None:
    """A terse question for a maths tool call, or None when this file has no wording for the tool."""
    a = {k: v for k, v in (args or {}).items() if v is not None}
    e = _m(a.get("expression", a.get("scalar_field", a.get("vector_field", a.get("equation", "")))))
    v = a.get("variable") or a.get("t_var") or "x"
    if tool == "differentiate_expr":
        order = int(a.get("order", 1) or 1)
        return f"{'second ' if order == 2 else 'third ' if order == 3 else ''}derivative of {e} with respect to {v}" if v != "x" or order > 1 else f"derivative of {e}"
    if tool == "integrate_expr":
        if "lower" in a and "upper" in a:
            return f"integrate {e} from {_bound(a['lower'])} to {_bound(a['upper'])}" + (f" with respect to {v}" if v != "x" else "")
        return f"integral of {e} d{v}"
    if tool == "solve_equation":
        eq = e if "=" in e else f"{e} = 0"
        return f"solve {eq} for {a.get('variable', 'x')}"
    if tool == "simplify_expr":
        return f"simplify {e}"
    if tool == "factor_expression":
        return f"factor {e}"
    if tool == "expand_expression":
        return f"expand {e}"
    if tool == "compute_limit":
        return f"limit of {e} as {v} -> {_bound(a.get('point', 0))}"
    if tool == "series_expansion":
        n = a.get("order", a.get("n", 6))
        return f"series expansion of {e} at {v} = {_bound(a.get('point', 0))} to order {n}"
    if tool == "summation":
        return f"sum of {e} for {v} from {_bound(a.get('lower', 1))} to {_bound(a.get('upper', 'oo'))}"
    if tool == "laplace_transform":
        return f"Laplace transform of {e}" + (f" from {a['t_var']} to {a['s_var']}" if a.get("t_var") not in (None, "t") or a.get("s_var") not in (None, "s") else "")
    if tool == "fourier_transform":
        return f"Fourier transform of {e}"
    if tool == "gradient":
        return f"gradient of {e}" + (f" in {a['variables']}" if a.get("variables") else "")
    if tool == "divergence":
        return f"divergence of ({e})"
    if tool == "curl":
        return f"curl of ({e})"
    if tool == "matrix_determinant":
        return f"determinant of {a.get('matrix')}"
    if tool == "matrix_eigenvalues":
        return f"eigenvalues of {a.get('matrix')}"
    if tool == "matrix_inverse":
        return f"inverse of {a.get('matrix')}"
    if tool == "matrix_multiply":
        return f"{a.get('matrix_a')} * {a.get('matrix_b')}"
    if tool == "solve_linear_system":
        return f"solve {a.get('matrix_a')} x = {a.get('vector_b')}"
    if tool == "solve_ode":
        eq = e if "=" in e else f"{e} = 0"
        f, var = a.get("func", "y"), a.get("variable", "x")
        # day one (2026-10-30): the oracle solved in t and the question did not say so, so Mentat's y(x) was marked wrong; the
        # question now names the function and its variable whenever the variable is not x
        return f"solve {eq}" + (f" for {f}({var})" if var != "x" else "")
    return None


# ── what is already asked (novelty) ────────────────────────────────────────────────────────────────────────────────────

def _get_lines(url: str, path: str) -> list[str]:
    with urllib.request.urlopen(url.rstrip("/") + path, timeout=60) as r:
        return r.read().decode("utf-8", "replace").splitlines()


def known_questions(trickle_url: str | None) -> set[str]:
    seen: set[str] = set()
    for p in (REPO / "local-cache" / "router_math_suite.json",):
        if p.is_file():
            for it in json.loads(p.read_text(encoding="utf-8")).get("items", []):
                seen.add(D._norm(it["question"]))
    if trickle_url:
        try:
            for line in _get_lines(trickle_url, "/export"):
                line = line.strip()
                if line:
                    seen.add(D._norm(json.loads(line)["question"]))
        except Exception as e:  # noqa: BLE001 — no trickle reachable: novelty against the local suite only, and say so
            print(f"  (trickle export not read: {type(e).__name__}: {str(e)[:80]}; novelty checked against the local suite only)", file=sys.stderr)
    return seen


# ── the three makers ───────────────────────────────────────────────────────────────────────────────────────────────────

async def make_math(session, n: int, seed: int, forbidden: set[str]) -> list[dict]:
    seeds, _report = await RM.generate(session, per=max(2, n // 6), seed=seed, guard=None, holdout=True, forbidden=set(forbidden))
    rng = random.Random(f"distill-math:{seed}")
    rng.shuffle(seeds)
    out = []
    for s in seeds:
        exp = s.get("expected") or {}
        q = terse_math(exp.get("tool", ""), exp.get("args") or {})
        if not q or D._norm(q) in forbidden:
            continue
        forbidden.add(D._norm(q))
        out.append({"question": q, "question_long": s["question"], "domain": "math", "kind": "spec",
                    "expected": {"tool": exp.get("tool"), "args": exp.get("args"), "result": exp.get("result"), "spec": exp.get("spec"),
                                 "oracle": s["meta"]["oracle"]}})      # agrees | unchecked (a DISAGREES was dropped by the generator)
        if len(out) >= n:
            break
    return out


async def make_recipes(session, n: int, seed: int, forbidden: set[str], domains: set[str]) -> list[dict]:
    rng = random.Random(f"distill-recipes:{seed}")
    recipes = [r for r in RR.RECIPES if FAMILY_DOMAIN.get(r.family) in domains]
    out, attempts = [], 0
    while len(out) < n and attempts < n * 30 and recipes:
        attempts += 1
        r = rng.choice(recipes)
        args = {name: RR._sample_slot(name, spec, rng) for name, spec in r.slots.items()}
        template = min(r.templates, key=len)                       # the shortest wording: closest to what Wolfram parses
        q = RR.render(r, args, template)
        if D._norm(q) in forbidden:
            continue
        steps = RR.build_plan(r, args)
        results, why = await RR.run_plan(session, steps, q, r.bounds)
        if not results:
            continue
        last = results[-1]
        forbidden.add(D._norm(q))
        out.append({"question": q, "domain": FAMILY_DOMAIN[r.family], "kind": "spec",
                    "expected": {"tool": last["tool"], "args": last["args"], "result": last["value"], "units": last.get("units"),
                                 "plan": steps, "recipe": r.id, "oracle": "tool"}})
    return out


def make_wild(n: int, domain: str, seed: int, forbidden: set[str], ollama_url: str, model: str) -> tuple[list[dict], dict]:
    """Questions written by Core's chat model. No oracle (unverified). Returns (items, record of the call)."""
    body = {"model": model, "stream": False, "think": False, "options": {"temperature": 0.9, "seed": seed},
            "messages": [{"role": "user", "content": WILD_PROMPT.format(n=n + 3, domain=domain)}]}
    rec = {"model": model, "url": ollama_url, "prompt_sha256": hashlib.sha256(WILD_PROMPT.encode()).hexdigest()[:16], "seed": seed, "domain": domain}
    try:
        req = urllib.request.Request(ollama_url.rstrip("/") + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            text = (json.loads(r.read().decode("utf-8", "replace")).get("message") or {}).get("content") or ""
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        return [], rec
    out = []
    for line in text.splitlines():
        q = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"').rstrip("?").strip()
        if len(q) < 8 or len(q) > 140 or any(ch in q for ch in "`{}[]**") or D._norm(q) in forbidden:
            continue
        forbidden.add(D._norm(q))
        out.append({"question": q, "domain": domain, "kind": "wild", "expected": {"tool": None, "args": None, "result": None, "oracle": "none"}})
        if len(out) >= n:
            break
    rec["kept"] = len(out)
    return out, rec


# ── the day ────────────────────────────────────────────────────────────────────────────────────────────────────────────

def parse_rota(text: str | None) -> dict[str, int]:
    if not text:
        return dict(DEFAULT_ROTA)
    rota = {}
    for part in text.split(","):
        k, v = part.split("=")
        rota[k.strip()] = int(v)
    return rota


async def build_day(date: str, rota: dict[str, int], trickle_url: str | None, ollama_url: str | None, wild_model: str, session=None) -> dict:
    seed = int(hashlib.sha256(date.encode()).hexdigest()[:8], 16)
    forbidden = known_questions(trickle_url)
    session = session or LocalSession()
    items: list[dict] = []
    record: dict = {"date": date, "seed": seed, "rota": rota, "known_before": len(forbidden), "wild": []}
    if rota.get("math"):
        items += await make_math(session, rota["math"], seed, forbidden)
    if rota.get("physics"):
        items += await make_recipes(session, rota["physics"], seed, forbidden, {"physics"})
    for domain in ("chemistry", "astronomy"):
        want = rota.get(domain, 0)
        if not want:
            continue
        got = await make_recipes(session, max(1, want // 2), seed + 1, forbidden, {domain})
        items += got
        rest = want - len(got)
        if rest > 0 and ollama_url:
            wild, rec = make_wild(rest, domain, seed, forbidden, ollama_url, wild_model)
            items += wild
            record["wild"].append(rec)
    for i, it in enumerate(items):
        it["id"] = f"distill-{date}-{i:03d}"
    record["made"] = {d: sum(1 for it in items if it["domain"] == d) for d in rota}
    record["oracles"] = {o: sum(1 for it in items if it["expected"]["oracle"] == o) for o in ("agrees", "unchecked", "tool", "none")}
    return {"suite": f"distill-{date}", "generator": record, "items": items}


def enqueue(trickle_url: str, day: dict) -> dict:
    items = [{"id": it["id"], "question": it["question"], "priority": 1000 - i, "expected": {**it["expected"], "kind": it["kind"], "domain": it["domain"]}}
             for i, it in enumerate(day["items"])]
    req = urllib.request.Request(trickle_url.rstrip("/") + "/enqueue", data=json.dumps({"suite": day["suite"], "items": items}, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--n", type=int, default=None, help="total questions; scales the default rota")
    ap.add_argument("--rota", default=None, help="domain=count,... (default math=20,physics=12,chemistry=5,astronomy=3)")
    ap.add_argument("--url", default=os.environ.get("WA_TRICKLE_URL", "http://127.0.0.1:3010"), help="the wa-trickle service")
    ap.add_argument("--no-trickle", action="store_true", help="do not read the trickle's export for novelty")
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:17434"), help="Core's chat model for the wild questions")
    ap.add_argument("--no-wild", action="store_true")
    ap.add_argument("--enqueue", action="store_true", help="queue the set in the trickle (default: write the set file only)")
    ap.add_argument("--out", type=Path, default=REPO / "sigma_ground" / "mcp" / "benchmark" / "results" / "distill",
                    help="where the day's set is written (committed beside the scored days: the set is the record of what was asked)")
    a = ap.parse_args(argv)
    rota = parse_rota(a.rota)
    if a.n:
        total = sum(rota.values())
        rota = {k: max(1, round(v * a.n / total)) for k, v in rota.items()}
    a.out.mkdir(parents=True, exist_ok=True)
    existing = a.out / f"distill-{a.date}.set.json"
    if existing.is_file():
        # A day is made once. On 2026-10-30 a second run re-made a different set under the same ids: the trickle ignored the duplicate
        # ids (one question per id, ever) while this file was overwritten, so the record no longer matched what was asked.
        print(f"distill-{a.date} already made ({existing}); nothing to do. Use another --date for another set.")
        return 0
    day = asyncio.run(build_day(a.date, rota, None if a.no_trickle else a.url, None if a.no_wild else a.ollama_url, WILD_MODEL))
    path = existing
    path.write_text(json.dumps(day, indent=1, ensure_ascii=False, default=str), encoding="utf-8", newline="\n")
    g = day["generator"]
    print(f"{day['suite']}: {len(day['items'])} questions {g['made']}; oracles {g['oracles']}; known before: {g['known_before']} -> {path}")
    for it in day["items"][:8]:
        print(f"  [{it['domain']:9s} {it['expected']['oracle']:9s}] {it['question']}")
    if a.enqueue:
        print("enqueued:", enqueue(a.url, day))
    return 0


if __name__ == "__main__":
    sys.exit(main())
