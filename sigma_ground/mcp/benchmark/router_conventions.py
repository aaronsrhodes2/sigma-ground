"""Training seeds and a held-out suite for the explicit conventions of the math tools (2026-10-15).

fourier_transform takes a `convention` ("ordinary" e^(-2 pi i k x), "angular" e^(-i k x), "unitary" angular over sqrt(2 pi)) and series_expansion takes one
("exclusive": terms below x^order, "inclusive": through x^order). When a question names the convention the plan says it; when it does not, the plan leaves the
argument out and the tool's default applies (the existing math seeds). These seeds teach the first case. Like the other math seeds they need no teacher:
the wording comes from many hand-written templates, held-out wordings go only to the suite, and every result is checked against an independent numerical
method (a quadrature of the transform with the right kernel; the series check of router_math).

    python -m sigma_ground.mcp.benchmark.router_conventions --out local-cache/router_conventions_v32 --per 60
    python -m sigma_ground.mcp.benchmark.router_conventions --suite local-cache/router_conventions_suite.json --per 12
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

from sigma_ground.mcp.benchmark import router_dataset as D
from sigma_ground.mcp.benchmark import router_math as M

# ── wordings (train) and held-out wordings (suite only) ────────────────────────────────────────────

FOURIER = {
    "angular": {
        "train": ["Find the Fourier transform of {e} using angular frequency.", "Fourier transform of {e}, angular frequency convention.",
                  "What is the angular-frequency Fourier transform of {e}?", "Compute the integral of {e} times e^(-i {k} {v}) over all {v}.",
                  "Transform {e} with F({k}) = ∫ f({v}) e^(-i {k} {v}) d{v}.", "Fourier transform of {e} in terms of angular frequency {k}.",
                  "Give the Fourier transform of {e} where the kernel is e^(-i {k} {v}), no prefactor.", "Angular frequency Fourier transform: {e}"],
        "holdout": ["Take the Fourier transform of {e}, in angular frequency.", "Apply F({k}) = ∫ f({v}) e^(-i {k} {v}) d{v} to {e}."],
    },
    "unitary": {
        "train": ["Find the unitary Fourier transform of {e}.", "Fourier transform of {e} with the symmetric 1/√(2π) convention.",
                  "Compute (1/√(2π)) ∫ {e} e^(-i {k} {v}) d{v}.", "What is the unitary angular-frequency Fourier transform of {e}?",
                  "Fourier transform of {e}, unitary convention.", "Transform {e} using the symmetric (unitary) Fourier convention.",
                  "Give the Fourier transform of {e} normalised by 1/sqrt(2π), angular frequency."],
        "holdout": ["Use the unitary convention to Fourier transform {e}.", "Fourier transform {e} with a 1/sqrt(2π) prefactor and kernel e^(-i {k} {v})."],
    },
    "ordinary": {
        "train": ["Find the Fourier transform of {e} in ordinary frequency (cycles per unit).", "Fourier transform of {e} with the kernel e^(-2π i {k} {v}).",
                  "Fourier transform of {e}, ordinary frequency {k} in hertz.", "Compute the integral of {e} times e^(-2πi {k} {v}) over all {v}.",
                  "What is the ordinary-frequency Fourier transform of {e}?"],
        "holdout": ["Transform {e} to ordinary frequency with the e^(-2πi {k} {v}) kernel."],
    },
}

SERIES = {
    "inclusive": {
        "train": ["Expand {e} as a Taylor series about {v} = {p}, up to and including order {n}.", "Taylor series of {e} around {v} = {p} through order {n}.",
                  "Give the series of {e} about {v} = {p}, keeping every term up to and including {pw}.", "Series for {e} near {v} = {p} that includes the {pw} term.",
                  "Taylor expansion of {e} about {v} = {p}, through the {pw} term.", "What is the degree-{n} Taylor polynomial of {e} at {v} = {p}?",
                  "Find the Taylor polynomial of degree {n} for {e} about {v} = {p}.", "Expand {e} about {v} = {p} to order {n} inclusive."],
        "holdout": ["Approximate {e} near {v} = {p} with a polynomial of degree {n}.", "Series of {e} about {v} = {p}, order {n} included."],
    },
    "exclusive": {
        "train": ["Taylor series of {e} about {v} = {p}, keeping only the terms below {pw}.", "Expand {e} about {v} = {p} up to but not including the {pw} term.",
                  "Series of {e} about {v} = {p} with the error O({pw}).", "Give the expansion of {e} around {v} = {p} up to O({pw}).",
                  "Taylor expansion of {e} about {v} = {p} that stops before the {pw} term."],
        "holdout": ["Expand {e} about {v} = {p} with remainder O({pw})."],
    },
}

# (sympy form, textbook form, spoken form), each with the variable in place: renaming x afterwards cannot see the x in "4x"
GAUSS = [("exp(-{v}**2)", "e^(-{v}^2)", "e to the minus {v} squared"), ("exp(-2*{v}**2)", "e^(-2{v}^2)", "e to the minus 2 {v} squared"),
         ("exp(-3*{v}**2)", "e^(-3{v}^2)", "e to the minus 3 {v} squared"), ("exp(-{v}**2/2)", "e^(-{v}^2/2)", "e to the minus {v} squared over 2"),
         ("exp(-4*{v}**2)", "e^(-4{v}^2)", "e to the minus 4 {v} squared")]
KERNEL_VARS = {"angular": [("x", "k")] * 4 + [("t", "w"), ("x", "w"), ("t", "omega")], "unitary": [("x", "k")] * 4 + [("t", "w"), ("x", "w"), ("t", "omega")],
               "ordinary": [("x", "k")] * 5 + [("t", "f"), ("t", "nu")]}


def p_fourier(rng: random.Random, holdout: bool, conv: str | None = None) -> M.P:
    conv = conv or rng.choice(["angular", "angular", "unitary", "unitary", "ordinary"])
    xv, kv = rng.choice(KERNEL_VARS[conv])
    s, c, w = (form.format(v=xv) for form in rng.choice(GAUSS))
    pool = FOURIER[conv]["holdout" if holdout else "train"]
    template = rng.choice(pool)
    q = template.format(e=rng.choice([c, s, w]), v=xv, k=kv)
    if kv != "k" and "{k}" not in template:           # the label names the transform variable, so the question must (the expression already shows the other)
        q = q.rstrip("?.") + rng.choice([f", writing the result in terms of {kv}", f" (transform variable {kv})", f", as a function of {kv}"])
    q = M.finish(q, rng)
    args: dict = {"expression": s}
    if xv != "x":
        args["x_var"] = xv
    if kv != "k":
        args["k_var"] = kv
    args["convention"] = conv
    return M.P("math_fourier_conv", "fourier_transform", args, q, {"kind": "fourier_conv", "expr": s, "xvar": xv, "kvar": kv, "conv": conv})


_FUNCS = {"exp": ("exp({v})", "e^{v}", "e to the {v}"), "sin": ("sin({v})", "sin({v})", "sine of {v}"), "cos": ("cos({v})", "cos({v})", "cosine of {v}"),
          "geo": ("1/(1 - {v})", "1/(1 - {v})", "1 over (1 minus {v})"), "log1p": ("log(1 + {v})", "ln(1 + {v})", "natural log of (1 plus {v})"),
          "recip1p": ("1/(1 + {v})", "1/(1 + {v})", "1 over (1 plus {v})"), "atan": ("atan({v})", "arctan({v})", "arctangent of {v}"),
          "sqrt1p": ("sqrt(1 + {v})", "sqrt(1 + {v})", "square root of (1 plus {v})")}


def p_series(rng: random.Random, holdout: bool, conv: str | None = None) -> M.P:
    conv = conv or rng.choice(["inclusive", "inclusive", "inclusive", "exclusive"])
    v = M.var_choice(rng, (("x", 14), ("t", 1)))
    name, pt = rng.choice(M.SERIES)
    s, c, w = (t.format(v=v) for t in _FUNCS[name])
    n = rng.choice([2, 3, 4, 5])
    ptext = M.bound_text(pt, rng) if pt in M.BOUND_WORDS else pt
    pw = f"{v}^{n}" if pt == "0" else f"({v} - {ptext})^{n}"
    pool = SERIES[conv]["holdout" if holdout else "train"]
    q = M.finish(rng.choice(pool).format(e=rng.choice([c, w, s]), v=v, p=ptext, n=n, pw=pw), rng)
    args: dict = {"expression": s}
    if v != "x" or rng.random() < 0.7:
        args["variable"] = v
    args.update({"point": pt, "order": n, "convention": conv})
    return M.P("math_series_conv", "series_expansion", args, q, {"kind": "series", "expr": s, "var": v, "point": pt, "order": n + (1 if conv == "inclusive" else 0)})


GENERATORS = {"math_fourier_conv": p_fourier, "math_series_conv": p_series}


def check(spec: dict, raw) -> bool | None:
    """The independent verdict: a quadrature of the transform with the convention's own kernel; the series check of router_math otherwise."""
    if spec["kind"] != "fourier_conv":
        return M.check(spec, raw)
    try:
        import scipy.integrate as si
        f = M.sp.lambdify(M.sp.Symbol(spec["xvar"]), M._sym(spec["expr"]), "numpy")
        point = 0.9
        omega = 2 * math.pi * point if spec["conv"] == "ordinary" else point            # the kernel's own argument
        want, _ = si.quad(lambda x: float(f(x)) * math.cos(omega * x), -np.inf, np.inf, limit=200)       # the gaussians are even: the sine part vanishes
        if spec["conv"] == "unitary":
            want /= math.sqrt(2 * math.pi)
        got = M._num(M._sym(raw), **{spec["kvar"]: point})
        return None if got is None else M._close(want, got, 1e-5)
    except Exception:  # noqa: BLE001 — an unexpected shape: no independent verdict
        return None


async def generate(session, per: int, seed: int, holdout: bool = False, forbidden: set | None = None):
    """-> (seeds, report): about `per` problems for each convention of each family, every one executed on the real server and checked."""
    rng = random.Random(f"conv:{seed}:{'h' if holdout else 't'}")
    seeds, seen = [], set(forbidden or ())
    report = {"made": {}, "drops": {}, "oracle_disagree": [], "unchecked": {}}
    plan = [("fourier", c) for c in ("angular", "unitary", "ordinary")] + [("series", c) for c in ("inclusive", "exclusive")]
    for family, conv in plan:
        fn = p_fourier if family == "fourier" else p_series
        want = per if conv in ("angular", "unitary", "inclusive") else max(8, per // 2)
        made = attempts = 0
        while made < want and attempts < want * 60:
            attempts += 1
            p = fn(rng, holdout, conv)
            key = D._norm(p.question)
            if key in seen:
                report["drops"]["duplicate"] = report["drops"].get("duplicate", 0) + 1
                continue
            res = await D.execute(session, p.tool, p.args, p.question, timeout_s=60, keep_raw=True)
            if res is None or res.get("raw") is None:
                report["drops"][f"server returned no value: {p.recipe}"] = report["drops"].get(f"server returned no value: {p.recipe}", 0) + 1
                continue
            verdict = check(p.check, res["raw"])
            if verdict is False:
                report["oracle_disagree"].append({"recipe": p.recipe, "question": p.question, "args": p.args, "result": res["raw"]})
                report["drops"]["oracle disagrees"] = report["drops"].get("oracle disagrees", 0) + 1
                continue
            seen.add(key)
            made += 1
            if verdict is None:
                report["unchecked"][p.recipe] = report["unchecked"].get(p.recipe, 0) + 1
            seeds.append(M.seed_of(p, res["raw"], verdict, len(seeds), seed))
            if holdout:
                seeds[-1]["expected"] = {"tool": p.tool, "args": p.args, "result": res["raw"], "spec": p.check}
        report["made"][f"{family}:{conv}"] = made
        print(f"  {family}:{conv:10s} {made:4d}/{want}", flush=True)
    return seeds, report


async def _amain(a) -> int:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    async with stdio_client(D.server_params(a.server_mem_gb)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await D.execute(session, "free_fall_time", {"height_m": 1.0}, "", timeout_s=120)
            if a.suite:
                suite, report = await generate(session, a.per, a.seed + 1000, holdout=True)
                items = [{"id": f"conv_{i:04d}", "question": s["question"], "expected": s["expected"], "oracle": s["meta"]["oracle"],
                          "recipe": s["meta"]["recipe"], "kind": "templated-holdout"} for i, s in enumerate(suite)]
                a.suite.write_text(json.dumps({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "items": items,
                                               "report": {k: v for k, v in report.items() if k != "oracle_disagree"},
                                               "oracle_disagreements": report["oracle_disagree"]}, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
                print(f"Wrote {a.suite}: {len(items)} convention-suite questions; oracle disagreements: {len(report['oracle_disagree'])}")
                return 0
            forbidden = {D._norm(i["question"]) for f in a.holdout_file for i in json.loads(f.read_text(encoding="utf-8"))["items"] if f.exists()}
            seeds, report = await generate(session, a.per, a.seed, forbidden=forbidden)
    manifest = M.write_seeds(seeds, report, a.out, a.per, a.seed)
    print(f"\nWrote {a.out}: {manifest['seeds']} seeds; oracle {manifest['oracle']}; drops {manifest['drops']}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=D.REPO / "local-cache" / "router_conventions_v32")
    p.add_argument("--per", type=int, default=60, help="problems per convention (the 'ordinary' and 'exclusive' ones get half)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--suite", type=Path, help="write the held-out convention suite here instead of training seeds")
    p.add_argument("--holdout-file", type=Path, nargs="*", default=[D.REPO / "local-cache" / "router_conventions_suite.json",
                                                                      D.REPO / "local-cache" / "router_math_suite.json"],
                   help="suites whose questions are never generated as training data")
    p.add_argument("--server-mem-gb", type=float, default=6)
    return asyncio.run(_amain(p.parse_args()))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
