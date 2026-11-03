"""The fair contest's report (v2, C26 M-B): per body and lead, ours against reality, JPL's prediction at the time against reality, the dynamics gap between the two, and the
sign test of who was closer; plus the blanket arrays (issue year x target year) saved explicitly this time (the 2026-10-11 mesh kept only summaries).

Reads the contest's jsonl (one run per line: {vintage, start, jd, bodies: {body: {lead: {eO, eV, dyn, e0[, eT]}}}, [predictor, truth]}). Writes <out>.md and <out>.json.
The pre-registered M-B targets (ledger C26) are printed beside the numbers: dynamics gap at +5 yr at or below half of the 2026-10-12 contest's (Moon 3.40, Mars 2.11, Earth 0.90,
Mercury 1.20 km), and Venus to Pluto indistinguishable from JPL-at-the-time (sign test p > 0.05).

usage: python scripts/fair_contest_report.py RUNS.jsonl [--out fixtures/vintages/FAIR_CONTEST_v2] [--label "contest v2"]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
BODIES = ["Mercury", "Venus", "Earth", "Moon", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
BASELINE_GAP_5YR = {"Moon": 3.40, "Mars": 2.11, "Earth": 0.90, "Mercury": 1.20}          # the 2026-10-12 contest (FAIR_CONTEST.md), old model at 0.05 d


def sign_test_p(wins: int, n: int) -> float:
    """Two-sided binomial sign test, p = 0.5."""
    if n == 0:
        return float("nan")
    k = min(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path)
    ap.add_argument("--out", type=Path, default=_ROOT / "sigma_ground/field/interface/fixtures/vintages/FAIR_CONTEST_v2")
    ap.add_argument("--label", default="fair contest v2")
    a = ap.parse_args()
    from vintage_deviance_mesh import VINTAGES
    rel = {name: r.year + (r.timetuple().tm_yday - 1) / 365.25 for name, _, r in VINTAGES}
    runs = [json.loads(ln) for ln in a.runs.read_text(encoding="utf-8").splitlines() if ln.strip()]
    truth = runs[0].get("truth", "DE440")
    predictor = runs[0].get("predictor", "headline stack (2026-10-12 path)")
    leads_all = sorted({int(round(float(k))) for r in runs for b in r["bodies"].values() for k in b})
    table = {}
    for n in BODIES:
        table[n] = {}
        for L in leads_all:
            rows = []
            for r in runs:
                d = r["bodies"].get(n, {})
                x = d.get(str(L)) or d.get(L)
                if x:
                    rows.append(x)
            if not rows:
                continue
            wins = sum(1 for x in rows if x["eO"] < x["eV"])
            cell = {"eO_med": statistics.median(x["eO"] for x in rows), "eV_med": statistics.median(x["eV"] for x in rows), "dyn_med": statistics.median(x["dyn"] for x in rows),
                    "e0_med": statistics.median(x["e0"] for x in rows), "ours_wins": wins, "n": len(rows), "p_sign": sign_test_p(wins, len(rows))}
            if any("eT" in x for x in rows):
                cell["eT_med"] = statistics.median(x["eT"] for x in rows if "eT" in x)
            table[n][str(L)] = cell
    # the blankets: issue year (vintage) x target year, median over the runs whose lead lands there
    vintages = sorted({r["vintage"] for r in runs if r["vintage"] in rel}, key=lambda v: rel[v])
    cells: dict = {}
    for r in runs:
        if r["vintage"] not in rel:
            continue
        y0 = 2000 + round((r["jd"] - 2451545.0) / 365.25)
        for n, d in r["bodies"].items():
            for L, x in d.items():
                c = cells.setdefault((n, r["vintage"], y0 + int(round(float(L)))), {"eO": [], "eV": [], "eT": []})
                c["eO"].append(x["eO"]); c["eV"].append(x["eV"])
                if "eT" in x:
                    c["eT"].append(x["eT"])
    targets = sorted({k[2] for k in cells})
    blankets = {}
    for n in BODIES:
        if not any(k[0] == n for k in cells):
            continue
        g = lambda key: [[(statistics.median(cells[(n, v, t)][key]) if (n, v, t) in cells and cells[(n, v, t)][key] else None) for t in targets] for v in vintages]
        blankets[n] = {"ours": g("eO"), "jpl": g("eV"), "band": g("eT")}
    verdict = {}
    for n, base in BASELINE_GAP_5YR.items():
        c = table.get(n, {}).get("5")
        if c:
            verdict[n] = {"gap_5yr_km": c["dyn_med"], "baseline_km": base, "target_km": base / 2, "met": c["dyn_med"] <= base / 2}
    parity = {n: {"p_sign_5yr": table[n]["5"]["p_sign"], "met": table[n]["5"]["p_sign"] > 0.05} for n in ("Venus", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto") if table.get(n, {}).get("5")}
    doc = {"metadata": {"what": a.label, "runs": len(runs), "truth": truth, "predictor": predictor, "source": a.runs.name, "vintages": vintages, "issue_years": [rel[v] for v in vintages],
                        "targets": targets, "generator": "scripts/fair_contest_report.py"},
           "table": table, "verdict_M_B": {"gap_halved": verdict, "parity_outer": parity}, "blankets": blankets}
    a.out.with_suffix(".json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    lines = [f"# {a.label}: {len(runs)} propagations, predictor `{predictor}`, scored against {truth}", "",
             "Each run starts from a JPL ephemeris's OWN state at a yearly date after its release and is compared with that ephemeris's own continuation (JPL's prediction at the time) and with "
             f"{truth} (reality). Median km over runs. `eT` is DE440 against {truth}: the width of reality's own spread. `p` is the two-sided sign test of ours-closer-than-JPL.", "",
             "| body | lead | ours vs reality | JPL-at-the-time vs reality | ours vs JPL (dynamics gap) | start offset | reality spread | ours closer | p | runs |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n in BODIES:
        for L in ("1", "3", "5", "10", "20"):
            c = table.get(n, {}).get(L)
            if c:
                lines.append(f"| {n} | +{L} yr | {c['eO_med']:.3f} | {c['eV_med']:.3f} | {c['dyn_med']:.3f} | {c['e0_med']:.2f} | {c.get('eT_med', float('nan')):.3f} | {c['ours_wins']}/{c['n']} | {c['p_sign']:.3f} | {c['n']} |")
    lines += ["", "## The pre-registered M-B targets (ledger C26)", "", "| body | dynamics gap +5 yr | 2026-10-12 baseline | target (half) | met |", "|---|---|---|---|---|"]
    for n, v in verdict.items():
        lines.append(f"| {n} | {v['gap_5yr_km']:.3f} | {v['baseline_km']:.2f} | {v['target_km']:.2f} | {'YES' if v['met'] else 'no'} |")
    lines += ["", "| body | sign test p at +5 yr | indistinguishable (p > 0.05) |", "|---|---|---|"]
    for n, v in parity.items():
        lines.append(f"| {n} | {v['p_sign_5yr']:.3f} | {'YES' if v['met'] else 'no'} |")
    a.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", a.out.with_suffix(".md"))
    for n, v in verdict.items():
        print(f"  {n}: gap {v['gap_5yr_km']:.3f} km (target {v['target_km']:.2f}) {'MET' if v['met'] else 'not met'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
