"""Report for the fair JPL contest (scripts/fair_jpl_contest.py): tables, a sign test per body, and a figure.

For every (vintage, start epoch, body, lead) the contest stored four heliocentric position distances in km:
  eO   our propagation of the vintage's own initial state vs DE440,
  eV   the vintage's own continuation of that state vs DE440,
  dyn  our propagation vs the vintage's own continuation (disagreement of the two dynamics given identical initial conditions),
  e0   the initial offset (recomputed here HELIOCENTRICALLY from the kernels; the worker stored it in the barycentric frame).
Reading: if eO < eV in about half the runs and dyn << eV, the two dynamics are equivalent and the error is the initial state's (the data of the day);
a clear majority eO < eV (sign-test p small) would mean our dynamics beat JPL's old ones on the same input; a clear majority eO > eV, the reverse.

usage: python scripts/fair_contest_report.py JSONL --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
V_DIR = Path("D:/datasets/jpl/vintages")
TRUTH = Path("D:/datasets/jpl/de440s.bsp")
FILES = {"DE403": "de403s.bsp", "DE405": "de405.bsp", "DE410": "de410s.bsp", "DE413": "de413.bsp", "DE418": "de418.bsp", "DE421": "de421.bsp", "DE432": "de432s.bsp"}
CHAINS = {"Mercury": ((0, 1), (1, 199)), "Venus": ((0, 2), (2, 299)), "Earth": ((0, 3), (3, 399)), "Moon": ((0, 3), (3, 301)), "Mars": ((0, 4),),
          "Jupiter": ((0, 5),), "Saturn": ((0, 6),), "Uranus": ((0, 7),), "Neptune": ((0, 8),), "Pluto": ((0, 9),)}
BODIES = list(CHAINS)


def sign_test_p(wins: int, n: int) -> float:
    """Two-sided exact binomial p-value for wins out of n at p = 0.5 (ties already excluded)."""
    if n == 0:
        return float("nan")
    k = min(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def pos(k, name, jd):
    p = 0.0
    for c, t in CHAINS[name]:
        p = p + k[c, t].compute(jd)
    return np.asarray(p, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from jplephem.spk import SPK
    rows = [json.loads(ln) for ln in a.jsonl.read_text().splitlines() if ln.strip()]
    kt = SPK.open(str(TRUTH))
    kv = {n: SPK.open(str(V_DIR / f)) for n, f in FILES.items()}
    sun = lambda k, jd: np.asarray(k[0, 10].compute(jd), dtype=float)
    cells = defaultdict(list)                                     # (body, lead) -> list of dict
    by_vintage = defaultdict(list)
    for r in rows:
        k = kv[r["vintage"]]
        for b in BODIES:
            e0 = float(np.linalg.norm((pos(k, b, r["jd"]) - sun(k, r["jd"])) - (pos(kt, b, r["jd"]) - sun(kt, r["jd"]))))
            for lead, x in r["bodies"][b].items():
                d = dict(x, e0=e0, vintage=r["vintage"], start=r["start"])
                cells[(b, int(lead))].append(d)
                by_vintage[(r["vintage"], b, int(lead))].append(d)
    summary = {}
    md = ["# Fair JPL contest: same initial state, different dynamics", "",
          f"{len(rows)} propagations (vintage x start epoch), 5 yr each, our headline-stack engine at 0.05 d against each JPL vintage's own continuation, "
          "all measured against DE440 (heliocentric position, km). Medians over runs.", "",
          "| body | lead | ours vs DE440 (eO) | vintage vs DE440 (eV) | ours vs vintage (dynamics gap) | initial offset (e0) | eO < eV | sign-test p | runs |", "|---|---|---|---|---|---|---|---|---|"]
    for b in BODIES:
        for lead in (1, 3, 5):
            xs = cells.get((b, lead), [])
            if not xs:
                continue
            wins = sum(1 for x in xs if x["eO"] < x["eV"])
            ties = sum(1 for x in xs if x["eO"] == x["eV"])
            p = sign_test_p(wins, len(xs) - ties)
            med = lambda key: statistics.median(x[key] for x in xs)
            summary[f"{b}|{lead}"] = {"eO": med("eO"), "eV": med("eV"), "dyn": med("dyn"), "e0": med("e0"), "wins": wins, "n": len(xs), "p": p}
            md.append(f"| {b} | +{lead} yr | {med('eO'):.2f} | {med('eV'):.2f} | {med('dyn'):.2f} | {med('e0'):.1f} | {wins}/{len(xs)} | {p:.3f} | {len(xs)} |")
    md += ["", "## By vintage (+5 yr, median km)", "", "| vintage | body | ours eO | vintage eV | dynamics gap | runs |", "|---|---|---|---|---|---|"]
    for v in FILES:
        for b in ("Mercury", "Earth", "Mars", "Jupiter", "Saturn", "Neptune"):
            xs = by_vintage.get((v, b, 5), [])
            if xs:
                md.append(f"| {v} | {b} | {statistics.median(x['eO'] for x in xs):.2f} | {statistics.median(x['eV'] for x in xs):.2f} | {statistics.median(x['dyn'] for x in xs):.2f} | {len(xs)} |")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "FAIR_CONTEST.md").write_text("\n".join(md))
    (a.out / "FAIR_CONTEST.json").write_text(json.dumps({"n_runs": len(rows), "summary": summary}, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(17, 7))
    for ax, b in zip(axes.ravel(), BODIES):
        xs = [x for lead in (1, 2, 3, 4, 5) for x in cells.get((b, lead), [])]
        ev = np.array([x["eV"] for x in xs]) + 1e-3
        eo = np.array([x["eO"] for x in xs]) + 1e-3
        ax.scatter(ev, eo, s=6, alpha=0.5)
        lo, hi = min(ev.min(), eo.min()), max(ev.max(), eo.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(b, fontsize=10)
        ax.set_xlabel("vintage's own continuation vs DE440 (km)")
        ax.set_ylabel("ours from the same state vs DE440 (km)")
    fig.suptitle("Fair contest: each dot = one body at one lead of one run; on the diagonal = our dynamics and JPL's old dynamics are equivalent", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out / "FAIR_CONTEST_scatter.png", dpi=105)
    plt.close(fig)
    print("\n".join(md[:60]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
