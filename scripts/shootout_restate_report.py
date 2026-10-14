"""Restated shoot-out report: the reduced model at 0.025 d with the seven heaviest asteroids and the Moon's C22 figure, against DE440, beside the r2 baseline.

Reads the DE440-start jobs written by ``scripts/fair_jpl_contest.py --plan misc/shootout_restate_plan.json`` (labels cand_long, cand_short, base_dt0025) and the stored
r2 shoot-out (headline stack, 0.05 d, full model, no asteroids). Heliocentric position error in km at +1..+5 yr (16 windows J2005-J2020) and +10, +20 yr (7 windows J2000-J2006).

Comparisons that are fair and the ones that are not:
  * Mercury, Venus, Earth, Moon, Mars: the reduced model is the full model's inner system, so candidate vs r2 on the same windows is a like-for-like paired comparison
    (the step also changes, 0.05 -> 0.025 d; the base_dt0025 control, same reduced model without asteroids or C22 on 6 windows, separates step from physics).
  * Jupiter ... Pluto: the reduced model carries SYSTEM barycentres (planet + moons), r2 carries the planet centre, so they are not compared with r2; only against DE440's barycentres.

usage: python scripts/shootout_restate_report.py RESTATE_DIR [--out OUTDIR]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
V2 = _ROOT / "sigma_ground/field/interface/fixtures/v2physics"
FIXTURE = _ROOT / "sigma_ground/field/interface/fixtures/de440_state_vectors_v2physics.json"
AU = 149597870.7
INNER = ["Mercury", "Venus", "Earth", "Moon", "Mars"]
OUTER = ["Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
SHORT_WINDOWS = [f"j{y}" for y in range(2005, 2021)]
LONG_WINDOWS = [f"j{y}" for y in range(2000, 2007)]
BASE_WINDOWS = [f"j{y}" for y in (2005, 2008, 2011, 2014, 2017, 2020)]


def load_runs(restate_dir: Path) -> dict:
    """{label: {window: {body: {lead: eO km}}}}."""
    runs: dict = {}
    for ln in (restate_dir / "fair_contest_validate.jsonl").read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        label = r["vintage"].split("|", 1)[1]
        runs.setdefault(label, {})[r["start"]] = {b: {int(k): v["eO"] for k, v in d.items()} for b, d in r["bodies"].items()}
    return runs


def load_r2() -> dict:
    """{window_key: {body: {lead: km}}} of the r2 headline stack, from both passes (a window in both passes is taken from the long pass)."""
    de = json.loads(FIXTURE.read_text())
    jd_key = {round(s["epoch"]["jd_tdb"], 3): k for k, s in de["snapshots"].items() if k.startswith("j")}
    out: dict = {}
    for f in ("rolling_shootout_short_v2physics_r2.json", "rolling_shootout_long_v2physics_r2.json"):
        for s in json.loads((V2 / f).read_text(encoding="utf-8"))["samples"]:
            if s["predictor"] != "de440_stack" or s.get("error_au") is None:
                continue
            lead = int(round((s["sample_jd"] - s["window_start_jd"]) / 365.25))
            if lead < 1:
                continue
            key = jd_key.get(round(s["window_start_jd"], 3))
            if key:
                out.setdefault(key, {}).setdefault(s["body"], {})[lead] = s["error_au"] * AU
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def cell(runs, windows, body, lead):
    return [runs.get(w, {}).get(body, {}).get(lead) for w in windows]


def fmt(x):
    return "-" if x is None else (f"{x:.2f}" if x < 100 else f"{x:,.0f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("restate_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    runs = load_runs(a.restate_dir)
    cand = {**runs.get("cand_short", {}), **{w: runs["cand_long"][w] for w in ("j2005", "j2006") if w in runs.get("cand_long", {})}}   # 16 five-year windows
    long_ = runs.get("cand_long", {})
    base = runs.get("base_dt0025", {})
    r2 = load_r2()
    lines = ["# Shoot-out restated: reduced model, step 0.025 d, seven asteroids + the Moon's C22 figure, against DE440", "",
             f"Windows: {len(cand)} five-year (J2005-J2020), {len(long_)} twenty-year (J2000-J2006), control without asteroids and C22 on {len(base)} windows. Median heliocentric position error, km.", ""]
    rows = []
    for title, bodies in (("Inner system (like-for-like with the r2 full model)", INNER), ("Giants and Pluto as system barycentres (not comparable with r2's planet centres)", OUTER)):
        lines += [f"## {title}", "", "| body | lead | r2 (0.05 d) | control (0.025 d, no ast/C22) | candidate | candidate vs r2, windows better | candidate vs control, windows better |", "|---|---|---|---|---|---|---|"]
        for b in bodies:
            for lead in (1, 3, 5, 10, 20):
                wins = LONG_WINDOWS if lead > 5 else SHORT_WINDOWS
                src = long_ if lead > 5 else cand
                c = cell(src, wins, b, lead)
                if med(c) is None:
                    continue
                r2c = [r2.get(w, {}).get(b, {}).get(lead) for w in wins]
                bc = cell(base, BASE_WINDOWS, b, lead) if lead <= 5 else []
                pairs_r2 = [(x, y) for x, y in zip(c, r2c) if x is not None and y is not None] if b in INNER else []
                pairs_b = [(src.get(w, {}).get(b, {}).get(lead), base.get(w, {}).get(b, {}).get(lead)) for w in BASE_WINDOWS] if lead <= 5 else []
                pairs_b = [(x, y) for x, y in pairs_b if x is not None and y is not None]
                vs_r2 = f"{sum(x < y for x, y in pairs_r2)}/{len(pairs_r2)}" if pairs_r2 else "-"
                vs_b = f"{sum(x < y for x, y in pairs_b)}/{len(pairs_b)}" if pairs_b else "-"
                lines.append(f"| {b} | +{lead} yr | {fmt(med(r2c)) if b in INNER else '-'} | {fmt(med(bc))} | **{fmt(med(c))}** | {vs_r2} | {vs_b} |")
                rows.append({"body": b, "lead": lead, "r2": med(r2c) if b in INNER else None, "control": med(bc), "candidate": med(c), "n": len(c)})
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "SHOOTOUT_RESTATED.md").write_text(text + "\n", encoding="utf-8")
        (a.out / "SHOOTOUT_RESTATED.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
