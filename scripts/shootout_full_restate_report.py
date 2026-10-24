"""Full-system shoot-out restated: the headline stack at step 0.025 d with and without DE440's asteroids, Kuiper-belt ring and the Moon's C22 figure, against DE440, for all 33 bodies.

Reads the merged output of ``scripts/run_shootout_parallel.py --plan`` (rolling_shootout_short.json, rolling_shootout_long.json) and the stored r2 baseline (step 0.05 d, no
candidate layers). Predictors: ``de440_stack`` (control, step 0.025 d), ``restated_ast_ring_c22`` (asteroids + ring + C22), ``restated_ast_c22`` (no ring).

Windows: candidate five-year numbers use J2005-J2020 (J2005 and J2006 from the 20-year jobs), twenty-year numbers J2000-J2006; the control runs six windows (J2005, 08, 11, 14, 17,
20); the ring pair (with and without) runs J2000 and J2003. Heliocentric position error in km. The fast inner moons (Phobos, Deimos, Mimas, Enceladus, Io, Miranda) are
integrator-limited at this step and are scored in their own table, never in the headline averages (the Captain's rule, 2026-10-10).

usage: python scripts/shootout_full_restate_report.py RESTATE_DIR [--out OUTDIR]
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
GIANTS = ["Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
MOONS = {"Mars": ["Phobos", "Deimos"], "Jupiter": ["Io", "Europa", "Ganymede", "Callisto", "Amalthea", "Thebe", "Himalia"],
         "Saturn": ["Mimas", "Enceladus", "Tethys", "Dione", "Rhea", "Titan", "Hyperion", "Iapetus", "Phoebe", "Janus", "Epimetheus", "Prometheus", "Pandora", "Atlas", "Pan", "Helene"],
         "Uranus": ["Miranda", "Ariel", "Umbriel", "Titania", "Oberon", "Puck", "Portia", "Juliet", "Belinda"], "Neptune": ["Triton", "Proteus", "Nereid", "Larissa", "Galatea", "Despina"],
         "Pluto": ["Charon", "Nix", "Hydra", "Kerberos", "Styx"]}
# Not scoreable at the global step (period under about 2 d, or under 4 d for Io's resonant neighbours): own table, never in the headline (the Captain's rule, 2026-10-10). Moons absent from a run are skipped.
LIMITED = ("Phobos", "Deimos", "Mimas", "Enceladus", "Io", "Miranda", "Amalthea", "Thebe", "Janus", "Epimetheus", "Prometheus", "Pandora", "Atlas", "Pan", "Puck", "Portia", "Juliet", "Belinda",
           "Proteus", "Larissa", "Galatea", "Despina")
SHORT5 = [f"j{y}" for y in range(2005, 2021)]
LONG20 = [f"j{y}" for y in range(2000, 2007)]
CONTROL_W = [f"j{y}" for y in (2005, 2008, 2011, 2014, 2017, 2020)]
RING_W = ["j2000", "j2003"]


def jd_to_key() -> dict:
    de = json.loads(FIXTURE.read_text())
    return {round(s["epoch"]["jd_tdb"], 3): k for k, s in de["snapshots"].items() if k.startswith("j")}


def load(paths, predictor_filter=None) -> dict:
    """{predictor: {window: {body: {lead: km}}}}"""
    key_of = jd_to_key()
    out: dict = {}
    for f in paths:
        if not Path(f).exists():
            continue
        for s in json.loads(Path(f).read_text())["samples"]:
            if s.get("error_au") is None:
                continue
            if predictor_filter and s["predictor"] not in predictor_filter:
                continue
            lead = int(round((s["sample_jd"] - s["window_start_jd"]) / 365.25))
            w = key_of.get(round(s["window_start_jd"], 3))
            if lead >= 1 and w:
                out.setdefault(s["predictor"], {}).setdefault(w, {}).setdefault(s["body"], {})[lead] = s["error_au"] * AU
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(x):
    return "-" if x is None else (f"{x:.2f}" if x < 100 else f"{x:,.0f}")


def cells(d, windows, body, lead):
    return [d.get(w, {}).get(body, {}).get(lead) for w in windows]


def paired(a, b):
    p = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    return f"{sum(x < y for x, y in p)}/{len(p)}" if p else "-"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("restate_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--candidate", default="restated_ast_ring_c22", help="predictor name of the candidate stack (r4 run: restated_ast_ring_kbo_c22)")
    ap.add_argument("--label", default="SHOOTOUT_FULL_RESTATED", help="output file stem (a new label keeps the earlier tables)")
    a = ap.parse_args()
    new = load([a.restate_dir / "rolling_shootout_short.json", a.restate_dir / "rolling_shootout_long.json"])
    r2 = load([V2 / "rolling_shootout_short_v2physics_r2.json", V2 / "rolling_shootout_long_v2physics_r2.json"], {"de440_stack"})["de440_stack"]
    cand = new.get(a.candidate, {})
    ctrl = new.get("de440_stack", {})
    noring = new.get("restated_ast_c22", {})
    lines = ["# Full-system shoot-out restated (step 0.025 d)", "",
             "Median heliocentric position error, km. r2 = stored baseline (0.05 d, no asteroids, ring or C22); control = same stack at 0.025 d; candidate = + DE440's 343 asteroids + 36-point ring + the Moon's C22" + (" + seven identified Kuiper-belt objects (run on the r4 fixture)." if "kbo" in a.candidate else "."),
             f"Windows: candidate {len([w for w in SHORT5 if w in cand])} five-year / {len([w for w in LONG20 if w in cand])} twenty-year; control {len([w for w in CONTROL_W if w in ctrl])}; ring pair {len([w for w in RING_W if w in noring])}.", ""]
    rows = []

    def table(title, bodies, leads=(1, 3, 5, 10, 20)):
        lines.extend([f"## {title}", "", "| body | lead | r2 | control | candidate | cand vs r2 (windows better) | cand vs control |", "|---|---|---|---|---|---|---|"])
        for b in bodies:
            for lead in leads:
                wins = LONG20 if lead > 5 else SHORT5
                c = cells(cand, wins, b, lead)
                if med(c) is None:
                    continue
                r2c = cells(r2, wins, b, lead)
                k_c = cells(ctrl, CONTROL_W, b, lead) if lead <= 5 else []
                kc_pair = paired([cand.get(w, {}).get(b, {}).get(lead) for w in CONTROL_W], k_c) if k_c else "-"
                lines.append(f"| {b} | +{lead} yr | {fmt(med(r2c))} | {fmt(med(k_c))} | **{fmt(med(c))}** | {paired(c, r2c)} | {kc_pair} |")
                rows.append({"body": b, "lead": lead, "r2": med(r2c), "control": med(k_c), "candidate": med(c), "n": len(c)})
        lines.append("")

    table("Inner system", INNER)
    table("Giants and Pluto (planet centres, like r2)", GIANTS)
    heads = [m for ms in MOONS.values() for m in ms if m not in LIMITED]
    table("Moons (headline: not integrator-limited at this step)", heads, leads=(1, 5, 20))
    table("Integrator-limited fast moons (own table, never averaged into the headline)", [m for m in LIMITED], leads=(1, 5))
    lines.extend(["## The ring, with and without (windows J2000, J2003, 20 years)", "", "| body | lead | candidate without ring" + (" or KBOs" if "kbo" in a.candidate else "") + " | candidate with ring" + (" and KBOs" if "kbo" in a.candidate else "") + " | change |", "|---|---|---|---|---|"])
    for b in ["Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]:
        for lead in (5, 10, 20):
            n0 = med(cells(noring, RING_W, b, lead))
            n1 = med(cells(cand, RING_W, b, lead))
            if n0 is not None and n1 is not None:
                lines.append(f"| {b} | +{lead} yr | {fmt(n0)} | {fmt(n1)} | {n1 - n0:+.2f} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"{a.label}.md").write_text(text + "\n", encoding="utf-8")
        (a.out / f"{a.label}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
