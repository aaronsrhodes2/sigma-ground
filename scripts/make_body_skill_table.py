"""Build fixtures/body_skill_table.json: how well this model predicts each body, from the runs on disk, with the rule that sorts a body into a class.

Sources (all measured against JPL DE440, sha256 recorded):
  * v2physics_restated/rolling_shootout_r8_full_short_0.025.json and _long_: the full-system shoot-out of predictor restated_zon at step 0.025 d, 55 bodies. Median heliocentric position error over the
    windows: 14 five-year (J2007-J2020) and 7 twenty-year (J2000-J2006); leads 1, 3, 5 use all 21 windows, 10 and 20 the 7 long ones (C16).
  * v2physics_restated/SYSTEM_LOCAL_MOONS.json: the 43 moons in system-local models at step = shortest period / 300 (C14), position error relative to the planet at +1, +3, +5 yr.
  * prediction_error_table.json: the forecast's own hindcast error (4 starts) for the 10 planets and the Moon, which is the number body_position quotes.

Class rule, from the data: a moon LOSES ORBITAL PHASE when its converged +1 yr error is at least a tenth of its orbital radius. The 43 moons separate with a gap of two orders of magnitude (the largest ratio among the
others is 0.004, the smallest among the 14 that lose phase is 0.41), so the threshold is not a fit.

usage: python scripts/make_body_skill_table.py [--check]       (--check: rebuild in memory and fail if the stored file differs)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
FIX = _ROOT / "sigma_ground/field/interface/fixtures"
RES = FIX / "v2physics_restated"
OUT = FIX / "body_skill_table.json"
AU_KM = 149597870.7
PHASE_LOSS_RATIO = 0.1
FORECAST_STEP_D = 0.025
STEPS_PER_ORBIT_TARGET = 300
DISAGREE_FACTOR = 10.0
JD_REF = 2461000.5                      # elements for the orbital radius are taken at the snapshot nearest this date
NOT_CONVERGED_NOTES = {
    "Phobos": "scored only at 0.025 d, which gives it about 13 steps per orbit; at the converged step C14 reports about 1,088 km after 3 yr (probably tidal), not in the table",
    "Deimos": "scored only at 0.025 d, about 50 steps per orbit; at the converged step C14 reports about 176 km after 3 yr, not in the table",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def shootout_medians() -> dict[str, dict[str, dict]]:
    short = json.loads((RES / "rolling_shootout_r8_full_short_0.025.json").read_text(encoding="utf-8"))
    long_ = json.loads((RES / "rolling_shootout_r8_full_long_0.025.json").read_text(encoding="utf-8"))
    per: dict[tuple[str, int], list[float]] = {}
    for doc, leads in ((short, (1, 3, 5)), (long_, (1, 3, 5, 10, 20))):
        for s in doc["samples"]:
            lead = round((s["sample_jd"] - s["window_start_jd"]) / 365.25)
            if lead in leads:
                per.setdefault((s["body"], lead), []).append(s["error_au"] * AU_KM)
    out: dict[str, dict[str, dict]] = {}
    for (b, lead), v in sorted(per.items()):
        out.setdefault(b, {})[str(lead)] = {"median_km": statistics.median(v), "max_km": max(v), "windows": len(v)}
    return out


def build() -> dict:
    from sigma_ground.field.interface import celestial as C
    shoot = shootout_medians()
    local = {m["moon"]: m for m in json.loads((RES / "SYSTEM_LOCAL_MOONS.json").read_text(encoding="utf-8"))}
    hind = json.loads((FIX / "prediction_error_table.json").read_text(encoding="utf-8"))
    bodies: dict[str, dict] = {}
    for name, leads in shoot.items():
        b: dict = {"forecast_step_0.025d": leads}
        if name in local:
            m = local[name]
            a = C.elements(name, m["system"], JD_REF)["a_km"]
            ratio = m["e1"] / a
            b["converged"] = {"relative_to": m["system"], "period_days": m["period_d"], "step_days": m["dt_d"], "error_km": {"1": m["e1"], "3": m["e3"], "5": m["e5"]}, "orbital_radius_km": a,
                              "error_at_1yr_over_orbital_radius": ratio}
            full5 = leads["5"]["median_km"]
            step_limited = m["period_d"] / FORECAST_STEP_D < STEPS_PER_ORBIT_TARGET
            b["converged"]["steps_per_orbit_at_forecast_step"] = m["period_d"] / FORECAST_STEP_D
            b["converged"]["converged_over_full_system_at_5yr"] = m["e5"] / full5
            if ratio >= PHASE_LOSS_RATIO:
                b["class"] = "loses_orbital_phase"
            elif not step_limited and not (1 / DISAGREE_FACTOR <= m["e5"] / full5 <= DISAGREE_FACTOR):
                b["class"] = "measurements_disagree"
                b["note"] = (f"the converged system-local model (Sun, planet and its moons only) gives {m['e5']:.0f} km at +5 yr, the full-system run {full5:.1f} km: {m['e5'] / full5:.0f} times apart. The step is adequate in both, so the "
                             "step does not explain it; the system-local model leaves out the other planets, the likeliest cause, not tested here. Neither number is offered as the skill")
            else:
                b["class"] = "predictable"
            no_claim = b["class"] in ("measurements_disagree", "loses_orbital_phase")          # a body with no position claim has no best estimate; its errors stay in the table as data
            b["best_estimate_5yr_km"] = None if no_claim else (m["e5"] if step_limited else full5)
            b["best_estimate_basis"] = None if no_claim else ("system-local model at the converged step (C14); the forecast step cannot integrate this moon" if step_limited else "full-system run (the step is adequate for this moon and the two agree)")
        elif name in NOT_CONVERGED_NOTES:
            b["class"] = "not_scored_at_a_converged_step"
            b["note"] = NOT_CONVERGED_NOTES[name]
        else:
            b["class"] = "predictable"
            b["best_estimate_5yr_km"] = leads["5"]["median_km"]
            b["best_estimate_basis"] = "full-system run at the forecast step"
        if name in hind["table"]:
            b["hindcast"] = hind["table"][name]
            b["best_estimate_5yr_km"] = hind["table"][name]["5"]["median_km"]
            b["best_estimate_basis"] = "the forecast's own hindcast (what body_position quotes)"
        bodies[name] = b
    return {"metadata": {
        "what": "measured skill of the N-body model against JPL DE440 by body: class, the forecast-step errors, the converged errors for moons, and the forecast's hindcast error",
        "predictor": "restated_zon", "disagreement_rule": f"for a moon whose step is adequate (period over the forecast step at least {STEPS_PER_ORBIT_TARGET}) the converged system-local +5 yr error and the full-system one must agree within a factor {DISAGREE_FACTOR:g}, else the class is measurements_disagree", "phase_loss_rule": f"a moon loses orbital phase when its converged +1 yr error is at least {PHASE_LOSS_RATIO} of its orbital radius (the data separate at 0.004 and 0.41)",
        "orbital_radius_epoch_jd_tdb_near": JD_REF, "generator": "scripts/make_body_skill_table.py",
        "sources": {p: sha(f) for p, f in (("v2physics_restated/rolling_shootout_r8_full_short_0.025.json", RES / "rolling_shootout_r8_full_short_0.025.json"),
                                            ("v2physics_restated/rolling_shootout_r8_full_long_0.025.json", RES / "rolling_shootout_r8_full_long_0.025.json"),
                                            ("v2physics_restated/SYSTEM_LOCAL_MOONS.json", RES / "SYSTEM_LOCAL_MOONS.json"), ("prediction_error_table.json", FIX / "prediction_error_table.json"))},
        "ledger": "misc/CANDIDATE_LEDGER.md C14, C16, C17, C19"}, "bodies": bodies}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    doc = build()
    text = json.dumps(doc, indent=1, sort_keys=False) + "\n"
    if a.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print("body_skill_table.json is missing or differs from a rebuild", file=sys.stderr)
            return 1
        print("body_skill_table.json matches a rebuild")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    classes = {}
    for n, b in doc["bodies"].items():
        classes.setdefault(b["class"], []).append(n)
    print(f"wrote {OUT.relative_to(_ROOT)}: {len(doc['bodies'])} bodies; " + "; ".join(f"{k}: {len(v)}" for k, v in classes.items()))
    print("loses_orbital_phase:", ", ".join(sorted(classes.get("loses_orbital_phase", []))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
