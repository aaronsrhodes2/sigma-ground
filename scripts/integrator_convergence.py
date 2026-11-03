"""Phase 3 of C26 (2026-11-02): can a higher-order step give today's accuracy at a fraction of the computation?

One year of the reduced model (`restated_zon`'s physics) from DE440's state at a start date, each arm = (integrator, step): Forest-Ruth at 0.0125 d (the converged control, C19) and
0.025 d, Yoshida-6 and Yoshida-8 at 0.05, 0.1 and 0.2 d. Each arm is scored two ways: its heliocentric position error against the truth kernel (DE442) at +1 yr, body by body, and
its distance from the Forest-Ruth 0.0125 d control (which is what "the same answer" means here); the force evaluations are counted. Pre-registered (ledger C26): Yoshida-8 at 0.2 d
within 1 % of the control on Mercury and the Moon at a third of the evaluations or fewer.

usage: python scripts/integrator_convergence.py [--start 2015] [--years 1] [--out fixtures/v2physics_restated/INTEGRATOR_CONVERGENCE] [--arms fr4:0.0125 fr4:0.025 y6:0.1 y8:0.2 ...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
DEFAULT_OUT = _ROOT / "sigma_ground/field/interface/fixtures/v2physics_restated/INTEGRATOR_CONVERGENCE"
DEFAULT_ARMS = ["fr4:0.0125", "fr4:0.025", "y6:0.05", "y6:0.1", "y8:0.1", "y8:0.2"]


def jd_jan1(year: int) -> float:
    return dt.date(year, 1, 1).toordinal() + 1721424.5


def run_arm(kernel, truth, jd0: float, jd1: float, integrator: str, dt_days: float, predictor: str):
    from sigma_ground.field.interface import reduced_model as rm
    system, order = rm.build_system(kernel, jd0, predictor, dt_days=dt_days, horizon_days=jd1 - jd0 + 10.0, fixture_key="j2010")
    calls = [0]
    real = system.compute_accelerations

    def counted():
        calls[0] += 1
        return real()
    system.compute_accelerations = counted
    t0 = time.time()
    d = rm.propagate_dense(system, jd0, jd1, dt_days, every=10 ** 9, integrator=integrator)
    secs = time.time() - t0
    pos = d["pos_km"][-1]
    sun = order.index("Sun")
    hel = {n: pos[i] - pos[sun] for i, n in enumerate(order) if n != "Sun"}
    err = {n: float(np.linalg.norm(hel[n] - (rm.kernel_state(truth, n, jd1)[0] - rm.kernel_state(truth, "Sun", jd1)[0]))) for n in hel}
    return {"integrator": integrator, "dt_days": dt_days, "force_evaluations": calls[0], "seconds": round(secs, 1), "error_km": err, "_hel": {n: v.tolist() for n, v in hel.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--years", type=float, default=1.0)
    ap.add_argument("--predictor", default="restated_zon")
    ap.add_argument("--arms", nargs="*", default=DEFAULT_ARMS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    from jplephem.spk import SPK
    from sigma_ground.field import datasets
    kernel = SPK.open(str(datasets.de440_kernel()))
    truth_file, truth_name = datasets.truth_kernel()
    truth = SPK.open(str(truth_file))
    jd0 = jd_jan1(a.start)
    jd1 = jd0 + a.years * 365.25
    arms = []
    for spec in a.arms:
        integ, step = spec.split(":")
        r = run_arm(kernel, truth, jd0, jd1, integ, float(step), a.predictor)
        print(f"{integ} {step} d: {r['force_evaluations']} evaluations, {r['seconds']} s, Mercury {r['error_km']['Mercury']:.4f} Moon {r['error_km']['Moon']:.4f} km", flush=True)
        arms.append(r)
    control = next((r for r in arms if r["integrator"] == "fr4" and r["dt_days"] == 0.0125), arms[0])
    ref = {n: np.array(v) for n, v in control["_hel"].items()}             # taken before any arm's positions are dropped (the control is one of the arms)
    for r in arms:
        r["vs_control_km"] = {n: float(np.linalg.norm(np.array(r["_hel"][n]) - ref[n])) for n in r["_hel"]}
        r["evaluations_vs_control"] = round(r["force_evaluations"] / control["force_evaluations"], 3)
    for r in arms:
        del r["_hel"]
    doc = {"metadata": {"what": "one-year integrator comparison on the reduced model", "predictor": a.predictor, "start": a.start, "years": a.years, "truth": truth_name,
                        "control": "fr4 at 0.0125 d", "generator": "scripts/integrator_convergence.py", "date": dt.date.today().isoformat()}, "arms": arms}
    a.out.with_suffix(".json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    bodies = [n for n in arms[0]["error_km"]]
    lines = [f"# Integrator comparison: one year from {a.start}-01-01, reduced model `{a.predictor}`, scored against {truth_name} (C26 Phase 3)", "",
             "Control = Forest-Ruth at 0.0125 d. `vs control` is the distance from the control's own answer (km); `error` is against the truth kernel (km).", "",
             "| arm | evaluations | relative | " + " | ".join(f"{b} error / vs control" for b in bodies) + " |", "|---|---|---|" + "---|" * len(bodies)]
    for r in arms:
        cells = " | ".join(f"{r['error_km'][b]:.4f} / {r['vs_control_km'][b]:.4f}" for b in bodies)
        lines.append(f"| {r['integrator']} {r['dt_days']} d | {r['force_evaluations']} | {r['evaluations_vs_control']} | {cells} |")
    a.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", a.out.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
