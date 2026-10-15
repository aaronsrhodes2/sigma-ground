"""Step-size convergence of every moon in the full 33-body system: the headline stack from one DE440 window, one year, at several steps (the error at +1 yr: the fixture has truth only at yearly snapshots).

Why: at step 0.025 d the shoot-out still shows moon errors of 1e4 to 1e5 km (Europa, Io, Tethys, Dione ...). Whether that is the integrator (phase error, which falls as dt^4 for a
fourth-order scheme) or the fixture/physics shows up as the error's scaling with dt. A moon is CONVERGED at a step when halving the step changes its error by less than 10 %.

usage: python scripts/moon_step_convergence.py --out DIR [--window j2010] [--years 1] [--dts 0.05 0.025 0.0125 0.00625]
       (worker mode: --worker DT OUT)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
FIXTURE = _ROOT / "sigma_ground/field/interface/fixtures/de440_state_vectors_v2physics.json"
AU = 149597870.7


def run_dt(dt: float, window: str, years: float) -> dict:
    import sigma_ground.field.interface.rolling_shootout as rs
    fx = json.loads(FIXTURE.read_text())
    pred = [p for p in rs.canonical_predictors(dt) if p.name == "de440_stack"][0]
    t0 = time.time()
    samples = rs.run_rolling_shootout([window], years, 1.0, predictors=[pred], de440=fx, verbose=False, output_path=None)
    out = {}
    for s in samples:
        lead = round((s.sample_jd - s.window_start_jd) / 365.25, 2)
        if lead > 0 and s.error_au is not None:                   # truth exists only at the fixture's yearly snapshots
            out.setdefault(s.body, {})[str(lead)] = s.error_au * AU
    return {"dt_days": dt, "window": window, "years": years, "elapsed_s": time.time() - t0, "error_km": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--window", default="j2010")
    ap.add_argument("--years", type=float, default=1.0)
    ap.add_argument("--dts", type=float, nargs="+", default=[0.05, 0.025, 0.0125, 0.00625])
    ap.add_argument("--worker", nargs=2, metavar=("DT", "OUT"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        Path(a.worker[1]).write_text(json.dumps(run_dt(float(a.worker[0]), a.window, a.years)))
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    procs = {}
    for dt in a.dts:
        f = a.out / f"moon_conv_dt{dt}.json"
        if not f.exists():
            procs[dt] = subprocess.Popen([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker", str(dt), str(f), "--window", a.window, "--years", str(a.years)],
                                         stdout=open(a.out / f"moon_conv_dt{dt}.log", "w"), stderr=subprocess.STDOUT)
    for dt, p in procs.items():
        if p.wait() != 0:
            print(f"dt {dt} FAILED")
            return 1
    res = {dt: json.loads((a.out / f"moon_conv_dt{dt}.json").read_text()) for dt in a.dts}
    lead = str(round(a.years, 2))                                    # the key the workers store: str(round(years, 2))
    bodies = [b for b in res[a.dts[0]]["error_km"]]
    lines = [f"Error (km) at +{lead} yr from {a.window}, headline stack, by step (days):", "", "| body | " + " | ".join(str(d) for d in a.dts) + " | converged at |", "|---|" + "---|" * (len(a.dts) + 1)]
    for b in bodies:
        v = [res[d]["error_km"].get(b, {}).get(lead) for d in a.dts]
        conv = "-"
        for k in range(1, len(v)):
            if v[k] is not None and v[k - 1] is not None and abs(v[k] - v[k - 1]) <= 0.1 * max(v[k], 1e-9):
                conv = str(a.dts[k - 1])
                break
        lines.append(f"| {b} | " + " | ".join("-" if x is None else f"{x:,.2f}" for x in v) + f" | {conv} |")
    text = "\n".join(lines)
    (a.out / "MOON_STEP_CONVERGENCE.md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
