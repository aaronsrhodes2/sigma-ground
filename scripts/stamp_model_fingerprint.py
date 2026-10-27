"""Stamp the physics fingerprint on a forecast or error table that was built BEFORE builders recorded one (the first corrected build of 2026-10-26 was already running).

A stored table is a cache of a calculation. Its fingerprint (reduced_model.model_fingerprint: the physics code, the predictor's configuration, the body parameters and the mass data) lets body_position notice that the
calculation has changed since. A late stamp is only honest if nothing the fingerprint covers changed between the START of the build and now, so this script checks that with git (no commit touching those files since the
start, no uncommitted edit to them) and refuses otherwise. The stamp records that it was made after the build.

usage: python scripts/stamp_model_fingerprint.py --forecast        # the newest entry of fixtures/forecast_manifest.json
       python scripts/stamp_model_fingerprint.py --error-table     # fixtures/prediction_error_table.json (its start is taken from --since)
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
FIX = _ROOT / "sigma_ground/field/interface/fixtures"


def physics_changed_since(start: datetime.datetime) -> str | None:
    """Why the fingerprint cannot be stamped late, or None if the covered files are unchanged since `start` (UTC)."""
    from sigma_ground.field.interface import reduced_model as rm
    base = Path(rm.__file__).resolve().parent
    norm = [(base / rel).resolve().relative_to(_ROOT).as_posix() for rel in rm.FINGERPRINT_CODE + rm.FINGERPRINT_DATA]
    log = subprocess.run(["git", "log", f"--since={start.strftime('%Y-%m-%dT%H:%M:%SZ')}", "--format=%h %s", "--", *norm], cwd=_ROOT, capture_output=True, text=True).stdout.strip()
    if log:
        return "commits touched the physics files since the build started: " + log.splitlines()[0]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *norm], cwd=_ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        return "uncommitted edits to the physics files: " + dirty.splitlines()[0]
    return None


def reproduce(entry: dict, years: float) -> float:
    """Run the CURRENT code from the table's start state and return the largest position difference (km, any body, any stored sample) from the stored table over the first `years` years."""
    import numpy as np
    from jplephem.spk import SPK
    from sigma_ground.field import datasets
    from sigma_ground.field.interface import reduced_model as rm
    table = np.load(datasets.datasets_root() / "forecast" / entry["file"])
    kernel = SPK.open(str(datasets.de440_kernel()))
    span = (entry["end_jd_tdb"] - entry["start_jd_tdb"]) + 10.0                              # the build's perturber horizon
    system, order = rm.build_system(kernel, entry["start_jd_tdb"], entry["predictor"], dt_days=entry["dt_days"], horizon_days=span, fixture_key="j2026")
    assert list(order) == [str(n) for n in table["names"]], "body order differs from the stored table"
    jd1 = entry["start_jd_tdb"] + years * 365.25
    d = rm.propagate_dense(system, entry["start_jd_tdb"], jd1, entry["dt_days"], every=entry["stored_every_steps"])
    n = len(d["jd"])
    assert np.allclose(d["jd"], table["jd"][:n], rtol=0, atol=1e-6), "sample times differ from the stored table"
    return float(np.abs(d["pos_km"] - table["pos_km"][:n]).max())


def reproduce_errors(doc: dict, start_year: int, leads: int) -> float:
    """Rerun the CURRENT code for the hindcast that started on 1 January of `start_year` and return the largest difference (km) between its heliocentric error against DE440 at leads 1..`leads` and the stored per-start value."""
    import numpy as np
    from jplephem.spk import SPK
    from sigma_ground.field import datasets
    from sigma_ground.field.interface import reduced_model as rm
    import make_prediction_error_table as m                                                   # same helpers as the builder
    meta = doc["metadata"]
    kernel = SPK.open(str(datasets.de440_kernel()))
    jd0 = m.jd_jan1(start_year)
    stored_leads = [k for k in range(1, 21) if start_year + k <= meta["targets_end"]]
    jd_end = jd0 + stored_leads[-1] * 365.25                                                   # the builder's perturber horizon, not the shorter run's
    system, order = rm.build_system(kernel, jd0, meta["predictor"], dt_days=meta["dt_days"], horizon_days=jd_end - jd0 + 10.0, fixture_key="j2010")
    every = int(round(365.25 / meta["dt_days"]))
    d = rm.propagate_dense(system, jd0, jd0 + leads * 365.25, meta["dt_days"], every=every)
    sun_i = order.index("Sun")
    worst = 0.0
    for row, jd in enumerate(d["jd"]):
        k = int(round((jd - jd0) / 365.25))
        if k < 1 or k > leads:
            continue
        for i, n in enumerate(order):
            if n == "Sun":
                continue
            truth = rm.kernel_state(kernel, n, jd)[0] - rm.kernel_state(kernel, "Sun", jd)[0]
            e = float(np.linalg.norm((d["pos_km"][row, i] - d["pos_km"][row, sun_i]) - truth))
            worst = max(worst, abs(e - doc["per_start"][str(start_year)][n][str(k)]))
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecast", action="store_true")
    ap.add_argument("--error-table", action="store_true")
    ap.add_argument("--since", help="UTC time the build started (ISO), required for --error-table")
    ap.add_argument("--verify-years", type=float, default=0.0, help="if the covered files changed since the build, rerun the current code and stamp only if it reproduces the stored result: --forecast reruns this many years of the table, --error-table the 2015 hindcast out to this many leads")
    ap.add_argument("--verify-tol-km", type=float, default=1e-4, help="largest position difference (km, any body, any stored sample) that still counts as reproducing the table")
    a = ap.parse_args()
    from sigma_ground.field.interface import reduced_model as rm
    now = datetime.datetime.now(datetime.timezone.utc)
    if a.forecast:
        f = FIX / "forecast_manifest.json"
        doc = json.loads(f.read_text(encoding="utf-8"))
        e = doc["tables"][-1]
        created = datetime.datetime.strptime(e["created_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
        start = created - datetime.timedelta(seconds=float(e["wall_seconds"]))
        why = physics_changed_since(start)
        if why and not a.verify_years:
            raise SystemExit("NOT STAMPED: " + why + "  (or pass --verify-years N to test that the current code reproduces the stored table)")
        if why:
            worst = reproduce(e, a.verify_years)
            if worst > a.verify_tol_km:
                raise SystemExit(f"NOT STAMPED: the current code differs from the stored table by up to {worst:.3g} km over {a.verify_years:g} yr (tolerance {a.verify_tol_km:g} km); rebuild the table")
            e["model_fingerprint_note"] = (f"stamped {now:%Y-%m-%dT%H:%M:%SZ} after the build. {why.split(':')[0]}, so the current code was RUN again from the same start state for {a.verify_years:g} yr "
                                           f"at the same step: it reproduces every stored sample (all bodies) to within {worst:.3g} km (tolerance {a.verify_tol_km:g} km)")
        else:
            e["model_fingerprint_note"] = f"stamped {now:%Y-%m-%dT%H:%M:%SZ} after the build; no commit or edit touched the covered physics files between the build's start ({start:%Y-%m-%dT%H:%M:%SZ}) and the stamp"
        e["model_fingerprint"] = rm.model_fingerprint(e["predictor"])
        f.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        print("stamped", e["file"], e["model_fingerprint"][:16])
    if a.error_table:
        if not a.since:
            raise SystemExit("--since is required for --error-table")
        start = datetime.datetime.fromisoformat(a.since.replace("Z", "+00:00"))
        why = physics_changed_since(start)
        if why and not a.verify_years:
            raise SystemExit("NOT STAMPED: " + why + "  (or pass --verify-years N to test that the current code reproduces the stored 2015 hindcast)")
        f = FIX / "prediction_error_table.json"
        doc = json.loads(f.read_text(encoding="utf-8"))
        if why:
            leads = int(a.verify_years)
            worst = reproduce_errors(doc, 2015, leads)
            if worst > a.verify_tol_km:
                raise SystemExit(f"NOT STAMPED: the current code differs from the stored hindcast error by up to {worst:.3g} km (tolerance {a.verify_tol_km:g} km); rebuild the table")
            doc["metadata"]["model_fingerprint_note"] = (f"stamped {now:%Y-%m-%dT%H:%M:%SZ} after the build. {why.split(':')[0]}, so the current code was RUN again for the hindcast that starts 2015-01-01, leads 1..{leads} yr: "
                                                         f"every body's error against DE440 equals the stored one to within {worst:.3g} km (tolerance {a.verify_tol_km:g} km)")
        else:
            doc["metadata"]["model_fingerprint_note"] = f"stamped {now:%Y-%m-%dT%H:%M:%SZ} after the build; the covered physics files were unchanged since {start:%Y-%m-%dT%H:%M:%SZ}"
        doc["metadata"]["model_fingerprint"] = rm.model_fingerprint(doc["metadata"]["predictor"])
        f.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        print("stamped the error table", doc["metadata"]["model_fingerprint"][:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
