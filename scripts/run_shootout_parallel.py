"""Run the N-body shoot-out in parallel: one process per (predictor, window), resumable, merged into the standard schema.

The shoot-out is embarrassingly parallel across windows and predictors (each window starts from a DE440 snapshot and each
predictor is independent), but ``rolling_shootout.run_rolling_shootout`` runs them serially. This driver spawns one worker
process per job, skips jobs whose output already exists (so an interrupted run resumes), and merges the per-job files into
``rolling_shootout_<pass>.json`` with the same metadata/samples layout the analysis code reads.

Passes (as in rolling_shootout's CLI):  short = 16 windows J2005..J2020, 5 yr horizon;  long = 7 windows J2000..J2006, 20 yr.
Predictors default to the canonical scoreboard set (headline stack, its one-layer-off ablations, Newton at the same step, the
honest Kepler forecast); ``--set all`` adds the diagnostics. Output goes to an untracked folder by default.

usage:
    python scripts/run_shootout_parallel.py --pass short --dt-days 0.1 --workers 12 [--predictors de440_stack ...]
    python scripts/run_shootout_parallel.py --pass short --merge-only        # just merge what exists
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

PASSES = {
    "short": ([f"j{y}" for y in range(2005, 2021)], 5.0),
    "long":  ([f"j{y}" for y in range(2000, 2007)], 20.0),
}
_DEFAULT_OUT = _ROOT / "local-cache" / "shootout_recalc"


def _pool(dt_days: float | None):
    import sigma_ground.field.interface.rolling_shootout as rs
    pool = {p.name: p for p in rs.PREDICTORS}
    if dt_days is not None:
        pool.update({p.name: p for p in rs.canonical_predictors(dt_days)})
    return rs, pool


def _worker(pred_name: str, key: str, horizon: float, dt_days: float | None, out: str, fixture: str | None,
            ckpt_dir: str | None = None) -> None:
    rs, pool = _pool(dt_days)
    pred = pool[pred_name]
    de440 = json.loads(Path(fixture).read_text()) if fixture and fixture != "none" else None
    rs.run_rolling_shootout([key], horizon, 1.0, predictors=[pred], de440=de440, verbose=False, output_path=out,
                            checkpoint_dir=None if not ckpt_dir or ckpt_dir == "none" else ckpt_dir)


def _job_file(outdir: Path, pass_name: str, pred: str, key: str) -> Path:
    return outdir / pass_name / f"{pred}__{key}.json"


def _cost_weight(pred: str, horizon: float) -> float:
    return horizon * (0.02 if pred.startswith("kepler") else 1.0)


def merge(outdir: Path, pass_name: str, pred_names: list[str], dt_days: float | None,
          label: str | None = None, fixture: str | None = None) -> Path | None:
    keys, horizon = PASSES[pass_name]
    samples, got, elapsed = [], [], 0.0
    # scan the whole pass folder (not just this invocation's predictors) so separate invocations accumulate
    for f in sorted((outdir / pass_name).glob("*__*.json")):
        pred, key = f.stem.split("__", 1)
        d = json.loads(f.read_text())
        samples.extend(d["samples"])
        elapsed += d["metadata"].get("elapsed_s", 0.0)
        got.append((pred, key))
    pred_names = sorted({g[0] for g in got}, key=lambda n: pred_names.index(n) if n in pred_names else 99)
    if not samples:
        return None
    payload = {
        "metadata": {
            "n_windows": len({k for _, k in got}),
            "window_start_keys": [k for k in keys if any(k == g[1] for g in got)],
            "prediction_horizon_yr": horizon,
            "sample_interval_yr": 1.0,
            "predictors": [p for p in pred_names if any(p == g[0] for g in got)],
            "dt_days": dt_days,
            "jobs_present": len(got),
            "jobs_expected_if_full_grid": len(pred_names) * len(keys),
            "elapsed_s_total_cpu": round(elapsed, 1),
            "baseline_label": label,
            "input_fixture": fixture,
            "generator": "scripts/run_shootout_parallel.py",
        },
        "samples": samples,
    }
    out = outdir / f"rolling_shootout_{pass_name}.json"
    out.write_text(json.dumps(payload, separators=(",", ":")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pass", dest="which", choices=["short", "long", "both"], default="short")
    ap.add_argument("--predictors", nargs="+", default=None)
    ap.add_argument("--set", dest="pred_set", choices=["canonical", "all"], default="canonical")
    ap.add_argument("--dt-days", type=float, default=None, help="step for the canonical predictors")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--outdir", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--thin-predictors", nargs="+", default=[], help="predictors run on --thin-windows only (e.g. the ablations)")
    ap.add_argument("--thin-windows", nargs="+", default=[], help="window keys used for --thin-predictors")
    ap.add_argument("--fixture", type=Path, default=None,
                    help="DE440 state-vector fixture to use (default: the original tracked one). Past data is never overwritten: "
                         "recalculations read a *_v2physics fixture and write to their own labelled folder.")
    ap.add_argument("--label", default=None, help="baseline label stored in the merged metadata, e.g. v2physics_2026-10-10")
    ap.add_argument("--worker", nargs=7, metavar=("PRED", "KEY", "HORIZON", "DT", "OUT", "FIXTURE", "CKPT"), help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.worker:
        pred, key, horizon, dt, out, fixture, ckpt = a.worker
        _worker(pred, key, float(horizon), None if dt == "none" else float(dt), out, fixture, ckpt)
        return 0

    rs, pool = _pool(a.dt_days)
    if a.predictors:
        names = a.predictors
    elif a.pred_set == "canonical":
        names = [p.name for p in rs.CANONICAL_PREDICTORS]
    else:
        names = [p.name for p in rs.PREDICTORS]
    unknown = [n for n in names if n not in pool]
    if unknown:
        print("unknown predictors:", unknown)
        return 2
    passes = ["short", "long"] if a.which == "both" else [a.which]
    a.outdir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for ps in passes:
        keys, horizon = PASSES[ps]
        (a.outdir / ps).mkdir(parents=True, exist_ok=True)
        for pred in names:
            for key in keys:
                if pred in a.thin_predictors and a.thin_windows and key not in a.thin_windows:
                    continue
                f = _job_file(a.outdir, ps, pred, key)
                if not f.exists():
                    jobs.append((ps, pred, key, horizon, f))
    jobs.sort(key=lambda j: -_cost_weight(j[1], j[3]))
    if not a.merge_only:
        print(f"{len(jobs)} jobs to run, {a.workers} workers, dt = {a.dt_days or 'predictor default'} d, out = {a.outdir}", flush=True)
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        running: list[tuple[subprocess.Popen, tuple, float]] = []
        horizon_of = {(j[0], j[2]): j[3] for j in jobs}
        pending, done, failed, t0 = list(jobs), 0, [], time.time()
        attempts: dict[tuple, int] = {}
        MAX_ATTEMPTS = 3
        while pending or running:
            while pending and len(running) < a.workers:
                ps, pred, key, horizon, f = pending.pop(0)
                tmp = f.with_suffix(".part")
                cmd = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker", pred, key, str(horizon),
                       "none" if a.dt_days is None else str(a.dt_days), str(tmp), str(a.fixture) if a.fixture else "none",
                       str(a.outdir / "ckpt")]
                attempts[(ps, pred, key)] = attempts.get((ps, pred, key), 0) + 1
                running.append((subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                                                 stderr=open(f.with_suffix(".log"), "a")), (ps, pred, key, f), time.time()))
            time.sleep(5)
            for item in list(running):
                proc, (ps, pred, key, f), t1 = item
                if proc.poll() is None:
                    continue
                running.remove(item)
                tmp = f.with_suffix(".part")
                if proc.returncode == 0 and tmp.exists():
                    tmp.replace(f)
                    (a.outdir / "ckpt" / f"{pred}__{key}.npz").unlink(missing_ok=True)
                    done += 1
                    print(f"[{time.time() - t0:7.0f}s] done {ps}/{pred}/{key} in {time.time() - t1:.0f}s  ({done}/{len(jobs)})", flush=True)
                else:
                    n_try = attempts.get((ps, pred, key), 1)
                    print(f"[{time.time() - t0:7.0f}s] FAILED {ps}/{pred}/{key} exit code {proc.returncode} after "
                          f"{time.time() - t1:.0f}s, attempt {n_try}/{MAX_ATTEMPTS} (see {f.with_suffix('.log')})", flush=True)
                    if n_try < MAX_ATTEMPTS:
                        pending.insert(0, (ps, pred, key, horizon_of[(ps, key)], f))   # resumes from its checkpoint
                    else:
                        failed.append((ps, pred, key))
        print(f"finished: {done} ok, {len(failed)} failed, {time.time() - t0:.0f}s wall", flush=True)
    for ps in passes:
        out = merge(a.outdir, ps, names, a.dt_days, a.label, str(a.fixture) if a.fixture else None)
        print("merged ->", out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
