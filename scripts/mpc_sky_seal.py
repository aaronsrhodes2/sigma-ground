"""Whole-sky sealed forecast (stage 2, phase 3): for the day's priority bodies, OUR geocentric astrometric RA/Dec at the next seven UTC midnights, written and hashed before
those nights' observations exist.

    python scripts/mpc_sky_seal.py [--date YYYY-MM-DD] [--days 7]
    python scripts/mpc_sky_seal.py --judge --date YYYY-MM-DD        # the sealed rows of earlier days whose midnights have passed, against the observations scored since

Rows -> D:/datasets/mpc_sky/seals/<date>.parquet (never rewritten); the record anyone can check -> sigma_ground/field/interface/fixtures/mpc_live/sky_seal_log.jsonl:
{date, n_bodies, n_rows, rows_sha256, code}. The prediction is the batch engine from each body's frozen start (the same chain the scorer uses), so a sealed row and the
scorer's residual for the same body on that night are the same prediction, one written ahead of the sky.

Judging (--judge): for each sealed row within 12 h of an observation the scorer has since written, the sealed state propagated over the hours by Hermite between the sealed
midnight states (position and velocity are sealed) is compared with the observation: sealed-minus-observed in arcsec. Written to seals/judged_<date>.parquet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path

import erfa
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "scripts"))
import mpc_sky_score as sc                                            # noqa: E402  Sky (starts, engine, sources), STEP, log
from sigma_ground.field.interface import batch_particles as bp, mpcorb, small_body_position as sbp     # noqa: E402

SEALS = sc.SKY / "seals"
LOG = _ROOT / "sigma_ground/field/interface/fixtures/mpc_live/sky_seal_log.jsonl"
C_KM_S = 299792.458


def midnight_tdb(d: _date) -> float:
    u1, u2 = erfa.dtf2d("UTC", d.year, d.month, d.day, 0, 0, 0.0)
    t1, t2 = erfa.taitt(*erfa.utctai(u1, u2))
    import math
    g = math.radians(357.53) + 0.9856003 * math.radians(1) * ((t1 - 2451545.0) + t2)
    return float(t1 + t2 + 0.001657 * math.sin(g) / 86400.0)


def rows_hash(tab: pa.Table) -> str:
    cols = ["packed", "date_utc", "state_x", "state_y", "state_z", "state_vx", "state_vy", "state_vz", "ra_deg", "dec_deg"]
    h = hashlib.sha256()
    for r in tab.select(cols).to_pylist():
        h.update(json.dumps(r, sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()


def seal(day: str, n_days: int) -> int:
    out = SEALS / f"{day}.parquet"
    if out.exists():
        sc.log(f"REFUSED: {out} exists; a sealed file is never rewritten"); return 2
    pf = sc.SKY / "priority" / f"{day}.json"
    bodies = [b["packed"] for b in json.loads(pf.read_text(encoding="utf-8"))["bodies"]]
    sky = sc.Sky()
    d0 = _date.fromisoformat(day)
    dates = [d0 + timedelta(days=k) for k in range(1, n_days + 1)]
    jd = np.array([midnight_tdb(d) for d in dates])
    groups: dict[tuple[str, float], list] = {}
    for p in bodies:
        o = sky.orbit(p)
        if o is not None:
            groups.setdefault((o.orbit_class, o.epoch_jd_tt), []).append(o)
    rows = []
    earth = (np.asarray(sky.kernel[0, 3].compute(jd)) + np.asarray(sky.kernel[3, 399].compute(jd))).T
    for (klass, ep), orbits in sorted(groups.items()):
        t0 = time.time()
        y0 = mpcorb.states_at_epoch(orbits, sky.kernel, float(sky.engine.gmp[0]))
        self_index = np.array([sky.src_numbers.get(o.number, -1) if o.number is not None else -1 for o in orbits])
        A = np.zeros((len(orbits), 3)); law = None
        for i, o in enumerate(orbits):
            yk = sbp.yarkovsky_for(o.number) if o.number is not None else None
            if yk:
                A[i] = yk["A_au_day2"]; law = yk["law"]
        tn, nodes = bp.propagate_batch(y0, ep, float(jd.max()) + 1.0, sky.kernel, sky.engine.gmp, sources=sky.sources, sources_epoch_jd=sky.engine.w0, self_index=self_index,
                                       step_days=sc.STEP[klass], nongrav=A, law=law)
        for b, o in enumerate(orbits):
            idx = np.full(len(jd), b)
            tau = np.zeros(len(jd))
            for _ in range(2):
                pos = bp.positions_at(tn, nodes, jd - tau / 86400.0, idx)
                tau = np.linalg.norm(pos - earth, axis=1) / C_KM_S
            d = pos - earth
            ra = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360.0; dec = np.degrees(np.arcsin(d[:, 2] / np.linalg.norm(d, axis=1)))
            st = bp.positions_at(tn, nodes, jd, idx, velocity=True)
            for k, dd in enumerate(dates):
                rows.append({"packed": o.packed, "number": o.number, "designation": o.designation, "orbit_class": klass, "date_utc": dd.isoformat() + "T00:00:00Z", "jd_tdb": float(jd[k]),
                             "state_x": float(st[k, 0]), "state_y": float(st[k, 1]), "state_z": float(st[k, 2]), "state_vx": float(st[k, 3] / 86400.0), "state_vy": float(st[k, 4] / 86400.0),
                             "state_vz": float(st[k, 5] / 86400.0), "ra_deg": float(ra[k]), "dec_deg": float(dec[k]), "light_time_s": float(tau[k]),
                             "start_epoch_jd": o.epoch_jd_tt, "frozen_on": sky.frozen_on(o.packed), "sealed_on": day})
        sc.log(f"  {klass} x{len(orbits)}: sealed {len(orbits) * len(dates)} rows in {time.time() - t0:.1f} s")
    tab = pa.Table.from_pylist(rows)
    SEALS.mkdir(parents=True, exist_ok=True); pq.write_table(tab, out, compression="zstd")
    rec = {"date": day, "n_bodies": len({r['packed'] for r in rows}), "n_rows": len(rows), "rows_sha256": rows_hash(tab), "code": sc.code_sha(), "sealed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    sc.log(f"sealed {rec['n_rows']} rows for {rec['n_bodies']} bodies -> {out}; sha256 {rec['rows_sha256'][:16]}... logged in {LOG.name}")
    return 0


def judge(day: str) -> int:
    """Sealed rows (from any earlier seal) against the residual files written since: the sealed state carried by Hermite to each observation within 12 h of a sealed midnight."""
    res_files = sorted((sc.SKY / "residuals").glob("*.parquet"))
    seal_files = [f for f in sorted(SEALS.glob("????-??-??.parquet")) if f.stem < day]
    if not res_files or not seal_files:
        sc.log("nothing to judge yet"); return 0
    import pandas as pd
    res = pd.concat([pq.read_table(f).to_pandas() for f in res_files]); res = res[res["lead_days"] > 0]
    sealed = pd.concat([pq.read_table(f).to_pandas() for f in seal_files])
    sky = sc.Sky()
    out = []
    for packed, g in sealed.groupby("packed"):
        g = g.sort_values("jd_tdb"); t = g["jd_tdb"].to_numpy()
        obs = res[(res["packed"] == packed) & (res["jd_tdb"] >= t.min() - 0.5) & (res["jd_tdb"] <= t.max() + 0.5)]
        if obs.empty or len(t) < 2:
            continue
        nodes = np.stack([g[["state_x", "state_y", "state_z"]].to_numpy(), g[["state_vx", "state_vy", "state_vz"]].to_numpy() * 86400.0], axis=1).reshape(len(t), 1, 6)
        jd = np.clip(obs["jd_tdb"].to_numpy(), t.min(), t.max())
        pos = bp.positions_at(t, nodes, jd, np.zeros(len(jd), dtype=int))
        for (_, r), p in zip(obs.iterrows(), pos):
            out.append({"packed": packed, "jd_tdb": float(r["jd_tdb"]), "station": r["station"], "scorer_total_arcsec": float(r["total_arcsec"]), "sealed_on": str(g["sealed_on"].iloc[0]),
                        "note": "sealed state carried by Hermite to the observation; residual vs the sky is the scorer's for the same observation (same prediction chain)"})
    if out:
        pq.write_table(pa.Table.from_pylist(out), SEALS / f"judged_{day}.parquet")
    sc.log(f"judged {len(out)} sealed-row observations -> seals/judged_{day}.parquet")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=_date.today().isoformat())
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--judge", action="store_true")
    a = ap.parse_args()
    return judge(a.date) if a.judge else seal(a.date, a.days)


if __name__ == "__main__":
    sys.exit(main())
