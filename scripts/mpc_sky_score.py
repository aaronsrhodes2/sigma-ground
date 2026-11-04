"""Whole-sky scorer (stage 2, phase 2; misc/CANDIDATE_LEDGER.md C36): every observation in hand of every body with a frozen start, against OUR batch-engine prediction from that start.

    python scripts/mpc_sky_score.py --night YYYY-MM-DD [--source api|bulk] [--bodies PACKED ...] [--max-bodies N] [--class nea|mba|outer]

For each body: the frozen start (starts.parquet: the MPC elements of the day the body was first seen, never replaced) becomes a barycentric state at its epoch
(mpcorb.states_at_epoch); the bodies of one step group (nea 0.25 d, mba/outer 1 d) are carried together by batch_particles.propagate_batch, backward and forward from the
epoch to cover their observations, with the Sun's 1PN term, the 343 prescribed sources and the C31 non-gravitational term where one exists; positions at each observation
come from the recorded nodes by Hermite with two light-time iterations; the residual is O - C in arcsec (dRA cos dec, dDec) against the station's barycentric position
(eros_sky_residuals.Obs). A body inside 0.02 au of the Earth at any node is left to the per-body scorer (flagged, not scored here).

Rows (one per observation) -> D:/datasets/mpc_sky/residuals/<night>.parquet; one row per body per night -> sky_scoreboard.parquet (append-only); the watermark per body (last
observation time scored) in sky_state.json. lead_days = observation time minus the day the start was frozen: a row with lead_days > 0 is a forward prediction, a row at or
before it is the fit's own residual (the baseline, kept and labelled). Observations are taken from the day's API files (--source api, the priority set) or the archive parquet
(--source bulk).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import date as _date, datetime, timezone
from pathlib import Path

import erfa
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "scripts"))
import eros_sky_residuals as esr                                      # noqa: E402
import mpc_live_score as m                                            # noqa: E402  the engine's kernel, GM, perturber nodes, W0
from sigma_ground.field.interface import batch_particles as bp, mpcorb, obs80, small_body_position as sbp     # noqa: E402
from sigma_ground.field.interface.prescribed_sources import PrescribedSources                                   # noqa: E402

SKY = Path("D:/datasets/mpc_sky")
ARC = 206264.80624709636
STEP = {"nea": 0.25, "mba": 1.0, "outer": 1.0}
CLOSE_AU = 0.02
BACKFILL_DAYS = 400.0                                                 # a body's first scoring reaches back this far before its start epoch: the fit's own residual as the baseline, no further
AU_KM = 149597870.7
DAY = 86400.0


def log(s: str):
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {s}", flush=True)


def code_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                                 # noqa: BLE001
        return "unknown"


def jd_utc_to_tdb(y, mo, d):
    """Arrays -> JD TDB, the scoreboard's conventions (eros_sky_residuals.Obs)."""
    day = np.floor(d); u1 = np.empty(len(y)); u2 = np.empty(len(y))
    for i in range(len(y)):
        u1[i], u2[i] = erfa.dtf2d("UTC", int(y[i]), int(mo[i]), int(day[i]), 0, 0, 0.0)
    u2 = u2 + (d - day)
    t1, t2 = erfa.taitt(*erfa.utctai(u1, u2))
    g = math.radians(357.53) + 0.9856003 * np.radians(1) * ((t1 - 2451545.0) + t2)
    return t1 + t2 + 0.001657 * np.sin(g) / DAY, (u1, u2)


class Sky:
    def __init__(self):
        self.engine = m.Engine(log)
        t = self.engine.pp.to_prescribed_sources(self.engine.nodes, self.engine.gm_all, 1.0)
        self.sources = PrescribedSources(t.gm, t.t0, t.h, t.pos, t.vel)
        self.src_numbers = {int(n): i for i, n in enumerate(self.engine.numbers)}
        self.codes = esr.parse_obscodes(m.MPC_OLD / "ObsCodes.html")
        self.kernel = self.engine.kernel
        self.starts = pq.read_table(SKY / "starts.parquet").to_pandas().set_index("packed")
        self.window = (m.W0, m.W0 + m.HORIZON_DAYS - 2.0)              # the prescribed-source table's reach

    def orbit(self, packed: str) -> mpcorb.Orbit | None:
        if packed not in self.starts.index:
            return None
        r = self.starts.loc[packed]
        return mpcorb.Orbit(packed=packed, number=None if r["number"] != r["number"] else int(r["number"]), designation=r["designation"], h_mag=r["h_mag"], g_slope=r["g_slope"],
                            epoch_jd_tt=float(r["epoch_jd_tt"]), ma_deg=float(r["ma_deg"]), w_deg=float(r["w_deg"]), om_deg=float(r["om_deg"]), i_deg=float(r["i_deg"]),
                            e=float(r["e"]), n_deg_day=float(r["n_deg_day"]), a_au=float(r["a_au"]), u_param=str(r["u_param"]), reference=str(r["reference"]),
                            n_obs=None, n_opp=None, arc=str(r["arc"]), rms_arcsec=None, flags_hex=str(r["flags_hex"]), readable=str(r["readable"]), last_obs_yyyymmdd=str(r["last_obs_yyyymmdd"]))

    def frozen_on(self, packed: str) -> str:
        return str(self.starts.loc[packed]["frozen_on"])


def load_api_obs(night: str, packed_list: list[str] | None) -> dict[str, list]:
    """packed -> optical rows from the API files of the day (and earlier days' files when the day has none for a body)."""
    out = {}
    days = sorted((SKY / "obs_api").glob("*"))
    for d in reversed(days):
        if d.name > night:
            continue
        for f in d.glob("*.txt"):
            p = f.stem.replace("_", " ") if " " in f.stem else f.stem
            if (packed_list and p not in packed_list) or p in out:
                continue
            rows = [r for r in (obs80.parse_line(ln) for ln in f.read_text(encoding="utf-8", errors="replace").splitlines()) if r is not None]
            if rows:
                out[p] = rows
    return out


def load_bulk_obs(batch: str, packed_list: list[str] | None) -> dict[str, list]:
    tabs = [pq.read_table(f) for f in sorted((SKY / "obs" / batch).glob("*_part*.parquet"))]
    if not tabs:
        return {}
    t = pa.concat_tables(tabs).to_pandas()
    if packed_list:
        t = t[t["packed"].isin(packed_list)]
    out = {}
    for p, g in t.groupby("packed"):
        out[p] = list(zip(g["packed_number"], g["packed_prov"], g["note2"], g["year"], g["month"], g["day"], g["ra_rad"], g["dec_rad"], g["mag"], g["band"], g["station"]))
    return out


def sky_track_components(sky: Sky, t_nodes, nodes, b: int, obs, pos, res) -> tuple[np.ndarray, np.ndarray]:
    """The sky residual split along and across the body's apparent motion (O - C projected on the unit direction of d(RA cos dec, Dec)/dt as seen from the observer).

    Along-track is where an along-orbit timing error (a mean-motion offset, a transverse acceleration) shows; cross-track is where a plane or start error shows. The direction
    comes from the predicted state (Hermite velocity) and the observer's barycentric velocity (the Earth's, the station's rotation neglected: it changes the direction by under
    a part in a thousand for a body moving faster than 10 arcsec per hour)."""
    st = bp.positions_at(t_nodes, nodes, obs.jd_tdb, np.full(obs.n, b), velocity=True)
    ev = np.asarray(sky.kernel[0, 3].compute_and_differentiate(obs.jd_tdb)[1]).T + np.asarray(sky.kernel[3, 399].compute_and_differentiate(obs.jd_tdb)[1]).T   # km/day
    d = pos - obs.x_obs; v = st[:, 3:] - ev
    rn = np.linalg.norm(d, axis=1); east = np.stack([-d[:, 1], d[:, 0], np.zeros(len(d))], axis=1); east /= np.linalg.norm(east, axis=1)[:, None]
    north = np.cross(d / rn[:, None], east)
    mu = np.stack([np.einsum("ni,ni->n", v, east), np.einsum("ni,ni->n", v, north)], axis=1)         # sky-plane motion (east, north) in km/day
    mn = np.linalg.norm(mu, axis=1); ok = mn > 0
    u = np.zeros_like(mu); u[ok] = mu[ok] / mn[ok][:, None]
    along = res[:, 0] * u[:, 0] + res[:, 1] * u[:, 1]
    cross = -res[:, 0] * u[:, 1] + res[:, 1] * u[:, 0]
    return along, cross


def score_group(sky: Sky, group: list[tuple[str, mpcorb.Orbit, list]], step: float, night: str, state: dict, code: str) -> tuple[list[dict], list[dict], list[str]]:
    """One step group: all bodies' observations newer than their watermark, scored from their frozen starts. Returns (observation rows, body rows, flagged packed)."""
    bodies, orbits, obs_rows = [], [], []
    for packed, o, rows in group:
        wm = state.get(packed, {}).get("watermark_jd", -1e9)
        rows = [r for r in rows if r[10] in sky.codes]
        if not rows:
            continue
        y = np.array([r[3] for r in rows]); mo = np.array([r[4] for r in rows]); d = np.array([r[5] for r in rows])
        jd, _ = jd_utc_to_tdb(y, mo, d)
        keep = (jd > wm) & (jd > o.epoch_jd_tt - BACKFILL_DAYS)
        if not keep.any():
            continue
        bodies.append(packed); orbits.append(o); obs_rows.append(([rows[i] for i in np.nonzero(keep)[0]], jd[keep]))
    if not bodies:
        return [], [], []
    y0 = mpcorb.states_at_epoch(orbits, sky.kernel, float(sky.engine.gmp[0]))
    epochs = np.array([o.epoch_jd_tt for o in orbits])
    if np.ptp(epochs) > 1e-6:                                            # one epoch per group here (MPCORB uses a common epoch); a mixed group is split by the caller
        raise ValueError("score_group: mixed epochs")
    jd0 = float(epochs[0])
    lo = min(float(j.min()) for _, j in obs_rows) - 1.0; hi = max(float(j.max()) for _, j in obs_rows) + 1.0
    lo, hi = max(lo, sky.window[0] + 1.0), min(hi, sky.window[1] - 1.0)
    self_index = np.array([sky.src_numbers.get(o.number, -1) if o.number is not None else -1 for o in orbits])
    A = np.zeros((len(orbits), 3)); law = None
    for i, o in enumerate(orbits):
        yk = sbp.yarkovsky_for(o.number) if o.number is not None else None
        if yk:
            A[i] = yk["A_au_day2"]; law = yk["law"]
    t0 = time.time()
    parts = []
    if hi > jd0:
        parts.append(bp.propagate_batch(y0, jd0, hi, sky.kernel, sky.engine.gmp, sources=sky.sources, sources_epoch_jd=sky.engine.w0, self_index=self_index, step_days=step, nongrav=A, law=law))
    if lo < jd0:
        tb, nb = bp.propagate_batch(y0, jd0, lo, sky.kernel, sky.engine.gmp, sources=sky.sources, sources_epoch_jd=sky.engine.w0, self_index=self_index, step_days=step, nongrav=A, law=law)
        parts.append((tb[::-1], nb[::-1]))
    t_nodes = np.concatenate([p[0] for p in sorted(parts, key=lambda p: p[0][0])]); nodes = np.concatenate([p[1] for p in sorted(parts, key=lambda p: p[0][0])])
    keep = np.concatenate([[True], np.diff(t_nodes) > 1e-9]); t_nodes, nodes = t_nodes[keep], nodes[keep]
    dt_prop = time.time() - t0
    # the Earth's barycentric path at the nodes: a body within CLOSE_AU at any node is flagged for the per-body path
    earth = (np.asarray(sky.kernel[0, 3].compute(t_nodes)) + np.asarray(sky.kernel[3, 399].compute(t_nodes))).T
    dmin = np.min(np.linalg.norm(nodes[:, :, :3] - earth[:, None, :], axis=2), axis=0) / AU_KM
    flagged = [bodies[i] for i in np.nonzero(dmin < CLOSE_AU)[0]]
    rows_out, body_out = [], []
    for b, (packed, o, (rows, jd)) in enumerate(zip(bodies, orbits, obs_rows)):
        if packed in flagged:
            continue
        obs = esr.Obs([(r[3], r[4], r[5], r[6], r[7], r[10]) for r in rows], sky.codes, sky.kernel)
        pos, tau = bp.light_time_positions(t_nodes, nodes, obs.jd_tdb, np.full(obs.n, b), obs.x_obs)
        res = esr.radec_residuals(obs, pos)
        tot = np.hypot(res[:, 0], res[:, 1])
        along, cross = sky_track_components(sky, t_nodes, nodes, b, obs, pos, res)
        frozen = sky.frozen_on(packed)
        fy, fm, fd = map(int, frozen.split("-")); jd_frozen = float(sum(erfa.dtf2d("UTC", fy, fm, fd, 0, 0, 0.0)))
        lead = obs.jd_tdb - jd_frozen
        for i in range(obs.n):
            rows_out.append({"night": night, "packed": packed, "number": o.number, "designation": o.designation, "orbit_class": o.orbit_class, "jd_tdb": float(obs.jd_tdb[i]),
                             "station": obs.code[i], "dra_cosdec_arcsec": float(res[i, 0]), "ddec_arcsec": float(res[i, 1]), "total_arcsec": float(tot[i]),
                             "along_arcsec": float(along[i]), "cross_arcsec": float(cross[i]),
                             "lead_days": float(lead[i]), "light_time_s": float(tau[i]), "mag": rows[i][8], "code": code, "start_epoch_jd": o.epoch_jd_tt, "frozen_on": frozen})
        fwd = lead > 0
        body_out.append({"night": night, "packed": packed, "number": o.number, "designation": o.designation, "orbit_class": o.orbit_class, "n_obs": int(obs.n),
                         "n_forward": int(fwd.sum()), "median_arcsec": float(np.median(tot)), "rms_arcsec": float(np.sqrt(np.mean(tot ** 2))),
                         "median_forward_arcsec": float(np.median(tot[fwd])) if fwd.any() else None, "mean_dra_cosdec": float(res[:, 0].mean()), "mean_ddec": float(res[:, 1].mean()),
                         "mean_along_arcsec": float(along.mean()), "mean_cross_arcsec": float(cross.mean()), "sem_along_arcsec": float(along.std(ddof=1) / math.sqrt(obs.n)) if obs.n > 1 else None,
                         "lead_days_max": float(lead.max()), "first_jd": float(obs.jd_tdb.min()), "last_jd": float(obs.jd_tdb.max()), "step_days": step, "code": code,
                         "start_epoch_jd": o.epoch_jd_tt, "frozen_on": frozen, "h_mag": o.h_mag})
        state[packed] = {"watermark_jd": float(obs.jd_tdb.max()), "night": night}
    tot_all = np.array([r["total_arcsec"] for r in rows_out]); fwd_all = np.array([r["lead_days"] > 0 for r in rows_out])
    med = f"median {np.median(tot_all):.3f} arcsec ({int(fwd_all.sum())} forward of {len(tot_all)})" if len(tot_all) else "no rows"
    log(f"  group {orbits[0].orbit_class} x{len(bodies)}: propagated {t_nodes[0]:.1f}..{t_nodes[-1]:.1f} in {dt_prop:.1f} s; {len(rows_out):,} observations scored, {med}, {len(flagged)} flagged close")
    return rows_out, body_out, flagged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--night", default=_date.today().isoformat())
    ap.add_argument("--source", choices=["api", "bulk"], default="api")
    ap.add_argument("--batch", default=None, help="--source bulk: the archive batch directory under D:/datasets/mpc_sky/obs")
    ap.add_argument("--bodies", nargs="*", default=None, help="packed designations")
    ap.add_argument("--max-bodies", type=int, default=None)
    ap.add_argument("--class", dest="klass", choices=["nea", "mba", "outer"], default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sky = Sky()
    code = code_sha()
    obs = load_api_obs(a.night, a.bodies) if a.source == "api" else load_bulk_obs(a.batch, a.bodies)
    log(f"{len(obs):,} bodies with observation files ({a.source})")
    state_f = SKY / "sky_state.json"
    state = json.loads(state_f.read_text(encoding="utf-8")) if state_f.exists() else {}
    groups: dict[tuple[str, float], list] = {}
    missing = 0
    for packed, rows in obs.items():
        o = sky.orbit(packed)
        if o is None:
            missing += 1; continue
        if a.klass and o.orbit_class != a.klass:
            continue
        groups.setdefault((o.orbit_class, o.epoch_jd_tt), []).append((packed, o, rows))
    if a.max_bodies:
        for k in groups:
            groups[k] = groups[k][: a.max_bodies]
    log(f"{sum(len(g) for g in groups.values()):,} bodies in {len(groups)} step/epoch groups; {missing} without a frozen start")
    all_rows, all_bodies, flagged = [], [], []
    for (klass, ep), g in sorted(groups.items()):
        r, b, f = score_group(sky, g, STEP[klass], a.night, state, code)
        all_rows += r; all_bodies += b; flagged += f
    if a.dry_run:
        log(f"DRY RUN: {len(all_rows):,} rows, {len(all_bodies)} bodies, flagged {flagged}"); return 0
    out = SKY / "residuals"; out.mkdir(parents=True, exist_ok=True)
    f = out / f"{a.night}.parquet"
    if all_rows:
        tab = pa.Table.from_pylist(all_rows)
        if f.exists():
            tab = pa.concat_tables([pq.read_table(f), tab], promote_options="default")
        pq.write_table(tab, f, compression="zstd")
    sb = SKY / "sky_scoreboard.parquet"
    if all_bodies:
        tab = pa.Table.from_pylist(all_bodies)
        if sb.exists():
            tab = pa.concat_tables([pq.read_table(sb), tab], promote_options="default")
        tmp = sb.with_suffix(".tmp.parquet"); pq.write_table(tab, tmp); tmp.replace(sb)
    state_f.write_text(json.dumps(state, indent=0), encoding="utf-8")
    (SKY / "flagged_close.jsonl").open("a", encoding="utf-8").write("".join(json.dumps({"night": a.night, "packed": p}) + "\n" for p in flagged))
    log(f"finished: {len(all_rows):,} observations of {len(all_bodies)} bodies -> {f}; scoreboard {sb}; flagged close {len(flagged)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
