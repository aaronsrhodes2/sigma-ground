"""Sealed forward predictions for the tracked small bodies (misc/CANDIDATE_LEDGER.md C32).

The daily scoreboard (scripts/mpc_live_score.py) scores a prediction only after the sky has been observed, by re-running the engine. This writes the prediction FIRST:
for every tracked body, from the same fixed start state the scoreboard ages, the engine's barycentric state and the geocentric astrometric RA/Dec at fixed future
instants, hashed and committed before any of those instants arrive. Anyone can later compare the sealed numbers with what the sky and JPL's final solutions say.

    python scripts/mpc_forecast_seal.py --seal                      # write the sealed file (refuses to overwrite one)
    python scripts/mpc_forecast_seal.py --check 433 99942           # recompute these bodies and compare with the sealed rows (reproducibility, km and arcsec)
    python scripts/mpc_forecast_seal.py --seal-missing              # new roster bodies (C33) into today's file; the daily driver runs this
    python scripts/mpc_forecast_seal.py --judge                     # rows whose instant has passed: fetch JPL's geocentric astrometric RA/Dec from Horizons and compare

What a row is: `state_bary_km_kms` is the engine's barycentric ICRF position and velocity at the TDB instant; `ra_deg`/`dec_deg` are the geocentric astrometric direction
(light-time from our own path, geometric from the Earth's centre, ICRF, no aberration), which is what Horizons OBSERVER quantity 1 at centre 500 reports. The scoreboard
tests against topocentric MPC observations; a sealed geocentric row is matched to them by propagating the sealed state over the hours to the observation, which adds nothing
at this level. The sealed file is never edited: a later seal is a new file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import erfa
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
import mpc_live_score as m                                            # noqa: E402  the scoreboard's engine, starts and Horizons helpers
import eros_sky_residuals as esr                                      # noqa: E402

SEAL_DIR = m.FIX / "mpc_live"
DATES_ALL = ["2027-01-01", "2028-01-01", "2029-01-01", "2030-01-01", "2031-01-01"]         # 1 to 5 years after the heavies' 2026-01-01 start; Eros 7 to 11
DATES_EXTRA = {99942: ["2029-04-13", "2029-04-14"]}                                        # Apophis's approach (closest 2029-04-13 ~21:46 UTC): the test of C31's Yarkovsky term
ARC = esr.ARC if hasattr(esr, "ARC") else 180.0 / math.pi * 3600.0


def roster_dates() -> dict[int, list[str]]:
    """Near-miss roster bodies (C33): the UTC date of each body's close approach and the day after, when the astrometry is densest."""
    f = SEAL_DIR / "neo_roster.json"
    if not f.exists():
        return {}
    out: dict[int, list[str]] = {}
    for b in json.loads(f.read_text(encoding="utf-8"))["bodies"]:
        d = datetime.strptime(b["approach_date"][:11], "%Y-%b-%d").date()
        out.setdefault(b["number"], []).extend([d.isoformat(), (d + timedelta(days=1)).isoformat()])
    return out


def dates_for(number: int) -> list[str]:
    """The instants sealed for a body: the January set, then any extra dates, in order, no repeats."""
    out = []
    for d in DATES_ALL + DATES_EXTRA.get(number, []) + roster_dates().get(number, []):
        if d not in out:
            out.append(d)
    return out


def free_seal_path(day: str) -> Path:
    """Today's sealed file name, or the next free one when the day already has a seal (a sealed file is never rewritten)."""
    p = SEAL_DIR / f"forecast_sealed_{day}.json"
    k = 2
    while p.exists():
        p = SEAL_DIR / f"forecast_sealed_{day}_{k}.json"; k += 1
    return p


def tdb_of_utc_midnight(iso: str) -> float:
    """JD TDB of 00:00:00 UTC on the date, the scoreboard's conventions (erfa UTC -> TAI -> TT, 1.7 ms periodic TDB term)."""
    y, mo, d = map(int, iso.split("-"))
    u1, u2 = erfa.dtf2d("UTC", y, mo, d, 0, 0, 0.0)
    t1, t2 = erfa.taitt(*erfa.utctai(u1, u2))
    g = math.radians(357.53) + 0.9856003 * math.radians(1) * ((t1 - 2451545.0) + t2)
    return float(t1 + t2 + 0.001657 * math.sin(g) / m.DAY)


def states(engine: m.Engine, number: int, jd0: float, y0: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """(N,6) barycentric state km, km/day at `targets` from the scoreboard engine, Yarkovsky included where the scorer includes it."""
    from sigma_ground.field.interface import comet_orbits as co
    yk = m.yarkovsky(number)
    extra = {"nongrav": list(yk["A_au_day2"]), "law": yk["law"]} if yk else {}
    out = co.propagate(list(y0), jd0, [float(t) for t in targets], engine.kernel, engine.gmp, gr=True, sources=engine.sources(number), sources_epoch_jd=engine.w0, **extra)
    return np.array([out[float(t)] for t in targets])


def geocentric_radec(engine: m.Engine, number: int, jd0: float, y0: np.ndarray, jd_tdb: np.ndarray, iters: int = 3):
    """Geocentric astrometric RA/Dec (deg, ICRF) at the TDB instants, light time from our own path; also the states at the instants themselves and the light times (s)."""
    earth = (np.asarray(engine.kernel[0, 3].compute(jd_tdb)) + np.asarray(engine.kernel[3, 399].compute(jd_tdb))).T
    tau = np.zeros(len(jd_tdb))
    for _ in range(iters):
        pos = states(engine, number, jd0, y0, jd_tdb - tau / m.DAY)[:, :3]
        tau = np.linalg.norm(pos - earth, axis=1) / esr.C_KM_S
    d = pos - earth
    ra = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360.0
    dec = np.degrees(np.arcsin(d[:, 2] / np.linalg.norm(d, axis=1)))
    at = states(engine, number, jd0, y0, jd_tdb)
    return ra, dec, tau, at, np.linalg.norm(d, axis=1)


def code_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def rows_hash(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def forecast_rows(number: int, name: str, engine: m.Engine, start, dates: list[str], log) -> list[dict]:
    jd0, y0, start_src = start
    jd = np.array([tdb_of_utc_midnight(d) for d in dates])
    t0 = time.time()
    ra, dec, tau, at, dist = geocentric_radec(engine, number, jd0, y0, jd)
    sun = np.asarray(engine.kernel[0, 10].compute(jd)).T
    log(f"  {number} {name}: {len(dates)} instants in {time.time() - t0:.0f} s")
    out = []
    for i, d in enumerate(dates):
        out.append({"body": number, "name": name, "date_utc": d + "T00:00:00Z", "jd_tdb": float(jd[i]), "lead_days": float(jd[i] - jd0),
                    "start_epoch_jd_tdb": jd0, "start_source": start_src,
                    "state_bary_km_kms": [float(x) for x in at[i, :3]] + [float(x / m.DAY) for x in at[i, 3:]],
                    "ra_deg": float(ra[i]), "dec_deg": float(dec[i]), "light_time_s": float(tau[i]), "geocentric_distance_km": float(dist[i]),
                    "heliocentric_distance_km": float(np.linalg.norm(at[i, :3] - sun[i])),
                    "yarkovsky": "JPL's fitted A2 on the (1 au / r)^2 law (C31)" if m.yarkovsky(number) else None})
    return out


def seal(path: Path, bodies: list[int] | None, log) -> int:
    if path.exists():
        log(f"REFUSED: {path} exists; a sealed file is never rewritten (seal a new dated file instead)"); return 2
    engine = m.Engine(log)
    day_dir = m.ROOT / "forecast_seal"; day_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, name in m.tracked():
        if bodies and number not in bodies:
            continue
        start = m.start_state(number, day_dir, log)
        if start is None:
            log(f"  {number} {name}: no start state; skipped"); continue
        rows += forecast_rows(number, name, engine, start, dates_for(number), log)
    doc = {"metadata": {"sealed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "code": code_sha(),
                        "engine": "C24 (scripts/eros_sky_residuals.py): DE440s planets, Sun 1PN, DE440 asteroids as prescribed sources; Yarkovsky from JPL's fitted A2 where it is a detection (C31)",
                        "kernel": "D:/datasets/jpl/de440s.bsp", "frame": "ICRF, barycentric km and km/s; RA/Dec geocentric astrometric (light time from our path, no aberration)",
                        "rows_sha256": rows_hash(rows), "n_rows": len(rows), "bodies": sorted({r["body"] for r in rows}),
                        "how_to_judge": "python scripts/mpc_forecast_seal.py --judge (JPL's solution at judging time, Horizons centre 500 quantity 1); the sky itself through the scoreboard rows nearest each date"},
           "rows": rows}
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    log(f"sealed {len(rows)} rows -> {path}\nrows_sha256 {doc['metadata']['rows_sha256']}")
    return 0


def check(path: Path, bodies: list[int], log) -> int:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if rows_hash(doc["rows"]) != doc["metadata"]["rows_sha256"]:
        log("HASH MISMATCH: the sealed rows were edited"); return 1
    engine = m.Engine(log)
    worst_km = worst_arc = 0.0
    for number in bodies:
        sealed = [r for r in doc["rows"] if r["body"] == number]
        start = m.start_state(number, m.ROOT / "forecast_seal", log)
        fresh = forecast_rows(number, sealed[0]["name"], engine, start, [r["date_utc"][:10] for r in sealed], log)
        for s, f in zip(sealed, fresh):
            dkm = float(np.linalg.norm(np.array(s["state_bary_km_kms"][:3]) - np.array(f["state_bary_km_kms"][:3])))
            darc = math.hypot((s["ra_deg"] - f["ra_deg"]) * math.cos(math.radians(s["dec_deg"])), s["dec_deg"] - f["dec_deg"]) * 3600.0
            worst_km, worst_arc = max(worst_km, dkm), max(worst_arc, darc)
            log(f"  {number} {s['date_utc'][:10]}: {dkm:.6f} km, {darc:.6f} arcsec from the sealed row")
    log(f"worst {worst_km:.6f} km, {worst_arc:.6f} arcsec")
    return 0 if worst_km < 1e-3 else 1


def judge(path: Path, log) -> int:
    """For rows whose instant has passed: JPL's geocentric astrometric RA/Dec (Horizons centre 500, quantity 1) at the instant, and the separation from the sealed row."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)
    out_dir = m.ROOT / "forecast_judge" / now.date().isoformat(); out_dir.mkdir(parents=True, exist_ok=True)
    due = [r for r in doc["rows"] if datetime.fromisoformat(r["date_utc"].replace("Z", "+00:00")) <= now]
    if not due:
        log("no sealed instant has passed yet"); return 0
    results = []
    for r in due:
        y, mo, d = map(int, r["date_utc"][:10].split("-"))
        jd_utc = float(sum(erfa.dtf2d("UTC", y, mo, d, 0, 0, 0.0)))
        f = out_dir / f"horizons_500_{r['body']}_{r['date_utc'][:10]}.json"
        dd = m.fetch_url(m.horizons_observer_url(r["body"], "500", [jd_utc]), f, log, lambda res, n=r["body"]: m.parse_horizons_observer(res, n))
        if dd is None:
            continue
        _, ra, dec = m.parse_horizons_observer(dd["result"], r["body"])
        sep = math.hypot((r["ra_deg"] - ra[0]) * math.cos(math.radians(r["dec_deg"])), r["dec_deg"] - dec[0]) * 3600.0
        results.append({**{k: r[k] for k in ("body", "name", "date_utc", "lead_days")}, "jpl_ra_deg": float(ra[0]), "jpl_dec_deg": float(dec[0]), "sealed_minus_jpl_arcsec": sep,
                        "judged_at_utc": now.isoformat(timespec="seconds"), "jpl_solution": "Horizons at judging time"})
        log(f"  {r['body']} {r['name']} {r['date_utc'][:10]} (lead {r['lead_days']:.0f} d): sealed minus JPL {sep:.4f} arcsec")
    (out_dir / "judged.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seal", action="store_true")
    ap.add_argument("--seal-missing", action="store_true", help="seal only the tracked bodies no sealed file holds yet (new roster bodies) into today's file; nothing to do is exit 0")
    ap.add_argument("--check", nargs="*", type=int)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--file", type=Path, default=None, help="the sealed file (default fixtures/mpc_live/forecast_sealed_<today>.json for --seal, the newest for --check/--judge)")
    ap.add_argument("--bodies", nargs="*", type=int, help="--seal: only these bodies (a partial seal, for a trial file)")
    a = ap.parse_args()
    log = lambda s: print(s, flush=True)
    if a.seal_missing:
        sealed = {r["body"] for f in SEAL_DIR.glob("forecast_sealed_*.json") for r in json.loads(f.read_text(encoding="utf-8"))["rows"]}
        missing = [n for n, _ in m.tracked() if n not in sealed]
        if not missing:
            log("every tracked body is sealed"); return 0
        log(f"sealing {len(missing)} new bodies: {missing}")
        return seal(a.file or free_seal_path(datetime.now(timezone.utc).date().isoformat()), missing, log)
    if a.seal:
        return seal(a.file or SEAL_DIR / f"forecast_sealed_{datetime.now(timezone.utc).date().isoformat()}.json", a.bodies, log)
    newest = a.file or sorted(SEAL_DIR.glob("forecast_sealed_*.json"))[-1]
    if a.check is not None:
        return check(newest, a.check or [433, 99942], log)
    if a.judge:
        return judge(newest, log)
    ap.print_help(); return 2


if __name__ == "__main__":
    sys.exit(main())
