"""Whole-sky ingest, daily (stage 2, phase 1; misc/CANDIDATE_LEDGER.md C36): the MPC's orbit files to parquet, the frozen starts, and the day's priority fetch of observations.

    python scripts/mpc_sky_ingest.py [--date YYYY-MM-DD] [--no-fetch-orbits] [--fetch-obs] [--budget 1000] [--delay 20]

Daily (after MPCORB.DAT lands, 13:27 UT):
  1. MPCORB.DAT.gz -> D:/datasets/mpc_sky/mpcorb/<date>.parquet (every asteroid's elements; the raw .gz is kept on the first of each month, else removed after parsing);
     DAILY.DAT -> touched/<date>.parquet (the orbits updated by the latest Daily Orbit Update: the bodies with fresh observations).
  2. FREEZE AT FIRST SIGHT: every designation not yet in starts.parquet is appended with today's elements, epoch and date. A start is never replaced (the prediction ages; a
     refreshed start is only ever a control arm). The first run freezes the whole catalogue.
  3. The priority list, priority/<date>.json: the near-miss roster, every near-Earth body in today's touched list, then a rotating sample of the other touched bodies up to
     --budget; with --fetch-obs their observation files are fetched from the MPC get-obs API, one request per --delay seconds (20 s: the MPC's robots.txt crawl delay, C24;
     never in parallel; 1,000 bodies is 5.6 h), to obs_api/<date>/<packed>.txt.
     Everything else is scored from the monthly archive (scripts/mpc_bulk_to_parquet.py).
Nothing is overwritten; a rerun for a date reuses what it finds.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import shutil
import sys
import time
import urllib.request
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "scripts"))
from sigma_ground.field.interface import mpcorb                        # noqa: E402

SKY = Path("D:/datasets/mpc_sky")
MPCORB_URL = "https://www.minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz"
DAILY_URL = "https://www.minorplanetcenter.net/iau/MPCORB/DAILY.DAT"
MPC_API = "https://data.minorplanetcenter.net/api/get-obs"
UA = "sigma-ground research (aaron.s.rhodes@gmail.com; one request at a time)"
ROSTER = _ROOT / "sigma_ground/field/interface/fixtures/mpc_live/neo_roster.json"
FIELDS = ["packed", "number", "designation", "h_mag", "g_slope", "epoch_jd_tt", "ma_deg", "w_deg", "om_deg", "i_deg", "e", "n_deg_day", "a_au", "u_param", "reference", "n_obs",
          "n_opp", "arc", "rms_arcsec", "flags_hex", "readable", "last_obs_yyyymmdd"]


def log(s: str):
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {s}", flush=True)


def download(url: str, out: Path) -> Path:
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=600) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(out)
    log(f"fetched {url} -> {out} ({out.stat().st_size:,} bytes)")
    return out


def orbits_table(orbits) -> pa.Table:
    cols = {f: [getattr(o, f) for o in orbits] for f in FIELDS}
    cols["orbit_class"] = [o.orbit_class for o in orbits]
    cols["q_au"] = [o.q_au for o in orbits]
    return pa.table(cols)


def parse_file(path: Path) -> pa.Table:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="latin-1") as f:
        orbits = mpcorb.parse_lines(f)
    return orbits_table(orbits)


def freeze_starts(today: pa.Table, day: str) -> int:
    """Append today's elements for every designation not yet frozen; returns how many were added."""
    f = SKY / "starts.parquet"
    have = set(pq.read_table(f, columns=["packed"]).column("packed").to_pylist()) if f.exists() else set()
    packed = today.column("packed").to_pylist()
    keep = [i for i, p in enumerate(packed) if p not in have]
    if not keep:
        return 0
    new = today.take(keep).append_column("frozen_on", pa.array([day] * len(keep)))
    if f.exists():
        new = pa.concat_tables([pq.read_table(f), new], promote_options="default")
    tmp = f.with_suffix(".tmp.parquet"); pq.write_table(new, tmp); tmp.replace(f)
    return len(keep)


def priority_list(touched: pa.Table, day: str, budget: int) -> dict:
    roster = [b["number"] for b in json.loads(ROSTER.read_text(encoding="utf-8"))["bodies"]] if ROSTER.exists() else []
    t = touched.to_pydict()
    rows = list(zip(t["packed"], t["number"], t["designation"], t["orbit_class"], t["last_obs_yyyymmdd"]))
    neas = [r for r in rows if r[3] == "nea"]
    rest = [r for r in rows if r[3] != "nea"]
    seed = int(day.replace("-", "")) % max(1, len(rest))                                    # the rotating sample: a different slice each day, deterministic
    rest = rest[seed:] + rest[:seed]
    chosen = {}
    for n in roster:
        chosen[mpcorb.pack_number(n)] = {"packed": mpcorb.pack_number(n), "number": n, "why": "roster"}
    for r in neas:
        chosen.setdefault(r[0], {"packed": r[0], "number": r[1], "designation": r[2], "why": "nea touched", "last_obs": r[4]})
    for r in rest:
        if len(chosen) >= budget:
            break
        chosen.setdefault(r[0], {"packed": r[0], "number": r[1], "designation": r[2], "why": "sample", "last_obs": r[4]})
    return {"date": day, "budget": budget, "n": len(chosen), "roster": len(roster), "nea_touched": len(neas), "bodies": list(chosen.values())}


def fetch_obs(desig: str, out: Path) -> bool:
    if out.exists() and out.stat().st_size > 0:
        return True
    for attempt in range(1, 4):
        req = urllib.request.Request(MPC_API, data=json.dumps({"desigs": [desig], "output_format": ["OBS80"]}).encode(), method="GET",
                                     headers={"Content-Type": "application/json", "User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            txt = d[0]["OBS80"] if isinstance(d, list) else d.get("OBS80")
            if not txt:
                raise ValueError("no OBS80 in the response")
            tmp = out.with_suffix(".part"); tmp.write_text(txt, encoding="utf-8"); tmp.replace(out)
            return True
        except Exception as e:                                        # noqa: BLE001
            log(f"  get-obs {desig}: attempt {attempt}/3 failed {e!r}"); time.sleep(20 * attempt)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=_date.today().isoformat())
    ap.add_argument("--no-fetch-orbits", action="store_true", help="reuse the day's downloaded files; fail if absent")
    ap.add_argument("--fetch-obs", action="store_true", help="also fetch the priority list's observation files from the MPC API")
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--delay", type=float, default=20.0)
    ap.add_argument("--queue-night", action="store_true", help="queue tonight's whole-sky scoring, seal and report in the night window (scripts/night_queue.py)")
    a = ap.parse_args()
    day = a.date
    raw = SKY / "raw" / day
    mp_gz, daily = raw / "MPCORB.DAT.gz", raw / "DAILY.DAT"
    if not a.no_fetch_orbits:
        download(MPCORB_URL, mp_gz); download(DAILY_URL, daily)
    out_m, out_t = SKY / "mpcorb" / f"{day}.parquet", SKY / "touched" / f"{day}.parquet"
    if not out_m.exists():
        t0 = time.time(); tab = parse_file(mp_gz); out_m.parent.mkdir(parents=True, exist_ok=True); pq.write_table(tab, out_m)
        log(f"MPCORB: {tab.num_rows:,} orbits parsed in {time.time() - t0:.0f} s -> {out_m}")
    else:
        tab = pq.read_table(out_m)
    if not out_t.exists():
        tt = parse_file(daily); out_t.parent.mkdir(parents=True, exist_ok=True); pq.write_table(tt, out_t)
        log(f"DAILY: {tt.num_rows:,} touched orbits -> {out_t}")
    else:
        tt = pq.read_table(out_t)
    added = freeze_starts(tab, day)
    log(f"starts: {added:,} new designations frozen on {day} (total {pq.read_metadata(SKY / 'starts.parquet').num_rows:,})")
    pl = priority_list(tt, day, a.budget)
    pf = SKY / "priority" / f"{day}.json"; pf.parent.mkdir(parents=True, exist_ok=True)
    if not pf.exists():
        pf.write_text(json.dumps(pl, indent=1), encoding="utf-8")
    log(f"priority: {pl['n']} bodies (roster {pl['roster']}, NEAs touched {pl['nea_touched']}, budget {a.budget}) -> {pf}")
    if day[-2:] != "01" and mp_gz.exists():
        mp_gz.unlink()                                                   # the parquet is the record; the raw 94 MB is kept only on the first of the month
    if a.queue_night:
        import subprocess
        py = sys.executable; nq = [py, "-X", "utf8", str(_ROOT / "scripts" / "night_queue.py"), "add"]
        tomorrow = (_date.fromisoformat(day) + timedelta(days=1)).isoformat()
        subprocess.run(nq + ["--name", f"sky_seal_{day}", "--est-minutes", "10", "--", py, "-X", "utf8", str(_ROOT / "scripts" / "mpc_sky_seal.py"), "--date", day], check=False)
        subprocess.run(nq + ["--name", f"sky_score_{tomorrow}", "--est-minutes", "45", "--resumable", "--", py, "-X", "utf8", str(_ROOT / "scripts" / "mpc_sky_score.py"), "--night", tomorrow], check=False)
        subprocess.run(nq + ["--name", f"sky_report_{tomorrow}", "--est-minutes", "5", "--", py, "-X", "utf8", str(_ROOT / "scripts" / "mpc_sky_report.py")], check=False)
        subprocess.run(nq + ["--name", f"sky_growth_{tomorrow}", "--est-minutes", "10", "--", py, "-X", "utf8", str(_ROOT / "scripts" / "residual_growth.py")], check=False)
        log("queued tonight's sky seal, score and report")
    if a.fetch_obs:
        od = SKY / "obs_api" / day; od.mkdir(parents=True, exist_ok=True)
        ok = bad = 0
        for b in json.loads(pf.read_text(encoding="utf-8"))["bodies"]:
            desig = str(b["number"]) if b.get("number") else b.get("designation", b["packed"])
            f = od / f"{b['packed'].replace(' ', '_').replace('/', '_')}.txt"
            if f.exists():
                continue
            if fetch_obs(desig, f):
                ok += 1
            else:
                bad += 1
            time.sleep(a.delay)
        log(f"obs fetched: {ok} ok, {bad} failed -> {od}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
