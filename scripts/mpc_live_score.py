"""The MPC live scoreboard (misc/CANDIDATE_LEDGER.md C27): OUR N-body prediction and JPL's against the newest raw sky positions of a tracked set of asteroids, one JSON line per body per window.

For each tracked body: fetch its Minor Planet Center observations (OBS80, the whole file, one request per 20 s), keep the ones newer than the last run's watermark (incremental by observation
date), propagate OUR state from the body's fixed start epoch (never moved: the prediction is meant to age) through the C24 pipeline (`scripts/eros_sky_residuals.py`), fetch JPL's own
topocentric astrometric RA/Dec from Horizons at the same instants, and score both. Every response is saved under D:/datasets/mpc_live/<date>/, never overwritten; the rows go to
D:/datasets/mpc_live/scoreboard.jsonl and the day's rows are mirrored to sigma_ground/field/interface/fixtures/mpc_live/<date>.json.

usage: python scripts/mpc_live_score.py --date 2026-11-02 [--bodies 433 1 4] [--no-fetch] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path

import erfa
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
import eros_sky_residuals as esr                                   # the C24 pipeline: parse_obs80, parse_obscodes, Obs, light_time_targets, radec_residuals  # noqa: E402

FIX = _ROOT / "sigma_ground/field/interface/fixtures"
ROOT = Path("D:/datasets/mpc_live")
MPC_OLD = Path("D:/datasets/mpc")                                  # C24's files: ObsCodes.html and the Eros Horizons vectors (the Eros start state)
HORIZONS = "https://ssd.jpl.nasa.gov/api/horizons.api"
MPC_API = "https://data.minorplanetcenter.net/api/get-obs"
UA = "sigma-ground research (personal; aaron.s.rhodes@gmail.com); one request per 20 s"
MPC_DELAY_S = 20.0                                                  # the MPC robots.txt crawl delay (C24)
HORIZONS_DELAY_S = 0.5
DAY = 86400.0
JD_2026 = 2461041.5                                                 # 2026-01-01 00:00 TDB: the fixture's last snapshot epoch and the heavies' start epoch
JD_EROS = esr.JD0                                                   # 2458849.5, C24's start (2020-01-01 TDB)
W0 = 2458845.5                                                      # 2019-12-28 TDB: the prescribed-source table starts four days before Eros's start
HORIZON_DAYS = 4100.0                                               # to 2031-03: the table must cover every time the integration visits
BACKFILL_FROM = "2026-01-01"
WINDOW_DAYS = 30
CHUNK = 200                                                         # instants per Horizons request (URL length)
OUTLIER_ARCSEC = 5.0                                                # C24's rule, on JPL's residual, applied to both arms


def tracked() -> list[tuple[int, str]]:
    cat = json.loads((FIX / "de440_perturbers_v2physics.json").read_text(encoding="utf-8"))["asteroids"]
    top = sorted(cat, key=lambda a: -a["gm_km3_s2"])[:30]
    return [(a["number"], a["name"]) for a in top] + [(433, "Eros"), (99942, "Apophis")]


# ----------------------------------------------------------------------------- bookkeeping (pure; tested)
def obs_datetime(row) -> datetime:
    y, m, d = row[0], row[1], row[2]
    day = int(math.floor(d))
    return datetime(y, m, day, tzinfo=timezone.utc) + timedelta(days=d - day)


def select_new(rows, watermark_iso: str | None, before_iso: str, backfill_from: str = BACKFILL_FROM):
    """Rows with observation time strictly after the watermark (the backfill start when there is none) and before `before_iso` 00:00 UTC, oldest first."""
    lo = datetime.fromisoformat(watermark_iso) if watermark_iso else datetime.fromisoformat(backfill_from)
    if lo.tzinfo is None:
        lo = lo.replace(tzinfo=timezone.utc)
    hi = datetime.fromisoformat(before_iso).replace(tzinfo=timezone.utc)
    keep = [(obs_datetime(r), r) for r in rows]
    keep = [(t, r) for t, r in keep if lo < t < hi]
    keep.sort(key=lambda x: x[0])
    return [r for _, r in keep], [t for t, _ in keep]


def windows(times: list[datetime], first_run: bool, window_days: int = WINDOW_DAYS) -> list[tuple[int, int]]:
    """Index ranges [i0, i1) of `times` (sorted) to score as separate rows: one per `window_days` of observation date on a first run (backfill), one window otherwise."""
    if not times:
        return []
    if not first_run:
        return [(0, len(times))]
    out, i0, t0 = [], 0, times[0]
    for i, t in enumerate(times):
        if (t - t0).days >= window_days:
            out.append((i0, i)); i0, t0 = i, t
    out.append((i0, len(times)))
    return out


def robust(res: np.ndarray) -> dict:
    """Median and robust sigma (1.4826 x MAD) of the total residual, and the medians of the two components."""
    tot = np.hypot(res[:, 0], res[:, 1])
    med = float(np.median(tot))
    return {"median_arcsec": med, "robust_sigma_arcsec": float(1.4826 * np.median(np.abs(tot - med))), "rms_arcsec": float(np.sqrt(np.mean(tot ** 2))),
            "median_dra_cosdec": float(np.median(res[:, 0])), "median_ddec": float(np.median(res[:, 1]))}


def horizons_observer_url(number: int, code: str, jd_utc: list[float]) -> str:
    q = {"format": "json", "COMMAND": f"'{number};'", "OBJ_DATA": "'NO'", "MAKE_EPHEM": "'YES'", "EPHEM_TYPE": "'OBSERVER'", "CENTER": f"'{code}@399'", "QUANTITIES": "'1'",
         "TLIST": "'" + " ".join(f"{t:.7f}" for t in jd_utc) + "'", "TLIST_TYPE": "'JD'", "TIME_TYPE": "'UT'", "ANG_FORMAT": "'DEG'", "EXTRA_PREC": "'YES'", "CSV_FORMAT": "'YES'",
         "CAL_FORMAT": "'JD'", "TIME_DIGITS": "'FRACSEC'"}
    return HORIZONS + "?" + urllib.parse.urlencode(q, quote_via=urllib.parse.quote)


def horizons_vectors_url(number: int, jd_tdb: float) -> str:
    q = {"format": "json", "COMMAND": f"'{number};'", "OBJ_DATA": "'YES'", "MAKE_EPHEM": "'YES'", "EPHEM_TYPE": "'VECTORS'", "CENTER": "'500@0'", "TLIST": f"'{jd_tdb:.2f}'", "TLIST_TYPE": "'JD'",
         "TIME_TYPE": "'TDB'", "VEC_TABLE": "'2'", "REF_PLANE": "'FRAME'", "REF_SYSTEM": "'ICRF'", "OUT_UNITS": "'KM-D'", "CSV_FORMAT": "'YES'"}
    return HORIZONS + "?" + urllib.parse.urlencode(q, quote_via=urllib.parse.quote)


def parse_horizons_observer(result: str, number: int):
    """(jd_utc, ra_deg, dec_deg) arrays from an OBSERVER CSV table, or raise ValueError (wrong body, no table, apparent instead of astrometric)."""
    m = re.search(r"Target body name:\s*(.+?)\s*\{", result)
    if m is None or not re.match(rf"\s*{number}\b", m.group(1)):
        raise ValueError(f"unexpected body: {m.group(1) if m else 'no name'}")
    if "$$SOE" not in result:
        raise ValueError("no ephemeris block: " + result[:200].replace("\n", " "))
    head, body = result.split("$$SOE")
    rows = body.split("$$EOE")[0].strip().splitlines()
    header = [ln for ln in head.strip().splitlines() if "R.A." in ln][-1]
    cols = [c.strip() for c in header.split(",")]
    ira = next(i for i, c in enumerate(cols) if c.startswith("R.A."))
    idec = next(i for i, c in enumerate(cols) if c.startswith("DEC"))
    if "ICRF" not in cols[ira]:
        raise ValueError(f"RA column is not astrometric ICRF: {cols[ira]}")
    jd, ra, dec = [], [], []
    for ln in rows:
        c = [x.strip() for x in ln.split(",")]
        jd.append(float(c[0])); ra.append(float(c[ira])); dec.append(float(c[idec]))
    return np.array(jd), np.array(ra), np.array(dec)


def parse_horizons_vector_row(result: str, number: int, jd_tdb: float) -> np.ndarray:
    m = re.search(r"Target body name:\s*(.+?)\s*\{", result)
    if m is None or not re.match(rf"\s*{number}\b", m.group(1)):
        raise ValueError(f"unexpected body: {m.group(1) if m else 'no name'}")
    units = re.search(r"Output units\s*:\s*(KM-[SD])", result)
    if units is None:
        raise ValueError("no 'Output units' line (KM-S or KM-D expected)")
    vscale = DAY if units.group(1) == "KM-S" else 1.0                      # C24's Eros vectors are km/s (as its load_horizons scales them); the asteroid group is km/day
    for ln in result.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines():
        c = [x.strip() for x in ln.split(",")]
        if abs(float(c[0]) - jd_tdb) < 1e-6:
            y = np.array([float(x) for x in c[2:8]])
            y[3:] *= vscale
            if not 1e5 < np.linalg.norm(y[3:]) < 1e7:                   # km/day: a heliocentric orbit inside 10 au moves 0.4-4 million km a day
                raise ValueError(f"velocity {np.linalg.norm(y[3:]):.3g} km/day is not a solar-system orbit; units misread")
            return y
    raise ValueError(f"no row at JD {jd_tdb}")


# ----------------------------------------------------------------------------- fetches
def fetch_mpc(number: int, out: Path, log) -> Path | None:
    if out.exists() and out.stat().st_size > 0:
        log(f"  mpc {number}: kept {out.name} ({out.stat().st_size} bytes)")
        return out
    for attempt in range(1, 4):
        req = urllib.request.Request(MPC_API, data=json.dumps({"desigs": [str(number)], "output_format": ["OBS80"]}).encode(), method="GET",
                                     headers={"Content-Type": "application/json", "User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            txt = d[0]["OBS80"] if isinstance(d, list) else d.get("OBS80")
            if not txt:
                raise ValueError("no OBS80 in the response: " + json.dumps(d)[:200])
            tmp = out.with_suffix(".part")
            tmp.write_text(txt, encoding="utf-8")
            tmp.replace(out)
            log(f"  mpc {number}: {len(txt.encode())} bytes, {txt.count(chr(10)) + 1} lines")
            return out
        except Exception as e:
            log(f"  mpc {number}: attempt {attempt}/3 FAILED {e!r}; retrying after {MPC_DELAY_S} s")
            time.sleep(MPC_DELAY_S)
    return None


def fetch_url(url: str, out: Path, log, check) -> dict | None:
    """One Horizons GET, saved whole; `check(result_text)` raises ValueError on a wrong response, which is kept under rejected/."""
    if out.exists():
        try:
            d = json.loads(out.read_text(encoding="utf-8")); check(d["result"]); return d
        except Exception:
            out.replace(out.parent / "rejected" / (out.name + ".old"))
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "rejected").mkdir(exist_ok=True)
    for attempt in range(1, 4):
        tmp = out.with_suffix(".part")
        r = subprocess.run(["curl", "-sS", "--max-time", "120", "-o", str(tmp), "-w", "%{http_code}", url], capture_output=True, text=True)
        time.sleep(HORIZONS_DELAY_S)
        try:
            if r.stdout.strip() != "200":
                raise ValueError(f"HTTP {r.stdout.strip()} {r.stderr.strip()[:80]}")
            d = json.loads(tmp.read_text(encoding="utf-8"))
            check(d["result"])
            tmp.replace(out)
            return d
        except Exception as e:
            log(f"  horizons {out.name}: attempt {attempt}/3 FAILED {e!r}")
            if tmp.exists():
                tmp.replace(out.parent / "rejected" / f"{out.stem}_attempt{attempt}.json")
            time.sleep(2.0 * attempt)
    return None


# ----------------------------------------------------------------------------- our arm
def start_state(number: int, day_dir: Path, log) -> tuple[float, np.ndarray, str] | None:
    """(jd_tdb, barycentric state km and km/day, source) for the fixed start epoch of a tracked body."""
    if number == 433:
        f = MPC_OLD / "eros" / "vec_2019-12-31_2022-01-02.json"
        y = parse_horizons_vector_row(json.loads(f.read_text(encoding="utf-8"))["result"], 433, JD_EROS)
        return JD_EROS, y, f"JPL Horizons barycentric state at JD {JD_EROS} (2020-01-01 TDB), C24's start, {f}"
    f = Path("D:/datasets/jpl/horizons_asteroids") / f"horizons_ast_{number}.json"
    if f.exists():
        y = parse_horizons_vector_row(json.loads(f.read_text(encoding="utf-8"))["result"], number, JD_2026)
        return JD_2026, y, f"JPL Horizons barycentric state at JD {JD_2026} (2026-01-01 TDB), fetched 2026-10-16, {f}"
    f = ROOT / "start_states" / f"horizons_vec_{number}.json"
    d = fetch_url(horizons_vectors_url(number, JD_2026), f, log, lambda res: parse_horizons_vector_row(res, number, JD_2026))
    if d is None:
        return None
    return JD_2026, parse_horizons_vector_row(d["result"], number, JD_2026), f"JPL Horizons barycentric state at JD {JD_2026} (2026-01-01 TDB), fetched {datetime.now(timezone.utc).date()}, {f}"


class Engine:
    """The C24 engine: DE440s planets, the Sun's 1PN term, DE440's 343 asteroids as prescribed sources (the body's own entry zeroed per run)."""

    def __init__(self, log, w0: float = W0, horizon_days: float = HORIZON_DAYS, asteroids: bool = True):
        from jplephem.spk import SPK
        from comet_ladder import gm_planets
        from sigma_ground.field.interface import perturber_paths as pp
        from sigma_ground.field.interface.asteroid_perturbers import Perturber
        self.kernel = SPK.open("D:/datasets/jpl/de440s.bsp")
        cat = [x for x in json.loads((FIX / "de440_perturbers_v2physics.json").read_text(encoding="utf-8"))["asteroids"] if x["elements"]]
        self.numbers = np.array([x["number"] for x in cat])
        self.gm_all = np.array([x["gm_km3_s2"] for x in cat])
        self.gmp = gm_planets()
        self.w0, self.asteroids = w0, asteroids
        self.nodes = None
        if asteroids:
            cache = ROOT / f"perturber_nodes_{w0:.1f}_{int(horizon_days)}d.npy"
            if cache.exists():
                self.nodes = np.load(cache)
            else:
                pert = [Perturber(str(x["number"]), x["gm_km3_s2"], 0.0, *[x["elements"][k] for k in ("epoch_jd_tdb", "a_au", "e", "i_deg", "om_deg", "w_deg", "ma_deg")]) for x in cat]
                t0 = time.time()
                self.nodes = pp.asteroid_window_paths(pert, self.kernel, np.array(self.gmp), w0, horizon_days, 1.0)
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, self.nodes)
                log(f"perturber paths {self.nodes.shape} in {time.time() - t0:.0f} s -> {cache}")
        self.pp = pp
        self._src_cache: dict[int, object] = {}

    def sources(self, number: int):
        if not self.asteroids:
            return None
        if number not in self._src_cache:
            from sigma_ground.field.interface.prescribed_sources import PrescribedSources
            gm = np.where(self.numbers == number, 0.0, self.gm_all)             # the target cannot pull on itself (as C24 and asteroid_ladder.py)
            t = self.pp.to_prescribed_sources(self.nodes, gm, 1.0)
            self._src_cache[number] = PrescribedSources(t.gm, t.t0, t.h, t.pos, t.vel)
        return self._src_cache[number]

    def positions(self, number: int, jd0: float, y0: np.ndarray, targets: np.ndarray) -> np.ndarray:
        from sigma_ground.field.interface import comet_orbits as co
        out = co.propagate(list(y0), jd0, [float(t) for t in targets], self.kernel, self.gmp, gr=True, sources=self.sources(number), sources_epoch_jd=self.w0)
        return np.array([out[float(t)][:3] for t in targets])


# ----------------------------------------------------------------------------- one body, one window
def jd_of_date(iso: str) -> float:
    y, m, d = map(int, iso.split("-"))
    u1, u2 = erfa.dtf2d("UTC", y, m, d, 0, 0, 0.0)
    return float(u1 + u2)


def utc_jd(rows) -> np.ndarray:
    out = np.empty(len(rows))
    for i, r in enumerate(rows):
        day = int(math.floor(r[2]))
        u1, u2 = erfa.dtf2d("UTC", r[0], r[1], day, 0, 0, 0.0)
        out[i] = u1 + u2 + (r[2] - day)
    return out


def fetch_part(number: int, code: str, part: list[int], times, jd_utc: np.ndarray, ra_j: np.ndarray, dec_j: np.ndarray, files: list, day_dir: Path, log, dry_run: bool = False) -> list[int]:
    """One Horizons OBSERVER request for the observation indices `part` of station `code`, filling ra_j/dec_j (deg). On failure the part is split in two and each half tried, down to
    single instants (Horizons answered 502 to every 81-120-instant request for station C40 on 2026-11-02 and 200 to a 2-instant one: the limit is request size per site, not the site).
    Returns the indices that got no answer."""
    f = day_dir / "horizons" / f"obs_{number}_{code}_{times[part[0]].strftime('%Y%m%dT%H%M%S')}_{len(part)}.json"
    url = horizons_observer_url(number, code, [float(jd_utc[i]) for i in part])
    if dry_run:
        log(f"  DRY RUN would send: {url[:160]}... ({len(part)} instants)"); return []
    d = fetch_url(url, f, log, lambda res: parse_horizons_observer(res, number))
    ok_rows = False
    if d is not None:
        jd, ra, dec = parse_horizons_observer(d["result"], number)
        ok_rows = len(jd) == len(part) and np.abs(jd - jd_utc[part]).max() <= 2.0 / DAY
        if not ok_rows:
            log(f"  horizons {f.name}: {len(jd)} rows for {len(part)} instants or times differ; skipped")
    if ok_rows:
        ra_j[part] = ra; dec_j[part] = dec
        files.append(str(f))
        return []
    if len(part) >= 2:
        h = len(part) // 2
        log(f"  horizons {code}: {len(part)} instants refused, splitting into {h} + {len(part) - h}")
        return fetch_part(number, code, part[:h], times, jd_utc, ra_j, dec_j, files, day_dir, log) + fetch_part(number, code, part[h:], times, jd_utc, ra_j, dec_j, files, day_dir, log)
    return list(part)


def score_window(number: int, name: str, rows, times, engine: Engine, start, codes, day_dir: Path, log, dry_run: bool) -> dict:
    jd0, y0, start_src = start
    obs = esr.Obs(rows, codes, engine.kernel)
    jd_utc = utc_jd([r for r in rows if r[5] in codes])
    row = {"body": number, "name": name, "start_epoch_jd_tdb": jd0, "start_source": start_src, "n_obs": obs.n,
           "window_utc": [times[0].isoformat(), times[-1].isoformat()], "lead_days": float(obs.jd_tdb.max() - jd0), "lead_days_first": float(obs.jd_tdb.min() - jd0)}
    if obs.n == 0:
        row["note"] = "observations exist but none from a station with ObsCodes constants"
        return row
    # ours: light time from our own path, as C24
    t0 = time.time()
    fn = lambda T: engine.positions(number, jd0, y0, T)
    T, tau = esr.light_time_targets(fn, obs, iters=2)                  # each iteration is a full propagation (about 20 s from Eros's 2020 start); after two the times move by well under a millisecond
    res_ours = esr.radec_residuals(obs, fn(T))
    log(f"  ours: {obs.n} obs propagated in {time.time() - t0:.0f} s, median {float(np.median(np.hypot(res_ours[:, 0], res_ours[:, 1]))):.4f} arcsec")
    # JPL: Horizons topocentric astrometric RA/Dec at the observation instants, per station, chunks of CHUNK
    ra_j = np.full(obs.n, np.nan); dec_j = np.full(obs.n, np.nan)
    files, failed_sites = [], {}
    by_code: dict[str, list[int]] = {}
    for i, c in enumerate(obs.code):
        by_code.setdefault(c, []).append(i)
    for code, idx in sorted(by_code.items()):
        for k in range(0, len(idx), CHUNK):
            lost = fetch_part(number, code, idx[k:k + CHUNK], times, jd_utc, ra_j, dec_j, files, day_dir, log, dry_run)
            if lost:
                failed_sites[code] = failed_sites.get(code, 0) + len(lost)
    have = ~np.isnan(ra_j)
    row["files"] = {"horizons": files}
    row["n_no_jpl"] = int((~have).sum())
    if failed_sites:
        row["jpl_failed_sites"] = failed_sites
    if dry_run or not have.any():
        row["note"] = "dry run" if dry_run else "no JPL prediction obtained"
        row["ours_all"] = robust(res_ours)
        return row
    dra = np.angle(np.exp(1j * (obs.ra - np.radians(ra_j)))) * np.cos(obs.dec) * esr.ARC
    ddec = (obs.dec - np.radians(dec_j)) * esr.ARC
    res_jpl = np.stack([dra, ddec], axis=1)
    ok = have & (np.hypot(res_jpl[:, 0], res_jpl[:, 1]) <= OUTLIER_ARCSEC)
    diff = res_ours[ok] - res_jpl[ok]
    row.update({"n_scored": int(ok.sum()), "n_dropped_outliers": int((have & ~ok).sum()), "ours": robust(res_ours[ok]), "jpl": robust(res_jpl[ok]), "ours_minus_jpl": robust(diff),
                "stations": {c: int(sum(1 for i in idx if ok[i])) for c, idx in sorted(by_code.items())}})
    per_obs = day_dir / f"residuals_{number}_{times[0].strftime('%Y%m%d')}_{times[-1].strftime('%Y%m%d')}.json"
    per_obs.write_text(json.dumps({"body": number, "jd_utc": jd_utc.tolist(), "code": obs.code, "ours_dra_ddec": res_ours.tolist(), "jpl_dra_ddec": res_jpl.tolist(), "scored": ok.tolist()}), encoding="utf-8")
    row["files"]["per_observation"] = str(per_obs)
    log(f"  scored {int(ok.sum())}: ours median {row['ours']['median_arcsec']:.4f} sigma {row['ours']['robust_sigma_arcsec']:.4f} | JPL median {row['jpl']['median_arcsec']:.4f} "
        f"sigma {row['jpl']['robust_sigma_arcsec']:.4f} | ours-JPL median {row['ours_minus_jpl']['median_arcsec']:.4f}  (dropped {row['n_dropped_outliers']}, no JPL {row['n_no_jpl']})")
    return row


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=_date.today().isoformat(), help="the run date; observations before this day 00:00 UTC are scored")
    ap.add_argument("--bodies", nargs="+", type=int, default=None, help="asteroid numbers (default: the tracked set)")
    ap.add_argument("--no-fetch", action="store_true", help="use the day's saved MPC files only")
    ap.add_argument("--dry-run", action="store_true", help="fetch MPC files and propagate, but send no Horizons request and append nothing")
    ap.add_argument("--state", type=Path, default=ROOT / "state.json")
    ap.add_argument("--scoreboard", type=Path, default=ROOT / "scoreboard.jsonl")
    a = ap.parse_args()
    bodies = [(n, nm) for n, nm in tracked() if a.bodies is None or n in a.bodies]
    day_dir = ROOT / a.date
    day_dir.mkdir(parents=True, exist_ok=True)
    logf = open(day_dir / "score.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    log(f"mpc_live_score --date {a.date}: {len(bodies)} bodies, sigma-ground {git_sha()}")
    state = json.loads(a.state.read_text(encoding="utf-8")) if a.state.exists() else {}
    codes = esr.parse_obscodes(MPC_OLD / "ObsCodes.html")
    engine = Engine(log)
    day_rows, failures = [], []
    for k, (number, name) in enumerate(bodies):
        log(f"[{k + 1}/{len(bodies)}] {number} {name}")
        obs_file = day_dir / f"obs80_{number}.txt"
        if not a.no_fetch and not obs_file.exists():
            if k and MPC_DELAY_S:
                time.sleep(MPC_DELAY_S)
            if fetch_mpc(number, obs_file, log) is None:
                failures.append(number); log(f"  {number}: MPC fetch failed, body skipped this run"); continue
        if not obs_file.exists():
            failures.append(number); log(f"  {number}: no MPC file for {a.date}"); continue
        st = state.get(str(number), {})
        rows_all = esr.parse_obs80(obs_file)
        rows, times = select_new(rows_all, st.get("last_obs_utc"), a.date)
        n_unknown_station = sum(1 for r in rows if r[5] not in codes)
        rows, times = [r for r in rows if r[5] in codes], [t for r, t in zip(rows, times) if r[5] in codes]
        base = {"date": a.date, "body": number, "name": name, "files": {"obs80": str(obs_file)}, "engine": "C24 (scripts/eros_sky_residuals.py): DE440s planets, Sun 1PN, DE440 asteroids as prescribed sources",
                "code": git_sha(), "n_obs_in_file": len(rows_all), "n_new_unknown_station": n_unknown_station}
        start = start_state(number, day_dir, log)
        if start is None:
            failures.append(number); log(f"  {number}: no start state, body skipped"); continue
        if not rows:
            row = dict(base, n_obs=0, start_epoch_jd_tdb=start[0], start_source=start[2], note=f"no optical C/B observations after {st.get('last_obs_utc') or BACKFILL_FROM} and before {a.date}",
                       lead_days=float(jd_of_date(a.date) - start[0]))
            log(f"  {number}: {row['note']}")
            day_rows.append(row); continue
        first_run = "last_obs_utc" not in st
        for i0, i1 in windows(times, first_run):
            log(f"  window {times[i0].date()} .. {times[i1 - 1].date()}: {i1 - i0} observations{' (backfill)' if first_run else ''}")
            try:
                r = score_window(number, name, rows[i0:i1], times[i0:i1], engine, start, codes, day_dir, log, a.dry_run)
            except Exception as e:
                failures.append(number); log(f"  {number}: window FAILED {e!r}"); import traceback; traceback.print_exc(); break
            row = dict(base, **r, backfill=first_run)
            row["files"] = dict(base["files"], **r.get("files", {}))
            day_rows.append(row)
            if not a.dry_run:
                with open(a.scoreboard, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                state[str(number)] = {"last_obs_utc": times[i1 - 1].isoformat(), "start_epoch_jd_tdb": start[0], "start_source": start[2], "updated": a.date}
                a.state.write_text(json.dumps(state, indent=1), encoding="utf-8")
    if not a.dry_run:
        for row in day_rows:
            if row.get("n_obs") == 0:
                with open(a.scoreboard, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
        mirror = FIX / "mpc_live"
        mirror.mkdir(exist_ok=True)
        (mirror / f"{a.date}.json").write_text(json.dumps({"date": a.date, "ledger": "misc/CANDIDATE_LEDGER.md C27", "rows": day_rows, "failed_bodies": failures}, indent=1), encoding="utf-8")
    scored = [r for r in day_rows if r.get("n_scored")]
    log(f"finished: {len(scored)} rows scored, {sum(1 for r in day_rows if r.get('n_obs') == 0)} bodies without new observations, {len(failures)} failed {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
