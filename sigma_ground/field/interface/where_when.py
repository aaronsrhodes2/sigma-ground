"""Where is body X at time Y? A known value from the past, or a labelled prediction.

Planets, the Sun and the Moon (Mars to Pluto as planet-system barycentres, which is what the JPL planetary kernel holds).

* A date up to today is answered from JPL's DE440 kernel and labelled ``known``. DE440 was fitted to observations that end about 2020; for 2021 to today the value is JPL's model, not new measurements, and the answer says so.
* A date after today is a ``prediction``: our own N-body integration (field/interface/reduced_model.py, the layers of the `restated_zon` predictor) started from DE440's state at 2026-01-01, stored densely in a sealed table
  (fixtures/forecast_manifest.json; the table is on the data volume). The answer carries (a) the same instant from JPL's DE440 extrapolation, when the kernel is installed, (b) the difference between the two, and
  (c) the error this model measured against DE440 at that lead when started from DE440's own state in the past (fixtures/prediction_error_table.json). It is not a claim to beat JPL, and it is not independent of JPL: the start state is JPL's.
* Without the kernel or without the table the answer says what is missing instead of guessing.

Time: an ISO date or date-time is UTC; "JD 2461234.5" or a bare Julian date is TDB; "now" is the clock. UTC is converted to TDB with astropy.
"""
from __future__ import annotations

import datetime
import functools
import hashlib
import json
import math
import re
import warnings
from pathlib import Path

import numpy as np

_FIX = Path(__file__).parent / "fixtures"
FIT_DATA_END_JD = 2459215.5                      # 2021-01-01: DE440's observations end about 2020 (Park et al. 2021)
AU_KM = 149597870.7
C_KM_S = 299792.458
SUPPORTED = ("Sun", "Mercury", "Venus", "Earth", "Moon", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto")
_BARYCENTRES = ("Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto")
CENTERS = ("sun", "ssb", "earth")


class WhereWhenError(ValueError):
    """The question cannot be answered as asked (unknown body, unreadable time, data not installed); the message says what to change."""


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def parse_when(when: str, now: datetime.datetime | None = None) -> tuple[float, str, str]:
    """(jd_tdb, iso_utc or '' for a Julian date, how it was read)."""
    now = now or _now()
    s = str(when).strip()
    if s.lower() in ("now", "today", "current"):
        return _utc_to_jd_tdb(now), now.strftime("%Y-%m-%dT%H:%M:%SZ"), "the clock (UTC)"
    m = re.fullmatch(r"(?:jd\s*)?(\d{7}(?:\.\d*)?)", s, re.I)
    if m:
        return float(m.group(1)), "", "a Julian date, read as TDB"
    try:
        t = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        raise WhereWhenError(f"cannot read the time '{when}': use an ISO date or date-time such as 2031-06-15 or 2031-06-15T12:00:00 (UTC), a Julian date such as 'JD 2462000.5' (TDB), or 'now'") from None
    if t.tzinfo is None:
        t = t.replace(tzinfo=datetime.timezone.utc)
    t = t.astimezone(datetime.timezone.utc)
    return _utc_to_jd_tdb(t), t.strftime("%Y-%m-%dT%H:%M:%SZ"), "an ISO time, read as UTC"


def _utc_to_jd_tdb(t: datetime.datetime) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from astropy.time import Time
        return float(Time(t.replace(tzinfo=None), scale="utc").tdb.jd)


def canonical_body(name: str) -> str:
    from sigma_ground.field.interface.body_aliases import BODY_ALIASES
    key = re.sub(r"\s+", " ", name.strip().lower().replace("_", " ").replace("-", " "))
    key = BODY_ALIASES.get(key, key)
    for b in SUPPORTED:
        if b.lower() == key:
            return b
    raise WhereWhenError(f"'{name}' is not supported yet. Supported: {', '.join(SUPPORTED)}. (Moons, asteroids and comets are not in this tool yet; body_constants has their masses and sizes.)")


@functools.lru_cache(maxsize=1)
def _kernel():
    from sigma_ground.field import datasets
    f = datasets.de440_kernel()
    if f is None:
        return None
    from jplephem.spk import SPK
    return SPK.open(str(f))


@functools.lru_cache(maxsize=1)
def _forecast():
    """(table dict, manifest entry) for the newest sealed table that is installed and whose sha256 matches, else (None, reason)."""
    from sigma_ground.field import datasets
    mf = _FIX / "forecast_manifest.json"
    if not mf.exists():
        return None, "no forecast has been built (fixtures/forecast_manifest.json is missing)"
    root = datasets.datasets_root()
    if root is None:
        return None, "the data volume is not installed here (set SIGMA_DATASETS)"
    for entry in reversed(json.loads(mf.read_text(encoding="utf-8"))["tables"]):
        f = root / "forecast" / entry["file"]
        if not f.is_file():
            continue
        if hashlib.sha256(f.read_bytes()).hexdigest() != entry["sha256"]:
            return None, f"{entry['file']} does not match its sealed sha256: refusing to serve a table that was changed after it was made"
        z = np.load(f)
        return {"jd": z["jd"], "pos": z["pos_km"], "vel": z["vel_km_s"], "names": [str(n) for n in z["names"]]}, entry
    return None, "the forecast table named in the manifest is not on this machine's data volume (forecast/ is mirrored by sync_datasets.sh)"


@functools.lru_cache(maxsize=2)
def _current_fingerprint(predictor: str) -> str:
    from sigma_ground.field.interface import reduced_model as rm
    return rm.model_fingerprint(predictor)


def freshness(recorded: str | None, predictor: str) -> dict:
    """Does the physics code that built a stored table still match the physics code now? A stored table is a cache of a calculation: when the equations, constants or configuration change, the table is
    the answer of the OLD calculation and must be rebuilt (scripts/make_forecast_table.py), so the answer says so instead of passing it off as current."""
    if not recorded:
        return {"checked": False, "stale": None, "note": "this table was sealed without a physics fingerprint, so it cannot be checked against the current code"}
    try:
        now = _current_fingerprint(predictor)
    except Exception as e:                                         # a machine without the full engine data cannot compute it: say so, do not guess
        return {"checked": False, "stale": None, "note": f"the current physics fingerprint could not be computed here ({type(e).__name__})"}
    if now == recorded:
        return {"checked": True, "stale": False, "note": "the physics code and configuration are the ones this table was built with"}
    return {"checked": True, "stale": True, "note": "STALE: the physics code, constants or configuration have changed since this table was built, so it is the answer of the earlier calculation. Rebuild it with scripts/make_forecast_table.py"}


def _kernel_state(name: str, jd: float):
    from sigma_ground.field.interface import reduced_model as rm
    k = _kernel()
    return rm.kernel_state(k, name, jd)


def _table_state(table: dict, name: str, jd: float):
    from sigma_ground.field.interface import reduced_model as rm
    i = table["names"].index(name)
    p, v = rm.hermite(jd, table["jd"], table["pos"][:, i], table["vel"][:, i])
    return p, v


@functools.lru_cache(maxsize=1)
def _error_table() -> dict:
    f = _FIX / "prediction_error_table.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def expected_error(body: str, lead_years: float) -> dict | None:
    """The hindcast error of this model against DE440 at that lead: log-log interpolation of the measured medians; flagged when outside the measured leads."""
    doc = _error_table()
    t = doc.get("table", {}).get(body)
    if not t:
        return None
    ks = sorted(int(k) for k in t)
    med = {k: t[str(k)]["median_km"] for k in ks}
    mx = {k: t[str(k)]["max_km"] for k in ks}
    flag = None
    if lead_years <= ks[0]:
        m = med[ks[0]] * max(lead_years, 0.0) / ks[0]
        x = mx[ks[0]] * max(lead_years, 0.0) / ks[0]
        flag = "shorter than the first measured lead (1 yr): scaled linearly from it"
    elif lead_years >= ks[-1]:
        m, x = med[ks[-1]], mx[ks[-1]]
        flag = f"beyond the longest measured lead ({ks[-1]} yr): the value is the {ks[-1]}-yr error and errors grow with lead, so it is a floor"
    else:
        lo = max(k for k in ks if k <= lead_years)
        hi = min(k for k in ks if k >= lead_years)
        if lo == hi:
            m, x = med[lo], mx[lo]
        else:
            f = (math.log(lead_years) - math.log(lo)) / (math.log(hi) - math.log(lo))
            m = math.exp(math.log(med[lo]) + f * (math.log(med[hi]) - math.log(med[lo])))
            x = math.exp(math.log(mx[lo]) + f * (math.log(mx[hi]) - math.log(mx[lo])))
    meta = doc.get("metadata", {})
    fr = freshness(meta.get("model_fingerprint"), meta.get("predictor", "restated_zon"))
    if fr.get("stale"):
        flag = (flag + "; " if flag else "") + "the error table itself is stale (" + fr["note"] + ")"
    n = t[str(min(ks, key=lambda k: abs(k - lead_years)))]["n_starts"]
    return {"typical_km": m, "largest_seen_km": x, "lead_years": lead_years, "hindcast_starts_at_that_lead": n,
            "basis": f"median and maximum distance from DE440 of this same model started from DE440's state at past epochs {meta.get('starts')} (targets inside DE440's fitted span)", "note": flag}


def position_at(body: str, when: str, center: str = "sun", now: datetime.datetime | None = None) -> dict:
    now = now or _now()
    b = canonical_body(body)
    c = center.strip().lower()
    c = {"sun": "sun", "heliocentric": "sun", "ssb": "ssb", "barycentre": "ssb", "barycenter": "ssb", "earth": "earth", "geocentric": "earth"}.get(c)
    if c is None:
        raise WhereWhenError(f"unknown center '{center}': use 'sun' (default), 'ssb' (solar-system barycentre) or 'earth'")
    if c == "earth" and b == "Earth":
        raise WhereWhenError("the position of the Earth seen from the Earth is zero: pick another center ('sun' or 'ssb') or another body")
    jd, iso, how = parse_when(when, now)
    now_jd = _utc_to_jd_tdb(now)
    kind = "known" if jd <= now_jd else "prediction"
    out: dict = {"body": b, "center": {"sun": "Sun", "ssb": "solar-system barycentre", "earth": "Earth"}[c], "when_read_as": how, "when_utc": iso or None, "when_jd_tdb": jd, "kind": kind, "frame": "ICRF (J2000 equatorial), km and km/s",
                 "reference_point": "planet-system barycentre (planet plus all its moons)" if b in _BARYCENTRES else "body centre", "notes": []}
    if b in _BARYCENTRES:
        out["notes"].append("For Mars to Pluto the position is the planet-system barycentre, which is what the JPL planetary kernel holds; the planet's centre differs from it by up to about a hundred km (Jupiter).")

    def combine(get):
        pb, vb = get(b)
        if c == "ssb":
            return pb, vb
        ref = "Sun" if c == "sun" else "Earth"
        pr, vr = get(ref)
        return pb - pr, vb - vr

    if kind == "known":
        if _kernel() is None:
            tb, why = _forecast()
            if tb is not None and jd >= tb["jd"][0]:
                p, v = combine(lambda n: _table_state(tb, n, jd))
                out.update(method="our own N-body integration (the DE440 kernel is not installed here)", basis="a past date answered from the sealed forecast table because the JPL kernel is missing")
                kind = "known"
            else:
                raise WhereWhenError("the JPL DE440 kernel (de440s.bsp) is not installed on this machine, so past dates cannot be answered here; set SIGMA_DATASETS to the folder that holds jpl/de440s.bsp")
        else:
            k = _kernel()
            lo, hi = k[0, 3].start_jd, k[0, 3].end_jd
            if not lo <= jd <= hi:
                raise WhereWhenError(f"the DE440s kernel covers JD {lo:.1f} to {hi:.1f} (years 1849 to 2150); that date is outside it")
            p, v = combine(lambda n: _kernel_state(n, jd))
            beyond = jd >= FIT_DATA_END_JD
            out.update(method="JPL DE440 ephemeris (de440s kernel)", basis=("JPL's fit to observations through about 2020" if not beyond else
                                                                            "JPL's model after the end of the data DE440 was fitted to (about 2020): a value JPL's dynamics give, not a new observation"))
            out["beyond_fit_data"] = bool(beyond)
    else:
        tb, info = _forecast()
        lead = (jd - (tb["jd"][0] if tb else 2461041.5)) / 365.25
        k = _kernel()
        jpl = None
        if k is not None and k[0, 3].start_jd <= jd <= k[0, 3].end_jd:
            jpl = combine(lambda n: _kernel_state(n, jd))
        if tb is not None and tb["jd"][0] <= jd <= tb["jd"][-1]:
            p, v = combine(lambda n: _table_state(tb, n, jd))
            out.update(method="our own N-body integration from DE440's state at 2026-01-01 (asteroids, ring, Kuiper-belt objects, the Moon's figure, measured zonals; Sun, planets, Moon)",
                       basis=f"a prediction made on {info['created_utc'][:10]} (table sealed, sha256 {info['sha256'][:12]}, commit {info['git_commit'][:10]}), {lead:.2f} yr after its start state, which is JPL's",
                       forecast_table={"file": info["file"], "sha256": info["sha256"], "created_utc": info["created_utc"], "git_commit": info["git_commit"]})
            out["lead_years_from_start"] = lead
            out["table_freshness"] = freshness(info.get("model_fingerprint"), info["predictor"])
            if out["table_freshness"].get("stale"):
                out["notes"].append(out["table_freshness"]["note"])
            out["expected_distance_from_de440"] = expected_error(b, lead)
            if jpl is not None:
                out["cross_check"] = {"jpl_de440_extrapolation_position_km": [float(x) for x in jpl[0]], "difference_km": float(np.linalg.norm(p - jpl[0])),
                                      "note": "DE440 after about 2020 is itself an extrapolation, so this is two predictions compared, not a prediction compared with a measurement"}
        elif jpl is not None:
            p, v = jpl
            why = info if tb is None else f"the date is outside the table (JD {tb['jd'][0]:.1f} to {tb['jd'][-1]:.1f})"
            out.update(method="JPL DE440 extrapolation (our integration is not available for this date)", basis=f"our forecast table is not usable here: {why}")
            out["notes"].append("This is JPL's own prediction, not ours.")
        else:
            raise WhereWhenError("a prediction for that date needs either the forecast table or the JPL kernel (to 2150); neither is available here" + (f" ({info})" if tb is None else ""))
    out["position_km"] = [float(x) for x in p]
    out["velocity_km_s"] = [float(x) for x in v]
    r = float(np.linalg.norm(p))
    out["distance_km"] = r
    out["distance_au"] = r / AU_KM
    out["light_time_min"] = r / C_KM_S / 60.0
    if c == "earth":
        out["ra_deg"] = float(math.degrees(math.atan2(p[1], p[0])) % 360.0)
        out["dec_deg"] = float(math.degrees(math.asin(p[2] / r)))
        out["notes"].append("RA and Dec are geometric (no light-time or aberration correction), in ICRF.")
    return out
