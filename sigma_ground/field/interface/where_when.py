"""Where is body X at time Y? A known value from JPL's kernel, or our own calculation made now.

Planets, the Sun and the Moon (Mars to Pluto as planet-system barycentres, which is what the JPL planetary kernel holds).

The Captain's rule (2026-11-02): when sigma-ground is asked where a body is at a moment, the answer is our own calculation computed then, not a stored
table; stored data is allowed only for what is known fact, the JPL kernels.

* A date up to the edge of reality is ``known``: JPL's DE442 kernel (datasets.truth_kernel()), the fit to observations JPL made in May 2024. Its fit data
  end about 2024; the edge used here is 2024-01-01 TDB, a date certainly inside the data (DE440 fallback: 2021-01-01).
* A date after the edge is a ``prediction`` computed when asked: the reduced N-body model (reduced_model.py, the layers of the served predictor,
  SERVED below) is started from the kernel's state at the edge and integrated to the date asked, here, now. No table is read for the position. The
  answer carries the lead from the edge, the predictor and its physics fingerprint, the integrator and step, how long the calculation took, JPL's
  DE442 value for the same instant (its own extrapolation past the edge) with the difference, and the error this model MEASURED against DE440 at that
  lead when started from DE440's own state in the past (fixtures/prediction_error_table.json; a measurement of the calculation, so it stays).
* The one exception to "nothing kept": a memo in this process of the last live integration, keyed by the physics fingerprint and the start, so a
  second question continues the same integration instead of restarting it. The answer says when the memo was used. Nothing is written to disk.
* A time budget (LIVE_BUDGET_S): if the integration cannot reach the date in budget the answer says how far it got, what it would cost to finish, and
  that asking again continues; it never hangs the chat. JPL's own value for the asked date is given beside that, labelled as JPL's.
* The sealed forecast table of 2026-10-27 (fixtures/forecast_manifest.json, D:/datasets/forecast) is a RECORD of a prediction made that day, to be judged
  later; it is no longer served. _forecast() still reads it for audits and tests.

Time: an ISO date or date-time is UTC; "JD 2461234.5" or a bare Julian date is TDB; "now" is the clock. UTC is converted to TDB with astropy.
"""
from __future__ import annotations

import bisect
import datetime
import functools
import hashlib
import json
import math
import re
import threading
import time
import warnings
from pathlib import Path

import numpy as np

_FIX = Path(__file__).parent / "fixtures"
FIT_DATA_END_JD = 2459215.5                      # 2021-01-01: DE440's observations end about 2020 (Park et al. 2021)
DE442_DATA_END_JD = 2460310.5                    # 2024-01-01: DE442 (integrated 2024-05-13) adds "four years" of Mars and Juno ranging to DE440's; a date certainly inside its data
AU_KM = 149597870.7
C_KM_S = 299792.458
DAY = 86400.0
SUPPORTED = ("Sun", "Mercury", "Venus", "Earth", "Moon", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto")
_BARYCENTRES = ("Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto")
CENTERS = ("sun", "ssb", "earth")

# The ONE place that says what body_position serves (read it, never copy it): the predictor label, the integrator and its step.
# restated_math_tide (2026-11-02, misc/CANDIDATE_LEDGER.md C26 verdicts): restated_zon with the Earth's J2-J4 on their reference radius and DE430's
# time-delayed Earth tide; the Moon 0.245 -> 0.097 km at +5 yr against DE442 on four starts, the other bodies unchanged. Yoshida-6 at 0.1 d gives
# the fine Forest-Ruth answer (0.0125 d) for 58 % of its force evaluations (C26 Phase 3; the same-answer distance is measured in C29).
# The error table served beside each answer (fixtures/prediction_error_table.json) is this predictor's, four starts, 0.0125 d, against DE442.
SERVED = {"predictor": "restated_math_tide", "integrator": "y6", "dt_days": 0.1}
LIVE_BUDGET_S = 60.0                             # the chat must not hang: Mentat's ask route times out at 90 s
LIVE_MIN_HORIZON_DAYS = 5.0 * 365.25             # the perturber paths are built this far at least (lead + 1 yr beyond that); further leads rebuild


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


def _jd_to_iso(jd: float) -> str:
    """Calendar date (TDB) of a Julian date, for the reader; not used in any calculation."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from astropy.time import Time
        return Time(jd, format="jd", scale="tdb").isot[:19] + " TDB"


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
    """The DE440s kernel (the model's start and perturber kernel; celestial.py reads it too), or None."""
    from sigma_ground.field import datasets
    f = datasets.de440_kernel()
    if f is None:
        return None
    from jplephem.spk import SPK
    return SPK.open(str(f))


@functools.lru_cache(maxsize=1)
def _truth():
    """(kernel, name, edge_jd): the kernel that is reality (DE442, else DE440 with the name saying so) and the last date its fit data certainly cover."""
    from sigma_ground.field import datasets
    f, name = datasets.truth_kernel()
    if f is None:
        return None, "none", None
    from jplephem.spk import SPK
    edge = DE442_DATA_END_JD if name == "DE442" else FIT_DATA_END_JD
    return SPK.open(str(f)), name, edge


@functools.lru_cache(maxsize=1)
def _forecast():
    """The sealed table of 2026-10-27, for audits and tests only (body_position does not serve it): (table dict, manifest entry) for the newest sealed table that is installed and whose sha256 matches, else (None, reason)."""
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


def _kernel_state(kernel, name: str, jd: float):
    from sigma_ground.field.interface import reduced_model as rm
    return rm.kernel_state(kernel, name, jd)


def _table_state(table: dict, name: str, jd: float):
    from sigma_ground.field.interface import reduced_model as rm
    i = table["names"].index(name)
    p, v = rm.hermite(jd, table["jd"], table["pos"][:, i], table["vel"][:, i])
    return p, v


# What the converged step showed about two of the error table's rows (misc/CANDIDATE_LEDGER.md C19, windows J2008/J2014/J2020, +5 yr, 0.025 d -> 0.0125 d). The table is the measured error of the
# model as run at 0.025 d, so it stays the right number to quote; these say what that number is made of.
_STEP_CAVEATS = {
    "Mercury": "At the finer step of 0.0125 d this model's Mercury error against DE440 is about 2.5 times larger (0.068 against 0.027 km at +5 yr; C19): the low value at 0.025 d is partly integration error cancelling a model deficit, so it is the measured error of that run, not a floor for the physics.",
    "Moon": "At the finer step of 0.0125 d this model's Moon error against DE440 is 0.72 times as large (0.317 against 0.441 km at +5 yr; C19): a quarter of the Moon's error at the 0.025 d step is integration error.",
}


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
    fr = freshness(meta.get("model_fingerprint"), meta.get("predictor", SERVED["predictor"]))
    if fr.get("stale"):
        flag = (flag + "; " if flag else "") + "the error table itself is stale (" + fr["note"] + ")"
    n = t[str(min(ks, key=lambda k: abs(k - lead_years)))]["n_starts"]
    truth = meta.get("truth", "DE440")
    caveat = _STEP_CAVEATS.get(body) if meta.get("dt_days") == 0.025 else None          # the caveats describe the 0.025 d table; a converged-step table needs none
    return {"typical_km": m, "largest_seen_km": x, "lead_years": lead_years, "hindcast_starts_at_that_lead": n, "integration_step_days": meta.get("dt_days"), "step_caveat": caveat,
            "basis": f"median and maximum distance from {truth} of this same model started from DE440's state at past epochs {meta.get('starts')} (targets inside the fitted span), integrated with Forest-Ruth at {meta.get('dt_days')} d; a measurement of the calculation, not of this run's integrator",
            "note": flag}


# ── the live integration ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

_LIVE: dict = {}                                 # the per-process memo (see the module docstring); empty until the first question past the edge
_LIVE_LOCK = threading.Lock()


def live_key(start_jd: float, start_kernel_name: str) -> tuple:
    return (_current_fingerprint(SERVED["predictor"]), float(start_jd), start_kernel_name, SERVED["predictor"], SERVED["integrator"], float(SERVED["dt_days"]))


def _memo_matches(start_jd: float, start_kernel_name: str, horizon_days: float) -> bool:
    return bool(_LIVE) and _LIVE["key"] == live_key(start_jd, start_kernel_name) and _LIVE["horizon_days"] >= horizon_days


def _start_memo(start_jd: float, start_kernel, start_kernel_name: str, horizon_days: float) -> None:
    """Build the system from the kernel's state at start_jd and open an empty memo (row 0 = the start)."""
    from sigma_ground.field.interface import reduced_model as rm
    t0 = time.perf_counter()
    system, order = rm.build_system(start_kernel, start_jd, SERVED["predictor"], dt_days=SERVED["dt_days"], horizon_days=horizon_days)
    count = [0]
    real = system.compute_accelerations

    def counted():
        count[0] += 1
        return real()
    system.compute_accelerations = counted
    _LIVE.clear()
    _LIVE.update(key=live_key(start_jd, start_kernel_name), system=system, order=order, advance=rm._step_function(system, SERVED["integrator"]), evaluations=count, horizon_days=float(horizon_days),
                 jd=[float(start_jd)], pos=[np.array([b.position_m / 1e3 for b in system.bodies])], vel=[np.array([b.velocity_m_s / 1e3 for b in system.bodies])],
                 seconds=time.perf_counter() - t0, build_seconds=time.perf_counter() - t0, started_utc=_now().strftime("%Y-%m-%dT%H:%M:%SZ"), start_kernel=start_kernel_name,
                 force_backend=system.force_backend)


def _extend_memo(jd_target: float, deadline: float, progress=None) -> None:
    """Integrate the memo's system forward in fixed steps until jd_target is bracketed or the deadline passes; every step is stored.
    `progress(info)` is called every 20 steps with percent done, years integrated, force evaluations and seconds so far (the live count the chat shows)."""
    dt_s = SERVED["dt_days"] * DAY
    system, advance = _LIVE["system"], _LIVE["advance"]
    t0 = time.perf_counter()
    k = 0
    jd0 = _LIVE["jd"][0]
    while _LIVE["jd"][-1] < jd_target:
        advance(dt_s)
        _LIVE["jd"].append(jd0 + system.time / DAY)
        _LIVE["pos"].append(np.array([b.position_m / 1e3 for b in system.bodies]))
        _LIVE["vel"].append(np.array([b.velocity_m_s / 1e3 for b in system.bodies]))
        k += 1
        if k % 20 == 0:
            if progress is not None:
                done = _LIVE["jd"][-1] - jd0
                progress({"percent": 100.0 * min(done / max(jd_target - jd0, 1e-9), 1.0), "years_integrated": done / 365.25, "years_asked": (jd_target - jd0) / 365.25,
                          "force_evaluations": _LIVE["evaluations"][0], "seconds": time.perf_counter() - t0, "steps": len(_LIVE["jd"]) - 1})
            if time.perf_counter() > deadline:
                break
    _LIVE["seconds"] += time.perf_counter() - t0


@functools.lru_cache(maxsize=1)
def machine() -> dict:
    """The machine the calculation runs on, for the receipt: no supercomputer's timeshare, one PC."""
    import platform
    out = {"host": platform.node(), "processor": platform.processor() or platform.machine(), "python": platform.python_version(), "numpy": np.__version__, "os": f"{platform.system()} {platform.release()}"}
    try:
        import psutil
        out["cores"] = psutil.cpu_count(logical=False)
        out["threads"] = psutil.cpu_count()
    except Exception:
        pass
    return out


def _cpu_load_percent() -> float | None:
    """How busy the machine was with OTHER work while this ran (the speed exhibit has to say the load it was measured under)."""
    try:
        import psutil
        return float(psutil.cpu_percent(interval=0.2))
    except Exception:
        return None


# What each force layer is and where its numbers come from: the receipt's list. Keyed by the layer_effect table's names where the ledger measured the layer's size (C23b).
_LAYER_SOURCES = {
    "newton": "point-mass gravity of the Sun, Mercury, Venus, Earth, Moon, Mars and the Jupiter, Saturn, Uranus, Neptune and Pluto systems (a barycentre carries its moons' mass); G from CODATA (field/constants.py), masses from DE440's header (fixtures/de440_state_vectors_v2physics.json)",
    "eih": "general relativity to first post-Newtonian order, the Einstein-Infeld-Hoffmann N-body equations (Will 1993 Box 6.2; Soffel et al. 2003), the force model DE440 itself uses; implicit, time-symmetric velocity kicks",
    "zonals": "the oblateness terms J2, J3, J4 of the Sun, Earth, Moon and Mars about their IAU poles, with the reaction on the oblate body (Vallado 2013 sec. 9.4 for J2, the gradients of the P3 and P4 potentials for J3 and J4); values from the NASA Planetary Fact Sheets, Williams 2024 and Pavlis et al. 2008 (rolling_shootout._BODY_PARAMS); Jupiter's J2 from Juno and Saturn's J2 and J4 from the Cassini Grand Finale (_MEASURED_ZONALS) act only on the giants' own figure, which a barycentre does not carry",
    "tides": "the static equilibrium tide: each body's k2 Love-number bulge raised by its dominant perturber (Love 1911; the pull on every other body and the reaction), Love numbers per body from _BODY_PARAMS",
    "de_tide": "DE430's time-delayed three-band Earth tide on the Moon (Folkner et al. 2014, IPN 42-196 eq. 31-32) with DE440's header k2 and delays",
    "c22": "the Moon's triaxial figure C22 (GRAIL, 2.2382e-5 on 1738 km) as a MacCullagh quadrupole, its long axis from the IAU WGCCRE rotation model (Archinal et al. 2011), with the reaction (figure_forces.py, lunar_figure.py)",
    "asteroids": "DE440's 343 asteroid perturbers as prescribed Newtonian sources with DE440's own masses, their paths integrated as test particles from their SBDB elements over the perturber horizon (asteroid_perturbers.py, perturber_paths.py)",
    "ring": "DE440's 36-point Kuiper-belt ring (8917 km^3/s^2 in all) as prescribed sources, with DE440's mass per point",
    "kbo": "the 7 Kuiper-belt objects of DE440's 30 that could be placed, as prescribed sources with DE440's masses (the other 23 are inside the ring's mass)",
    "srp": "solar radiation pressure on each body's disc, L_sun = 3.828e26 W (IAU 2015), a body's area and reflectivity from _BODY_PARAMS",
    "gm_rate": "the Sun's GM decreasing at DE440's header GMSDOT",
}


def layers_applied(system, body: str | None = None, lead_years: float | None = None) -> list[dict]:
    """The physics layers the served system applies, each with its source and, where the ledger measured it (fixtures/layer_effect_table.json, C23b: the forecast model with that one layer
    removed, 0.0125 d, four hindcast starts, against DE440), the size of the layer for `body` at the nearest measured lead: the ratio by which the error grows without it and the error in km without it."""
    t = system.toggles
    rows = [("newton", True)]
    rows += [("eih", bool(t.eih_cross)), ("zonals", bool(t.j2_zonal or t.j3_zonal or t.j4_zonal)), ("tides", bool(t.tidal_force)), ("de_tide", bool(t.earth_tide_de)), ("c22", bool(t.sectoral_c22)),
             ("srp", bool(t.srp and system.solar_luminosity_W > 0.0)), ("gm_rate", bool(t.gm_rate))]
    n_src = int(len(system.prescribed_sources.gm)) if (t.prescribed_sources and system.prescribed_sources is not None) else 0
    rows += [("asteroids", n_src > 0), ("ring", n_src > 0), ("kbo", n_src > 0)]
    effect = None
    if body is not None and lead_years is not None:
        try:
            from sigma_ground.field.interface import layer_effect as LE
            effect = LE.layer_effect(body)
        except Exception:
            effect = None
    out = []
    for key, on in rows:
        row = {"layer": key, "applied": on, "source": _LAYER_SOURCES[key]}
        if key in ("asteroids", "ring", "kbo") and on:
            row["prescribed_sources_in_all"] = n_src
        if effect and key in effect.get("layers", {}):
            e = effect["layers"][key]
            leads = sorted(e["ratio"], key=lambda s: abs(float(s) - lead_years))
            if leads:
                ld = leads[0]
                row["measured_size"] = {"at_lead_years": float(ld), "error_ratio_without_it": e["ratio"][ld], "error_without_it_km": e["without_layer_km"][ld], "error_with_every_layer_km": effect.get("error_with_all_layers_km", {}).get(ld),
                                        "basis": "C23b ablation: this model with that one layer removed, against DE440, median of four hindcast starts, Forest-Ruth 0.0125 d"}
        out.append(row)
    return out


def _memo_state(name: str, jd: float):
    """Cubic Hermite between the two stored steps that bracket jd (the same interpolation the sealed table used; at a 0.1 d step its error is below a metre for the Moon, far less for the planets)."""
    from sigma_ground.field.interface import reduced_model as rm
    i = bisect.bisect_right(_LIVE["jd"], jd) - 1
    i = min(max(i, 0), len(_LIVE["jd"]) - 2)
    b = _LIVE["order"].index(name)
    jd2 = np.array(_LIVE["jd"][i:i + 2])
    pos2 = np.stack([_LIVE["pos"][i][b], _LIVE["pos"][i + 1][b]])
    vel2 = np.stack([_LIVE["vel"][i][b], _LIVE["vel"][i + 1][b]])
    return rm.hermite(jd, jd2, pos2, vel2)


def live_states(jd_target: float, budget_s: float = LIVE_BUDGET_S, start_jd: float | None = None, start_kernel=None, start_kernel_name: str | None = None, progress=None) -> dict:
    """Our own integration from the edge of reality (or from start_jd in start_kernel, for audits and tests: the sealed table's own start, say) to jd_target, continued from the memo when it is the same
    integration, stopped at the budget. Returns {"reached": bool, "jd_end": float, "lead_days": float, "seconds_this_call", "memo": {...}, "rate_s_per_year": float, "state": name -> (pos_km, vel_km_s) or None}."""
    if start_jd is None:
        k, name, edge = _truth()
        if k is None:
            raise WhereWhenError("no JPL kernel is installed here (jpl/de442/de442.bsp or de440s.bsp under the data volume): there is no start state to integrate from; set SIGMA_DATASETS")
        start_jd, start_kernel, start_kernel_name = edge, k, name
    if jd_target < start_jd:
        raise ValueError(f"live_states integrates forward only: target JD {jd_target} is before the start JD {start_jd}")
    horizon = max(LIVE_MIN_HORIZON_DAYS, jd_target - start_jd + 365.25)
    t_call = time.perf_counter()
    deadline = t_call + float(budget_s)
    with _LIVE_LOCK:
        fresh = not _memo_matches(start_jd, start_kernel_name, horizon)
        if fresh:
            _start_memo(start_jd, start_kernel, start_kernel_name, horizon)
        rows_before = len(_LIVE["jd"])
        evals_before = _LIVE["evaluations"][0]
        _extend_memo(jd_target, deadline, progress)
        reached = _LIVE["jd"][-1] >= jd_target
        steps_this_call = len(_LIVE["jd"]) - rows_before
        secs = time.perf_counter() - t_call
        integ_secs = max(_LIVE["seconds"] - _LIVE["build_seconds"], 1e-9)
        rate = integ_secs / max((_LIVE["jd"][-1] - _LIVE["jd"][0]) / 365.25, 1e-9)               # seconds per year of integration, measured on this run
        system = _LIVE["system"]
        out = {"reached": reached, "jd_end": _LIVE["jd"][-1], "lead_days": _LIVE["jd"][-1] - start_jd, "seconds_this_call": secs, "steps_this_call": steps_this_call,
               "force_evaluations_this_call": _LIVE["evaluations"][0] - evals_before, "rate_s_per_year": rate, "system": system, "n_bodies": len(system.bodies),
               "n_prescribed_sources": int(len(system.prescribed_sources.gm)) if system.prescribed_sources is not None else 0,
               "memo": {"fresh_this_call": fresh, "rows": len(_LIVE["jd"]), "started_utc": _LIVE["started_utc"], "start_jd_tdb": _LIVE["jd"][0], "start_kernel": _LIVE["start_kernel"],
                        "model_fingerprint": _LIVE["key"][0], "predictor": SERVED["predictor"], "integrator": SERVED["integrator"], "dt_days": SERVED["dt_days"], "force_backend": _LIVE["force_backend"],
                        "force_evaluations_total": _LIVE["evaluations"][0], "seconds_total": _LIVE["seconds"], "build_seconds": _LIVE["build_seconds"], "horizon_days": _LIVE["horizon_days"],
                        "kept": "in this process only, keyed by the physics fingerprint and the start; nothing on disk"},
               "state": {n: _memo_state(n, jd_target) for n in _LIVE["order"]} if reached else None}
    return out


def receipt(r: dict, body: str, lead_years: float, kname: str, edge: float, budget_s: float) -> dict:
    """The receipt of a live answer: what ran, where, how long, from what, with which physics (the Captain, 2026-11-02: the time it took is the exhibit; show the work, not a chart)."""
    m = r["memo"]
    return {"what": "our own N-body integration, computed when asked, on one PC",
            "machine": {**machine(), "cpu_load_percent_from_other_work": _cpu_load_percent()},
            "wall_seconds_this_call": r["seconds_this_call"], "wall_seconds_whole_integration": m["seconds_total"], "of_which_build_seconds": m["build_seconds"],
            "measured_rate_seconds_per_year": r["rate_s_per_year"], "time_budget_seconds": budget_s,
            "integrator": {"name": m["integrator"], "scheme": "Yoshida 1990 6th-order symplectic composition of velocity-Verlet, implicit time-symmetric kicks for the relativistic layer" if m["integrator"] == "y6" else m["integrator"],
                           "step_days": m["dt_days"], "force_backend": m["force_backend"]},
            "steps_this_call": r["steps_this_call"], "force_evaluations_this_call": r["force_evaluations_this_call"], "force_evaluations_whole_integration": m["force_evaluations_total"],
            "bodies_integrated": r["n_bodies"], "prescribed_sources": r["n_prescribed_sources"],
            "start": {"kernel": kname, "jd_tdb": edge, "date": _jd_to_iso(edge)[:10], "what_it_is": f"JPL's measured state: the end of the data {kname} was fitted to; the one quantity ever taken from observations"},
            "lead_years": lead_years, "predictor": m["predictor"], "model_fingerprint": m["model_fingerprint"],
            "memo": {"used": not m["fresh_this_call"], "started_utc": m["started_utc"], "rows_stored": m["rows"], "kept": m["kept"]},
            "layers": layers_applied(r["system"], body, lead_years),
            "position_between_steps": f"cubic Hermite between the two stored {m['dt_days']} d steps that bracket the instant (below a metre for the Moon, less for the planets)"}


def forget_live() -> None:
    """Drop the memo (tests)."""
    with _LIVE_LOCK:
        _LIVE.clear()


def position_at(body: str, when: str, center: str = "sun", now: datetime.datetime | None = None, budget_s: float = LIVE_BUDGET_S, progress=None) -> dict:
    """The answer dict. For a date past the edge, `progress(info)` is called as the integration runs (percent, years integrated, force evaluations, seconds) and the answer carries a `receipt`."""
    now = now or _now()
    c = center.strip().lower()
    c = {"sun": "sun", "heliocentric": "sun", "ssb": "ssb", "barycentre": "ssb", "barycenter": "ssb", "earth": "earth", "geocentric": "earth"}.get(c)
    if c is None:
        raise WhereWhenError(f"unknown center '{center}': use 'sun' (default), 'ssb' (solar-system barycentre) or 'earth'")
    planet_miss = None
    try:
        b = canonical_body(body)
    except WhereWhenError as e:
        b, planet_miss = None, e
    if b == "Earth" and c == "earth":
        raise WhereWhenError("the position of the Earth seen from the Earth is zero: pick another center ('sun' or 'ssb') or another body")
    jd, iso, how = parse_when(when, now)
    k, kname, edge = _truth()
    if k is None:
        raise WhereWhenError("no JPL kernel is installed on this machine (jpl/de442/de442.bsp or de440s.bsp under the data volume), so neither a known value nor a start state for our own calculation is available; set SIGMA_DATASETS")
    lo, hi = k[0, 3].start_jd, k[0, 3].end_jd
    if not lo <= jd <= hi:
        raise WhereWhenError(f"the {kname} kernel covers JD {lo:.1f} to {hi:.1f} ({_jd_to_iso(lo)[:4]} to {_jd_to_iso(hi)[:4]}); that date is outside it")
    if b is None:                                                                            # an asteroid, Kuiper-belt object or comet: our test-particle integration from a stated start
        from sigma_ground.field.interface import small_body_position as sbp
        try:
            out = sbp.position_at(body, jd, c, k, kname, budget_s, progress=progress)
        except sbp.SmallBodyError as e:
            raise WhereWhenError(str(e)) from planet_miss
        out.update(center={"sun": "Sun", "ssb": "solar-system barycentre", "earth": "Earth"}[c], when_read_as=how, when_utc=iso or None, when_jd_tdb=jd,
                   reality_edge={"kernel": kname, "jd_tdb": edge, "date": _jd_to_iso(edge)[:10], "note": "the planet field is the kernel's; the small body itself has no kernel value here, so the answer is a calculation at every date"})
        if out.get("position_km") is not None:
            _finish_geometry(out, np.array(out["position_km"]), c)
        return out
    kind = "known" if jd <= edge else "prediction"
    out: dict = {"body": b, "center": {"sun": "Sun", "ssb": "solar-system barycentre", "earth": "Earth"}[c], "when_read_as": how, "when_utc": iso or None, "when_jd_tdb": jd, "kind": kind, "frame": "ICRF (J2000 equatorial), km and km/s",
                 "reference_point": "planet-system barycentre (planet plus all its moons)" if b in _BARYCENTRES else "body centre", "reality_edge": {"kernel": kname, "jd_tdb": edge, "date": _jd_to_iso(edge)[:10]}, "notes": []}
    if b in _BARYCENTRES:
        out["notes"].append("For Mars to Pluto the position is the planet-system barycentre, which is what the JPL planetary kernel holds; the planet's centre differs from it by up to about a hundred km (Jupiter).")

    def combine(get):
        pb, vb = get(b)
        if c == "ssb":
            return pb, vb
        ref = "Sun" if c == "sun" else "Earth"
        pr, vr = get(ref)
        return pb - pr, vb - vr

    jpl = combine(lambda n: _kernel_state(k, n, jd))
    if kind == "known":
        p, v = jpl
        out.update(method=f"JPL {kname} ephemeris (kernel)", basis=f"JPL's fit to observations; its data end about {_jd_to_iso(edge)[:4]}, and this date is inside them", beyond_fit_data=False)
    else:
        lead = (jd - edge) / 365.25
        r = live_states(jd, budget_s=budget_s, progress=progress)
        m = r["memo"]
        out["lead_years_from_edge"] = lead
        out["receipt"] = receipt(r, b, lead, kname, edge, budget_s)
        out["live"] = {"start": f"{kname}'s state at the edge, JD {edge} ({_jd_to_iso(edge)[:10]})", "predictor": m["predictor"], "integrator": m["integrator"], "dt_days": m["dt_days"], "force_backend": m["force_backend"],
                       "model_fingerprint": m["model_fingerprint"], "seconds_this_call": r["seconds_this_call"], "steps_this_call": r["steps_this_call"], "measured_rate_s_per_year": r["rate_s_per_year"],
                       "memo": {"used": not m["fresh_this_call"], "started_utc": m["started_utc"], "rows": m["rows"], "force_evaluations_total": m["force_evaluations_total"], "seconds_total": m["seconds_total"], "kept": m["kept"]}}
        out["jpl_same_instant"] = {"kernel": kname, "position_km": [float(x) for x in jpl[0]], "velocity_km_s": [float(x) for x in jpl[1]],
                                   "note": f"{kname} after the edge of its data is JPL's own extrapolation, so this is two calculations compared, not a calculation compared with a measurement"}
        out["expected_distance_from_de440"] = expected_error(b, lead)
        if not r["reached"]:
            reached_lead = r["lead_days"] / 365.25
            remaining = (jd - r["jd_end"]) / 365.25 * r["rate_s_per_year"]
            out.update(status="over_budget", method="our own N-body integration, computed now, from the kernel's state at the edge of its data (not finished within the time budget)",
                       basis=f"{reached_lead:.3f} yr of the {lead:.3f} yr lead integrated so far at {r['rate_s_per_year']:.0f} s per year ({r['seconds_this_call']:.0f} s this call, budget {budget_s:.0f} s); about {remaining:.0f} s more are needed",
                       position_km=None, velocity_km_s=None,
                       offer={"reached_jd_tdb": r["jd_end"], "reached_date": _jd_to_iso(r["jd_end"])[:10], "reached_lead_years": reached_lead, "seconds_to_finish_estimate": remaining,
                              "how": "ask again for this date (the integration continues from where it stopped, in this process), ask for a date up to reached_date, or raise budget_s"})
            out["notes"].append("No position of ours yet: the budget ran out. jpl_same_instant is JPL's value, not ours.")
            return out
        p, v = combine(lambda n: r["state"][n])
        out.update(status="computed", method="our own N-body integration, computed now, from the kernel's state at the edge of its data",
                   basis=(f"a live calculation: {m['predictor']} layers (asteroids, ring, Kuiper-belt objects, the Moon's C22, measured zonals; Sun, planets, Moon) started from {kname}'s state at the edge, "
                          f"{m['integrator']} at {m['dt_days']} d, {lead:.3f} yr of lead; {r['steps_this_call']} steps in {r['seconds_this_call']:.1f} s this call"
                          + (f", the rest from this process's memo of the same integration (started {m['started_utc']})" if not m["fresh_this_call"] or r["steps_this_call"] < m["rows"] - 1 else "")
                          + "; the start state is JPL's, so this is a propagation from JPL's state, not an independent measurement of the sky"))
        out["cross_check"] = {"jpl_kernel": kname, "difference_km": float(np.linalg.norm(p - jpl[0])), "note": out["jpl_same_instant"]["note"]}
        if out["expected_distance_from_de440"] and out["expected_distance_from_de440"].get("note") and "stale" in out["expected_distance_from_de440"]["note"]:
            out["notes"].append("The error table was measured with an earlier state of the physics code (its fingerprint differs); it is the last measurement there is until scripts/make_prediction_error_table.py is run again.")
    out["position_km"] = [float(x) for x in p]
    out["velocity_km_s"] = [float(x) for x in v]
    _finish_geometry(out, p, c)
    return out


def _finish_geometry(out: dict, p: np.ndarray, c: str) -> None:
    """Distance, light time and (from the Earth) geometric RA/Dec, added to a finished answer."""
    r_ = float(np.linalg.norm(p))
    out["distance_km"] = r_
    out["distance_au"] = r_ / AU_KM
    out["light_time_min"] = r_ / C_KM_S / 60.0
    if c == "earth":
        out["ra_deg"] = float(math.degrees(math.atan2(p[1], p[0])) % 360.0)
        out["dec_deg"] = float(math.degrees(math.asin(p[2] / r_)))
        out["notes"].append("RA and Dec are geometric (no light-time or aberration correction), in ICRF.")
