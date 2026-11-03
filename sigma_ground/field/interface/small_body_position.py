"""Where a small body (asteroid, Kuiper-belt object, comet) is at a time: our own test-particle integration from a stated start, computed now.

The planets and the Moon are where_when.py's: the full reduced N-body model from the truth kernel's edge. A small body has no kernel here, so there is no
"known" value and the answer is always a calculation: the body as a test particle in the planet field (the Sun and the nine planet systems from the truth
kernel), the Sun's 1PN term, DE440's 343 asteroids as prescribed sources (the body's own entry zeroed) and, for comets, the Marsden non-gravitational
terms the Horizons solution gives: the C24 engine (misc/CANDIDATE_LEDGER.md C24, C27, C28; comet_orbits.propagate, perturber_paths). Nothing is read
from a table for the position; the start state is the only stored input, and the answer says which it was:

  tier 1  own_fit    OUR orbit solution from the Minor Planet Center's observations alone (fixtures/small_body_starts/own_fit_<n>.json; Eros, C28).
  tier 2  horizons   JPL's own barycentric state at 2026-01-01 TDB (fixtures/horizons_asteroids, small_body_starts/horizons_vec_<n>.json): the 30 heaviest
                     of DE440's asteroids and Apophis, the bodies the daily MPC scoreboard tracks (C27).
  tier 3  catalogue  JPL SBDB osculating elements at their epoch (small_body_data: DE440's 343 asteroids, its 30 Kuiper-belt objects) and the Horizons
                     elements and non-gravitational terms of the 20 comets.
  tier 4  sbdb       JPL SBDB osculating elements of every IAU-named main-belt asteroid (24,370; fixtures/sbdb_asteroids_named_mba.json).

A two-body state from elements is exact at the epoch; the propagation from it is ours. Where the daily scoreboard has measured this body against fresh
telescope positions (fixtures/mpc_live/scoreboard_portal.json), the latest measurement is given beside the answer: that is the only comparison with the
sky there is; no JPL kernel value exists for these bodies on this machine.
"""
from __future__ import annotations

import functools
import json
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_FIX = Path(__file__).parent / "fixtures"
_STARTS = _FIX / "small_body_starts"
DAY = 86400.0
AU_KM = 149597870.7
JD_2026 = 2461041.5                                   # 2026-01-01 TDB: the Horizons fixtures' last snapshot epoch
SECONDS_PER_YEAR_GUESS = 1.6                          # measured 2026-11-03: Eros 7 yr in 9.2 s on the PC (dopri5, rtol 1e-13, 343 sources); comets with DT iterate 2-3 passes
PATH_PAD_DAYS = 4.0


class SmallBodyError(ValueError):
    """The question cannot be answered as asked; the message says what to change."""


@dataclass(frozen=True)
class Start:
    number: int | None
    name: str
    kind: str                                         # asteroid | kuiper-belt object | comet
    tier: str                                         # own_fit | horizons | catalogue | sbdb
    jd0: float
    y0: tuple                                         # barycentric ICRF equatorial km, km/day
    source: str
    gm_km3_s2: float = 0.0
    nongrav: tuple | None = None                      # (A1, A2, A3) au/day^2 for comets
    nongrav_dt_days: float = 0.0
    law: dict | None = None
    extra: dict | None = None


# ----------------------------------------------------------------------------- the planet field
@functools.lru_cache(maxsize=1)
def gm_planets() -> list[float]:
    """GM (km^3/s^2) of the Sun then the nine planet systems, from the DE440 state-vector fixture (the same list scripts/comet_ladder.py and the scoreboard use)."""
    bodies = json.loads((_FIX / "de440_state_vectors_v2physics.json").read_text(encoding="utf-8"))["snapshots"]["j2010"]["bodies"]
    gm = {x["name"]: x["gm_km3_s2"] for x in bodies}
    anc = {x["name"]: x.get("gravitational_anchor") for x in bodies}
    sysgm = lambda n: gm[n] + sum(g for m, g in gm.items() if anc.get(m) == n)
    return [gm["Sun"], sysgm("Mercury"), sysgm("Venus"), gm["Earth"] + gm["Moon"], sysgm("Mars"), sysgm("Jupiter"), sysgm("Saturn"), sysgm("Uranus"), sysgm("Neptune"), sysgm("Pluto")]


@functools.lru_cache(maxsize=1)
def _perturbers():
    from sigma_ground.field.interface.asteroid_perturbers import Perturber
    cat = [x for x in json.loads((_FIX / "de440_perturbers_v2physics.json").read_text(encoding="utf-8"))["asteroids"] if x["elements"]]
    pert = [Perturber(str(x["number"]), x["gm_km3_s2"], 0.0, *[x["elements"][k] for k in ("epoch_jd_tdb", "a_au", "e", "i_deg", "om_deg", "w_deg", "ma_deg")]) for x in cat]
    return pert, np.array([x["number"] for x in cat]), np.array([x["gm_km3_s2"] for x in cat])


_PATHS: dict = {}
_LOCK = threading.Lock()


def _sources_for(kernel, number: int | None, lo: float, hi: float):
    """The prescribed-source table (343 asteroids, the target's own entry zeroed) covering [lo, hi], rebuilt only when the window grows; memo in this process."""
    from sigma_ground.field.interface import perturber_paths as pp
    from sigma_ground.field.interface.prescribed_sources import PrescribedSources
    pert, numbers, gm_all = _perturbers()
    w0 = math.floor(lo - PATH_PAD_DAYS)
    end = math.ceil(hi + PATH_PAD_DAYS)
    with _LOCK:
        cur = _PATHS.get("nodes")
        if cur is None or not (cur["w0"] <= w0 and cur["w0"] + cur["horizon"] >= end):
            if cur is not None:                                                   # grow the window to cover both the old and the new span
                w0, end = min(w0, cur["w0"]), max(end, cur["w0"] + cur["horizon"])
            horizon = max(end - w0, 2 * 365.25)
            t0 = time.time()
            nodes = pp.asteroid_window_paths(pert, kernel, np.array(gm_planets()), w0, float(horizon), 1.0)
            _PATHS["nodes"] = {"w0": w0, "horizon": horizon, "nodes": nodes, "seconds": time.time() - t0}
            _PATHS.pop("tables", None)
        cur = _PATHS["nodes"]
        tables = _PATHS.setdefault("tables", {})
        if number not in tables:
            gm = np.where(numbers == number, 0.0, gm_all) if number is not None else gm_all
            t = pp.to_prescribed_sources(cur["nodes"], gm, 1.0)
            tables[number] = PrescribedSources(t.gm, t.t0, t.h, t.pos, t.vel)
        return tables[number], float(cur["w0"]), cur


# ----------------------------------------------------------------------------- start states
def _horizons_vector_row(result: str, number: int, jd_tdb: float) -> np.ndarray:
    """The barycentric state (km, km/day) at jd_tdb from a Horizons VECTORS response (ported from scripts/mpc_live_score.py)."""
    m = re.search(r"Target body name:\s*(.+?)\s*\{", result)
    if m is None or not re.match(rf"\s*{number}\b", m.group(1)):
        raise SmallBodyError(f"unexpected body in the Horizons file: {m.group(1) if m else 'no name'}")
    units = re.search(r"Output units\s*:\s*(KM-[SD])", result)
    if units is None:
        raise SmallBodyError("no 'Output units' line (KM-S or KM-D expected)")
    vscale = DAY if units.group(1) == "KM-S" else 1.0
    for ln in result.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines():
        c = [x.strip() for x in ln.split(",")]
        if abs(float(c[0]) - jd_tdb) < 1e-6:
            y = np.array([float(x) for x in c[2:8]])
            y[3:] *= vscale
            if not 1e4 < np.linalg.norm(y[3:]) < 1e7:
                raise SmallBodyError(f"velocity {np.linalg.norm(y[3:]):.3g} km/day is not a solar-system orbit; units misread")
            return y
    raise SmallBodyError(f"no row at JD {jd_tdb} in the Horizons file")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _number_of(name: str) -> int | None:
    m = re.match(r"^\s*\(?(\d+)\)?(?:\s+\D.*)?$", name.strip())
    return int(m.group(1)) if m else None


@functools.lru_cache(maxsize=1)
def own_fits() -> dict[int, dict]:
    out = {}
    for f in sorted(_STARTS.glob("own_fit_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        d["_file"] = f.name
        out[int(d["number"])] = d
    return out


@functools.lru_cache(maxsize=1)
def horizons_starts() -> dict[int, Path]:
    out = {}
    for f in (_FIX / "horizons_asteroids").glob("horizons_ast_*.json"):
        out[int(f.stem.split("_")[-1])] = f
    for f in _STARTS.glob("horizons_vec_*.json"):
        out[int(f.stem.split("_")[-1])] = f
    return out


@functools.lru_cache(maxsize=1)
def horizons_names() -> dict[str, int]:
    """Normalised name -> number for the Horizons start files (their 'Target body name' line), so Apophis is found by name as well as by 99942."""
    out = {}
    for num, f in horizons_starts().items():
        m = re.search(r"Target body name:\s*\d+\s+([^(\n]+?)\s*(?:\(|\{|$)", f.read_text(encoding="utf-8")[:4000].replace("\\n", "\n"))
        if m:
            out[_norm(m.group(1))] = num
    return out


@functools.lru_cache(maxsize=1)
def _sbdb():
    d = json.loads((_FIX / "sbdb_asteroids_named_mba.json").read_text(encoding="utf-8"))
    fields = d["fields"]
    i = {k: fields.index(k) for k in ("pdes", "name", "epoch", "e", "a", "i", "om", "w", "ma", "GM")}
    by_name = {_norm(r[i["name"]]): r for r in d["data"] if r[i["name"]]}
    by_num = {int(r[i["pdes"]]): r for r in d["data"] if r[i["pdes"]] and str(r[i["pdes"]]).isdigit()}
    return i, by_name, by_num, d["metadata"]


def _elements_to_state(kernel, epoch: float, a_au: float, e: float, i_deg: float, om_deg: float, w_deg: float, ma_deg: float) -> np.ndarray:
    """Heliocentric ecliptic osculating elements -> barycentric ICRF equatorial state (km, km/day) at the epoch: the exact two-body solution plus the Sun's kernel state."""
    from sigma_ground.field.interface.asteroid_perturbers import Perturber, kepler_state_ecliptic
    from sigma_ground.field.interface.fast_ephemeris import _ECL_TO_EQ
    if not (0.0 <= e < 1.0) or a_au <= 0.0:
        raise SmallBodyError(f"the catalogue orbit has e = {e} and a = {a_au} au: parabolic and hyperbolic orbits are not propagated here")
    r, v = kepler_state_ecliptic(Perturber("x", 0.0, 0.0, epoch, a_au, e, i_deg, om_deg, w_deg, ma_deg), gm_planets()[0])
    sp, sv = kernel[0, 10].compute_and_differentiate(epoch)
    return np.concatenate([_ECL_TO_EQ @ r + np.asarray(sp, float), (_ECL_TO_EQ @ v) * DAY + np.asarray(sv, float)])


def _mean_anomaly_from_perihelion_time(epoch: float, tp: float, a_au: float) -> float:
    n = math.sqrt(gm_planets()[0] / (a_au * AU_KM) ** 3) * DAY          # rad/day
    return math.degrees((n * (epoch - tp)) % (2.0 * math.pi))


@functools.lru_cache(maxsize=1)
def planetary_moons() -> dict[str, str]:
    """Normalised moon name -> its planet, from the DE440 state-vector fixture: a bare 'Io' or 'Europa' means the moon, never the asteroid of the same name (85 Io, 52 Europa)."""
    bodies = json.loads((_FIX / "de440_state_vectors_v2physics.json").read_text(encoding="utf-8"))["snapshots"]["j2010"]["bodies"]
    return {_norm(x["name"]): x["gravitational_anchor"] for x in bodies if x.get("gravitational_anchor") not in (None, "Sun") and x["name"] not in ("Sun", "Moon")}


def lookup(name: str) -> dict | None:
    """Name or number -> {number, name, kind, tier} for a small body body_position can place, without a kernel or any state (for callers that only need to know
    whether a phrase names a solvable body, such as Mentat's position rule). None for a planet, a moon, or an unknown name. The order is resolve()'s."""
    from sigma_ground.field.interface import small_body_data as sbd
    s = name.strip()
    if not s or _norm(s) in planetary_moons():
        return None
    num = _number_of(s)
    entry, _ = sbd.resolve(s)
    if num is None and entry is not None and entry.get("number") is not None:
        num = int(entry["number"])
    if num is None and entry is None:
        num = horizons_names().get(_norm(s))
    if num is not None and num in own_fits():
        return {"number": num, "name": own_fits()[num]["name"], "kind": "asteroid", "tier": "own_fit"}
    if num is not None and num in horizons_starts():
        nm = entry["name"] if entry else next((n.title() for n, k in horizons_names().items() if k == num), s)
        return {"number": num, "name": nm, "kind": entry["kind"] if entry else "asteroid", "tier": "horizons"}
    if entry is not None and (entry.get("elements") or entry.get("kind") == "comet"):
        return {"number": num, "name": entry["name"], "kind": entry["kind"], "tier": "catalogue"}
    i, by_name, by_num, _ = _sbdb()
    row = by_num.get(num) if num is not None else by_name.get(_norm(s))
    if row is not None:
        return {"number": int(row[i["pdes"]]), "name": row[i["name"]], "kind": "asteroid", "tier": "sbdb"}
    return None


def resolve(name: str, kernel) -> Start:
    """The start state for a small body by name or number, from the highest tier that has it; SmallBodyError (with the near misses) when none does."""
    from sigma_ground.field.interface import small_body_data as sbd
    from sigma_ground.field.interface import where_when as ww
    num = _number_of(name)
    if num is None and _norm(name) in planetary_moons():
        raise SmallBodyError(f"'{name}' is a moon of {planetary_moons()[_norm(name)]}: planetary moons are not in body_position (the live model carries no moon but ours; body_skill says how well each could be predicted). "
                             f"Supported: {', '.join(ww.SUPPORTED)}, plus asteroids, Kuiper-belt objects and comets by name or number (solvable_bodies lists them); the ASTEROID of that name is reachable by its number, e.g. '85 Io', '52 Europa'.")
    entry, close = sbd.resolve(name)
    if num is None and entry is not None and entry.get("number") is not None:
        num = int(entry["number"])
    if num is None and entry is None:
        num = horizons_names().get(_norm(name))
    if num is None and entry is None:
        i, by_name, by_num, _ = _sbdb()
        row = by_name.get(_norm(name))
        if row is not None:
            num = int(row[i["pdes"]])
    # tier 1: our own orbit solution
    if num is not None and num in own_fits():
        d = own_fits()[num]
        return Start(num, d["name"], "asteroid", "own_fit", float(d["epoch_jd_tdb"]), tuple(float(x) for x in d["state_km_kmday"]),
                     f"{d['what_it_is']} Fit window {d['fit_window'][0]} to {d['fit_window'][1]}, {d['observations_used']} of {d['observations_total']} observations kept, rms {d['fit_rms_arcsec']:.2f} arcsec ({d['ledger']}).",
                     gm_km3_s2=float(entry["gm_km3_s2"]) if entry and entry.get("gm_km3_s2") else 0.0, extra={"measured": d.get("measured"), "difference_vs_jpl_state": d.get("difference_vs_jpl_state")})
    # tier 2: JPL's own state at 2026-01-01
    if num is not None and num in horizons_starts():
        f = horizons_starts()[num]
        y = _horizons_vector_row(json.loads(f.read_text(encoding="utf-8"))["result"], num, JD_2026)
        nm = entry["name"] if entry else next((n.title() for n, k in horizons_names().items() if k == num), name)
        return Start(num, nm, entry["kind"] if entry else "asteroid", "horizons", JD_2026, tuple(float(x) for x in y),
                     f"JPL Horizons barycentric state of {num} {nm} at JD {JD_2026} (2026-01-01 TDB), JPL's own orbit solution (fixtures/{f.relative_to(_FIX).as_posix()}); our calculation starts from it.",
                     gm_km3_s2=float(entry["gm_km3_s2"]) if entry and entry.get("gm_km3_s2") else 0.0)
    # tier 3: the catalogue (DE440's asteroids and KBOs, the 20 comets)
    if entry is not None:
        if entry["kind"] == "comet":
            o = entry["orbit"]
            a, e = o.get("semi_major_axis_au"), o.get("eccentricity")
            if a is None or e is None or e >= 1.0:
                raise SmallBodyError(f"{entry['name']}: the Horizons solution gives e = {e}; parabolic and hyperbolic orbits are not propagated here")
            ma = o.get("mean_anomaly_deg")
            if ma is None:
                ma = _mean_anomaly_from_perihelion_time(float(o["epoch_jd_tdb"]), float(o["perihelion_time_jd_tdb"]), float(a))
            y = _elements_to_state(kernel, float(o["epoch_jd_tdb"]), float(a), float(e), float(o["inclination_deg"]), float(o["node_deg"]), float(o["arg_perihelion_deg"]), float(ma))
            ng = entry.get("nongravitational") or {}
            A = tuple(float(ng.get(k) or 0.0) for k in ("A1_au_per_day2", "A2_au_per_day2", "A3_au_per_day2"))
            gf = ng.get("g_function") if isinstance(ng.get("g_function"), dict) else {}
            law = {k: gf[h] for k, h in (("alpha", "ALN"), ("k", "NK"), ("m", "NM"), ("n", "NN"), ("r0", "R0")) if gf.get(h) is not None} or None     # comet_truth's names for marsden_g
            return Start(None, entry["name"], "comet", "catalogue", float(o["epoch_jd_tdb"]), tuple(float(x) for x in y),
                         f"{entry['sources']['orbit']}: osculating elements at JD {o['epoch_jd_tdb']}" + (f" with the Marsden terms A1, A2, A3 = {A} au/day^2, DT = {ng.get('DT_days') or 0} d" if any(A) else "")
                         + "; the two-body state at the epoch is exact, the propagation from it is ours.",
                         nongrav=A if any(A) else None, nongrav_dt_days=float(ng.get("DT_days") or 0.0), law=law)
        el = entry.get("elements")
        if el:
            y = _elements_to_state(kernel, float(el["epoch_jd_tdb"]), float(el["a_au"]), float(el["e"]), float(el["i_deg"]), float(el["om_deg"]), float(el["w_deg"]), float(el["ma_deg"]))
            return Start(num, entry["name"], entry["kind"], "catalogue", float(el["epoch_jd_tdb"]), tuple(float(x) for x in y),
                         f"{entry['sources'].get('elements', 'JPL SBDB osculating elements')} at JD {el['epoch_jd_tdb']}; the two-body state at the epoch is exact, the propagation from it is ours.",
                         gm_km3_s2=float(entry.get("gm_km3_s2") or 0.0))
    # tier 4: every named main-belt asteroid (SBDB)
    i, by_name, by_num, meta = _sbdb()
    row = by_num.get(num) if num is not None else by_name.get(_norm(name))
    if row is not None:
        g = lambda k: float(row[i[k]])
        y = _elements_to_state(kernel, g("epoch"), g("a"), g("e"), g("i"), g("om"), g("w"), g("ma"))
        return Start(int(row[i["pdes"]]), row[i["name"]], "asteroid", "sbdb", g("epoch"), tuple(float(x) for x in y),
                     f"{meta['source']} osculating elements of {row[i['pdes']]} {row[i['name']]} at JD {row[i['epoch']]} (catalogue of {meta['created_utc'][:10]}); the two-body state at the epoch is exact, the propagation from it is ours.",
                     gm_km3_s2=float(row[i["GM"]]) if row[i["GM"]] else 0.0)
    hint = f" Near misses: {', '.join(close)}." if close else ""
    raise SmallBodyError(f"'{name}' is not a body this tool can place: not a planet, the Sun or the Moon, and not in the small-body catalogues (DE440's 343 asteroids and 30 Kuiper-belt objects, 20 comets, the 24,370 named main-belt asteroids).{hint} "
                         "Use solvable_bodies() for the roster; body_constants and small_body give masses and orbits.")


# ----------------------------------------------------------------------------- the roster
def solvable_bodies() -> dict:
    """Which bodies body_position can place, by tier, with what each tier starts from."""
    from sigma_ground.field.interface import small_body_data as sbd
    from sigma_ground.field.interface import where_when as ww
    cat = sbd._catalog()
    asts = sorted({e["name"] for e in cat.values() if e.get("kind") == "asteroid"})
    kbos = sorted({e["name"] for e in cat.values() if e.get("kind") == "kuiper-belt object"})
    comets = sorted({e["name"] for e in cat.values() if e.get("kind") == "comet"})
    _, _, by_num, meta = _sbdb()
    hz = sorted(horizons_starts())
    own = own_fits()
    return {
        "full_n_body_live": {"bodies": list(ww.SUPPORTED), "count": len(ww.SUPPORTED),
                             "how": f"the served reduced N-body model ({ww.SERVED['predictor']}, {ww.SERVED['integrator']} at {ww.SERVED['dt_days']} d) from the truth kernel's state at the edge of its data; a known kernel value before the edge"},
        "small_bodies_as_test_particles": {
            "how": "the body as a test particle in the planet field (truth kernel), the Sun's 1PN term, DE440's 343 asteroids as prescribed sources (its own entry zeroed), comets with their Marsden terms; dopri5 at rtol 1e-13 (the C24 engine)",
            "tiers_in_order": [
                {"tier": "own_fit", "start": "OUR orbit solution from MPC observations alone (C28)", "bodies": [f"{n} {d['name']}" for n, d in sorted(own.items())], "count": len(own)},
                {"tier": "horizons", "start": "JPL's own barycentric state at 2026-01-01 TDB; tracked daily against fresh MPC positions (C27)", "bodies": [f"{n}" for n in hz], "count": len(hz)},
                {"tier": "catalogue", "start": "JPL SBDB / Horizons osculating elements at their epoch", "asteroids": len(asts), "kuiper_belt_objects": len(kbos), "comets": len(comets), "examples": asts[:5] + kbos[:3] + comets[:3]},
                {"tier": "sbdb", "start": f"{meta['source']} osculating elements, every IAU-named main-belt asteroid ({meta['created_utc'][:10]})", "count": len(by_num)},
            ],
            "not_solved": ["unnamed and un-numbered asteroids (not in the catalogues)", "parabolic and hyperbolic comets (e >= 1)", "planetary moons (the full model carries no moon but ours; body_skill lists them)", "spacecraft"],
        },
        "ask": "body_position(body_name, when, central_body='sun'|'ssb'|'earth', budget_s)",
    }


# ----------------------------------------------------------------------------- the answer
def scoreboard_row(number: int | None) -> dict | None:
    """The latest line of the daily MPC scoreboard for this body (fixtures/mpc_live/scoreboard_portal.json), or None."""
    if number is None:
        return None
    f = _FIX / "mpc_live" / "scoreboard_portal.json"
    if not f.is_file():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    for b in d.get("bodies", []):
        if int(b["body"]) == int(number) and b.get("series"):
            s = max(b["series"], key=lambda r: r.get("lead_days", 0.0))
            return {"what": "both predictions scored against fresh Minor Planet Center observations (O minus C on the sky), the only comparison with reality there is for this body",
                    "scoreboard_generated_utc": d.get("generated_utc"), "ledger": d.get("ledger"), "start_of_the_scoreboard_run": b.get("start_source"),
                    "window_utc": s.get("window_utc"), "lead_years_at_window": s.get("lead_days", 0.0) / 365.25, "observations_scored": s.get("n_scored"), "stations": s.get("stations"),
                    "ours_median_arcsec": s.get("ours_median_arcsec"), "jpl_median_arcsec": s.get("jpl_median_arcsec"), "ours_minus_jpl_median_arcsec": s.get("ours_minus_jpl_median_arcsec"),
                    "telescope_scatter_sigma_arcsec": s.get("jpl_sigma_arcsec")}
    return None


def position_at(name: str, jd: float, center: str, kernel, kname: str, budget_s: float, progress=None) -> dict:
    """The small-body half of where_when.position_at: the answer dict for a body resolve() knows, at JD (TDB) `jd`, relative to `center` in ('sun', 'ssb', 'earth')."""
    from sigma_ground.field.interface import comet_orbits as co
    from sigma_ground.field.interface import where_when as ww
    st = resolve(name, kernel)
    lead = (jd - st.jd0) / 365.25
    est = SECONDS_PER_YEAR_GUESS * abs(lead) * (2.5 if st.nongrav and st.nongrav_dt_days else 1.0) + 3.0
    label = f"{st.number} {st.name}" if st.number is not None else st.name
    out: dict = {"body": label, "kind": "prediction", "frame": "ICRF (J2000 equatorial), km and km/s", "reference_point": "body centre",
                 "start": {"tier": st.tier, "jd_tdb": st.jd0, "date": ww._jd_to_iso(st.jd0)[:10], "what_it_is": st.source, "lead_years_from_start": lead, "direction": "forward" if jd >= st.jd0 else "backward"},
                 "notes": []}
    if st.tier == "own_fit":
        out["notes"].append("The start is OUR orbit solution from telescope observations alone: nothing of JPL's enters this answer.")
    else:
        out["notes"].append("The start state is JPL's (a state or osculating elements), so this is our propagation from JPL's solution, not an independent measurement of the sky.")
    if est > budget_s:
        out.update(status="over_budget", method="our own test-particle integration (not started: the estimate exceeds the time budget)",
                   basis=f"about {est:.0f} s are needed for {abs(lead):.1f} yr from the start at roughly {SECONDS_PER_YEAR_GUESS} s per year on this machine; budget {budget_s:.0f} s",
                   position_km=None, velocity_km_s=None, offer={"seconds_estimate": est, "how": "raise budget_s, or ask for a date nearer the start epoch"})
        return out
    t0 = time.time()
    lo, hi = min(jd, st.jd0), max(jd, st.jd0)
    if st.nongrav and st.nongrav_dt_days:                                     # the delayed-g pass integrates DT + 5 d beyond the targets (comet_orbits.propagate)
        lo, hi = lo - abs(st.nongrav_dt_days) - 6.0, hi + abs(st.nongrav_dt_days) + 6.0
    src, w0, nodes = _sources_for(kernel, st.number, lo, hi)
    t_paths = time.time() - t0
    kw = {}
    if st.nongrav:
        kw.update(nongrav=list(st.nongrav), law=st.law, dt_days=st.nongrav_dt_days)
    y = co.propagate(list(st.y0), st.jd0, [jd], kernel, gm_planets(), gr=True, sources=src, sources_epoch_jd=w0, **kw)[jd]
    secs = time.time() - t0
    if progress is not None:
        try:
            progress({"percent": 100.0, "years_integrated": abs(lead), "years_asked": abs(lead), "force_evaluations": None, "seconds": secs})
        except Exception:
            pass
    p, v = np.array(y[:3]), np.array(y[3:]) / DAY
    if center != "ssb":
        pr, vr = ww._kernel_state(kernel, "Sun" if center == "sun" else "Earth", jd)
        p, v = p - pr, v - vr
    pert, _, gm_all = _perturbers()
    out.update(status="computed", method="our own test-particle integration, computed now, from the stated start",
               basis=(f"a live calculation: {st.name} as a test particle in the field of the Sun and nine planet systems ({kname} kernel), the Sun's 1PN term, DE440's {len(pert)} asteroids as prescribed sources"
                      + (" (its own entry zeroed)" if st.number is not None and st.number in set(int(x.name) for x in pert) else "") + (", the Marsden non-gravitational terms" if st.nongrav else "")
                      + f"; dopri5 at rtol 1e-13, {abs(lead):.3f} yr {out['start']['direction']} from the start, {secs:.1f} s"),
               receipt={"machine": ww.machine(), "wall_seconds_this_call": secs, "of_which_perturber_paths_s": t_paths, "integrator": {"name": "dopri5", "scheme": "Dormand-Prince 5(4), adaptive, dense output", "rtol": 1e-13, "atol_km": 1e-6},
                        "planet_field": {"kernel": kname, "bodies": "Sun + Mercury, Venus, Earth-Moon, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto (system barycentres)", "gm_source": "DE440 header constants (fixtures/de440_state_vectors_v2physics.json)"},
                        "prescribed_sources": {"count": len(pert), "what": "DE440's asteroid perturbers on their own two-body-plus-planets paths (perturber_paths.asteroid_window_paths)", "window_jd": [nodes["w0"], nodes["w0"] + nodes["horizon"]], "built_in_s": nodes["seconds"], "memo_used": t_paths < 0.5},
                        "relativity": "Sun's 1PN (EIH one-body) term", "non_gravitational": ({"A_au_day2": st.nongrav, "DT_days": st.nongrav_dt_days, "law": st.law or "standard Marsden"} if st.nongrav else None),
                        "start": out["start"], "engine": "the C24 engine (comet_orbits.propagate), the one the daily MPC scoreboard (C27) and the Eros orbit fit (C28) run"})
    sb = scoreboard_row(st.number)
    out["measured_against_the_sky"] = sb if sb else {"what": "no scoreboard measurement for this body: it is not in the daily MPC set; the engine's measured accuracy on the tracked bodies is in fixtures/mpc_live/scoreboard_portal.json"}
    if st.extra and st.extra.get("measured"):
        out["start"]["measured"] = st.extra["measured"]
    out["position_km"] = [float(x) for x in p]
    out["velocity_km_s"] = [float(x) for x in v]
    return out
