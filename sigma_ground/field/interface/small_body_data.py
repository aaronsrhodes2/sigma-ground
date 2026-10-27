"""Asteroids, Kuiper-belt objects and comets: orbit, physical data and (for comets) the non-gravitational force model, each with its source.

Built from data already in the repo, nothing fetched here:
  * DE440's 343 asteroids and 30 Kuiper-belt objects (fixtures/de440_perturbers_v2physics.json): osculating elements from JPL SBDB, GM from the DE440 header (via body_data.py).
  * the 30 heaviest asteroids' Horizons headers (diameter-class physical data: radius, rotation period, H, G, albedo, spectral type; parsed in body_data.py).
  * 20 comets' Horizons headers (fixtures/horizons_comets/): osculating elements, nuclear radius, magnitude parameters, and the Marsden non-gravitational parameters A1, A2, A3, DT with the solution date and
    observation count, exactly as JPL gave them.

Derived quantities (perihelion, aphelion, period, mean motion) are CALCULATED here from the elements and the Sun's GM when asked, never stored (the Captain, 2026-10-26: keep calculations as calculations).
Elements are osculating at their epoch and drift; this module gives no positions.
"""
from __future__ import annotations

import difflib
import functools
import glob
import math
import re
from pathlib import Path

_FIX = Path(__file__).parent / "fixtures"
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?"            # Horizons writes EC= .5375 with no leading zero


def _norm(s: str) -> str:
    s = s.strip().lower().replace("'", "").replace("-", " ").replace("_", " ")
    s = re.sub(r"[()]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def derived_orbit(a_au: float, e: float, gm_sun_km3_s2: float | None = None) -> dict:
    """Perihelion, aphelion, period and mean motion from a and e: q = a(1-e), Q = a(1+e), P = 2 pi sqrt(a^3 / GM_sun), n = 2 pi / P. Calculated from the Sun's DE440 GM (or the one given)."""
    from sigma_ground.field.interface import de440_constants as dc
    gm = gm_sun_km3_s2 if gm_sun_km3_s2 is not None else dc.gm_km3_s2("Sun")
    au_km = dc.load()["raw"]["AU"]
    out = {"perihelion_au": a_au * (1.0 - e), "aphelion_au": a_au * (1.0 + e) if e < 1.0 else None}
    if a_au > 0.0:
        p_s = 2.0 * math.pi * math.sqrt((a_au * au_km) ** 3 / gm)
        out.update(period_days=p_s / 86400.0, period_years=p_s / 86400.0 / 365.25, mean_motion_deg_per_day=360.0 / (p_s / 86400.0))
    return out


def _parse_comet(res: str) -> dict:
    head = res[:res.find("$$SOE")] if "$$SOE" in res else res
    first = head.split("\n")
    name = ""
    for line in first:
        m = re.match(r"JPL/HORIZONS\s+(.+?)\s{2,}\d{4}-\w{3}-\d\d", line)
        if m:
            name = m.group(1).strip()
            break
    out: dict = {"horizons_name": name}
    m = re.search(r"Soln\.date:\s*(\S+)\s+# obs:\s*(\d+)\s*\(([^)]*)\)", head)
    if m:
        out["solution_date"], out["observations"], out["observation_span"] = m.group(1), int(m.group(2)), m.group(3)
    m = re.search(r"soln ref\.=\s*(\S+)", head)
    if m:
        out["solution_reference"] = m.group(1)
    el: dict = {}
    for key, val in re.findall(r"\b([A-Z][A-Z0-9]*)=\s*(" + _NUM + r"|n\.a\.)", head.split("Comet physical")[0]):
        el.setdefault(key, None if val == "n.a." else float(val))
    out["elements_raw"] = el
    phys: dict = {}
    seg = head[head.find("Comet physical"):head.find("Comet non-gravitational")] if "Comet physical" in head else ""
    for key, val in re.findall(r"\b([A-Za-z][A-Za-z0-9]*)=\s*(" + _NUM + r"|n\.a\.)", seg):
        phys.setdefault(key, None if val == "n.a." else float(val))
    out["physical_raw"] = phys
    ng: dict = {}
    seg = head[head.find("Comet non-gravitational"):head.find("COMET comments")] if "Comet non-gravitational" in head else ""
    for key, val in re.findall(r"\b([A-Za-z][A-Za-z0-9]*)=\s*(" + _NUM + r"|n\.a\.)", seg):
        ng.setdefault(key, None if val == "n.a." else float(val))
    out["nongrav_raw"] = ng
    return out


@functools.lru_cache(maxsize=1)
def comets() -> dict[str, dict]:
    """{designation: parsed header} for the 20 comets (the solution is the same in each of a comet's three files; the newest solution date wins)."""
    best: dict[str, dict] = {}
    for f in sorted(glob.glob(str(_FIX / "horizons_comets" / "horizons_*_j20*.json"))):
        import json
        des = re.search(r"horizons_(\w+?)_j20", Path(f).name).group(1)
        p = _parse_comet(json.loads(Path(f).read_text(encoding="utf-8"))["result"])
        p["designation"] = des
        p["file"] = Path(f).name
        if des not in best or p.get("solution_date", "") > best[des].get("solution_date", ""):
            best[des] = p
    return best


def _comet_entry(p: dict) -> dict:
    el, ph, ng = p["elements_raw"], p["physical_raw"], p["nongrav_raw"]
    src = f"JPL Horizons solution {p.get('solution_reference', '?')} of {p.get('solution_date', '?')} ({p.get('observations', '?')} observations, {p.get('observation_span', '?')})"
    orbit = {"epoch_jd_tdb": el.get("EPOCH"), "eccentricity": el.get("EC"), "perihelion_distance_au_given": el.get("QR"), "perihelion_time_jd_tdb": el.get("TP"), "node_deg": el.get("OM"),
             "arg_perihelion_deg": el.get("W"), "inclination_deg": el.get("IN"), "semi_major_axis_au": el.get("A"), "mean_anomaly_deg": el.get("MA"), "period_years_given": el.get("PER"), "moid_au": el.get("MOID")}
    if orbit["semi_major_axis_au"] is not None and orbit["eccentricity"] is not None:
        orbit["calculated"] = derived_orbit(orbit["semi_major_axis_au"], orbit["eccentricity"])
    physical = {"nucleus_radius_km": ph.get("RAD"), "total_magnitude_M1": ph.get("M1"), "total_magnitude_slope_k1": ph.get("k1"), "nuclear_magnitude_M2": ph.get("M2")}
    nongrav = {"A1_au_per_day2": ng.get("A1"), "A2_au_per_day2": ng.get("A2"), "A3_au_per_day2": ng.get("A3"), "DT_days": ng.get("DT"), "AMRAT_m2_per_kg": ng.get("AMRAT"),
               "g_function": {k: ng.get(k) for k in ("R0", "ALN", "NK", "NM", "NN") if ng.get(k) is not None} or "standard Marsden (the Horizons header lists no non-standard model)"}
    e = {"name": p["horizons_name"], "kind": "comet", "designation": p["designation"], "parent": "Sun", "orbit": orbit, "physical": {k: v for k, v in physical.items() if v is not None},
         "nongravitational": nongrav, "sources": {"orbit": src, "physical": src, "nongravitational": src}, "notes": []}
    if ph.get("GM") is None:
        e["mass_known"] = False
        e["notes"].append("No published mass for this comet (Horizons gives GM = n.a.).")
    e["notes"].append("Osculating elements at their epoch; non-gravitational terms are the Marsden A1 (radial), A2 (transverse), A3 (normal) accelerations at 1 au, with the time delay DT. No position is given here.")
    return e


@functools.lru_cache(maxsize=1)
def _catalog() -> dict[str, dict]:
    from sigma_ground.field.interface import body_data
    cat: dict[str, dict] = {}
    for p in comets().values():
        e = _comet_entry(p)
        full = p["horizons_name"]
        keys = {full, p["designation"], re.sub(r"^\d+[A-Z]/", "", full), re.sub(r"^\d+[A-Z]/", "", full).replace("Comet ", "")}
        for k in keys:
            cat.setdefault(_norm(k), e)
    for key, e in body_data.catalog().items():
        if e["kind"] in ("asteroid", "kuiper-belt object"):
            cat.setdefault(key, e)
    return cat


def resolve(name: str) -> tuple[dict | None, list[str]]:
    key = _norm(name)
    key = re.sub(r"^comet ", "", key)
    cat = _catalog()
    if key in cat:
        return cat[key], []
    m = re.match(r"^(\d+)\b", key)
    if m and m.group(1) in cat:
        return cat[m.group(1)], []
    return None, difflib.get_close_matches(key, list(cat), n=5, cutoff=0.6)


def describe(entry: dict) -> dict:
    """The entry as a flat answer: comets as parsed; asteroids and Kuiper-belt objects with their elements (SBDB) and the orbit quantities CALCULATED from them."""
    from sigma_ground.field.interface import body_data
    if entry["kind"] == "comet":
        return {k: v for k, v in entry.items()}
    d = body_data.describe(entry)
    el = entry.get("elements")
    out = {"name": d["name"], "kind": d["kind"], "number": d.get("number"), "parent": "Sun", "sources": d["sources"], "notes": d["notes"]}
    if el:
        orbit = {"epoch_jd_tdb": el["epoch_jd_tdb"], "semi_major_axis_au": el["a_au"], "eccentricity": el["e"], "inclination_deg": el["i_deg"], "node_deg": el["om_deg"], "arg_perihelion_deg": el["w_deg"],
                 "mean_anomaly_deg": el["ma_deg"], "calculated": derived_orbit(el["a_au"], el["e"])}
        out["orbit"] = orbit
    else:
        out["notes"] = out["notes"] + ["No orbital elements are held for this object."]
    phys = {k: d[k] for k in ("radius_km", "diameter_km", "rotation_period_h", "abs_magnitude_H", "slope_G", "albedo", "spectral_type") if k in d}
    if phys:
        out["physical"] = phys
    if "gm_km3_s2" in d:
        out["gm_km3_s2"] = d["gm_km3_s2"]
        out["mass_kg"] = d["mass_kg"]
    else:
        out["mass_known"] = False
    return out
