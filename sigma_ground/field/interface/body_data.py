"""What we know about each body, in one place: GM, mass, size, gravity field, orbit and physical data, each with the source it came from.

Built from data already in the repo (nothing is fetched here):
  * DE440's header (de440_constants.py): the Sun, Earth, Moon GM and fields, and the planet-SYSTEM GMs.
  * the 56-body fixture (de440_state_vectors_v2physics_r8.json): the GM of each planet and moon with the satellite ephemeris it came from (JUP365, SAT441, URA111, NEP095, PLU058, MAR097, Horizons header),
    11 moons have none published and are reported as "mass unknown", never as zero.
  * the Horizons headers of the 23 added moons and the 30 heaviest asteroids (radius, albedo, density, mean orbit, rotation, H, G, spectral type).
  * DE440's 343 asteroids and 30 Kuiper-belt objects (de440_perturbers_v2physics.json): GM from the header constants MAnnnn, and the orbital elements held.
  * the predictor's own table (rolling_shootout._BODY_PARAMS) for a radius where nothing better exists, labelled as that.

A GM from JPL is exact to its published digits. A mass in kg needs G, which is known to 2.2e-5, so ``mass_kg`` carries that uncertainty and says so: the GM is the exact quantity, the mass is derived.
"""
from __future__ import annotations

import difflib
import functools
import glob
import json
import re
from pathlib import Path

_FIX = Path(__file__).parent / "fixtures"
_G = 6.6743e-11                      # m^3 kg^-1 s^-2, CODATA 2022 (scipy.constants), the same value as field.constants.G
_G_REL_UNC = 1.5e-15 / 6.6743e-11    # 2.2474e-5, CODATA 2022 standard uncertainty of G
_PLANETS = ("Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto")
_SYSTEM_KEY = {"Jupiter": "Jupiter system", "Saturn": "Saturn system", "Uranus": "Uranus system", "Neptune": "Neptune system", "Pluto": "Pluto system", "Mars": "Mars system"}


def _norm(s: str) -> str:
    s = s.strip().lower().replace("'", "").replace("’", "").replace("-", " ").replace("_", " ")
    s = re.sub(r"[()]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def mass_from_gm(gm_km3_s2: float) -> dict:
    m = gm_km3_s2 * 1e9 / _G
    return {"value": m, "uncertainty": m * _G_REL_UNC, "basis": "GM / G with G = 6.6743e-11 m^3/(kg s^2), relative uncertainty 2.2e-5 (CODATA 2022); the GM itself is exact to its published digits"}


def _parse_moon_header(res: str) -> dict:
    h = res[:res.find("$$SOE")] if "$$SOE" in res else res
    out: dict = {}

    def num(pat):
        m = re.search(pat, h)
        return float(m.group(1).replace(",", "")) if m else None
    r = num(r"Mean radius \(km\)\s*=\s*([\d.]+)")
    if r is not None:
        out["radius_km"] = r
    else:
        m = re.search(r"Radius \(km\)\s*=\s*([^\n]*?)\s{2,}", h)
        if m and m.group(1).strip():
            out["radius_text_km"] = m.group(1).strip()
    for key, pat in (("density_g_cm3", r"[Dd]ensity \(g cm\^-3\)\s*=\s*([\d.]+)"), ("albedo", r"Geometric Albedo\s*=\s*([\d.]+)"), ("period_d", r"Orbital period\s*[=~]\s*([\d.]+)"),
                     ("inclination_deg", r"Inclination, i  \(deg\)\s*[=~]\s*([\d.]+)")):
        v = num(pat)
        if v is not None:
            out[key] = v
    m = re.search(r"Semi-major axis, a \(km\)\s*[=~]\s*([\d.,]+)(\s*\(10\^3\))?", h)
    if m:
        out["semi_major_axis_km"] = float(m.group(1).replace(",", "")) * (1e3 if m.group(2) else 1.0)
    m = re.search(r"GM \(km\^3/s\^2\)\s*=\s*([\d.]+(?:[eE][-+]?\d+)?)(?:\s*\+-\s*([\d.]+(?:[eE][-+]?\d+)?))?", h)
    if m:
        out["gm_header_km3_s2"] = float(m.group(1))
        if m.group(2):
            out["gm_header_uncertainty_km3_s2"] = float(m.group(2))
    return out


def _parse_asteroid_header(res: str) -> dict:
    h = res[:res.find("$$SOE")] if "$$SOE" in res else res
    out = {}
    for key, name in (("gm_horizons_km3_s2", "GM"), ("radius_km", "RAD"), ("rotation_period_h", "ROTPER"), ("abs_magnitude_H", "H"), ("slope_G", "G"), ("albedo", "ALBEDO")):
        m = re.search(r"(?<![A-Z-])" + name + r"=\s*([-\d.]+(?:[eE][-+]?\d+)?)", h)
        if m:
            out[key] = float(m.group(1))
    m = re.search(r"STYP=\s*(\S+)", h)
    if m:
        out["spectral_type"] = m.group(1)
    return out


@functools.lru_cache(maxsize=1)
def catalog() -> dict[str, dict]:
    """{normalised key: entry}. Several keys point at one entry (name, number, 'number name')."""
    from sigma_ground.field.interface import de440_constants as dc
    from sigma_ground.field.interface import rolling_shootout as rs
    fix = json.loads((_FIX / "de440_state_vectors_v2physics_r8.json").read_text(encoding="utf-8"))["snapshots"]["j2010"]["bodies"]
    moon_hdr = {}
    for f in glob.glob(str(_FIX / "horizons_moons" / "horizons_moon_*.json")):
        name = re.search(r"horizons_moon_\d+_(.+)\.json", f.replace("\\", "/").split("/")[-1]).group(1)
        moon_hdr[name] = _parse_moon_header(json.loads(Path(f).read_text(encoding="utf-8"))["result"])
    cat: dict[str, dict] = {}
    for b in fix:
        name = b["name"]
        anchor = b.get("gravitational_anchor")
        kind = "star" if name == "Sun" else "planet" if name in _PLANETS else "moon"
        e: dict = {"name": name, "kind": kind, "parent": "Sun" if kind == "planet" else anchor, "sources": {}, "notes": []}
        gm, src = b["gm_km3_s2"], b.get("gm_source", "")
        if src == "massless test body" or gm == 0.0:
            e["mass_known"] = False
            e["notes"].append("No published mass in the data held (Horizons gives none for this body); it is treated as a massless test particle in our model. This is not a claim that its mass is zero.")
        else:
            e["gm_km3_s2"] = gm
            e["sources"]["gm_km3_s2"] = src
        if name in moon_hdr:
            for k, v in moon_hdr[name].items():
                if not k.startswith("gm_header"):
                    e[k] = v
                    e["sources"][k] = "JPL Horizons satellite physical data (header)"
            if "gm_header_uncertainty_km3_s2" in moon_hdr[name]:
                e["gm_uncertainty_km3_s2"] = moon_hdr[name]["gm_header_uncertainty_km3_s2"]
        tab = rs._BODY_PARAMS.get(name)
        if tab is not None and "radius_km" not in e and "radius_text_km" not in e:
            e["radius_km"] = tab.radius_km
            e["sources"]["radius_km"] = "predictor table (rolling_shootout._BODY_PARAMS): a Fact-Sheet-era radius, not independently sourced"
        cat[_norm(name)] = e
    # DE440 header values override the fixture's for the bodies the header covers, and add the field constants and the system GMs
    for name in ("Sun", "Earth", "Moon", "Jupiter", "Saturn"):
        d = dc.body(name)
        e = cat[_norm(name)]
        if "gm_km3_s2" in d:
            e["gm_km3_s2"] = d["gm_km3_s2"]
            e["sources"]["gm_km3_s2"] = d["source"]
        e["reference_radius_km"] = d["reference_radius_km"]
        e["sources"]["reference_radius_km"] = d["source"]
        for k in ("j2", "j3", "j4"):
            if k in d:
                e[k] = d[k]
                e["sources"][k] = d["source"]
        if name == "Moon":
            e["notes"].append("The Moon's gravity field in DE440 runs to degree 6 (C22 = 2.23809e-5 ...); only J2-J4 are shown here.")
    for n in ("Mercury", "Venus"):
        gm = dc.gm_km3_s2(n)
        cat[_norm(n)]["gm_km3_s2"] = gm
        cat[_norm(n)]["sources"]["gm_km3_s2"] = "JPL DE440 header (header.440; Park et al. 2021)"
    for n, key in _SYSTEM_KEY.items():
        e = cat[_norm(n)]
        e["system_gm_km3_s2"] = dc.gm_km3_s2(key)
        e["sources"]["system_gm_km3_s2"] = "JPL DE440 header (planet plus all its moons)"
        e["notes"].append(f"gm_km3_s2 is the planet alone ({e['sources'].get('gm_km3_s2', 'n/a')}); system_gm_km3_s2 is the planet plus its moons (DE440), the quantity the Sun sees.")
    cat[_norm("Earth-Moon barycentre")] = {"name": "Earth-Moon barycentre", "kind": "barycentre", "parent": "Sun", "gm_km3_s2": dc.gm_km3_s2("Earth-Moon barycentre"),
                                           "sources": {"gm_km3_s2": "JPL DE440 header (GMB)"}, "notes": ["Earth plus Moon."]}
    # asteroids and Kuiper-belt objects
    pc = json.loads((_FIX / "de440_perturbers_v2physics.json").read_text(encoding="utf-8"))
    ast_hdr = {}
    for f in glob.glob(str(_FIX / "horizons_asteroids" / "horizons_ast_*.json")):
        n = int(re.search(r"horizons_ast_(\d+)\.json", f.replace("\\", "/")).group(1))
        ast_hdr[n] = _parse_asteroid_header(json.loads(Path(f).read_text(encoding="utf-8"))["result"])
    for a in pc["asteroids"]:
        nm = (a.get("name") or "").strip()
        e = {"name": nm or f"asteroid {a['number']}", "number": a["number"], "kind": "asteroid", "parent": "Sun", "gm_km3_s2": a["gm_km3_s2"],
             "sources": {"gm_km3_s2": "JPL DE440 header (constant MA%04d; Park et al. 2021)" % a["number"]}, "notes": []}
        if a.get("diameter_km"):
            e["diameter_km"] = a["diameter_km"]
            e["sources"]["diameter_km"] = "the DE440 perturber catalogue (SBDB)"
        if a.get("elements"):
            e["elements"] = a["elements"]
            e["sources"]["elements"] = "JPL SBDB osculating elements at the catalogue epoch (heliocentric ecliptic J2000)"
        h = ast_hdr.get(a["number"])
        if h:
            for k, v in h.items():
                e[k] = v
                e["sources"][k] = "JPL Horizons asteroid physical parameters (header)"
            if "gm_horizons_km3_s2" in h and abs(h["gm_horizons_km3_s2"] - a["gm_km3_s2"]) > 1e-9:
                e["notes"].append(f"Horizons' own GM for this body is {h['gm_horizons_km3_s2']} km^3/s^2; the DE440 value above is the one the ephemeris was fitted with.")
        cat[_norm(str(a["number"]))] = e
        if nm:
            cat[_norm(f"{a['number']} {nm}")] = e
            holder = cat.get(_norm(nm))
            if holder is None:
                cat[_norm(nm)] = e
            elif holder is not e:                                       # a planet or moon of the same name keeps the bare name (Io, Europa, Pan, Atlas ...); the asteroid is reached by its number
                holder.setdefault("notes", []).append(f"An asteroid has the same name: ask for '{a['number']} {nm}' (or just {a['number']}) to get asteroid {a['number']} {nm}.")
    for k in pc["kbo"]:
        nm = (k.get("name") or "").strip()
        e = {"name": nm or f"DE440 Kuiper-belt object {k['id']}", "kind": "kuiper-belt object", "parent": "Sun", "gm_km3_s2": k["gm_km3_s2"],
             "sources": {"gm_km3_s2": "JPL DE440 header (constant MA%d)" % k["id"]}, "notes": []}
        if k.get("elements"):
            e["elements"] = k["elements"]
            e["sources"]["elements"] = "JPL SBDB osculating elements at the catalogue epoch (heliocentric ecliptic J2000)"
        if not nm:
            e["notes"].append("DE440 lists this object's mass but the object could not be identified by the 1 % mass match to the paper's table (ambiguous or absent), so it has no name here.")
        if k.get("match_pct") is not None:
            e["notes"].append(f"Identified by mass: matches the paper's GM to {k['match_pct']:.2f} %.")
        cat[_norm(str(k["id"]))] = e
        if nm:
            short = re.sub(r"^\d+\s+", "", re.sub(r"\s*\(.*\)", "", nm)).strip()          # '136199 Eris (2003 UB313)' -> 'Eris'
            for key in (nm, short, re.sub(r"\s*\(.*\)", "", nm), *re.findall(r"\(([^)]*)\)", nm)):        # also the provisional designation: '2002 UX25'
                cat.setdefault(_norm(key), e)
    return cat


def resolve(name: str) -> tuple[dict | None, list[str]]:
    """(entry or None, close names when not found). Accepts planet/moon names, nicknames (Luna, Terra), asteroid numbers and names ('1', 'Ceres', '1 Ceres', '(1) Ceres')."""
    from sigma_ground.field.interface.body_aliases import BODY_ALIASES
    cat = catalog()
    key = _norm(name)
    key = BODY_ALIASES.get(key, key)
    if key in cat:
        return cat[key], []
    m = re.match(r"^(\d+)\b", key)
    if m and m.group(1) in cat:
        return cat[m.group(1)], []
    return None, difflib.get_close_matches(key, list(cat), n=5, cutoff=0.6)


def describe(entry: dict) -> dict:
    """The entry as a flat dict for a tool answer, with mass derived from GM (and its uncertainty) and the sources and notes kept."""
    out = {k: v for k, v in entry.items() if k not in ("sources", "notes")}
    if "gm_km3_s2" in entry:
        out["mass_kg"] = mass_from_gm(entry["gm_km3_s2"])
    if "system_gm_km3_s2" in entry:
        out["system_mass_kg"] = mass_from_gm(entry["system_gm_km3_s2"])
    out["sources"] = dict(entry.get("sources", {}))
    if "mass_kg" in out:
        out["sources"]["mass_kg"] = out["mass_kg"]["basis"]
    out["notes"] = list(entry.get("notes", []))
    return out
