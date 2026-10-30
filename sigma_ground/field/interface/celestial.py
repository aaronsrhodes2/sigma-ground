"""Two-body celestial mechanics of the bodies we hold: osculating elements, Hill radius, sphere of influence, Roche limit, Lagrange points, synodic period, Hohmann transfer, steps per orbit, GR perihelion advance.

Inputs are DE440 states and GMs already in the repo, nothing is fetched:
  * planets, the Sun and the Moon at any date: the JPL DE440 kernel (Mars to Pluto are planet-SYSTEM barycentres, as the kernel holds them; their GM is the system's);
  * every other body (moons): the r8 annual snapshots (de440_state_vectors_v2physics_r8.json), taken at the snapshot epoch NEAREST the date asked, within one year, and said so.
Elements are osculating (the instantaneous two-body orbit), in the ICRF equatorial frame, with mu = GM(central) + GM(body). They are not mean elements: the Earth's centre wobbles with the Moon, the giant planets'
semi-major axes move by a fraction of a percent with each other's pull. Each function returns the number and what it was computed from, never a bare value.

Kept out of nbody.py and reduced_model.py: those are part of the physics fingerprint (reduced_model.FINGERPRINT_CODE) and nothing here changes the integration.
"""
from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import numpy as np

from ..constants import C as _C
from . import body_data, de440_constants

_FIX = Path(__file__).parent / "fixtures"
_SNAP_FILE = _FIX / "de440_state_vectors_v2physics_r8.json"
_KERNEL_BODIES = ("Sun", "Mercury", "Venus", "Earth", "Moon", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto")
_SYSTEMS = ("Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto")
STEPS_PER_ORBIT_TARGET = 300        # misc/CANDIDATE_LEDGER.md C14: the step rule, shortest period / 300
SNAPSHOT_TOLERANCE_DAYS = 365.25     # a moon's elements are taken at the snapshot nearest the date, within a year
ROCHE_FLUID_COEFF = 2.44            # Roche (1849, Memoires de l'Academie des sciences de Montpellier 1, p. 258), the same value as nbody._ROCHE_FLUID_COEFF
ROUTH_CRITICAL_MU = (1.0 - math.sqrt(23.0 / 27.0)) / 2.0   # L4 and L5 are linearly stable for mass ratio m2/(m1+m2) below this (Routh 1875; 0.03852)
DAY = 86400.0


class CelestialError(ValueError):
    """A question these functions cannot answer from the data held; the message says what is held."""


# ── states ─────────────────────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _snapshots() -> dict:
    doc = json.loads(_SNAP_FILE.read_text(encoding="utf-8"))
    out = {}
    for key, s in doc["snapshots"].items():
        bodies = {}
        for b in s["bodies"]:
            sv = b["state_vector"]
            bodies[b["name"]] = (np.array([sv["x_km"], sv["y_km"], sv["z_km"]]), np.array([sv["vx_km_s"], sv["vy_km_s"], sv["vz_km_s"]]), float(b.get("gm_km3_s2") or 0.0))
        out[key] = {"jd": float(s["epoch"]["jd_tdb"]), "iso": s["epoch"]["iso"], "bodies": bodies}
    return out


def snapshot_epochs() -> dict[str, float]:
    return {k: v["jd"] for k, v in _snapshots().items()}


def _nearest_snapshot(jd: float) -> tuple[str, dict]:
    key = min(_snapshots(), key=lambda k: abs(_snapshots()[k]["jd"] - jd))
    s = _snapshots()[key]
    if abs(s["jd"] - jd) > SNAPSHOT_TOLERANCE_DAYS:
        eps = snapshot_epochs()
        raise CelestialError(f"moons are held as annual snapshots {min(eps.values()):.1f} to {max(eps.values()):.1f} (JD TDB, 2000 to 2026); JD {jd:.1f} is more than a year from the nearest one")
    return key, s


def canonical(name: str) -> str:
    """The name the data uses (Moon, Io, Jupiter ...), through the aliases the tools already accept."""
    entry, close = body_data.resolve(name)
    if entry is None or entry.get("kind") not in ("planet", "moon", "star"):
        raise CelestialError(f"'{name}' is not a planet or moon held here" + (f" (close: {', '.join(close)})" if close else "") + "; small bodies are in small_body")
    return entry["name"]


def parent_of(name: str) -> str | None:
    entry, _ = body_data.resolve(name)
    p = entry.get("parent") if entry else None
    return p if p and p != name else None


def gm_of(name: str, *, system: bool = False) -> tuple[float, str]:
    """(GM km^3/s^2, source). system=True for a planet-system barycentre (Mars to Pluto: the planet plus its moons)."""
    if system and name in _SYSTEMS:
        return de440_constants.gm_km3_s2(f"{name} system"), "JPL DE440 header, planet-system GM (planet plus moons)"
    g = de440_constants.gm_km3_s2(name)
    if g is not None:
        return g, "JPL DE440 header"
    entry, _ = body_data.resolve(name)
    if entry and entry.get("gm_km3_s2"):
        return float(entry["gm_km3_s2"]), "; ".join(f"{k}: {v}" for k, v in entry.get("sources", {}).items() if k == "gm_km3_s2") or "sigma-ground body_data"
    return 0.0, "mass not published (GM taken as 0, the body is a test particle)"


def _kernel_pair(a: str, b: str, jd: float):
    from . import where_when
    ka = where_when._kernel()
    if ka is None:
        raise CelestialError("the JPL DE440 kernel is not installed on this machine (set SIGMA_DATASETS)")
    from . import reduced_model as rm

    def state(n):
        if n == "Earth-Moon barycentre":
            p, v = ka[0, 3].compute_and_differentiate(jd)           # the Earth-Moon barycentre about the SSB; the Earth's own centre wobbles by the Moon's pull with a month's period
            return np.asarray(p, float), np.asarray(v, float) / DAY
        return rm.kernel_state(ka, n, jd)
    try:
        pa, va = state(a)
        pb, vb = state(b)
    except Exception as e:
        if type(e).__name__ == "OutOfRangeError":                    # jplephem: the kernel's span; say it as a question the data cannot answer, not a crash
            raise CelestialError(f"the DE440 kernel held here: {e}") from e
        raise
    return np.asarray(pb) - np.asarray(pa), np.asarray(vb) - np.asarray(va)


def relative_state(body: str, central: str, jd: float) -> dict:
    """Position (km) and velocity (km/s) of `body` relative to `central`, ICRF, with mu = GM(central) + GM(body) and where the numbers came from."""
    b, c = canonical(body), canonical(central)
    if b == c:
        raise CelestialError("a body has no orbit about itself")
    if b in _KERNEL_BODIES and c in _KERNEL_BODIES:
        emb = b == "Earth" and c == "Sun"                            # the Earth's heliocentric orbit is the Earth-Moon barycentre's
        r, v = _kernel_pair(c, "Earth-Moon barycentre" if emb else b, jd)
        via_sys = c == "Sun"
        gm_c, s_c = gm_of(c, system=False)
        gm_b, s_b = (de440_constants.gm_km3_s2("Earth-Moon barycentre"), "JPL DE440 header, Earth-Moon barycentre GM") if emb else gm_of(b, system=via_sys)
        src = f"JPL DE440 kernel at JD {jd:.4f} TDB; GM {s_c} / {s_b}" + ("; the Earth's heliocentric orbit is that of the Earth-Moon barycentre (the Earth's centre alone wobbles by the Moon's pull)" if emb else "")
        if b in _SYSTEMS or c in _SYSTEMS:
            src += ". Mars to Pluto are planet-system barycentres in the kernel (the planet's centre differs by up to about a hundred km, Jupiter)"
        epoch = jd
        note = None
    else:
        key, s = _nearest_snapshot(jd)
        if b not in s["bodies"] or c not in s["bodies"]:
            raise CelestialError(f"{b if b not in s['bodies'] else c} is not in the snapshots")
        pb, vb, gmb = s["bodies"][b]
        pc, vc, gmc = s["bodies"][c]
        r, v = pb - pc, vb - vc
        gm_c = gmc or gm_of(c)[0]
        gm_b = gmb or gm_of(b)[0]
        epoch = s["jd"]
        src = f"DE440 r8 snapshot {key} (JD {epoch:.2f} TDB, {s['iso']}); body centres, GM from the satellite ephemerides"
        note = (f"elements taken at the snapshot epoch {epoch:.2f}, {abs(epoch - jd):.1f} days from the date asked" if abs(epoch - jd) > 0.5 else None)
    if gm_b == 0.0 and b not in _KERNEL_BODIES:
        note = (note + "; " if note else "") + f"{b}'s mass is not published, mu is the GM of {c} alone"
    return {"body": b, "central": c, "r_km": np.asarray(r, float), "v_km_s": np.asarray(v, float), "mu_km3_s2": gm_c + gm_b, "gm_central": gm_c, "gm_body": gm_b, "epoch_jd_tdb": epoch, "source": src, "note": note,
            "is_parent": parent_of(b) == c}


def elements_from_state(r_km, v_km_s, mu: float) -> dict:
    """Osculating two-body elements from a state, ICRF equatorial. Hyperbolic and parabolic states return a<0 or inf and no period."""
    r, v = np.asarray(r_km, float), np.asarray(v_km_s, float)
    rn, vn = float(np.linalg.norm(r)), float(np.linalg.norm(v))
    h = np.cross(r, v)
    hn = float(np.linalg.norm(h))
    eps = vn * vn / 2.0 - mu / rn
    evec = np.cross(v, h) / mu - r / rn
    e = float(np.linalg.norm(evec))
    a = -mu / (2.0 * eps) if eps != 0.0 else math.inf
    inc = math.degrees(math.acos(max(-1.0, min(1.0, h[2] / hn))))
    nvec = np.array([-h[1], h[0], 0.0])
    nn = float(np.linalg.norm(nvec))
    node = math.degrees(math.atan2(nvec[1], nvec[0])) % 360.0 if nn > 1e-12 * hn else None
    argp = None
    if nn > 1e-12 * hn and e > 1e-12:
        argp = math.degrees(math.acos(max(-1.0, min(1.0, float(nvec @ evec) / (nn * e)))))
        if evec[2] < 0:
            argp = 360.0 - argp
    out = {"a_km": a, "e": e, "i_deg": inc, "node_deg": node, "argument_of_periapsis_deg": argp, "specific_energy_km2_s2": eps, "radius_km": rn, "speed_km_s": vn}
    if eps < 0 and math.isfinite(a):
        out["period_days"] = 2.0 * math.pi * math.sqrt(a ** 3 / mu) / DAY
        out["periapsis_km"] = a * (1.0 - e)
        out["apoapsis_km"] = a * (1.0 + e)
    return out


def elements(body: str, central: str, jd: float) -> dict:
    st = relative_state(body, central, jd)
    el = elements_from_state(st["r_km"], st["v_km_s"], st["mu_km3_s2"])
    el.update({k: st[k] for k in ("body", "central", "mu_km3_s2", "epoch_jd_tdb", "source", "note", "is_parent")})
    return el


# ── what the elements give ────────────────────────────────────────────────────────────────────────────

def hill_radius(body: str, central: str, jd: float) -> dict:
    """Hill radius r_H = a (1 - e) (m / (3 M))^(1/3) of the body's gravity against the central body's tide (m, M the GMs); the standard first-order result for m << M."""
    el = elements(body, central, jd)
    if el["e"] >= 1 or "period_days" not in el:
        raise CelestialError(f"{el['body']} is not on a bound orbit about {el['central']} at that time (e = {el['e']:.3f}); there is no Hill radius")
    M, m = _heavier_central(el)
    el["hill_radius_km"] = el["periapsis_km"] * (m / (3.0 * M)) ** (1.0 / 3.0)
    el["hill_radius_at_a_km"] = el["a_km"] * (m / (3.0 * M)) ** (1.0 / 3.0)
    return el


def _gm_central(el: dict) -> float:
    return gm_of(el["central"])[0]


def _heavier_central(el: dict) -> tuple[float, float]:
    """(M, m): the central body's GM and the body's, refusing when the 'central' body is the lighter one (these formulas are for a small body about a heavier one)."""
    M = _gm_central(el)
    m = el["mu_km3_s2"] - M
    if m >= M:
        raise CelestialError(f"{el['central']} is not more massive than {el['body']}: ask for {el['body']} as the central body of {el['central']}, not the other way round")
    return M, m


def sphere_of_influence(body: str, central: str, jd: float) -> dict:
    """Laplace sphere of influence r = a (m/M)^(2/5): the radius inside which the body, not the central body, is the better centre for a two-body orbit."""
    el = elements(body, central, jd)
    if "period_days" not in el:
        raise CelestialError(f"{el['body']} is not on a bound orbit about {el['central']} at that time")
    M, m = _heavier_central(el)
    el["sphere_of_influence_km"] = el["a_km"] * (m / M) ** 0.4
    return el


def roche_limit(satellite: str, primary: str) -> dict:
    """Roche limit of `satellite` about `primary`, from the radii and GMs of body_constants. Rigid: the distance at which the primary's tidal stretch 2GM_p R_s/d^3 equals the satellite's surface gravity
    GM_s/R_s^2, d = R_s (2 M_p/M_s)^(1/3). Fluid: Roche's 2.44 R_s (M_p/M_s)^(1/3). Also the comparison with where the satellite is."""
    s, p = canonical(satellite), canonical(primary)
    if s == p:
        raise CelestialError("a body has no Roche limit about itself")
    es, _ = body_data.resolve(s)
    gms, src_s = gm_of(s)
    gmp, src_p = gm_of(p)
    rs = es.get("radius_km")
    if not gms or not rs:
        raise CelestialError(f"{s} has no published mass or radius here, so no Roche limit can be computed")
    ratio = gmp / gms
    if ratio <= 1.0:
        raise CelestialError(f"{p} is not more massive than {s}: the Roche limit is the distance of a satellite about its heavier primary (try roche_limit('{p}', '{s}'))")
    return {"satellite": s, "primary": p, "rigid_km": rs * (2.0 * ratio) ** (1.0 / 3.0), "fluid_km": ROCHE_FLUID_COEFF * rs * ratio ** (1.0 / 3.0), "satellite_radius_km": rs, "gm_primary": gmp, "gm_satellite": gms,
            "source": f"radius and GM of the bodies (body_constants); GM: {src_p} / {src_s}; rigid and fluid forms as in nbody.NBodySystem.roche_limit"}


def lagrange_points(body: str, central: str, jd: float) -> dict:
    """The five Lagrange points of the circular restricted three-body problem for the pair (central, body), at the pair's current separation, in km from the central body along the line to the body (L1, L2, L3)
    and in the orbital plane at 60 degrees (L4, L5). Found by solving the zero of the force in the rotating frame; stability of L4 and L5 by Routh's criterion."""
    el = elements(body, central, jd)
    M, m = _heavier_central(el)
    mu = m / (M + m)
    sep = el["radius_km"]                                      # the pair's separation now; circular restricted problem uses it as the unit

    def fx(x):                                                  # force along the line, units of G(M+m)/sep^2, central at -mu, body at 1-mu, rotating frame
        return x - (1 - mu) * (x + mu) / abs(x + mu) ** 3 - mu * (x - 1 + mu) / abs(x - 1 + mu) ** 3

    def root(lo, hi):
        flo, fhi = fx(lo), fx(hi)
        if flo * fhi > 0:
            raise CelestialError("no sign change bracketing a Lagrange point; the mass ratio is outside what was tested")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            fm = fx(mid)
            if fm == 0.0 or (hi - lo) < 1e-15:
                break
            if flo * fm < 0:
                hi = mid
            else:
                lo, flo = mid, fm
        return 0.5 * (lo + hi)

    eps = 1e-9
    x1 = root(-mu + 1e-12 + (1 - mu - (-mu)) * 1e-6, 1 - mu - eps)           # L1 between the two bodies
    x2 = root(1 - mu + eps, 3.0)                                              # L2 beyond the smaller body
    x3 = root(-3.0, -mu - eps)                                                # L3 opposite
    stable = mu < ROUTH_CRITICAL_MU
    el.update({"mass_ratio": mu, "separation_km": sep,
               "L1_km_from_central": (x1 + mu) * sep, "L2_km_from_central": (x2 + mu) * sep, "L3_km_from_central": (x3 + mu) * sep,
               "L4_L5_distance_km": sep, "L4_L5_angle_deg": 60.0, "L4_L5_stable": stable, "routh_critical_mass_ratio": ROUTH_CRITICAL_MU})
    return el


def synodic_period_days(body: str, other: str, central: str, jd: float) -> dict:
    """Mean time between successive alignments of two bodies about the same central body, 1/|1/P1 - 1/P2|, from the two osculating periods."""
    if canonical(body) == canonical(other):
        raise CelestialError("a body has no synodic period with itself")
    e1, e2 = elements(body, central, jd), elements(other, central, jd)
    if "period_days" not in e1 or "period_days" not in e2:
        raise CelestialError("both bodies need bound orbits about the central body")
    p1, p2 = e1["period_days"], e2["period_days"]
    if math.isclose(p1, p2, rel_tol=1e-12):
        raise CelestialError("the two periods are equal, there is no synodic period")
    return {"body": e1["body"], "other": e2["body"], "central": e1["central"], "synodic_days": 1.0 / abs(1.0 / p1 - 1.0 / p2), "period_body_days": p1, "period_other_days": p2, "epoch_jd_tdb": e1["epoch_jd_tdb"],
            "source": e1["source"]}


def hohmann(body: str, other: str, central: str, jd: float) -> dict:
    """Hohmann transfer between the two bodies' orbits taken as circular and coplanar at their current distances from the central body (the textbook idealisation; the real orbits are eccentric and inclined).
    Two burns, vis-viva; the transfer ellipse has periapsis at the inner radius and apoapsis at the outer."""
    if canonical(body) == canonical(other):
        raise CelestialError("a transfer needs two different bodies")
    e1, e2 = elements(body, central, jd), elements(other, central, jd)
    mu = gm_of(e1["central"])[0]
    r1, r2 = e1["radius_km"], e2["radius_km"]
    inner, outer = (r1, r2) if r1 <= r2 else (r2, r1)
    at = 0.5 * (inner + outer)
    v_c1, v_c2 = math.sqrt(mu / inner), math.sqrt(mu / outer)
    v_t1 = math.sqrt(mu * (2.0 / inner - 1.0 / at))
    v_t2 = math.sqrt(mu * (2.0 / outer - 1.0 / at))
    dv1, dv2 = abs(v_t1 - v_c1), abs(v_c2 - v_t2)
    return {"from": e1["body"], "to": e2["body"], "central": e1["central"], "r_from_km": r1, "r_to_km": r2, "dv1_km_s": dv1, "dv2_km_s": dv2, "dv_total_km_s": dv1 + dv2,
            "transfer_time_days": math.pi * math.sqrt(at ** 3 / mu) / DAY, "transfer_semi_major_axis_km": at, "epoch_jd_tdb": e1["epoch_jd_tdb"], "source": e1["source"],
            "assumption": "circular, coplanar orbits at the distances the two bodies have at that date"}


def gr_advance_per_orbit(mu_km3_s2: float, a_km: float, e: float) -> float:
    """First post-Newtonian periapsis advance per orbit, 6 pi mu / (c^2 a (1 - e^2)), radians (Einstein 1915; the test-particle limit with mu = GM central + GM body)."""
    c_km_s = _C / 1000.0
    return 6.0 * math.pi * mu_km3_s2 / (c_km_s ** 2 * a_km * (1.0 - e ** 2))


def gr_perihelion_advance(body: str, central: str, jd: float) -> dict:
    """The general-relativistic advance of the periapsis, 6 pi mu / (c^2 a (1 - e^2)) per orbit (first post-Newtonian, test particle limit with mu = GM central + GM body), and per century at the osculating period.
    This is the GR part only; the planets' Newtonian pull and the Sun's oblateness add their own, and the engine's layer ablations (C-ledger) carry those."""
    el = elements(body, central, jd)
    if "period_days" not in el:
        raise CelestialError(f"{el['body']} is not on a bound orbit about {el['central']} at that time")
    per_orbit = gr_advance_per_orbit(el["mu_km3_s2"], el["a_km"], el["e"])
    orbits_per_century = 36525.0 / el["period_days"]
    el["gr_advance_rad_per_orbit"] = per_orbit
    el["gr_advance_arcsec_per_century"] = math.degrees(per_orbit) * 3600.0 * orbits_per_century
    return el


# ── the step rule ────────────────────────────────────────────────────────────────────────────────────

def step_rows(central: str, step_s: float, jd: float, target: float = STEPS_PER_ORBIT_TARGET) -> list[dict]:
    """Steps per orbit that an integration step of `step_s` seconds gives each body that orbits `central` (its satellites, or the planets for the Sun), shortest period first.

    Measured on the real Jupiter system (misc/CANDIDATE_LEDGER.md C19, Forest-Ruth, +5 yr against DE440): Io at 71 steps per orbit is 118,000 km off, at 142 steps 7,610 km (a factor 15.5 for a halved step,
    convergence order 3.95); Europa and Ganymede the same. The rule of C14 is shortest period / 300."""
    c = canonical(central)
    cat = body_data.catalog()
    names = [e["name"] for e in cat.values() if e.get("parent") == c and e.get("kind") in ("moon", "planet") and e["name"] != c]
    rows = []
    for n in sorted(set(names)):
        try:
            el = elements(n, c, jd)
        except CelestialError:
            continue
        if "period_days" not in el:
            continue
        spo = el["period_days"] * DAY / step_s
        rows.append({"body": n, "period_days": el["period_days"], "steps_per_orbit": spo, "adequate": spo >= target, "target": target, "epoch_jd_tdb": el["epoch_jd_tdb"]})
    if not rows:
        raise CelestialError(f"{c} has no satellites held here (moons are held for Mars, Jupiter, Saturn, Uranus, Neptune, Pluto and the Earth; the Sun's are the planets)")
    return sorted(rows, key=lambda r: r["period_days"])
