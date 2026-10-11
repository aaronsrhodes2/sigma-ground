"""Instant ephemeris: where will a body be, from Earth's view or from the body's own view.

``where("Mars", days_ahead=120)`` answers in well under a millisecond from a closed-form model: one Keplerian ellipse per body
whose six elements drift linearly in time, plus a short sum of sinusoids at combinations of the planets' own mean motions
(the same structure as the classic analytic planetary theories, with far fewer terms). Nothing is integrated and nothing is
looked up at query time.

The coefficients are produced offline by ``scripts/fit_fast_ephemeris.py``, which fits them to JPL DE440 over a training span and
scores them on dates it never saw; the error table that comes with the coefficients is what ``where`` reports as its
uncertainty. This module is pure numpy and has no dependence on the fitting code or on the DE440 kernel.

Frames: heliocentric / geocentric vectors are ICRF (J2000 equatorial), km. The fit works in the J2000 ecliptic frame; the rotation
uses the IAU 2006 obliquity ``OBLIQUITY_J2000_ARCSEC`` (a defined convention, not a measurement).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

J2000 = 2451545.0
C_KM_S = 299792.458                      # SI definition of the metre: exact
AU_KM = 149597870.7                      # IAU 2012 definition: exact
OBLIQUITY_J2000_ARCSEC = 84381.406       # IAU 2006 convention
_EPS = math.radians(OBLIQUITY_J2000_ARCSEC / 3600.0)
_ECL_TO_EQ = np.array([[1.0, 0.0, 0.0],
                       [0.0, math.cos(_EPS), -math.sin(_EPS)],
                       [0.0, math.sin(_EPS), math.cos(_EPS)]])

DEFAULT_COEFFS = Path(__file__).parent / "fixtures" / "fast_ephemeris" / "fast_ephemeris_coeffs.json"

# element order in a parameter block: a [km], e, i [rad], Omega [rad], varpi [rad], L [rad]; each is (value at J2000, per century)
ELEMENTS = ("a", "e", "i", "Omega", "varpi", "L")
PLANETS = ("Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune")
_ALIASES = {"emb": "EMB", "earth-moon barycenter": "EMB"}


def kepler_ecliptic(p: np.ndarray, t_days: np.ndarray) -> np.ndarray:
    """Position (km) in the J2000 ecliptic frame from drifting elements ``p`` (6 x polynomial coefficients per century), shape (N, 3)."""
    t = np.atleast_1d(np.asarray(t_days, dtype=float))
    T = t / 36525.0
    pw = [np.ones_like(T)]
    for _ in range(p.shape[1] - 1):
        pw.append(pw[-1] * T)
    a, e, inc, Om, vp, L = (sum(p[k, j] * pw[j] for j in range(p.shape[1])) for k in range(6))
    M = np.mod(L - vp + math.pi, 2.0 * math.pi) - math.pi
    E = M + e * np.sin(M)
    for _ in range(8):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    xp = a * (np.cos(E) - e)
    yp = a * np.sqrt(1.0 - e * e) * np.sin(E)
    w = vp - Om
    cw, sw, cO, sO, ci, si = np.cos(w), np.sin(w), np.cos(Om), np.sin(Om), np.cos(inc), np.sin(inc)
    x = (cw * cO - sw * sO * ci) * xp + (-sw * cO - cw * sO * ci) * yp
    y = (cw * sO + sw * cO * ci) * xp + (-sw * sO + cw * cO * ci) * yp
    z = (sw * si) * xp + (cw * si) * yp
    return np.stack([x, y, z], axis=-1)


def periodic_sum(freqs: np.ndarray, coef: np.ndarray, poly: np.ndarray, t_days: np.ndarray) -> np.ndarray:
    """Sum of sinusoids plus a constant and linear drift, per ecliptic axis.

    freqs (K,) rad/day; coef (K, 2, 3) = cos and sin amplitudes in km; poly (2, 3) = constant and per-century drift in km.
    """
    t = np.atleast_1d(np.asarray(t_days, dtype=float))
    out = poly[0][None, :] + (t / 36525.0)[:, None] * poly[1][None, :]
    if len(freqs):
        ph = np.outer(t, freqs)
        out = out + np.cos(ph) @ coef[:, 0, :] + np.sin(ph) @ coef[:, 1, :]
    return out


def jd_now() -> float:
    import time
    return time.time() / 86400.0 + 2440587.5


def _unit_to_radec(v: np.ndarray) -> tuple[float, float]:
    r = float(np.linalg.norm(v))
    ra = math.degrees(math.atan2(v[1], v[0])) % 360.0
    dec = math.degrees(math.asin(max(-1.0, min(1.0, v[2] / r))))
    return ra, dec


class FastEphemeris:
    """Closed-form ephemeris for the eight planets, the Moon and the Sun, fitted to DE440."""

    def __init__(self, coeffs_path: str | Path | None = None, terms: int | None = None, doc: dict | None = None) -> None:
        if doc is None:
            path = Path(coeffs_path) if coeffs_path else DEFAULT_COEFFS
            doc = json.loads(path.read_text())
        self._doc = doc
        self.meta = self._doc["metadata"]
        self._terms = terms
        self._emrat_inv = float(self.meta["earth_from_emb_over_moon_geo"])   # |Earth-EMB| / |Moon-Earth|, derived from the kernel
        self._cache: dict = {}

    # ---- model pieces -------------------------------------------------------------------------------------------------
    def _block(self, key: str) -> tuple:
        b = self._doc["bodies"][key]
        ladder = b["ladder"]
        want = self._terms if self._terms is not None else int(b.get("default_terms", max(int(k) for k in ladder)))
        k = max((int(x) for x in ladder if int(x) <= want), default=min(int(x) for x in ladder))
        return b, ladder[str(k)], k

    def _arrays(self, key: str):
        b, lad, k = self._block(key)
        hit = self._cache.get((key, k))
        if hit is None:
            hit = (np.array(b["elements"], dtype=float) if b.get("elements") else None, np.array(lad["freq_rad_day"], dtype=float),
                   np.array(lad["coef_km"], dtype=float).reshape(-1, 2, 3), np.array(lad["poly_km"], dtype=float).reshape(2, 3))
            self._cache[(key, k)] = hit
        return hit

    def _ecl(self, key: str, jd: np.ndarray) -> np.ndarray:
        p, freqs, coef, poly = self._arrays(key)
        t = np.atleast_1d(np.asarray(jd, dtype=float)) - J2000
        reflex = self._doc["bodies"][key].get("reflex")
        if reflex:                                            # the Sun about the barycentre: a fixed combination of the giants' orbits
            out = sum(c * self._ecl(name, jd) for name, c in reflex.items())
            return out + periodic_sum(freqs, coef, poly, t)
        base = kepler_ecliptic(p, t) if p is not None else 0.0
        return base + periodic_sum(freqs, coef, poly, t)

    def heliocentric(self, body: str, jd) -> np.ndarray:
        """ICRF heliocentric position in km, shape (N, 3) (or (3,) for scalar jd)."""
        scalar = np.ndim(jd) == 0
        name = self._canon(body)
        if name == "Sun":
            out = np.zeros((np.atleast_1d(jd).size, 3))
        elif name == "Moon":                                  # heliocentric Moon = Earth + geocentric Moon
            out = self.heliocentric("Earth", jd).reshape(-1, 3) + self.geocentric_moon(jd).reshape(-1, 3)
        elif name == "Earth":
            emb = self._ecl("EMB", jd) @ _ECL_TO_EQ.T
            moon = self._ecl("Moon", jd) @ _ECL_TO_EQ.T
            out = emb - self._emrat_inv * moon
        elif self._doc["bodies"][name].get("frame") == "bary":   # outer planets: orbit about the barycentre, minus the Sun's wobble
            out = (self._ecl(name, jd) - self._ecl("SunBary", jd)) @ _ECL_TO_EQ.T
        else:
            out = self._ecl(name, jd) @ _ECL_TO_EQ.T
        return out[0] if scalar else out

    def geocentric_moon(self, jd) -> np.ndarray:
        scalar = np.ndim(jd) == 0
        out = self._ecl("Moon", jd) @ _ECL_TO_EQ.T
        return out[0] if scalar else out

    @staticmethod
    def _canon(body: str) -> str:
        s = str(body).strip()
        low = s.lower()
        if low in _ALIASES:
            return _ALIASES[low]
        for n in PLANETS + ("Sun", "Moon"):
            if n.lower() == low:
                return n
        raise ValueError(f"unknown body {body!r}; available: {', '.join(PLANETS + ('Moon', 'Sun'))}")

    # ---- observer-relative answers ------------------------------------------------------------------------------------
    def _relative(self, target: str, observer: str, jd: float, light_time: bool) -> tuple[np.ndarray, float]:
        """Vector observer -> target (ICRF, km) with the target taken at the retarded time; returns (vector, light-time s)."""
        obs = self.heliocentric(observer, jd)
        tau = 0.0
        vec = self.heliocentric(target, jd) - obs
        if light_time:
            for _ in range(3):
                tau = float(np.linalg.norm(vec)) / C_KM_S
                vec = self.heliocentric(target, jd - tau / 86400.0) - obs
        return vec, float(np.linalg.norm(vec)) / C_KM_S

    def geocentric(self, body: str, jd: float, light_time: bool = True) -> np.ndarray:
        return self._relative(self._canon(body), "Earth", jd, light_time)[0]

    def uncertainty(self, body: str, jd: float) -> dict:
        """Measured error for this body at this distance from the training span: median and 95th percentile, in arcsec and km."""
        tbl = self._doc["errors"].get(self._canon(body))
        if not tbl:
            return {}
        end = float(self.meta["train_end_jd"])
        yrs = abs(jd - end) / 365.25 if jd > end else 0.0
        for row in tbl["buckets"]:
            if yrs <= row["to_years"]:
                return {"bucket": f"{row['from_years']}-{row['to_years']} yr past the training span",
                        "median_arcsec": row["median_arcsec"], "p95_arcsec": row["p95_arcsec"],
                        "median_km": row["median_km"], "p95_km": row["p95_km"]}
        row = tbl["buckets"][-1]
        return {"bucket": f"beyond {row['to_years']} yr past the training span: NOT VALIDATED, last bucket shown",
                "median_arcsec": row["median_arcsec"], "p95_arcsec": row["p95_arcsec"],
                "median_km": row["median_km"], "p95_km": row["p95_km"]}

    def where(self, body: str, days_ahead: float = 0.0, *, jd: float | None = None, view: str = "earth",
              light_time: bool = True) -> dict:
        """Where ``body`` is ``days_ahead`` days after ``jd`` (default: now), from Earth's view or from the body's own view.

        view="earth": direction (right ascension, declination, ICRF), distance, light time, and the measured uncertainty.
        view="body":  the body's view of the sky: the direction and distance to Earth and to the Sun, and the angle between
                      them (the Earth-Sun separation in that body's sky, i.e. whether Earth is lost in the Sun's glare).
        """
        name = self._canon(body)
        t = (jd_now() if jd is None else float(jd)) + float(days_ahead)
        unc = self.uncertainty(name, t)
        base = {"body": name, "jd": t, "view": view, "uncertainty": unc,
                "frame": "ICRF/J2000 equatorial", "model": self.meta["model_name"]}
        if view == "earth":
            if name == "Earth":
                raise ValueError("Earth's view of Earth is undefined")
            vec, tau = self._relative(name, "Earth", t, light_time)
            ra, dec = _unit_to_radec(vec)
            d = float(np.linalg.norm(vec))
            base.update({"ra_deg": ra, "dec_deg": dec, "distance_km": d, "distance_au": d / AU_KM, "light_time_s": tau,
                         "vector_km": vec.tolist()})
            return base
        if view == "body":
            if name == "Sun":
                raise ValueError("the Sun's own view is not modelled (no pole or rotation data in this model)")
            to_earth, tau_e = self._relative("Earth", name, t, light_time)
            to_sun, tau_s = self._relative("Sun", name, t, light_time)
            ra_e, dec_e = _unit_to_radec(to_earth)
            ra_s, dec_s = _unit_to_radec(to_sun)
            cosang = float(np.dot(to_earth, to_sun) / (np.linalg.norm(to_earth) * np.linalg.norm(to_sun)))
            base.update({"earth": {"ra_deg": ra_e, "dec_deg": dec_e, "distance_km": float(np.linalg.norm(to_earth)),
                                   "light_time_s": tau_e},
                         "sun": {"ra_deg": ra_s, "dec_deg": dec_s, "distance_km": float(np.linalg.norm(to_sun)),
                                 "light_time_s": tau_s},
                         "earth_sun_separation_deg": math.degrees(math.acos(max(-1.0, min(1.0, cosang))))})
            return base
        raise ValueError("view must be 'earth' or 'body'")


_DEFAULT: FastEphemeris | None = None


def where(body: str, days_ahead: float = 0.0, **kw) -> dict:
    """Module-level convenience: ``where("Mars", 120)``; see :meth:`FastEphemeris.where`."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = FastEphemeris()
    return _DEFAULT.where(body, days_ahead, **kw)
