"""SGEphemeris -- SSBM-patched ephemeris adapter.

Provides planet/body positions using a priority chain:

  1. DE440 annual snapshot fixture   (exact; only AT the 27 snapshot epochs,
                                      J2000.0-J2026.0)
  2. SSBM 30-year forecast fixture   (SSBM's own N-body forecast; only AT its
                                      stored monthly epochs, 2025-2055)
  3. jplephem kernel                 (exact DE440, if jplephem + de440s.bsp
                                      are installed; 1849-2150; Mars..Pluto
                                      converted from system barycentre to
                                      planet centre)
  4. Keplerian orbit fit             (last resort, no external deps; any epoch;
                                      a satellite's fitted orbit is added to
                                      its planet as served by tiers 1-4)

A tier serves a query only when it holds a value for that epoch: the
fixtures are samples, not continuous ephemerides, so an epoch between two
samples falls through to the kernel or the Kepler fit instead of being
snapped to the nearest sample.

All positions returned in ICRF barycentric equatorial km.  The tier that
serves the body also supplies the Sun, so sg_heliocentric returns the
tier's own heliocentric vector.  Where a tier stores heliocentric vectors
only (forecast, Kepler), the Sun's barycentric position comes from the
definition of the barycentre: GM_sun r_sun + sum_i GM_i (r_sun + h_i) = 0,
with DE440 system GMs.
sg_ prefix marks all SSBM-patched query methods.

Usage
-----
    from sigma_ground.field.interface.adapters.ephemeris import SGEphemeris

    eph = SGEphemeris()

    # Query a planet at any JD (past or future):
    pos_km = eph.sg_position("Mars", jd=2461000.0)  # (x, y, z) in km

    # Heliocentric (Sun subtracted):
    hel_km = eph.sg_heliocentric("Earth", jd=2461000.0)

    # With the served tier's error estimate:
    pos_km, err_au = eph.sg_position_with_uncertainty("Jupiter", jd=2465000.0)
"""

from __future__ import annotations

import bisect
import json
import math
from pathlib import Path

import numpy as np

from sigma_ground.field.interface.orbital import (
    ANCHOR_MAP, DE440_GM, SATELLITE_GM, fit_orbit, _ecliptic_to_equatorial,
)

AU_KM = 1.495978707e8
KM_TO_M = 1000.0

_FIXTURES = Path(__file__).parent.parent / "fixtures"
_DE440_FILE    = _FIXTURES / "de440_state_vectors.json"
_FORECAST_FILE = _FIXTURES / "ssbm_future_predictions.json"

# Two epochs closer than this are the same epoch (0.0864 s; a double JD near
# 2.46e6 resolves 4.7e-10 d).  Earth moves 2.6 km in that time.
_EPOCH_TOL_D = 1e-6

# Heliocentric bodies whose DE440 system GMs place the Sun relative to the
# solar-system barycentre.  Earth carries the Earth-Moon GM (its centre is
# within 4.7e3 km of the EMB: a 0.014 km effect on the Sun).
_BARYCENTRE_BODIES = ("Mercury", "Venus", "Earth", "Mars", "Jupiter",
                      "Saturn", "Uranus", "Neptune", "Pluto")
_BARYCENTRE_GM: dict[str, float] = {
    name: (DE440_GM["Earth"] + DE440_GM["Moon"] if name == "Earth"
           else DE440_GM[name])
    for name in _BARYCENTRE_BODIES
}

# Kepler fits are deterministic functions of the fixture: share them across
# instances (each planet fit costs ~0.5 s).
_FIT_CACHE: dict[str, object] = {}

# de440s gives Mars..Pluto as planetary-system barycentres (NAIF 4-9), not
# planet centres; the kernel tier converts them (_barycentre_offset_km).
_KERNEL_BARYCENTRE_PLANETS = ("Mars", "Jupiter", "Saturn", "Uranus",
                              "Neptune", "Pluto")

# Measured error (km) of the kernel tier's planet centre, per planet (cache;
# see SGEphemeris._kernel_centre_bound_km).
_KERNEL_BOUND_CACHE: dict[str, float] = {}


def sun_ssb_from_heliocentric(hel_km: dict[str, np.ndarray]) -> np.ndarray:
    """Sun's barycentric position from heliocentric planet positions (km).

    The barycentre definition GM_sun r_sun + sum_i GM_i (r_sun + h_i) = 0
    gives r_sun = -sum_i GM_i h_i / (GM_sun + sum_i GM_i).  GM_i are DE440
    system values (Park et al. 2021, AJ 161, 105); bodies missing from
    hel_km are omitted (Pluto: about 40 km).  Bound: with DE440's own planet
    positions (fixture, 27 epochs) this reproduces DE440's Sun to 116-118 km;
    the residual is the trans-Neptunian mass DE440 carries and this sum
    omits (it points to RA 1.6 h, Dec ~0 deg, where Eris is: GM ~1.1e3
    km^3/s^2 at ~96 AU gives ~120 km).
    """
    num = np.zeros(3)
    den = DE440_GM["Sun"]
    for name, h in hel_km.items():
        gm = _BARYCENTRE_GM.get(name)
        if gm is None:
            continue
        num += gm * np.asarray(h, dtype=float)
        den += gm
    return -num / den


class SGEphemeris:
    """SSBM-patched ephemeris with four-tier fallback.

    Parameters
    ----------
    use_kernel : consult the jplephem kernel tier (default True).  False
                 reproduces a machine without de440s.bsp.

    Thread-safety: the fixture caches are loaded on first access (lazy).
    The object is safe to create once and reuse across calls.
    """

    def __init__(self, use_kernel: bool = True) -> None:
        self._de440:    dict | None = None
        self._forecast: dict | None = None
        self._forecast_jds: list[float] | None = None
        self._kepler_cache: dict = _FIT_CACHE
        self._use_kernel = use_kernel
        self._names: dict[str, str] | None = None

    # ── data loaders ─────────────────────────────────────────────────────

    def _load_de440(self) -> dict:
        if self._de440 is None:
            with open(_DE440_FILE) as f:
                self._de440 = json.load(f)
        return self._de440

    def _load_forecast(self) -> dict:
        if self._forecast is None:
            if _FORECAST_FILE.exists():
                with open(_FORECAST_FILE) as f:
                    self._forecast = json.load(f)
            else:
                self._forecast = {}
            preds = self._forecast.get("predictions", [])
            self._forecast_jds = [float(e["jd"]) for e in preds]
        return self._forecast

    def _kepler(self, body_name: str):
        """Cached Keplerian orbit fit for body_name."""
        if body_name not in self._kepler_cache:
            data = self._load_de440()
            orbit = fit_orbit(body_name, data)
            self._kepler_cache[body_name] = orbit
        return self._kepler_cache[body_name]

    def _canonical(self, body_name: str) -> str:
        """Fixture spelling of a body name (lookups ignore case)."""
        if self._names is None:
            names = set()
            for snap in self._load_de440()["snapshots"].values():
                names.update(b["name"] for b in snap["bodies"])
            self._names = {n.lower(): n for n in names}
        return self._names.get(str(body_name).lower(), body_name)

    # ── internal: per-tier lookups ────────────────────────────────────────

    def _snapshot_at(self, jd: float) -> dict | None:
        """The DE440 snapshot taken at jd (within _EPOCH_TOL_D), or None."""
        for snap in self._load_de440()["snapshots"].values():
            if abs(snap["epoch"]["jd_tdb"] - jd) <= _EPOCH_TOL_D:
                return snap
        return None

    @staticmethod
    def _snap_pos(snap: dict, body_name: str) -> np.ndarray | None:
        for b in snap["bodies"]:
            if b["name"] == body_name:
                sv = b["state_vector"]
                return np.array([sv["x_km"], sv["y_km"], sv["z_km"]])
        return None

    def _de440_position(self, body_name: str, jd: float) -> np.ndarray | None:
        """Exact DE440 snapshot position (SSB-equatorial, km).

        Served only at a snapshot epoch: one day away Earth has moved
        2.6e6 km, so a nearby snapshot is not an answer."""
        snap = self._snapshot_at(jd)
        if snap is None:
            return None
        return self._snap_pos(snap, body_name)

    def _forecast_entry(self, jd: float) -> dict | None:
        """The stored forecast entry at jd (within _EPOCH_TOL_D), or None.

        Uses each entry's stored jd (the forecast steps whole days, so the
        stored dates sit up to 0.5 d off start + m*month_days).  Epochs
        outside the forecast, or between two stored months, get None."""
        fc = self._load_forecast()
        jds = self._forecast_jds
        if not fc or not jds:
            return None
        i = bisect.bisect_left(jds, jd)
        for k in (i - 1, i):
            if 0 <= k < len(jds) and abs(jds[k] - jd) <= _EPOCH_TOL_D:
                return fc["predictions"][k]
        return None

    def _forecast_heliocentric(self, body_name: str, jd: float) -> np.ndarray | None:
        entry = self._forecast_entry(jd)
        if entry is None:
            return None
        ssbm = entry.get("ssbm", {})
        if body_name == "Sun":
            return np.zeros(3) if ssbm else None
        if body_name not in ssbm:
            return None
        return np.array(ssbm[body_name], dtype=float)

    def _forecast_sun_ssb(self, jd: float) -> np.ndarray | None:
        """The Sun's SSB position implied by the forecast's own planets."""
        entry = self._forecast_entry(jd)
        if entry is None or not entry.get("ssbm"):
            return None
        hel = {n: np.array(v, dtype=float) for n, v in entry["ssbm"].items()}
        return sun_ssb_from_heliocentric(hel)

    def _forecast_position(self, body_name: str, jd: float) -> np.ndarray | None:
        """SSBM forecast heliocentric position → SSB (km), at a stored epoch.

        Forecast stores heliocentric equatorial coords.  The Sun's SSB
        position at the same epoch comes from the forecast's own planets
        (barycentre definition), not from a DE440 snapshot of another date.
        """
        hel = self._forecast_heliocentric(body_name, jd)
        if hel is None:
            return None
        sun = self._forecast_sun_ssb(jd)
        if sun is None:
            return None
        return hel + sun

    def _kepler_heliocentric(self, body_name: str, jd: float) -> np.ndarray | None:
        """Heliocentric equatorial position (km) from the Kepler fits.

        A moon is its anchor's heliocentric position at jd plus its own
        anchor-relative orbit at jd (both evaluated at the same epoch)."""
        if body_name == "Sun":
            return np.zeros(3)
        orbit = self._kepler(body_name)
        if orbit is None:
            return None
        rel = np.array(_ecliptic_to_equatorial(*orbit.predict_relative_ecliptic(jd)))
        if orbit.anchor == "Sun":
            return rel
        anchor = self._kepler_heliocentric(orbit.anchor, jd)
        if anchor is None:
            return None
        return anchor + rel

    def _kepler_sun_ssb(self, jd: float) -> np.ndarray | None:
        hel = {}
        for name in _BARYCENTRE_BODIES:
            h = self._kepler_heliocentric(name, jd)
            if h is None:
                return None
            hel[name] = h
        return sun_ssb_from_heliocentric(hel)

    def _kepler_position(self, body_name: str, jd: float) -> np.ndarray | None:
        """Keplerian orbit prediction (SSB-equatorial, km) at jd.

        Anchor and Sun are evaluated at jd too (before 2026-10-10 the anchor
        was frozen at the latest snapshot: the Moon came out 2 AU off)."""
        hel = self._kepler_heliocentric(body_name, jd)
        if hel is None:
            return None
        sun = self._kepler_sun_ssb(jd)
        if sun is None:
            return None
        return hel + sun

    def _barycentre_offset_km(self, planet: str, jd: float) -> np.ndarray:
        """System barycentre minus planet centre (km, equatorial) at jd.

        From the barycentre definition GM_sys r_bary = GM_p r_p + sum_i GM_i
        r_i with r_i = r_p + r_i,rel:  r_bary - r_p = sum_i GM_i r_i,rel /
        GM_sys.  GM_sys: DE440 system GM; GM_i: SATELLITE_GM; r_i,rel: the
        Kepler fits of the planet's moons in ANCHOR_MAP (relative to the
        planet centre).  Moons without a fit are omitted (Saturn: Iapetus,
        Rhea, Dione, Tethys, Mimas, ~12 km together).
        """
        num = np.zeros(3)
        for moon, anchor in ANCHOR_MAP.items():
            if anchor != planet or moon not in SATELLITE_GM:
                continue
            orbit = self._kepler(moon)
            if orbit is None:
                continue
            rel = np.array(_ecliptic_to_equatorial(
                *orbit.predict_relative_ecliptic(jd)))
            num += SATELLITE_GM[moon] * rel
        return num / DE440_GM[planet]

    def _jplephem_position(self, body_name: str, jd: float) -> np.ndarray | None:
        """jplephem kernel lookup (km) -- requires jplephem + downloaded kernel.

        Mars..Pluto: de440s holds the system barycentre; the planet centre
        is barycentre - _barycentre_offset_km.  FIXED 2026-10-10: the
        barycentre was returned as the planet (Pluto 2131 km, Uranus up to
        1163 km, Saturn ~300 km from the fixture's planet centres) with
        uncertainty 0.  The remaining error is _kernel_centre_bound_km.
        First use fits the planet's moons (about 1-3 s per planet, cached
        per process).
        """
        if not self._use_kernel:
            return None
        try:
            from sigma_ground.field.interface.adapters._jplephem_bridge import (
                query_jplephem,
            )
            pos = query_jplephem(body_name, jd)
        except Exception:
            return None
        if pos is None:
            return None
        name = self._canonical(body_name)
        if name in _KERNEL_BARYCENTRE_PLANETS:
            pos = pos - self._barycentre_offset_km(name, jd)
        return pos

    def _kernel_centre_bound_km(self, planet: str) -> float:
        """Measured error (km) of the kernel tier's planet centre.

        The largest |kernel-tier centre - Horizons planet centre| over the
        27 DE440 snapshots and the fixture's bary_to_center_offsets epochs
        (Horizons N vs N99; the J2015 one is 0.25 d off its snapshot, so it
        also checks the moon fits between their samples).  0 for the bodies
        de440s gives directly (Sun, Mercury, Venus, Earth, Moon).  See
        sg_position_with_uncertainty for the measured values.
        """
        if planet not in _KERNEL_BARYCENTRE_PLANETS:
            return 0.0
        if planet in _KERNEL_BOUND_CACHE:
            return _KERNEL_BOUND_CACHE[planet]
        from sigma_ground.field.interface.adapters._jplephem_bridge import (
            query_jplephem,
        )
        data = self._load_de440()
        worst = 0.0
        for snap in data["snapshots"].values():
            truth = self._snap_pos(snap, planet)
            pos = self._jplephem_position(planet, snap["epoch"]["jd_tdb"])
            if truth is None or pos is None:
                continue
            worst = max(worst, float(np.linalg.norm(pos - truth)))
        offsets = data.get("bary_to_center_offsets", {})
        for key, rec in offsets.get("bodies", {}).get(planet, {}).items():
            jd = offsets["epochs"][key]["jd_tdb"]
            pos = self._jplephem_position(planet, jd)
            bary = query_jplephem(planet, jd)
            if pos is None or bary is None:
                continue
            truth = bary + np.array([rec["dx_km"], rec["dy_km"], rec["dz_km"]])
            worst = max(worst, float(np.linalg.norm(pos - truth)))
        _KERNEL_BOUND_CACHE[planet] = worst
        return worst

    def _serve(self, body_name: str, jd: float):
        """(tier, SSB position, heliocentric position) from the best tier."""
        body_name = self._canonical(body_name)
        jd = float(jd)

        # Tier 1: DE440 exact snapshot
        snap = self._snapshot_at(jd)
        if snap is not None:
            pos = self._snap_pos(snap, body_name)
            sun = self._snap_pos(snap, "Sun")
            if pos is not None and sun is not None:
                return "de440", pos, pos - sun

        # Tier 2: SSBM forecast at a stored epoch
        hel = self._forecast_heliocentric(body_name, jd)
        if hel is not None:
            sun = self._forecast_sun_ssb(jd)
            if sun is not None:
                return "forecast", hel + sun, hel

        # Tier 3: jplephem kernel (exact DE440)
        pos = self._jplephem_position(body_name, jd)
        if pos is not None:
            sun = pos if body_name == "Sun" else self._jplephem_position("Sun", jd)
            if sun is not None:
                return "jplephem", pos, pos - sun

        # Tier 4: Keplerian orbit fit.  A satellite's fit is relative to its
        # planet, so it is added to the planet as served at jd (any tier),
        # with that tier's Sun: moon - planet is then the fitted relative
        # orbit whichever tier serves the planet.  FIXED 2026-10-10: with
        # the kernel on, the planet came from the kernel and the moon from a
        # Kepler planet (Charon-Pluto 1.78e6 km on 2012-07-02, true 19 596).
        if body_name == "Sun":
            sun = self._kepler_sun_ssb(jd)
            if sun is not None:
                return "kepler", sun, np.zeros(3)
        else:
            orbit = self._kepler(body_name)
            if orbit is not None:
                rel = np.array(_ecliptic_to_equatorial(
                    *orbit.predict_relative_ecliptic(jd)))
                if orbit.anchor == "Sun":
                    sun = self._kepler_sun_ssb(jd)
                    if sun is not None:
                        return "kepler", rel + sun, rel
                else:
                    try:
                        _, a_pos, a_hel = self._serve(orbit.anchor, jd)
                    except ValueError:
                        pass
                    else:
                        return "kepler", a_pos + rel, a_hel + rel

        kernel = ("disabled" if not self._use_kernel else
                  "unavailable, outside its span, or without this body")
        raise ValueError(
            f"Cannot determine position for {body_name!r} at JD={jd:.6f}: "
            f"not a DE440 snapshot epoch with this body, not a stored "
            f"forecast epoch with this body, jplephem kernel {kernel}, and "
            f"no Kepler fit for this body (sg_available_bodies() lists the "
            f"bodies served at every epoch)."
        )

    # ── public sg_ API ────────────────────────────────────────────────────

    def sg_position(self, body_name: str, jd: float) -> np.ndarray:
        """SSB-equatorial barycentric position (km) using best available source.

        Priority: DE440 snapshot > SSBM forecast > jplephem kernel > Keplerian.

        Parameters
        ----------
        body_name : body name matching the DE440/forecast fixture (e.g. 'Mars';
                    case-insensitive)
        jd        : Julian Date (TDB)

        Returns
        -------
        np.ndarray shape (3,) in km, ICRF equatorial
        """
        return self._serve(body_name, jd)[1]

    def sg_heliocentric(self, body_name: str, jd: float) -> np.ndarray:
        """Heliocentric equatorial position (km).

        Body and Sun come from the same tier, so a forecast or Kepler answer
        is that tier's own heliocentric vector.
        """
        return self._serve(body_name, jd)[2]

    def sg_distance_au(self, body_name: str, jd: float) -> float:
        """Heliocentric distance in AU."""
        hel = self.sg_heliocentric(body_name, jd)
        return float(np.linalg.norm(hel)) / AU_KM

    def sg_position_with_uncertainty(
        self, body_name: str, jd: float,
    ) -> tuple[np.ndarray, float]:
        """Position (km) with the served tier's error estimate in AU.

        - 'de440' tier, and 'jplephem' for the Sun, Mercury, Venus, Earth
          and Moon: 0.0 -- the values are DE440 itself (DE440's own error,
          Park et al. 2021, is not included).
        - 'jplephem' for Mars..Pluto: the measured error of the conversion
          from system barycentre to planet centre (_kernel_centre_bound_km),
          i.e. the largest disagreement with Horizons' planet centres.
          Measured 2026-10-10: Mars < 1 m, Jupiter 0.4 km, Saturn 15 km
          (moons without a Kepler fit, mainly Iapetus), Neptune 3 km
          (Triton's fit), Pluto 0.4 km; Uranus 1.1e3 km: Horizons' Uranus
          centre (799) sits 352-1163 km from DE440's Uranus barycentre over
          2000-2026, while the five major moons (605 km^3/s^2) put the
          centre within 35 km of it.  That disagreement is not resolved here.
        - 'forecast' tier with the kernel available: the measured error
          |forecast - DE440 kernel| at jd.
        - 'forecast' tier without the kernel: |forecast - Kepler fit|, a
          cross-check between independent predictors (the Kepler fit is
          generally the worse of the two).
        - 'kepler' tier: nan.  It serves only where no exact value exists,
          so there is no independent reference and no honest number (its
          in-sample fit residual says nothing about errors between the
          yearly fit epochs, e.g. for the Moon).  A satellite served by this
          tier sits on its planet as served by the planet's own tier.

        Returns
        -------
        pos_km  : np.ndarray (3,) -- best position estimate
        err_au  : float           -- error estimate in AU (nan if unknown)
        """
        tier, pos, _ = self._serve(body_name, jd)
        if tier == "de440":
            return pos, 0.0
        if tier == "jplephem":
            bound = self._kernel_centre_bound_km(self._canonical(body_name))
            return pos, bound / AU_KM
        if tier == "forecast":
            name = self._canonical(body_name)
            ref = self._jplephem_position(name, jd)
            if ref is None:
                ref = self._kepler_position(name, jd)
            if ref is not None:
                return pos, float(np.linalg.norm(pos - ref)) / AU_KM
        return pos, float("nan")

    def sg_available_bodies(self) -> list[str]:
        """Bodies whose position queries succeed at every epoch.

        These are the bodies with a Kepler fit (the any-epoch tier) plus the
        Sun.  Fixture bodies without one (Mimas, Tethys, Dione and Rhea,
        which are not in orbital.ANCHOR_MAP) are served only at the DE440
        snapshot epochs.
        """
        bodies: set[str] = {"Sun"}
        data = self._load_de440()
        names: set[str] = set()
        for snap in data["snapshots"].values():
            for b in snap["bodies"]:
                names.add(b["name"])
        for name in names:
            if name != "Sun" and self._kepler_heliocentric_ok(name):
                bodies.add(name)
        return sorted(bodies)

    def _kepler_heliocentric_ok(self, body_name: str) -> bool:
        from sigma_ground.field.interface.orbital import ANCHOR_MAP
        chain = body_name
        while chain != "Sun":
            if chain not in ANCHOR_MAP:
                return False
            chain = ANCHOR_MAP[chain]
        return True

    def sg_tier(self, body_name: str, jd: float) -> str:
        """Which data tier would serve this query: 'de440', 'forecast', 'jplephem', 'kepler', 'none'."""
        try:
            return self._serve(body_name, jd)[0]
        except ValueError:
            return "none"
