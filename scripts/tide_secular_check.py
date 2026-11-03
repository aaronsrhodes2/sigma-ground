"""The falsifier for lever L3 (C26): does DE430's time-delayed Earth tide, with the DE440 header's constants, give the Moon the tidal acceleration DE430 reports,
-25.82 +- 0.03 arcsec/century^2 in mean longitude (Folkner et al. 2014, section III.C)?

Sun, Earth and Moon only, started from DE440 at 2010-01-01, Forest-Ruth at the given step, two runs: the static Love-number tide (every stored predictor) and the DE tide. The
geocentric along-track difference of the Moon between the two runs is fitted with a + b t + c t^2; the secular acceleration of the mean longitude is 2c / a_moon, printed in
arcsec/century^2. The static tide has no secular term, so the difference IS the DE tide's secular effect. Independent of the shoot-out; costs a few minutes.

usage: python scripts/tide_secular_check.py [--years 8] [--dt-days 0.025]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=float, default=8.0)
    ap.add_argument("--dt-days", type=float, default=0.025)
    ap.add_argument("--out", type=Path, default=_ROOT / "sigma_ground/field/interface/fixtures/v2physics_restated/TIDE_SECULAR_CHECK.json")
    a = ap.parse_args()
    import dataclasses
    from jplephem.spk import SPK
    from sigma_ground.field import datasets
    from sigma_ground.field.constants import G
    from sigma_ground.field.interface import de440_constants, reduced_model as rm, rolling_shootout as rs
    from sigma_ground.field.interface.nbody import CelestialBody, NBodySystem
    kernel = SPK.open(str(datasets.de440_kernel()))
    jd0 = rm.jd_of(2010, 1, 1) if hasattr(rm, "jd_of") else 2455197.5
    names = ["Sun", "Earth", "Moon"]
    de = json.loads(rm._FIXTURE.read_text())
    gm = rm.reduced_masses(de["snapshots"]["j2010"])
    full, fnames, _, _ = rs._build_bodies_at_snapshot("j2010", de, next(p for p in rs.PREDICTORS if p.name == "restated_zon"))
    by = dict(zip(fnames, full))

    def bodies():
        out = []
        for n in names:
            p, v = rm.kernel_state(kernel, n, jd0)
            b = by[n]
            out.append(CelestialBody(mass_kg=gm[n] / G, position_m=p * rm.KM, velocity_m_s=v * rm.KM, radius_m=b.radius_m, love_number_k2=b.love_number_k2, area_m2=b.area_m2,
                                     reflectivity=b.reflectivity, j2=b.j2, j3=b.j3, j4=b.j4, pole_axis_unit=b.pole_axis_unit))
        return out
    zon = next(p for p in rs.PREDICTORS if p.name == "restated_zon")
    tog = dataclasses.replace(zon.toggles, prescribed_sources=False, sectoral_c22=False)
    static = NBodySystem(bodies(), toggles=tog, solar_luminosity_W=rs._L_SUN_W)
    bs = bodies()
    bs[1] = bs[1].replace(love_number_k2=0.0)
    c = de440_constants.load()["raw"]
    de_tide = {"earth": 1, "moon": 2, "sun": 0, "k2": (c["K2E0"], c["K2E1"], c["K2E2"]), "tau_orbit_s": tuple(86400.0 * c[k] for k in ("TAUE0", "TAUE1", "TAUE2")),
               "tau_rot_s": (0.0, 86400.0 * c["TAUER1"], 86400.0 * c["TAUER2"]), "radius_m": 1000.0 * c["RE"], "theta_dot": rs.EARTH_ROTATION_RATE_RAD_S}
    tide = NBodySystem(bs, toggles=dataclasses.replace(tog, earth_tide_de=True), solar_luminosity_W=rs._L_SUN_W, de_tide=de_tide)
    n_steps = int(round(a.years * 365.25 / a.dt_days))
    dts = a.dt_days * 86400.0
    every = max(1, int(round(1.0 / a.dt_days)))
    t_yr, along_m, radial_m = [], [], []
    for s in range(1, n_steps + 1):
        static.forest_ruth_step(dts)
        tide.forest_ruth_step(dts)
        if s % every == 0:
            r0 = static.bodies[2].position_m - static.bodies[1].position_m
            v0 = static.bodies[2].velocity_m_s - static.bodies[1].velocity_m_s
            r1 = tide.bodies[2].position_m - tide.bodies[1].position_m
            rh = r0 / np.linalg.norm(r0)
            nh = np.cross(r0, v0)
            nh /= np.linalg.norm(nh)
            th = np.cross(nh, rh)
            d = r1 - r0
            t_yr.append(s * a.dt_days / 365.25)
            along_m.append(float(d @ th))
            radial_m.append(float(d @ rh))
    t = np.array(t_yr)
    coef = np.polyfit(t, np.array(along_m), 2)                       # along = c2 t^2 + c1 t + c0 (m, yr)
    a_moon = 384400e3
    sec_rad_per_yr2 = 2.0 * coef[0] / a_moon                          # d^2(lambda)/dt^2 in rad/yr^2 (along-track = a * d lambda)
    sec_arcsec_per_cy2 = sec_rad_per_yr2 * (180.0 / math.pi) * 3600.0 * 100.0 ** 2
    radial_rate_mm_yr = np.polyfit(t, np.array(radial_m), 1)[0] * 1000.0
    doc = {"what": "secular effect of the DE tide against the static tide, Sun-Earth-Moon from DE440 at 2010-01-01", "years": a.years, "dt_days": a.dt_days,
           "mean_longitude_acceleration_arcsec_per_cy2": sec_arcsec_per_cy2, "de430_reference_arcsec_per_cy2": -25.82, "radial_drift_mm_per_yr": radial_rate_mm_yr,
           "de430_reference_recession_mm_per_yr": 38.08, "fit_along_track_m_coefficients": coef.tolist(), "samples": len(t)}
    a.out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
