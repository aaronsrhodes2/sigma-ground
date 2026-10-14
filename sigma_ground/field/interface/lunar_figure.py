"""The Moon's triaxial figure: its C22 coefficient and rotation model, for the engine's sectoral-figure layer (PhysicsToggles.sectoral_c22).

The force law lives in figure_forces.py (shared with the engine); this module holds what is specific to the Moon: the coefficient, the IAU rotation model that orients the long axis,
and a helper that builds the Moon body fields. Use: body = CelestialBody(..., **moon_figure_fields()); NBodySystem(bodies, toggles=PhysicsToggles(sectoral_c22=True),
epoch_jd_tdb=jd0, rotation_models={moon_index: moon_axes}).

Background (unchanged): the C22 sectoral term of its gravity field, for a body that always points its long axis at the Earth.

The N-body engine carries the Moon's axisymmetric J2 about its pole. The real Moon is triaxial (its long axis, the smallest moment A, points at the Earth; B > A across it; C about the spin axis),
and because it is tidally locked that bulge never averages away: it adds an orbit-averaged quadrupole that changes the Moon's mean motion and the rate at which its perigee advances.

Derivation chain (Golden Rule 1):
  FIRST_PRINCIPLES: MacCullagh's formula for the potential energy of a point mass m at r from a body with inertia tensor I:  U = -G m [ M/r + (tr I - 3 r^T I r / r^2) / (2 r^3) ].
      Writing I = (tr I / 3) 1 + T (T traceless) the quadrupole part is U = +3 G m (r^T T r) / (2 r^5), and the force on m is  F = 3 G m [ (5 q / (2 r^7)) r - T r / r^5 ],  q = r^T T r.
      The reaction on the body is -F (Newton III). For the axisymmetric T = (2/3) J2 M R^2 diag(-1/2, -1/2, 1) this is the engine's own J2 acceleration (tested).
      With J2 = [C - (A+B)/2] / (M R^2) and C22 = (B - A) / (4 M R^2), the C22 part alone is  T = 2 C22 M R^2 (y y^T - x x^T)  in the body frame (x toward the Earth, y = z cross x).
  MEASURED (LIT): C22 of the Moon, 2.2382e-5 for the reference radius 1738 km (GRAIL, Lemoine et al. 2013; JPL DE430/DE440 lunar models carry the same field). GM of the Moon from the DE440 fixture.
Golden Rule 2: ``regime`` of the model is SAFE for a test mass farther than ~3 lunar radii from the Moon (a multipole expansion needs r > R); the engine's softening is not applied here.
PHYSICS_GAP: the long axis is taken as the instantaneous direction to the Earth (physical librations, up to ~8 deg, are ignored: they change the orbit-averaged effect at the few-percent level);
PHYSICS_GAP: the torque of the Earth on the Moon's figure is not integrated (the rotation is prescribed); degree 3 and higher field terms, and the tesseral C21/S21, are not included.
"""
from __future__ import annotations

import numpy as np

from sigma_ground.field.interface.figure_forces import axisymmetric_T, c22_T, quadrupole_acceleration, sectoral_extra_accelerations  # noqa: F401  (re-exported)

C22_MOON = 2.2382e-5            # [LIT] unnormalised, reference radius 1738 km (GRAIL GRGM660)
R_REF_MOON_M = 1.738e6          # m, the reference radius the coefficient belongs to


# ---- the Moon's orientation: IAU WGCCRE rotation model (Archinal et al. 2011, Table 2; the 2015 report keeps it) --------------------------------------------------
# W = 38.3213 + 13.17635815 d - 1.4e-12 d^2 + sum(a_k sin E_k), d = days from J2000.0 TDB. The pole (alpha0, delta0) is the repo's own table (rolling_shootout._IAU_POLE_MODELS,
# checked there against NAIF pck00011); the E_k below are the same arguments as in that table (their rates per century match it: E1 -0.0529921 deg/d = -1935.5364525 deg/cy, E13 12.9590088 deg/d = 473327.7964 deg/cy).
# The prime-meridian constants W0, the rate and the amplitudes are from the same report (LIT, entered from the report, validated by the test against the DE440 Moon-Earth direction:
# the model's long axis must agree with the true direction to the Earth on average to < 0.5 deg and oscillate about it by the optical libration).
_E_ARGS = [(125.045, -0.0529921), (250.089, -0.1059842), (260.008, 13.0120009), (176.625, 13.3407154), (357.529, 0.9856003), (311.589, 26.4057084), (134.963, 13.0649930),
           (276.617, 0.3287146), (34.226, 1.7484877), (15.134, -0.1589763), (119.743, 0.0036096), (239.961, 0.1643573), (25.053, 12.9590088)]
_W_TERMS = [3.5610, 0.1208, -0.0642, 0.0158, 0.0252, -0.0066, -0.0047, -0.0046, 0.0028, 0.0052, 0.0040, 0.0019, -0.0044]
_JD_J2000 = 2451545.0


def moon_prime_meridian_deg(jd_tdb: float) -> float:
    d = jd_tdb - _JD_J2000
    w = 38.3213 + 13.17635815 * d - 1.4e-12 * d * d
    for amp, (e0, e1) in zip(_W_TERMS, _E_ARGS):
        w += amp * np.sin(np.radians(e0 + e1 * d))
    return w


def moon_axes(jd_tdb: float) -> tuple:
    """(x_hat, y_hat, z_hat) of the Moon's body frame in the ICRF at TDB Julian date jd_tdb: z the IAU pole, x toward the prime meridian (the mean direction to the Earth)."""
    from sigma_ground.field.interface.rolling_shootout import _iau_pole_radec
    ra, dec = (np.radians(v) for v in _iau_pole_radec("Moon", jd_tdb))
    w = np.radians(moon_prime_meridian_deg(jd_tdb))
    z = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    node = np.array([-np.sin(ra), np.cos(ra), 0.0])                     # the ascending node of the Moon's equator on the ICRF equator
    x = np.cos(w) * node + np.sin(w) * np.cross(z, node)
    y = np.cross(z, x)
    return x, y, z


def c22_extra_accelerations(bodies, moon_index: int, earth_index: int, c22: float = C22_MOON, ref_radius_m: float = R_REF_MOON_M, jd_tdb: float | None = None) -> np.ndarray:
    """The accelerations (m/s^2, shape (n, 3)) that the Moon's C22 figure adds to every body, with the reaction on the Moon (sum m_i a_i = 0 to round-off).

    With ``jd_tdb`` the long axis comes from the IAU rotation model; without it the long axis is locked to the instantaneous direction to the Earth (a simplification kept for tests).
    """
    n = len(bodies)
    if c22 == 0.0:
        return np.zeros((n, 3))
    moon, earth = bodies[moon_index], bodies[earth_index]
    if jd_tdb is not None:
        x_hat, y_hat, _ = moon_axes(jd_tdb)
    else:
        z = moon.pole_axis_unit / np.linalg.norm(moon.pole_axis_unit)
        d = earth.position_m - moon.position_m
        x_hat = d - (d @ z) * z
        x_hat /= np.linalg.norm(x_hat)
        y_hat = np.cross(z, x_hat)
    T = c22_T(moon.mass_kg, ref_radius_m, c22, x_hat, y_hat)
    return sectoral_extra_accelerations(np.array([b.position_m for b in bodies]), np.array([b.mass_kg for b in bodies]), moon_index, T)


def moon_figure_fields() -> dict:
    """Keyword arguments that give a CelestialBody the Moon's C22 figure (use with PhysicsToggles(sectoral_c22=True) and rotation_models={moon_index: moon_axes})."""
    return {"c22": C22_MOON, "figure_ref_radius_m": R_REF_MOON_M}
