"""The Moon's triaxial figure: the C22 sectoral term of its gravity field, for a body that always points its long axis at the Earth.

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

from sigma_ground.field.constants import G
from sigma_ground.field.interface.nbody import NBodySystem

C22_MOON = 2.2382e-5            # [LIT] unnormalised, reference radius 1738 km (GRAIL GRGM660)
R_REF_MOON_M = 1.738e6          # m, the reference radius the coefficient belongs to


def quadrupole_acceleration(r_vec: np.ndarray, T_inertial: np.ndarray) -> np.ndarray:
    """Acceleration (m/s^2) of a test mass at r_vec (m, from the body) due to the traceless quadrupole T (kg m^2, inertial frame) of the body: 3 G [ (5 q / 2 r^7) r - T r / r^5 ]."""
    r2 = float(r_vec @ r_vec)
    r = np.sqrt(r2)
    Tr = T_inertial @ r_vec
    q = float(r_vec @ Tr)
    return 3.0 * G * ((2.5 * q / (r2 * r2 * r2 * r)) * r_vec - Tr / (r2 * r2 * r))


def axisymmetric_T(mass_kg: float, radius_m: float, j2: float, pole: np.ndarray) -> np.ndarray:
    """Traceless inertia tensor of an axisymmetric body: (2/3) J2 M R^2 (3 n n^T - 1)/2... written as (J2 M R^2 / 3)(3 n n^T - 1)."""
    n = pole / np.linalg.norm(pole)
    return (j2 * mass_kg * radius_m ** 2 / 3.0) * (3.0 * np.outer(n, n) - np.eye(3))


def c22_T(mass_kg: float, radius_m: float, c22: float, x_hat: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """The C22 part of the traceless inertia tensor: 2 C22 M R^2 (y y^T - x x^T), x the long axis (toward the Earth)."""
    return 2.0 * c22 * mass_kg * radius_m ** 2 * (np.outer(y_hat, y_hat) - np.outer(x_hat, x_hat))


def c22_extra_accelerations(bodies, moon_index: int, earth_index: int, c22: float = C22_MOON, ref_radius_m: float = R_REF_MOON_M) -> np.ndarray:
    """The accelerations (m/s^2, shape (n, 3)) that the Moon's C22 figure adds to every body, long axis locked toward the Earth, spin axis the Moon body's ``pole_axis_unit``.

    Includes the reaction on the Moon, so sum(m_i a_i) = 0 to round-off (Newton III).
    """
    n = len(bodies)
    extra = np.zeros((n, 3))
    if c22 == 0.0:
        return extra
    moon, earth = bodies[moon_index], bodies[earth_index]
    z = moon.pole_axis_unit / np.linalg.norm(moon.pole_axis_unit)
    d = earth.position_m - moon.position_m
    x_hat = d - (d @ z) * z
    x_hat /= np.linalg.norm(x_hat)
    y_hat = np.cross(z, x_hat)
    T = c22_T(moon.mass_kg, ref_radius_m, c22, x_hat, y_hat)
    reaction = np.zeros(3)
    for i, b in enumerate(bodies):
        if i == moon_index:
            continue
        a_i = quadrupole_acceleration(b.position_m - moon.position_m, T)
        extra[i] = a_i
        reaction -= b.mass_kg * a_i
    extra[moon_index] = reaction / moon.mass_kg
    return extra


class LockedMoonSystem(NBodySystem):
    """NBodySystem plus the Moon's C22 figure (see ``c22_extra_accelerations``)."""

    def __init__(self, bodies, *args, moon_index: int, earth_index: int, c22: float = C22_MOON, ref_radius_m: float = R_REF_MOON_M, **kw):
        super().__init__(bodies, *args, **kw)
        self.moon_index = moon_index
        self.earth_index = earth_index
        self.c22 = c22
        self.ref_radius_m = ref_radius_m

    def compute_accelerations(self):
        acc = super().compute_accelerations()
        if self.c22 == 0.0:
            return acc
        return acc + c22_extra_accelerations(self.bodies, self.moon_index, self.earth_index, self.c22, self.ref_radius_m)
