"""Gravitational quadrupole forces of an extended body, independent of the N-body engine (so both nbody.py and lunar_figure.py can use them).

Derivation chain (Golden Rule 1):
  FIRST_PRINCIPLES: MacCullagh's formula for the potential energy of a point mass m at r from a body of mass M with inertia tensor I:
      U = -G m [ M/r + (tr I - 3 r^T I r / r^2) / (2 r^3) ].
      With I = (tr I / 3) 1 + T (T traceless) the quadrupole part is U = +3 G m (r^T T r) / (2 r^5) and the force on m is
          F = 3 G m [ (5 q / (2 r^7)) r - T r / r^5 ],   q = r^T T r.
      The reaction on the body is -F (Newton III).
      With J2 = [C - (A+B)/2] / (M R^2) and C22 = (B - A) / (4 M R^2) (A < B < C the principal moments about the body axes x, y, z):
          axisymmetric part  T_J2  = (J2 M R^2 / 3)(3 n n^T - 1),
          sectoral C22 part  T_C22 = 2 C22 M R^2 (y y^T - x x^T),   x the long axis (smallest moment).
  Tested against: the engine's own J2 acceleration, the spherical-harmonic derivatives of the C22 potential on both principal axes and at 10/20/45 deg, momentum conservation.
Golden Rule 2: valid for a test mass outside the body (r > R); the multipole expansion is not softened here.
PHYSICS_GAP: degree-2 only (degree 3 and higher, and the tesseral C21/S21, are not included); the body's rotation is prescribed by the caller, not integrated.
"""
from __future__ import annotations

import numpy as np

from sigma_ground.field.constants import G


def quadrupole_acceleration(r_vec: np.ndarray, T_inertial: np.ndarray) -> np.ndarray:
    """Acceleration (m/s^2) of a test mass at r_vec (m, from the body) due to the traceless quadrupole T (kg m^2, inertial frame) of the body: 3 G [ (5 q / 2 r^7) r - T r / r^5 ]."""
    r2 = float(r_vec @ r_vec)
    r = np.sqrt(r2)
    Tr = T_inertial @ r_vec
    q = float(r_vec @ Tr)
    return 3.0 * G * ((2.5 * q / (r2 * r2 * r2 * r)) * r_vec - Tr / (r2 * r2 * r))


def axisymmetric_T(mass_kg: float, radius_m: float, j2: float, pole: np.ndarray) -> np.ndarray:
    """Traceless inertia tensor of an axisymmetric body: (J2 M R^2 / 3)(3 n n^T - 1)."""
    n = pole / np.linalg.norm(pole)
    return (j2 * mass_kg * radius_m ** 2 / 3.0) * (3.0 * np.outer(n, n) - np.eye(3))


def c22_T(mass_kg: float, radius_m: float, c22: float, x_hat: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """The C22 part of the traceless inertia tensor: 2 C22 M R^2 (y y^T - x x^T), x the long axis."""
    return 2.0 * c22 * mass_kg * radius_m ** 2 * (np.outer(y_hat, y_hat) - np.outer(x_hat, x_hat))


def sectoral_extra_accelerations(positions_m: np.ndarray, masses_kg: np.ndarray, body_index: int, T_inertial: np.ndarray) -> np.ndarray:
    """Accelerations (n, 3) that the quadrupole T of body ``body_index`` adds to every body, including the reaction on that body (sum m_i a_i = 0 to round-off)."""
    n = len(masses_kg)
    extra = np.zeros((n, 3))
    reaction = np.zeros(3)
    centre = positions_m[body_index]
    for i in range(n):
        if i == body_index:
            continue
        a_i = quadrupole_acceleration(positions_m[i] - centre, T_inertial)
        extra[i] = a_i
        reaction -= masses_kg[i] * a_i
    extra[body_index] = reaction / masses_kg[body_index]
    return extra
