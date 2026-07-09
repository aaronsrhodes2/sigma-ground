"""
Scale factor (standard, inert).

scale_ratio(sigma) = e^sigma is a dimensionless scale factor. At sigma = 0
(our spacetime; SIGMA_HERE ~ 1e-61) it is exactly 1.0, so all physics is
standard. The field-interface modules are parameterized on this hook; it is
retained as an inert identity so those signatures keep working.

schwarzschild_radius(M) = 2GM/c^2 is the standard GR horizon radius.
"""

import math

from .constants import G, C


def scale_ratio(sigma):
    """e^sigma — dimensionless scale factor. Exactly 1.0 at sigma = 0 (standard).

    Guard: exp overflows float64 near sigma ~ 709.8, so clamp rather than crash.
    """
    if sigma > 709.0:
        return float('inf')
    if sigma < -709.0:
        return 0.0
    return math.exp(sigma)


def schwarzschild_radius(M_kg):
    """Schwarzschild radius r_s = 2GM/c^2 (meters)."""
    return 2 * G * M_kg / C**2
