"""
General Relativity Basics — Schwarzschild geometry.

Schwarzschild metric quantities, horizons, and Hawking radiation, with every
constant derived from sigma_ground.field.constants.

References:
  Misner, Thorne & Wheeler "Gravitation" (Princeton, 1973)
  Wald "General Relativity" (Chicago, 1984)
  Hawking 1974, Comm. Math. Phys. 43, 199
  Bardeen, Press & Teukolsky 1972, ApJ 178, 347
  Page 1976, Phys. Rev. D 13, 198; Page 1993, PRL 71, 3743

Dependency: G, C, HBAR, K_B from constants; schwarzschild_radius from scale.
"""

import math

from .constants import G, C, HBAR, K_B
from .scale import schwarzschild_radius        # r_s = 2GM/c^2

__all__ = [
    'schwarzschild_radius',
    'gravitational_redshift',
    'time_dilation_gr',
    'escape_velocity',
    'isco_radius',
    'photon_sphere_radius',
    'hawking_temperature',
    'hawking_luminosity',
    'hawking_evaporation_time',
    'page_time_sm',
    'bekenstein_hawking_entropy',
    'tidal_force',
]

# Page 1993 (PRL 71, 3743): for a Schwarzschild BH evaporating into a thermal
# environment, the radiation's entanglement entropy peaks at ~half the BH's
# initial entropy at t ~ 0.5384 t_evap (single massless-field reference).
_PAGE_FRACTION_SM = 0.5384


# ── Schwarzschild geometry ──────────────────────────────────────────────

def gravitational_redshift(M, r):
    """Gravitational redshift z = 1/sqrt(1 - r_s/r) - 1.

    A photon climbing out of a gravitational well loses energy. At r = r_s,
    z -> infinity (event horizon).

    Args:
        M: mass (kg)
        r: coordinate radius (m), must be > r_s

    Raises:
        ValueError: if r <= r_s.
    """
    rs = schwarzschild_radius(M)
    if r <= rs:
        raise ValueError(f"r={r:.4e} m <= r_s={rs:.4e} m: inside/at the horizon")
    return 1.0 / math.sqrt(1.0 - rs / r) - 1.0


def time_dilation_gr(M, r):
    """Gravitational time dilation factor tau/t = sqrt(1 - r_s/r).

    Ratio of proper time at radius r to coordinate time at infinity. At
    r = r_s the factor -> 0 (time stops at the horizon).

    Args:
        M: mass (kg)
        r: coordinate radius (m), must be > r_s

    Raises:
        ValueError: if r <= r_s.
    """
    rs = schwarzschild_radius(M)
    if r <= rs:
        raise ValueError(f"r={r:.4e} m <= r_s={rs:.4e} m: inside/at the horizon")
    return math.sqrt(1.0 - rs / r)


def escape_velocity(M, r):
    """Newtonian escape velocity v_esc = sqrt(2GM/r).

    At r = r_s, v_esc = c (which is why light cannot escape).

    Args:
        M: mass (kg)
        r: radius (m), must be > 0
    """
    if r <= 0:
        raise ValueError(f"r={r} <= 0: radius must be positive")
    return math.sqrt(2.0 * G * M / r)


# ── Special radii ───────────────────────────────────────────────────────

def isco_radius(M):
    """Innermost Stable Circular Orbit radius for a Schwarzschild BH.

    r_ISCO = 6 GM/c^2 = 3 r_s. Inside it, no stable circular orbits exist.
    Reference: Bardeen, Press & Teukolsky 1972, ApJ 178, 347.
    """
    return 6.0 * G * M / C**2


def photon_sphere_radius(M):
    """Photon sphere radius r_ph = 3 GM/c^2 = 1.5 r_s.

    Photons travel in unstable circular orbits at this radius.
    """
    return 3.0 * G * M / C**2


# ── Hawking radiation ───────────────────────────────────────────────────

def hawking_temperature(M):
    """Hawking temperature T_H = hbar c^3 / (8 pi G M k_B).

    Smaller black holes are hotter. Solar-mass BH ~ 6.17e-8 K.
    Reference: Hawking 1974, Comm. Math. Phys. 43, 199.

    Args:
        M: black hole mass (kg), must be > 0.
    """
    if M <= 0:
        raise ValueError(f"M={M} <= 0: mass must be positive")
    return HBAR * C**3 / (8.0 * math.pi * G * M * K_B)


def hawking_luminosity(M):
    """Hawking luminosity L_H = hbar c^6 / (15360 pi G^2 M^2) (W).

    Reference: Page 1976, Phys. Rev. D 13, 198.

    Args:
        M: black hole mass (kg), must be > 0.
    """
    if M <= 0:
        raise ValueError(f"M={M} <= 0: mass must be positive")
    return HBAR * C**6 / (15360.0 * math.pi * G**2 * M**2)


def hawking_evaporation_time(M):
    """Black-hole evaporation time t_evap = 5120 pi G^2 M^3 / (hbar c^4) (s).

    Reference: Hawking 1974; Page 1976.

    Args:
        M: initial black hole mass (kg), must be > 0.
    """
    if M <= 0:
        raise ValueError(f"M={M} <= 0: mass must be positive")
    return 5120.0 * math.pi * G**2 * M**3 / (HBAR * C**4)


def page_time_sm(M):
    """Page time t_Page ~ 0.5384 t_evap (s).

    When the Hawking radiation's entanglement entropy has built up to half
    the BH's initial Bekenstein-Hawking entropy. Reference: Page 1993
    (PRL 71, 3743).

    Args:
        M: initial black hole mass (kg), must be > 0.
    """
    if M <= 0:
        raise ValueError(f"M={M} <= 0: mass must be positive")
    return _PAGE_FRACTION_SM * hawking_evaporation_time(M)


def bekenstein_hawking_entropy(M):
    """Bekenstein-Hawking entropy S_BH = k_B * 4 pi G M^2 / (hbar c) (J/K).

    Reference: Bekenstein 1973 (PRD 7, 2333); Hawking 1975.

    Args:
        M: black hole mass (kg), must be > 0.
    """
    if M <= 0:
        raise ValueError(f"M={M} <= 0: mass must be positive")
    return K_B * 4.0 * math.pi * G * M * M / (HBAR * C)


# ── Tidal forces ────────────────────────────────────────────────────────

def tidal_force(M, r, dr, m=1.0):
    """Tidal (differential gravitational) force dF = 2 G M m dr / r^3.

    Args:
        M: gravitating mass (kg)
        r: distance to center of mass (m), must be > 0
        dr: radial separation of the two test points (m)
        m: test mass (kg); default 1.0 returns acceleration difference (m/s^2)
    """
    if r <= 0:
        raise ValueError(f"r={r} <= 0: radius must be positive")
    return 2.0 * G * M * m * dr / r**3
