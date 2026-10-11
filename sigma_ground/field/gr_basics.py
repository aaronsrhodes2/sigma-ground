"""
General Relativity Basics — Schwarzschild geometry.

Schwarzschild metric quantities, horizons, and Hawking radiation, with every
constant derived from sigma_ground.field.constants.

References:
  Misner, Thorne & Wheeler "Gravitation" (Princeton, 1973)
  Wald "General Relativity" (Chicago, 1984)
  Hawking 1974, Comm. Math. Phys. 43, 199
  Bardeen, Press & Teukolsky 1972, ApJ 178, 347
  Page 1976, Phys. Rev. D 13, 198 (greybody emission rates; for comparison)
  Page 2013, JCAP 09, 028, arXiv:1301.4995 (Page time fraction)

Dependency: G, C, HBAR, K_B from constants; schwarzschild_radius from scale.
"""

import math

from .constants import G, C, HBAR, K_B
from .scale import schwarzschild_radius as _scale_schwarzschild_radius

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

# Page 2013 (JCAP 09, 028; arXiv:1301.4995, eq. 9): for a large nonrotating
# hole emitting photons and gravitons, the radiation's coarse-grained entropy
# equals the remaining Bekenstein-Hawking entropy (about 59.75 % of the
# initial) at t* = eps t_decay, eps = 1 - (beta/(beta+1))^(3/2) = 0.53810.
# FIXED 2026-10-10: was 0.5384, cited to Page 1993 (whose abstract has no
# time fraction) and described as "half the initial entropy".
_PAGE_FRACTION_SM = 0.53810


# ── Input contract (2026-10-10 audit) ───────────────────────────────────
# Every function here needs M > 0 and finite: before, M <= 0 gave negative
# radii, a blueshift for "redshift", or a bare "math domain error", while
# the Hawking functions already raised. NaN fails the check.

def _check_mass(M):
    if not (M > 0.0 and M != math.inf):
        raise ValueError(f"M={M!r}: mass must be positive and finite")


# ── Schwarzschild geometry ──────────────────────────────────────────────

def schwarzschild_radius(M):
    """Schwarzschild radius r_s = 2GM/c^2 (m), from field.scale.

    Raises:
        ValueError: if M is not positive and finite.
    """
    _check_mass(M)
    return _scale_schwarzschild_radius(M)


def gravitational_redshift(M, r):
    """Gravitational redshift z = 1/sqrt(1 - r_s/r) - 1.

    A photon climbing out of a gravitational well loses energy. At r = r_s,
    z -> infinity (event horizon).

    Evaluated as z = expm1(-1/2 log1p(-r_s/r)), the same quantity without
    the cancellation of "1/sqrt(1 - x) - 1": in a weak field (x = r_s/r =
    1.4e-9 at Earth's surface) the old form lost ~8 digits of z, and the
    redshift difference across 1 m (1.09e-16, Pound-Rebka scale) came out 0.
    FIXED 2026-10-10 (A24).

    Args:
        M: mass (kg)
        r: coordinate radius (m), must be > r_s

    Raises:
        ValueError: if r <= r_s.
    """
    rs = schwarzschild_radius(M)
    if not r > rs:
        raise ValueError(f"r={r:.4e} m <= r_s={rs:.4e} m: inside/at the horizon")
    return math.expm1(-0.5 * math.log1p(-rs / r))


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
    if not r > rs:
        raise ValueError(f"r={r:.4e} m <= r_s={rs:.4e} m: inside/at the horizon")
    return math.sqrt(1.0 - rs / r)


def escape_velocity(M, r):
    """Newtonian escape velocity v_esc = sqrt(2GM/r).

    At r = r_s, v_esc = c (which is why light cannot escape).

    Args:
        M: mass (kg)
        r: radius (m), must be > 0
    """
    _check_mass(M)
    if not r > 0:
        raise ValueError(f"r={r} <= 0: radius must be positive")
    return math.sqrt(2.0 * G * M / r)


# ── Special radii ───────────────────────────────────────────────────────

def isco_radius(M):
    """Innermost Stable Circular Orbit radius for a Schwarzschild BH.

    r_ISCO = 6 GM/c^2 = 3 r_s. Inside it, no stable circular orbits exist.
    Reference: Bardeen, Press & Teukolsky 1972, ApJ 178, 347.
    """
    _check_mass(M)
    return 6.0 * G * M / C**2


def photon_sphere_radius(M):
    """Photon sphere radius r_ph = 3 GM/c^2 = 1.5 r_s.

    Photons travel in unstable circular orbits at this radius.
    """
    _check_mass(M)
    return 3.0 * G * M / C**2


# ── Hawking radiation ───────────────────────────────────────────────────

def hawking_temperature(M):
    """Hawking temperature T_H = hbar c^3 / (8 pi G M k_B).

    Smaller black holes are hotter. Solar-mass BH ~ 6.17e-8 K.
    Reference: Hawking 1974, Comm. Math. Phys. 43, 199.

    Args:
        M: black hole mass (kg), must be > 0.
    """
    _check_mass(M)
    return HBAR * C**3 / (8.0 * math.pi * G * M * K_B)


def hawking_luminosity(M):
    """Photon-only horizon-blackbody estimate of the Hawking luminosity (W):
    L = hbar c^6 / (15360 pi G^2 M^2) = sigma_SB T_H^4 * 4 pi r_s^2.

    This is the textbook geometric-optics estimate: one species (photons),
    the horizon as the radiating surface, no greybody factors. It is NOT
    Page's computed rate. Page 1976 (PRD 13, 198), with greybody factors and
    massless neutrinos, photons and gravitons, gives 2.0e-4 hbar c^6/(G^2 M^2)
    for M >> 1e17 g: 9.65x this value (81 % neutrinos, 17 % photons, 2 %
    gravitons); his photon-only rate is about 1.6x this value.
    (Relabelled 2026-10-10; the law itself is unchanged.)

    Args:
        M: black hole mass (kg), must be > 0.
    """
    _check_mass(M)
    return HBAR * C**6 / (15360.0 * math.pi * G**2 * M**2)


def hawking_evaporation_time(M):
    """Evaporation time of the photon-only blackbody estimate (s):
    t_evap = 5120 pi G^2 M^3 / (hbar c^4) = M c^2 / (3 L), with L from
    hawking_luminosity (mass loss dM/dt = -L/c^2 integrated to zero).

    An order-of-magnitude estimate, not a computed lifetime: Page 2013
    (JCAP 09, 028, eq. 4; photons and gravitons with greybody factors) gives
    t_decay = 8895 M^3 (Planck units) = 1.159e67 yr (M/M_sun)^3, i.e. 0.553x
    this value (5120 pi = 16085 in the same units). With Standard-Model
    species the holes evaporating today start near 5e11 kg (MacGibbon, Carr
    and Page 2008); this formula puts that at 1.7e11 kg.
    (Relabelled 2026-10-10; the law itself is unchanged.)

    Args:
        M: initial black hole mass (kg), must be > 0.
    """
    _check_mass(M)
    return 5120.0 * math.pi * G**2 * M**3 / (HBAR * C**4)


def page_time_sm(M):
    """Page time t_Page = 0.53810 t_evap (s).

    When the radiation's entropy has grown to equal the black hole's
    remaining Bekenstein-Hawking entropy (about 59.75 % of the initial).
    Fraction from Page 2013 (JCAP 09, 028, eq. 9; photons + gravitons).
    Caveat: Page applies it to his own t_decay (8895 M^3, Planck units);
    applied here to the photon-only blackbody t_evap (16085 M^3), the
    absolute Page time is 1.81x Page's t* = 6.236e66 yr (M/M_sun)^3.

    Args:
        M: initial black hole mass (kg), must be > 0.
    """
    _check_mass(M)
    return _PAGE_FRACTION_SM * hawking_evaporation_time(M)


def bekenstein_hawking_entropy(M):
    """Bekenstein-Hawking entropy S_BH = k_B * 4 pi G M^2 / (hbar c) (J/K).

    Reference: Bekenstein 1973 (PRD 7, 2333); Hawking 1975.

    Args:
        M: black hole mass (kg), must be > 0.
    """
    _check_mass(M)
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
    _check_mass(M)
    if not r > 0:
        raise ValueError(f"r={r} <= 0: radius must be positive")
    return 2.0 * G * M * m * dr / r**3
