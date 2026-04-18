"""
General Relativity Basics — σ-grounded Schwarzschild geometry.

Schwarzschild metric quantities, horizons, and Hawking radiation,
with every constant derived from sigma_ground.field.constants.

Note: schwarzschild_radius() and sigma_at_event_horizon() already live
in sigma_ground.field.scale (the σ-field module that originally needed
them).  This module imports and re-exports them for completeness, and
adds the broader GR toolkit (redshift, ISCO, Hawking radiation, tidal
forces, etc.).

References:
  Misner, Thorne & Wheeler "Gravitation" (Princeton, 1973)
  Wald "General Relativity" (Chicago, 1984)
  Hawking 1974, Comm. Math. Phys. 43, 199
  Bardeen, Press & Teukolsky 1972, ApJ 178, 347

σ-connection
------------
sigma_at_horizon() confirms the universal result σ = ξ/2 ≈ 0.079 at any
Schwarzschild radius — a key consistency check of the σ-field framework.
The σ-field value at the horizon is mass-independent, which is why black
holes of all masses produce the same ratio of scale transition.

Dependency: G, C, HBAR, K_B, M_SUN_KG from constants; scale.py
"""

import math

from .constants import G, C, HBAR, K_B, M_SUN_KG, ETA, L_PLANCK
from .scale import (
    schwarzschild_radius,        # r_s = 2GM/c²
    sigma_at_event_horizon,      # always ξ/2
    sigma_from_potential,        # σ = ξGM/(rc²)
)

# Re-export scale.py helpers under the GR module namespace
__all__ = [
    'schwarzschild_radius',
    'sigma_at_event_horizon',
    'gravitational_redshift',
    'time_dilation_gr',
    'escape_velocity',
    'isco_radius',
    'photon_sphere_radius',
    'hawking_temperature',
    'hawking_luminosity',
    'hawking_evaporation_time',
    'page_time_sm',
    'page_time_sigma_ground',
    'replica_wormhole_entropy_contribution',
    'bekenstein_hawking_entropy',
    'eta_as_bit_thread_fraction',
    'tidal_force',
    'sigma_at_horizon',
]


# ── Replica wormholes (arXiv:2504.18663) ──────────────────────────────
# Simplicial-quantum-gravity construction of the replica wormhole saddle
# that reproduces the Page curve. Provides a path-integral mechanism for
# the saturation of entanglement entropy at t_Page.
#
# Scaffolded stub: the paper's simplicial numerics are not lifted here.
# Until they are, the function refuses to return a fabricated value.

def replica_wormhole_entropy_contribution(M, n_replicas=None):
    """Entropy contribution of the replica-wormhole saddle at a given BH mass.

    Intended signature for future implementation of arXiv:2504.18663's
    simplicial-QG replica computation. The full calculation needs:
    - the replica-index n and continuation to n→1
    - the simplicial regularization parameter
    - the saddle action evaluated in the relevant geometry

    None of these are lifted yet. The function raises until they are.

    Args:
        M: black-hole mass (kg).
        n_replicas: integer replica index n; None means "analytically
            continued n→1" (Rényi → von Neumann entropy).

    Raises:
        NotImplementedError: until arXiv:2504.18663's simplicial action
        coefficients are lifted into constants.py and this function.
    """
    _ = M
    _ = n_replicas
    raise NotImplementedError(
        "Replica-wormhole simplicial numerics from arXiv:2504.18663 not yet "
        "lifted. Pending: replica-index continuation, simplicial action "
        "coefficients, saddle-geometry parameters."
    )


# ── Bit-thread entropy (arXiv:2508.18941) ──────────────────────────────
# Differential entropy on AdS boundary equals the Bekenstein-Hawking bulk
# entropy, via the bit-thread reformulation of Ryu-Takayanagi. The
# sigma-ground hook (UP-007): η as the bit-thread density fraction on the
# σ=σ_conv iso-surface.

def bekenstein_hawking_entropy(M):
    """Bekenstein-Hawking entropy S_BH = k_B × 4π G M² / (ℏ c).

    Units: Joules per Kelvin (SI entropy), or equivalently k_B × N where
    N is the number of Planck-area bits on the horizon.

    Reference: Bekenstein 1973 (PRD 7, 2333); Hawking 1975.

    Args:
        M: black hole mass (kg), > 0.

    Returns:
        S_BH in J/K.
    """
    if M <= 0:
        raise ValueError(f"M={M} ≤ 0: mass must be positive")
    return K_B * 4.0 * math.pi * G * M * M / (HBAR * C)


def eta_as_bit_thread_fraction(M):
    """Report η's interpretation as bit-thread density fraction (UP-007).

    Prediction: the bit-thread density through the σ = σ_conv iso-surface
    equals η × (1 Planck area)⁻¹. This provides a geometric interpretation
    of η as the "active fraction" of Planck cells on the conversion surface.

    The full prediction requires an entanglement-entropy computation on a
    Schwarzschild-AdS background with a sigma-ground-specified σ-field
    profile, which is not yet set up. For now we report the trivial
    consequence: N_bits = S_BH / k_B, and N_active = η × N_bits.

    Args:
        M: black hole mass (kg), > 0.

    Returns:
        dict with 'n_bits_total', 'n_bits_active', 'eta', 'note'.
    """
    if M <= 0:
        raise ValueError(f"M={M} ≤ 0")
    s_bh = bekenstein_hawking_entropy(M)
    n_bits_total = s_bh / K_B            # dimensionless
    n_bits_active = ETA * n_bits_total
    return {
        'n_bits_total': n_bits_total,
        'n_bits_active': n_bits_active,
        'eta': ETA,
        'note': 'UP-007 prediction: bit-thread density on sigma_conv = eta / Planck_area',
    }


# ── Page time ──────────────────────────────────────────────────────────
# Page-time prediction gate (Phase XII.a.2, arXiv:2502.04430).
# Page 1993 (Phys. Rev. Lett. 71, 3743) showed that for a Schwarzschild BH
# evaporating into a thermal environment, the radiation's entanglement
# entropy peaks at about half the BH's initial entropy at t ≈ 0.5384 × t_evap.
# The exact numerical coefficient depends on the massless-species count;
# we use Page's single-field result as the SM reference.
_PAGE_FRACTION_SM = 0.5384


# ── Schwarzschild Geometry ─────────────────────────────────────────────

def gravitational_redshift(M, r):
    """Gravitational redshift z = 1/√(1 − r_s/r) − 1.

    A photon climbing out of a gravitational well loses energy.
    At r = r_s: z → ∞ (photons cannot escape — event horizon).

    Args:
        M: mass (kg)
        r: coordinate radius (m), must be > r_s

    Returns:
        z (dimensionless, ≥ 0)

    Raises:
        ValueError: if r ≤ r_s (inside or at the horizon)
    """
    rs = schwarzschild_radius(M)
    if r <= rs:
        raise ValueError(
            f"r={r:.4e} m ≤ r_s={rs:.4e} m: inside or at the event horizon"
        )
    return 1.0 / math.sqrt(1.0 - rs / r) - 1.0


def time_dilation_gr(M, r):
    """Gravitational time dilation factor τ/t = √(1 − r_s/r).

    The ratio of proper time τ (at radius r) to coordinate time t (at
    infinity).  A clock near a massive object runs slow.

    At r = r_s: factor → 0 (time stops at the horizon).
    At r → ∞: factor → 1 (standard time at infinity).

    Args:
        M: mass (kg)
        r: coordinate radius (m), must be > r_s

    Returns:
        τ/t (dimensionless, 0 < factor ≤ 1)

    Raises:
        ValueError: if r ≤ r_s
    """
    rs = schwarzschild_radius(M)
    if r <= rs:
        raise ValueError(
            f"r={r:.4e} m ≤ r_s={rs:.4e} m: inside or at the event horizon"
        )
    return math.sqrt(1.0 - rs / r)


def escape_velocity(M, r):
    """Newtonian escape velocity v_esc = √(2GM/r).

    At r = r_s: v_esc = c (which is why light cannot escape the BH).
    Note: beyond GR, the relativistic result is more complex, but
    v_esc = c at r = r_s holds exactly in both frameworks.

    Args:
        M: mass (kg)
        r: radius (m), must be > 0

    Returns:
        v_esc (m/s)
    """
    if r <= 0:
        raise ValueError(f"r={r} ≤ 0: radius must be positive")
    return math.sqrt(2.0 * G * M / r)


# ── Special Radii ──────────────────────────────────────────────────────

def isco_radius(M):
    """Innermost Stable Circular Orbit radius for a Schwarzschild BH.

    r_ISCO = 6 GM/c² = 3 r_s

    Inside r_ISCO, no stable circular orbits exist — particles spiral in.
    This is the inner edge of the accretion disk in the Schwarzschild case.

    Reference: Bardeen, Press & Teukolsky 1972, ApJ 178, 347

    Args:
        M: black hole mass (kg)

    Returns:
        r_ISCO (m)
    """
    return 6.0 * G * M / C**2


def photon_sphere_radius(M):
    """Photon sphere radius r_ph = 3 GM/c² = 1.5 r_s.

    Photons travel in unstable circular orbits at this radius.
    Outside: photons can escape.  Inside: captured by the BH.

    Args:
        M: black hole mass (kg)

    Returns:
        r_ph (m)
    """
    return 3.0 * G * M / C**2


# ── Hawking Radiation ──────────────────────────────────────────────────

def hawking_temperature(M):
    """Hawking temperature T_H = ℏc³ / (8πGMk_B).

    Black holes emit thermal radiation at this temperature.
    Smaller black holes are hotter.

    Reference: Hawking 1974, Comm. Math. Phys. 43, 199

    Benchmarks:
      Solar-mass BH (M ≈ 2e30 kg): T_H ≈ 6.17e-8 K
      Primordial BH (M = 1e12 kg): T_H ≈ 1.23e11 K (very hot)

    Args:
        M: black hole mass (kg), must be > 0

    Returns:
        T_H (K)
    """
    if M <= 0:
        raise ValueError(f"M={M} ≤ 0: mass must be positive")
    return HBAR * C**3 / (8.0 * math.pi * G * M * K_B)


def hawking_luminosity(M):
    """Hawking luminosity L_H ∝ ℏc⁶ / (15360π G² M²).

    Power emitted by Hawking radiation (black body at T_H).

    Reference: Page 1976, Phys. Rev. D 13, 198

    Args:
        M: black hole mass (kg), must be > 0

    Returns:
        L_H (W)
    """
    if M <= 0:
        raise ValueError(f"M={M} ≤ 0: mass must be positive")
    return HBAR * C**6 / (15360.0 * math.pi * G**2 * M**2)


def hawking_evaporation_time(M):
    """Time for a black hole to fully evaporate via Hawking radiation.

    t_evap = 5120π G² M³ / (ℏ c⁴)

    Reference: Hawking 1974; Page 1976

    Args:
        M: initial black hole mass (kg), must be > 0

    Returns:
        t_evap (s)
    """
    if M <= 0:
        raise ValueError(f"M={M} ≤ 0: mass must be positive")
    return 5120.0 * math.pi * G**2 * M**3 / (HBAR * C**4)


def page_time_sm(M):
    """Standard-Model Page time: t_Page ≈ 0.5384 × t_evap.

    The Page time is when the Hawking radiation's entanglement entropy
    with the black hole has built up to half the BH's initial
    Bekenstein-Hawking entropy. Beyond this point, purity is being
    returned to the exterior rather than accumulated in the radiation.

    Reference: Page 1993 (Phys. Rev. Lett. 71, 3743);
               arXiv:2502.04430 for SM/BSM Page-time comparison.

    [THEORETICAL] gate prediction; matches SM single-field radiation.

    Args:
        M: initial black hole mass (kg), must be > 0.

    Returns:
        Page time in seconds.
    """
    if M <= 0:
        raise ValueError(f"M={M} ≤ 0: mass must be positive")
    return _PAGE_FRACTION_SM * hawking_evaporation_time(M)


def page_time_sigma_ground(M):
    """Sigma-ground Page time: t_Page,SG = (1 − η/2) × t_Page,SM.

    In sigma-ground, only the fraction (1 − η/2) of Hawking-radiation
    entanglement thermalizes with the σ=0 exterior — the remaining η/2
    is retained in the cross-hadron entanglement channel that defines η
    itself. The Page time shifts earlier by the same factor.

    Prediction (UP-002 in misc/unpublished_predictions.md):
        t_Page,SG / t_Page,SM = (1 − η/2) ≈ 0.7924

    [SPECULATIVE] — this is a falsifiable sigma-ground prediction.
    arXiv:2502.04430 provides the SM + BSM Page-time benchmark to test against.

    Args:
        M: initial black hole mass (kg), must be > 0.

    Returns:
        Sigma-ground Page time in seconds.
    """
    if M <= 0:
        raise ValueError(f"M={M} ≤ 0: mass must be positive")
    return (1.0 - ETA / 2.0) * page_time_sm(M)


# ── Tidal Forces ───────────────────────────────────────────────────────

def tidal_force(M, r, dr, m=1.0):
    """Tidal (differential gravitational) force on a test mass.

    ΔF = 2 G M m dr / r³

    This is the radial tidal acceleration difference across a distance dr.
    At the Schwarzschild radius of a stellar BH, tidal forces are lethal.

    Args:
        M: gravitating mass (kg)
        r: distance to center of mass (m), must be > 0
        dr: separation of the two test mass points along the radial (m)
        m: test mass (kg). Default: 1.0 (returns acceleration difference in m/s²)

    Returns:
        tidal force (N) for mass m, or tidal acceleration (m/s²) for m=1
    """
    if r <= 0:
        raise ValueError(f"r={r} ≤ 0: radius must be positive")
    return 2.0 * G * M * m * dr / r**3


# ── σ-Connection ───────────────────────────────────────────────────────

def sigma_at_horizon(M):
    """σ-field value at the Schwarzschild radius of a black hole.

    σ = ξ G M / (r_s c²) = ξ G M / (2 G M) = ξ/2

    This is mass-independent — every black hole, regardless of mass, has
    exactly σ = ξ/2 ≈ 0.079 at its event horizon.

    In the SSBM framework, this is the universal 'entry condition' for
    the σ-transition.  The bond failure cascade from ξ/2 to σ_conv ≈ 1.849
    happens inside the event horizon, not at it.

    Re-exports sigma_at_event_horizon() from sigma_ground.field.scale.

    Args:
        M: black hole mass (kg) — provided for API symmetry but does not
           affect the result (σ is mass-independent)

    Returns:
        σ at the event horizon = ξ/2 (dimensionless)
    """
    return sigma_at_event_horizon(M)
