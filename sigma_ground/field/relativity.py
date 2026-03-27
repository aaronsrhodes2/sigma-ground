"""
Special Relativity — σ-grounded SR kinematics.

Standard SR with every constant derived from sigma_ground.field.constants.
No magic numbers. All formulas reference Einstein 1905 or Jackson
"Classical Electrodynamics" Ch. 11.

σ-connection
------------
Inside a compressed spacetime pocket (σ > 0), the local clock runs slow
relative to the external frame by the scale factor e^σ.
See sigma_time_dilation().

Dependency: C from constants only — the most fundamental SR module.
"""

import math

from .constants import C
from .scale import scale_ratio


# ── Lorentz Factor ─────────────────────────────────────────────────────

def lorentz_factor(v):
    """Lorentz factor γ = 1/√(1−v²/c²).

    Domain: 0 ≤ |v| < c.

    Args:
        v: speed (m/s), may be negative (sign is ignored)

    Returns:
        γ (dimensionless, ≥ 1.0)

    Raises:
        ValueError: if |v| ≥ c
    """
    v = abs(v)
    if v >= C:
        raise ValueError(
            f"v={v:.6e} m/s ≥ c={C:.6e} m/s: Lorentz factor undefined at or above c"
        )
    beta_sq = (v / C) ** 2
    return 1.0 / math.sqrt(1.0 - beta_sq)


def beta(v):
    """Velocity ratio β = v/c.

    Args:
        v: speed (m/s)

    Returns:
        β (dimensionless, 0 ≤ β < 1)
    """
    return abs(v) / C


# ── Energy & Momentum ──────────────────────────────────────────────────

def rest_energy(m0):
    """Rest energy E₀ = m₀c².

    Args:
        m0: rest mass (kg)

    Returns:
        rest energy (J)
    """
    return m0 * C**2


def relativistic_energy(m0, v):
    """Total relativistic energy E = γm₀c².

    Args:
        m0: rest mass (kg)
        v: speed (m/s)

    Returns:
        total energy (J)
    """
    return lorentz_factor(v) * m0 * C**2


def kinetic_energy_rel(m0, v):
    """Relativistic kinetic energy K = (γ−1)m₀c².

    Equals the classical ½m₀v² at v ≪ c.

    Args:
        m0: rest mass (kg)
        v: speed (m/s)

    Returns:
        kinetic energy (J)
    """
    return (lorentz_factor(v) - 1.0) * m0 * C**2


def momentum_rel(m0, v):
    """Relativistic momentum p = γm₀v.

    Args:
        m0: rest mass (kg)
        v: speed (m/s)

    Returns:
        momentum magnitude (kg·m/s)
    """
    return lorentz_factor(v) * m0 * abs(v)


def energy_momentum_invariant(m0):
    """Rest-mass invariant: E² − (pc)² = (m₀c²)².

    Returns (m₀c²)² in J² — a Lorentz invariant.

    Args:
        m0: rest mass (kg)

    Returns:
        (m₀c²)² in J²
    """
    return (m0 * C**2) ** 2


# ── Kinematics ─────────────────────────────────────────────────────────

def velocity_addition(u, v):
    """Relativistic collinear velocity addition.

    w = (u + v) / (1 + uv/c²)

    At u, v ≪ c: approaches u + v (classical limit).

    Args:
        u: first velocity (m/s), signed
        v: second velocity (m/s), signed

    Returns:
        combined velocity (m/s), −c < w < c
    """
    return (u + v) / (1.0 + u * v / C**2)


def length_contraction(L0, v):
    """Length contraction L = L₀/γ.

    The proper length L₀ is measured in the object's rest frame.

    Args:
        L0: proper length (m)
        v: relative speed (m/s)

    Returns:
        contracted length in the moving frame (m)
    """
    return L0 / lorentz_factor(v)


def time_dilation(t0, v):
    """Time dilation t = γt₀  (coordinate time from proper time).

    A moving clock ticks slow: proper time t₀ in the moving frame
    corresponds to coordinate time γt₀ in the lab frame.

    Args:
        t0: proper time interval (s)
        v: speed of the moving clock (m/s)

    Returns:
        coordinate time interval (s)
    """
    return lorentz_factor(v) * t0


def doppler_factor(v, cos_theta):
    """Relativistic Doppler factor D.

    D = 1 / (γ (1 − β cos θ))

    Observed frequency f_obs = D × f_emitted.

    Args:
        v: source speed (m/s)
        cos_theta: cosine of the angle between source velocity and the
                   direction from source to observer.
                   cos_theta = +1: source approaching head-on (max blueshift).
                   cos_theta = −1: source receding (max redshift).
                   cos_theta = 0: transverse motion (pure time-dilation redshift).

    Returns:
        Doppler factor D (dimensionless, > 0)
    """
    gamma = lorentz_factor(v)
    b = v / C
    return 1.0 / (gamma * (1.0 - b * cos_theta))


# ── σ-Connection ───────────────────────────────────────────────────────

def sigma_time_dilation(sigma, t0):
    """Coordinate time elapsed outside a σ-compressed spacetime pocket.

    Inside a region of elevated σ (compressed spacetime), the local clock
    runs slow relative to the external frame by the scale factor e^σ:

        t_coord = t₀ × e^σ

    At σ = 0: no dilation — standard flat spacetime.
    At σ_conv ≈ 1.849: dilation factor e^1.849 ≈ 6.35.
    At the event horizon of a solar-mass BH: σ ≈ ξ/2 ≈ 0.079, factor ≈ 1.08.

    This is the σ-field analogue of gravitational time dilation.
    Uses scale_ratio(σ) = e^σ from sigma_ground.field.scale.

    Args:
        sigma: σ-field value (dimensionless, ≥ 0)
        t0: proper time interval inside the compressed pocket (s)

    Returns:
        coordinate time interval in the external frame (s)
    """
    return t0 * scale_ratio(sigma)
