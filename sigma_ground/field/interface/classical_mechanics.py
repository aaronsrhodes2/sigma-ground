"""
Classical mechanics fundamentals — energy, momentum, work, power.

These are the Newtonian building blocks that other modules
(rotational, projectile, dynamics) depend on.

Derivation chains:

  1. Gravitational potential energy
     U = mgh
     FIRST_PRINCIPLES: Newton (1687), Principia.

  2. Kinetic energy
     KE = ½mv²
     FIRST_PRINCIPLES: Leibniz (1695), vis viva.

  3. Work-energy theorem
     W = Fd cos θ = ΔKE
     FIRST_PRINCIPLES: Coriolis (1829).

  4. Momentum and impulse
     p = mv,  J = FΔt = Δp
     FIRST_PRINCIPLES: Newton's second law in integral form.

  5. Mechanical power
     P = Fv = dW/dt
     FIRST_PRINCIPLES: Watt (1782).

  σ-dependence:
    Mass scales as m(σ) = m₀ × scale_ratio(σ) through QCD energy.
    All quantities that depend on mass (KE, PE, momentum, work)
    inherit the shift. Energy and momentum conservation hold at
    every σ — this is Wheeler invariance.
"""

import math

from ..constants import SIGMA_HERE
from ..scale import scale_ratio


# ── Constants ────────────────────────────────────────────────────────

G_EARTH = 9.80665   # m/s², standard gravity (MEASURED, NIST)


# ── Energy ───────────────────────────────────────────────────────────

def gravitational_pe(mass, height, g=G_EARTH, sigma=SIGMA_HERE):
    """Gravitational potential energy U = mgh (Joules).

    FIRST_PRINCIPLES: Newton (1687).

    Args:
        mass: mass in kg
        height: height above reference in m
        g: gravitational acceleration (m/s²)
        sigma: σ-field value

    Returns:
        Potential energy in Joules.
    """
    m = mass * scale_ratio(sigma)
    return m * g * height


def kinetic_energy(mass, velocity, sigma=SIGMA_HERE):
    """Translational kinetic energy KE = ½mv² (Joules).

    FIRST_PRINCIPLES: Leibniz (1695), vis viva.

    Args:
        mass: mass in kg
        velocity: speed in m/s
        sigma: σ-field value

    Returns:
        Kinetic energy in Joules.
    """
    m = mass * scale_ratio(sigma)
    return 0.5 * m * velocity ** 2


def rotational_ke(inertia, angular_velocity, sigma=SIGMA_HERE):
    """Rotational kinetic energy KE_rot = ½Iω² (Joules).

    FIRST_PRINCIPLES: Euler (1750).

    Args:
        inertia: moment of inertia in kg·m²
        angular_velocity: angular speed in rad/s
        sigma: σ-field value (shifts inertia through mass)

    Returns:
        Rotational kinetic energy in Joules.
    """
    I = inertia * scale_ratio(sigma)
    return 0.5 * I * angular_velocity ** 2


def total_mechanical_energy(mass, velocity, height, g=G_EARTH,
                            inertia=0.0, angular_velocity=0.0,
                            sigma=SIGMA_HERE):
    """Total mechanical energy E = KE + KE_rot + PE (Joules).

    For conservative systems, E is constant.

    Args:
        mass: mass in kg
        velocity: translational speed in m/s
        height: height above reference in m
        g: gravitational acceleration (m/s²)
        inertia: moment of inertia in kg·m² (0 for point mass)
        angular_velocity: angular speed in rad/s (0 for non-rotating)
        sigma: σ-field value

    Returns:
        Total mechanical energy in Joules.
    """
    ke = kinetic_energy(mass, velocity, sigma)
    pe = gravitational_pe(mass, height, g, sigma)
    ke_rot = rotational_ke(inertia, angular_velocity, sigma)
    return ke + ke_rot + pe


# ── Work and Power ───────────────────────────────────────────────────

def work_done(force, distance, angle=0.0):
    """Work W = Fd cos θ (Joules).

    FIRST_PRINCIPLES: Coriolis (1829).

    Args:
        force: magnitude of force in N
        distance: displacement in m
        angle: angle between force and displacement in radians

    Returns:
        Work in Joules.
    """
    return force * distance * math.cos(angle)


def power_mechanical(force, velocity):
    """Mechanical power P = Fv (Watts).

    FIRST_PRINCIPLES: instantaneous power.

    Args:
        force: force in N (along velocity direction)
        velocity: speed in m/s

    Returns:
        Power in Watts.
    """
    return force * velocity


def friction_dissipation(friction_force, distance):
    """Energy lost to friction W_f = f × d (Joules).

    FIRST_PRINCIPLES: non-conservative work.
    Always positive (energy removed from mechanical system).

    Args:
        friction_force: friction magnitude in N
        distance: sliding distance in m

    Returns:
        Dissipated energy in Joules (positive).
    """
    return abs(friction_force) * abs(distance)


# ── Momentum and Impulse ─────────────────────────────────────────────

def momentum(mass, velocity, sigma=SIGMA_HERE):
    """Linear momentum p = mv (kg·m/s).

    FIRST_PRINCIPLES: Newton (1687), second law.

    Args:
        mass: mass in kg
        velocity: velocity in m/s (signed)
        sigma: σ-field value

    Returns:
        Momentum in kg·m/s.
    """
    m = mass * scale_ratio(sigma)
    return m * velocity


def impulse(force, duration):
    """Impulse J = FΔt (N·s = kg·m/s).

    FIRST_PRINCIPLES: Newton's second law in integral form.

    Args:
        force: average force in N
        duration: time interval in s

    Returns:
        Impulse in N·s.
    """
    return force * duration


def velocity_from_impulse(impulse_val, mass, v_initial=0.0, sigma=SIGMA_HERE):
    """Final velocity after impulse: v_f = v_i + J/m.

    Args:
        impulse_val: impulse in N·s
        mass: mass in kg
        v_initial: initial velocity in m/s
        sigma: σ-field value

    Returns:
        Final velocity in m/s.
    """
    m = mass * scale_ratio(sigma)
    return v_initial + impulse_val / m


# ── Collision (1D) ───────────────────────────────────────────────────

def elastic_collision_velocities(m1, v1, m2, v2, sigma=SIGMA_HERE):
    """Final velocities after perfectly elastic 1D collision.

    FIRST_PRINCIPLES: conservation of momentum and kinetic energy.

    Args:
        m1, v1: mass and velocity of body 1
        m2, v2: mass and velocity of body 2
        sigma: σ-field value (shifts both masses equally)

    Returns:
        (v1_final, v2_final) in m/s.
    """
    # σ cancels in the ratio — elastic collision velocities are
    # independent of absolute mass scale. This is physically correct:
    # a heavier universe bounces the same way.
    M = m1 + m2
    v1f = ((m1 - m2) * v1 + 2 * m2 * v2) / M
    v2f = ((m2 - m1) * v2 + 2 * m1 * v1) / M
    return v1f, v2f


def inelastic_collision_velocity(m1, v1, m2, v2, cor=1.0, sigma=SIGMA_HERE):
    """Final velocities after 1D collision with coefficient of restitution.

    COR = 1.0: perfectly elastic
    COR = 0.0: perfectly inelastic (stick together)
    0 < COR < 1: real collision

    FIRST_PRINCIPLES: Newton's experimental law of restitution.

    Args:
        m1, v1: mass and velocity of body 1
        m2, v2: mass and velocity of body 2
        cor: coefficient of restitution (0 to 1)
        sigma: σ-field value

    Returns:
        (v1_final, v2_final) in m/s.
    """
    # Again, σ cancels in mass ratios
    M = m1 + m2
    v_cm = (m1 * v1 + m2 * v2) / M
    v1f = v_cm + cor * m2 * (v2 - v1) / M
    v2f = v_cm + cor * m1 * (v1 - v2) / M
    return v1f, v2f


def collision_energy_loss(m1, v1, m2, v2, cor, sigma=SIGMA_HERE):
    """Energy dissipated in a 1D collision (Joules).

    FIRST_PRINCIPLES: ΔKE = ½ μ (1 - e²) (v1 - v2)²
    where μ = m1·m2/(m1+m2) is the reduced mass.

    Args:
        m1, v1: mass and velocity of body 1
        m2, v2: mass and velocity of body 2
        cor: coefficient of restitution
        sigma: σ-field value

    Returns:
        Energy lost in Joules (always ≥ 0).
    """
    s = scale_ratio(sigma)
    mu = (m1 * m2) / (m1 + m2) * s  # reduced mass, σ-shifted
    return 0.5 * mu * (1 - cor ** 2) * (v1 - v2) ** 2


# ── σ-field coupling ─────────────────────────────────────────────────

def sigma_energy_shift(energy, sigma):
    """Energy at σ relative to energy at σ=0.

    All mechanical energies scale with mass, which scales as e^σ.

    FIRST_PRINCIPLES: Wheeler invariance — E = mc² holds at every σ.

    Args:
        energy: energy at σ=0 in Joules
        sigma: σ-field value

    Returns:
        Energy at σ in Joules.
    """
    return energy * scale_ratio(sigma)


# ── Nagatha Export ───────────────────────────────────────────────────

def mechanics_report(mass, velocity, height=0.0, g=G_EARTH,
                     inertia=0.0, angular_velocity=0.0,
                     sigma=SIGMA_HERE):
    """Export classical mechanics state in Nagatha-compatible format.

    Args:
        mass: mass in kg
        velocity: speed in m/s
        height: height in m
        g: gravitational acceleration (m/s²)
        inertia: moment of inertia in kg·m²
        angular_velocity: angular speed in rad/s
        sigma: σ-field value

    Returns:
        Dict with all mechanical quantities and origin tags.
    """
    s = scale_ratio(sigma)
    m_eff = mass * s
    ke = 0.5 * m_eff * velocity ** 2
    pe = m_eff * g * height
    ke_rot = 0.5 * (inertia * s) * angular_velocity ** 2
    p = m_eff * velocity

    return {
        'mass_kg': m_eff,
        'velocity_m_s': velocity,
        'height_m': height,
        'sigma': sigma,
        'kinetic_energy_J': ke,
        'potential_energy_J': pe,
        'rotational_ke_J': ke_rot,
        'total_energy_J': ke + pe + ke_rot,
        'momentum_kg_m_s': p,
        'origin': (
            'FIRST_PRINCIPLES: Newtonian mechanics. '
            'KE=½mv², PE=mgh, KE_rot=½Iω². '
            'σ-shift through mass scaling (Wheeler invariance).'
        ),
    }
