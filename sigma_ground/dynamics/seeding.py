"""Phase E seeding — ONE set of initial conditions, consumed by BOTH engines.

The fluid-vs-gravity comparison (Phase E) runs the same mass distribution
twice: once through the n-body integrator (field.interface.nbody — pure
gravity, symplectic) and once through the SPH continuum bridge
(dynamics.continuum with self_gravity_G — gravity + pressure + viscosity).
For the residual between them to mean anything, both must start from
byte-identical initial conditions. This module is that single source:
deterministic seeds (random.Random(seed), never global random) producing
engine-neutral SeedBody lists, plus adapters into each engine's native
objects.

Two scenarios, per the Phase E plan (2026-07-29 addendum):

  seed_ring_annulus()   — a scaled Saturn-ring analogue: a central mass
      with a thin annulus of particles on circular Keplerian orbits.
      Real ring science treats rings BOTH ways (viscous fluid disk AND
      self-gravitating n-body swarm); the discrepancy is a real published
      phenomenon (self-gravity wakes), which is what makes this the
      primary comparison scenario. roche_limit() (nbody.py) supplies the
      physical anchor for WHERE an annulus is a swarm rather than a moon.

  seed_rotating_cloud() — a rotating, self-gravitating cloud: the textbook
      SPH-vs-n-body cross-validation setup. The n-body side undergoes
      pressureless collapse; the SPH side halts at pressure support. The
      residual (pressure) is the known, well-understood divergence.

Every dimensional number is a caller-supplied parameter — nothing here
pretends to BE Saturn; scaled scenarios are Choices the caller must
disclose. All randomness is seeded and reproducible.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .vec import Vec3

# CODATA-measured G, the same source nbody.py uses (field.constants.G).
G_CODATA = 6.67430e-11    # m^3 kg^-1 s^-2


@dataclass(frozen=True)
class SeedBody:
    """One engine-neutral body: plain floats, no engine types."""
    mass_kg: float
    position_m: tuple      # (x, y, z)
    velocity_m_s: tuple    # (x, y, z)
    radius_m: float
    label: str = ""


def _zero_total_momentum(bodies):
    """Shift every velocity into the zero-momentum (barycentric) frame, so
    neither engine sees a net drift the other doesn't."""
    total_m = sum(b.mass_kg for b in bodies)
    px = sum(b.mass_kg * b.velocity_m_s[0] for b in bodies) / total_m
    py = sum(b.mass_kg * b.velocity_m_s[1] for b in bodies) / total_m
    pz = sum(b.mass_kg * b.velocity_m_s[2] for b in bodies) / total_m
    return [SeedBody(b.mass_kg,
                     b.position_m,
                     (b.velocity_m_s[0] - px,
                      b.velocity_m_s[1] - py,
                      b.velocity_m_s[2] - pz),
                     b.radius_m, b.label) for b in bodies]


def seed_ring_annulus(n, *, central_mass_kg, central_radius_m,
                      r_inner_m, r_outer_m, ring_mass_kg,
                      particle_density_kg_m3,
                      thickness_m=0.0, velocity_dispersion_frac=0.0,
                      G=G_CODATA, seed=0):
    """Central mass + n equal-mass particles in a thin Keplerian annulus.

    Radii are sampled sqrt-uniform (uniform SURFACE density — the annulus
    area element is 2πr·dr, so uniform-in-r would crowd the inner edge);
    azimuth uniform; z uniform in ±thickness_m/2. Each particle starts on
    the circular Keplerian orbit for its radius, v = sqrt(G·M_central/r),
    tangential, prograde about +z, with an optional isotropic Gaussian
    dispersion of velocity_dispersion_frac·v (a disclosed Choice — real
    rings are dynamically cold but not perfectly so).

    Returns [central, particle_0, ..., particle_{n-1}] in the barycentric
    zero-momentum frame.
    """
    if not 0 < r_inner_m < r_outer_m:
        raise ValueError("need 0 < r_inner_m < r_outer_m")
    rng = random.Random(seed)
    m_p = ring_mass_kg / n
    r_p = (3.0 * m_p / (4.0 * math.pi * particle_density_kg_m3)) ** (1.0 / 3.0)

    bodies = [SeedBody(central_mass_kg, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                       central_radius_m, "central")]
    for i in range(n):
        u = rng.random()
        r = math.sqrt(r_inner_m ** 2 + u * (r_outer_m ** 2 - r_inner_m ** 2))
        th = rng.random() * 2.0 * math.pi
        z = (rng.random() - 0.5) * thickness_m
        x, y = r * math.cos(th), r * math.sin(th)
        v = math.sqrt(G * central_mass_kg / r)
        vx, vy, vz = -v * math.sin(th), v * math.cos(th), 0.0
        if velocity_dispersion_frac:
            s = velocity_dispersion_frac * v
            vx += rng.gauss(0.0, s)
            vy += rng.gauss(0.0, s)
            vz += rng.gauss(0.0, s)
        bodies.append(SeedBody(m_p, (x, y, z), (vx, vy, vz), r_p,
                               f"ring_{i}"))
    return _zero_total_momentum(bodies)


def seed_rotating_cloud(n, *, cloud_mass_kg, cloud_radius_m,
                        omega_rad_s, particle_density_kg_m3,
                        jitter_frac=0.0, seed=0):
    """n equal-mass particles uniformly filling a sphere, in solid-body
    rotation about +z: v = ω × r. Optional isotropic velocity jitter of
    jitter_frac·(ω·R) breaks perfect symmetry (a disclosed Choice; real
    clouds are turbulent, and a perfectly symmetric collapse hides the
    fragmentation physics the comparison is looking for).

    Returns the particle list in the barycentric zero-momentum frame.
    """
    rng = random.Random(seed)
    m_p = cloud_mass_kg / n
    r_p = (3.0 * m_p / (4.0 * math.pi * particle_density_kg_m3)) ** (1.0 / 3.0)
    v_ref = abs(omega_rad_s) * cloud_radius_m

    bodies = []
    for i in range(n):
        # uniform-in-volume sphere sample: r ~ R·u^(1/3), direction isotropic
        r = cloud_radius_m * rng.random() ** (1.0 / 3.0)
        cos_t = 2.0 * rng.random() - 1.0
        sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
        ph = rng.random() * 2.0 * math.pi
        x = r * sin_t * math.cos(ph)
        y = r * sin_t * math.sin(ph)
        z = r * cos_t
        # solid-body rotation about +z: v = ω ẑ × r = ω(−y, x, 0)
        vx, vy, vz = -omega_rad_s * y, omega_rad_s * x, 0.0
        if jitter_frac:
            s = jitter_frac * v_ref
            vx += rng.gauss(0.0, s)
            vy += rng.gauss(0.0, s)
            vz += rng.gauss(0.0, s)
        bodies.append(SeedBody(m_p, (x, y, z), (vx, vy, vz), r_p,
                               f"cloud_{i}"))
    return _zero_total_momentum(bodies)


# ── adapters: the SAME SeedBody list into each engine's native objects ──────

def to_nbody(bodies, *, softening_m=0.0):
    """SeedBody list → field.interface.nbody.NBodySystem (pure gravity)."""
    import numpy as np
    from ..field.interface.nbody import CelestialBody, NBodySystem
    cbs = [CelestialBody(mass_kg=b.mass_kg,
                         position_m=np.array(b.position_m, dtype=float),
                         velocity_m_s=np.array(b.velocity_m_s, dtype=float),
                         radius_m=b.radius_m,
                         love_number_k2=0.0)
           for b in bodies]
    return NBodySystem(cbs, softening_m=softening_m)


def to_continuum(bodies, *, material_key, G=G_CODATA, smoothing_h=None,
                 gravity_softening_m=None, temperature_K=300.0):
    """SeedBody list → ContinuumScene with self-gravity (fluid side).

    Parcel radius is re-derived from each body's MASS and the material's
    real density — so both engines carry identical masses even though the
    SPH side's density is a material fact, not a free parameter. The seed's
    particle_density_kg_m3 should be the same material's density for the
    radii to agree too; a mismatch changes packing, not mass, and is the
    caller's disclosed Choice.

    KNOWN LIMIT (from the Phase E plan): the scene uses ONE global
    smoothing length. A huge central body among small ring particles is
    fine for GRAVITY (uncutoff, no kernel involved) but the central parcel
    must sit farther than 2h from ring particles for the kernel terms to
    vanish — true by construction for any sane annulus geometry.
    """
    from .continuum import ContinuumParcel, ContinuumScene
    from ..field.interface.surface import MATERIALS
    rho = MATERIALS[material_key]['density_kg_m3']
    parcels = []
    for b in bodies:
        r = (3.0 * b.mass_kg / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
        parcels.append(ContinuumParcel(
            material_key, r, temperature=temperature_K,
            position=Vec3(*b.position_m), velocity=Vec3(*b.velocity_m_s),
            label=b.label))
    return ContinuumScene(parcels, gravity=Vec3(0.0, 0.0, 0.0), ground=False,
                          smoothing_h=smoothing_h, self_gravity_G=G,
                          gravity_softening_m=gravity_softening_m)


__all__ = ["SeedBody", "seed_ring_annulus", "seed_rotating_cloud",
           "to_nbody", "to_continuum", "G_CODATA"]
