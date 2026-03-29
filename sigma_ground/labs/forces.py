"""
External force callbacks for the simulation engine.

These functions compute forces beyond gravity that act on parcels:
  - Aerodynamic drag (Stokes + turbulent regimes)
  - Buoyancy (Archimedes' principle)
  - Combined callback for the stepper

Physics:
  Drag force:
    F_drag = -½ ρ_medium v² C_d A v̂

    C_d depends on Reynolds number Re = ρvL/μ:
      Re < 1:    C_d = 24/Re        (Stokes regime — viscous dominates)
      1 < Re < 1000: C_d = 24/Re × (1 + 0.15 Re^0.687)  (Schiller-Naumann)
      Re > 1000: C_d = 0.44         (Newton regime — pressure drag dominates)

    FIRST_PRINCIPLES: Stokes drag from Navier-Stokes linearization.
    MEASURED: the 0.44 plateau and Schiller-Naumann transition.

  Buoyancy:
    F_buoy = -ρ_medium V g
    FIRST_PRINCIPLES: Archimedes' principle (displaced fluid weight).

σ-dependence:
  None directly. Medium density and viscosity are σ-invariant
  (electromagnetic properties). Object mass and volume carry σ
  through the material cascade.
"""

import math
from ..dynamics.vec import Vec3


def drag_force(parcel, medium_density, medium_viscosity):
    """Compute aerodynamic drag on a parcel.

    Args:
        parcel:           PhysicsParcel with .velocity, .radius, .shape
        medium_density:   ρ of surrounding medium (kg/m³)
        medium_viscosity: μ of surrounding medium (Pa·s)

    Returns:
        Vec3 drag force in newtons.
    """
    if medium_density < 1e-15:
        return Vec3(0, 0, 0)

    v = parcel.velocity
    speed = v.length()
    if speed < 1e-12:
        return Vec3(0, 0, 0)

    # Characteristic length = diameter for spheres
    L = 2.0 * parcel.radius

    # Reynolds number
    if medium_viscosity > 1e-15:
        Re = medium_density * speed * L / medium_viscosity
    else:
        Re = 1e6  # inviscid limit → turbulent

    # Drag coefficient (sphere)
    if Re < 1.0:
        Cd = 24.0 / max(Re, 1e-10)
    elif Re < 1000.0:
        # Schiller-Naumann correlation
        Cd = 24.0 / Re * (1.0 + 0.15 * Re ** 0.687)
    else:
        Cd = 0.44

    # Cross-sectional area
    A = parcel.shape.cross_section()

    # F = -½ ρ v² Cd A v̂
    F_mag = 0.5 * medium_density * speed * speed * Cd * A
    # Direction: opposite to velocity
    v_hat = v * (-1.0 / speed)
    return v_hat * F_mag


def buoyancy_force(parcel, medium_density, gravity):
    """Compute buoyancy force on a parcel.

    F_buoy = -ρ_medium × V_object × g

    Args:
        parcel:         PhysicsParcel with .shape.volume()
        medium_density: ρ of surrounding medium (kg/m³)
        gravity:        Vec3 gravitational acceleration

    Returns:
        Vec3 buoyancy force in newtons.
    """
    if medium_density < 1e-15:
        return Vec3(0, 0, 0)

    V = parcel.shape.volume()
    # Buoyancy opposes gravity: F = -ρ_medium * V * g
    return gravity * (-medium_density * V)


def combined_forces(medium_density, medium_viscosity, gravity):
    """Build a force callback for the stepper.

    Returns a function f(parcel) -> Vec3 that computes the total
    external force (drag + buoyancy) on a parcel.

    Args:
        medium_density:   ρ of surrounding medium (kg/m³)
        medium_viscosity: μ of surrounding medium (Pa·s)
        gravity:          Vec3 gravitational acceleration

    Returns:
        Callable[[PhysicsParcel], Vec3]
    """
    def callback(parcel):
        fd = drag_force(parcel, medium_density, medium_viscosity)
        fb = buoyancy_force(parcel, medium_density, gravity)
        return fd + fb

    return callback
