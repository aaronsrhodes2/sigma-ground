"""Rotational dynamics + atomic angular-momentum tools (standard physics).

Composite tools cascading through field.interface.{rotational,
angular_momentum}. Rolling-from-geometry uses a real Shape instance.
"""
from __future__ import annotations

import math
from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_SRC = "sigma_ground.field.interface (rotational, angular_momentum)"


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        return None


def rotational_dynamics(mass_kg: float, radius_m: float, angle_deg: float = 30.0,
                        angular_velocity_rad_s: float = 10.0) -> dict[str, Any]:
    """Rotational dynamics bundle: rod moment of inertia, parallel-axis shift,
    angular momentum, torque & angular acceleration, plus rolling-down-a-ramp
    speed/distance/time and shape-geometry moment of inertia + rolling
    acceleration. e.g. rotational_dynamics(2, 0.5)."""
    from sigma_ground.field.interface import rotational as R
    try:
        from sigma_ground.shapes import Sphere
        sph = Sphere(radius_m)
    except Exception:
        sph = None
    ang = math.radians(angle_deg)
    I_rod = _safe(R.moment_of_inertia_rod, mass_kg, 2.0 * radius_m)
    tau = _safe(R.torque, 10.0, radius_m)
    results = {
        "moment_of_inertia_rod_kg_m2": I_rod,
        "parallel_axis_I_kg_m2": _safe(R.parallel_axis, I_rod if I_rod else 1.0, mass_kg, radius_m),
        "angular_momentum_kg_m2_s": _safe(R.angular_momentum, I_rod if I_rod else 1.0, angular_velocity_rad_s),
        "torque_Nm": tau,
        "angular_acceleration_rad_s2": _safe(R.angular_acceleration, tau if tau else 5.0, I_rod if I_rod else 1.0),
        "rolling_speed_from_1m_drop_m_s": _safe(R.rolling_speed_from_height, 1.0),
        "rolling_distance_on_flat_m": _safe(R.rolling_distance_on_flat, 5.0, 0.05),
        "rolling_time_on_flat_s": _safe(R.rolling_time_on_flat, 5.0, 0.05),
        "ramp_to_flat_distance_m": _safe(R.ramp_to_flat_distance, 1.0, ang, 0.05),
        "shape_moment_of_inertia_kg_m2": _safe(R.shape_moment_of_inertia, sph, mass_kg) if sph else None,
        "shape_rolling_acceleration_m_s2": _safe(R.shape_rolling_acceleration, sph, ang) if sph else None,
        "shape_rolling_speed_m_s": _safe(R.shape_rolling_speed_from_height, sph, 1.0) if sph else None,
    }
    return ToolResult(value=results, units="kg.m^2, N.m, rad/s^2, m, m/s", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="I_rod=mL^2/12; L=Iw; tau=rF; alpha=tau/I; a_roll=g sinth/(1+I/mr^2)",
                      inputs={"mass_kg": mass_kg, "radius_m": radius_m,
                              "angle_deg": angle_deg,
                              "angular_velocity_rad_s": angular_velocity_rad_s}).to_dict()


def atomic_angular_momentum(total_j: float = 1.5,
                            spin_orbit_constant_ev: float = 0.05) -> dict[str, Any]:
    """Atomic angular momentum: |J| magnitude and allowed m_j values, plus
    spin-orbit coupling energy/splitting and the Lande interval check for a
    term (L=2, S=1/2). e.g. atomic_angular_momentum(1.5)."""
    from sigma_ground.field.interface import angular_momentum as AM
    L, S = 2, 0.5
    results = {
        "magnitude_J_s": _safe(AM.angular_momentum_magnitude, total_j),
        "m_j_values": _safe(AM.angular_momentum_z_values, total_j),
        "spin_orbit_energy_eV": _safe(AM.spin_orbit_energy_eV, spin_orbit_constant_ev, L, S, total_j),
        "spin_orbit_splitting_eV": _safe(AM.spin_orbit_splitting_eV, spin_orbit_constant_ev, L, S),
        "lande_interval_check": _safe(AM.lande_interval_check, spin_orbit_constant_ev, L, S),
        "spin_expectation_z": _safe(AM.spin_expectation, 0.6, 0.8),
    }
    return ToolResult(value=results, units="J.s / eV", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="|J|=hbar sqrt(j(j+1)); E_SO=(A/2)[J(J+1)-L(L+1)-S(S+1)]",
                      inputs={"total_j": total_j,
                              "spin_orbit_constant_ev": spin_orbit_constant_ev}).to_dict()
