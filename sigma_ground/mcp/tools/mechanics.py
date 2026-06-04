"""Mechanics analysis tools (standard physics).

Each tool takes a few realistic inputs and runs a full standard-mechanics
cascade, so one natural question exercises many underlying
field.interface.{classical_mechanics, projectile} functions at once.
All angles are accepted in DEGREES and converted internally.
"""
from __future__ import annotations

import math
from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_SRC = "sigma_ground.field.interface (classical_mechanics, projectile)"


def _decline(notes: str, **inp) -> dict[str, Any]:
    return ToolResult(value=None, source=_SRC, provenance_tag="SPECULATIVE-PENDING",
                      notes=notes, inputs=inp).to_dict()


def collision_analysis(mass1_kg: float, velocity1_m_s: float,
                       mass2_kg: float, velocity2_m_s: float,
                       restitution: float = 0.8) -> dict[str, Any]:
    """1D two-body collision: elastic final velocities, the inelastic outcome
    for a coefficient of restitution, and the kinetic energy dissipated.
    e.g. collision_analysis(2, 3, 1, -1)."""
    from sigma_ground.field.interface import classical_mechanics as cm
    try:
        v1e, v2e = cm.elastic_collision_velocities(mass1_kg, velocity1_m_s,
                                                   mass2_kg, velocity2_m_s)
        inel = cm.inelastic_collision_velocity(mass1_kg, velocity1_m_s, mass2_kg,
                                               velocity2_m_s, cor=float(restitution))
        eloss = cm.collision_energy_loss(mass1_kg, velocity1_m_s, mass2_kg,
                                         velocity2_m_s, float(restitution))
    except Exception as e:
        return _decline(f"collision_analysis failed: {e}",
                        mass1_kg=mass1_kg, mass2_kg=mass2_kg)
    return ToolResult(
        value={"elastic_v1_m_s": v1e, "elastic_v2_m_s": v2e,
               "inelastic_outcome_m_s": inel, "energy_lost_J": eloss},
        units="m/s, J", source=_SRC, provenance_tag="DERIVED",
        formula="1D momentum+energy conservation; dKE for restitution e",
        inputs={"mass1_kg": mass1_kg, "velocity1_m_s": velocity1_m_s,
                "mass2_kg": mass2_kg, "velocity2_m_s": velocity2_m_s,
                "restitution": restitution}).to_dict()


def work_energy_analysis(mass_kg: float, velocity_m_s: float,
                         height_m: float = 10.0, force_n: float = 20.0,
                         distance_m: float = 5.0) -> dict[str, Any]:
    """Work-energy bundle: work done, mechanical power, friction loss,
    gravitational PE, rotational KE, impulse, post-impulse velocity, and
    total mechanical energy of a moving body."""
    from sigma_ground.field.interface import classical_mechanics as cm
    try:
        impulse = cm.impulse(force_n, 2.0)
        results = {
            "work_done_J": cm.work_done(force_n, distance_m),
            "power_W": cm.power_mechanical(force_n, velocity_m_s),
            "friction_loss_J": cm.friction_dissipation(0.3 * force_n, distance_m),
            "gravitational_pe_J": cm.gravitational_pe(mass_kg, height_m),
            "rotational_ke_J": cm.rotational_ke(0.5 * mass_kg, velocity_m_s),
            "impulse_Ns": impulse,
            "velocity_after_impulse_m_s": cm.velocity_from_impulse(impulse, mass_kg),
            "total_mechanical_energy_J": cm.total_mechanical_energy(mass_kg, velocity_m_s, height_m),
        }
    except Exception as e:
        return _decline(f"work_energy_analysis failed: {e}", mass_kg=mass_kg)
    return ToolResult(value=results, units="J / W / N*s / (m/s)",
                      source=_SRC, provenance_tag="DERIVED",
                      formula="W=Fd; P=Fv; U=mgh; KE_rot=0.5 I w^2; J=F dt; E=KE+PE",
                      inputs={"mass_kg": mass_kg, "velocity_m_s": velocity_m_s,
                              "height_m": height_m, "force_n": force_n,
                              "distance_m": distance_m}).to_dict()


def projectile_analysis(speed_m_s: float, angle_deg: float, mass_kg: float = 1.0,
                        drag_coefficient: float = 0.47,
                        area_m2: float = 0.01) -> dict[str, Any]:
    """Full projectile analysis from a launch speed + angle (degrees): vacuum
    range, apex height and flight time, plus drag force at launch, terminal
    velocity, and the drag-corrected range. e.g. projectile_analysis(50, 45)."""
    from sigma_ground.field.interface import projectile as pj
    try:
        ang = math.radians(angle_deg)
        traj = pj.projectile_trajectory(speed_m_s, ang)
        drag_range = pj.projectile_with_drag(speed_m_s, ang, mass_kg,
                                             drag_coefficient, area_m2)
        results = {
            "range_m": pj.projectile_range(speed_m_s, ang),
            "max_height_m": pj.projectile_max_height(speed_m_s, ang),
            "time_of_flight_s": pj.projectile_time_of_flight(speed_m_s, ang),
            "drag_force_at_launch_N": pj.drag_force(speed_m_s, drag_coefficient, area_m2),
            "terminal_velocity_m_s": pj.terminal_velocity(mass_kg, drag_coefficient, area_m2),
            "trajectory_points": len(traj) if hasattr(traj, "__len__") else None,
        }
        if isinstance(drag_range, (int, float)):
            results["drag_corrected_range_m"] = drag_range
    except Exception as e:
        return _decline(f"projectile_analysis failed: {e}",
                        speed_m_s=speed_m_s, angle_deg=angle_deg)
    return ToolResult(value=results, units="m, s, N, m/s",
                      source=_SRC, provenance_tag="DERIVED",
                      formula="R=v^2 sin2th/g; H=v^2 sin^2th/2g; T=2v sinth/g; + quadratic drag",
                      inputs={"speed_m_s": speed_m_s, "angle_deg": angle_deg,
                              "mass_kg": mass_kg, "drag_coefficient": drag_coefficient,
                              "area_m2": area_m2}).to_dict()


def incline_analysis(angle_deg: float, friction_coefficient: float = 0.3,
                     speed_m_s: float = 10.0, height_m: float = 5.0) -> dict[str, Any]:
    """Incline mechanics: the critical angle at which sliding begins, the
    distance a body slides up the incline before stopping, and the slide
    speed reached at the bottom."""
    from sigma_ground.field.interface import projectile as pj
    try:
        ang = math.radians(angle_deg)
        results = {
            "critical_sliding_angle_deg": math.degrees(pj.incline_critical_angle(friction_coefficient)),
            "sliding_distance_up_m": pj.incline_sliding_distance(speed_m_s, ang, friction_coefficient),
            "speed_at_bottom_m_s": pj.incline_speed_at_bottom(height_m, ang, friction_coefficient),
        }
    except Exception as e:
        return _decline(f"incline_analysis failed: {e}", angle_deg=angle_deg)
    return ToolResult(value=results, units="deg, m, m/s",
                      source=_SRC, provenance_tag="DERIVED",
                      formula="th_c=arctan(mu); d=v^2/(2g(sin th+mu cos th)); v=sqrt(2gh(1-mu/tan th))",
                      inputs={"angle_deg": angle_deg, "friction_coefficient": friction_coefficient,
                              "speed_m_s": speed_m_s, "height_m": height_m}).to_dict()
