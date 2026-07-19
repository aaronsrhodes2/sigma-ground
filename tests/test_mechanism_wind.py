"""Wind-rotor gates — the natural-drive model (nothing plugged).

Closed forms (derived in mechanisms/wind.py's docstring, re-derived here):
  - at rest, per-blade normal flow u_n = U*cos(beta); summed axis torque
        tau_x(0) = N * R_c * (1/2*rho*C_N*A) * U^2 * cos^2(beta) * sin(beta)
    (the tangential components of blade force cancel pairwise by symmetry);
  - zero pitch => zero axis torque: a face-square blade's force passes
    parallel to the axis — geometry, not scripting;
  - the rotor self-limits where blade tangential motion cancels normal
    inflow: omega* = U * cot(beta) / R_c — approached from below, never
    overshot (torque reverses sign above it).
"""
import math

import pytest

from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.parcel import PhysicsParcel
from sigma_ground.dynamics.scene import PhysicsScene
from sigma_ground.dynamics.stepper import step
from sigma_ground.dynamics.joints import RevoluteJoint
from sigma_ground.dynamics.mechanisms.wind import (RotorWind, build_rotor_blades,
                                                   terminal_omega,
                                                   RHO_AIR_SEA_LEVEL,
                                                   C_N_FLAT_PLATE)
from sigma_ground.dynamics.mechanisms.bearing import RigidBearing


class _Mat:
    density_kg_m3 = 1000.0
    restitution = 0.5

    def density_at_sigma(self, s):
        return 1000.0


def _rotor(I=0.05):
    return PhysicsParcel(0.05, _Mat(), position=Vec3(0, 0, 0), mass=1.0,
                         inertia_body=(I, I, I))


def _tau0_closed_form(n, r_c, area, U, beta):
    k = 0.5 * RHO_AIR_SEA_LEVEL * C_N_FLAT_PLATE * area
    return n * r_c * k * U * U * math.cos(beta) ** 2 * math.sin(beta)


def test_static_rotor_torque_matches_closed_form_and_scales_with_wind_squared():
    beta = math.radians(30.0)
    blades, area = build_rotor_blades(4, 0.6, 0.5, 0.15, beta)
    rotor = _rotor()
    for U in (5.0, 10.0):
        wind = RotorWind(rotor, blades, Vec3(U, 0, 0))
        F, tau = wind(rotor)
        assert tau.x == pytest.approx(_tau0_closed_form(4, 0.6, area, U, beta),
                                      rel=1e-9)
        # tangential force components cancel by 4-blade symmetry: the net
        # force is purely axial (downwind), absorbed by the pivot
        assert abs(F.y) < 1e-12 and abs(F.z) < 1e-12
        assert F.x > 0.0
    # tau(10) / tau(5) == 4: quadratic in wind speed
    t5 = RotorWind(rotor, blades, Vec3(5, 0, 0))(rotor)[1].x
    t10 = RotorWind(rotor, blades, Vec3(10, 0, 0))(rotor)[1].x
    assert t10 / t5 == pytest.approx(4.0, rel=1e-9)


def test_zero_pitch_yields_zero_axis_torque():
    blades, _ = build_rotor_blades(4, 0.6, 0.5, 0.15, 0.0)
    rotor = _rotor()
    _, tau = RotorWind(rotor, blades, Vec3(10, 0, 0))(rotor)
    assert abs(tau.x) < 1e-12


def test_pitch_sign_flips_spin_direction():
    rotor = _rotor()
    for sign in (+1.0, -1.0):
        blades, _ = build_rotor_blades(4, 0.6, 0.5, 0.15,
                                       sign * math.radians(30.0))
        _, tau = RotorWind(rotor, blades, Vec3(10, 0, 0))(rotor)
        assert math.copysign(1.0, tau.x) == sign


def test_spinup_approaches_terminal_tip_speed_from_below():
    """The emergent behavior: monotone spin-up that self-limits at the
    closed-form omega* — the physics finds the equilibrium, nothing
    prescribes the final speed. Mounted on a RigidBearing (see the xfail
    below for why not a RevoluteJoint), whose absorbed-energy ledger must
    stay negligible against the spin energy."""
    U, beta, r_c = 10.0, math.radians(40.0), 0.6
    blades, _ = build_rotor_blades(4, r_c, 0.5, 0.15, beta)
    rotor = _rotor(I=0.05)
    bearing = RigidBearing(rotor, rotor.position, Vec3(1, 0, 0))
    scene = PhysicsScene([rotor], gravity=Vec3(0, 0, 0), ground=False)
    wind = RotorWind(rotor, blades, Vec3(U, 0, 0))
    w_star = terminal_omega(U, beta, r_c)

    dt = 1.0 / 960.0
    prev = 0.0
    for k in range(int(5.0 / dt)):
        step(scene, dt=dt, sub_steps=1, external_forces=wind)
        bearing.project()
        w = bearing.omega()
        assert w >= prev - 1e-9, "spin-up must be monotone"
        assert w <= w_star * 1.01, "must never overshoot the equilibrium"
        prev = w
    assert prev == pytest.approx(w_star, rel=0.05)
    # the ideal bearing's projection is workless to second order: what it
    # absorbed must be negligible against the rotor's spin energy
    ke_spin = 0.5 * 0.05 * prev ** 2
    assert bearing.absorbed_energy_j < 1e-6 * ke_spin


@pytest.mark.xfail(reason="PHYSICS_GAP: RevoluteJoint's swing-correction "
                   "loop (SHAKE/RATTLE) is exponentially unstable under "
                   "orientation-coupled EXTERNAL torque (wind responding to "
                   "axis tilt) — transverse error grows ~x2.5/step from float "
                   "noise and the rotor tumbles, at every dt (1/960..1/3840) "
                   "and iteration count (10..30) tried. The identical aero "
                   "model on an ideal projection is cleanly stable (the test "
                   "above), isolating the gap to the joint loop. See "
                   "misc/KNOWN_GAPS.md; fixing the solver un-xfails this.",
                   strict=True)
def test_wind_rotor_on_revolute_joint_stays_swing_stable():
    U, beta, r_c = 10.0, math.radians(40.0), 0.6
    blades, _ = build_rotor_blades(4, r_c, 0.5, 0.15, beta)
    rotor = _rotor(I=0.05)
    joint = RevoluteJoint(rotor, None, rotor.position, Vec3(1, 0, 0))
    scene = PhysicsScene([rotor], gravity=Vec3(0, 0, 0), ground=False,
                         constraints=[joint])
    wind = RotorWind(rotor, blades, Vec3(U, 0, 0))
    dt = 1.0 / 960.0
    for k in range(int(3.0 / dt)):
        step(scene, dt=dt, sub_steps=1, external_forces=wind)
        w = rotor.angular_velocity
        assert math.hypot(w.y, w.z) < 0.01 * max(1.0, abs(w.x)), \
            "rotor tumbled off its hinge axis"
