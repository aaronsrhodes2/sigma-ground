"""Plug doctrine gates — force provenance for artificial stand-ins.

The doctrine (2026-07-15): motion requires a mover; artificial stand-ins
for absent machinery must be flagged in-scene with their substituted
variables. A Plug is load-bearing (constructing it IS what configures the
motor), so the flag can never drift from the physics.
"""
import math

import pytest

from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.parcel import PhysicsParcel
from sigma_ground.dynamics.scene import PhysicsScene
from sigma_ground.dynamics.stepper import step
from sigma_ground.dynamics.joints import RevoluteJoint
from sigma_ground.dynamics.mechanisms.plug import Plug
from sigma_ground.dynamics.mechanisms.spring import HUGE_SPEED


class _Mat:
    density_kg_m3 = 1000.0
    restitution = 0.5

    def density_at_sigma(self, s):
        return 1000.0


def _disc():
    return PhysicsParcel(0.05, _Mat(), position=Vec3(0, 0, 0), mass=1.0)


def test_speed_plug_configures_the_motor_and_reports_variables():
    d = _disc()
    j = RevoluteJoint(d, None, d.position, Vec3(0, 0, 1))
    p = Plug(j, "artificial rotation from external source",
             motor_speed_rad_s=-3.0, motor_max_torque_nm=2.0,
             equivalent="a motor that is not modeled")
    assert j.motor_speed == -3.0
    assert j.motor_max_torque == 2.0
    d_dict = p.to_dict()
    assert d_dict["description"] == "artificial rotation from external source"
    assert d_dict["variables"]["motor_speed_rad_s"] == -3.0
    assert "equivalent" in d_dict["variables"]
    assert "PLUGGED" in p.cite()


def test_torque_plug_degenerates_to_pure_torque_source():
    d = _disc()
    j = RevoluteJoint(d, None, d.position, Vec3(0, 0, 1))
    Plug(j, "as if a 20hp engine were attached, without the engine",
         torque_nm=0.02)
    assert abs(j.motor_speed) == HUGE_SPEED
    assert j.motor_max_torque == 0.02
    scene = PhysicsScene([d], ground=False, constraints=[j])
    for _ in range(48):
        step(scene, dt=1.0 / 960.0, sub_steps=1)
    # constant-alpha spin-up, same closed form gated in test_mechanism_spring
    I = 0.4 * 1.0 * 0.05 ** 2
    assert d.angular_velocity.length() == pytest.approx(
        (0.02 / I) * 48 / 960.0, rel=1e-6)


def test_plug_requires_exactly_one_mode():
    d = _disc()
    j = RevoluteJoint(d, None, d.position, Vec3(0, 0, 1))
    with pytest.raises(ValueError):
        Plug(j, "x")                                   # neither
    with pytest.raises(ValueError):
        Plug(j, "x", motor_speed_rad_s=1.0, torque_nm=1.0)   # both


def test_support_plug_declares_artificial_holders():
    s = Plug.support("held as if by a frame, without the frame")
    d = s.to_dict()
    assert d["kind"] == "support"
    assert not d["adjustable"]                 # supports default non-adjustable
    assert "PLUGGED SUPPORT" in s.cite()


def test_scenes_declare_both_drive_and_support_plugs():
    """The extended doctrine: artificial MOVERS and artificial HOLDERS both
    get cited. A motor demo has one of each (nothing turns the disc, and
    nothing holds it up); the clock has NO drive plug (the mainspring is
    present, modeled machinery) but its arbors still hang on world anchors
    with no clock plates modeled — a support plug."""
    from sigma_ground.radiance.trajectory import (record_motor_spin,
                                                  record_clock)
    plugged = record_motor_spin(t_max=0.1)
    kinds = sorted(p["kind"] for p in plugged["scene"]["plugs"])
    assert kinds == ["drive", "support"]
    drive = next(p for p in plugged["scene"]["plugs"] if p["kind"] == "drive")
    assert "artificial rotation" in drive["description"]
    assert drive["adjustable"]                 # artificial variables: adjustable by default
    assert "PLUGGED" in plugged["scene"]["source"]

    natural = record_clock(t_max=0.5)
    kinds = [p["kind"] for p in natural["scene"]["plugs"]]
    assert kinds == ["support"]                # holders cited; no drive plug
    assert "plates" in natural["scene"]["plugs"][0]["description"]
