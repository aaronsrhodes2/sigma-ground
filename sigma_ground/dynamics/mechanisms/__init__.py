"""Mechanisms — clock/windmill orchestration built ENTIRELY from existing
dynamics/joints.py primitives (RevoluteJoint's motor + limits). No new
solver math lives here: every file in this package is bookkeeping and
Python-level orchestration over the ONE stepper (dynamics/stepper.py),
which is why these are gated separately, in isolation, against closed-form
solutions before ever touching a real gear train.
"""
from .spring import MainspringState, HUGE_SPEED
from .escapement import Escapement
from .plug import Plug
from .wind import (Blade, RotorWind, build_rotor_blades, terminal_omega,
                   RHO_AIR_SEA_LEVEL, C_N_FLAT_PLATE)
from .bearing import RigidBearing
from .bearing_gear_coupling import BearingGearCoupling
from .pump import ReciprocatingPumpState
from .choice import Choice
from .actuator import OscillatingRevoluteActuator

__all__ = ["MainspringState", "HUGE_SPEED", "Escapement", "Plug", "Choice",
           "Blade", "RotorWind", "build_rotor_blades", "terminal_omega",
           "RHO_AIR_SEA_LEVEL", "C_N_FLAT_PLATE", "RigidBearing",
           "BearingGearCoupling", "ReciprocatingPumpState",
           "OscillatingRevoluteActuator"]
