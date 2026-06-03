"""Materia — the combined physics of matter and energy.

Resurrected name (Captain Aaron Rhodes, 2026-06). Materia is the capstone
layer that *combines* the two halves sigma-ground already owns:

  MATTER  — composition → mass, density, geometry
            (sigma_ground.inventory / field.interface.surface)
  ENERGY  — the forces acting on that matter: gravity, aerodynamic drag,
            buoyancy, heat (sigma_ground.field / sigma_ground.labs.forces)

…and evolves the *pair* through time with the symplectic leapfrog integrator
(sigma_ground.dynamics.stepper), then hands back a worked, provenance-tagged,
self-validated answer — the same "show your work + cite the source + flag if
unsure" discipline the Q&A procedures layer uses, but for time-evolution.

Materia does not re-implement physics. Every number it produces traces to a
library function. Its job is to (1) assemble matter + energy into a scenario,
(2) run the existing engine, (3) extract the physically meaningful answer, and
(4) cross-check it against a closed-form result so it can never be silently
wrong.

Layer position:
    quarksum / inventory   (matter)        ─┐
    field / labs / dynamics (energy+motion) ─┼─► Materia ─► MatterShaper (render)
                                            ─┘   (this)
"""
from __future__ import annotations

from .engine import (
    simulate_fall,
    analytic_terminal_velocity,
    drag_heating_temperature_rise,
    drag_coefficient_mach,
    simulate_drag_run,
    MateriaStep,
    MateriaResult,
)
from .scenarios import (terminal_velocity_drop, drag_heating_drop,
                        high_altitude_descent, supersonic_projectile,
                        vertical_launch, orbital_velocity, material_profile,
                        structural_response, thermal_response,
                        rotational_dynamics, material_full_profile, SCENARIOS)
from .spec import SimulationSpec, SpecStep, run_spec, synthesize_chain
from .manifest import VERB_MANIFEST
from .translator import translate, answer

__all__ = [
    "simulate_fall",
    "analytic_terminal_velocity",
    "drag_heating_temperature_rise",
    "drag_coefficient_mach",
    "simulate_drag_run",
    "MateriaStep",
    "MateriaResult",
    "terminal_velocity_drop",
    "drag_heating_drop",
    "high_altitude_descent",
    "supersonic_projectile",
    "vertical_launch",
    "orbital_velocity",
    "material_profile",
    "structural_response",
    "thermal_response",
    "rotational_dynamics",
    "material_full_profile",
    "SCENARIOS",
    "SimulationSpec",
    "SpecStep",
    "run_spec",
    "synthesize_chain",
    "VERB_MANIFEST",
    "translate",
    "answer",
]
