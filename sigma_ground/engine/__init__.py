"""The game engine layer — tier 3 (peer of materia/radiance/labs).

Wraps the existing physics backends (dynamics.stepper, dynamics.continuum,
field.interface.nbody, radiance.push_off.StackSim) behind an ECS and a
real-time Region scheduler. Never re-implements physics: every Region
adapter is a thin binding onto a backend that already exists and stays the
authoritative implementation. See the game-engine plan for the full
architecture and phased build order.
"""
