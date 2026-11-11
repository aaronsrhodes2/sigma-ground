"""The matter record, for SkippyCraft: a thin door onto ``sigma_ground.field.interface.matter_record``, where the record is built.

Nothing is computed here. The record (invariant / state as of t / look, with sigma-ground's optics as tables for a clear material) is
sigma-ground's, portable to any renderer; the SkippyCraft mod (stage/skippycraft/mod) draws from it and carries no physics.
"""
from sigma_ground.field.interface.matter_record import DEPTH_CELLS, LIGHT_LEVELS, SCHEMA, matter_record as record, to_wire  # noqa: F401

MAX_LIGHT = LIGHT_LEVELS
