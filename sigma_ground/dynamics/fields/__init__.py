"""dynamics.fields — physics LAWS that sweep a voxel material field.

Cell-based field solvers (heat first; momentum/fluid later) that exchange flux
between neighbouring cells each timestep. These are the math only — the per-cell
material properties (diffusivity, density, …) are supplied by the orchestrator
(Materia/Deckard, which may import the field physics); the law here stays
import-clean (stdlib + lazy numpy/scipy), so `dynamics` never reaches up a tier.

DECREE: a field solver sweeps the WHOLE grid, independent of the camera; the
renderer traces the frozen result afterward (two passes, one substrate).
"""
