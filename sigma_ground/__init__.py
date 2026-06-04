"""
sigma_ground — the Mentat physics + rendering stack
===================================================

Pure Python, zero external dependencies. "Mentat" is the umbrella brand (the
MCP server is its public face); the import root stays ``sigma_ground``. Service
roles (see ARCHITECTURE.md for the tier contract):

  sigma_ground.kernel     — shared geometry/math primitives (shapes, csg, Vec3).
  sigma_ground.field      — σ-field scalar physics + authoritative constants.
  sigma_ground.inventory  — particle inventory & mass closure (Quarksum).
  sigma_ground.dynamics   — N-body, SPH fluid, Barnes-Hut, integrators.
  sigma_ground.deckard    — matter compiler: a name → a validated Construct.
  sigma_ground.materia    — physics / movement engine.
  sigma_ground.radiance   — renderer (SDF ray-march + entangler push renderer).
  sigma_ground.mcp        — the Mentat MCP face over every service.

Quick start::

    # Particle inventory
    from sigma_ground.inventory import stoq, build_quick_structure
    s = build_quick_structure("Iron", 1.0)
    result = stoq(s)

    # physical constants
    from sigma_ground.field.constants import HBAR, C

    # N-body dynamics
    from sigma_ground.dynamics.scene import PhysicsScene

Physics / rendering boundary
----------------------------
Physics never imports renderers; renderers may import physics. Rendering now
lives in ``sigma_ground.radiance`` (folded in from the former matter-shaper repo).

Author: Aaron Rhodes
"""

from .constants import (
    G, C, HBAR, L_PLANCK,
    K_B, ALPHA, MU_0, M_ELECTRON_KG,
    M_SUN_KG, L_SUN_W, AU_M, YEAR_S,
)
from .dynamics.vec import Vec3
from .shapes import (
    Shape, Sphere, HollowSphere, Cylinder, Box, Ellipsoid, Cone,
    Torus, Ring, Plane, Structure,
    sphere, cylinder, box, ellipsoid, cone, torus, plane,
    mass_from_shape, moment_of_inertia,
)
from .parts import (
    list_parts, convert_to_primitives, sample_boundary,
    shape_budget_from_source,
    pipe_structure, bolt_structure, beam_structure,
    PIPES, ISO_BOLTS, W_BEAMS,
)
from .csg import (
    ComposedSDF,
    sdf_union, sdf_subtract, sdf_intersect, sdf_smooth_union,
)

__all__ = [
    # Fundamental constants
    'G', 'C', 'HBAR', 'L_PLANCK',
    'K_B', 'ALPHA', 'MU_0', 'M_ELECTRON_KG',
    # Astronomical
    'M_SUN_KG', 'L_SUN_W', 'AU_M', 'YEAR_S',
    # Math primitives
    'Vec3',
    # Geometric shapes — structure = material + shape
    'Shape', 'Sphere', 'HollowSphere', 'Cylinder', 'Box',
    'Ellipsoid', 'Cone', 'Torus', 'Ring', 'Plane', 'Structure',
    'sphere', 'cylinder', 'box', 'ellipsoid', 'cone', 'torus', 'plane',
    'mass_from_shape', 'moment_of_inertia',
    # Standard parts catalog
    'list_parts', 'convert_to_primitives', 'sample_boundary',
    'shape_budget_from_source',
    'pipe_structure', 'bolt_structure', 'beam_structure',
    'PIPES', 'ISO_BOLTS', 'W_BEAMS',
    # CSG composition (A3DVS)
    'ComposedSDF',
    'sdf_union', 'sdf_subtract', 'sdf_intersect', 'sdf_smooth_union',
]
