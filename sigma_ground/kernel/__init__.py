"""sigma_ground.kernel — the shared tier-0 substrate every Mentat service builds on.

Canonical home for the geometry primitives (``shapes``, ``csg``, ``parts``) and
the ``Vec3`` math primitive. The legacy top-level paths
(``sigma_ground.shapes``, ``sigma_ground.csg``, ``sigma_ground.parts``,
``sigma_ground.dynamics.vec``) remain as thin re-export shims for backward
compatibility — new code should import from here.

Pure stdlib; nothing here imports above tier 0 (the layering guard enforces it).
"""
from .vec import *     # noqa: F401,F403
from .shapes import *  # noqa: F401,F403
from .csg import *     # noqa: F401,F403
from .parts import *   # noqa: F401,F403
from .voxel import Voxel  # noqa: F401  (voxel SDF shape; indexes numpy, never imports it)
