"""Compatibility shim — ``Vec3`` now lives in ``sigma_ground.kernel.vec``.

``from sigma_ground.dynamics.vec import Vec3`` keeps working (re-exported below),
but new code should import from ``sigma_ground.kernel``. Edit ``kernel/vec.py``,
not this file.
"""
from sigma_ground.kernel.vec import *  # noqa: F401,F403
