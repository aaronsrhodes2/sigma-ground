"""Compatibility shim — geometry primitives now live in ``sigma_ground.kernel.shapes``.

``from sigma_ground.shapes import ...`` keeps working (re-exported below), but
new code should import from ``sigma_ground.kernel``. Edit ``kernel/shapes.py``,
not this file.
"""
from sigma_ground.kernel.shapes import *  # noqa: F401,F403
