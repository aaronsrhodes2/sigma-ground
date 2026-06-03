"""Compatibility shim — CSG composition now lives in ``sigma_ground.kernel.csg``.

``from sigma_ground.csg import ...`` keeps working (re-exported below), but new
code should import from ``sigma_ground.kernel``. Edit ``kernel/csg.py``, not this
file.
"""
from sigma_ground.kernel.csg import *  # noqa: F401,F403
