"""Compatibility shim — the parts catalog now lives in ``sigma_ground.kernel.parts``.

``from sigma_ground.parts import ...`` keeps working (re-exported below), but new
code should import from ``sigma_ground.kernel``. Edit ``kernel/parts.py``, not
this file.
"""
from sigma_ground.kernel.parts import *  # noqa: F401,F403
