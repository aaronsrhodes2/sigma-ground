"""Compatibility shim — labs now lives in ``sigma_ground.materia.labs``.

labs is Materia's private simulation toolbox (scene/environment/runner/forces).
``from sigma_ground.labs import ...`` keeps working (re-exported below), but new
code should import from ``sigma_ground.materia.labs``.
"""
from sigma_ground.materia.labs import *  # noqa: F401,F403
