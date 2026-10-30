"""The step rule of the N-body engine: how many steps per orbit a given integration step gives each body (misc/CANDIDATE_LEDGER.md C14 and C19).

Kept out of nbody.py on purpose: nbody.py is part of the physics fingerprint (reduced_model.FINGERPRINT_CODE), and this reads the bodies but does not change the integration."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..constants import G as _G

STEPS_PER_ORBIT_TARGET = 300   # C14: the step rule, shortest period / 300 (Phobos at 150 and 300 steps per orbit differ by 12 %)


def step_adequacy(bodies: Sequence, dt_s: float, target: float = STEPS_PER_ORBIT_TARGET, names: Sequence[str] | None = None) -> list[dict]:
    """Steps per orbit of every body against the step `dt_s`, so a caller can see which bodies this step cannot integrate.

    `bodies` are nbody.CelestialBody (mass_kg, position_m). A body's orbit is taken round the more massive body that gives it the SHORTEST period (the Moon about the Earth, not the
    Sun, although the Sun pulls harder); that is the period the step has to resolve. Returns one row per body that has a more massive partner, shortest period first:
    {"body", "index", "partner", "period_s", "steps_per_orbit", "adequate"} (adequate = steps_per_orbit >= target); `names` labels the rows.

    Measured on the real Jupiter system (C19, Forest-Ruth, +5 yr against DE440): Io at 71 steps per orbit is 118,000 km off, at 142 steps 7,610 km (a factor 15.5 for a halved
    step, convergence order 3.95); Europa and Ganymede the same. This function reports the number; it does not change the integration.
    """
    rows = []
    for i, b in enumerate(bodies):
        best = None
        for j, c in enumerate(bodies):
            if j == i or c.mass_kg <= b.mass_kg:
                continue
            r = float(np.linalg.norm(np.asarray(b.position_m, float) - np.asarray(c.position_m, float)))
            if r <= 0.0:
                continue
            p = 2.0 * math.pi * math.sqrt(r ** 3 / (_G * (b.mass_kg + c.mass_kg)))
            if best is None or p < best[0]:
                best = (p, j)
        if best is not None:
            spo = best[0] / dt_s
            rows.append({"body": names[i] if names else f"body[{i}]", "index": i, "partner": best[1], "period_s": best[0], "steps_per_orbit": spo, "adequate": spo >= target})
    return sorted(rows, key=lambda r: r["period_s"])
