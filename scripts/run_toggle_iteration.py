"""Iteration loop: each Tier-A toggle individually + cumulative accumulations.

Generates 8 predictor variants (all at dt=0.1d, FR4) and runs them across
4 widely-spaced windows at 3y horizon. Saves to a custom JSON path so the
existing baseline/finedt data is not overwritten.

Predictor list:
  finedt          -- baseline (no Tier-A filters)
  finedt_tidal    -- + tidal_force                       (individual 1)
  finedt_j4       -- + j4_zonal                          (individual 2)
  finedt_j3       -- + j3_zonal                          (individual 3)
  finedt_gr2pn    -- + gr_2pn                            (individual 4)
  finedt_tidal_j4         -- + tidal + j4                (cumulative step 2)
  finedt_tidal_j4_j3      -- + tidal + j4 + j3           (cumulative step 3)
  finedt_all      -- + tidal + j4 + j3 + gr_2pn          (cumulative step 4)

Windows: j2005, j2010, j2015, j2020 (4, widely spaced)
Horizon: 3y
dt: 0.1d
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

from sigma_ground.field.interface.nbody import PhysicsToggles
from sigma_ground.field.interface.rolling_shootout import (
    Predictor, run_rolling_shootout, _RESULTS_FILE,
)


# Base toggle bundle for all variants in this iteration: the current "best"
# = finedt's physics (Newton + 1PN GR + SRP + J₂) at dt=0.1d.
_BASE_TOGGLES = PhysicsToggles(
    gr_1pn=True, srp=True, j2_zonal=True,
    # New Tier-A filters all default OFF; each variant flips one+.
)


# 8 predictor variants
PREDICTORS_ITERATION: list[Predictor] = [
    # Baseline -- finedt physics, no new filters
    Predictor(
        name="finedt",
        toggles=_BASE_TOGGLES,
        integrator="fr4", dt_days=0.1,
        description="Newton + 1PN GR + SRP + J2 (the morning baseline)",
    ),
    # Individual filter tests
    Predictor(
        name="finedt_tidal",
        toggles=replace(_BASE_TOGGLES, tidal_force=True),
        integrator="fr4", dt_days=0.1,
        description="+ tidal_force (OURS, mutual tidal bulge as effective J2)",
    ),
    Predictor(
        name="finedt_j4",
        toggles=replace(_BASE_TOGGLES, j4_zonal=True),
        integrator="fr4", dt_days=0.1,
        description="+ j4_zonal (BORROWED, gas-giant J4 zonal harmonic)",
    ),
    Predictor(
        name="finedt_j3",
        toggles=replace(_BASE_TOGGLES, j3_zonal=True),
        integrator="fr4", dt_days=0.1,
        description="+ j3_zonal (BORROWED, asymmetric pear-shape zonal)",
    ),
    Predictor(
        name="finedt_gr2pn",
        toggles=replace(_BASE_TOGGLES, gr_2pn=True),
        integrator="fr4", dt_days=0.1,
        description="+ gr_2pn (BORROWED, single-body 2PN Schwarzschild c^-4)",
    ),
    # Cumulative accumulations (in predicted impact order)
    Predictor(
        name="finedt_tidal_j4",
        toggles=replace(_BASE_TOGGLES, tidal_force=True, j4_zonal=True),
        integrator="fr4", dt_days=0.1,
        description="Cumulative: + tidal + j4",
    ),
    Predictor(
        name="finedt_tidal_j4_j3",
        toggles=replace(_BASE_TOGGLES,
                        tidal_force=True, j4_zonal=True, j3_zonal=True),
        integrator="fr4", dt_days=0.1,
        description="Cumulative: + tidal + j4 + j3",
    ),
    Predictor(
        name="finedt_all",
        toggles=replace(_BASE_TOGGLES,
                        tidal_force=True, j4_zonal=True,
                        j3_zonal=True, gr_2pn=True),
        integrator="fr4", dt_days=0.1,
        description="Cumulative: + tidal + j4 + j3 + gr_2pn (all Tier-A on)",
    ),
]


WINDOW_START_KEYS = ["j2005", "j2010", "j2015", "j2020"]
PREDICTION_HORIZON_YR = 3.0
SAMPLE_INTERVAL_YR    = 1.0

_OUTPUT_PATH = (
    Path(__file__).parent.parent
    / "sigma_ground" / "field" / "interface" / "fixtures"
    / "rolling_shootout_toggle_iteration.json"
)


def main() -> None:
    print(f"\n  TOGGLE ITERATION PASS")
    print(f"  Predictors:  {len(PREDICTORS_ITERATION)}")
    print(f"  Windows:     {WINDOW_START_KEYS}")
    print(f"  Horizon:     {PREDICTION_HORIZON_YR}y")
    print(f"  Total runs:  {len(PREDICTORS_ITERATION) * len(WINDOW_START_KEYS)}")
    print(f"  Output:      {_OUTPUT_PATH.name}")
    print()

    t0 = time.time()
    run_rolling_shootout(
        window_start_keys=WINDOW_START_KEYS,
        prediction_horizon_yr=PREDICTION_HORIZON_YR,
        sample_interval_yr=SAMPLE_INTERVAL_YR,
        predictors=PREDICTORS_ITERATION,
        verbose=True,
    )

    # run_rolling_shootout writes to _RESULTS_FILE; move it to our target.
    if _RESULTS_FILE.exists():
        shutil.move(str(_RESULTS_FILE), str(_OUTPUT_PATH))
        print(f"\n  Saved -> {_OUTPUT_PATH.name}")
    print(f"  Total elapsed: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
