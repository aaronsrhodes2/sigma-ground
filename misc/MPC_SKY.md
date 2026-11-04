# The whole sky against our frozen predictions

Generated 2026-11-04T20:23:22+00:00. Nights: 2026-11-04 .. 2026-11-04 (1). Bodies 53; observations 23,002, of which 147 are forward predictions (observed after the start was frozen).

Every prediction runs from the MPC elements of the day the body was first seen, never refreshed; a residual at lead <= 0 is the fit's own residual (the baseline), a residual at lead > 0 is ours ahead of the sky.

| class | bodies | observations | median arcsec (all) | median arcsec (forward) | mean dRA cos dec | mean dDec |
|---|---|---|---|---|---|---|
| nea | 53 | 23,002 | 0.208 | 0.199 | -0.002 | +0.021 |

## Residual against lead (all classes)

| lead | observations | median arcsec | 90th percentile |
|---|---|---|---|
| fit (lead <= 0) | 22,855 | 0.208 | 0.619 |
| 0-30 d | 147 | 0.199 | 0.514 |

Source files: `D:/datasets/mpc_sky/residuals/<night>.parquet`, `sky_scoreboard.parquet`; sealed forecasts in `seals/` with their hashes in `fixtures/mpc_live/sky_seal_log.jsonl`.
