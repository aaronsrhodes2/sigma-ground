# The whole sky against our frozen predictions

Generated 2026-11-11T02:35:41+00:00. Nights: 2026-11-04 .. 2026-11-10 (7). Bodies 4,927; observations 1,405,089, of which 39,030 are forward predictions (observed after the start was frozen).

Every prediction runs from the MPC elements of the day the body was first seen, never refreshed; a residual at lead <= 0 is the fit's own residual (the baseline), a residual at lead > 0 is ours ahead of the sky.

| class | bodies | observations | median arcsec (all) | median arcsec (forward) | mean dRA cos dec | mean dDec |
|---|---|---|---|---|---|---|
| mba | 4,324 | 1,336,045 | 0.184 | 0.195 | -0.082 | +0.010 |
| nea | 602 | 68,996 | 0.241 | 0.388 | +54.080 | +20.933 |
| outer | 1 | 48 | 0.461 | 0.716 | +0.525 | -0.002 |

## Residual against lead (all classes)

| lead | observations | median arcsec | 90th percentile |
|---|---|---|---|
| fit (lead <= 0) | 1,366,059 | 0.186 | 0.573 |
| 0-30 d | 39,030 | 0.206 | 0.700 |

Source files: `D:/datasets/mpc_sky/residuals/<night>.parquet`, `sky_scoreboard.parquet`; sealed forecasts in `seals/` with their hashes in `fixtures/mpc_live/sky_seal_log.jsonl`.
