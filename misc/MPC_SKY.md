# The whole sky against our frozen predictions

Generated 2026-11-10T09:00:05+00:00. Nights: 2026-11-04 .. 2026-11-09 (6). Bodies 4,357; observations 1,187,679, of which 29,260 are forward predictions (observed after the start was frozen).

Every prediction runs from the MPC elements of the day the body was first seen, never refreshed; a residual at lead <= 0 is the fit's own residual (the baseline), a residual at lead > 0 is ours ahead of the sky.

| class | bodies | observations | median arcsec (all) | median arcsec (forward) | mean dRA cos dec | mean dDec |
|---|---|---|---|---|---|---|
| mba | 3,809 | 1,123,097 | 0.183 | 0.187 | -0.092 | +0.004 |
| nea | 547 | 64,534 | 0.239 | 0.388 | +40.082 | +14.634 |
| outer | 1 | 48 | 0.461 | 0.716 | +0.525 | -0.002 |

## Residual against lead (all classes)

| lead | observations | median arcsec | 90th percentile |
|---|---|---|---|
| fit (lead <= 0) | 1,158,419 | 0.185 | 0.572 |
| 0-30 d | 29,260 | 0.200 | 0.689 |

Source files: `D:/datasets/mpc_sky/residuals/<night>.parquet`, `sky_scoreboard.parquet`; sealed forecasts in `seals/` with their hashes in `fixtures/mpc_live/sky_seal_log.jsonl`.
