# Orbit fit of 433 from observations alone (C28)

Generated 2026-11-03T04:19:37+00:00 by `scripts/orbit_fit.py`; data `D:/datasets/mpc_live/orbit_fit/433/`. Epoch JD 2458849.5 (TDB); fit window 2015-01-01 to 2019-12-31; prediction window from 2020-01-01.

## The fit

* Observations kept 1544 of 1563 (3-sigma rejection at 1.5 arcsec after the first pass); post-fit RMS **0.3692 arcsec**; converged: True; unit-weight variance 0.273 (sigma assumed 0.5 arcsec).

| component | fitted | JPL | fit minus JPL | formal sigma | scaled sigma | difference / scaled sigma |
|---|---|---|---|---|---|---|
| x km | 161271673.056750 | 161271684.952737 | -11.895988 | 44.895046 | 23.462093 | -0.51 |
| y km | -194363117.895118 | -194363097.475571 | -20.419547 | 34.306113 | 17.928331 | -1.14 |
| z km | -81267542.900819 | -81267565.374852 | 22.474033 | 26.330704 | 13.760392 | 1.63 |
| vx km/day | 1301614.356109 | 1301614.320912 | 0.035196 | 0.214807 | 0.112258 | 0.31 |
| vy km/day | 832264.211896 | 832264.367582 | -0.155686 | 0.274203 | 0.143298 | -1.09 |
| vz km/day | 708144.650583 | 708144.514039 | 0.136544 | 0.140652 | 0.073505 | 1.86 |

Position difference 32.61 km, speed difference 2.43 mm/s.

| element | fitted | JPL (same epoch, from its state) | difference |
|---|---|---|---|
| a_au | 1.45818052 | 1.45818051 | 1.60e-08 |
| e | 0.22284892 | 0.22284893 | -1.11e-08 |
| i_deg | 30.79301635 | 30.79301517 | 1.18e-06 |
| om_deg | 342.34951934 | 342.34953737 | -1.80e-05 |
| w_deg | 138.93643122 | 138.93641435 | 1.69e-05 |
| ma_deg | 186.55636098 | 186.55636643 | -5.45e-06 |

## The three curves on the prediction window

4435 observations, 0 dropped by the 5-arcsec rule on (c). Residuals in arcsec: median of the total residual (robust sigma); the last two columns are the per-observation differences.

| year | n | lead yr | (a) fitted | (b) from JPL's state | (c) JPL | (a)-(c) median | (b)-(c) median |
|---|---|---|---|---|---|---|---|
| 2020 | 240 | 0.73 | 0.3924 (0.323) | 0.3890 (0.323) | 0.3890 (0.323) | 0.0253 | 0.0000 |
| 2021 | 294 | 1.62 | 0.1182 (0.068) | 0.1005 (0.065) | 0.1005 (0.065) | 0.0403 | 0.0000 |
| 2023 | 955 | 3.74 | 0.1391 (0.089) | 0.1267 (0.087) | 0.1266 (0.087) | 0.0640 | 0.0002 |
| 2024 | 207 | 4.06 | 0.1902 (0.145) | 0.1730 (0.130) | 0.1729 (0.130) | 0.0403 | 0.0002 |
| 2025 | 1579 | 5.71 | 0.1945 (0.131) | 0.1234 (0.079) | 0.1237 (0.079) | 0.1776 | 0.0013 |
| 2026 | 1160 | 6.26 | 0.2199 (0.107) | 0.1375 (0.098) | 0.1380 (0.098) | 0.1623 | 0.0014 |

All years: (a) 0.1868, (b) 0.1319, (c) 0.1319; (a)-(c) 0.1182, (b)-(c) 0.0008 arcsec (medians).
