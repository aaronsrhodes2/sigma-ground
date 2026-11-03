# fair contest v3 (restated_math_tide, 0.0125 d, against DE442): 146 propagations, predictor `restated_math_tide`, scored against DE442

Each run starts from a JPL ephemeris's OWN state at a yearly date after its release and is compared with that ephemeris's own continuation (JPL's prediction at the time) and with DE442 (reality). Median km over runs. `eT` is DE440 against DE442: the width of reality's own spread. `p` is the two-sided sign test of ours-closer-than-JPL.

| body | lead | ours vs reality | JPL-at-the-time vs reality | ours vs JPL (dynamics gap) | start offset | reality spread | ours closer | p | runs |
|---|---|---|---|---|---|---|---|---|---|
| Mercury | +1 yr | 1.273 | 1.167 | 0.060 | 1.15 | 0.004 | 48/146 | 0.000 | 146 |
| Mercury | +3 yr | 1.597 | 1.272 | 0.261 | 1.22 | 0.004 | 37/125 | 0.000 | 125 |
| Mercury | +5 yr | 1.840 | 1.322 | 0.425 | 1.19 | 0.005 | 27/102 | 0.000 | 102 |
| Venus | +1 yr | 0.553 | 0.534 | 0.063 | 0.51 | 0.008 | 89/146 | 0.010 | 146 |
| Venus | +3 yr | 0.587 | 0.573 | 0.218 | 0.56 | 0.008 | 71/125 | 0.152 | 125 |
| Venus | +5 yr | 0.707 | 0.680 | 0.374 | 0.67 | 0.008 | 51/102 | 1.000 | 102 |
| Earth | +1 yr | 0.207 | 0.208 | 0.031 | 0.20 | 0.011 | 76/146 | 0.679 | 146 |
| Earth | +3 yr | 0.230 | 0.225 | 0.134 | 0.20 | 0.011 | 72/125 | 0.107 | 125 |
| Earth | +5 yr | 0.246 | 0.266 | 0.323 | 0.22 | 0.012 | 58/102 | 0.198 | 102 |
| Moon | +1 yr | 0.219 | 0.209 | 0.138 | 0.20 | 0.011 | 44/146 | 0.000 | 146 |
| Moon | +3 yr | 0.486 | 0.226 | 0.465 | 0.20 | 0.011 | 33/125 | 0.000 | 125 |
| Moon | +5 yr | 1.007 | 0.265 | 0.741 | 0.22 | 0.011 | 22/102 | 0.000 | 102 |
| Mars | +1 yr | 0.464 | 0.450 | 0.045 | 0.42 | 0.018 | 76/146 | 0.679 | 146 |
| Mars | +3 yr | 0.486 | 0.520 | 0.141 | 0.42 | 0.018 | 63/125 | 1.000 | 125 |
| Mars | +5 yr | 0.524 | 0.604 | 0.268 | 0.42 | 0.018 | 50/102 | 0.921 | 102 |
| Jupiter | +1 yr | 47.548 | 47.546 | 0.034 | 45.43 | 10.091 | 36/146 | 0.000 | 146 |
| Jupiter | +3 yr | 52.720 | 52.572 | 0.343 | 46.02 | 10.396 | 24/125 | 0.000 | 125 |
| Jupiter | +5 yr | 54.431 | 54.364 | 1.235 | 53.39 | 10.458 | 17/102 | 0.000 | 102 |
| Saturn | +1 yr | 88.921 | 88.929 | 0.018 | 88.96 | 0.113 | 68/146 | 0.456 | 146 |
| Saturn | +3 yr | 143.219 | 143.146 | 0.151 | 143.12 | 0.114 | 57/125 | 0.371 | 125 |
| Saturn | +5 yr | 177.867 | 178.005 | 0.342 | 187.78 | 0.119 | 38/102 | 0.013 | 102 |
| Uranus | +1 yr | 884.915 | 884.916 | 0.029 | 846.66 | 573.448 | 38/146 | 0.000 | 146 |
| Uranus | +3 yr | 939.675 | 939.470 | 0.246 | 822.79 | 585.539 | 38/125 | 0.000 | 125 |
| Uranus | +5 yr | 983.659 | 983.481 | 0.638 | 823.04 | 600.908 | 42/102 | 0.092 | 102 |
| Neptune | +1 yr | 1981.759 | 1981.695 | 0.067 | 1860.10 | 197.767 | 32/146 | 0.000 | 146 |
| Neptune | +3 yr | 2463.868 | 2463.414 | 0.609 | 2091.06 | 203.674 | 25/125 | 0.000 | 125 |
| Neptune | +5 yr | 2973.436 | 2971.965 | 1.668 | 2256.44 | 209.837 | 17/102 | 0.000 | 102 |
| Pluto | +1 yr | 3246.071 | 3246.038 | 0.097 | 2969.75 | 0.790 | 63/146 | 0.116 | 146 |
| Pluto | +3 yr | 4749.801 | 4750.199 | 0.880 | 4254.56 | 0.790 | 56/125 | 0.283 | 125 |
| Pluto | +5 yr | 5414.612 | 5413.310 | 2.417 | 4607.97 | 0.790 | 48/102 | 0.621 | 102 |

## The pre-registered M-B targets (ledger C26)

| body | dynamics gap +5 yr | 2026-10-12 baseline | target (half) | met |
|---|---|---|---|---|
| Moon | 0.741 | 3.40 | 1.70 | YES |
| Mars | 0.268 | 2.11 | 1.05 | YES |
| Earth | 0.323 | 0.90 | 0.45 | YES |
| Mercury | 0.425 | 1.20 | 0.60 | YES |

| body | sign test p at +5 yr | indistinguishable (p > 0.05) |
|---|---|---|
| Venus | 1.000 | YES |
| Jupiter | 0.000 | no |
| Saturn | 0.013 | no |
| Uranus | 0.092 | YES |
| Neptune | 0.000 | no |
| Pluto | 0.621 | YES |
