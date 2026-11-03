# Hindcast, `restated_geo` against `restated_zon` (4 starts: 2000, 2005, 2010, 2015)

Same code, step (0.0125 d), fixture snapshot (j2010) and DE440 answer key; only the predictor's configuration differs. Median heliocentric position error over the starts that reach the lead, km. Ratio below 1 means the new configuration is closer to DE440.

| body | lead | control | new | ratio | starts better |
|---|---|---|---|---|---|
| Earth | +1 yr | 0.0277 | 0.0279 | 1.006 | 1/4 |
| Earth | +2 yr | 0.0486 | 0.0487 | 1.004 | 2/4 |
| Earth | +3 yr | 0.0711 | 0.0707 | 0.994 | 2/4 |
| Earth | +5 yr | 0.113 | 0.112 | 0.997 | 3/4 |
| Earth | +10 yr | 0.208 | 0.203 | 0.980 | 2/3 |
| Jupiter | +1 yr | 6.64 | 6.64 | 1.000 | 2/4 |
| Jupiter | +2 yr | 7.15 | 7.15 | 1.000 | 1/4 |
| Jupiter | +3 yr | 12.5 | 12.5 | 1.000 | 3/4 |
| Jupiter | +5 yr | 12.2 | 12.2 | 1.000 | 4/4 |
| Jupiter | +10 yr | 13.9 | 13.9 | 1.000 | 3/3 |
| Mars | +1 yr | 0.0281 | 0.0281 | 1.000 | 0/4 |
| Mars | +2 yr | 0.0513 | 0.0513 | 1.000 | 2/4 |
| Mars | +3 yr | 0.0676 | 0.0676 | 1.000 | 0/4 |
| Mars | +5 yr | 0.119 | 0.119 | 1.000 | 1/4 |
| Mars | +10 yr | 0.278 | 0.279 | 1.001 | 2/3 |
| Mercury | +1 yr | 0.0192 | 0.0193 | 1.007 | 0/4 |
| Mercury | +2 yr | 0.0344 | 0.0348 | 1.011 | 0/4 |
| Mercury | +3 yr | 0.0506 | 0.0514 | 1.014 | 0/4 |
| Mercury | +5 yr | 0.0896 | 0.0913 | 1.019 | 0/4 |
| Mercury | +10 yr | 0.164 | 0.168 | 1.022 | 1/3 |
| Moon | +1 yr | 0.0713 | 0.0368 | 0.515 | 4/4 |
| Moon | +2 yr | 0.131 | 0.0483 | 0.368 | 4/4 |
| Moon | +3 yr | 0.156 | 0.0557 | 0.358 | 4/4 |
| Moon | +5 yr | 0.245 | 0.113 | 0.463 | 4/4 |
| Moon | +10 yr | 0.443 | 0.28 | 0.632 | 2/3 |
| Neptune | +1 yr | 120 | 120 | 1.000 | 0/4 |
| Neptune | +2 yr | 128 | 128 | 1.000 | 0/4 |
| Neptune | +3 yr | 137 | 137 | 1.000 | 0/4 |
| Neptune | +5 yr | 156 | 156 | 1.000 | 0/4 |
| Neptune | +10 yr | 181 | 181 | 1.000 | 0/3 |
| Pluto | +1 yr | 0.791 | 0.791 | 1.000 | 0/4 |
| Pluto | +2 yr | 0.762 | 0.762 | 1.000 | 0/4 |
| Pluto | +3 yr | 0.733 | 0.733 | 1.000 | 0/4 |
| Pluto | +5 yr | 0.731 | 0.731 | 1.000 | 0/4 |
| Pluto | +10 yr | 0.655 | 0.655 | 1.000 | 0/3 |
| Saturn | +1 yr | 0.0905 | 0.0905 | 1.000 | 0/4 |
| Saturn | +2 yr | 0.096 | 0.096 | 1.000 | 0/4 |
| Saturn | +3 yr | 0.106 | 0.106 | 1.000 | 0/4 |
| Saturn | +5 yr | 0.131 | 0.131 | 1.000 | 0/4 |
| Saturn | +10 yr | 0.353 | 0.353 | 1.000 | 0/3 |
| Uranus | +1 yr | 485 | 485 | 1.000 | 0/4 |
| Uranus | +2 yr | 493 | 493 | 1.000 | 0/4 |
| Uranus | +3 yr | 502 | 502 | 1.000 | 0/4 |
| Uranus | +5 yr | 520 | 520 | 1.000 | 0/4 |
| Uranus | +10 yr | 543 | 543 | 1.000 | 2/3 |
| Venus | +1 yr | 0.0199 | 0.0199 | 1.000 | 1/4 |
| Venus | +2 yr | 0.0315 | 0.0314 | 0.995 | 3/4 |
| Venus | +3 yr | 0.0481 | 0.0479 | 0.995 | 4/4 |
| Venus | +5 yr | 0.0753 | 0.0746 | 0.990 | 4/4 |
| Venus | +10 yr | 0.148 | 0.145 | 0.979 | 3/3 |
