"""Textbook puzzles with one exact answer: Monty Hall, the birthday problem, two bodies closing on each other.

The adversarial suite asks these (adv_puzzle_001..003, 2026-10-30) and the router had no tool to name: it sent the trains to
relativistic_velocity_addition and the birthdays to day_of_year. Each is plain arithmetic with a named source; nothing here is estimated.
"""

from __future__ import annotations

import math

from sigma_ground.mcp.provenance import ToolResult


def monty_hall_switch(n: int = 3) -> ToolResult:
    """Probability of winning the Monty Hall game by switching, with n doors, one prize, and a host who opens every other losing door.

    P(win | switch) = (n - 1) / n; P(win | stay) = 1 / n. Three doors: 2/3 against 1/3 (vos Savant 1990; Selvin 1975).
    """
    inputs = {"n": n}
    if not isinstance(n, int) or isinstance(n, bool) or n < 3:
        return ToolResult(value=None, source="invalid input", inputs=inputs, notes="n is the number of doors, at least 3.")
    p = (n - 1) / n
    return ToolResult(value=p, units="", source="combinatorics (Selvin 1975; vos Savant 1990)", formula="P(switch) = (n - 1) / n",
                      provenance_tag="DERIVED", inputs=inputs,
                      notes=f"Yes, switch: {p:.4g} by switching against {1 / n:.4g} by staying. The host knows where the prize is and always opens a losing door.")


def birthday_collision(n: int) -> ToolResult:
    """Probability that at least two of n people share a birthday (365 equally likely days, no leap day).

    P = 1 - 365! / (365^n (365 - n)!); 23 people give 0.5073 (the birthday paradox).
    """
    inputs = {"n": n}
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        return ToolResult(value=None, source="invalid input", inputs=inputs, notes="n is the number of people, at least 1.")
    if n > 365:
        p = 1.0
    else:
        log_no_match = sum(math.log((365 - k) / 365.0) for k in range(n))
        p = 1.0 - math.exp(log_no_match)
    return ToolResult(value=p, units="", source="combinatorics (the birthday problem; von Mises 1939)",
                      formula="P = 1 - prod_{k=0}^{n-1} (365 - k) / 365", provenance_tag="DERIVED", inputs=inputs,
                      notes="365 equally likely birthdays, no 29 February; real birth dates are slightly clustered, which raises P a little.")


def closing_time(distance_m: float, velocity1_m_s: float, velocity2_m_s: float) -> ToolResult:
    """Time for two bodies approaching each other head-on to meet: t = d / (v1 + v2). Non-relativistic.

    Two trains 100 km apart at 60 and 40 km/h meet after 100 / (60 + 40) h = 1 h.
    """
    inputs = {"distance_m": distance_m, "velocity1_m_s": velocity1_m_s, "velocity2_m_s": velocity2_m_s}
    closing = velocity1_m_s + velocity2_m_s
    if distance_m < 0 or closing <= 0:
        return ToolResult(value=None, source="invalid input", inputs=inputs,
                          notes="distance is non-negative and the two speeds (both taken towards each other) must add to more than zero.")
    t = distance_m / closing
    return ToolResult(value=t, units="s", source="kinematics (relative motion)", formula="t = d / (v1 + v2)", provenance_tag="DERIVED",
                      inputs=inputs, notes=f"Closing speed {closing:g} m/s. Speeds are given as magnitudes towards each other; "
                      f"classical addition (both far below c).")
