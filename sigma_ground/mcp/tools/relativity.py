"""Special-relativity tools.

Thin wrappers around `sigma_ground.field.relativity` -- the underlying
implementations already handle the math and edge cases. We just add
the MCP-side ToolResult, citations, and input validation.

The σ-derived functions (`sigma_time_dilation`, `effective_liv_scale_gev`)
live in the EXTENDED tier of the SSBM theoretical layer and are NOT
exposed here per the MCP-positioning decision.
"""

from __future__ import annotations

import math

from sigma_ground.mcp.provenance import ToolResult


def _nonfinite(inputs: dict):
    """ToolResult refusing NaN / inf inputs, or None (2026-10-10 audit: NaN
    and inf passed through to NaN, inf or 0 under physics labels)."""
    for k, x in inputs.items():
        if not (isinstance(x, (int, float)) and math.isfinite(x)):
            return ToolResult(value=None, source="invalid input",
                              notes=f"{k} must be a finite number (got {x!r})",
                              inputs=inputs)
    return None


def lorentz_factor(velocity_m_s: float) -> ToolResult:
    """gamma = 1 / sqrt(1 - v^2/c^2)."""
    bad = _nonfinite({"velocity_m_s": velocity_m_s})
    if bad:
        return bad
    from sigma_ground.field.constants import C
    if abs(velocity_m_s) >= C:
        return ToolResult(
            value=None, source="invalid input",
            notes=f"velocity_m_s must be subluminal (|v| < c = {C} m/s)",
            inputs={"velocity_m_s": velocity_m_s},
        )
    from sigma_ground.field.relativity import lorentz_factor as _gamma
    gamma = _gamma(velocity_m_s)
    return ToolResult(
        value=gamma,
        units="dimensionless",
        source="sigma-ground (special relativity)",
        formula="gamma = 1 / sqrt(1 - (v/c)^2)",
        inputs={"velocity_m_s": velocity_m_s},
        notes=("At v = 0.5c gamma ~ 1.155; at v = 0.99c gamma ~ 7.09. "
                "Diverges at v = c."),
    )


def relativistic_time_dilation(rest_time_s: float,
                                 velocity_m_s: float) -> ToolResult:
    """Moving clock's tick interval as seen from rest frame: t = gamma t0."""
    bad = _nonfinite({"rest_time_s": rest_time_s, "velocity_m_s": velocity_m_s})
    if bad:
        return bad
    from sigma_ground.field.constants import C
    if abs(velocity_m_s) >= C:
        return ToolResult(
            value=None, source="invalid input",
            notes=f"velocity_m_s must be subluminal",
            inputs={"rest_time_s": rest_time_s, "velocity_m_s": velocity_m_s},
        )
    from sigma_ground.field.relativity import time_dilation as _td
    t = _td(rest_time_s, velocity_m_s)
    return ToolResult(
        value=t,
        units="s",
        source="sigma-ground (SR time dilation)",
        formula="t = gamma * t0 = t0 / sqrt(1 - (v/c)^2)",
        inputs={"rest_time_s": rest_time_s, "velocity_m_s": velocity_m_s},
        notes=("t > t0: moving clocks tick slower when measured from the "
                "rest frame. Classic muon-lifetime experiment confirms "
                "this to high precision."),
    )


def relativistic_length_contraction(rest_length_m: float,
                                     velocity_m_s: float) -> ToolResult:
    """Moving ruler's length as seen from rest frame: L = L0 / gamma."""
    bad = _nonfinite({"rest_length_m": rest_length_m, "velocity_m_s": velocity_m_s})
    if bad:
        return bad
    from sigma_ground.field.constants import C
    if abs(velocity_m_s) >= C:
        return ToolResult(
            value=None, source="invalid input",
            notes="velocity_m_s must be subluminal",
            inputs={"rest_length_m": rest_length_m,
                    "velocity_m_s": velocity_m_s},
        )
    from sigma_ground.field.relativity import length_contraction as _lc
    L = _lc(rest_length_m, velocity_m_s)
    return ToolResult(
        value=L,
        units="m",
        source="sigma-ground (SR length contraction)",
        formula="L = L0 * sqrt(1 - (v/c)^2) = L0 / gamma",
        inputs={"rest_length_m": rest_length_m,
                "velocity_m_s": velocity_m_s},
    )


def relativistic_energy(rest_mass_kg: float, velocity_m_s: float) -> ToolResult:
    """Total relativistic energy E = gamma m c^2."""
    bad = _nonfinite({"rest_mass_kg": rest_mass_kg, "velocity_m_s": velocity_m_s})
    if bad:
        return bad
    from sigma_ground.field.constants import C
    if abs(velocity_m_s) >= C:
        return ToolResult(value=None, source="invalid input",
                           notes="velocity_m_s must be subluminal",
                           inputs={"rest_mass_kg": rest_mass_kg,
                                   "velocity_m_s": velocity_m_s})
    if rest_mass_kg < 0:
        return ToolResult(value=None, source="invalid input",
                           notes="rest_mass_kg must be non-negative",
                           inputs={"rest_mass_kg": rest_mass_kg,
                                   "velocity_m_s": velocity_m_s})
    from sigma_ground.field.relativity import relativistic_energy as _re
    E = _re(rest_mass_kg, velocity_m_s)
    return ToolResult(
        value=E,
        units="J",
        source="sigma-ground (SR total energy)",
        formula="E = gamma m c^2",
        inputs={"rest_mass_kg": rest_mass_kg, "velocity_m_s": velocity_m_s},
        notes="Includes rest energy m c^2 plus kinetic. At v=0 reduces to m c^2.",
    )


def relativistic_momentum(rest_mass_kg: float,
                            velocity_m_s: float) -> ToolResult:
    """Relativistic momentum p = gamma m v (signed, along v).

    FIXED 2026-10-10: returned the magnitude (the sign of v was dropped,
    contrary to the formula label) and accepted negative masses.
    """
    bad = _nonfinite({"rest_mass_kg": rest_mass_kg, "velocity_m_s": velocity_m_s})
    if bad:
        return bad
    from sigma_ground.field.constants import C
    if abs(velocity_m_s) >= C:
        return ToolResult(value=None, source="invalid input",
                           notes="velocity_m_s must be subluminal",
                           inputs={"rest_mass_kg": rest_mass_kg,
                                   "velocity_m_s": velocity_m_s})
    if rest_mass_kg < 0:
        return ToolResult(value=None, source="invalid input",
                           notes="rest_mass_kg must be non-negative",
                           inputs={"rest_mass_kg": rest_mass_kg,
                                   "velocity_m_s": velocity_m_s})
    from sigma_ground.field.relativity import momentum_rel as _pr
    p = math.copysign(_pr(rest_mass_kg, velocity_m_s), velocity_m_s)
    return ToolResult(
        value=p,
        units="kg m/s",
        source="sigma-ground (SR momentum)",
        formula="p = gamma m v",
        inputs={"rest_mass_kg": rest_mass_kg, "velocity_m_s": velocity_m_s},
        notes="Reduces to Newtonian p = mv when v << c.",
    )


def relativistic_velocity_addition(u_m_s: float, v_m_s: float) -> ToolResult:
    """Einstein velocity addition: (u + v) / (1 + uv/c^2).

    For collinear velocities u and v (both magnitudes signed along same
    axis). Returns the resultant speed in the original frame.
    """
    bad = _nonfinite({"u_m_s": u_m_s, "v_m_s": v_m_s})
    if bad:
        return bad
    from sigma_ground.field.constants import C
    if abs(u_m_s) >= C or abs(v_m_s) >= C:
        return ToolResult(value=None, source="invalid input",
                           notes="both u and v must be subluminal",
                           inputs={"u_m_s": u_m_s, "v_m_s": v_m_s})
    from sigma_ground.field.relativity import velocity_addition as _va
    w = _va(u_m_s, v_m_s)
    return ToolResult(
        value=w,
        units="m/s",
        source="sigma-ground (Einstein velocity addition)",
        formula="w = (u + v) / (1 + u v / c^2)",
        inputs={"u_m_s": u_m_s, "v_m_s": v_m_s},
        notes=("If u = v = 0.9c, classical addition would give 1.8c, but "
                "the relativistic answer is ~0.994c < c, as required."),
    )


def doppler_shift_factor(velocity_m_s: float,
                           angle_to_los_deg: float = 0.0) -> ToolResult:
    """Relativistic Doppler factor for light: lambda_obs / lambda_emit (= 1 + z).

    Angle convention (stated explicitly; every quantity in the OBSERVER's
    rest frame, at the moment the light is received):

        lambda_obs / lambda_emit = gamma (1 + beta cos(theta))

    theta is the angle between the source's velocity and the line of sight
    pointing AWAY from the observer (observer -> source). theta = 0: the
    source recedes head-on (redshift sqrt((1+b)/(1-b))); theta = 180 deg:
    it approaches head-on (blueshift); theta = 90 deg: transverse in the
    observer's frame, a pure time-dilation REDSHIFT, factor gamma.
    (Transverse in the SOURCE's frame instead gives 1/gamma; aberration
    makes the two differ.) Reference: MIT 8.20 Special Relativity, IAP 2008,
    problem set 3 solution 1, "lambda = lambda0 gamma (1 + beta cos theta)"
    with theta measured from the direction away from the observer.

    Parameters
    ----------
    velocity_m_s : float
        Source speed along the direction set by the angle; a positive value
        with theta = 0 is a receding source. A negative value reverses the
        velocity, which is the same as replacing theta by 180 deg - theta.
    angle_to_los_deg : float
        theta above, in degrees.
    """
    import math
    from sigma_ground.field.constants import C
    inputs = {"velocity_m_s": velocity_m_s, "angle_to_los_deg": angle_to_los_deg}
    if not (math.isfinite(velocity_m_s) and math.isfinite(angle_to_los_deg)):
        return ToolResult(value=None, source="invalid input",
                           notes="velocity_m_s and angle_to_los_deg must be finite",
                           inputs=inputs)
    if abs(velocity_m_s) >= C:
        return ToolResult(value=None, source="invalid input",
                           notes="velocity_m_s must be subluminal",
                           inputs=inputs)
    cos_theta = math.cos(math.radians(angle_to_los_deg))
    from sigma_ground.field.relativity import doppler_factor as _df
    # FIXED 2026-10-10 (A37): field.doppler_factor is the FREQUENCY ratio with
    # the angle measured from the source->observer direction; this tool's angle
    # is measured from the observer->source direction, so cos flips sign and
    # the wavelength ratio is the reciprocal: 1/D(v, -cos) = gamma (1 + b cos).
    factor = 1.0 / _df(velocity_m_s, -cos_theta)
    return ToolResult(
        value=factor,
        units="dimensionless",
        source="sigma-ground (relativistic Doppler)",
        formula="lambda_obs / lambda_emit = gamma (1 + beta cos(theta))",
        inputs=inputs,
        notes=("theta = angle, in the observer's frame, between the source "
                "velocity and the line of sight from the observer to the "
                "source (0 = receding, 180 = approaching). Value > 1 is a "
                "redshift; the frequency ratio is its reciprocal. At "
                "theta = 90 deg the transverse Doppler factor is gamma, a "
                "redshift (pure time dilation, no classical analog)."),
    )
