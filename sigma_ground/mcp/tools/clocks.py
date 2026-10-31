"""Clock rates in orbit: the net relativistic drift of a satellite clock against a ground clock.

GPS clocks run fast by about 38 microseconds a day: gravitational time dilation (+45.7, the clock sits higher in Earth's potential)
minus special-relativistic time dilation (-7.2, it moves at 3.9 km/s). Weak-field, circular orbit, non-rotating ground clock
(Ashby 2003, "Relativity in the Global Positioning System", Living Rev. Relativity 6).
"""

from __future__ import annotations

from sigma_ground.mcp.provenance import ToolResult

_C = 299792458.0
_DAY = 86400.0


def _earth():
    from sigma_ground.field.interface import body_data
    entry, _ = body_data.resolve("earth")
    d = body_data.describe(entry)
    return entry["gm_km3_s2"] * 1e9, entry["reference_radius_km"] * 1e3, d["sources"]["gm_km3_s2"], d["sources"]["reference_radius_km"]


def orbital_clock_drift(altitude_km: float) -> ToolResult:
    """Net rate of a clock in a circular orbit at `altitude_km` above Earth's surface against a clock on the surface, in microseconds per day.

    drift = [GM/c^2 (1/R - 1/r) - GM/(2 c^2 r)] x 86400 s; positive = the orbiting clock runs fast. GPS (20200 km): +38.5 us/day.
    First order in GM/(c^2 r); Earth's rotation and eccentricity neglected (the periodic terms, not the secular rate).
    """
    inputs = {"altitude_km": altitude_km}
    altitude_m = altitude_km * 1e3
    if altitude_m <= 0:
        return ToolResult(value=None, source="invalid input", inputs=inputs, notes="altitude above the surface must be positive.")
    gm, R, gm_src, r_src = _earth()
    r = R + altitude_m
    gr_term = gm / _C**2 * (1.0 / R - 1.0 / r)
    sr_term = gm / (2.0 * _C**2 * r)
    net = (gr_term - sr_term) * _DAY * 1e6
    return ToolResult(
        value=net, units="us/day", source=f"weak-field GR + SR, circular orbit (Ashby 2003); GM: {gm_src}; R: {r_src}",
        formula="drift = [GM/c^2 (1/R - 1/r) - GM/(2 c^2 r)] * 86400 s", provenance_tag="DERIVED",
        inputs={**inputs, "orbit_radius_m": r, "gm_m3_s2": gm, "earth_radius_m": R},
        notes=f"gravitational +{gr_term * _DAY * 1e6:.1f} us/day, velocity -{sr_term * _DAY * 1e6:.1f} us/day; "
              f"the clock in orbit runs {'fast' if net > 0 else 'slow'} by {abs(net):.1f} us/day. The two cancel at r = 1.5 R (altitude about 3,190 km).",
    )


def altitude_for_clock_drift(drift_us_per_day: float) -> float | None:
    """The altitude (m) above 1.5 R at which the net drift equals `drift_us_per_day` (the branch GPS sits on); None when no such altitude.
    Bisection on orbital_clock_drift, which rises monotonically above the cancellation altitude."""
    gm, R, _, _ = _earth()
    lo, hi = 0.5 * R + 1.0, 400.0 * R          # just above the cancellation radius 1.5 R, out to beyond geostationary and the Moon
    f = lambda h: orbital_clock_drift(h / 1e3).value - drift_us_per_day  # noqa: E731
    if f(lo) > 0 or f(hi) < 0:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1.0:
            break
    return 0.5 * (lo + hi)
