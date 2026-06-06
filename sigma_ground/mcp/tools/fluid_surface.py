"""Free-surface fluid-dynamics analysis tools (standard physics).

Composite tools over field.interface.fluid_surface (buoyancy, wind shear stress,
capillary-gravity ripples), using NIST-anchored water seeds from liquid_water.
Built for the wind-over-a-pond scenario (misc/WATER_SIM_PLAN.md): these give
Materia/Deckard the buoyancy verdict and the ripple field (wavelength, speed,
amplitude) that parameterizes the water surface.
"""
from __future__ import annotations

import math
from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_SRC = "sigma_ground.field.interface.fluid_surface"


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        return None


def _water_seeds(temperature_k: float):
    from sigma_ground.field.interface import liquid_water as W
    return W.water_surface_tension(temperature_k), W.water_density(temperature_k)


def buoyancy_analysis(material_key: str = "copper", temperature_k: float = 293.15,
                      fluid_density_kg_m3: float | None = None) -> dict[str, Any]:
    """Will it float, and how deep does it sit? Archimedes: submerged fraction =
    rho_body / rho_fluid (the body sinks if >= 1). Default is a copper body in
    water -- copper sinks, so a 'half-sunk' ball in a shallow pond rests on the
    bottom (half above the waterline), it is NOT floating. Body keys from the
    materials DB (copper, water_ice, wood_oak, lead, aluminum, gold, ...).
    e.g. buoyancy_analysis('water_ice')  -> floats, ~92%% submerged (iceberg)."""
    from sigma_ground.field.interface import fluid_surface as FS
    from sigma_ground.field.interface import surface as S
    body = (S.MATERIALS.get(material_key) or {}).get("density_kg_m3")
    if fluid_density_kg_m3 is None:
        _, fluid_density_kg_m3 = _water_seeds(temperature_k)
    frac = _safe(FS.submerged_fraction, body, fluid_density_kg_m3) if body else None
    fl = _safe(FS.floats, body, fluid_density_kg_m3) if body else None
    results = {
        "body_density_kg_m3": body,
        "fluid_density_kg_m3": fluid_density_kg_m3,
        "floats": fl,
        "submerged_fraction": frac,
    }
    note = ""
    if body and fluid_density_kg_m3:
        if not fl:
            note = (f"{material_key} (rho={body:.0f}) is denser than the fluid "
                    f"(rho={fluid_density_kg_m3:.0f}) -> it SINKS; a 'half-sunk' ball "
                    f"is shallow-pond geometry (on the floor, half above water), not floating.")
        else:
            note = f"{material_key} floats with {frac * 100:.0f}% of its volume submerged."
    return ToolResult(value=results, units="kg/m^3, dimensionless", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="f_submerged = rho_body / rho_fluid (Archimedes)",
                      notes=note,
                      inputs={"material_key": material_key,
                              "temperature_k": temperature_k}).to_dict()


def wind_wave_analysis(wind_speed_m_s: float = 5.0, temperature_k: float = 293.15,
                       wavelength_m: float | None = None) -> dict[str, Any]:
    """Wind blowing across water -> surface stress + capillary-gravity ripples.

    Reports the wind shear stress (tau = C_d rho_air U^2), the friction velocity,
    the slowest possible wave (capillary-gravity minimum: ~0.23 m/s at ~1.7 cm for
    water -- a wind must beat this to raise ripples), and the ripple field
    (wavelength, phase & group speed, frequency, order-of-magnitude amplitude)
    that a renderer needs to displace the water surface.
    e.g. wind_wave_analysis(5.0)."""
    from sigma_ground.field.interface import fluid_surface as FS
    sigma, rho = _water_seeds(temperature_k)
    tau = _safe(FS.wind_shear_stress, wind_speed_m_s)
    ustar = _safe(FS.friction_velocity, tau, rho) if tau is not None else None
    cl = _safe(FS.capillary_gravity_min_phase_speed, sigma, rho)
    c_min, lam_min = cl if cl else (None, None)
    # A wind first excites waves near lambda_min (the easiest scale to raise).
    lam = wavelength_m if wavelength_m else lam_min
    c = _safe(FS.capillary_gravity_phase_speed, lam, sigma, rho) if lam else None
    omega = _safe(FS.capillary_gravity_angular_frequency, lam, sigma, rho) if lam else None
    cg = _safe(FS.capillary_gravity_group_speed, lam, sigma, rho) if lam else None
    amp = (0.1 * lam / (2.0 * math.pi)) if lam else None   # gentle-ripple estimate
    raises = bool(c_min is not None and wind_speed_m_s > c_min)
    results = {
        "wind_shear_stress_Pa": tau,
        "friction_velocity_m_s": ustar,
        "min_phase_speed_m_s": c_min,
        "min_wavelength_m": lam_min,
        "raises_ripples": raises,
        "ripple_wavelength_m": lam,
        "ripple_phase_speed_m_s": c,
        "ripple_angular_freq_rad_s": omega,
        "ripple_group_speed_m_s": cg,
        "ripple_amplitude_estimate_m": amp,
    }
    note = ""
    if c_min and lam:
        note = (f"Wind {wind_speed_m_s} m/s {'>' if raises else '<='} the "
                f"{c_min:.2f} m/s minimum wave speed, so it "
                f"{'raises' if raises else 'cannot raise'} ripples. Dominant initial "
                f"wavelength ~{lam * 100:.1f} cm. NOTE: amplitude is an order-of-"
                f"magnitude estimate (gentle steepness ~0.1), not a fetch/duration "
                f"wave-growth solution.")
    return ToolResult(value=results, units="Pa, m/s, m, rad/s", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="tau=C_d rho_air U^2; c=sqrt(g/k + sigma k/rho); "
                              "c_min=sqrt(2)(g sigma/rho)^(1/4)",
                      notes=note,
                      inputs={"wind_speed_m_s": wind_speed_m_s,
                              "temperature_k": temperature_k,
                              "wavelength_m": wavelength_m}).to_dict()
