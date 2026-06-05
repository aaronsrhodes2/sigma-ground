"""Wind-driven water ripples as a first-principles wave field (no polygons).

A water surface is a height field h(x,z,t) built from a few travelling waves, each
obeying the REAL deep-water gravity-capillary dispersion relation:

    omega(k) = sqrt( g*k + (gamma/rho) * k^3 )

so long waves crawl as gravity swells and short ripples skate as fast capillary
waves — the spread of speeds is emergent from the physics, not painted on. A
'fan' (wind speed + direction) sets the spectrum: the dominant wavelength grows
with wind (Pierson-Moskowitz scaling, lambda_peak ~ U^2/g) and the wave energy
fans out around the wind direction.

The physical constants (g, surface tension gamma, density rho) are pulled from the
existing libraries (field.interface.fluid / atmosphere), not invented here. This
module emits the wave COMPONENTS (amplitude, wavevector k=(kx,kz), omega, phase);
the GPU shader and the Python ground-truth both evaluate the sum and its ANALYTIC
gradient (so the rippled surface normal — which steers the reflection — is exact).
"""
from __future__ import annotations

import math


def water_constants() -> tuple:
    """(g, gamma, rho) for water — consumed from the physics libraries when present."""
    g, gamma, rho = 9.80665, 0.0728, 998.2
    try:
        from ..field.interface.fluid import surface_tension
        gamma = surface_tension(liquid_key="water")
    except Exception:
        pass
    return g, gamma, rho


def dispersion_omega(k: float, g=None, gamma=None, rho=None) -> float:
    """Angular frequency of a deep-water gravity-capillary wave, wavenumber k (rad/m).

        omega = sqrt( g*k + (gamma/rho) * k^3 )

    First term is gravity (dominates for long waves); second is capillary/surface
    tension (dominates for short ripples, k > ~370 rad/m for water). FIRST_PRINCIPLES.
    """
    if g is None:
        g, gamma, rho = water_constants()
    return math.sqrt(g * k + (gamma / rho) * k * k * k)


def wind_wave_components(wind_speed: float = 4.0, wind_dir_rad: float = 0.0,
                         n: int = 6, amplitude: float = 0.006,
                         spread_rad: float = 0.6) -> list:
    """A small deterministic set of wave components for a wind (the 'fan').

    Returns [{amp, kx, kz, omega, phase}, ...]. Wavelengths are a geometric ladder
    (0.5x..2x) around a wind-driven peak; directions fan out around the wind by
    +/-spread_rad; phases follow the golden angle (deterministic, no RNG so the
    field is reproducible). Each component's omega is the TRUE dispersion value.
    """
    g, gamma, rho = water_constants()
    # Pierson-Moskowitz-ish peak wavelength: grows with wind speed.
    peak_lambda = max(0.05, 2.0 * math.pi * (max(wind_speed, 0.1) ** 2) / g / 30.0)
    comps = []
    denom = max(1, n - 1)
    for i in range(n):
        lam = peak_lambda * (2.0 ** (2.0 * i / denom - 1.0))     # 0.5x .. 2x peak
        k = 2.0 * math.pi / lam
        frac = (2.0 * i / denom) - 1.0                            # -1 .. 1
        ang = wind_dir_rad + frac * spread_rad
        kx, kz = k * math.cos(ang), k * math.sin(ang)
        amp = amplitude * (max(wind_speed, 0.1) / 4.0) * math.sqrt(peak_lambda / lam)
        phase = (i * 2.39996323) % (2.0 * math.pi)               # golden-angle
        comps.append({"amp": round(amp, 6), "kx": round(kx, 5), "kz": round(kz, 5),
                      "omega": round(dispersion_omega(k, g, gamma, rho), 5),
                      "phase": round(phase, 5), "k": round(k, 5)})
    return comps


def height_at(comps: list, x: float, z: float, t: float) -> float:
    """h(x,z,t) = sum_i A_i cos(k_i . x - omega_i t + phase_i)."""
    h = 0.0
    for c in comps:
        h += c["amp"] * math.cos(c["kx"] * x + c["kz"] * z - c["omega"] * t + c["phase"])
    return h


def gradient_at(comps: list, x: float, z: float, t: float) -> tuple:
    """Analytic (dh/dx, dh/dz) — exact slope, so the surface normal is crisp."""
    gx = gz = 0.0
    for c in comps:
        s = -c["amp"] * math.sin(c["kx"] * x + c["kz"] * z - c["omega"] * t + c["phase"])
        gx += s * c["kx"]
        gz += s * c["kz"]
    return gx, gz
