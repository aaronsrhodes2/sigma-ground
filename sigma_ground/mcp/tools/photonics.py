"""Photonics & luminescence analysis tools (standard optics).

Composite tools cascading through field.interface.{photonics, optics, phosphor}.
"""
from __future__ import annotations

from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_SRC = "sigma_ground.field.interface (photonics, optics, phosphor)"


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        return None


def optical_waveguide_analysis(wavelength_m: float = 1.55e-6,
                               core_thickness_m: float = 5.0e-6,
                               n_core: float = 1.50,
                               n_clad: float = 1.48) -> dict[str, Any]:
    """Symmetric slab dielectric waveguide: numerical aperture, normalized
    frequency (V-number), and the guided TE-mode count.
    e.g. optical_waveguide_analysis(1.55e-6, 5e-6, 1.50, 1.48)."""
    from sigma_ground.field.interface import photonics as P
    results = {
        "numerical_aperture": _safe(P.numerical_aperture, n_core, n_clad),
        "v_number": _safe(P.v_number_slab, core_thickness_m, wavelength_m, n_core, n_clad),
        "guided_TE_modes": _safe(P.slab_modes_count, core_thickness_m, wavelength_m, n_core, n_clad),
    }
    return ToolResult(value=results, units="dimensionless", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="NA=sqrt(n_core^2-n_clad^2); V=(2 pi d/lambda) NA; m=floor(2V/pi)",
                      inputs={"wavelength_m": wavelength_m,
                              "core_thickness_m": core_thickness_m,
                              "n_core": n_core, "n_clad": n_clad}).to_dict()


def photonic_bandgap_analysis(design_wavelength_m: float = 550.0e-9,
                              n_low: float = 1.46,
                              n_high: float = 2.35,
                              n_pairs: int = 10) -> dict[str, Any]:
    """Quarter-wave dielectric Bragg mirror (1D photonic bandgap): center
    wavelength, fractional stop-band width, and peak reflectance for N layer
    pairs. e.g. photonic_bandgap_analysis(550e-9, 1.46, 2.35, 10)."""
    from sigma_ground.field.interface import photonics as P
    d_low = design_wavelength_m / (4.0 * n_low)
    d_high = design_wavelength_m / (4.0 * n_high)
    results = {
        "bragg_center_wavelength_m": _safe(P.bragg_wavelength, n_low, d_low, n_high, d_high),
        "bandwidth_fraction": _safe(P.bragg_bandwidth_fraction, n_low, n_high),
        "peak_reflectance": _safe(P.bragg_reflectance, n_low, n_high, n_pairs),
    }
    return ToolResult(value=results, units="m, dimensionless", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="lambda_B=2(n1 d1+n2 d2); dlambda/lambda=(4/pi)arcsin(|dn|/sum); R quarter-wave stack",
                      inputs={"design_wavelength_m": design_wavelength_m,
                              "n_low": n_low, "n_high": n_high,
                              "n_pairs": n_pairs}).to_dict()


def nonlinear_optics_analysis(intensity_w_m2: float = 1.0e13,
                              wavelength_m: float = 1.064e-6,
                              n0: float = 1.45,
                              n2_m2_w: float = 2.7e-20,
                              length_m: float = 0.01,
                              d_eff_pm_v: float = 0.39,
                              n_omega: float = 1.494,
                              n_2omega: float = 1.51) -> dict[str, Any]:
    """Nonlinear optics: Kerr intensity-dependent index, B-integral (nonlinear
    phase), self-focusing critical power, and a second-harmonic-generation
    efficiency factor. e.g. nonlinear_optics_analysis(1e13, 1.064e-6)."""
    from sigma_ground.field.interface import photonics as P
    results = {
        "kerr_index": _safe(P.kerr_refractive_index, n0, n2_m2_w, intensity_w_m2),
        "b_integral_rad": _safe(P.nonlinear_phase_shift, n2_m2_w, intensity_w_m2, length_m, wavelength_m),
        "self_focusing_critical_power_W": _safe(P.self_focusing_critical_power, wavelength_m, n0, n2_m2_w),
        "shg_efficiency_factor": _safe(P.shg_efficiency_factor, d_eff_pm_v, length_m, n_omega, n_2omega, wavelength_m),
    }
    return ToolResult(value=results, units="dimensionless, rad, W, a.u.", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="n=n0+n2 I; phi=2pi n2 I L/lambda; P_cr=3.77 lambda^2/(8 pi n0 n2)",
                      inputs={"intensity_w_m2": intensity_w_m2,
                              "wavelength_m": wavelength_m, "n0": n0,
                              "n2_m2_w": n2_m2_w, "length_m": length_m}).to_dict()


def material_color_analysis(category: str = "metal", key: str = "copper",
                            dye_key: str | None = None,
                            substrate_key: str | None = None) -> dict[str, Any]:
    """Physically-derived sRGB color of a material (metal Drude reflectance,
    organic spectrum, or dye-on-substrate transmission).
    e.g. material_color_analysis('metal', 'gold')."""
    from sigma_ground.field.interface import optics as O
    rgb = _safe(O.get_material_color, category, key, dye_key, substrate_key)
    if rgb:
        bit = tuple(int(round(max(0.0, min(1.0, c)) * 255)) for c in rgb)
        results = {
            "rgb_0_1": [round(c, 4) for c in rgb],
            "rgb_8bit": list(bit),
            "hex": "#%02X%02X%02X" % bit,
        }
    else:
        results = {"rgb_0_1": None, "rgb_8bit": None, "hex": None}
    return ToolResult(value=results, units="sRGB (0-1)",
                      source="sigma_ground.field.interface.optics",
                      provenance_tag="DERIVED",
                      formula="physical reflectance/transmission -> sRGB",
                      inputs={"category": category, "key": key}).to_dict()


def phosphor_decay_analysis(time_s: float = 10.0e-3,
                            tau_s: float = 5.0e-3,
                            initial_brightness: float = 1.0) -> dict[str, Any]:
    """Phosphor / luminescence afterglow brightness I(t)=I0 exp(-t/tau) at a
    time t after excitation, plus the surviving fraction.
    e.g. phosphor_decay_analysis(10e-3, 5e-3)."""
    from sigma_ground.field.interface import phosphor as PH
    bright = _safe(PH.phosphor_brightness, initial_brightness, time_s, tau_s)
    frac = (bright / initial_brightness) if (bright is not None and initial_brightness) else None
    results = {"brightness": bright, "fraction_remaining": frac}
    return ToolResult(value=results, units="relative",
                      source="sigma_ground.field.interface.phosphor",
                      provenance_tag="DERIVED",
                      formula="I(t)=I0 exp(-t/tau)",
                      inputs={"time_s": time_s, "tau_s": tau_s,
                              "initial_brightness": initial_brightness}).to_dict()
