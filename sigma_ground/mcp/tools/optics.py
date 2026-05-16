"""Optics tools: refraction, reflection, lens equation, diffraction, Rydberg.

Pure formulas. Refractive-index lookup data for common materials lives
in tools/materials.py.
"""

from __future__ import annotations

import math

from sigma_ground.mcp.provenance import ToolResult


def snells_law_refraction_angle(n1: float, n2: float,
                                  incident_angle_deg: float) -> ToolResult:
    """Refracted ray angle from Snell's law: n1 sin(theta1) = n2 sin(theta2).

    Returns the refraction angle (in same medium notation as n2).
    Returns total internal reflection notice if angle exceeds critical.
    """
    if n1 <= 0 or n2 <= 0:
        return ToolResult(value=None, source="invalid input",
                           notes="refractive indices must be positive",
                           inputs={"n1": n1, "n2": n2,
                                   "incident_angle_deg": incident_angle_deg})
    if not 0 <= incident_angle_deg <= 90:
        return ToolResult(value=None, source="invalid input",
                           notes="incident_angle_deg must be in [0, 90]",
                           inputs={"n1": n1, "n2": n2,
                                   "incident_angle_deg": incident_angle_deg})
    sin_theta2 = (n1 / n2) * math.sin(math.radians(incident_angle_deg))
    if abs(sin_theta2) > 1:
        # Total internal reflection
        theta_c = math.degrees(math.asin(n2 / n1)) if n1 > n2 else None
        return ToolResult(
            value=None,
            source="sigma-ground (Snell's law -- TIR detected)",
            inputs={"n1": n1, "n2": n2,
                    "incident_angle_deg": incident_angle_deg},
            notes=(f"Total internal reflection: incident angle exceeds "
                    f"critical angle. Critical angle = {theta_c:.2f} deg "
                    f"for n1={n1}, n2={n2}."),
        )
    theta2 = math.degrees(math.asin(sin_theta2))
    return ToolResult(
        value=theta2,
        units="deg",
        source="sigma-ground (Snell's law)",
        formula="n1 sin(theta1) = n2 sin(theta2)",
        inputs={"n1": n1, "n2": n2,
                "incident_angle_deg": incident_angle_deg},
    )


def critical_angle_for_tir(n_dense: float, n_rare: float) -> ToolResult:
    """Critical angle for total internal reflection at a dense->rare boundary.

    Above this angle, all light is reflected. Below it, light refracts.
    """
    if n_dense <= n_rare or n_rare <= 0:
        return ToolResult(value=None, source="invalid input",
                           notes="need n_dense > n_rare > 0",
                           inputs={"n_dense": n_dense, "n_rare": n_rare})
    theta_c = math.degrees(math.asin(n_rare / n_dense))
    return ToolResult(
        value=theta_c,
        units="deg",
        source="sigma-ground (Snell's law at TIR boundary)",
        formula="theta_c = arcsin(n_rare / n_dense)",
        inputs={"n_dense": n_dense, "n_rare": n_rare},
        notes=("Glass-air (n=1.5): theta_c=41.8 deg. "
                "Water-air (n=1.33): theta_c=48.6 deg. "
                "Diamond-air (n=2.42): theta_c=24.4 deg (why diamonds sparkle)."),
    )


def thin_lens_focal_length(object_distance_m: float,
                             image_distance_m: float) -> ToolResult:
    """1/f = 1/d_o + 1/d_i. Solve for focal length given d_o and d_i.

    Sign convention: positive distances are 'real' (object on incoming
    side, image on outgoing side). Virtual images use negative d_i.
    """
    if object_distance_m == 0 or image_distance_m == 0:
        return ToolResult(value=None, source="invalid input",
                           notes="distances must be non-zero",
                           inputs={"object_distance_m": object_distance_m,
                                   "image_distance_m": image_distance_m})
    inv_f = 1.0/object_distance_m + 1.0/image_distance_m
    if inv_f == 0:
        return ToolResult(value=float("inf"), source="sigma-ground (thin lens)",
                           notes="parallel rays (1/d_o + 1/d_i = 0)",
                           inputs={"object_distance_m": object_distance_m,
                                   "image_distance_m": image_distance_m})
    f = 1.0/inv_f
    return ToolResult(
        value=f,
        units="m",
        source="sigma-ground (thin lens equation)",
        formula="1/f = 1/d_o + 1/d_i",
        inputs={"object_distance_m": object_distance_m,
                "image_distance_m": image_distance_m},
        notes=("Positive f = converging lens. Negative f = diverging lens. "
                "Thin-lens approximation: lens thickness << f."),
    )


def thin_lens_image_distance(object_distance_m: float,
                               focal_length_m: float) -> ToolResult:
    """1/d_i = 1/f - 1/d_o. Image distance given object distance and focal length."""
    if object_distance_m == 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"object_distance_m": object_distance_m,
                                   "focal_length_m": focal_length_m})
    inv_di = 1.0/focal_length_m - 1.0/object_distance_m
    if inv_di == 0:
        return ToolResult(value=float("inf"),
                           source="sigma-ground (thin lens; object at focal point)",
                           inputs={"object_distance_m": object_distance_m,
                                   "focal_length_m": focal_length_m})
    di = 1.0/inv_di
    return ToolResult(
        value=di,
        units="m",
        source="sigma-ground (thin lens equation)",
        formula="d_i = 1 / (1/f - 1/d_o)",
        inputs={"object_distance_m": object_distance_m,
                "focal_length_m": focal_length_m},
        notes=("Negative d_i = virtual image (same side as object). "
                "Positive d_i = real image (opposite side)."),
    )


def lens_magnification(object_distance_m: float,
                         image_distance_m: float) -> ToolResult:
    """Magnification m = -d_i / d_o. Negative = inverted image."""
    if object_distance_m == 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"object_distance_m": object_distance_m,
                                   "image_distance_m": image_distance_m})
    m = -image_distance_m / object_distance_m
    return ToolResult(
        value=m,
        units="dimensionless",
        source="sigma-ground (thin lens magnification)",
        formula="m = -d_i / d_o",
        inputs={"object_distance_m": object_distance_m,
                "image_distance_m": image_distance_m},
        notes=("|m|>1: enlarged. |m|<1: reduced. m<0: inverted. m>0: upright."),
    )


def rydberg_hydrogen_wavelength(n_initial: int, n_final: int) -> ToolResult:
    """Wavelength of hydrogen emission/absorption: 1/lambda = R_inf (1/n_f^2 - 1/n_i^2).

    For emission, n_i > n_f. For absorption, n_i < n_f (returns negative
    wavelength magnitude, take abs).

    R_infinity = 1.0973731568160e7 /m (CODATA 2018).
    """
    if not isinstance(n_initial, int) or not isinstance(n_final, int):
        return ToolResult(value=None, source="invalid input",
                           notes="quantum numbers must be integers",
                           inputs={"n_initial": n_initial,
                                   "n_final": n_final})
    if n_initial < 1 or n_final < 1 or n_initial == n_final:
        return ToolResult(value=None, source="invalid input",
                           notes="need positive integer n_i, n_f, and n_i != n_f",
                           inputs={"n_initial": n_initial,
                                   "n_final": n_final})
    R_inf = 1.0973731568160e7  # 1/m
    inv_lambda = R_inf * abs(1.0/n_final**2 - 1.0/n_initial**2)
    lam = 1.0 / inv_lambda
    # Series identification
    series_map = {1: "Lyman (UV)", 2: "Balmer (visible)", 3: "Paschen (IR)",
                   4: "Brackett (IR)", 5: "Pfund (IR)"}
    n_low = min(n_initial, n_final)
    series = series_map.get(n_low, "higher (IR/radio)")
    return ToolResult(
        value=lam,
        units="m",
        source="sigma-ground (Rydberg formula, CODATA 2018 R_inf)",
        formula="1/lambda = R_inf |1/n_f^2 - 1/n_i^2|",
        inputs={"n_initial": n_initial, "n_final": n_final},
        notes=(f"{series} series. Notable lines: H-alpha (n=3->2) = 656.3 nm; "
                f"Lyman-alpha (n=2->1) = 121.6 nm. For hydrogen-like ions "
                f"(He+, Li2+), multiply 1/lambda by Z^2."),
    )


def double_slit_fringe_spacing(wavelength_m: float, slit_separation_m: float,
                                 screen_distance_m: float) -> ToolResult:
    """Fringe spacing on screen: y = lambda L / d.

    Young's double-slit interference pattern.
    """
    if wavelength_m <= 0 or slit_separation_m <= 0 or screen_distance_m <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"wavelength_m": wavelength_m,
                                   "slit_separation_m": slit_separation_m,
                                   "screen_distance_m": screen_distance_m})
    y = wavelength_m * screen_distance_m / slit_separation_m
    return ToolResult(
        value=y,
        units="m",
        source="sigma-ground (Young's double-slit, far-field)",
        formula="y = lambda L / d",
        inputs={"wavelength_m": wavelength_m,
                "slit_separation_m": slit_separation_m,
                "screen_distance_m": screen_distance_m},
        notes=("Spacing between adjacent bright fringes. Far-field "
                "(Fraunhofer) approximation: L >> d. For 633 nm laser, "
                "d=0.1 mm slits, L=1 m: y=6.33 mm."),
    )


def single_slit_first_minimum_angle(wavelength_m: float,
                                      slit_width_m: float) -> ToolResult:
    """First-minimum diffraction angle: sin(theta) = lambda / a."""
    if wavelength_m <= 0 or slit_width_m <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"wavelength_m": wavelength_m,
                                   "slit_width_m": slit_width_m})
    if wavelength_m > slit_width_m:
        return ToolResult(value=None, source="diffraction limit",
                           notes="wavelength > slit width; no first minimum",
                           inputs={"wavelength_m": wavelength_m,
                                   "slit_width_m": slit_width_m})
    theta = math.degrees(math.asin(wavelength_m / slit_width_m))
    return ToolResult(
        value=theta,
        units="deg",
        source="sigma-ground (single-slit Fraunhofer diffraction)",
        formula="sin(theta) = lambda / a",
        inputs={"wavelength_m": wavelength_m,
                "slit_width_m": slit_width_m},
        notes=("Half-width of the central diffraction maximum. For visible "
                "light through 0.1 mm slit: ~0.4 deg. For radio waves "
                "through a dish, this sets the beamwidth."),
    )


def diffraction_grating_angle(wavelength_m: float, grating_spacing_m: float,
                                order: int = 1) -> ToolResult:
    """Diffraction grating maximum angle: d sin(theta) = m lambda."""
    if wavelength_m <= 0 or grating_spacing_m <= 0 or order < 1:
        return ToolResult(value=None, source="invalid input",
                           inputs={"wavelength_m": wavelength_m,
                                   "grating_spacing_m": grating_spacing_m,
                                   "order": order})
    sin_theta = order * wavelength_m / grating_spacing_m
    if abs(sin_theta) > 1:
        return ToolResult(value=None, source="no diffraction maximum",
                           notes=f"order {order} not possible with these params",
                           inputs={"wavelength_m": wavelength_m,
                                   "grating_spacing_m": grating_spacing_m,
                                   "order": order})
    theta = math.degrees(math.asin(sin_theta))
    return ToolResult(
        value=theta,
        units="deg",
        source="sigma-ground (diffraction grating)",
        formula="d sin(theta) = m lambda",
        inputs={"wavelength_m": wavelength_m,
                "grating_spacing_m": grating_spacing_m, "order": order},
    )
