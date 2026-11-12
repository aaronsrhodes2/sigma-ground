"""
Clear-sky radiance — what a mirror (a pool, a pane, a polished floor) shows of the sky.

Physics chain (a renderer looks the answer up; it computes none of this):

  Sun         E0(λ) = π B(λ, T_sun) (R_sun / AU)²                    blackbody Sun at the top of the atmosphere
  Air         n(λ) → σ_R(λ) = 24π³ ((n²−1)/(n²+2))² F / (λ⁴ N_s²)   Rayleigh cross-section per molecule
              N(z) = N0 exp(−z/H),  H from atmosphere.scale_height
  Scattering  L(view) = ∫ β(P) p(θ) E0 exp(−τ_sun(P) − τ_view(P)) ds,  p = 3/(16π)(1+cos²θ)
              single scattering along the view ray; the sun ray from each sample point is blocked by the
              Earth's shadow (a spherical Earth), so the sky goes through dusk and then dark
  Ground      Lambertian, albedo ρ_g: L_g = ρ_g (E_direct/π + ⟨L_sky⟩)  (⟨L_sky⟩ the cosine-weighted mean of the sky)
  Display     the table is white-balanced to a white diffuse card in full sun at the zenith (1.0 = that card); sRGB-encoded
              by optics.srgb_encode. This is a fixed exposure, like a camera left on one setting: the sky is dim at dusk
              and dark at night because the scene is, not because anything was fitted.

The three colour samples are the library's own (optics.LAMBDA_R/G/B = 650/550/450 nm).

Origin tags:
  - Rayleigh scattering: FIRST_PRINCIPLES (Rayleigh 1871; Lorentz-Lorenz polarisability from the measured index)
  - n_air(λ): MEASURED (Peck & Reeder 1972, standard air at 15 °C, 101325 Pa)
  - King correction F: MEASURED (Bates 1984, air at 550 nm)
  - Sun: blackbody at the IAU 2015 nominal T_eff and R_sun (a real solar spectrum has Fraunhofer lines, a measured one is not used)
  - Earth radius, top of atmosphere: MEASURED / convention (IAU nominal mean radius; 100 km, the Kármán line)
  - Ground albedo: EMPIRICAL-INPUT (a typical land surface; the renderer does not know what the ground is made of)

Not in it (SKY_GAPS): ozone, aerosols and Mie scattering (no sun halo, no haze), multiple scattering, clouds and rain, the
sun's disc (a mirror's sun glint), the moon and stars, an observer above sea level, a real solar spectrum.
"""

import math

from ..constants import AU_M, K_B
from . import atmosphere as _atm
from .optics import LAMBDA_B, LAMBDA_G, LAMBDA_R, srgb_encode
from .thermal_emission import planck_spectral_radiance

# ── Inputs (each tagged) ─────────────────────────────────────────────────
R_EARTH_M = 6.371e6          # MEASURED: IAU 2015 nominal mean Earth radius
R_SUN_M = 6.957e8            # MEASURED: IAU 2015 Resolution B3 nominal solar radius
T_SUN_K = 5772.0             # MEASURED: IAU 2015 Resolution B3 nominal effective temperature
TOP_M = 100.0e3              # convention: the Kármán line; the air above it is under 1e-5 of the surface density
KING_F = 1.048               # MEASURED: Bates (1984), the depolarisation correction for air at 550 nm
GROUND_ALBEDO = 0.25         # EMPIRICAL-INPUT: typical land surface albedo (the mean over Earth's land is 0.25-0.3)
T_AIR_K = _atm._T_STANDARD   # the standard atmosphere the index was measured in
P_AIR_PA = _atm._P_STANDARD

WAVELENGTHS_M = (LAMBDA_R, LAMBDA_G, LAMBDA_B)

# The table's axes (degrees). Denser where the sky changes fastest: near the horizon, near the sun, and around dusk.
SUN_ELEVATION_DEG = (-18.0, -12.0, -8.0, -4.0, 0.0, 2.0, 5.0, 10.0, 15.0, 25.0, 40.0, 60.0, 90.0)
VIEW_ELEVATION_DEG = (0.0, 1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 30.0, 40.0, 52.0, 65.0, 78.0, 90.0)
RELATIVE_AZIMUTH_DEG = (0.0, 10.0, 20.0, 35.0, 55.0, 80.0, 110.0, 140.0, 180.0)

SKY_GAPS = (
    "No ozone: the real twilight zenith is bluer (the Chappuis band) than Rayleigh alone gives.",
    "No aerosols or Mie scattering: no haze, no halo round the sun, a horizon bluer than a real one.",
    "Single scattering only: the real sky is a little brighter, and bluer in the shade, from multiple scattering and the ground bounce.",
    "No clouds or rain: an overcast sky is a different radiance.",
    "The sun's disc is not in the table: a mirror's sun glint is not drawn.",
    "No moon or stars: after dusk the table is dark.",
    "The observer is at sea level; the sun is a blackbody at T_eff, not a measured spectrum.",
    "Three colour samples (650, 550, 450 nm), not a spectrum.",
)

_H = None


def _scale_height_m() -> float:
    global _H
    if _H is None:
        _H = _atm.scale_height(T_AIR_K)
    return _H


def surface_number_density() -> float:
    """Air molecules per m³ at the surface of the standard atmosphere, P / (k T)."""
    return P_AIR_PA / (K_B * T_AIR_K)


def air_refractive_index(lam_m: float) -> float:
    """Refractive index of standard dry air (15 °C, 101325 Pa) at a vacuum wavelength.

    MEASURED (Peck & Reeder 1972): (n − 1) × 1e8 = 5791817 / (238.0185 − s²) + 167909 / (57.362 − s²), s = 1/λ in µm⁻¹.
    """
    s2 = (1.0e-6 / lam_m) ** 2
    return 1.0 + (5791817.0 / (238.0185 - s2) + 167909.0 / (57.362 - s2)) * 1.0e-8


def rayleigh_cross_section_m2(lam_m: float) -> float:
    """Rayleigh scattering cross-section of one air molecule (m²).

    FIRST_PRINCIPLES (Rayleigh; Lorentz-Lorenz): σ = 24π³ ((n²−1)/(n²+2))² F / (λ⁴ N_s²), N_s the number density at which n is measured.
    """
    n = air_refractive_index(lam_m)
    ns = surface_number_density()
    return 24.0 * math.pi ** 3 * ((n * n - 1.0) / (n * n + 2.0)) ** 2 * KING_F / (lam_m ** 4 * ns * ns)


def rayleigh_extinction_per_m(lam_m: float, altitude_m: float = 0.0) -> float:
    """β(λ, z): the Rayleigh extinction (and scattering: air does not absorb here) coefficient at an altitude, 1/m."""
    return rayleigh_cross_section_m2(lam_m) * surface_number_density() * math.exp(-altitude_m / _scale_height_m())


def vertical_optical_depth(lam_m: float) -> float:
    """Rayleigh optical depth of the whole atmosphere straight up from the surface."""
    h = _scale_height_m()
    return rayleigh_extinction_per_m(lam_m) * h * (1.0 - math.exp(-TOP_M / h))


def solar_irradiance_top(lam_m: float) -> float:
    """Spectral irradiance of sunlight at the top of the atmosphere, W / (m² · m): π B(λ, T_sun) (R_sun / AU)²."""
    return math.pi * planck_spectral_radiance(lam_m, T_SUN_K) * (R_SUN_M / AU_M) ** 2


# ── Geometry on a spherical Earth ────────────────────────────────────────

def _column(p, d, steps: int = 24):
    """∫ exp(−h/H) ds along the ray from p in direction d (a unit vector) to the top of the atmosphere, in metres of
    surface-density air; None when the Earth blocks the ray (the point is in the Earth's shadow)."""
    px, py, pz = p
    dx, dy, dz = d
    b = px * dx + py * dy + pz * dz
    pp = px * px + py * py + pz * pz
    if b < 0.0 and b * b - (pp - R_EARTH_M ** 2) >= 0.0:
        return None
    rt = R_EARTH_M + TOP_M
    t_max = -b + math.sqrt(max(0.0, b * b - (pp - rt * rt)))
    h = _scale_height_m()
    total = 0.0
    prev = 0.0
    for k in range(1, steps + 1):
        t = (k / steps) ** 3 * t_max
        mid = 0.5 * (prev + t)
        x, y, z = px + dx * mid, py + dy * mid, pz + dz * mid
        total += math.exp(-(math.sqrt(x * x + y * y + z * z) - R_EARTH_M) / h) * (t - prev)
        prev = t
    return total


def sky_radiance(sun_elevation_deg: float, view_elevation_deg: float, relative_azimuth_deg: float, steps: int = 64):
    """Single-scattering spectral radiance of the clear sky at R, G, B, W / (m² · sr · m), seen from the surface.

    The sun is at this elevation; the view is at its own elevation (0 horizon, 90 zenith) and at this angle round the
    horizon from the sun's azimuth.
    """
    s = math.radians(sun_elevation_deg)
    e = math.radians(view_elevation_deg)
    ph = math.radians(relative_azimuth_deg)
    sun = (math.cos(s), 0.0, math.sin(s))
    view = (math.cos(e) * math.cos(ph), math.cos(e) * math.sin(ph), math.sin(e))
    cos_theta = sun[0] * view[0] + sun[1] * view[1] + sun[2] * view[2]
    phase = 3.0 / (16.0 * math.pi) * (1.0 + cos_theta * cos_theta)
    origin = (0.0, 0.0, R_EARTH_M)
    rt = R_EARTH_M + TOP_M
    b = origin[2] * view[2]
    t_max = -b + math.sqrt(max(0.0, b * b - (R_EARTH_M ** 2 - rt * rt)))
    h = _scale_height_m()
    n0 = surface_number_density()
    sig = [rayleigh_cross_section_m2(l) for l in WAVELENGTHS_M]
    e0 = [solar_irradiance_top(l) for l in WAVELENGTHS_M]
    out = [0.0, 0.0, 0.0]
    col_view = 0.0
    prev = 0.0
    for k in range(1, steps + 1):
        t = (k / steps) ** 3 * t_max
        dt = t - prev
        mid = 0.5 * (prev + t)
        p = (view[0] * mid, view[1] * mid, R_EARTH_M + view[2] * mid)
        dens = math.exp(-(math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) - R_EARTH_M) / h)
        col_sun = _column(p, sun)
        seg = dens * dt
        col_to_mid = col_view + 0.5 * seg
        if col_sun is not None:
            for c in range(3):
                out[c] += sig[c] * n0 * seg * phase * e0[c] * math.exp(-sig[c] * n0 * (col_to_mid + col_sun))
        col_view += seg
        prev = t
    return tuple(out)


# ── The ground and the display scale ─────────────────────────────────────

_U = tuple((k + 0.5) / 12.0 for k in range(12))          # u = sin²(elevation), uniform: the cosine-weighted hemisphere
_PHI_DEG = tuple(180.0 * (j + 0.5) / 8.0 for j in range(8))


def _mean_sky(sun_elevation_deg: float):
    """The cosine-weighted mean sky radiance over the hemisphere, R, G, B: the diffuse irradiance is π times it."""
    acc = [0.0, 0.0, 0.0]
    for u in _U:
        el = math.degrees(math.asin(math.sqrt(u)))
        for ph in _PHI_DEG:
            L = sky_radiance(sun_elevation_deg, el, ph, steps=32)
            for c in range(3):
                acc[c] += L[c]
    n = len(_U) * len(_PHI_DEG)
    return [a / n for a in acc]


def _direct_irradiance(sun_elevation_deg: float):
    """Direct sunlight on a horizontal surface at the ground, W / (m² · m), R, G, B (zero when the Earth hides the sun)."""
    s = math.radians(sun_elevation_deg)
    col = _column((0.0, 0.0, R_EARTH_M), (math.cos(s), 0.0, math.sin(s)))
    if col is None or sun_elevation_deg <= 0.0:
        return [0.0, 0.0, 0.0]
    n0 = surface_number_density()
    return [solar_irradiance_top(l) * math.sin(s) * math.exp(-rayleigh_cross_section_m2(l) * n0 * col) for l in WAVELENGTHS_M]


def white_card_radiance():
    """Radiance of a white (albedo 1) horizontal diffuse card in full sun at the zenith, R, G, B: the display's 1.0."""
    direct = _direct_irradiance(90.0)
    mean = _mean_sky(90.0)
    return [direct[c] / math.pi + mean[c] for c in range(3)]


def ground_radiance(sun_elevation_deg: float, albedo: float = GROUND_ALBEDO):
    """Radiance of a Lambertian ground lit by the sun and the sky, R, G, B, W / (m² · sr · m)."""
    direct = _direct_irradiance(sun_elevation_deg)
    mean = _mean_sky(sun_elevation_deg)
    return [albedo * (direct[c] / math.pi + mean[c]) for c in range(3)]


def display(radiance, white):
    """A radiance as (linear, sRGB-encoded) RGB against the white card: [0, 1] each, clipped (the sun's glare saturates)."""
    lin = [max(0.0, min(1.0, radiance[c] / white[c])) for c in range(3)]
    return lin, [srgb_encode(v) for v in lin]


def sky_record() -> dict:
    """The sky as a table for a renderer: display colours of the sky by sun elevation, view elevation and azimuth from the
    sun, and of the ground by sun elevation. Everything physical was computed here; a renderer only interpolates."""
    white = white_card_radiance()
    sky_lin, sky_disp = [], []
    for s in SUN_ELEVATION_DEG:
        lin_s, disp_s = [], []
        for e in VIEW_ELEVATION_DEG:
            lin_e, disp_e = [], []
            for a in RELATIVE_AZIMUTH_DEG:
                lin, disp = display(sky_radiance(s, e, a), white)
                lin_e.append([round(v, 6) for v in lin])
                disp_e.append([round(v, 4) for v in disp])
            lin_s.append(lin_e)
            disp_s.append(disp_e)
        sky_lin.append(lin_s)
        sky_disp.append(disp_s)
    ground_lin, ground_disp = [], []
    for s in SUN_ELEVATION_DEG:
        lin, disp = display(ground_radiance(s), white)
        ground_lin.append([round(v, 6) for v in lin])
        ground_disp.append([round(v, 4) for v in disp])
    return {
        "schema": 1,
        "wavelengths_nm": [round(l * 1e9) for l in WAVELENGTHS_M],
        "sun_elevation_deg": list(SUN_ELEVATION_DEG),
        "view_elevation_deg": list(VIEW_ELEVATION_DEG),
        "relative_azimuth_deg": list(RELATIVE_AZIMUTH_DEG),
        "sky": {"rgb": sky_lin, "rgb_display": sky_disp},
        "ground": {"albedo": GROUND_ALBEDO, "rgb": ground_lin, "rgb_display": ground_disp},
        "white_card": "a white diffuse card in full sun at the zenith = 1.0 in each channel (white-balanced, fixed exposure)",
        "gaps": list(SKY_GAPS),
    }
