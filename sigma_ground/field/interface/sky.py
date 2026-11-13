"""
Clear-sky radiance — what a mirror (a pool, a pane, a polished floor) shows of the sky.

Physics chain (a renderer looks the answer up; it computes none of this):

  Sun         E0(λ) = π B(λ, T_sun) (R_sun / AU)²                    blackbody Sun at the top of the atmosphere
  Air         n(λ) → σ_R(λ) = 24π³ ((n²−1)/(n²+2))² F / (λ⁴ N_s²)   Rayleigh cross-section per molecule
              N(z) = N0 exp(−z/H),  H from atmosphere.scale_height
  Aerosol     β_a(λ) = β_a(550) (λ/550 nm)^−α  (Ångström), N_a(z) = exp(−z/H_a); scattering fraction ω0 of the
              extinction (the rest is absorbed), phase Henyey-Greenstein with asymmetry g (the forward peak: the
              halo round the sun, the whiter horizon)
  Scattering  L(view) = ∫ [β_R p_R(θ) + ω0 β_a p_a(θ)] E0 exp(−τ_sun(P) − τ_view(P)) ds,  p_R = 3/(16π)(1+cos²θ)
              single scattering along the view ray, τ the sum of air and aerosol extinction; the sun ray from each
              sample point is blocked by the Earth's shadow (a spherical Earth), so the sky goes through dusk and then dark
  Multiple    the first-order sky and the ground give a mean intensity at every altitude; the air scatters it once more (taken
              isotropic): L_ms = ∫ σ_s(P) Ī(z(P)) exp(−τ_view) ds. Second order only, plus the ground's light scattered once.
  Ground      Lambertian, albedo ρ_g: L_g = ρ_g (E_direct/π + ⟨L_sky⟩)  (⟨L_sky⟩ the cosine-weighted mean of the whole sky,
              which includes the light the ground itself sent up and the air sent back: solved in closed form)
  Glint       a horizontal sheet of water is a field of small tilted mirrors (Cox & Munk slopes for the wind, Smith masking); the sun
              reaches the eye off the facets that bisect it and the eye: L = E_perp F(w) P G / (4 cos(theta_v) cos^4(b)).
              ``water_mirror`` folds the sky's reflection and the glint into the colour and alpha of a water face.
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
  - Water glint: the slope statistics are MEASURED (Cox & Munk 1954/1956, a clean surface), Smith's masking is FIRST_PRINCIPLES
    geometry for Gaussian slopes, the Fresnel reflectance is the library's (optics.fresnel_interface, water's Cauchy index); the
    wind over the water is an EMPIRICAL-INPUT (the world has no weather)
  - Ground albedo: EMPIRICAL-INPUT (a typical land surface; the renderer does not know what the ground is made of)
  - Aerosol (optical depth, Ångström exponent, scale height, single-scattering albedo, asymmetry): EMPIRICAL-INPUT, the
    typical clear continental air (AERONET climatology). The library holds no aerosol composition or size distribution,
    so a Mie calculation is not possible: the phase function is the Henyey-Greenstein fit to one, not derived.

Not in it (SKY_GAPS): ozone, third and higher orders of scattering, clouds and rain, the sun's disc (a flat mirror's glint), the moon and
stars, an observer above sea level, a real solar spectrum; and the aerosol is one typical air, not this place's today.
"""

import functools
import math

from ..constants import AU_M, K_B
from . import atmosphere as _atm
from . import optics as _optics
from .optics import LAMBDA_B, LAMBDA_G, LAMBDA_R, srgb_encode
from .thermal_emission import planck_spectral_radiance

# ── Inputs (each tagged) ─────────────────────────────────────────────────
R_EARTH_M = 6.371e6          # MEASURED: IAU 2015 nominal mean Earth radius
R_SUN_M = 6.957e8            # MEASURED: IAU 2015 Resolution B3 nominal solar radius
T_SUN_K = 5772.0             # MEASURED: IAU 2015 Resolution B3 nominal effective temperature
TOP_M = 100.0e3              # convention: the Kármán line; the air above it is under 1e-5 of the surface density
KING_F = 1.048               # MEASURED: Bates (1984), the depolarisation correction for air at 550 nm
GROUND_ALBEDO = 0.25         # EMPIRICAL-INPUT: typical land surface albedo (the mean over Earth's land is 0.25-0.3)
AEROSOL_OD_550 = 0.10        # EMPIRICAL-INPUT: aerosol optical depth at 550 nm, clear continental air (AERONET: 0.05-0.15 on a clear day)
ANGSTROM_EXPONENT = 1.3      # EMPIRICAL-INPUT: Ångström (1929) exponent, the AERONET global mean ~1.3
AEROSOL_SCALE_HEIGHT_M = 1.2e3   # EMPIRICAL-INPUT: aerosols sit in the boundary layer (Elterman 1968 ~1.2 km)
AEROSOL_SSA = 0.92           # EMPIRICAL-INPUT: single-scattering albedo of continental aerosol at 550 nm (the rest of the extinction is absorbed)
AEROSOL_G = 0.70             # EMPIRICAL-INPUT: Henyey-Greenstein asymmetry of continental aerosol at 550 nm (0.65-0.75)
WIND_M_S = 3.0               # EMPIRICAL-INPUT: the wind over the water, a light breeze (Beaufort 2); the world has no weather of its own
COX_MUNK_BASE = 0.003        # MEASURED: Cox & Munk (1954, 1956), the mean-square slope of a clean sea surface at no wind
COX_MUNK_PER_M_S = 0.00512   # MEASURED: Cox & Munk (1956), the growth of that slope with wind speed (m/s)
GLINT_MIN_ELEVATION_DEG = 0.25   # the table's horizon row is evaluated a quarter degree up: at 0 the glint is 0 / 0
T_AIR_K = _atm._T_STANDARD   # the standard atmosphere the index was measured in
P_AIR_PA = _atm._P_STANDARD

WAVELENGTHS_M = (LAMBDA_R, LAMBDA_G, LAMBDA_B)

# The table's axes (degrees). Denser where the sky changes fastest: near the horizon, near the sun, and around dusk.
SUN_ELEVATION_DEG = (-18.0, -12.0, -8.0, -4.0, 0.0, 2.0, 5.0, 10.0, 15.0, 25.0, 40.0, 60.0, 90.0)
VIEW_ELEVATION_DEG = (0.0, 1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 30.0, 40.0, 52.0, 65.0, 78.0, 90.0)
RELATIVE_AZIMUTH_DEG = (0.0, 10.0, 20.0, 35.0, 55.0, 80.0, 110.0, 140.0, 180.0)
# The water's table has its own, finer axes: the glint is a lobe a few degrees wide, narrower than the sky's grid.
WATER_SUN_ELEVATION_DEG = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0, 90.0)
WATER_VIEW_ELEVATION_DEG = (0.0, 1.0, 2.0, 3.0, 4.0, 5.5, 7.0, 9.0, 11.0, 13.5, 16.0, 19.0, 22.0, 26.0, 30.0, 35.0, 40.0, 46.0,
                            52.0, 58.0, 65.0, 72.0, 78.0, 84.0, 90.0)
WATER_RELATIVE_AZIMUTH_DEG = (0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 110.0, 140.0, 180.0)

SKY_GAPS = (
    "No ozone: the real twilight zenith is bluer (the Chappuis band) than Rayleigh alone gives.",
    "The aerosol is one typical clear continental air with a Henyey-Greenstein phase function (empirical inputs, no Mie calculation: the library holds no aerosol composition); a hazy, smoky or maritime day is a different sky.",
    "The table's azimuth steps (10 degrees near the sun) are coarser than the real aureole: the glare round the sun is smoothed, then clipped at the white card.",
    "Multiple scattering is second order only (the air scatters the first-order sky and the ground's light once more, isotropically, with the sun's elevation taken at the observer for the whole ray): third and higher orders are missing, so the real sky is a little brighter still; near the horizon at dusk the observer's sun elevation is wrong for a distant point.",
    "No clouds or rain: an overcast sky is a different radiance.",
    "The sun's glint is drawn on water only, as the statistics of a wind-roughened surface (Cox-Munk slopes, Smith masking, the sun a point: its 0.27-degree disc is far narrower than the glint). A flat mirror (glass, still water) would show the sun's disc itself, which a colour per corner cannot; the wave field the water would really have is not simulated; one wind (WIND_M_S) is tabulated; sea foam and whitecaps are not.",
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


def _aod(aod550) -> float:
    return AEROSOL_OD_550 if aod550 is None else aod550


def aerosol_extinction_per_m(lam_m: float, aod550=None, altitude_m: float = 0.0) -> float:
    """β_a(λ, z): the aerosol extinction coefficient, 1/m. The 550 nm optical depth spread over the layer's scale height, and the
    Ångström power law in wavelength: β_a(λ) ∝ λ^−α."""
    ha = AEROSOL_SCALE_HEIGHT_M
    at_550 = _aod(aod550) / (ha * (1.0 - math.exp(-TOP_M / ha)))
    return at_550 * (lam_m / LAMBDA_G) ** (-ANGSTROM_EXPONENT) * math.exp(-altitude_m / ha)


def henyey_greenstein(cos_theta: float, g: float = AEROSOL_G) -> float:
    """Henyey-Greenstein phase function (per steradian, integrates to 1 over the sphere): the aerosol's forward peak."""
    return (1.0 - g * g) / (4.0 * math.pi * (1.0 + g * g - 2.0 * g * cos_theta) ** 1.5)


def solar_irradiance_top(lam_m: float) -> float:
    """Spectral irradiance of sunlight at the top of the atmosphere, W / (m² · m): π B(λ, T_sun) (R_sun / AU)²."""
    return math.pi * planck_spectral_radiance(lam_m, T_SUN_K) * (R_SUN_M / AU_M) ** 2


# ── Geometry on a spherical Earth ────────────────────────────────────────

def _columns(p, d, steps: int = 24):
    """(air, aerosol): ∫ exp(−h/H) ds along the ray from p in direction d (a unit vector) to the top of the atmosphere, for
    the air's scale height and for the aerosol layer's, in metres of surface-density medium; None when the Earth blocks the
    ray (the point is in the Earth's shadow)."""
    px, py, pz = p
    dx, dy, dz = d
    b = px * dx + py * dy + pz * dz
    pp = px * px + py * py + pz * pz
    if b < 0.0 and b * b - (pp - R_EARTH_M ** 2) >= 0.0:
        return None
    rt = R_EARTH_M + TOP_M
    t_max = -b + math.sqrt(max(0.0, b * b - (pp - rt * rt)))
    h = _scale_height_m()
    ha = AEROSOL_SCALE_HEIGHT_M
    air = 0.0
    aer = 0.0
    prev = 0.0
    for k in range(1, steps + 1):
        t = (k / steps) ** 3 * t_max
        mid = 0.5 * (prev + t)
        x, y, z = px + dx * mid, py + dy * mid, pz + dz * mid
        alt = math.sqrt(x * x + y * y + z * z) - R_EARTH_M
        air += math.exp(-alt / h) * (t - prev)
        aer += math.exp(-alt / ha) * (t - prev)
        prev = t
    return air, aer


def _lookup(table, altitude_m: float):
    """Linear interpolation of a per-altitude list of RGB triples on ALTITUDES_M (clamped to its ends)."""
    if altitude_m <= ALTITUDES_M[0]:
        return table[0]
    if altitude_m >= ALTITUDES_M[-1]:
        return table[-1]
    i = 0
    while ALTITUDES_M[i + 1] < altitude_m:
        i += 1
    f = (altitude_m - ALTITUDES_M[i]) / (ALTITUDES_M[i + 1] - ALTITUDES_M[i])
    lo, hi = table[i], table[i + 1]
    return (lo[0] + f * (hi[0] - lo[0]), lo[1] + f * (hi[1] - lo[1]), lo[2] + f * (hi[2] - lo[2]))


def _trace(p, d, sun, aod550, steps: int, col_steps: int = 24, mean_intensity=None):
    """March a ray from the point p (metres from the Earth's centre) in direction d to the top of the atmosphere, or to the
    ground it hits. Returns (single, ms_air, ms_ground, ground_t), each an RGB triple:

      single     first-order scattered radiance along the ray (sunlight scattered once, towards the ray's start),
                 W / (m² · sr · m)
      ms_air     the radiance from scattering the sky's own light once more (second order), when ``mean_intensity``
                 = (A, G) is given: ∫ σ_s(P) A(z(P)) exp(−τ(0→P)) ds, A the mean intensity of the first-order sky at that altitude
      ms_ground  the same with G, the fraction of the mean intensity that is light from the ground (per unit of the ground's
                 own radiance), so the ground's contribution is (ground radiance) × ms_ground
      ground_t   transmittance along the whole ray if it ends on the ground (zeros if it escapes to space): the ground's
                 own radiance reaches the start as (ground radiance) × ground_t
    """
    px, py, pz = p
    dx, dy, dz = d
    b = px * dx + py * dy + pz * dz
    pp = px * px + py * py + pz * pz
    disc = b * b - (pp - R_EARTH_M ** 2)
    hits = b < 0.0 and disc >= 0.0
    if hits:
        t_end = -b - math.sqrt(disc)
    else:
        rt = R_EARTH_M + TOP_M
        t_end = -b + math.sqrt(max(0.0, b * b - (pp - rt * rt)))
    h = _scale_height_m()
    ha = AEROSOL_SCALE_HEIGHT_M
    n0 = surface_number_density()
    sig = [rayleigh_cross_section_m2(l) * n0 for l in WAVELENGTHS_M]                 # air extinction at the surface, 1/m
    ba = [aerosol_extinction_per_m(l, aod550) for l in WAVELENGTHS_M]                # aerosol extinction at the surface, 1/m
    e0 = [solar_irradiance_top(l) for l in WAVELENGTHS_M]
    cos_theta = dx * sun[0] + dy * sun[1] + dz * sun[2]
    phase_r = 3.0 / (16.0 * math.pi) * (1.0 + cos_theta * cos_theta)
    phase_a = henyey_greenstein(cos_theta)
    single = [0.0, 0.0, 0.0]
    ms_air = [0.0, 0.0, 0.0]
    ms_gnd = [0.0, 0.0, 0.0]
    path_air = 0.0
    path_aer = 0.0
    prev = 0.0
    for k in range(1, steps + 1):
        # spacing: dense where the air is thick (the start of a ray that leaves the ground; the end of one that reaches it)
        f = k / steps
        t = t_end * (1.0 - (1.0 - f) ** 3) if hits else t_end * f ** 3
        dt = t - prev
        mid = 0.5 * (prev + t)
        x, y, z = px + dx * mid, py + dy * mid, pz + dz * mid
        alt = math.sqrt(x * x + y * y + z * z) - R_EARTH_M
        seg_air = math.exp(-alt / h) * dt
        seg_aer = math.exp(-alt / ha) * dt
        to_air = path_air + 0.5 * seg_air
        to_aer = path_aer + 0.5 * seg_aer
        sun_cols = _columns((x, y, z), sun, col_steps)
        a_z = _lookup(mean_intensity[0], alt) if mean_intensity is not None else None
        g_z = _lookup(mean_intensity[1], alt) if mean_intensity is not None else None
        for c in range(3):
            view_tau = sig[c] * to_air + ba[c] * to_aer
            if sun_cols is not None:
                tau = view_tau + sig[c] * sun_cols[0] + ba[c] * sun_cols[1]
                single[c] += (sig[c] * seg_air * phase_r + AEROSOL_SSA * ba[c] * seg_aer * phase_a) * e0[c] * math.exp(-tau)
            if a_z is not None:
                scatter = (sig[c] * seg_air + AEROSOL_SSA * ba[c] * seg_aer) * math.exp(-view_tau)    # isotropic: the 1/4π cancels the 4π
                ms_air[c] += scatter * a_z[c]
                ms_gnd[c] += scatter * g_z[c]
        path_air += seg_air
        path_aer += seg_aer
        prev = t
    ground_t = [math.exp(-(sig[c] * path_air + ba[c] * path_aer)) for c in range(3)] if hits else [0.0, 0.0, 0.0]
    return single, ms_air, ms_gnd, ground_t


# ── Multiple scattering ──────────────────────────────────────────────────
#
# The light that has been scattered once is a second source of light for the air: at every altitude the first-order sky and the
# ground give a mean intensity (the average over all directions), and the air scatters that once more. Taking that second
# scattering as isotropic (the multiply-scattered light is nearly so) the source at a point is σ_s(P) × (the mean intensity there),
# and the radiance it adds along a view ray is that integrated with the ray's own extinction. The mean intensity is tabulated
# by altitude for the sun's elevation: A (from the first-order sky) and G (the share that is the ground's light, per unit
# of the ground's radiance). The ground is Lambertian and lit by the sun and the whole sky, so its own radiance solves
#   L_g = ρ (E_direct/π + ⟨L1⟩ + ⟨ms_air⟩ + L_g ⟨ms_ground⟩)  →  L_g = ρ (E_direct/π + ⟨L1⟩ + ⟨ms_air⟩) / (1 − ρ ⟨ms_ground⟩),
# in closed form, because the two multiple-scattering terms are linear in L_g.

ALTITUDES_M = (0.0, 500.0, 1500.0, 3000.0, 5000.0, 8000.0, 12000.0, 17000.0, 24000.0, 33000.0, 45000.0, 65000.0, 100000.0)
_MU = tuple(-1.0 + (k + 0.5) * 2.0 / 20.0 for k in range(20))             # sin(elevation), uniform: equal solid angles over the sphere
_PHI_SPHERE = tuple(180.0 * (j + 0.5) / 8.0 for j in range(8))            # the sun's plane is a mirror: half the azimuths are enough
_U = tuple((k + 0.5) / 12.0 for k in range(12))          # u = sin²(elevation), uniform: the cosine-weighted hemisphere
_PHI_DEG = tuple(180.0 * (j + 0.5) / 8.0 for j in range(8))


def _direction(elevation_deg: float, azimuth_deg: float):
    e, a = math.radians(elevation_deg), math.radians(azimuth_deg)
    return (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))


def _direct_irradiance(sun_elevation_deg: float, aod550=None):
    """Direct sunlight on a horizontal surface at the ground, W / (m² · m), R, G, B (zero when the Earth hides the sun)."""
    s = math.radians(sun_elevation_deg)
    cols = _columns((0.0, 0.0, R_EARTH_M), (math.cos(s), 0.0, math.sin(s)))
    if cols is None or sun_elevation_deg <= 0.0:
        return [0.0, 0.0, 0.0]
    n0 = surface_number_density()
    return [solar_irradiance_top(l) * math.sin(s)
            * math.exp(-rayleigh_cross_section_m2(l) * n0 * cols[0] - aerosol_extinction_per_m(l, aod550) * cols[1])
            for l in WAVELENGTHS_M]


@functools.lru_cache(maxsize=256)
def _context(sun_elevation_deg: float, aod550: float, albedo: float):
    """Everything the multiple scattering needs for one sun elevation: the mean-intensity tables A and G by altitude, the
    ground's own radiance L_g, and the surface's cosine-weighted means of the first-order sky and of the two second-order terms."""
    s = math.radians(sun_elevation_deg)
    sun = (math.cos(s), 0.0, math.sin(s))
    tab_a, tab_g = [], []
    for alt in ALTITUDES_M:
        p = (0.0, 0.0, R_EARTH_M + alt)
        acc_a, acc_g = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        for mu in _MU:
            el = math.degrees(math.asin(mu))
            for ph in _PHI_SPHERE:
                single, _, _, ground_t = _trace(p, _direction(el, ph), sun, aod550, steps=16, col_steps=16)
                for c in range(3):
                    acc_a[c] += single[c]
                    acc_g[c] += ground_t[c]
        n = len(_MU) * len(_PHI_SPHERE)
        tab_a.append(tuple(v / n for v in acc_a))
        tab_g.append(tuple(v / n for v in acc_g))
    mean_i = (tab_a, tab_g)
    m1, ma, mg = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    origin = (0.0, 0.0, R_EARTH_M)
    for u in _U:
        el = math.degrees(math.asin(math.sqrt(u)))
        for ph in _PHI_DEG:
            single, ms_air, ms_gnd, _ = _trace(origin, _direction(el, ph), sun, aod550, steps=32, mean_intensity=mean_i)
            for c in range(3):
                m1[c] += single[c]
                ma[c] += ms_air[c]
                mg[c] += ms_gnd[c]
    n = len(_U) * len(_PHI_DEG)
    m1, ma, mg = [v / n for v in m1], [v / n for v in ma], [v / n for v in mg]
    direct = _direct_irradiance(sun_elevation_deg, aod550)
    lg = [albedo * (direct[c] / math.pi + m1[c] + ma[c]) / (1.0 - albedo * mg[c]) for c in range(3)]
    return mean_i, lg, m1, ma, mg


def sky_radiance(sun_elevation_deg: float, view_elevation_deg: float, relative_azimuth_deg: float, steps: int = 64,
                 aod550=None, multiple: bool = True):
    """Spectral radiance of the clear sky at R, G, B, W / (m² · sr · m), seen from the surface.

    The sun is at this elevation; the view is at its own elevation (0 horizon, 90 zenith) and at this angle round the
    horizon from the sun's azimuth. ``aod550`` is the aerosol optical depth at 550 nm (None: AEROSOL_OD_550; 0: air only).
    ``multiple`` adds the second-order scattering and the ground's light (False: sunlight scattered once, the closed-form case).
    """
    aod = _aod(aod550)
    s = math.radians(sun_elevation_deg)
    sun = (math.cos(s), 0.0, math.sin(s))
    origin = (0.0, 0.0, R_EARTH_M)
    view = _direction(view_elevation_deg, relative_azimuth_deg)
    if not multiple:
        return tuple(_trace(origin, view, sun, aod, steps)[0])
    mean_i, lg, _, _, _ = _context(round(sun_elevation_deg, 9), aod, GROUND_ALBEDO)
    single, ms_air, ms_gnd, _ = _trace(origin, view, sun, aod, steps, mean_intensity=mean_i)
    return tuple(single[c] + ms_air[c] + lg[c] * ms_gnd[c] for c in range(3))


# ── The ground and the display scale ─────────────────────────────────────

def _mean_sky(sun_elevation_deg: float, aod550=None):
    """The cosine-weighted mean sky radiance over the hemisphere, R, G, B: the diffuse irradiance is π times it. It includes the
    second-order scattering and the light the ground sends back up."""
    _, lg, m1, ma, mg = _context(round(sun_elevation_deg, 9), _aod(aod550), GROUND_ALBEDO)
    return [m1[c] + ma[c] + lg[c] * mg[c] for c in range(3)]


def white_card_radiance(aod550=None):
    """Radiance of a white (albedo 1) horizontal diffuse card in full sun at the zenith, R, G, B: the display's 1.0."""
    direct = _direct_irradiance(90.0, aod550)
    mean = _mean_sky(90.0, aod550)
    return [direct[c] / math.pi + mean[c] for c in range(3)]


def ground_radiance(sun_elevation_deg: float, albedo: float = GROUND_ALBEDO, aod550=None):
    """Radiance of a Lambertian surface of this albedo lying on the world's ground (albedo GROUND_ALBEDO), lit by the sun and
    the whole sky, R, G, B, W / (m² · sr · m)."""
    direct = _direct_irradiance(sun_elevation_deg, aod550)
    mean = _mean_sky(sun_elevation_deg, aod550)
    return [albedo * (direct[c] / math.pi + mean[c]) for c in range(3)]


# ── The sun's glint on water ─────────────────────────────────────────────
# A wind-roughened surface is a field of small tilted mirrors. Light from the sun reaches the eye only off the facets whose
# normal bisects the sun and the eye, so the glint is the sun's irradiance times the Fresnel reflectance at that facet times the
# fraction of facets tilted that way (Cox & Munk 1954), over the geometry of a mirror (Smith 1967 for the facets that hide each other):
#   L = E_perp F(w) P(tan b) G / (4 cos(theta_v) cos^4(b))     b the facet tilt, w its angle of incidence, P the slope density

def water_slope_variance(wind_m_s=None) -> float:
    """The mean-square slope of the sea surface in this wind (Cox & Munk 1956, a clean surface): 0.003 + 0.00512 W."""
    w = WIND_M_S if wind_m_s is None else wind_m_s
    return COX_MUNK_BASE + COX_MUNK_PER_M_S * max(0.0, float(w))


def _smith_lambda(cos_theta: float, slope_variance: float) -> float:
    """Smith's (1967) shadowing function for a Gaussian slope distribution: the fraction of facets hidden at this angle from the normal."""
    sin = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    if sin <= 0.0:
        return 0.0
    a = cos_theta / (sin * math.sqrt(slope_variance))
    if a <= 1e-9:
        return float("inf")
    return 0.5 * (math.exp(-a * a) / (a * math.sqrt(math.pi)) - math.erfc(a))


def sun_glint_radiance(sun_elevation_deg: float, view_elevation_deg: float, relative_azimuth_deg: float,
                       wind_m_s=None, aod550=None):
    """The sun's reflection in a horizontal sheet of wind-roughened water, R, G, B, W / (m² · sr · m), the same units as ``sky_radiance``.

    The reflected ray (the camera's ray mirrored in the surface) is at this elevation and this angle round from the sun's azimuth,
    as for the sky table. Zero with the sun below the horizon. ``wind_m_s`` None is WIND_M_S.
    """
    if sun_elevation_deg <= 0.0:
        return (0.0, 0.0, 0.0)
    s2 = water_slope_variance(wind_m_s)
    e = max(view_elevation_deg, GLINT_MIN_ELEVATION_DEG)
    sun = _direction(sun_elevation_deg, 0.0)
    r = _direction(e, relative_azimuth_deg)
    v = (-r[0], -r[1], r[2])                              # toward the eye: the reflected ray's horizontal part turned round
    h = (sun[0] + v[0], sun[1] + v[1], sun[2] + v[2])
    norm = math.sqrt(h[0] ** 2 + h[1] ** 2 + h[2] ** 2)
    cos_b = h[2] / norm                                   # the facet's tilt from the horizontal
    if cos_b <= 0.0:
        return (0.0, 0.0, 0.0)
    tan2_b = (1.0 - cos_b * cos_b) / (cos_b * cos_b)
    p = math.exp(-tan2_b / s2) / (math.pi * s2)           # the slope density (isotropic Gaussian)
    cos_v = r[2]
    cos_s = sun[2]
    cos_w = norm / 2.0                                    # the angle of incidence on the facet: s . h = |s + v| / 2
    g = 1.0 / (1.0 + _smith_lambda(cos_v, s2) + _smith_lambda(cos_s, s2))
    direct = _direct_irradiance(sun_elevation_deg, aod550)
    out = []
    for c, lam in enumerate(WAVELENGTHS_M):
        n = _optics.cauchy_n("water", lam)
        f, _ = _optics.fresnel_interface(1.0, n, cos_w)
        e_perp = direct[c] / cos_s                        # the sun's irradiance across its own beam
        out.append(e_perp * f * p * g / (4.0 * cos_v * cos_b ** 4))
    return tuple(out)


def water_mirror(sun_elevation_deg: float, view_elevation_deg: float, relative_azimuth_deg: float, white,
                 wind_m_s=None, aod550=None):
    """What a horizontal sheet of water shows, drawn as one face composited over what lies behind it: (display colour, alpha).

    The light that comes back is T = R L_sky + L_glint (R the Fresnel reflectance at the camera's own angle, L_sky the sky the
    mirror shows, L_glint the sun's reflection off the waves), as fractions of the white card. A face that blends colour c at
    alpha a over the scene gives a c + (1 - a) behind; this is the (c, a) that carries T: a = R and c = T / R while the light
    is below what the reflection alone can carry, and a raised until c reaches white once the glint outshines it.
    Without the glint it is exactly (the sky's display colour, R).
    """
    lin_sky, _ = display(sky_radiance(sun_elevation_deg, view_elevation_deg, relative_azimuth_deg, aod550=aod550), white)
    glint = sun_glint_radiance(sun_elevation_deg, view_elevation_deg, relative_azimuth_deg, wind_m_s, aod550)
    r, _ = _optics.fresnel_interface(1.0, _optics.cauchy_n("water", LAMBDA_G), math.sin(math.radians(view_elevation_deg)))
    total = [r * lin_sky[c] + glint[c] / white[c] for c in range(3)]
    alpha = max(r, min(1.0, max(total)))
    return [srgb_encode(min(1.0, t / alpha)) for t in total], alpha


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
    water_rgb, water_alpha = [], []
    for s in WATER_SUN_ELEVATION_DEG:
        rgb_s, alpha_s = [], []
        for e in WATER_VIEW_ELEVATION_DEG:
            rgb_e, alpha_e = [], []
            for a in WATER_RELATIVE_AZIMUTH_DEG:
                c, al = water_mirror(s, e, a, white)
                rgb_e.append([round(v, 4) for v in c])
                alpha_e.append(round(al, 4))
            rgb_s.append(rgb_e)
            alpha_s.append(alpha_e)
        water_rgb.append(rgb_s)
        water_alpha.append(alpha_s)
    return {
        "schema": 1,
        "wavelengths_nm": [round(l * 1e9) for l in WAVELENGTHS_M],
        "sun_elevation_deg": list(SUN_ELEVATION_DEG),
        "view_elevation_deg": list(VIEW_ELEVATION_DEG),
        "relative_azimuth_deg": list(RELATIVE_AZIMUTH_DEG),
        "sky": {"rgb": sky_lin, "rgb_display": sky_disp},
        "ground": {"albedo": GROUND_ALBEDO, "rgb": ground_lin, "rgb_display": ground_disp},
        "aerosol": {"od_550": AEROSOL_OD_550, "angstrom_exponent": ANGSTROM_EXPONENT, "scale_height_m": AEROSOL_SCALE_HEIGHT_M,
                    "single_scattering_albedo": AEROSOL_SSA, "asymmetry_g": AEROSOL_G},
        "water": {"material": "water", "sun_elevation_deg": list(WATER_SUN_ELEVATION_DEG), "view_elevation_deg": list(WATER_VIEW_ELEVATION_DEG),
                  "relative_azimuth_deg": list(WATER_RELATIVE_AZIMUTH_DEG), "wind_m_s": WIND_M_S, "mean_square_slope": round(water_slope_variance(), 6),
                  "index_550": round(_optics.cauchy_n("water", LAMBDA_G), 6),
                  "rgb_display": water_rgb, "alpha": water_alpha,
                  "meaning": "a horizontal sheet of water seen by its reflected ray: the colour (sRGB-encoded) and alpha of the face, "
                             "sky reflection plus the sun's glint on the waves, composited over what is behind it"},
        "white_card": "a white diffuse card in full sun at the zenith = 1.0 in each channel (white-balanced, fixed exposure)",
        "gaps": list(SKY_GAPS),
    }
