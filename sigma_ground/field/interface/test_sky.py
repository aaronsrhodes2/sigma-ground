"""Tests for the clear-sky radiance model (sky.py): the pieces against measured values, the closed form where there is one,
and the table a renderer reads."""
import math

import pytest

from sigma_ground.field.interface import sky
from sigma_ground.field.interface.optics import srgb_encode


def test_air_index_matches_the_measured_value():
    # Peck & Reeder / Edlen: standard air at 550 nm has n - 1 = 2.778e-4
    assert 2.77e-4 < sky.air_refractive_index(550e-9) - 1 < 2.79e-4


def test_rayleigh_extinction_and_optical_depth_at_550nm():
    # Bodhaine et al. (1999): beta(550 nm) ~ 1.15e-5 /m at the surface; the vertical optical depth of a standard atmosphere ~ 0.097
    assert 1.10e-5 < sky.rayleigh_extinction_per_m(550e-9) < 1.20e-5
    assert sky.vertical_optical_depth(550e-9) == pytest.approx(0.097, rel=0.04)


def test_rayleigh_follows_the_inverse_fourth_power_of_wavelength():
    # The index is nearly flat, so sigma goes as lambda^-4 (a little steeper: dispersion)
    ratio = sky.rayleigh_cross_section_m2(450e-9) / sky.rayleigh_cross_section_m2(650e-9)
    assert (650 / 450) ** 4 < ratio < (650 / 450) ** 4 * 1.05


def test_zenith_radiance_matches_the_plane_parallel_closed_form():
    # (air only) Sun and view both at the zenith: the sun-ray and view-ray optical depths add to the whole column at every height, so
    # L = E0 p(0) tau exp(-tau), p(0) = 3/(8 pi). The spherical Earth moves it by well under 2%.
    L = sky.sky_radiance(90.0, 90.0, 0.0, aod550=0.0, multiple=False)          # air only, scattered once: the closed form
    for c, lam in enumerate(sky.WAVELENGTHS_M):
        tau = sky.vertical_optical_depth(lam)
        want = sky.solar_irradiance_top(lam) * 3.0 / (8.0 * math.pi) * tau * math.exp(-tau)
        assert L[c] == pytest.approx(want, rel=0.02)


def test_the_midday_sky_is_blue():
    white = sky.white_card_radiance()
    lin, _ = sky.display(sky.sky_radiance(60.0, 60.0, 90.0), white)
    assert lin[2] > lin[1] > lin[0] > 0.0
    assert lin[2] / lin[0] > 1.8                        # aerosols whiten it: air alone is more than 3 (next lines)
    pure, _ = sky.display(sky.sky_radiance(60.0, 60.0, 90.0, aod550=0.0), sky.white_card_radiance(0.0))
    assert pure[2] / pure[0] > 3.0


def test_the_sky_reddens_toward_a_low_sun_and_is_dark_after_dusk():
    white = sky.white_card_radiance()
    noon, _ = sky.display(sky.sky_radiance(60.0, 60.0, 90.0), white)
    dusk, _ = sky.display(sky.sky_radiance(0.0, 2.0, 10.0), white)
    assert dusk[0] / dusk[2] > 5.0 > 1.0 > noon[0] / noon[2]          # the sunset sky is red-orange, the noon sky is not
    assert dusk[0] > dusk[1] > dusk[2]
    night, _ = sky.display(sky.sky_radiance(-18.0, 60.0, 0.0), white)
    assert max(night) < 1e-3


def test_the_earths_shadow_darkens_the_sky_through_twilight():
    white = sky.white_card_radiance()
    levels = [sum(sky.display(sky.sky_radiance(s, 40.0, 90.0), white)[0]) for s in (10.0, 0.0, -4.0, -8.0, -12.0)]
    assert all(a > b for a, b in zip(levels, levels[1:]))
    assert levels[-1] < 0.01 * levels[0]


def test_the_white_card_is_one_by_definition_and_the_ground_scales_with_albedo():
    white = sky.white_card_radiance()
    card = sky.ground_radiance(90.0, albedo=1.0)
    assert all(card[c] / white[c] == pytest.approx(1.0, rel=1e-9) for c in range(3))
    noon, _ = sky.display(sky.ground_radiance(90.0), white)
    assert noon == pytest.approx([sky.GROUND_ALBEDO] * 3, rel=1e-9)
    low, _ = sky.display(sky.ground_radiance(10.0), white)
    assert sum(low) < sum(noon)
    night, _ = sky.display(sky.ground_radiance(-8.0), white)
    assert max(night) < 1e-3


def test_display_clips_and_encodes():
    lin, enc = sky.display([2.0, 1.0, 0.0], [1.0, 1.0, 1.0])
    assert lin == [1.0, 1.0, 0.0] and enc == pytest.approx([1.0, 1.0, 0.0])
    lin, enc = sky.display([0.2, 0.2, 0.2], [1.0, 1.0, 1.0])
    assert enc[0] == pytest.approx(srgb_encode(0.2))


def test_the_table_is_complete_in_range_and_consistent():
    rec = sky.sky_record()
    S, V, A = len(sky.SUN_ELEVATION_DEG), len(sky.VIEW_ELEVATION_DEG), len(sky.RELATIVE_AZIMUTH_DEG)
    assert rec["wavelengths_nm"] == [650, 550, 450]
    for key in ("rgb", "rgb_display"):
        t = rec["sky"][key]
        assert len(t) == S and all(len(x) == V for x in t) and all(len(y) == A for x in t for y in x)
        assert all(len(c) == 3 and all(0.0 <= v <= 1.0 for v in c) for x in t for y in x for c in y)
        g = rec["ground"][key]
        assert len(g) == S and all(len(c) == 3 and all(0.0 <= v <= 1.0 for v in c) for c in g)
    for s in range(S):
        for v in range(V):
            for a in range(A):
                for c in range(3):
                    assert rec["sky"]["rgb_display"][s][v][a][c] == pytest.approx(srgb_encode(rec["sky"]["rgb"][s][v][a][c]), abs=2e-4)
    assert rec["gaps"] == list(sky.SKY_GAPS)


# ── Aerosol ──────────────────────────────────────────────────────────────

def test_aerosol_optical_depth_and_angstrom_law():
    ha = sky.AEROSOL_SCALE_HEIGHT_M
    # the 550 nm column integrates back to the optical depth the layer was given
    tau = sky.aerosol_extinction_per_m(550e-9) * ha * (1.0 - math.exp(-sky.TOP_M / ha))
    assert tau == pytest.approx(sky.AEROSOL_OD_550, rel=1e-9)
    # Angstrom: extinction goes as wavelength^-alpha
    ratio = sky.aerosol_extinction_per_m(450e-9) / sky.aerosol_extinction_per_m(650e-9)
    assert ratio == pytest.approx((650 / 450) ** sky.ANGSTROM_EXPONENT, rel=1e-9)
    assert sky.aerosol_extinction_per_m(550e-9, aod550=0.0) == 0.0
    assert sky.aerosol_extinction_per_m(550e-9, altitude_m=ha) == pytest.approx(sky.aerosol_extinction_per_m(550e-9) / math.e)


def test_henyey_greenstein_is_a_normalised_forward_peaked_phase_function():
    n = 4000
    total = mean_cos = 0.0
    for i in range(n):                       # integrate over cos(theta) in (-1, 1): d(omega) = 2 pi d(cos)
        mu = -1.0 + (i + 0.5) * 2.0 / n
        w = sky.henyey_greenstein(mu) * 2.0 * math.pi * 2.0 / n
        total += w
        mean_cos += w * mu
    assert total == pytest.approx(1.0, rel=2e-3)
    assert mean_cos == pytest.approx(sky.AEROSOL_G, rel=5e-3)           # the asymmetry parameter is the mean cosine
    assert sky.henyey_greenstein(1.0) > 50 * sky.henyey_greenstein(-1.0)


def test_aerosol_dims_the_direct_sun_by_its_optical_depth():
    clear = sky._direct_irradiance(90.0, aod550=0.0)
    hazy = sky._direct_irradiance(90.0)
    assert hazy[1] / clear[1] == pytest.approx(math.exp(-sky.AEROSOL_OD_550), rel=0.02)      # at 550 nm, straight overhead
    assert hazy[2] / clear[2] < hazy[0] / clear[0] < 1.0                                       # blue is dimmed more (Angstrom)


def test_aerosol_brightens_and_whitens_the_sky_and_puts_a_halo_round_the_sun():
    w0, w1 = sky.white_card_radiance(0.0), sky.white_card_radiance()
    clear = sky.display(sky.sky_radiance(60.0, 90.0, 0.0, aod550=0.0), w0)[0]
    hazy = sky.display(sky.sky_radiance(60.0, 90.0, 0.0), w1)[0]
    assert all(h > c for h, c in zip(hazy, clear))                     # single scattering only adds light
    assert hazy[2] / hazy[0] < clear[2] / clear[0]                     # and the sky is less blue
    # the horizon goes white-grey: a clear-air horizon is bluer than a hazy one
    c_h = sky.display(sky.sky_radiance(60.0, 2.0, 100.0, aod550=0.0), w0)[0]
    h_h = sky.display(sky.sky_radiance(60.0, 2.0, 100.0), w1)[0]
    assert h_h[2] / h_h[0] < c_h[2] / c_h[0]
    # the halo: the sky 15 degrees from the sun is far brighter than the sky 90 degrees away, more so with aerosol
    near = sky.sky_radiance(40.0, 40.0, 20.0)[1]
    far = sky.sky_radiance(40.0, 40.0, 140.0)[1]
    near0 = sky.sky_radiance(40.0, 40.0, 20.0, aod550=0.0)[1]
    far0 = sky.sky_radiance(40.0, 40.0, 140.0, aod550=0.0)[1]
    assert near / far > 2.0 * (near0 / far0)


def test_aerosol_shifts_the_table_and_is_recorded():
    rec = sky.sky_record()
    a = rec["aerosol"]
    assert a["od_550"] == sky.AEROSOL_OD_550 and a["asymmetry_g"] == sky.AEROSOL_G
    # high noon zenith, in the shipped table: brighter than the air-only sky (about 0.03 of a white card in green)
    s = list(sky.SUN_ELEVATION_DEG).index(90.0)
    v = list(sky.VIEW_ELEVATION_DEG).index(90.0)
    assert rec["sky"]["rgb"][s][v][0][1] > 0.05


# ── Multiple scattering ──────────────────────────────────────────────────

def test_multiple_scattering_adds_light_and_more_of_the_blue():
    for sun, view, az in ((60.0, 90.0, 0.0), (60.0, 30.0, 180.0), (40.0, 20.0, 100.0)):
        one = sky.sky_radiance(sun, view, az, multiple=False)
        many = sky.sky_radiance(sun, view, az)
        assert all(m > o for m, o in zip(many, one))
    # in clean air (no aerosol, which is grey) the air scatters blue more, so the blue gains more by the second order
    one = sky.sky_radiance(60.0, 90.0, 0.0, aod550=0.0, multiple=False)
    many = sky.sky_radiance(60.0, 90.0, 0.0, aod550=0.0)
    assert many[2] / one[2] > many[0] / one[0]
    # the second order is a fraction of the first, not a rival to it (the air is thin): between 5% and 60% at the zenith
    one = sky.sky_radiance(60.0, 90.0, 0.0, multiple=False)[1]
    assert 1.05 < sky.sky_radiance(60.0, 90.0, 0.0)[1] / one < 1.6


def test_light_from_the_ground_is_all_of_the_lower_half_at_the_surface():
    mean_i, _, _, _, _ = sky._context(60.0, sky.AEROSOL_OD_550, sky.GROUND_ALBEDO)
    assert mean_i[1][0] == pytest.approx([0.5, 0.5, 0.5], abs=1e-9)      # standing on it: half the sphere is the ground, no air between
    # from 100 km up the ground fills a smaller part of the sphere and the air between dims it, blue more than red
    top = mean_i[1][-1]
    assert 0.1 < top[2] < top[1] < top[0] < 0.5


def test_the_ground_radiance_solves_its_own_balance():
    # L_g = rho (E_direct/pi + mean sky), the mean sky including the light the ground itself sends up and the air sends back
    _, lg, _, _, _ = sky._context(60.0, sky.AEROSOL_OD_550, sky.GROUND_ALBEDO)
    assert sky.ground_radiance(60.0) == pytest.approx(lg, rel=1e-9)


def test_a_brighter_ground_brightens_the_sky_above_it(monkeypatch):
    base = sky.sky_radiance(60.0, 30.0, 180.0)
    monkeypatch.setattr(sky, "GROUND_ALBEDO", 0.9)               # snow
    snow = sky.sky_radiance(60.0, 30.0, 180.0)
    assert all(s > 1.1 * b for s, b in zip(snow, base))
    monkeypatch.setattr(sky, "GROUND_ALBEDO", 0.0)               # a black ground still has the sky's second order
    dark = sky.sky_radiance(60.0, 30.0, 180.0)
    one = sky.sky_radiance(60.0, 30.0, 180.0, multiple=False)
    assert all(o < d < b for o, d, b in zip(one, dark, base))


def test_twilight_is_dominated_by_multiple_scattering_in_the_earths_shadow():
    # sun 8 degrees down: the air overhead is in the Earth's shadow, so the single-scattered sunlight is almost nothing and what is
    # left is the second order from the lit air near the horizon
    one = sky.sky_radiance(-8.0, 90.0, 0.0, multiple=False)
    many = sky.sky_radiance(-8.0, 90.0, 0.0)
    assert all(m > 3.0 * o for m, o in zip(many, one))


# ── The sun's glint on water ─────────────────────────────────────────────

def test_the_slope_variance_is_the_measured_cox_munk_law():
    assert sky.water_slope_variance(0.0) == pytest.approx(0.003)
    assert sky.water_slope_variance(10.0) == pytest.approx(0.003 + 0.0512)
    assert sky.water_slope_variance() == pytest.approx(0.003 + 0.00512 * sky.WIND_M_S)


def _hemisphere_reflectance(sun_el, **kw):
    """The glint summed over every direction the eye can be in (cos theta_v dOmega), against the sun's irradiance on the water."""
    n_el, n_az = 90, 180
    total = 0.0
    for i in range(n_el):
        e = (i + 0.5) * 90.0 / n_el
        for j in range(n_az):
            a = (j + 0.5) * 360.0 / n_az
            total += (sky.sun_glint_radiance(sun_el, e, a, **kw)[1] * math.sin(math.radians(e)) * math.cos(math.radians(e))
                      * math.radians(90.0 / n_el) * math.radians(360.0 / n_az))
    return total / sky._direct_irradiance(sun_el)[1]


def test_the_glint_conserves_energy_against_the_fresnel_reflectance():
    # all the sunlight the waves send up equals the water's reflectance at the sun's own angle (a smooth sheet sends it all one way, a
    # rough one spreads it); facets hiding each other (Smith) take a little at a low sun, never add any
    from sigma_ground.field.interface.optics import cauchy_n, fresnel_interface
    for sun in (25.0, 40.0, 70.0):
        f, _ = fresnel_interface(1.0, cauchy_n("water", sky.LAMBDA_G), math.sin(math.radians(sun)))
        assert _hemisphere_reflectance(sun) == pytest.approx(f, rel=0.12)
    f5, _ = fresnel_interface(1.0, cauchy_n("water", sky.LAMBDA_G), math.sin(math.radians(5.0)))
    assert 0.3 * f5 < _hemisphere_reflectance(5.0) < 1.05 * f5


def test_the_glint_peaks_where_the_mirror_points_at_the_sun_and_widens_with_wind():
    on = sky.sun_glint_radiance(30.0, 30.0, 0.0)[1]               # the reflected ray is the sun's own direction
    away = sky.sun_glint_radiance(30.0, 30.0, 40.0)[1]
    assert on > 20 * away > 0
    calm = sky.sun_glint_radiance(30.0, 30.0, 0.0, wind_m_s=0.0)[1]
    gale = sky.sun_glint_radiance(30.0, 30.0, 0.0, wind_m_s=12.0)[1]
    assert calm > on > gale                                       # a rougher sea spreads the same light over more directions
    assert sky.sun_glint_radiance(30.0, 30.0, 40.0, wind_m_s=12.0)[1] > sky.sun_glint_radiance(30.0, 30.0, 40.0, wind_m_s=0.0)[1]


def test_no_glint_without_the_sun_or_on_the_far_side_of_it():
    assert sky.sun_glint_radiance(-5.0, 30.0, 0.0) == (0.0, 0.0, 0.0)
    assert sky.sun_glint_radiance(0.0, 30.0, 0.0) == (0.0, 0.0, 0.0)
    # looking away from the sun (reflected ray opposite): no facet can tilt that far
    assert sky.sun_glint_radiance(30.0, 30.0, 180.0)[1] < 1e-6 * sky.sun_glint_radiance(30.0, 30.0, 0.0)[1]


def test_the_glint_is_the_suns_own_colour_reddened_by_its_path():
    # the Fresnel reflectance barely depends on colour, so the glint is the sun's light after its way through the air: nearly white
    # high up, warmer as the sun lowers (against the white card, which is balanced for the zenith sun)
    white = sky.white_card_radiance()
    high = [g / w for g, w in zip(sky.sun_glint_radiance(70.0, 70.0, 0.0), white)]
    low = [g / w for g, w in zip(sky.sun_glint_radiance(10.0, 10.0, 0.0), white)]
    assert high[2] / high[1] > 0.85
    assert low[0] > low[1] > low[2] and low[2] / low[1] < high[2] / high[1]


def test_water_mirror_without_a_glint_is_the_sky_at_the_fresnel_alpha(monkeypatch):
    from sigma_ground.field.interface.optics import cauchy_n, fresnel_interface
    white = sky.white_card_radiance()
    # sun below the horizon: no glint, so the colour is the sky's display colour and the alpha the reflectance
    colour, alpha = sky.water_mirror(-8.0, 40.0, 60.0, white)
    _, disp = sky.display(sky.sky_radiance(-8.0, 40.0, 60.0), white)
    assert colour == pytest.approx(disp, abs=1e-9)
    r, _ = fresnel_interface(1.0, cauchy_n("water", sky.LAMBDA_G), math.sin(math.radians(40.0)))
    assert alpha == pytest.approx(r)


def test_water_mirror_carries_the_glint_as_light_over_the_scene():
    white = sky.white_card_radiance()
    lin_sky, _ = sky.display(sky.sky_radiance(30.0, 30.0, 0.0), white)
    glint = sky.sun_glint_radiance(30.0, 30.0, 0.0)
    colour, alpha = sky.water_mirror(30.0, 30.0, 0.0, white)
    # the light that comes back (a * c in linear light) is R L_sky + L_glint while it fits under 1, and the face grows to carry it
    from sigma_ground.field.interface.optics import cauchy_n, fresnel_interface
    r, _ = fresnel_interface(1.0, cauchy_n("water", sky.LAMBDA_G), math.sin(math.radians(30.0)))
    total = [r * lin_sky[c] + glint[c] / white[c] for c in range(3)]
    assert alpha >= r
    lin_colour = [_srgb_decode(v) for v in colour]
    for c in range(3):
        assert alpha * lin_colour[c] == pytest.approx(min(total[c], alpha), rel=1e-3, abs=1e-4)
    assert max(total) > r                                          # at the mirror angle the glint outshines the plain reflection


def _srgb_decode(v):
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def test_the_table_carries_the_water_and_stays_on_its_axes():
    rec = sky.sky_record()
    w = rec["water"]
    assert w["wind_m_s"] == sky.WIND_M_S
    ns, nv, na = len(w["sun_elevation_deg"]), len(w["view_elevation_deg"]), len(w["relative_azimuth_deg"])
    assert len(w["rgb_display"]) == ns and len(w["rgb_display"][0]) == nv and len(w["rgb_display"][0][0]) == na
    assert len(w["alpha"]) == ns and len(w["alpha"][0]) == nv and len(w["alpha"][0][0]) == na
    flat = [a for s in w["alpha"] for e in s for a in e]
    assert all(0.0 < a <= 1.0 for a in flat)
    # the grid really contains the glint: some face at some angle is lifted above its plain Fresnel alpha
    i, j = w["sun_elevation_deg"].index(25.0), w["view_elevation_deg"].index(26.0)
    assert w["alpha"][i][j][0] > w["alpha"][i][j][-1]


def test_the_glint_survives_the_tables_interpolation_between_grid_points():
    # a renderer interpolates the table linearly (colour and alpha apiece); against a dark scene the screen shows alpha * colour.
    # Between grid points that stays close to the direct value: the median error is small, the worst (a lobe's edge) is bounded
    import random
    white = sky.white_card_radiance()
    axes = (sky.WATER_SUN_ELEVATION_DEG, sky.WATER_VIEW_ELEVATION_DEG, sky.WATER_RELATIVE_AZIMUTH_DEG)
    cache = {}

    def at(pt):
        if pt not in cache:
            cache[pt] = sky.water_mirror(*pt, white)
        return cache[pt]

    rng = random.Random(3)
    errors = []
    for _ in range(40):
        x = (rng.uniform(3, 80), rng.uniform(3, 80), rng.uniform(0, 40))
        brackets = []
        for ax, v in zip(axes, x):
            k = max(i for i in range(len(ax) - 1) if ax[i] <= v)
            brackets.append((ax[k], ax[k + 1], (v - ax[k]) / (ax[k + 1] - ax[k])))
        colour, alpha = 0.0, 0.0
        for corner in range(8):
            w, pt = 1.0, []
            for (lo, hi, f), bit in zip(brackets, ((corner >> 2) & 1, (corner >> 1) & 1, corner & 1)):
                w *= f if bit else 1 - f
                pt.append(hi if bit else lo)
            c, al = at(tuple(pt))
            colour += w * c[1]
            alpha += w * al
        c, al = sky.water_mirror(*x, white)
        errors.append(abs(alpha * colour - al * c[1]) / max(al * c[1], 1e-3))
    errors.sort()
    assert errors[len(errors) // 2] < 0.06
    assert errors[-1] < 0.5
