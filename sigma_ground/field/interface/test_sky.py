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
    L = sky.sky_radiance(90.0, 90.0, 0.0, aod550=0.0)          # air only: the closed form has no aerosol
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
