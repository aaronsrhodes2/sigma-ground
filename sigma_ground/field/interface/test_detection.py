"""
Tests for detection.py -- thermal and reflected-sunlight passive detection
range physics, the sensor-sensitivity catalog, and the albedo reference
catalog.

Plain pytest functions (not unittest.TestCase), matching
sigma_ground/field/interface/test_propulsion.py and test_ore_abundance.py
style within this repo's own test suite.
"""
import math

import pytest

from sigma_ground.field.constants import AU_M, L_SUN_W
from sigma_ground.field.interface.detection import (
    AlbedoReference,
    ALBEDO_CATALOG,
    Sensor,
    SENSOR_CATALOG,
    combined_detection_report,
    detection_range_reflected,
    detection_range_thermal,
    incident_solar_flux_w_m2,
    reflected_power_w,
    thermal_radiated_power_w,
)
from sigma_ground.field.interface.thermal import thermal_emission_power


# ═══════════════════════════════════════════════════════════════════════════
# § 1. incident_solar_flux_w_m2
# ═══════════════════════════════════════════════════════════════════════════

def test_incident_solar_flux_at_1au_matches_the_real_solar_constant():
    # The real solar constant at 1 AU is ~1361 W/m^2 -- a strong external
    # sanity check that L_SUN_W and AU_M are wired correctly.
    flux = incident_solar_flux_w_m2(AU_M)
    assert 1360.0 < flux < 1362.0


def test_incident_solar_flux_inverse_square_scaling():
    # Hand-checkable invariant: flux at 2x distance is EXACTLY 1/4.
    d = 3.0 * AU_M
    f1 = incident_solar_flux_w_m2(d)
    f2 = incident_solar_flux_w_m2(2.0 * d)
    assert f2 == pytest.approx(f1 / 4.0, rel=1e-9)


def test_incident_solar_flux_hand_check():
    # F = L / (4*pi*D^2), a direct hand computation
    d = 1.0e11
    expected = L_SUN_W / (4.0 * math.pi * d ** 2)
    assert incident_solar_flux_w_m2(d) == pytest.approx(expected, rel=1e-12)


def test_incident_solar_flux_rejects_nonpositive_distance():
    with pytest.raises(ValueError):
        incident_solar_flux_w_m2(0.0)
    with pytest.raises(ValueError):
        incident_solar_flux_w_m2(-1.0)


# ═══════════════════════════════════════════════════════════════════════════
# § 2. thermal_radiated_power_w
# ═══════════════════════════════════════════════════════════════════════════

def test_thermal_radiated_power_hand_check_against_thermal_emission_power():
    r = 1000.0
    p = thermal_radiated_power_w(r, "iron", 500.0, 0.0)
    expected = thermal_emission_power("iron", 500.0, 0.0) * 4.0 * math.pi * r ** 2
    assert p == pytest.approx(expected, rel=1e-12)


def test_thermal_radiated_power_scales_with_surface_area():
    # Doubling radius quadruples the radiating surface area (4*pi*r^2)
    p1 = thermal_radiated_power_w(1000.0, "granite", 400.0, 0.0)
    p2 = thermal_radiated_power_w(2000.0, "granite", 400.0, 0.0)
    assert p2 == pytest.approx(4.0 * p1, rel=1e-9)


def test_thermal_radiated_power_zero_at_zero_temperature():
    # thermal_emission_power floors at T<=0 -> 0.0 (thermal.py's own floor)
    assert thermal_radiated_power_w(1000.0, "iron", 0.0, 0.0) == 0.0


def test_thermal_radiated_power_rejects_nonpositive_radius():
    with pytest.raises(ValueError):
        thermal_radiated_power_w(0.0, "iron", 300.0, 0.0)
    with pytest.raises(ValueError):
        thermal_radiated_power_w(-5.0, "iron", 300.0, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# § 3. detection_range_thermal (and the shared inverse-square inversion)
# ═══════════════════════════════════════════════════════════════════════════

def test_detection_range_thermal_hand_check_round_numbers():
    # R = sqrt(P / (4*pi*F_min)); pick P = 4*pi so R = sqrt(1/F_min)
    p = 4.0 * math.pi
    f_min = 1.0
    assert detection_range_thermal(p, f_min) == pytest.approx(1.0, rel=1e-9)

    f_min2 = 0.25
    assert detection_range_thermal(p, f_min2) == pytest.approx(2.0, rel=1e-9)


def test_detection_range_thermal_scales_as_sqrt_power():
    f_min = 1.0e-10
    r1 = detection_range_thermal(1.0e6, f_min)
    r2 = detection_range_thermal(4.0e6, f_min)
    assert r2 == pytest.approx(2.0 * r1, rel=1e-9)


def test_detection_range_thermal_zero_power_gives_zero_range():
    assert detection_range_thermal(0.0, 1.0e-10) == 0.0


def test_detection_range_thermal_rejects_nonpositive_sensor_threshold():
    with pytest.raises(ValueError):
        detection_range_thermal(1.0e6, 0.0)
    with pytest.raises(ValueError):
        detection_range_thermal(1.0e6, -1.0)


def test_detection_range_thermal_round_trips_into_a_recoverable_flux():
    # R_max is defined so F(R_max) == F_min exactly -- verify the round trip
    p = 5.0e9
    f_min = 2.3e-15
    r = detection_range_thermal(p, f_min)
    recovered_flux = p / (4.0 * math.pi * r ** 2)
    assert recovered_flux == pytest.approx(f_min, rel=1e-9)


# ═══════════════════════════════════════════════════════════════════════════
# § 4. reflected_power_w
# ═══════════════════════════════════════════════════════════════════════════

def test_reflected_power_hand_check():
    r = 500.0
    albedo = 0.2
    d = 2.0 * AU_M
    expected = incident_solar_flux_w_m2(d) * math.pi * r ** 2 * albedo
    assert reflected_power_w(r, albedo, d) == pytest.approx(expected, rel=1e-12)


def test_reflected_power_scales_linearly_with_albedo():
    p1 = reflected_power_w(1000.0, 0.1, AU_M)
    p2 = reflected_power_w(1000.0, 0.2, AU_M)
    assert p2 == pytest.approx(2.0 * p1, rel=1e-9)


def test_reflected_power_zero_albedo_gives_zero_power():
    assert reflected_power_w(1000.0, 0.0, AU_M) == 0.0


def test_reflected_power_rejects_bad_inputs():
    with pytest.raises(ValueError):
        reflected_power_w(0.0, 0.1, AU_M)
    with pytest.raises(ValueError):
        reflected_power_w(1000.0, -0.1, AU_M)
    with pytest.raises(ValueError):
        reflected_power_w(1000.0, 1.1, AU_M)
    with pytest.raises(ValueError):
        reflected_power_w(1000.0, 0.1, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# § 5. detection_range_reflected
# ═══════════════════════════════════════════════════════════════════════════

def test_detection_range_reflected_matches_thermal_formula_for_equal_inputs():
    # Both channels share the exact same inverse-square inversion once
    # power and threshold are known -- verify they agree bit-for-bit.
    p = 7.5e10
    f_min = 4.0e-14
    assert detection_range_reflected(p, f_min) == pytest.approx(
        detection_range_thermal(p, f_min), rel=1e-12)


def test_detection_range_reflected_zero_power_gives_zero_range():
    assert detection_range_reflected(0.0, 1.0e-10) == 0.0


def test_detection_range_reflected_rejects_nonpositive_sensor_threshold():
    with pytest.raises(ValueError):
        detection_range_reflected(1.0e6, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# § 6. Sensor catalog -- every conversion hand-checked against its cited
#      source figure
# ═══════════════════════════════════════════════════════════════════════════

REQUIRED_SENSOR_CATEGORIES = {
    "human_eye", "optical_telescope", "space_infrared_survey",
}


def test_sensor_catalog_covers_the_required_minimum_categories():
    have = {s.category for s in SENSOR_CATALOG.values()}
    assert REQUIRED_SENSOR_CATEGORIES <= have


def test_sensor_catalog_entries_are_all_facts_with_sources():
    for key, sensor in SENSOR_CATALOG.items():
        assert sensor.provenance == "fact", key
        assert sensor.source, f"{key} missing a source citation"
        assert sensor.limiting_flux_w_m2 > 0.0, key


def test_naked_eye_threshold_hand_check_from_first_principles():
    # Independent recomputation from the raw published figures, not a
    # reference back into the module's own private constants:
    #   V-band zero point (Fukugita, Shimasaku & Ichikawa 1995, PASP
    #   107:945): f_lambda(V=0) = 3.60e-9 erg/s/cm^2/A = 3.6e-11 W/m^2/nm,
    #   FWHM = 82.7 nm. Limiting magnitude m_V=6.5 (Curtis 1901, quoted in
    #   Crumey 2014). Pogson relation F(m) = F(0) * 10^(-m/2.5).
    v_zero_w_m2_nm = 3.6e-11
    fwhm_nm = 82.7
    f_zero_broadband = v_zero_w_m2_nm * fwhm_nm
    m_v = 6.5
    expected = f_zero_broadband * 10.0 ** (-m_v / 2.5)

    actual = SENSOR_CATALOG["naked_eye"].limiting_flux_w_m2
    assert actual == pytest.approx(expected, rel=1e-9)
    # sanity band around the cited ~7.5e-12 W/m^2 headline figure
    assert 7.0e-12 < actual < 8.0e-12


def test_telescope_threshold_is_exactly_1000x_fainter_than_naked_eye():
    # NexStar 8SE limiting mag 14.0 vs naked-eye 6.5: a difference of
    # exactly 7.5 magnitudes = 10^(7.5/2.5) = 10^3 = 1000x, by the Pogson
    # definition of the magnitude scale -- independent of the absolute
    # zero-point calibration.
    naked_eye = SENSOR_CATALOG["naked_eye"].limiting_flux_w_m2
    telescope = SENSOR_CATALOG["amateur_telescope_8in"].limiting_flux_w_m2
    assert telescope == pytest.approx(naked_eye / 1000.0, rel=1e-9)


def test_wise_w1_threshold_hand_check_from_first_principles():
    # Independent recomputation: 0.08 mJy 5-sigma sensitivity (Wright et
    # al. 2010, AJ 140:1868, Abstract) -> Jy -> SI spectral flux density
    # (1 Jy = 1e-26 W/m^2/Hz, definitional) -> broadband W/m^2 via the
    # effective bandwidth (~1.82e13 Hz, derived from the official W1
    # relative-spectral-response curve).
    mjy = 0.08
    jy_to_w_m2_hz = 1.0e-26
    s_nu_w_m2_hz = mjy * 1.0e-3 * jy_to_w_m2_hz
    assert s_nu_w_m2_hz == pytest.approx(8.0e-31, rel=1e-9)

    bandwidth_hz = 1.82e13
    expected = s_nu_w_m2_hz * bandwidth_hz

    actual = SENSOR_CATALOG["wise_w1"].limiting_flux_w_m2
    assert actual == pytest.approx(expected, rel=1e-9)
    # sanity band around the cited ~1.46e-17 W/m^2 headline figure
    assert 1.4e-17 < actual < 1.5e-17


def test_wise_w1_is_vastly_more_sensitive_than_optical_sensors():
    # A space IR survey's broadband threshold should be many orders of
    # magnitude below either optical sensor's -- physically sane ordering.
    wise = SENSOR_CATALOG["wise_w1"].limiting_flux_w_m2
    naked_eye = SENSOR_CATALOG["naked_eye"].limiting_flux_w_m2
    telescope = SENSOR_CATALOG["amateur_telescope_8in"].limiting_flux_w_m2
    assert wise < telescope < naked_eye


def test_sensor_rejects_bad_provenance():
    with pytest.raises(ValueError):
        Sensor(category="human_eye", name="x", limiting_flux_w_m2=1.0e-10,
               provenance="guess", source="none")


def test_sensor_rejects_nonpositive_flux():
    with pytest.raises(ValueError):
        Sensor(category="human_eye", name="x", limiting_flux_w_m2=0.0,
               provenance="fact", source="none")


# ═══════════════════════════════════════════════════════════════════════════
# § 7. Albedo reference catalog
# ═══════════════════════════════════════════════════════════════════════════

def test_albedo_catalog_entries_are_all_facts_with_sources():
    for key, ref in ALBEDO_CATALOG.items():
        assert ref.provenance == "fact", key
        assert ref.source, f"{key} missing a source citation"
        assert 0.0 <= ref.geometric_albedo <= 1.0, key


def test_albedo_catalog_planet_figures_match_cited_research():
    assert ALBEDO_CATALOG["venus"].geometric_albedo == pytest.approx(0.689)
    assert ALBEDO_CATALOG["mars"].geometric_albedo == pytest.approx(0.170)
    assert ALBEDO_CATALOG["jupiter"].geometric_albedo == pytest.approx(0.538)
    assert ALBEDO_CATALOG["saturn"].geometric_albedo == pytest.approx(0.499)
    assert ALBEDO_CATALOG["moon"].geometric_albedo == pytest.approx(0.12)


def test_albedo_catalog_taxonomy_figures_match_cited_research():
    # Mainzer et al. (2011) ApJ 741, 90, Table 1 median pV values
    assert ALBEDO_CATALOG["c_type"].geometric_albedo == pytest.approx(0.055)
    assert ALBEDO_CATALOG["s_type"].geometric_albedo == pytest.approx(0.210)
    assert ALBEDO_CATALOG["m_type"].geometric_albedo == pytest.approx(0.125)


def test_albedo_catalog_taxonomy_ordering_c_dark_s_bright():
    # C-type ("dark, carbonaceous") must be dimmer than S-type ("moderate,
    # siliceous") -- the defining physical distinction between the classes.
    assert (ALBEDO_CATALOG["c_type"].geometric_albedo
            < ALBEDO_CATALOG["m_type"].geometric_albedo
            < ALBEDO_CATALOG["s_type"].geometric_albedo)


def test_albedo_catalog_m_type_within_its_own_definitional_range():
    m = ALBEDO_CATALOG["m_type"]
    lo, hi = m.albedo_range
    assert lo <= m.geometric_albedo <= hi
    assert lo == pytest.approx(0.10)
    assert hi == pytest.approx(0.30)


def test_albedo_reference_rejects_out_of_range_albedo():
    with pytest.raises(ValueError):
        AlbedoReference(category="planet", name="x", geometric_albedo=1.5,
                        provenance="fact", source="none")
    with pytest.raises(ValueError):
        AlbedoReference(category="planet", name="x", geometric_albedo=-0.1,
                        provenance="fact", source="none")


# ═══════════════════════════════════════════════════════════════════════════
# § 8. combined_detection_report
# ═══════════════════════════════════════════════════════════════════════════

def test_combined_report_rejects_unknown_sensor():
    with pytest.raises(ValueError):
        combined_detection_report(1000.0, "iron", 300.0, 0.0, 0.1, AU_M,
                                  "not_a_real_sensor")


def test_combined_report_dominant_channel_matches_the_larger_range():
    report = combined_detection_report(
        50000.0, "granite", 40.0, 0.0, 0.055, 40.0 * AU_M, "wise_w1")
    if report["thermal_detection_range_m"] > report["reflected_detection_range_m"]:
        assert report["dominant_channel"] == "thermal"
    elif report["reflected_detection_range_m"] > report["thermal_detection_range_m"]:
        assert report["dominant_channel"] == "reflected"
    else:
        assert report["dominant_channel"] == "tie"


def test_combined_report_hot_large_nearby_beats_cold_small_distant():
    # A large, hot, nearby, bright body vs. a small, cold, distant, dark
    # one -- both channels' detection range should be far greater for the
    # first (physically sane ordering, derived from the actual formulas,
    # not asserted by construction).
    hot_bright_near = combined_detection_report(
        200000.0, "granite", 800.0, 0.0, 0.5, 1.0 * AU_M, "wise_w1")
    cold_dark_far = combined_detection_report(
        500.0, "granite", 40.0, 0.0, 0.05, 40.0 * AU_M, "wise_w1")

    assert (hot_bright_near["thermal_detection_range_m"]
            > cold_dark_far["thermal_detection_range_m"])
    assert (hot_bright_near["reflected_detection_range_m"]
            > cold_dark_far["reflected_detection_range_m"])


def test_combined_report_cold_dark_distant_body_favors_thermal_channel():
    # The real, well-documented observational-astronomy effect (Mainzer et
    # al. 2014; Stansberry et al. 2008 -- see module docstring): a cold,
    # dark, distant body's thermal-IR channel outperforms its reflected-
    # visible channel, because thermal emission depends only weakly on
    # albedo while reflected power scales directly with it -- a dark body
    # absorbs (and must eventually re-radiate) most of the sunlight it
    # intercepts instead of reflecting it.
    #
    # A 50 km, 40K, C-type-albedo (0.055) body at 40 AU (Kuiper belt):
    report = combined_detection_report(
        radius_m=50000.0, material_key="granite", T=40.0, sigma=0.0,
        albedo=ALBEDO_CATALOG["c_type"].geometric_albedo,
        heliocentric_distance_m=40.0 * AU_M, sensor_key="wise_w1")

    assert report["thermal_power_w"] > report["reflected_power_w"]
    assert (report["thermal_detection_range_m"]
            > report["reflected_detection_range_m"])
    assert report["dominant_channel"] == "thermal"


def test_combined_report_fields_are_internally_consistent():
    report = combined_detection_report(
        10000.0, "iron", 250.0, 0.0, 0.3, 2.0 * AU_M, "amateur_telescope_8in")

    expected_thermal_power = thermal_radiated_power_w(10000.0, "iron", 250.0, 0.0)
    expected_reflected_power = reflected_power_w(10000.0, 0.3, 2.0 * AU_M)
    sensor_flux = SENSOR_CATALOG["amateur_telescope_8in"].limiting_flux_w_m2

    assert report["thermal_power_w"] == pytest.approx(expected_thermal_power)
    assert report["reflected_power_w"] == pytest.approx(expected_reflected_power)
    assert report["thermal_detection_range_m"] == pytest.approx(
        detection_range_thermal(expected_thermal_power, sensor_flux))
    assert report["reflected_detection_range_m"] == pytest.approx(
        detection_range_reflected(expected_reflected_power, sensor_flux))
