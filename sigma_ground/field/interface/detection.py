"""
Detection — passive-sensor range physics for thermal (IR self-emission) and
reflected-sunlight (visible) signatures.

STANDALONE physics/optics module (P0c of the game-engine plan): the
blackbody/emissivity/geometric-cross-section machinery already in this
codebase (thermal.py's thermal_emission_power, surface.py's MATERIALS) is
wired only for RENDERING glow -- there was no "how far away could a real or
notional sensor see this object" function anywhere. This module is that
function, for two passive detection channels, plus a small cited catalog of
real sensor sensitivities and real body/taxonomy albedos to exercise it
against.

Physics chain
-------------
  1. THERMAL (infrared self-emission) channel.
     thermal_emission_power(material_key, T, sigma) [thermal.py] gives the
     Stefan-Boltzmann flux P/A = eps * sigma_SB * T^4 (W/m^2) radiated by the
     body's surface. Multiplying by the body's radiating surface area
     (4*pi*r^2 for a sphere) gives total radiated power:
       P_thermal = thermal_emission_power(...) * 4*pi*r^2      [W]
     FIRST_PRINCIPLES: Stefan-Boltzmann law (exact, from Planck's law),
     already established in thermal.py -- this module only adds the
     spherical-surface-area multiplication.
     SIMPLIFIED_MODEL (isotropic emission): treating that power as radiated
     equally in all directions (a point/spherical isotropic emitter) is the
     standard first-order treatment, but real bodies are not perfectly
     isotropic emitters -- day/night thermal asymmetry (a slowly-rotating
     body's sunlit hemisphere runs hotter than its night side, the basis of
     the real Yarkovsky effect) means the true angular emission pattern
     depends on rotation state, thermal inertia, and viewing geometry. This
     module deliberately does not model that: it is out of scope for a
     range-order-of-magnitude detection calculation and would require a
     per-body thermal-inertia model this codebase does not have.
     SIMPLIFIED_MODEL (bolometric power vs. a bandpass-limited sensor
     threshold): thermal_radiated_power_w is BOLOMETRIC -- Stefan-Boltzmann
     integrates the Planck spectrum over ALL wavelengths. Every
     SENSOR_CATALOG entry, by contrast, is BAND-LIMITED (V-band, FWHM 82.7nm,
     for naked_eye/amateur_telescope_8in; the WISE W1 3.4-micron band for
     wise_w1). Feeding a bolometric power straight into detection_range_thermal
     against a band-limited F_min implicitly assumes ALL of that power falls
     inside the sensor's passband -- true only near the source's Wien-peak
     wavelength (wien_peak_wavelength in thermal.py) and increasingly false
     the further the two diverge. This is a minor effect for wise_w1 (a
     mid-IR sensor evaluated against thermal-IR emitters, roughly the right
     regime) but becomes badly wrong for the naked_eye/amateur_telescope_8in
     V-band thresholds applied to cold bodies: a 40K body's Wien peak is deep
     in the far infrared, so in reality an unfiltered visual sensor would
     register essentially none of its thermal emission, no matter how the
     bolometric total compares numerically to the V-band F_min. This module
     does not implement a Planck-spectrum band-fraction correction (it would
     need the full spectral-radiance machinery, out of scope for this
     order-of-magnitude pass) -- callers pairing detection_range_thermal /
     combined_detection_report with an optical (V-band) sensor for cold
     bodies should treat the resulting range as a loose upper bound, not an
     achievable one.

  2. REFLECTED-SUNLIGHT (visible) channel.
     Incident solar flux at the body's heliocentric distance D, via
     inverse-square from the Sun's real luminosity (field/constants.py's
     L_SUN_W, IAU 2015 nominal solar luminosity):
       F_incident = L_SUN_W / (4*pi*D^2)                        [W/m^2]
     FIRST_PRINCIPLES: inverse-square law, exact geometry -- the same
     pattern already proven in this codebase by nbody.py's solar-radiation-
     pressure term (F_SRP = L*A*(1+C_R)/(4*pi*r^2*c)).
     Reflected power = incident flux * the body's cross-sectional area
     (pi*r^2, the disk the Sun sees) * its geometric albedo p_v:
       P_reflected = F_incident * pi*r^2 * p_v                  [W]
     SIMPLIFIED_MODEL (Lambertian isotropic scatterer): this module treats
     the reflected light as if it were then re-radiated isotropically in
     all directions (same inverse-square detection-range formula as the
     thermal channel), which is a first-order approximation. The
     astronomically correct treatment is the H-G (Bowell et al. 1989)
     photometric phase-integral system real minor-planet astronomy uses,
     which models how reflected brightness varies with solar phase angle
     (the Sun-object-observer angle) via a body-specific slope parameter G
     -- a real asteroid is brightest near zero phase angle ("opposition
     surge") and dims at large phase angles in a way this module does not
     capture. This module uses the simpler geometric-albedo * inverse-
     square treatment instead for three disclosed reasons: (a) engine
     performance -- H-G evaluation needs a phase-angle-dependent lookup
     every frame for every body in view; (b) it is a legitimate first-order
     accuracy floor for "can this sensor see this object at all," which is
     the question this module answers, not "what magnitude will it read";
     (c) no per-body H-G slope-parameter data exists anywhere in this
     codebase to drive the more-correct model even if it were built.

  3. Both channels reduce to the SAME inverse-square detection-range
     inversion once a total (isotropically-radiated) power P and a sensor's
     minimum detectable flux F_min are known:
       F(R) = P / (4*pi*R^2)  =>  R_max = sqrt(P / (4*pi*F_min))
     FIRST_PRINCIPLES: inverse-square law, exact geometry (same relation
     used both directions -- once as a source-to-sensor illumination law,
     once inverted to solve for range).

  4. A small, real, CITED sensor-sensitivity catalog (SENSOR_CATALOG) and a
     small, real, CITED albedo reference table (ALBEDO_CATALOG), so this
     module's functions have real numbers to be exercised against. Every
     sensor entry's raw published sensitivity figure (limiting visual
     magnitude, mJy spectral flux density -- whatever unit its source uses)
     is converted to a broadband W/m^2 threshold via a shown, cited
     conversion chain -- see each entry's derivation comment above
     SENSOR_CATALOG. Every albedo entry is a FACT: a specific number
     directly reported by a cited source (a NASA/IAU planetary fact sheet,
     or a specific median/statistic from a peer-reviewed asteroid-taxonomy
     survey table) -- never an invented range midpoint.

Why thermal-IR sometimes beats reflected-visible for cold, dark, distant
bodies (the real observational-astronomy effect combined_detection_report
surfaces when the numbers say so): reflected flux depends strongly on
albedo AND falls off with solar distance (dimming the illumination) *and*
observer distance; thermal flux depends on temperature/size and is only
weakly dependent on albedo (a darker body absorbs more sunlight and re-
radiates more heat, partially compensating). This is why the real WISE/
NEOWISE infrared survey found large, dark near-Earth objects that visible
surveys had missed: "NEOWISE detects asteroids based on their thermal
emission and is equally sensitive to high and low albedo objects;
consequently, NEOWISE-discovered NEOs tend to be large and dark" (Mainzer,
A. et al. 2014, "Initial Performance of the NEOWISE Reactivation Mission,"
ApJ 792:30, arXiv:1406.6025, Abstract -- verified verbatim). Cold, distant
Kuiper Belt Objects (equilibrium temperature ~40-80K) are a case where this
is well documented: Stansberry, J. et al. (2008), "Physical Properties of
Kuiper Belt and Centaur Objects: Constraints from Spitzer Space Telescope,"
in The Solar System Beyond Neptune, Univ. of Arizona Press
(arXiv:astro-ph/0702538) -- the "radiometric method" of combining visible
and thermal-IR photometry specifically to break the size/albedo degeneracy
that reflected-light-only observations cannot resolve for these bodies.
This module does not hardcode that verdict; combined_detection_report()
computes each channel's actual range from the given inputs and reports
whichever number comes out larger.

Doctrine: every number here is a FACT (a real published/measured figure,
cited), a DEDUCTION (one of this module's functions, a traceable formula
chain from the FIRST_PRINCIPLES physics above), or would be a disclosed
Choice if invented -- nothing here is invented. As with propulsion.py's
catalog, every SENSOR_CATALOG and ALBEDO_CATALOG entry traces to a specific
cited source, so none of them are Choices (see
sigma_ground/dynamics/mechanisms/choice.py for the pattern this module
deliberately does not need).

Scope: passive detection range only. No active sensing (radar cross-
section, lidar), no stealth/countermeasures mechanic, no UI/rendering
integration -- matching P0a (propulsion.py) / P0b (ore_abundance.py)'s
precedent exactly.

sigma-dependence: NONE. Stefan-Boltzmann emission, inverse-square geometry,
and reflected-light optics are all ordinary EM/thermodynamics at solar-
system scales (sigma ~ 0), not SSBM spacetime-compression physics -- this
module threads a `sigma` argument through to thermal_emission_power (so a
caller working in a compressed frame gets the right emissivity/Stefan-
Boltzmann inputs) but does not itself introduce any sigma-dependent physics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .thermal import thermal_emission_power
from ..constants import L_SUN_W as _L_SUN_W

_4PI = 4.0 * math.pi


# ═══════════════════════════════════════════════════════════════════════════
# § 1. Geometry helpers -- shared by both channels
# ═══════════════════════════════════════════════════════════════════════════

def incident_solar_flux_w_m2(heliocentric_distance_m: float) -> float:
    """Solar flux at a heliocentric distance D, via inverse-square law.

    F = L_SUN_W / (4*pi*D^2)

    FIRST_PRINCIPLES: inverse-square law, exact geometry. L_SUN_W is the
    IAU 2015 nominal solar luminosity (field/constants.py). Same pattern as
    nbody.py's solar-radiation-pressure term.

    Args:
        heliocentric_distance_m: distance from the Sun, in meters (> 0).

    Returns:
        Incident flux in W/m^2.
    """
    if heliocentric_distance_m <= 0:
        raise ValueError("heliocentric_distance_m must be > 0")
    return _L_SUN_W / (_4PI * heliocentric_distance_m ** 2)


def _max_detection_range_m(power_w: float, sensor_min_flux_w_m2: float) -> float:
    """R_max = sqrt(P / (4*pi*F_min)) -- inverse of the inverse-square law.

    FIRST_PRINCIPLES: exact geometry, the same relation
    incident_solar_flux_w_m2 uses in the forward direction, solved here for
    range instead of flux. Shared by both the thermal and reflected
    channels (see module docstring section 3) -- both channels reduce to
    "isotropic power P, sensor threshold F_min" once their respective power
    values are computed.

    Args:
        power_w: total power radiated/reflected isotropically, W (>= 0).
        sensor_min_flux_w_m2: sensor's minimum detectable flux, W/m^2 (> 0).

    Returns:
        Maximum detection range in meters. 0.0 if power_w <= 0 (nothing to
        detect).
    """
    if sensor_min_flux_w_m2 <= 0:
        raise ValueError("sensor_min_flux_w_m2 must be > 0")
    if power_w <= 0:
        return 0.0
    return math.sqrt(power_w / (_4PI * sensor_min_flux_w_m2))


# ═══════════════════════════════════════════════════════════════════════════
# § 2. Thermal (infrared self-emission) channel
# ═══════════════════════════════════════════════════════════════════════════

def thermal_radiated_power_w(radius_m: float, material_key: str,
                             T: float = 300.0, sigma: float = 0.0) -> float:
    """Total power radiated by a spherical body's thermal emission (W).

    P = thermal_emission_power(material_key, T, sigma) * 4*pi*r^2

    FIRST_PRINCIPLES: Stefan-Boltzmann law (exact, via thermal.py's
    thermal_emission_power -- P/A = eps*sigma_SB*T^4) times the exact
    surface area of a sphere.
    SIMPLIFIED_MODEL: isotropic emission -- see module docstring section 1
    for the disclosed day/night-asymmetry gap this does not model. Also
    note: this is a BOLOMETRIC power (all wavelengths); pairing it with a
    band-limited SENSOR_CATALOG threshold (V-band, WISE W1, ...) carries a
    disclosed accuracy gap for sensor/temperature pairs far from the body's
    Wien-peak wavelength -- see module docstring section 1.

    Args:
        radius_m: body radius, meters (> 0).
        material_key: key into surface.py's MATERIALS (via thermal.py).
        T: surface temperature, Kelvin.
        sigma: SSBM sigma-field value (passed through to
            thermal_emission_power; ordinary EM physics is sigma-invariant
            at solar-system scales -- see module docstring).

    Returns:
        Total radiated power in W. 0.0 if T <= 0 (thermal_emission_power's
        own floor).
    """
    if radius_m <= 0:
        raise ValueError("radius_m must be > 0")
    flux_w_m2 = thermal_emission_power(material_key, T, sigma)
    surface_area_m2 = _4PI * radius_m ** 2
    return flux_w_m2 * surface_area_m2


def detection_range_thermal(power_w: float,
                            sensor_min_flux_w_m2: float) -> float:
    """Max range a sensor could detect a body's thermal emission (m).

    R_max = sqrt(P / (4*pi*F_min)) -- see _max_detection_range_m.

    Args:
        power_w: total thermal power radiated, W (from
            thermal_radiated_power_w). >= 0.
        sensor_min_flux_w_m2: sensor's minimum detectable flux, W/m^2 (> 0;
            see SENSOR_CATALOG).

    Returns:
        Maximum detection range in meters.
    """
    return _max_detection_range_m(power_w, sensor_min_flux_w_m2)


# ═══════════════════════════════════════════════════════════════════════════
# § 3. Reflected-sunlight (visible) channel
# ═══════════════════════════════════════════════════════════════════════════

def reflected_power_w(radius_m: float, albedo: float,
                      heliocentric_distance_m: float) -> float:
    """Total power reflected by a spherical body's sunlit disk (W).

    P = incident_solar_flux_w_m2(D) * (pi*r^2) * p_v

    FIRST_PRINCIPLES: incident flux via inverse-square law (exact). Cross-
    section pi*r^2 is the exact projected disk area the Sun (and, in this
    module's simplified geometry, the observer) sees.
    SIMPLIFIED_MODEL: geometric albedo p_v * inverse-square Lambertian
    treatment instead of the real H-G photometric phase-integral system --
    see module docstring section 2 for the disclosed gap (no phase-angle/
    opposition-surge modeling) and why this module uses the simpler
    treatment.

    Args:
        radius_m: body radius, meters (> 0).
        albedo: geometric albedo p_v, dimensionless in [0, 1] (see
            ALBEDO_CATALOG for real cited figures).
        heliocentric_distance_m: distance from the Sun, meters (> 0).

    Returns:
        Total reflected power in W.
    """
    if radius_m <= 0:
        raise ValueError("radius_m must be > 0")
    if not (0.0 <= albedo <= 1.0):
        raise ValueError(f"albedo must be in [0, 1], got {albedo!r}")
    flux_incident = incident_solar_flux_w_m2(heliocentric_distance_m)
    cross_section_m2 = math.pi * radius_m ** 2
    return flux_incident * cross_section_m2 * albedo


def detection_range_reflected(power_w: float,
                              sensor_min_flux_w_m2: float) -> float:
    """Max range a sensor could detect a body's reflected sunlight (m).

    R_max = sqrt(P / (4*pi*F_min)) -- see _max_detection_range_m. Same
    inversion as the thermal channel (module docstring section 3): once
    P is known, the sensor doesn't care whether the photons started as
    thermal or reflected.

    Args:
        power_w: total reflected power, W (from reflected_power_w). >= 0.
        sensor_min_flux_w_m2: sensor's minimum detectable flux, W/m^2 (> 0;
            see SENSOR_CATALOG).

    Returns:
        Maximum detection range in meters.
    """
    return _max_detection_range_m(power_w, sensor_min_flux_w_m2)


# ═══════════════════════════════════════════════════════════════════════════
# § 4. Sensor-sensitivity catalog -- every entry a cited Fact, converted to
#      a broadband W/m^2 threshold via a shown, cited conversion chain.
# ═══════════════════════════════════════════════════════════════════════════

# ── V-band (Johnson) magnitude system: zero point and Pogson relation ──────
# Used by both the naked-eye and telescope entries (both are quoted as
# limiting VISUAL/stellar magnitudes, the standard astronomical unit for
# point-source brightness).
#
# f_lambda(V=0) = 3.60e-9 erg/s/cm^2/A, FWHM = 82.7 nm.
# Source: Fukugita, M., Shimasaku, K., Ichikawa, T. (1995), "Galaxy Colors
# in Various Photometric Band Systems," PASP 107:945, "Characteristics of
# Photometric Bands" table (V, Johnson-Morgan/Azusienis & Straizys 1969
# calibration). Verified verbatim against the primary source.
#
# Unit conversion (shown, not silently assumed):
#   3.60e-9 erg/s/cm^2/A
#     * 1e-7 W/erg                  -> erg/s to W
#     * 1e4 (1/cm^2 -> 1/m^2)       -> cm^2 to m^2
#     * 10  (1/A -> 1/nm)           -> per-Angstrom to per-nanometre
#                                       (1 nm spans 10 Angstrom)
#   = 3.60e-9 * 1e-7 * 1e4 * 10 = 3.6e-11 W/m^2/nm
_V_ZERO_POINT_W_M2_NM = 3.6e-11
_V_BAND_FWHM_NM = 82.7  # Fukugita, Shimasaku & Ichikawa (1995), PASP 107, 945

# Broadband (passband-integrated) V=0 reference flux:
#   F(0) = f_lambda(V=0) [W/m^2/nm] * FWHM [nm]
_V_ZERO_POINT_BROADBAND_W_M2 = _V_ZERO_POINT_W_M2_NM * _V_BAND_FWHM_NM
# = 3.6e-11 * 82.7 ~= 2.977e-9 W/m^2


def _flux_from_v_magnitude(m_v: float) -> float:
    """Pogson magnitude-flux relation: F(m) = F(0) * 10^(-m/2.5).

    FIRST_PRINCIPLES: the magnitude scale is logarithmic by definition
    (Pogson 1856), calibrated here against the real V-band zero-point flux
    above. A difference of 5 magnitudes is, by this definition, exactly a
    factor of 100 in flux -- the hand-checkable invariant this function
    obeys exactly regardless of the zero-point's absolute calibration.
    """
    return _V_ZERO_POINT_BROADBAND_W_M2 * 10.0 ** (-m_v / 2.5)


# ── Naked eye: dark-sky limiting visual magnitude m_V = 6.5 ────────────────
# Source: Curtis, H.D. (1901), quoted in Crumey, A. (2014), "Human contrast
# threshold and astronomical visibility," MNRAS 442(3):2600-2619
# (arXiv:1405.4209), Sec 1.6.5: "his limit for stars seen against the sky
# was 6.5". Verified verbatim against the primary source.
#
# Crumey (2014)'s own rigorous darkest-real-sky physical model (Sec 3.1,
# Eq 53, B=2e-4 cd/m^2) independently gives a range of 5.98-6.57 for the
# same regime (typical field-factor F=2 -> 6.18) -- corroborating 6.5 as a
# standard, slightly conservative headline figure. NOTE: an earlier draft
# of this research mis-attributed a "6.2-6.9" range to that same equation;
# the paper's own stated numbers are 5.98-6.57 (typical 6.18), not 6.2-6.9
# -- this module uses the verbatim-quoted 6.5 headline figure and does not
# repeat the incorrect range.
_NAKED_EYE_LIMITING_MAG_V = 6.5
_NAKED_EYE_THRESHOLD_W_M2 = _flux_from_v_magnitude(_NAKED_EYE_LIMITING_MAG_V)
# ~= 7.48e-12 W/m^2

# ── Amateur/mid-size optical telescope: Celestron NexStar 8SE (8", 203mm) ──
# Source: Celestron, official manufacturer product specification page,
# https://www.celestron.com/products/nexstar-8se-computerized-telescope,
# "Optical Tube Info" spec table: "Limiting Stellar Magnitude: 14". Verified
# directly against the live manufacturer page; cross-checked against the
# classical aperture-vs-limiting-magnitude relation m ~= 7.5 + 5*log10(D_cm)
# (astronomy.tools convention), which for D=20.3cm predicts ~14.04 --
# consistent with the rated spec to within the convention's own ~0.2 mag
# spread across sources. This is the manufacturer's rated ceiling under
# zero-light-pollution conditions, not a typical suburban-sky achieved
# figure -- disclosed here as the manufacturer's own framing, not silently
# reused as an "achieved under typical conditions" number.
_TELESCOPE_8IN_LIMITING_MAG_V = 14.0
_TELESCOPE_8IN_THRESHOLD_W_M2 = _flux_from_v_magnitude(_TELESCOPE_8IN_LIMITING_MAG_V)
# ~= 7.48e-15 W/m^2 (exactly 1000x fainter than the naked eye: 14.0 - 6.5 =
# 7.5 mag = 10^(7.5/2.5) = 10^3 = 1000x, a hand-checkable Pogson invariant)

# ── Space-based infrared survey: WISE W1 band (3.4 micron) ─────────────────
# Mission/band: Wright, E.L. et al. (2010), "The Wide-field Infrared Survey
# Explorer (WISE): Mission Description and Initial On-orbit Performance,"
# AJ 140:1868 (arXiv:1008.0031). Verified directly against the primary
# paper's PDF text (not a summary).
#
# 5-sigma point-source sensitivity, achieved (not theoretical) on-orbit
# performance, W1 band: 0.08 mJy. Source: Wright et al. 2010, Abstract:
# "WISE is achieving 5 sigma point source sensitivities better than 0.08,
# 0.11, 1 and 6 mJy in unconfused regions on the ecliptic in bands centered
# at wavelengths of 3.4, 4.6, 12 and 22 microns." (verified verbatim). This
# is a ceiling/benchmark figure for unconfused ecliptic-plane regions
# (sensitivity improves toward the ecliptic poles), used here as the
# standard single representative threshold, per how the follow-on WISE/
# NEOWISE literature itself treats this number.
_WISE_W1_5SIGMA_MJY = 0.08

# Jansky -> SI spectral flux density: 1 Jy = 1e-26 W/m^2/Hz (definitional,
# IAU/radio-astronomy convention).
_JY_TO_W_M2_HZ = 1.0e-26

# mJy -> Jy -> W/m^2/Hz (shown conversion chain):
#   0.08 mJy * 1e-3 Jy/mJy * 1e-26 W/m^2/Hz/Jy = 8.0e-31 W/m^2/Hz
_WISE_W1_5SIGMA_W_M2_HZ = _WISE_W1_5SIGMA_MJY * 1.0e-3 * _JY_TO_W_M2_HZ

# Effective (equivalent-width) bandwidth, derived from the official WISE W1
# relative-spectral-response curve (Delta_lambda_eff = integral(R(lambda)
# dlambda) / R_peak, trapezoidal integration of the full tabulated curve):
# Delta_lambda_eff ~= 0.68 micron, at isophotal wavelength lambda_center =
# 3.3526 micron (both from Wright et al. 2010 / the official IPAC RSR-W1.txt
# data file, irsa.ipac.caltech.edu -- both independently reproduced from the
# raw tabulated data to the cited precision).
# Converting a wavelength-domain bandwidth to a frequency-domain bandwidth
# (Delta_nu = c * Delta_lambda / lambda_center^2, the standard narrow-band
# approximation to d(nu)/d(lambda) = -c/lambda^2):
#   Delta_nu = (2.99792458e8 m/s * 0.68e-6 m) / (3.3526e-6 m)^2
#            ~= 1.82e13 Hz  (~18.2 THz)
_WISE_W1_BANDWIDTH_HZ = 1.82e13

# Broadband W1 5-sigma threshold: F = S_nu * Delta_nu
#   = 8.0e-31 W/m^2/Hz * 1.82e13 Hz ~= 1.456e-17 W/m^2
_WISE_W1_THRESHOLD_W_M2 = _WISE_W1_5SIGMA_W_M2_HZ * _WISE_W1_BANDWIDTH_HZ


@dataclass(frozen=True)
class Sensor:
    """One sensor-catalog entry -- provenance disclosed per-entry, same
    doctrine as PropulsionSystem in propulsion.py: every number here traces
    to a specific cited source, converted to a broadband W/m^2 threshold
    via a shown conversion chain (see the derivation comments above this
    class)."""
    category: str
    name: str
    limiting_flux_w_m2: float
    provenance: str
    source: str
    note: str = ""

    def __post_init__(self):
        if self.provenance not in ("fact", "choice"):
            raise ValueError(
                f"{self.name}: provenance must be 'fact' or 'choice', got "
                f"{self.provenance!r}")
        if self.limiting_flux_w_m2 <= 0:
            raise ValueError(f"{self.name}: limiting_flux_w_m2 must be > 0")

    def cite(self) -> str:
        tail = f" ({self.note})" if self.note else ""
        return (f"{self.name} [{self.category}]: "
                f"F_min={self.limiting_flux_w_m2:.3e} W/m^2{tail} -- "
                f"{self.source}")


SENSOR_CATALOG: dict[str, Sensor] = {
    "naked_eye": Sensor(
        category="human_eye",
        name="Human naked eye (dark-sky limit, m_V=6.5)",
        limiting_flux_w_m2=_NAKED_EYE_THRESHOLD_W_M2,
        provenance="fact",
        source="Curtis, H.D. (1901), quoted in Crumey, A. (2014), 'Human "
               "contrast threshold and astronomical visibility,' MNRAS "
               "442(3):2600-2619 (arXiv:1405.4209), Sec 1.6.5. V-band zero "
               "point: Fukugita, Shimasaku & Ichikawa (1995), PASP 107, "
               "945.",
        note="dark-sky, zero-light-pollution point-source limit; Crumey "
             "(2014)'s own rigorous model gives 5.98-6.57 (typical 6.18) "
             "for the same regime, corroborating 6.5"),
    "amateur_telescope_8in": Sensor(
        category="optical_telescope",
        name="8-inch (203mm) Schmidt-Cassegrain amateur telescope "
             "(Celestron NexStar 8SE)",
        limiting_flux_w_m2=_TELESCOPE_8IN_THRESHOLD_W_M2,
        provenance="fact",
        source="Celestron, NexStar 8SE Computerized Telescope product "
               "specification page (celestron.com/products/"
               "nexstar-8se-computerized-telescope), 'Optical Tube Info': "
               "Limiting Stellar Magnitude 14.0",
        note="manufacturer's rated ceiling under zero-light-pollution "
             "conditions, not a typical achieved-under-real-skies figure"),
    "wise_w1": Sensor(
        category="space_infrared_survey",
        name="WISE/NEOWISE W1 band (3.4 micron)",
        limiting_flux_w_m2=_WISE_W1_THRESHOLD_W_M2,
        provenance="fact",
        source="Wright, E.L. et al. (2010), 'The Wide-field Infrared "
               "Survey Explorer (WISE): Mission Description and Initial "
               "On-orbit Performance,' AJ 140:1868 (arXiv:1008.0031), "
               "Abstract (5-sigma sensitivity); official IPAC RSR-W1.txt "
               "relative-spectral-response curve (bandwidth)",
        note="achieved (not theoretical) on-orbit 5-sigma point-source "
             "sensitivity in unconfused ecliptic-plane regions; "
             "sensitivity improves toward the ecliptic poles"),
}


# ═══════════════════════════════════════════════════════════════════════════
# § 5. Albedo reference catalog -- every entry a cited Fact.
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AlbedoReference:
    """One albedo-catalog entry. Every geometric_albedo value below is a
    FACT: a specific number directly reported by the cited source (a NASA/
    IAU planetary fact sheet's own tabulated figure, or a specific median
    statistic from a peer-reviewed asteroid-taxonomy survey's own data
    table) -- never an invented range midpoint, matching ore_abundance.py's
    provenance discipline for the taxonomy figures specifically."""
    category: str  # "planet" | "asteroid_taxonomy"
    name: str
    geometric_albedo: float
    provenance: str
    source: str
    albedo_range: tuple[float, float] | None = None
    note: str = ""

    def __post_init__(self):
        if self.provenance not in ("fact", "choice"):
            raise ValueError(
                f"{self.name}: provenance must be 'fact' or 'choice', got "
                f"{self.provenance!r}")
        if not (0.0 <= self.geometric_albedo <= 1.0):
            raise ValueError(
                f"{self.name}: geometric_albedo {self.geometric_albedo} "
                f"out of [0, 1]")


ALBEDO_CATALOG: dict[str, AlbedoReference] = {
    "venus": AlbedoReference(
        category="planet", name="Venus", geometric_albedo=0.689,
        provenance="fact",
        source="Mallama, A. et al. (2017), Icarus 282:19-33 (V-band "
               "geometric albedo, Mallama & Hilton 2018 revision), as "
               "reproduced on the NASA NSSDCA Venus Fact Sheet",
        note="supersedes the older, widely-mirrored ~0.65-0.67 figure"),
    "mars": AlbedoReference(
        category="planet", name="Mars", geometric_albedo=0.170,
        provenance="fact",
        source="Mallama, A. (2007), 'The magnitude and albedo of Mars,' "
               "Icarus 192:404, as reproduced on the NASA NSSDCA Mars "
               "Fact Sheet"),
    "jupiter": AlbedoReference(
        category="planet", name="Jupiter", geometric_albedo=0.538,
        provenance="fact",
        source="NASA NSSDCA Jupiter Fact Sheet; corroborated by Mallama, "
               "A., Wang, D. & Howell, R.R. (2017), arXiv:1609.05048 "
               "(0.535 +/- 0.005, V-band)"),
    "saturn": AlbedoReference(
        category="planet", name="Saturn (disk only, excludes rings)",
        geometric_albedo=0.499, provenance="fact",
        source="NASA NSSDCA Saturn Fact Sheet"),
    "moon": AlbedoReference(
        category="planet", name="Earth's Moon", geometric_albedo=0.12,
        provenance="fact",
        source="NASA NSSDCA Lunar Fact Sheet",
        note="geometric albedo, not Bond albedo (~0.11) -- do not conflate "
             "with a radiative energy-balance calculation"),
    "c_type": AlbedoReference(
        category="asteroid_taxonomy",
        name="C-type (Tholen C / Bus-DeMeo C-complex) -- dark, "
             "carbonaceous",
        geometric_albedo=0.055, albedo_range=(0.03, 0.10),
        provenance="fact",
        source="Mainzer, A., Grav, T., Masiero, J. et al. (2011), "
               "'NEOWISE Studies of Spectrophotometrically Classified "
               "Asteroids: Preliminary Results,' ApJ 741, 90 "
               "(arXiv:1109.6407), Table 1: Tholen C, N=323, median "
               "pV=0.055"),
    "s_type": AlbedoReference(
        category="asteroid_taxonomy",
        name="S-type (Tholen S / Bus-DeMeo S-complex) -- moderate, "
             "siliceous",
        geometric_albedo=0.210, albedo_range=(0.10, 0.30),
        provenance="fact",
        source="Mainzer, A., Grav, T., Masiero, J. et al. (2011), ApJ "
               "741, 90 (arXiv:1109.6407), Table 1: Tholen S, N=502, "
               "median pV=0.210"),
    "m_type": AlbedoReference(
        category="asteroid_taxonomy",
        name="M-type (Tholen M) -- moderate-to-bright, metallic",
        geometric_albedo=0.125, albedo_range=(0.10, 0.30),
        provenance="fact",
        source="Definitional boundary: Tholen, D.J. (1984), PhD "
               "dissertation, Univ. of Arizona (0.1<pV<0.3). Measured "
               "statistic: Mainzer, A., Grav, T., Masiero, J. et al. "
               "(2011), ApJ 741, 90 (arXiv:1109.6407), Table 1: Tholen M, "
               "N=33, median pV=0.125"),
}


# ═══════════════════════════════════════════════════════════════════════════
# § 6. Combined report -- both channels, which one dominates
# ═══════════════════════════════════════════════════════════════════════════

def combined_detection_report(radius_m: float, material_key: str, T: float,
                              sigma: float, albedo: float,
                              heliocentric_distance_m: float,
                              sensor_key: str) -> dict:
    """Both channel detection ranges for one body/sensor pair, plus which
    channel dominates.

    Computes thermal_radiated_power_w / detection_range_thermal and
    reflected_power_w / detection_range_reflected for the SAME sensor
    threshold, and reports whichever range comes out longer as the
    "dominant_channel" -- computed from the actual inputs given, not
    hardcoded. See module docstring for the real, cited observational-
    astronomy background on why thermal-IR often dominates for cold, dark,
    distant bodies (Mainzer et al. 2014; Stansberry et al. 2008).

    CAVEAT: the thermal channel's power is bolometric while every
    SENSOR_CATALOG threshold is band-limited (see module docstring section
    1) -- results for the naked_eye/amateur_telescope_8in (V-band) sensors
    paired with cold bodies (Wien peak far outside V-band) should be read
    as a loose upper bound on thermal detectability, not an achievable
    range. wise_w1 (mid-IR) is the best-matched pairing in this catalog.

    Args:
        radius_m: body radius, meters (> 0).
        material_key: key into surface.py's MATERIALS (thermal channel).
        T: surface temperature, Kelvin (thermal channel).
        sigma: SSBM sigma-field value (passed through to
            thermal_emission_power; see module docstring on sigma-
            invariance).
        albedo: geometric albedo p_v, dimensionless in [0, 1] (reflected
            channel; see ALBEDO_CATALOG for real cited figures).
        heliocentric_distance_m: distance from the Sun, meters (> 0;
            reflected channel).
        sensor_key: key into SENSOR_CATALOG.

    Returns:
        dict with both channels' power/range, the sensor used, and
        dominant_channel in {"thermal", "reflected", "tie"}.
    """
    if sensor_key not in SENSOR_CATALOG:
        raise ValueError(
            f"Unknown sensor {sensor_key!r}. Recognized: "
            f"{sorted(SENSOR_CATALOG)}")
    sensor = SENSOR_CATALOG[sensor_key]

    p_thermal = thermal_radiated_power_w(radius_m, material_key, T, sigma)
    range_thermal_m = detection_range_thermal(
        p_thermal, sensor.limiting_flux_w_m2)

    p_reflected = reflected_power_w(radius_m, albedo, heliocentric_distance_m)
    range_reflected_m = detection_range_reflected(
        p_reflected, sensor.limiting_flux_w_m2)

    if range_thermal_m > range_reflected_m:
        dominant = "thermal"
    elif range_reflected_m > range_thermal_m:
        dominant = "reflected"
    else:
        dominant = "tie"

    return {
        "sensor_key": sensor_key,
        "sensor_name": sensor.name,
        "sensor_limiting_flux_w_m2": sensor.limiting_flux_w_m2,
        "radius_m": radius_m,
        "material_key": material_key,
        "temperature_K": T,
        "sigma": sigma,
        "albedo": albedo,
        "heliocentric_distance_m": heliocentric_distance_m,
        "thermal_power_w": p_thermal,
        "thermal_detection_range_m": range_thermal_m,
        "reflected_power_w": p_reflected,
        "reflected_detection_range_m": range_reflected_m,
        "dominant_channel": dominant,
        "note": (
            "Cold, dark, distant bodies are well documented to often be "
            "easier to detect via thermal IR than reflected visible light "
            "(Mainzer et al. 2014, ApJ 792:30, arXiv:1406.6025; Stansberry "
            "et al. 2008, arXiv:astro-ph/0702538) -- dominant_channel here "
            "is computed from the actual inputs given, not asserted."
        ),
    }


__all__ = [
    "incident_solar_flux_w_m2",
    "thermal_radiated_power_w",
    "detection_range_thermal",
    "reflected_power_w",
    "detection_range_reflected",
    "Sensor",
    "SENSOR_CATALOG",
    "AlbedoReference",
    "ALBEDO_CATALOG",
    "combined_detection_report",
]
