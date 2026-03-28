"""
Subsurface scattering — light transport through translucent materials.

Derivation chain:
  optics.py (refractive index, Drude scattering)
  + photonics.py (absorption coefficients)
  → subsurface.py (scattering length, diffusion, BSSRDF parameters)

Translucent materials (skin, wax, marble, milk, jade) neither fully
reflect nor fully transmit light. Photons enter the surface, scatter
multiple times off internal inhomogeneities, and exit at a different
point. The visual effect: soft, glowing appearance without sharp shadows.

The key parameters for a renderer:
  1. Mean free path (MFP): how far a photon travels between scattering events
  2. Absorption length: how far before the photon is absorbed
  3. Diffusion length: the effective "blur radius" of subsurface scattering
  4. Albedo: fraction of photons that eventually escape (not absorbed)

Derivation chains:

  1. Scattering Coefficient μ_s (FIRST_PRINCIPLES: Rayleigh/Mie)
     μ_s = N × σ_scatter

     Where:
       N = number density of scatterers (particles, cells, grain boundaries)
       σ_scatter = scattering cross-section

     For Rayleigh scattering (scatterer << λ):
       σ_scatter = (8π/3)(2πr/λ)⁴ × r² × ((m²-1)/(m²+2))²
     where m = n_particle/n_medium.

     For Mie scattering (scatterer ~ λ): σ ≈ 2πr² (geometric limit).

  2. Absorption Coefficient μ_a (FIRST_PRINCIPLES: Beer-Lambert)
     I(x) = I₀ exp(−μ_a x)

     μ_a depends on molecular absorption bands (chromophores).
     For biological tissue: hemoglobin, melanin, water.
     For minerals: transition metal ions.

  3. Transport Mean Free Path (FIRST_PRINCIPLES)
     l_tr = 1 / (μ_s' + μ_a)

     Where μ_s' = μ_s(1-g) is the reduced scattering coefficient,
     and g is the anisotropy factor (average cosine of scattering angle).
     g ≈ 0.8-0.95 for tissue (strongly forward-scattered).

  4. Diffusion Length (FIRST_PRINCIPLES: diffusion theory)
     L_d = √(D / μ_a) = 1 / √(3μ_a(μ_a + μ_s'))

     This is the characteristic distance that subsurface-scattered
     photons travel before being absorbed. It determines the visual
     "blur" radius.

  5. Diffuse Reflectance (Kubelka-Munk, FIRST_PRINCIPLES)
     R_d ≈ α' / (1 + α')
     where α' = √(μ_a / (μ_a + μ_s'))

     For highly scattering materials (μ_s' >> μ_a): R_d → 1 (white).
     For highly absorbing (μ_a >> μ_s'): R_d → 0 (black).

σ-dependence:
  Optical properties are EM → σ-INVARIANT.
  Scattering from structural features (grain boundaries, cells)
  depends on size ∝ lattice spacing → very weak σ-dependence.

Origin tags:
  - Rayleigh/Mie scattering: FIRST_PRINCIPLES (Maxwell's equations)
  - Beer-Lambert absorption: FIRST_PRINCIPLES (exponential attenuation)
  - Diffusion approximation: FIRST_PRINCIPLES (random walk theory)
  - Kubelka-Munk: FIRST_PRINCIPLES (two-flux model)
  - Tissue optical properties: MEASURED (Jacques 2013, Phys. Med. Biol.)
"""

import math


# ── Material Optical Properties ───────────────────────────────────
# MEASURED from integrating-sphere spectrophotometry and OCT.
#
# μ_a: absorption coefficient (1/m) at 550 nm (green, eye peak)
# μ_s_prime: reduced scattering coefficient (1/m) at 550 nm
# g: anisotropy factor (average cos θ of scattering)
# n: refractive index
#
# Sources:
#   Jacques (2013), Phys. Med. Biol. 58, R37 (tissue optics review)
#   Cheong et al. (1990), IEEE J. Quant. Electron. 26, 2166
#   Tuchin "Tissue Optics" 2nd ed. (2007)
#
# Rule 9: every common translucent material.

TRANSLUCENT_MATERIALS = {
    # Biological tissues
    'skin_caucasian': {
        'name': 'Human skin (Caucasian, dermis)',
        'mu_a_m': 40.0,           # MEASURED: 0.4 /cm = 40 /m (melanin + blood)
        'mu_s_prime_m': 20000.0,  # MEASURED: 200 /cm = 20000 /m (collagen fibers)
        'g': 0.90,                # MEASURED: strongly forward-scattered
        'n': 1.40,                # MEASURED
    },
    'skin_dark': {
        'name': 'Human skin (dark, dermis)',
        'mu_a_m': 200.0,          # Higher melanin absorption
        'mu_s_prime_m': 18000.0,  # Similar scattering (collagen same)
        'g': 0.90,
        'n': 1.40,
    },
    'fat': {
        'name': 'Adipose tissue (fat)',
        'mu_a_m': 10.0,           # Low absorption (transparent cells)
        'mu_s_prime_m': 12000.0,  # High scattering (lipid droplets)
        'g': 0.90,
        'n': 1.44,
    },
    'muscle': {
        'name': 'Skeletal muscle',
        'mu_a_m': 80.0,           # Myoglobin absorption
        'mu_s_prime_m': 5000.0,   # Moderate scattering (fibers)
        'g': 0.95,
        'n': 1.40,
    },
    'blood': {
        'name': 'Whole blood (oxygenated)',
        'mu_a_m': 30000.0,        # Very high (hemoglobin at 550nm!)
        'mu_s_prime_m': 20000.0,  # High (RBC scattering)
        'g': 0.99,                # Extremely forward
        'n': 1.37,
    },
    # Minerals and natural materials
    'marble_white': {
        'name': 'White Carrara marble',
        'mu_a_m': 2.0,            # Very low (pure CaCO₃)
        'mu_s_prime_m': 100000.0, # Very high (grain boundaries)
        'g': 0.85,
        'n': 1.56,
    },
    'jade_nephrite': {
        'name': 'Nephrite jade',
        'mu_a_m': 500.0,          # Fe²⁺ absorption (green tint)
        'mu_s_prime_m': 50000.0,  # Interlocking fibers
        'g': 0.80,
        'n': 1.61,
    },
    'alabaster': {
        'name': 'Alabaster (gypsum)',
        'mu_a_m': 5.0,
        'mu_s_prime_m': 80000.0,
        'g': 0.85,
        'n': 1.52,
    },
    # Waxes and organics
    'beeswax': {
        'name': 'Beeswax',
        'mu_a_m': 20.0,           # Slight yellow absorption
        'mu_s_prime_m': 30000.0,  # Crystallite scattering
        'g': 0.85,
        'n': 1.44,
    },
    'paraffin_wax': {
        'name': 'Paraffin wax',
        'mu_a_m': 5.0,            # Nearly colorless
        'mu_s_prime_m': 25000.0,  # Crystallite scattering
        'g': 0.85,
        'n': 1.43,
    },
    'milk_whole': {
        'name': 'Whole milk',
        'mu_a_m': 5.0,            # Very low
        'mu_s_prime_m': 110000.0, # Very high (fat globules, Mie scattering)
        'g': 0.75,                # Less forward than tissue
        'n': 1.35,
    },
    # Synthetic materials
    'soap_bar': {
        'name': 'Soap (white bar)',
        'mu_a_m': 10.0,
        'mu_s_prime_m': 60000.0,
        'g': 0.80,
        'n': 1.46,
    },
    'candle_wax': {
        'name': 'Candle wax (white)',
        'mu_a_m': 3.0,
        'mu_s_prime_m': 35000.0,
        'g': 0.85,
        'n': 1.44,
    },
}


# ═══════════════════════════════════════════════════════════════════
# SCATTERING PHYSICS
# ═══════════════════════════════════════════════════════════════════

def transport_mean_free_path(material_key):
    """Transport mean free path l_tr (metres).

    l_tr = 1 / (μ_s' + μ_a)

    FIRST_PRINCIPLES: the average distance a photon travels
    (after accounting for forward-scattering bias) before
    being scattered or absorbed.

    This is the length scale over which light "forgets" its
    original direction — the step size for diffusion.

    Args:
        material_key: key into TRANSLUCENT_MATERIALS

    Returns:
        Mean free path in metres.
    """
    data = TRANSLUCENT_MATERIALS[material_key]
    total = data['mu_s_prime_m'] + data['mu_a_m']
    if total <= 0:
        return float('inf')
    return 1.0 / total


def absorption_length(material_key):
    """Absorption length 1/μ_a (metres).

    The average distance before a photon is absorbed.

    Args:
        material_key: key into TRANSLUCENT_MATERIALS

    Returns:
        Absorption length in metres.
    """
    mu_a = TRANSLUCENT_MATERIALS[material_key]['mu_a_m']
    if mu_a <= 0:
        return float('inf')
    return 1.0 / mu_a


def scattering_length(material_key):
    """Reduced scattering length 1/μ_s' (metres).

    The average distance between effective scattering events
    (corrected for forward-scattering anisotropy).

    Args:
        material_key: key into TRANSLUCENT_MATERIALS

    Returns:
        Scattering length in metres.
    """
    mu_sp = TRANSLUCENT_MATERIALS[material_key]['mu_s_prime_m']
    if mu_sp <= 0:
        return float('inf')
    return 1.0 / mu_sp


def diffusion_length(material_key):
    """Diffusion length L_d (metres) — the SSS "blur radius".

    L_d = 1 / √(3 μ_a (μ_a + μ_s'))

    FIRST_PRINCIPLES: solution to photon diffusion equation.

    This is what a renderer needs: the characteristic distance
    that light travels inside the material before being absorbed.
    It determines the visible "glow" radius of subsurface scattering.

    Args:
        material_key: key into TRANSLUCENT_MATERIALS

    Returns:
        Diffusion length in metres.
    """
    data = TRANSLUCENT_MATERIALS[material_key]
    mu_a = data['mu_a_m']
    mu_sp = data['mu_s_prime_m']

    if mu_a <= 0:
        return float('inf')

    denom = 3.0 * mu_a * (mu_a + mu_sp)
    if denom <= 0:
        return float('inf')

    return 1.0 / math.sqrt(denom)


def diffusion_coefficient(material_key):
    """Photon diffusion coefficient D (m²/s... but units are m here).

    D = 1 / (3(μ_a + μ_s'))

    In the diffusion approximation, photon transport is modeled
    as a random walk with step size l_tr. D has units of length
    (it's really D_photon / c, but c cancels in the steady-state
    diffusion equation).

    Args:
        material_key: key into TRANSLUCENT_MATERIALS

    Returns:
        Diffusion coefficient in metres (effective).
    """
    data = TRANSLUCENT_MATERIALS[material_key]
    total = data['mu_a_m'] + data['mu_s_prime_m']
    if total <= 0:
        return float('inf')
    return 1.0 / (3.0 * total)


def single_scatter_albedo(material_key):
    """Single-scattering albedo a = μ_s' / (μ_a + μ_s').

    Probability that a photon scatters (vs being absorbed) at each
    interaction. High albedo → bright, translucent material.

    Args:
        material_key: key into TRANSLUCENT_MATERIALS

    Returns:
        Albedo a ∈ [0, 1].
    """
    data = TRANSLUCENT_MATERIALS[material_key]
    total = data['mu_a_m'] + data['mu_s_prime_m']
    if total <= 0:
        return 0.0
    return data['mu_s_prime_m'] / total


def diffuse_reflectance(material_key):
    """Diffuse reflectance R_d from Kubelka-Munk theory.

    R_d ≈ a' / (1 + a')
    where a' = √(3 μ_a / (μ_a + μ_s'))

    FIRST_PRINCIPLES: two-flux approximation (Kubelka-Munk 1931).
    More accurate version uses the effective reflection coefficient:

    R_d = (1 + exp(-4/3 × A × √(3(1-a))))
        / (1 + exp(+4/3 × A × √(3(1-a))))

    where A accounts for internal reflection at the boundary.

    We use the simpler form which is accurate to ±10% for
    thick, highly scattering media.

    Args:
        material_key: key into TRANSLUCENT_MATERIALS

    Returns:
        Diffuse reflectance R_d ∈ [0, 1].
    """
    data = TRANSLUCENT_MATERIALS[material_key]
    mu_a = data['mu_a_m']
    mu_sp = data['mu_s_prime_m']

    total = mu_a + mu_sp
    if total <= 0 or mu_a < 0:
        return 1.0

    a_prime = math.sqrt(3.0 * mu_a / total)

    if a_prime <= 0:
        return 1.0  # pure scatterer, no absorption

    return a_prime / (1.0 + a_prime) if a_prime < 100 else 1.0


# ═══════════════════════════════════════════════════════════════════
# RENDERER-ORIENTED OUTPUTS
# ═══════════════════════════════════════════════════════════════════

def bssrdf_parameters(material_key):
    """Parameters needed for a BSSRDF subsurface scattering renderer.

    Returns the five quantities a physically-based renderer needs
    to implement subsurface scattering (dipole diffusion model):

    1. σ_a: absorption coefficient (1/m)
    2. σ_s': reduced scattering coefficient (1/m)
    3. η: refractive index
    4. L_d: diffusion length (m) — the "blur radius"
    5. A: single-scattering albedo

    These plug directly into Jensen et al. (2001) "A Practical Model
    for Subsurface Light Transport" (the standard BSSRDF model).

    Args:
        material_key: key into TRANSLUCENT_MATERIALS

    Returns:
        dict with renderer parameters.
    """
    data = TRANSLUCENT_MATERIALS[material_key]

    return {
        'sigma_a': data['mu_a_m'],
        'sigma_s_prime': data['mu_s_prime_m'],
        'eta': data['n'],
        'g': data['g'],
        'diffusion_length_m': diffusion_length(material_key),
        'diffusion_length_mm': diffusion_length(material_key) * 1000.0,
        'albedo': single_scatter_albedo(material_key),
        'diffuse_reflectance': diffuse_reflectance(material_key),
    }


# ═══════════════════════════════════════════════════════════════════
# RAYLEIGH SCATTERING (from first principles)
# ═══════════════════════════════════════════════════════════════════

def rayleigh_scattering_coefficient(particle_radius_m, n_particle,
                                     n_medium, wavelength_m,
                                     number_density):
    """Rayleigh scattering coefficient μ_s (1/m).

    μ_s = N × σ_Rayleigh

    Where σ_Rayleigh = (8π/3)(2πr/λ)⁴ × r² × ((m²-1)/(m²+2))²

    FIRST_PRINCIPLES: Rayleigh (1871), electromagnetic scattering
    from particles much smaller than wavelength.

    Valid when r << λ (Rayleigh regime: r < λ/10).

    Args:
        particle_radius_m: scatterer radius (m)
        n_particle: refractive index of scatterer
        n_medium: refractive index of medium
        wavelength_m: wavelength in metres
        number_density: number of scatterers per m³

    Returns:
        Scattering coefficient in 1/m.
    """
    if wavelength_m <= 0 or n_medium <= 0:
        return 0.0

    m = n_particle / n_medium  # relative index
    size_param = 2.0 * math.pi * particle_radius_m / wavelength_m

    # Rayleigh cross-section
    sigma = (8.0 * math.pi / 3.0) * size_param ** 4 * particle_radius_m ** 2
    sigma *= ((m ** 2 - 1.0) / (m ** 2 + 2.0)) ** 2

    return number_density * sigma


# ── Diagnostics ───────────────────────────────────────────────────

def subsurface_report(material_key):
    """Complete subsurface scattering report."""
    data = TRANSLUCENT_MATERIALS[material_key]
    bssrdf = bssrdf_parameters(material_key)

    return {
        'name': data['name'],
        'material': material_key,
        'mu_a_per_m': data['mu_a_m'],
        'mu_s_prime_per_m': data['mu_s_prime_m'],
        'anisotropy_g': data['g'],
        'refractive_index': data['n'],
        'transport_mfp_mm': transport_mean_free_path(material_key) * 1000,
        'absorption_length_mm': absorption_length(material_key) * 1000,
        'scattering_length_mm': scattering_length(material_key) * 1000,
        'diffusion_length_mm': bssrdf['diffusion_length_mm'],
        'albedo': bssrdf['albedo'],
        'diffuse_reflectance': bssrdf['diffuse_reflectance'],
    }


def full_report():
    """Reports for ALL translucent materials. Rule 9."""
    return {key: subsurface_report(key) for key in TRANSLUCENT_MATERIALS}
