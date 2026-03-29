"""
Grain structure — microstructure-dependent mechanical properties.

The missing link between atomic-scale physics (mechanical.py, plasticity.py)
and real engineering materials. Single crystals are rare; real metals are
polycrystalline. Grain boundaries impede dislocation motion, which is why
grain size is the primary microstructural control on strength.

Derivation chains:

  1. Hall-Petch Relation (1951/1953, FIRST_PRINCIPLES reasoning)
     σ_y = σ_0 + k_HP / √d

     Where:
       σ_y = yield stress of polycrystal (Pa)
       σ_0 = lattice friction stress (Pa, MEASURED)
         (= Peierls-Nabarro stress + solid solution contribution)
       k_HP = Hall-Petch slope (Pa√m, MEASURED)
       d = mean grain diameter (m)

     Physics: dislocations pile up at grain boundaries under applied
     stress. The stress concentration at the pile-up tip scales as
     √(n_dislocations) ∝ √(grain_size). Yielding occurs when this
     concentrated stress activates a Frank-Read source in the
     neighboring grain.

     Eshelby-Frank-Nabarro pile-up model:
       τ* = n × τ_applied, where n ≈ d/(2b) (b = Burgers vector)
       → σ_y ∝ 1/√d

     Valid range: ~10 nm to ~1 mm grain size.
     Below ~10 nm: inverse Hall-Petch (grain boundary sliding dominates).

  2. Grain Growth Kinetics (Burke-Turnbull 1952, FIRST_PRINCIPLES)
     d² - d₀² = K_g × t

     Where:
       d = grain size at time t (m)
       d₀ = initial grain size (m)
       K_g = grain growth rate constant (m²/s)
       t = time at temperature (s)

     K_g = K_0 × exp(-Q_gg/(RT))

     Where:
       K_0 = pre-exponential (m²/s, MEASURED)
       Q_gg = activation energy for grain boundary migration (J/mol)
         ≈ 0.4-0.6 × Q_self-diffusion

     Physics: grain boundaries are driven to reduce total boundary area
     (minimize surface energy). Larger grains grow at expense of smaller
     ones (Ostwald ripening of grains).

  3. Grain Boundary Strengthening Energy
     ΔE = k_HP / √d × ε_p × V

     Energy absorbed by grain boundary hardening during plastic
     deformation. Important for crash energy absorption design.

  4. Inverse Hall-Petch (Chokshi 1989)
     Below d_critical ≈ 10-20 nm, yield stress DECREASES with
     decreasing grain size. Grain boundary sliding and diffusion
     creep dominate over dislocation pile-up.

     σ_y = σ_peak − k_inv × (1/d − 1/d_crit)  for d < d_crit

  5. Zener Pinning (1949)
     Maximum grain size limited by second-phase particles:
       d_max = 4r / (3f)

     Where:
       r = particle radius (m)
       f = volume fraction of particles

     Particles pin grain boundaries, preventing coarsening.
     This is why alloys (with precipitates) are more thermally stable.

σ-dependence:
  σ → E_coh → G → Burgers vector energy → k_HP
  σ → E_coh → Q_gg (grain boundary migration activation energy)
  Both channels: stronger bonds → higher k_HP, slower grain growth.

Origin tags:
  - Hall-Petch: FIRST_PRINCIPLES (Eshelby pile-up) + MEASURED (σ_0, k_HP)
  - Grain growth: FIRST_PRINCIPLES (curvature-driven boundary motion)
    + MEASURED (K_0, Q_gg)
  - Inverse Hall-Petch: MEASURED (observed in nanocrystalline metals)
  - Zener pinning: FIRST_PRINCIPLES (force balance on boundary)
  - σ-dependence: CORE (through □σ = −ξR → E_coh → G → k_HP)
"""

import math
from .mechanical import youngs_modulus, shear_modulus, MECHANICAL_DATA
from .surface import MATERIALS
from ..scale import scale_ratio
from ..constants import PROTON_QCD_FRACTION, K_B, SIGMA_HERE


# ── Grain Structure Data ─────────────────────────────────────────
# Rule 9 — If One, Then All: every material gets grain structure data.
#
# sigma_0_Pa: MEASURED lattice friction stress (Pa)
#   (Peierls stress + intrinsic resistance to dislocation motion)
# k_HP_Pa_sqrtm: MEASURED Hall-Petch slope (Pa·√m)
# d_typical_m: typical grain size for annealed material (m)
# Q_gg_eV: activation energy for grain boundary migration (eV)
# K0_gg_m2_s: pre-exponential for grain growth (m²/s)
# d_inverse_HP_m: critical grain size for inverse Hall-Petch (m)
# burgers_m: Burgers vector magnitude (m)
#
# Sources: Armstrong "60 Years of Hall-Petch" (2014),
#          Meyers & Chawla "Mechanical Behavior of Materials" (2009),
#          Humphreys & Hatherly "Recrystallization" (2004).

GRAIN_DATA = {
    'iron': {
        'sigma_0_Pa': 70e6,           # α-Fe lattice friction (BCC, high Peierls)
        'k_HP_Pa_sqrtm': 0.74e6,      # 0.74 MPa√m (strong HP effect in BCC)
        'd_typical_m': 50e-6,          # 50 μm annealed
        'Q_gg_eV': 1.4,               # ≈ 0.5 × Q_self-diffusion
        'K0_gg_m2_s': 1.0e-4,         # Pre-exponential
        'd_inverse_HP_m': 15e-9,       # ~15 nm
        'burgers_m': 2.48e-10,         # a√3/2 for BCC (a=2.867Å)
    },
    'copper': {
        'sigma_0_Pa': 25e6,            # FCC, low Peierls stress
        'k_HP_Pa_sqrtm': 0.11e6,      # 0.11 MPa√m (weak HP in FCC)
        'd_typical_m': 30e-6,          # 30 μm annealed
        'Q_gg_eV': 1.1,               # ≈ 0.5 × Q_self-diffusion
        'K0_gg_m2_s': 5.0e-5,
        'd_inverse_HP_m': 12e-9,       # ~12 nm
        'burgers_m': 2.56e-10,         # a/√2 for FCC (a=3.615Å)
    },
    'aluminum': {
        'sigma_0_Pa': 16e6,            # FCC, very low Peierls
        'k_HP_Pa_sqrtm': 0.07e6,      # 0.07 MPa√m (weak HP)
        'd_typical_m': 60e-6,          # 60 μm annealed
        'Q_gg_eV': 0.8,               # Low activation energy
        'K0_gg_m2_s': 1.0e-3,         # Fast grain growth (low T_melt)
        'd_inverse_HP_m': 10e-9,       # ~10 nm
        'burgers_m': 2.86e-10,         # a/√2 for FCC
    },
    'gold': {
        'sigma_0_Pa': 10e6,            # FCC, very soft
        'k_HP_Pa_sqrtm': 0.08e6,      # 0.08 MPa√m (weak HP)
        'd_typical_m': 40e-6,          # 40 μm annealed
        'Q_gg_eV': 0.9,
        'K0_gg_m2_s': 8.0e-4,
        'd_inverse_HP_m': 10e-9,
        'burgers_m': 2.88e-10,         # a/√2 for FCC
    },
    'silicon': {
        'sigma_0_Pa': 165e6,           # Covalent — very high Peierls stress
        'k_HP_Pa_sqrtm': 0.0,         # Brittle: no dislocation pile-up mechanism
        'd_typical_m': 1e-3,           # Czochralski: often single crystal
        'Q_gg_eV': 3.5,               # Very high (covalent bonds)
        'K0_gg_m2_s': 1.0e-8,         # Very slow
        'd_inverse_HP_m': 0.0,         # N/A for brittle materials
        'burgers_m': 3.84e-10,         # a/√2 for diamond cubic
    },
    'tungsten': {
        'sigma_0_Pa': 350e6,           # BCC refractory, very high Peierls
        'k_HP_Pa_sqrtm': 1.0e6,       # 1.0 MPa√m (strong HP in BCC)
        'd_typical_m': 20e-6,          # 20 μm sintered
        'Q_gg_eV': 3.2,               # High (refractory metal)
        'K0_gg_m2_s': 1.0e-6,
        'd_inverse_HP_m': 20e-9,       # ~20 nm
        'burgers_m': 2.74e-10,         # a√3/2 for BCC
    },
    'nickel': {
        'sigma_0_Pa': 40e6,            # FCC, moderate Peierls
        'k_HP_Pa_sqrtm': 0.16e6,      # 0.16 MPa√m
        'd_typical_m': 35e-6,          # 35 μm annealed
        'Q_gg_eV': 1.5,
        'K0_gg_m2_s': 3.0e-5,
        'd_inverse_HP_m': 14e-9,       # ~14 nm
        'burgers_m': 2.49e-10,         # a/√2 for FCC
    },
    'titanium': {
        'sigma_0_Pa': 80e6,            # HCP, moderate Peierls
        'k_HP_Pa_sqrtm': 0.40e6,      # 0.40 MPa√m (moderate HP)
        'd_typical_m': 40e-6,          # 40 μm annealed
        'Q_gg_eV': 1.4,
        'K0_gg_m2_s': 5.0e-5,
        'd_inverse_HP_m': 15e-9,       # ~15 nm
        'burgers_m': 2.95e-10,         # a for HCP (a=2.95Å)
    },
    # ── Metals ───────────────────────────────────────────────────
    'steel_mild': {
        'sigma_0_Pa': 100e6,           # Ferrite + pearlite friction stress
        'k_HP_Pa_sqrtm': 0.74e6,      # Similar to iron (BCC ferrite matrix)
        'd_typical_m': 25e-6,          # 25 μm hot-rolled
        'Q_gg_eV': 1.5,               # Slightly higher than pure Fe (C pinning)
        'K0_gg_m2_s': 5.0e-5,         # Slower than pure Fe (solute drag)
        'd_inverse_HP_m': 15e-9,       # ~15 nm (same as iron)
        'burgers_m': 2.48e-10,         # BCC ferrite, same as iron
    },
    'lead': {
        'sigma_0_Pa': 5e6,             # FCC, extremely soft
        'k_HP_Pa_sqrtm': 0.03e6,      # 0.03 MPa√m (very weak HP)
        'd_typical_m': 100e-6,         # 100 μm, coarsens easily
        'Q_gg_eV': 0.5,               # Very low (low T_melt = 600 K)
        'K0_gg_m2_s': 5.0e-3,         # Very fast grain growth
        'd_inverse_HP_m': 8e-9,        # ~8 nm
        'burgers_m': 3.50e-10,         # a/√2 for FCC (a=4.95Å)
    },
    'silver': {
        'sigma_0_Pa': 15e6,            # FCC, soft noble metal
        'k_HP_Pa_sqrtm': 0.08e6,      # 0.08 MPa√m (weak HP, like gold)
        'd_typical_m': 40e-6,          # 40 μm annealed
        'Q_gg_eV': 0.9,               # Similar to gold
        'K0_gg_m2_s': 6.0e-4,
        'd_inverse_HP_m': 10e-9,       # ~10 nm
        'burgers_m': 2.89e-10,         # a/√2 for FCC (a=4.086Å)
    },
    'platinum': {
        'sigma_0_Pa': 30e6,            # FCC, moderate friction stress
        'k_HP_Pa_sqrtm': 0.12e6,      # 0.12 MPa√m
        'd_typical_m': 30e-6,          # 30 μm annealed
        'Q_gg_eV': 1.3,               # High T_melt = 2041 K
        'K0_gg_m2_s': 2.0e-5,
        'd_inverse_HP_m': 12e-9,       # ~12 nm
        'burgers_m': 2.77e-10,         # a/√2 for FCC (a=3.924Å)
    },
    'depleted_uranium': {
        'sigma_0_Pa': 120e6,           # Orthorhombic α-U, high Peierls
        'k_HP_Pa_sqrtm': 0.60e6,      # 0.60 MPa√m (strong HP)
        'd_typical_m': 30e-6,          # 30 μm wrought
        'Q_gg_eV': 1.6,               # High activation energy
        'K0_gg_m2_s': 2.0e-5,
        'd_inverse_HP_m': 18e-9,       # ~18 nm
        'burgers_m': 2.85e-10,         # Orthorhombic α-U dominant slip
    },
    # ── Non-crystalline / Amorphous ──────────────────────────────
    'rubber': {
        'sigma_0_Pa': 2e6,             # Yield (onset of permanent set)
        'k_HP_Pa_sqrtm': 0.0,         # No grains, no Hall-Petch
        'd_typical_m': 1.0e-3,         # ~1 mm (no microstructural grains)
        'Q_gg_eV': 5.0,               # No grain growth (amorphous)
        'K0_gg_m2_s': 1.0e-15,        # Negligible
        'd_inverse_HP_m': 0.0,         # N/A
        'burgers_m': 0.0,              # No crystalline slip
    },
    'plastic_abs': {
        'sigma_0_Pa': 40e6,            # Yield stress of ABS
        'k_HP_Pa_sqrtm': 0.0,         # Amorphous, no Hall-Petch
        'd_typical_m': 1.0e-3,         # ~1 mm (no grains)
        'Q_gg_eV': 5.0,               # No grain growth
        'K0_gg_m2_s': 1.0e-15,        # Negligible
        'd_inverse_HP_m': 0.0,         # N/A
        'burgers_m': 0.0,              # No crystalline slip
    },
    'glass': {
        'sigma_0_Pa': 35e6,            # Compressive strength proxy (brittle)
        'k_HP_Pa_sqrtm': 0.0,         # Amorphous, no Hall-Petch
        'd_typical_m': 1.0e-3,         # No grains
        'Q_gg_eV': 5.0,               # No grain growth
        'K0_gg_m2_s': 1.0e-15,        # Negligible
        'd_inverse_HP_m': 0.0,         # N/A
        'burgers_m': 0.0,              # Amorphous
    },
    'concrete': {
        'sigma_0_Pa': 3e6,             # Tensile yield (~3 MPa)
        'k_HP_Pa_sqrtm': 0.0,         # Heterogeneous composite, no HP
        'd_typical_m': 1.0e-3,         # Aggregate scale ~mm
        'Q_gg_eV': 5.0,               # No grain growth
        'K0_gg_m2_s': 1.0e-15,        # Negligible
        'd_inverse_HP_m': 0.0,         # N/A
        'burgers_m': 0.0,              # No crystalline slip
    },
    'wood_oak': {
        'sigma_0_Pa': 40e6,            # Along-grain tensile yield
        'k_HP_Pa_sqrtm': 0.0,         # Biological, no Hall-Petch
        'd_typical_m': 1.0e-3,         # Fiber bundle scale ~mm
        'Q_gg_eV': 5.0,               # No grain growth
        'K0_gg_m2_s': 1.0e-15,        # Negligible
        'd_inverse_HP_m': 0.0,         # N/A
        'burgers_m': 0.0,              # No crystalline slip
    },
    # ── Ceramics / Ice ───────────────────────────────────────────
    'granite': {
        'sigma_0_Pa': 15e6,            # Tensile strength of granite
        'k_HP_Pa_sqrtm': 0.0,         # Brittle polycrystalline aggregate
        'd_typical_m': 2.0e-3,         # 2 mm mineral grain size
        'Q_gg_eV': 4.0,               # Very high (silicate bonds)
        'K0_gg_m2_s': 1.0e-10,        # Extremely slow
        'd_inverse_HP_m': 0.0,         # N/A for brittle
        'burgers_m': 5.0e-10,          # Quartz lattice parameter
    },
    'ceramic_alumina': {
        'sigma_0_Pa': 300e6,           # Flexural strength of Al2O3
        'k_HP_Pa_sqrtm': 0.0,         # Brittle ceramic: no pile-up mechanism
        'd_typical_m': 5.0e-6,         # 5 μm sintered alumina
        'Q_gg_eV': 3.8,               # High (ionic/covalent bonds)
        'K0_gg_m2_s': 1.0e-8,         # Slow grain growth
        'd_inverse_HP_m': 0.0,         # N/A for brittle
        'burgers_m': 4.76e-10,         # Corundum basal slip
    },
    'water_ice': {
        'sigma_0_Pa': 1e6,             # Ice Ih yield (very soft)
        'k_HP_Pa_sqrtm': 0.02e6,      # 0.02 MPa√m (weak HP observed)
        'd_typical_m': 1.0e-3,         # 1 mm typical glacier ice
        'Q_gg_eV': 0.4,               # Low (H-bond migration)
        'K0_gg_m2_s': 1.0e-2,         # Fast grain growth near T_melt
        'd_inverse_HP_m': 0.0,         # Not observed
        'burgers_m': 4.52e-10,         # Ice Ih basal slip a-axis
    },
    # ── Composites / Biological ──────────────────────────────────
    'bone': {
        'sigma_0_Pa': 100e6,           # Cortical bone tensile yield
        'k_HP_Pa_sqrtm': 0.0,         # No dislocation pile-up mechanism
        'd_typical_m': 200e-6,         # Osteon diameter ~200 μm
        'Q_gg_eV': 5.0,               # No grain growth (biological)
        'K0_gg_m2_s': 1.0e-15,        # Negligible
        'd_inverse_HP_m': 0.0,         # N/A
        'burgers_m': 0.0,              # No crystalline slip system
    },
    'carbon_fiber': {
        'sigma_0_Pa': 600e6,           # Composite matrix yield / transverse
        'k_HP_Pa_sqrtm': 0.0,         # Fiber composite, no HP
        'd_typical_m': 7.0e-6,         # Carbon fiber diameter ~7 μm
        'Q_gg_eV': 5.0,               # No grain growth
        'K0_gg_m2_s': 1.0e-15,        # Negligible
        'd_inverse_HP_m': 0.0,         # N/A
        'burgers_m': 3.35e-10,         # Graphite interlayer spacing
    },
    'kevlar': {
        'sigma_0_Pa': 400e6,           # Aramid fiber yield
        'k_HP_Pa_sqrtm': 0.0,         # Polymer fiber composite, no HP
        'd_typical_m': 12.0e-6,        # Kevlar fiber diameter ~12 μm
        'Q_gg_eV': 5.0,               # No grain growth
        'K0_gg_m2_s': 1.0e-15,        # Negligible
        'd_inverse_HP_m': 0.0,         # N/A
        'burgers_m': 0.0,              # Polymer, no crystalline slip
    },
}


# ── Hall-Petch ───────────────────────────────────────────────────

def hall_petch_yield(material_key, grain_size_m, sigma=SIGMA_HERE):
    """Yield stress from Hall-Petch relation (Pa).

    σ_y = σ_0 + k_HP / √d

    Valid for d > d_inverse_HP (typically > ~10 nm).
    Below that, inverse Hall-Petch applies.

    Args:
        material_key: key into GRAIN_DATA
        grain_size_m: mean grain diameter in meters
        sigma: σ-field value

    Returns:
        Yield stress in Pa.
    """
    data = GRAIN_DATA[material_key]
    sigma_0 = data['sigma_0_Pa']
    k_HP = data['k_HP_Pa_sqrtm']
    d_inv = data['d_inverse_HP_m']

    # σ-field scaling
    if sigma != SIGMA_HERE:
        r = scale_ratio(sigma)
        sigma_0 *= r
        k_HP *= r

    if grain_size_m <= 0:
        return sigma_0

    # Check for inverse Hall-Petch regime
    if d_inv > 0 and grain_size_m < d_inv:
        # Use peak stress at d_inv, then decrease
        sy_peak = sigma_0 + k_HP / math.sqrt(d_inv)
        # Linear decrease below critical size (simple model)
        ratio = grain_size_m / d_inv
        return sy_peak * ratio

    return sigma_0 + k_HP / math.sqrt(grain_size_m)


def hall_petch_slope(material_key, sigma=SIGMA_HERE):
    """Hall-Petch slope k_HP (Pa·√m).

    The rate at which yield stress increases with 1/√d.
    Higher for BCC metals (strong Peierls barrier) than FCC.

    Args:
        material_key: key into GRAIN_DATA
        sigma: σ-field value

    Returns:
        k_HP in Pa·√m.
    """
    k = GRAIN_DATA[material_key]['k_HP_Pa_sqrtm']
    if sigma != SIGMA_HERE:
        k *= scale_ratio(sigma)
    return k


# ── Grain Size from Yield Stress ─────────────────────────────────

def grain_size_for_yield(material_key, target_yield_pa, sigma=SIGMA_HERE):
    """Required grain size to achieve a target yield stress (m).

    Inverse Hall-Petch: d = (k_HP / (σ_y - σ_0))²

    Args:
        material_key: key into GRAIN_DATA
        target_yield_pa: desired yield stress (Pa)
        sigma: σ-field value

    Returns:
        Grain diameter in meters. Returns inf if target ≤ σ_0.
    """
    data = GRAIN_DATA[material_key]
    sigma_0 = data['sigma_0_Pa']
    k_HP = data['k_HP_Pa_sqrtm']

    if sigma != SIGMA_HERE:
        r = scale_ratio(sigma)
        sigma_0 *= r
        k_HP *= r

    delta = target_yield_pa - sigma_0

    if delta <= 0 or k_HP <= 0:
        return float('inf')

    return (k_HP / delta) ** 2


# ── Grain Growth Kinetics ────────────────────────────────────────

def grain_growth_rate_constant(material_key, temperature_K, sigma=SIGMA_HERE):
    """Grain growth rate constant K_g (m²/s) at temperature T.

    K_g = K_0 × exp(-Q_gg / (k_B × T))

    Args:
        material_key: key into GRAIN_DATA
        temperature_K: temperature in Kelvin
        sigma: σ-field value

    Returns:
        K_g in m²/s.
    """
    data = GRAIN_DATA[material_key]
    K0 = data['K0_gg_m2_s']
    Q = data['Q_gg_eV']

    # σ-field shifts activation energy through bond strength
    if sigma != SIGMA_HERE:
        Q *= scale_ratio(sigma)

    if temperature_K <= 0:
        return 0.0

    # Q in eV, k_B in eV/K
    k_B_eV = K_B / 1.602176634e-19  # Convert J/K to eV/K
    return K0 * math.exp(-Q / (k_B_eV * temperature_K))


def grain_size_after_anneal(material_key, initial_size_m, temperature_K,
                            time_s, sigma=SIGMA_HERE):
    """Grain size after isothermal annealing (m).

    d² = d₀² + K_g × t    (parabolic grain growth)

    Normal grain growth — curvature-driven boundary migration.

    Args:
        material_key: key into GRAIN_DATA
        initial_size_m: starting grain diameter (m)
        temperature_K: annealing temperature (K)
        time_s: hold time (s)
        sigma: σ-field value

    Returns:
        Final grain diameter in meters.
    """
    if time_s <= 0:
        return initial_size_m

    K_g = grain_growth_rate_constant(material_key, temperature_K, sigma)
    d_sq = initial_size_m ** 2 + K_g * time_s
    return math.sqrt(max(d_sq, 0.0))


def time_to_grain_size(material_key, initial_size_m, target_size_m,
                       temperature_K, sigma=SIGMA_HERE):
    """Time required to grow grains to a target size (s).

    t = (d² - d₀²) / K_g

    Args:
        material_key: key into GRAIN_DATA
        initial_size_m: starting grain size (m)
        target_size_m: desired grain size (m)
        temperature_K: annealing temperature (K)
        sigma: σ-field value

    Returns:
        Time in seconds. Returns inf if K_g ≈ 0.
    """
    if target_size_m <= initial_size_m:
        return 0.0

    K_g = grain_growth_rate_constant(material_key, temperature_K, sigma)

    if K_g <= 1e-50:
        return float('inf')

    return (target_size_m ** 2 - initial_size_m ** 2) / K_g


# ── Zener Pinning ────────────────────────────────────────────────

def zener_limit(particle_radius_m, volume_fraction):
    """Maximum grain size limited by Zener pinning (m).

    d_max = 4r / (3f)

    Second-phase particles exert a drag force on grain boundaries,
    limiting coarsening. This is why precipitation-hardened alloys
    retain fine grain size at high temperatures.

    FIRST_PRINCIPLES: force balance between boundary curvature
    driving force and particle pinning force.

    Args:
        particle_radius_m: mean particle radius (m)
        volume_fraction: volume fraction of particles (0 to 1)

    Returns:
        Maximum grain size in meters.
    """
    if volume_fraction <= 0 or particle_radius_m <= 0:
        return float('inf')

    return 4.0 * particle_radius_m / (3.0 * volume_fraction)


# ── Grain Boundary Area ─────────────────────────────────────────

def grain_boundary_area_per_volume(grain_size_m):
    """Grain boundary area per unit volume (1/m).

    S_v = 2/d  (for equiaxed grains, stereological relation)

    This is the driving force for grain growth — higher S_v means
    more boundary energy to be eliminated.

    Args:
        grain_size_m: mean grain diameter (m)

    Returns:
        Grain boundary area per volume in 1/m.
    """
    if grain_size_m <= 0:
        return 0.0
    return 2.0 / grain_size_m


def grain_boundary_energy_density(material_key, grain_size_m, sigma=SIGMA_HERE):
    """Total grain boundary energy per unit volume (J/m³).

    E_gb = γ_gb × S_v = γ_gb × 2/d

    Where γ_gb ≈ γ_surface / 3 (grain boundary energy is roughly
    1/3 of the free surface energy for metals).

    APPROXIMATION: γ_gb/γ_s ≈ 1/3 (Read-Shockley for high-angle boundaries).

    Args:
        material_key: key into MATERIALS
        grain_size_m: grain diameter (m)
        sigma: σ-field value

    Returns:
        Energy density in J/m³.
    """
    from .surface import surface_energy_at_sigma

    gamma_s = surface_energy_at_sigma(material_key, sigma)
    gamma_gb = gamma_s / 3.0  # Approximation for high-angle GB
    S_v = grain_boundary_area_per_volume(grain_size_m)
    return gamma_gb * S_v


# ── Dislocation Density Estimate ─────────────────────────────────

def dislocation_density_estimate(material_key, plastic_strain):
    """Estimate dislocation density from plastic strain (1/m²).

    ρ ≈ ε_p / (b × L)

    Where:
      b = Burgers vector
      L ≈ mean dislocation free path ≈ 1/√ρ (self-consistent)

    Solving: ρ ≈ (ε_p / b)^(2/3) × ρ_0^(1/3)

    Simpler Kocks-Mecking approximation:
      ρ ≈ ε_p / (b² × 1000)

    This gives order-of-magnitude estimates:
      Annealed: ρ ~ 10¹⁰ /m²
      Cold-worked: ρ ~ 10¹⁴-10¹⁵ /m²

    APPROXIMATION: highly simplified from full Kocks-Mecking model.

    Args:
        material_key: key into GRAIN_DATA
        plastic_strain: accumulated plastic strain

    Returns:
        Dislocation density in 1/m².
    """
    b = GRAIN_DATA[material_key]['burgers_m']
    rho_0 = 1e10  # Annealed dislocation density (1/m²)

    if plastic_strain <= 0:
        return rho_0

    # Kocks-Mecking simplified: ρ grows with strain
    rho = rho_0 + plastic_strain / (b * b * 1000.0)
    return rho


# ── Taylor Hardening ─────────────────────────────────────────────

def taylor_hardening_stress(material_key, dislocation_density, sigma=SIGMA_HERE):
    """Flow stress from Taylor hardening (Pa).

    σ = α × M × G × b × √ρ

    Where:
      α ≈ 0.3 (interaction strength between dislocations)
      M ≈ 3.06 (Taylor factor for FCC polycrystal)
      G = shear modulus
      b = Burgers vector
      ρ = dislocation density

    FIRST_PRINCIPLES: each dislocation creates a stress field;
    the forest of dislocations impedes motion of mobile dislocations.
    √ρ dependence from mean spacing between forest dislocations = 1/√ρ.

    Args:
        material_key: key into GRAIN_DATA
        dislocation_density: dislocation density (1/m²)
        sigma: σ-field value

    Returns:
        Flow stress contribution from dislocation hardening (Pa).
    """
    alpha = 0.3  # Interaction coefficient (MEASURED: Basinski & Basinski 1979)
    # Taylor factor — MEASURED per crystal structure (polycrystal slip simulations)
    #   FCC: Bishop & Hill 1951
    #   BCC: Kocks 1970
    #   HCP: Hosford 1993
    _TAYLOR_FACTOR = {
        'fcc': 3.06,
        'bcc': 2.75,
        'hcp': 4.5,
        'diamond_cubic': 3.06,  # same slip geometry as FCC
    }
    struct = MATERIALS[material_key]['crystal_structure']
    M = _TAYLOR_FACTOR.get(struct, 3.06)
    G = shear_modulus(material_key, sigma)
    b = GRAIN_DATA[material_key]['burgers_m']

    if dislocation_density <= 0:
        return 0.0

    return alpha * M * G * b * math.sqrt(dislocation_density)


# ── Combined Strength (Hall-Petch + Taylor) ──────────────────────

def polycrystal_yield(material_key, grain_size_m, plastic_strain=0.0,
                      sigma=SIGMA_HERE):
    """Combined yield/flow stress for a polycrystalline material (Pa).

    σ = σ_0 + k_HP/√d + α M G b √ρ

    Three additive contributions:
      1. Lattice friction (Peierls stress)
      2. Grain boundary strengthening (Hall-Petch)
      3. Dislocation forest hardening (Taylor)

    This is the "additive strengthening" model (Kocks 1970).

    Args:
        material_key: key into GRAIN_DATA
        grain_size_m: grain diameter (m)
        plastic_strain: accumulated plastic strain (for work hardening)
        sigma: σ-field value

    Returns:
        Total flow stress in Pa.
    """
    # Hall-Petch (includes σ_0)
    sy_hp = hall_petch_yield(material_key, grain_size_m, sigma)

    # Taylor hardening contribution
    if plastic_strain > 0:
        rho = dislocation_density_estimate(material_key, plastic_strain)
        rho_0 = 1e10  # Subtract annealed baseline (already in σ_0)
        delta_rho = max(rho - rho_0, 0.0)
        sy_taylor = taylor_hardening_stress(material_key, delta_rho, sigma)
    else:
        sy_taylor = 0.0

    return sy_hp + sy_taylor


# ── Annealing Profile ───────────────────────────────────────────

def annealing_profile(material_key, initial_size_m, temperature_K,
                      total_time_s, steps=100, sigma=SIGMA_HERE):
    """Simulate grain growth during isothermal annealing.

    Returns grain size, yield stress, and GB energy at each time step.

    Args:
        material_key: key into GRAIN_DATA
        initial_size_m: starting grain size (m)
        temperature_K: annealing temperature (K)
        total_time_s: total anneal time (s)
        steps: number of time steps
        sigma: σ-field value

    Returns:
        List of dicts with time_s, grain_size_m, yield_stress_pa, gb_energy_J_m3.
    """
    if steps < 1:
        steps = 1

    profile = []
    for i in range(steps + 1):
        t = total_time_s * i / steps
        d = grain_size_after_anneal(material_key, initial_size_m,
                                    temperature_K, t, sigma)
        sy = hall_petch_yield(material_key, d, sigma)
        E_gb = grain_boundary_energy_density(material_key, d, sigma)

        profile.append({
            'time_s': t,
            'grain_size_m': d,
            'yield_stress_pa': sy,
            'gb_energy_J_m3': E_gb,
        })

    return profile


# ── σ-field Coupling ─────────────────────────────────────────────

def sigma_hall_petch_shift(material_key, grain_size_m, sigma):
    """Ratio of HP yield stress at σ to HP yield stress at σ=0.

    Both σ_0 and k_HP scale with bond strength.

    Args:
        material_key: material key
        grain_size_m: grain diameter (m)
        sigma: σ-field value

    Returns:
        Ratio (dimensionless).
    """
    if sigma == SIGMA_HERE:
        return 1.0

    sy_0 = hall_petch_yield(material_key, grain_size_m, SIGMA_HERE)
    sy_s = hall_petch_yield(material_key, grain_size_m, sigma)

    if sy_0 <= 0:
        return 1.0

    return sy_s / sy_0


# ── Nagatha Export ───────────────────────────────────────────────

def grain_structure_properties(material_key, grain_size_m=None, sigma=SIGMA_HERE):
    """Export grain structure properties in Nagatha-compatible format.

    Args:
        material_key: key into GRAIN_DATA
        grain_size_m: grain size (default: typical for material)
        sigma: σ-field value

    Returns:
        Dict with grain structure properties and origin tags.
    """
    data = GRAIN_DATA[material_key]
    d = grain_size_m if grain_size_m is not None else data['d_typical_m']

    sy = hall_petch_yield(material_key, d, sigma)
    k_HP = hall_petch_slope(material_key, sigma)
    S_v = grain_boundary_area_per_volume(d)
    E_gb = grain_boundary_energy_density(material_key, d, sigma)
    rho_0 = dislocation_density_estimate(material_key, 0.0)
    shift = sigma_hall_petch_shift(material_key, d, sigma)

    return {
        'material': material_key,
        'sigma': sigma,
        'grain_size_m': d,
        'sigma_0_Pa': data['sigma_0_Pa'],
        'k_HP_Pa_sqrtm': k_HP,
        'yield_stress_pa': sy,
        'gb_area_per_volume_1_m': S_v,
        'gb_energy_density_J_m3': E_gb,
        'dislocation_density_annealed_1_m2': rho_0,
        'burgers_vector_m': data['burgers_m'],
        'd_inverse_HP_m': data['d_inverse_HP_m'],
        'sigma_hp_ratio': shift,
        'origin': (
            "Hall-Petch: σ_y = σ_0 + k_HP/√d (FIRST_PRINCIPLES pile-up model, "
            "Eshelby-Frank-Nabarro 1951). σ_0, k_HP: MEASURED. "
            "Grain growth: d²-d₀² = K_g×t (FIRST_PRINCIPLES curvature-driven, "
            "Burke-Turnbull 1952). K_g: MEASURED (Arrhenius). "
            "Taylor hardening: σ = αMGb√ρ (FIRST_PRINCIPLES dislocation interaction). "
            "Zener pinning: d_max = 4r/3f (FIRST_PRINCIPLES boundary-particle). "
            "σ-coupling: CORE (□σ = −ξR → E_coh → G → k_HP, Q_gg)."
        ),
    }
