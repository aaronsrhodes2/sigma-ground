"""
Wear physics — material removal under sliding contact.

The destructive consequence of friction. Links to hardness (from friction.py),
which links to yield stress through the Tabor relation.

Derivation chains:

  1. Archard Wear Law (1953, FIRST_PRINCIPLES form)
     V = K × F_n × d / H

     Where:
       V = volume of material removed (m³)
       K = dimensionless wear coefficient (MEASURED, 10⁻² to 10⁻⁷)
       F_n = normal force (N)
       d = sliding distance (m)
       H = hardness of softer material (Pa)

     FIRST_PRINCIPLES reasoning: each asperity contact has a probability K
     of producing a wear particle. Number of contacts ∝ F_n/H (real contact
     area). Each contact sweeps volume ∝ d × contact radius².
     Net: V ∝ K F_n d / H.

     K is the key material parameter — ranges from ~10⁻² (severe adhesive
     wear, unlubricated metals) to ~10⁻⁷ (mild oxidative wear, lubricated).

  2. Specific Wear Rate (FIRST_PRINCIPLES)
     k = K / H   (m³/(N·m) = m²/N = 1/Pa)

     Volume removed per unit load per unit sliding distance.
     More directly measurable than K alone.

  3. Wear Depth (FIRST_PRINCIPLES)
     h = K × p × d / H = k × p × d

     Where p = contact pressure (F_n / A_apparent).
     Wear depth = specific wear rate × pressure × distance.

  4. Sliding Wear Rate (FIRST_PRINCIPLES)
     dV/dt = K × F_n × v / H

     Where v = sliding velocity (m/s).
     Time-rate of volume removal.

  5. Adhesive-to-Abrasive Transition
     When hardness ratio H_abrasive/H_surface > ~1.2, the harder asperities
     cut into the softer surface (abrasive wear). Below this threshold,
     adhesive junction shearing dominates.

     K_abrasive ≈ K_adhesive × (H_ratio - 1.0) × 10
     (APPROXIMATION: empirical scaling for the transition)

  6. Oxidative Wear (Quinn 1962, FIRST_PRINCIPLES form)
     At moderate speeds, frictional heating generates oxide films at
     asperity contacts. Wear rate depends on oxide growth kinetics:

     V_oxide = A_ox × exp(-Q_ox/(RT_flash)) × F_n × d / H

     Flash temperature rise: ΔT ≈ μ × p × v × l / (4 × κ)
     where l = contact size, κ = thermal diffusivity.

     We use a simpler model: K_effective decreases when surface oxide
     is protective (Pilling-Bedworth ratio 1-2, from corrosion.py).

σ-dependence:
  σ → E_coh → G → τ_theoretical → H → wear rate
  Harder materials (higher σ) wear less. Both through H in the denominator
  and through shifts in K (stronger bonds → more resistant junctions).

Origin tags:
  - Archard law: FIRST_PRINCIPLES (contact mechanics, probabilistic)
  - K values: MEASURED (pin-on-disk, block-on-ring tests)
  - Specific wear rate: FIRST_PRINCIPLES (dimensional analysis)
  - Abrasive transition: APPROXIMATION (empirical hardness ratio)
  - σ-dependence: CORE (through □σ = −ξR → E_coh → H)
"""

import math
from .friction import _hardness
from .surface import MATERIALS
from ..scale import scale_ratio


# ── Wear Coefficient Data ────────────────────────────────────────
# Rule 9 — If One, Then All: every material in MATERIALS gets wear data.
#
# K_adhesive: MEASURED dry adhesive wear coefficient (dimensionless)
# K_abrasive: MEASURED abrasive wear coefficient (dimensionless)
# K_lubricated: MEASURED lubricated wear coefficient (dimensionless)
#
# Values from Rabinowicz "Friction and Wear of Materials" (1995),
# ASM Handbook Vol. 18 "Friction, Lubrication, and Wear Technology",
# Hutchings & Shipway "Tribology" (2017).
#
# K ranges:
#   Severe adhesive (metal-on-metal, dry): 10⁻² to 10⁻³
#   Mild adhesive (with oxide film): 10⁻⁴ to 10⁻⁵
#   Abrasive (hard particle cutting): 10⁻¹ to 10⁻²
#   Lubricated: 10⁻⁶ to 10⁻⁷

WEAR_DATA = {
    'iron': {
        'K_adhesive': 7.0e-3,       # Steel-on-steel, dry
        'K_abrasive': 5.0e-2,       # Abrasive grinding
        'K_lubricated': 1.0e-6,     # Oil-lubricated
    },
    'copper': {
        'K_adhesive': 1.5e-2,       # High adhesion, soft FCC
        'K_abrasive': 3.0e-2,       # Moderately soft
        'K_lubricated': 5.0e-6,     # Lubricated
    },
    'aluminum': {
        'K_adhesive': 2.0e-2,       # Very adhesive, galls easily
        'K_abrasive': 2.5e-2,       # Soft, easy to cut
        'K_lubricated': 3.0e-6,     # Lubricated
    },
    'gold': {
        'K_adhesive': 3.0e-2,       # Very soft, high adhesion
        'K_abrasive': 4.0e-2,       # Very soft
        'K_lubricated': 8.0e-6,     # Lubricated
    },
    'silicon': {
        'K_adhesive': 1.0e-4,       # Brittle — fracture, not adhesion
        'K_abrasive': 1.0e-2,       # Brittle chipping
        'K_lubricated': 5.0e-7,     # Very hard surface
    },
    'tungsten': {
        'K_adhesive': 5.0e-4,       # Very hard, low real contact area
        'K_abrasive': 1.0e-2,       # Hard but brittle at asperities
        'K_lubricated': 2.0e-7,     # Hard + lubricated
    },
    'titanium': {
        'K_adhesive': 5.0e-3,       # Moderate — oxide helps
        'K_abrasive': 2.0e-2,       # Medium hardness
        'K_lubricated': 1.0e-6,     # Lubricated
    },
    'nickel': {
        'K_adhesive': 8.0e-3,       # FCC, adhesive
        'K_abrasive': 3.5e-2,       # Moderate hardness
        'K_lubricated': 2.0e-6,     # Lubricated
    },
}


# ── Archard Wear Law ─────────────────────────────────────────────

def archard_wear_volume(material_key, normal_force_n, sliding_distance_m,
                        wear_mode='adhesive', sigma=0.0):
    """Volume of material removed by sliding wear (m³).

    V = K × F_n × d / H    (Archard 1953)

    FIRST_PRINCIPLES: probabilistic model of asperity contact and
    material removal. Each junction has probability K of producing
    a loose particle.

    Args:
        material_key: key into MATERIALS/WEAR_DATA
        normal_force_n: applied normal force (N)
        sliding_distance_m: total sliding distance (m)
        wear_mode: 'adhesive', 'abrasive', or 'lubricated'
        sigma: σ-field value

    Returns:
        Wear volume in m³.
    """
    if normal_force_n <= 0 or sliding_distance_m <= 0:
        return 0.0

    K = _get_wear_coefficient(material_key, wear_mode)
    H = _hardness(material_key, sigma)

    if H <= 0:
        return 0.0

    return K * normal_force_n * sliding_distance_m / H


def _get_wear_coefficient(material_key, wear_mode='adhesive'):
    """Look up wear coefficient K for given mode."""
    data = WEAR_DATA[material_key]
    key = f'K_{wear_mode}'
    if key not in data:
        raise ValueError(f"Unknown wear mode '{wear_mode}'. "
                         f"Use 'adhesive', 'abrasive', or 'lubricated'.")
    return data[key]


# ── Specific Wear Rate ───────────────────────────────────────────

def specific_wear_rate(material_key, wear_mode='adhesive', sigma=0.0):
    """Specific wear rate k = K/H (m²/N = m³/(N·m)).

    Volume removed per unit load per unit sliding distance.
    This is the quantity most directly measured in pin-on-disk tests.

    FIRST_PRINCIPLES: direct from Archard law, k = V/(F_n × d) = K/H.

    Args:
        material_key: key into MATERIALS/WEAR_DATA
        wear_mode: 'adhesive', 'abrasive', or 'lubricated'
        sigma: σ-field value

    Returns:
        Specific wear rate in m²/N (equivalently m³/(N·m)).
    """
    K = _get_wear_coefficient(material_key, wear_mode)
    H = _hardness(material_key, sigma)

    if H <= 0:
        return 0.0

    return K / H


# ── Wear Depth ───────────────────────────────────────────────────

def wear_depth(material_key, contact_pressure_pa, sliding_distance_m,
               wear_mode='adhesive', sigma=0.0):
    """Depth of material removed by wear (m).

    h = k × p × d = K × p × d / H

    For a flat contact of area A under load F_n:
      V = K × F_n × d / H
      h = V / A = K × (F_n/A) × d / H = K × p × d / H

    Args:
        material_key: key into MATERIALS/WEAR_DATA
        contact_pressure_pa: nominal contact pressure (Pa)
        sliding_distance_m: sliding distance (m)
        wear_mode: 'adhesive', 'abrasive', or 'lubricated'
        sigma: σ-field value

    Returns:
        Wear depth in meters.
    """
    if contact_pressure_pa <= 0 or sliding_distance_m <= 0:
        return 0.0

    k = specific_wear_rate(material_key, wear_mode, sigma)
    return k * contact_pressure_pa * sliding_distance_m


# ── Sliding Wear Rate (time-based) ──────────────────────────────

def sliding_wear_rate(material_key, normal_force_n, velocity_m_s,
                      wear_mode='adhesive', sigma=0.0):
    """Volume removal rate during sliding (m³/s).

    dV/dt = K × F_n × v / H

    Args:
        material_key: key into MATERIALS/WEAR_DATA
        normal_force_n: normal force (N)
        velocity_m_s: sliding velocity (m/s)
        wear_mode: 'adhesive', 'abrasive', or 'lubricated'
        sigma: σ-field value

    Returns:
        Volumetric wear rate in m³/s.
    """
    if normal_force_n <= 0 or velocity_m_s <= 0:
        return 0.0

    K = _get_wear_coefficient(material_key, wear_mode)
    H = _hardness(material_key, sigma)

    if H <= 0:
        return 0.0

    return K * normal_force_n * velocity_m_s / H


# ── Wear Mass Loss ──────────────────────────────────────────────

def wear_mass_loss(material_key, normal_force_n, sliding_distance_m,
                   wear_mode='adhesive', sigma=0.0):
    """Mass of material removed by wear (kg).

    m = ρ × V = ρ × K × F_n × d / H

    Args:
        material_key: key into MATERIALS/WEAR_DATA
        normal_force_n: normal force (N)
        sliding_distance_m: sliding distance (m)
        wear_mode: 'adhesive', 'abrasive', or 'lubricated'
        sigma: σ-field value

    Returns:
        Mass lost in kg.
    """
    V = archard_wear_volume(material_key, normal_force_n, sliding_distance_m,
                            wear_mode, sigma)
    rho = MATERIALS[material_key]['density_kg_m3']
    return rho * V


# ── Hardness Ratio Classification ────────────────────────────────

def wear_regime(mat_surface, mat_counter, sigma=0.0):
    """Classify the wear regime based on hardness ratio.

    When H_counter / H_surface > 1.2: abrasive regime
    When H_counter / H_surface ≈ 1.0: adhesive regime
    When H_counter / H_surface < 0.8: counter-body wears preferentially

    APPROXIMATION: threshold values are empirical (Hutchings 2017).

    Args:
        mat_surface: material being worn
        mat_counter: counter-body material
        sigma: σ-field value

    Returns:
        String: 'abrasive', 'adhesive', or 'counter-body-wears'
    """
    H_surface = _hardness(mat_surface, sigma)
    H_counter = _hardness(mat_counter, sigma)

    if H_surface <= 0 or H_counter <= 0:
        return 'adhesive'

    ratio = H_counter / H_surface

    if ratio > 1.2:
        return 'abrasive'
    elif ratio < 0.8:
        return 'counter-body-wears'
    else:
        return 'adhesive'


# ── Sliding Distance to Failure ──────────────────────────────────

def sliding_distance_to_depth(material_key, target_depth_m,
                              contact_pressure_pa,
                              wear_mode='adhesive', sigma=0.0):
    """Sliding distance required to reach a given wear depth (m).

    d = h × H / (K × p)

    Inverse of wear_depth(): given a tolerable wear depth, how far
    can the component slide before replacement?

    Args:
        material_key: key into MATERIALS/WEAR_DATA
        target_depth_m: acceptable wear depth (m)
        contact_pressure_pa: contact pressure (Pa)
        wear_mode: 'adhesive', 'abrasive', or 'lubricated'
        sigma: σ-field value

    Returns:
        Sliding distance in meters. Returns float('inf') if wear rate is zero.
    """
    if target_depth_m <= 0 or contact_pressure_pa <= 0:
        return 0.0

    k = specific_wear_rate(material_key, wear_mode, sigma)

    if k <= 0:
        return float('inf')

    return target_depth_m / (k * contact_pressure_pa)


# ── Comparative Wear Resistance ──────────────────────────────────

def relative_wear_resistance(material_key, reference_key='iron',
                             wear_mode='adhesive', sigma=0.0):
    """Wear resistance relative to a reference material (dimensionless).

    R = k_ref / k_mat = (K_ref / H_ref) / (K_mat / H_mat)

    Higher value = more wear-resistant than the reference.

    Args:
        material_key: material to evaluate
        reference_key: reference material (default: iron/steel)
        wear_mode: 'adhesive', 'abrasive', or 'lubricated'
        sigma: σ-field value

    Returns:
        Relative wear resistance (dimensionless).
    """
    k_mat = specific_wear_rate(material_key, wear_mode, sigma)
    k_ref = specific_wear_rate(reference_key, wear_mode, sigma)

    if k_mat <= 0:
        return float('inf')
    if k_ref <= 0:
        return 0.0

    return k_ref / k_mat


# ── σ-field Coupling ─────────────────────────────────────────────

def sigma_wear_shift(material_key, sigma, wear_mode='adhesive'):
    """Ratio of wear rate at σ to wear rate at σ=0.

    Wear rate ∝ 1/H. Since H ∝ G ∝ E_coh, and E_coh scales with σ:
      H(σ) / H(0) = scale_ratio(σ)
      wear(σ) / wear(0) = H(0) / H(σ) = 1 / scale_ratio(σ)

    Positive σ (stronger bonds) → lower wear rate.

    CORE: through □σ = −ξR → E_coh → G → H → V_wear.

    Args:
        material_key: material key
        sigma: σ-field value
        wear_mode: 'adhesive', 'abrasive', or 'lubricated'

    Returns:
        Ratio wear(σ)/wear(0). Values < 1 mean less wear.
    """
    if sigma == 0.0:
        return 1.0

    H_0 = _hardness(material_key, 0.0)
    H_s = _hardness(material_key, sigma)

    if H_0 <= 0 or H_s <= 0:
        return 1.0

    return H_0 / H_s


# ── Wear Profile Simulation ─────────────────────────────────────

def wear_profile(material_key, normal_force_n, velocity_m_s,
                 total_time_s, steps=100, wear_mode='adhesive', sigma=0.0):
    """Simulate wear depth vs time for a sliding contact.

    Returns a list of (time_s, depth_m, volume_m3, mass_kg) tuples.

    Useful for lifetime prediction: at what time does wear depth
    reach the tolerance limit?

    Args:
        material_key: material key
        normal_force_n: constant normal force (N)
        velocity_m_s: constant sliding velocity (m/s)
        total_time_s: total simulation time (s)
        steps: number of time steps
        wear_mode: 'adhesive', 'abrasive', or 'lubricated'
        sigma: σ-field value

    Returns:
        List of dicts with time_s, distance_m, depth_m, volume_m3, mass_kg.
    """
    if steps < 1:
        steps = 1

    rho = MATERIALS[material_key]['density_kg_m3']
    K = _get_wear_coefficient(material_key, wear_mode)
    H = _hardness(material_key, sigma)

    profile = []
    for i in range(steps + 1):
        t = total_time_s * i / steps
        d = velocity_m_s * t  # sliding distance
        V = K * normal_force_n * d / H if H > 0 else 0.0
        m = rho * V

        # Depth requires knowing contact area — we report volume
        # and let the caller divide by their contact area.
        # But also give depth assuming 1 m² for convenience.
        h = K * (normal_force_n / 1.0) * d / H if H > 0 else 0.0

        profile.append({
            'time_s': t,
            'distance_m': d,
            'volume_m3': V,
            'mass_kg': m,
            'depth_m_per_m2': h,
        })

    return profile


# ── Nagatha Export ───────────────────────────────────────────────

def wear_properties(material_key, sigma=0.0):
    """Export wear properties in Nagatha-compatible format.

    Args:
        material_key: key into MATERIALS/WEAR_DATA
        sigma: σ-field value

    Returns:
        Dict with wear properties and origin tags.
    """
    data = WEAR_DATA[material_key]
    H = _hardness(material_key, sigma)

    k_adh = specific_wear_rate(material_key, 'adhesive', sigma)
    k_abr = specific_wear_rate(material_key, 'abrasive', sigma)
    k_lub = specific_wear_rate(material_key, 'lubricated', sigma)

    sigma_shift = sigma_wear_shift(material_key, sigma)
    rel_resist = relative_wear_resistance(material_key, 'iron', 'adhesive', sigma)

    return {
        'material': material_key,
        'sigma': sigma,
        'hardness_pa': H,
        'K_adhesive': data['K_adhesive'],
        'K_abrasive': data['K_abrasive'],
        'K_lubricated': data['K_lubricated'],
        'specific_wear_rate_adhesive_m2_N': k_adh,
        'specific_wear_rate_abrasive_m2_N': k_abr,
        'specific_wear_rate_lubricated_m2_N': k_lub,
        'sigma_wear_ratio': sigma_shift,
        'relative_wear_resistance_vs_iron': rel_resist,
        'origin': (
            "Archard wear law (1953): FIRST_PRINCIPLES (probabilistic "
            "asperity contact model, V = K×F×d/H). "
            "K values: MEASURED (pin-on-disk, block-on-ring tests). "
            "Hardness from Frenkel shear → Tabor relation: "
            "FIRST_PRINCIPLES + APPROXIMATION. "
            "σ-coupling: CORE (□σ = −ξR → E_coh → G → H → wear)."
        ),
    }
