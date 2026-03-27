"""
Corrosion — oxidation kinetics, Pilling-Bedworth ratio, galvanic series.

Derivation chains:

  1. Pilling-Bedworth Ratio (FIRST_PRINCIPLES: geometry)
     PBR = (M_ox × ρ_metal) / (n × M_metal × ρ_oxide)

     Where:
       M_ox    = molar mass of oxide (g/mol)
       ρ_metal = density of metal (kg/m³)
       n       = number of metal atoms per oxide formula unit
       M_metal = molar mass of metal (g/mol)
       ρ_oxide = density of oxide (kg/m³)

     FIRST_PRINCIPLES: molar volume ratio of oxide to metal consumed.
     PBR < 1: oxide volume < metal volume → porous, non-protective (e.g., Mg)
     1 < PBR < 2: compressive stress but intact → protective film (e.g., Al, Ti)
     PBR > 2: tensile stress → film spalls → non-protective (e.g., Fe rust)

     Pilling & Bedworth (1923). No free parameters — pure geometry.

  2. Parabolic Oxidation (Wagner 1933, FIRST_PRINCIPLES)
     x² = k_eff(T) × t

     Where:
       x      = oxide thickness (m)
       k_eff  = effective parabolic rate constant (m²/s)
       t      = time (s)

     Temperature dependence (Arrhenius):
       k_eff(T) = k_ref × exp(Q/kT_ref − Q/kT)

     Where k_ref is the measured rate at T_ref = 300 K.

     FIRST_PRINCIPLES: rate-limiting step is solid-state diffusion of O²⁻ or
     metal cations through the growing oxide layer (Wagner's high-temperature
     oxidation theory). The parabolic law follows from Fick's first law across
     an oxide of growing thickness.

     The Arrhenius factor is exact for thermally activated diffusion.

  3. Galvanic Corrosion (FIRST_PRINCIPLES: electrochemistry)
     ΔE = E_a − E_b (V)

     The metal with more negative E_standard is anodic (corrodes).
     The metal with more positive E_standard is cathodic (protected).

     FIRST_PRINCIPLES: spontaneous oxidation is driven by ΔG = −nFΔE.
     When ΔE > 0 (cathode − anode convention), reaction is spontaneous.

  4. Corrosion Mass Loss Rate (APPROXIMATION)
     Derived from parabolic growth rate dx/dt = k_eff / (2x) at t = 1 s
     (initial rate before protective layer forms significantly).

     dm/dt = ρ_oxide × (dx/dt) = ρ_oxide × (k_eff)^(1/2) / 2   (kg/m²/s)

     APPROXIMATION: uses the parabolic law instantaneous rate at t = 1 s.
     Valid for initial-stage oxidation. For long times the rate falls as 1/√t.

σ-dependence:
  Activation energy Q scales with cohesive energy E_coh.
  Stiffer lattices (higher E_coh) suppress diffusion → slower oxidation.

  Q(σ) = Q_ref × (E_coh(σ) / E_coh_ref)

  E_coh(σ) shifts through the nuclear mass fraction:
    E_coh(σ) ≈ E_coh_ref × (1 + f_QCD × (scale_ratio(σ) − 1))

  where f_QCD ≈ 8% is the fraction of cohesive energy from QCD-mass
  (PROTON_QCD_FRACTION from constants). The dominant EM bond energy is
  σ-invariant; only the phonon/lattice stiffness piece shifts.

  At Earth (σ ~ 7×10⁻¹⁰): correction < 10⁻⁹, negligible.
  At σ_conv ~ 1.85: lattice destroyed, concept breaks down.

Origin tags:
  - Pilling-Bedworth ratio: FIRST_PRINCIPLES (geometry, Pilling & Bedworth 1923)
  - Parabolic law: FIRST_PRINCIPLES (Wagner 1933, Fickian diffusion)
  - Galvanic potential: FIRST_PRINCIPLES (electrochemistry, Gibbs / Nernst)
  - k_parabolic at 300 K: MEASURED (CRC Handbook, Birks, Meier & Pettit 2006)
  - Q_oxidation: MEASURED (Birks et al. 2006, Kofstad 1988)
  - Oxide densities / molar masses: MEASURED (CRC Handbook)
  - Standard electrode potentials: MEASURED (IUPAC, Bard et al. 1985)
  - σ-dependence: CORE (through □σ = −ξR via cohesive energy / nuclear mass)
"""

import math
from .surface import MATERIALS
from ..constants import K_B, PROTON_QCD_FRACTION

# ── Conversion ─────────────────────────────────────────────────────────────
_EV_TO_JOULE = 1.602176634e-19  # exact (2019 SI definition)

# ── Corrosion Material Database ─────────────────────────────────────────────
# All values are MEASURED.
# Sources: CRC Handbook of Chemistry and Physics (103rd ed.);
#          Birks, Meier & Pettit "Introduction to the High-Temperature
#          Oxidation of Metals" (2006);
#          Kofstad "High Temperature Corrosion" (1988);
#          Bard, Parsons & Jordan "Standard Potentials" (1985, IUPAC).
#
# Rule 9 compliance: every material carries every field.

CORROSION_DATA = {
    'iron': {
        # Standard electrode potential vs SHE, Fe → Fe²⁺ + 2e⁻ (MEASURED)
        'E_standard_V': -0.44,
        # Primary oxide: Fe₂O₃ (hematite), density (MEASURED, CRC)
        'oxide_density_kg_m3': 5250,
        # Molar mass of Fe₂O₃ (MEASURED): 2×55.845 + 3×15.999 = 159.687 g/mol
        'oxide_molar_mass_g': 159.69,
        # Molar mass of Fe (MEASURED, IUPAC 2021): 55.845 g/mol
        'metal_molar_mass_g': 55.845,
        # Fe₂O₃ has 2 iron atoms per formula unit
        'n_oxide_metal_atoms': 2,
        # Primary oxide name
        'oxide_name': 'Fe2O3',
        # Parabolic rate constant at 300 K (MEASURED, Birks et al. 2006)
        'k_parabolic_m2_s': 1.0e-21,
        # Activation energy for oxidation (MEASURED, Kofstad 1988)
        'Q_oxidation_eV': 1.0,
    },
    'copper': {
        # Cu → Cu⁺ + e⁻, Cu₂O (cuprite) is primary room-T oxide (MEASURED)
        'E_standard_V': +0.34,
        'oxide_density_kg_m3': 6100,
        # Molar mass of Cu₂O: 2×63.546 + 15.999 = 143.091 g/mol
        'oxide_molar_mass_g': 143.09,
        # Molar mass of Cu (MEASURED, IUPAC 2021)
        'metal_molar_mass_g': 63.546,
        'n_oxide_metal_atoms': 2,
        'oxide_name': 'Cu2O',
        'k_parabolic_m2_s': 1.0e-22,
        'Q_oxidation_eV': 0.8,
    },
    'aluminum': {
        # Al → Al³⁺ + 3e⁻ (MEASURED, IUPAC)
        'E_standard_V': -1.66,
        # Al₂O₃ (corundum/α-alumina), density (MEASURED, CRC)
        'oxide_density_kg_m3': 3950,
        # Molar mass of Al₂O₃: 2×26.982 + 3×15.999 = 101.961 g/mol
        'oxide_molar_mass_g': 101.96,
        'metal_molar_mass_g': 26.982,
        'n_oxide_metal_atoms': 2,
        'oxide_name': 'Al2O3',
        'k_parabolic_m2_s': 1.0e-26,
        'Q_oxidation_eV': 1.5,
    },
    'gold': {
        # Au³⁺ + 3e⁻ → Au (MEASURED, IUPAC)
        'E_standard_V': +1.50,
        # Au₂O₃, thermally unstable above ~160°C; extremely slow at 300 K
        # Density of Au₂O₃ (MEASURED, CRC): ~11340 kg/m³
        'oxide_density_kg_m3': 11340,
        # Molar mass of Au₂O₃: 2×196.967 + 3×15.999 = 441.931 g/mol
        'oxide_molar_mass_g': 441.93,
        'metal_molar_mass_g': 196.967,
        'n_oxide_metal_atoms': 2,
        'oxide_name': 'Au2O3',
        'k_parabolic_m2_s': 1.0e-30,
        'Q_oxidation_eV': 2.0,
    },
    'silicon': {
        # Si → SiO₂, not a metal in the classical sense but included as a
        # semiconductor corrosion target. E° vs SHE for Si/SiO₂ in water.
        'E_standard_V': -0.86,
        # SiO₂ (amorphous/fused quartz), density (MEASURED, CRC)
        'oxide_density_kg_m3': 2200,
        # Molar mass of SiO₂: 28.085 + 2×15.999 = 60.083 g/mol
        'oxide_molar_mass_g': 60.08,
        'metal_molar_mass_g': 28.085,
        'n_oxide_metal_atoms': 1,
        'oxide_name': 'SiO2',
        'k_parabolic_m2_s': 1.0e-23,
        'Q_oxidation_eV': 1.2,
    },
    'tungsten': {
        # W → WO₃ + 6e⁻ path; E_standard (MEASURED, Bard et al.)
        'E_standard_V': -0.09,
        # WO₃ (tungsten trioxide), density (MEASURED, CRC)
        'oxide_density_kg_m3': 7160,
        # Molar mass of WO₃: 183.84 + 3×15.999 = 231.837 g/mol
        'oxide_molar_mass_g': 231.84,
        'metal_molar_mass_g': 183.84,
        'n_oxide_metal_atoms': 1,
        'oxide_name': 'WO3',
        'k_parabolic_m2_s': 1.0e-22,
        'Q_oxidation_eV': 1.1,
    },
    'nickel': {
        # Ni²⁺ + 2e⁻ → Ni, E° (MEASURED, IUPAC)
        'E_standard_V': -0.26,
        # NiO (bunsenite), density (MEASURED, CRC)
        'oxide_density_kg_m3': 6670,
        # Molar mass of NiO: 58.693 + 15.999 = 74.692 g/mol
        'oxide_molar_mass_g': 74.69,
        'metal_molar_mass_g': 58.693,
        'n_oxide_metal_atoms': 1,
        'oxide_name': 'NiO',
        'k_parabolic_m2_s': 1.0e-22,
        'Q_oxidation_eV': 1.0,
    },
    'titanium': {
        # Ti²⁺ + 2e⁻ → Ti, E° (MEASURED, IUPAC)
        'E_standard_V': -1.63,
        # TiO₂ (rutile), density (MEASURED, CRC)
        'oxide_density_kg_m3': 4230,
        # Molar mass of TiO₂: 47.867 + 2×15.999 = 79.865 g/mol
        'oxide_molar_mass_g': 79.87,
        'metal_molar_mass_g': 47.867,
        'n_oxide_metal_atoms': 1,
        'oxide_name': 'TiO2',
        'k_parabolic_m2_s': 1.0e-25,
        'Q_oxidation_eV': 1.4,
    },
}

# Reference temperature for k_parabolic (MEASURED at this T)
_T_REF = 300.0  # K


# ── Pilling-Bedworth Ratio ──────────────────────────────────────────────────

def pilling_bedworth_ratio(material_key):
    """Pilling-Bedworth ratio for a metal/oxide pair.

    PBR = (M_oxide × ρ_metal) / (n × M_metal × ρ_oxide)

    Where:
      M_oxide  = molar mass of oxide (g/mol)
      ρ_metal  = density of metal (kg/m³)
      n        = number of metal atoms per oxide formula unit
      M_metal  = molar mass of metal (g/mol)
      ρ_oxide  = density of oxide (kg/m³)

    The ratio compares the molar volume of oxide produced to the molar
    volume of metal consumed. Protective oxides have 1 < PBR < 2.

    FIRST_PRINCIPLES: geometry of molar volume ratio (Pilling & Bedworth 1923).
    No empirical fitting — pure molar volume accounting.

    Derivation chain:
      Volume of oxide per mol metal = M_oxide / (n × ρ_oxide)   [m³/mol metal]
      Volume of metal consumed      = M_metal / ρ_metal          [m³/mol metal]
      PBR = oxide volume / metal volume
          = (M_oxide / (n × ρ_oxide)) / (M_metal / ρ_metal)
          = (M_oxide × ρ_metal) / (n × M_metal × ρ_oxide)

    Args:
        material_key: key in CORROSION_DATA (and MATERIALS)

    Returns:
        PBR (dimensionless, float > 0)

    Raises:
        KeyError: if material_key is not in CORROSION_DATA
    """
    if material_key not in CORROSION_DATA:
        raise KeyError(f"Unknown material: {material_key!r}")

    cd = CORROSION_DATA[material_key]
    mat = MATERIALS[material_key]

    M_ox = cd['oxide_molar_mass_g']       # g/mol (ratio cancels g/mol units)
    rho_m = mat['density_kg_m3']          # kg/m³
    n = cd['n_oxide_metal_atoms']
    M_m = cd['metal_molar_mass_g']        # g/mol
    rho_ox = cd['oxide_density_kg_m3']    # kg/m³

    return (M_ox * rho_m) / (n * M_m * rho_ox)


# ── Oxide Classification ────────────────────────────────────────────────────

def oxide_classification(material_key):
    """Classify oxide film as 'protective', 'porous', or 'spalling'.

    Classification by Pilling-Bedworth ratio:
      PBR < 1.0       → 'porous'     (oxide too small, cracks open to metal)
      1.0 ≤ PBR ≤ 2.0 → 'protective' (compressive stress seals film)
      PBR > 2.0       → 'spalling'   (excessive compressive → film fractures)

    FIRST_PRINCIPLES: mechanical stress in the oxide film.
    Pilling & Bedworth (1923); Evans (1945) for spalling threshold.

    Args:
        material_key: key in CORROSION_DATA

    Returns:
        str: 'porous', 'protective', or 'spalling'
    """
    pbr = pilling_bedworth_ratio(material_key)
    if pbr < 1.0:
        return 'porous'
    elif pbr <= 2.0:
        return 'protective'
    else:
        return 'spalling'


# ── Parabolic Oxide Thickness ───────────────────────────────────────────────

def parabolic_oxide_thickness(material_key, time_s, temperature=300.0):
    """Oxide layer thickness from Wagner parabolic oxidation law (m).

    x(t, T) = sqrt(k_eff(T) × t)

    where the temperature-dependent rate constant is:

      k_eff(T) = k_ref × exp(Q × e / (k_B × T_ref) − Q × e / (k_B × T))
               = k_ref × exp((Q × e / k_B) × (1/T_ref − 1/T))

    k_ref is the MEASURED value at T_ref = 300 K.

    FIRST_PRINCIPLES: Wagner (1933) high-temperature oxidation theory.
    Rate-limiting step is ionic diffusion (O²⁻ inward or M^n+ outward)
    through the growing oxide of thickness x. Fick's first law across
    the oxide gives J ∝ 1/x, which integrates to x² = k × t.
    The Arrhenius temperature dependence is exact for activated diffusion.

    Derivation chain:
      Fick's 1st law: J = D × ΔC / x  (flux through oxide)
      Growth rate:    dx/dt = J × V_m / e  (molar volume of oxide consumed)
      Integrating:    x² = 2 × D × ΔC × V_m × t / e = k_parabolic × t
      T-dependence:   D ∝ exp(−Q/kT)  →  k ∝ exp(−Q/kT) (factor of 2 absorbed)

    Args:
        material_key: key in CORROSION_DATA
        time_s: elapsed time (s), must be > 0
        temperature: temperature in Kelvin (default 300 K)

    Returns:
        Oxide layer thickness in metres (float ≥ 0)

    Raises:
        KeyError: unknown material
        ValueError: time_s ≤ 0 or temperature ≤ 0
    """
    if material_key not in CORROSION_DATA:
        raise KeyError(f"Unknown material: {material_key!r}")
    if time_s <= 0:
        raise ValueError(f"time_s must be positive, got {time_s}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive (K), got {temperature}")

    cd = CORROSION_DATA[material_key]
    k_ref = cd['k_parabolic_m2_s']
    Q_eV = cd['Q_oxidation_eV']

    # Arrhenius scaling from T_ref to T
    Q_J = Q_eV * _EV_TO_JOULE
    exponent = (Q_J / K_B) * (1.0 / _T_REF - 1.0 / temperature)
    k_eff = k_ref * math.exp(exponent)

    return math.sqrt(k_eff * time_s)


# ── Galvanic Potential ──────────────────────────────────────────────────────

def galvanic_potential(material_a, material_b):
    """Galvanic potential difference between two metals (V).

    ΔE = E_standard(a) − E_standard(b)

    Convention: positive ΔE means metal_a is cathodic (protected),
    metal_b is anodic (corrodes). When ΔE > 0, metal_b is the anode.

    FIRST_PRINCIPLES: from ΔG = −nFΔE (electrochemistry).
    Spontaneous dissolution of the more active metal (more negative E°)
    is driven by the thermodynamic potential difference.

    Derivation chain:
      Oxidation half-reaction at anode:   M_b → M_b^n+ + ne⁻   (−E_b)
      Reduction half-reaction at cathode: M_a^n+ + ne⁻ → M_a   (+E_a)
      Net cell potential: ΔE = E_a − E_b
      Spontaneous if ΔE > 0  (ΔG = −nFΔE < 0)

    Args:
        material_a: key in CORROSION_DATA (potential cathode)
        material_b: key in CORROSION_DATA (potential anode)

    Returns:
        ΔE in Volts. Positive means a is cathodic, b corrodes.

    Raises:
        KeyError: if either material is not in CORROSION_DATA
    """
    for k in (material_a, material_b):
        if k not in CORROSION_DATA:
            raise KeyError(f"Unknown material: {k!r}")

    E_a = CORROSION_DATA[material_a]['E_standard_V']
    E_b = CORROSION_DATA[material_b]['E_standard_V']
    return E_a - E_b


# ── Galvanic Series Rank ────────────────────────────────────────────────────

def galvanic_series_rank():
    """Return materials sorted by standard electrode potential (most anodic first).

    Most anodic (most negative E°) = most susceptible to corrosion.
    Most noble (most positive E°) = most corrosion-resistant.

    MEASURED: standard electrode potentials (IUPAC, Bard et al. 1985).

    Returns:
        List of (material_key, E_standard_V) tuples, sorted ascending E°
        (most anodic = most easily oxidized first).
    """
    return sorted(
        [(k, v['E_standard_V']) for k, v in CORROSION_DATA.items()],
        key=lambda x: x[1]
    )


# ── Corrosion Rate Estimate ─────────────────────────────────────────────────

def corrosion_rate_estimate(material_key, temperature=300.0):
    """Estimated mass loss rate from parabolic oxidation law (kg/m²/s).

    At the initial stage (t → 0⁺), the instantaneous parabolic rate is:

      dx/dt = k_eff(T) / (2 × x)

    As x → 0, the rate diverges; we evaluate at t = 1 s (x = √k_eff) to
    get a characteristic initial rate:

      (dx/dt)|_{t=1s} = √k_eff / 2

    Mass loss rate per unit surface area:
      dm/dt = ρ_oxide × dx/dt = ρ_oxide × √k_eff / 2   (kg/m²/s)

    APPROXIMATION: initial parabolic rate at t = 1 s. Represents worst-case
    (freshly exposed surface). Long-term rate falls as 1/√t.

    Derivation chain:
      x² = k_eff × t  →  x = √(k_eff × t)
      dx/dt = √(k_eff / (4t)) = √k_eff / (2√t)
      At t = 1 s:  dx/dt = √k_eff / 2
      dm/dt = ρ_oxide × dx/dt

    Args:
        material_key: key in CORROSION_DATA
        temperature: temperature in Kelvin (default 300 K)

    Returns:
        Mass loss rate in kg/(m²·s)

    Raises:
        KeyError: unknown material
        ValueError: temperature ≤ 0
    """
    if material_key not in CORROSION_DATA:
        raise KeyError(f"Unknown material: {material_key!r}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive (K), got {temperature}")

    cd = CORROSION_DATA[material_key]
    k_ref = cd['k_parabolic_m2_s']
    Q_eV = cd['Q_oxidation_eV']
    rho_ox = cd['oxide_density_kg_m3']

    Q_J = Q_eV * _EV_TO_JOULE
    exponent = (Q_J / K_B) * (1.0 / _T_REF - 1.0 / temperature)
    k_eff = k_ref * math.exp(exponent)

    # dx/dt at t = 1 s
    dxdt = math.sqrt(k_eff) / 2.0
    return rho_ox * dxdt


# ── σ-field Corrosion Shift ─────────────────────────────────────────────────

def sigma_corrosion_shift(material_key, sigma):
    """Corrosion rate under σ-field (kg/m²/s at T = 300 K).

    The σ-field shifts the activation energy Q through its effect on
    cohesive energy E_coh.  Stiffer lattices (higher E_coh) suppress
    ion diffusion through the oxide → lower oxidation rate.

    Mechanism (CORE):
      Q(σ) = Q_ref × (E_coh(σ) / E_coh_ref)

      E_coh(σ) = E_coh_ref × [1 + f_QCD × (scale_ratio(σ) − 1)]

    where f_QCD = PROTON_QCD_FRACTION ≈ 0.08 is the QCD mass fraction
    of the proton.  The dominant EM contribution to cohesive energy is
    σ-invariant; only the lattice stiffness (phonon) piece shifts.

    The modified activation energy feeds directly into the Arrhenius
    factor of the parabolic rate constant:

      k_eff(σ, T) = k_ref × exp((Q(σ)/kT_ref − Q(σ)/kT))

    At σ = 0 this reduces exactly to corrosion_rate_estimate().

    Derivation chain (σ → E_coh → Q → k_eff → dm/dt):
      σ   →  scale_ratio(σ)  [from ..scale]
          →  mass_ratio(σ)   [nuclear mass shift]
          →  E_coh(σ)        [lattice stiffness shift]
          →  Q(σ)            [activation energy for diffusion]
          →  k_eff(σ, T)     [parabolic rate constant]
          →  dm/dt(σ)        [corrosion mass loss rate]

    CORE: σ-dependence from □σ = −ξR through nuclear mass to cohesive energy.

    Args:
        material_key: key in CORROSION_DATA (and MATERIALS)
        sigma: σ-field value (dimensionless; 0 = Earth-like)

    Returns:
        Mass loss rate in kg/(m²·s) at T = 300 K under σ-field

    Raises:
        KeyError: unknown material
    """
    if material_key not in CORROSION_DATA:
        raise KeyError(f"Unknown material: {material_key!r}")

    cd = CORROSION_DATA[material_key]
    mat = MATERIALS[material_key]

    Q_ref_eV = cd['Q_oxidation_eV']
    E_coh_ref = mat['cohesive_energy_ev']
    rho_ox = cd['oxide_density_kg_m3']
    k_ref = cd['k_parabolic_m2_s']

    if sigma == 0.0:
        return corrosion_rate_estimate(material_key, temperature=_T_REF)

    from ..scale import scale_ratio
    f_qcd = PROTON_QCD_FRACTION

    # Nuclear mass ratio under σ
    sr = scale_ratio(sigma)
    # Cohesive energy shift (only QCD mass fraction of bond energy shifts)
    E_coh_sigma = E_coh_ref * (1.0 + f_qcd * (sr - 1.0))

    # Q scales proportionally with cohesive energy
    Q_sigma_eV = Q_ref_eV * (E_coh_sigma / E_coh_ref)
    Q_sigma_J = Q_sigma_eV * _EV_TO_JOULE

    # k_eff at T_ref under modified Q
    # k_eff(T) = k_ref × exp((Q/kT_ref) − (Q/kT))
    # At T = T_ref: exponent = 0 → k_eff = k_ref (correct reference)
    # We are evaluating at T = T_ref with new Q:
    # Arrhenius from T_ref to T_ref gives 0 exponent, but the k_ref value
    # itself was measured with Q_ref at T_ref, so we scale k_ref by the
    # ratio of the two Boltzmann factors at T_ref:
    #   k_sigma_ref = k_ref × exp(−Q_sigma/kT_ref + Q_ref/kT_ref)
    #                = k_ref × exp((Q_ref − Q_sigma)/(k_B × T_ref))
    delta_Q_J = (Q_ref_eV - Q_sigma_eV) * _EV_TO_JOULE
    k_sigma_ref = k_ref * math.exp(delta_Q_J / (K_B * _T_REF))

    dxdt = math.sqrt(k_sigma_ref) / 2.0
    return rho_ox * dxdt


# ── Nagatha Export ──────────────────────────────────────────────────────────

def corrosion_properties(material_key, time_s=3.15e7, T=300.0, sigma=0.0):
    """Corrosion properties in Nagatha-compatible export format.

    Collects all corrosion observables for a material into a single dict
    for use by the Nagatha rendering pipeline (matter-shaper).

    Default time_s = 3.15e7 s ≈ 1 year.

    Args:
        material_key: key in CORROSION_DATA
        time_s: exposure time in seconds (default 1 year = 3.15×10⁷ s)
        T: temperature in Kelvin (default 300 K)
        sigma: σ-field value (default 0)

    Returns:
        Dict with keys:
          material           — material key
          oxide_name         — primary oxide formula string
          E_standard_V       — standard electrode potential vs SHE (V)
          pilling_bedworth_ratio — PBR (dimensionless)
          oxide_classification   — 'protective', 'porous', or 'spalling'
          oxide_thickness_m  — parabolic thickness at time_s and T (m)
          corrosion_rate_kg_m2_s — initial mass loss rate at T (kg/m²/s)
          sigma_corrosion_rate_kg_m2_s — rate under sigma (kg/m²/s)
          galvanic_rank      — position in galvanic series (0 = most anodic)
          time_s             — input time (s)
          temperature_K      — input temperature (K)
          sigma              — input σ-field
          origin_tag         — derivation provenance string

    Raises:
        KeyError: unknown material
    """
    if material_key not in CORROSION_DATA:
        raise KeyError(f"Unknown material: {material_key!r}")

    cd = CORROSION_DATA[material_key]
    pbr = pilling_bedworth_ratio(material_key)
    classification = oxide_classification(material_key)
    thickness = parabolic_oxide_thickness(material_key, time_s, T)
    rate = corrosion_rate_estimate(material_key, T)
    rate_sigma = sigma_corrosion_shift(material_key, sigma)

    series = galvanic_series_rank()
    rank = next(i for i, (k, _) in enumerate(series) if k == material_key)

    return {
        'material': material_key,
        'oxide_name': cd['oxide_name'],
        'E_standard_V': cd['E_standard_V'],
        'pilling_bedworth_ratio': pbr,
        'oxide_classification': classification,
        'oxide_thickness_m': thickness,
        'corrosion_rate_kg_m2_s': rate,
        'sigma_corrosion_rate_kg_m2_s': rate_sigma,
        'galvanic_rank': rank,
        'time_s': time_s,
        'temperature_K': T,
        'sigma': sigma,
        'origin_tag': (
            "MEASURED: k_parabolic, Q_oxidation (Birks, Meier & Pettit 2006; "
            "Kofstad 1988); E_standard (IUPAC/Bard 1985); oxide properties "
            "(CRC Handbook). "
            "FIRST_PRINCIPLES: Pilling-Bedworth (1923) geometry; "
            "Wagner (1933) parabolic law. "
            "CORE: sigma-field shift through cohesive energy / nuclear mass "
            "(from □σ = −ξR)."
        ),
    }
