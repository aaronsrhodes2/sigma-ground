"""
Composite materials — effective properties of multi-phase systems.

Derivation chain:
  mechanical.py (elastic moduli of phases)
  + thermal.py (thermal conductivity, expansion of phases)
  → composites.py (effective composite properties)

This generalizes the Voigt-Reuss bounds used in organic_materials.py
(wood, bone) into a complete composite mechanics framework applicable
to ANY multi-phase material: fiber composites, particulate composites,
laminates, foams, and concrete.

Derivation chains:

  1. Voigt Bound (Upper Bound, FIRST_PRINCIPLES)
     E_voigt = Σ fᵢ × Eᵢ   (iso-strain assumption)

     All phases strain equally (parallel loading). This gives the
     MAXIMUM possible modulus. Exact for continuous fibers loaded
     along their axis.

  2. Reuss Bound (Lower Bound, FIRST_PRINCIPLES)
     1/E_reuss = Σ fᵢ/Eᵢ   (iso-stress assumption)

     All phases carry equal stress (series loading). This gives the
     MINIMUM possible modulus. Exact for laminae loaded transversely.

  3. Hashin-Shtrikman Bounds (FIRST_PRINCIPLES: variational)
     Tighter bounds than Voigt-Reuss for isotropic composites.

     HS upper: K_U = K₂ + f₁/[1/(K₁−K₂) + 3f₂/(3K₂+4G₂)]
     HS lower: K_L = K₁ + f₂/[1/(K₂−K₁) + 3f₁/(3K₁+4G₁)]

     Where phase 1 has the lower modulus (K₁ < K₂).
     Hashin & Shtrikman (1963), J. Mech. Phys. Solids 11, 127.

  4. Halpin-Tsai (FIRST_PRINCIPLES form + MEASURED shape parameter)
     E_c = E_m × (1 + ξηf) / (1 − ηf)
     η = (E_f/E_m − 1) / (E_f/E_m + ξ)

     Where:
       E_f = fiber/filler modulus
       E_m = matrix modulus
       f = filler volume fraction
       ξ = shape parameter (MEASURED):
           ξ = 2 for spheres
           ξ = 2L/d for aligned fibers (aspect ratio)

     Halpin & Tsai (1969), Primer on Composite Materials.

  5. Gibson-Ashby Foam Model (FIRST_PRINCIPLES)
     E_foam = E_solid × C × (ρ_foam/ρ_solid)ⁿ

     Where:
       C ≈ 1.0 (open-cell) or ≈ 0.3 (closed-cell)
       n = 2 (bending-dominated, open-cell)
       n = 3 (stretching-dominated, closed-cell)

     Gibson & Ashby "Cellular Solids" (1997), Cambridge.

  6. Rule of Mixtures for Transport Properties (FIRST_PRINCIPLES)
     Thermal conductivity: same Voigt/Reuss structure
     Thermal expansion: α_c = Σ fᵢαᵢ (Voigt) or Turner model
     Density: ρ_c = Σ fᵢρᵢ (exact)

σ-dependence:
  Composite properties inherit σ-dependence from their phase properties.
  Each phase's modulus, conductivity, and density shift with σ through
  the standard cohesive energy → modulus chain. The composite bounds
  are purely geometric — the σ-field enters only through the phases.

  For organic composites (wood, bone, polymers): σ-INVARIANT
  (electromagnetic bonding only, no QCD contribution).

Origin tags:
  - Voigt/Reuss bounds: FIRST_PRINCIPLES (exact variational bounds)
  - Hashin-Shtrikman: FIRST_PRINCIPLES (optimal variational bounds)
  - Halpin-Tsai: FIRST_PRINCIPLES form + MEASURED ξ
  - Gibson-Ashby: FIRST_PRINCIPLES (beam mechanics) + MEASURED C, n
  - Density rule of mixtures: FIRST_PRINCIPLES (exact)
"""

import math


# ═══════════════════════════════════════════════════════════════════
# ELASTIC MODULUS BOUNDS
# ═══════════════════════════════════════════════════════════════════

def voigt_bound(moduli, fractions):
    """Voigt (upper) bound for effective modulus.

    E_voigt = Σ fᵢ × Eᵢ

    FIRST_PRINCIPLES: iso-strain assumption (parallel rule of mixtures).
    Exact for aligned continuous fibers loaded longitudinally.

    Args:
        moduli: list of phase moduli (any consistent units)
        fractions: list of volume fractions (must sum to 1.0)

    Returns:
        Voigt upper bound in same units as moduli.
    """
    return sum(f * E for f, E in zip(fractions, moduli))


def reuss_bound(moduli, fractions):
    """Reuss (lower) bound for effective modulus.

    1/E_reuss = Σ fᵢ/Eᵢ

    FIRST_PRINCIPLES: iso-stress assumption (series rule of mixtures).
    Exact for laminae loaded transversely.

    Args:
        moduli: list of phase moduli (any consistent units)
        fractions: list of volume fractions (must sum to 1.0)

    Returns:
        Reuss lower bound in same units as moduli.
    """
    inv = sum(f / E for f, E in zip(fractions, moduli) if E > 0)
    if inv <= 0:
        return 0.0
    return 1.0 / inv


def voigt_reuss_hill(moduli, fractions):
    """Voigt-Reuss-Hill average (arithmetic mean of bounds).

    E_VRH = ½(E_voigt + E_reuss)

    APPROXIMATION: Hill (1952) showed this average is a reasonable
    estimate for random polycrystals. Often within ±10% of experiment.

    Args:
        moduli: list of phase moduli
        fractions: list of volume fractions

    Returns:
        VRH average modulus.
    """
    return 0.5 * (voigt_bound(moduli, fractions) +
                   reuss_bound(moduli, fractions))


def hashin_shtrikman_bounds(K1, G1, f1, K2, G2):
    """Hashin-Shtrikman bounds for bulk modulus of two-phase composite.

    FIRST_PRINCIPLES: variational bounds (Hashin & Shtrikman 1963).

    Tighter than Voigt-Reuss for isotropic composites. Exact for
    the "composite spheres" microstructure (concentric shells).

    Phase 1 is the SOFTER phase (K1 < K2). If the input is reversed,
    we swap internally.

    Args:
        K1: bulk modulus of phase 1 (Pa)
        G1: shear modulus of phase 1 (Pa)
        f1: volume fraction of phase 1
        K2: bulk modulus of phase 2 (Pa)
        G2: shear modulus of phase 2 (Pa)

    Returns:
        (K_lower, K_upper) — HS bounds on effective bulk modulus (Pa).
    """
    f2 = 1.0 - f1

    # Ensure phase 1 is softer
    if K1 > K2:
        K1, G1, f1, K2, G2, f2 = K2, G2, f2, K1, G1, f1

    # Lower bound (soft matrix)
    if abs(K2 - K1) > 0 and (3 * K1 + 4 * G1) > 0:
        K_lower = K1 + f2 / (1.0 / (K2 - K1) + 3 * f1 / (3 * K1 + 4 * G1))
    else:
        K_lower = K1

    # Upper bound (stiff matrix)
    if abs(K1 - K2) > 0 and (3 * K2 + 4 * G2) > 0:
        K_upper = K2 + f1 / (1.0 / (K1 - K2) + 3 * f2 / (3 * K2 + 4 * G2))
    else:
        K_upper = K2

    return (K_lower, K_upper)


# ═══════════════════════════════════════════════════════════════════
# HALPIN-TSAI MODEL
# ═══════════════════════════════════════════════════════════════════

def halpin_tsai(E_fiber, E_matrix, f_fiber, xi=2.0):
    """Halpin-Tsai effective modulus for fiber/particle composites.

    E_c = E_m × (1 + ξηf) / (1 − ηf)
    η = (E_f/E_m − 1) / (E_f/E_m + ξ)

    FIRST_PRINCIPLES form + MEASURED shape parameter ξ.

    Shape parameter ξ:
      ξ = 2         → spherical particles
      ξ = 2L/d      → aligned short fibers (L/d = aspect ratio)
      ξ → ∞         → continuous fibers (recovers Voigt bound)
      ξ → 0         → thin discs (recovers Reuss bound)

    Args:
        E_fiber: fiber/filler modulus (Pa)
        E_matrix: matrix modulus (Pa)
        f_fiber: fiber volume fraction (0-1)
        xi: shape parameter (default 2.0 for spheres)

    Returns:
        Effective composite modulus (Pa).
    """
    if E_matrix <= 0:
        return 0.0
    if f_fiber <= 0:
        return E_matrix
    if f_fiber >= 1.0:
        return E_fiber

    ratio = E_fiber / E_matrix
    eta = (ratio - 1.0) / (ratio + xi)

    denom = 1.0 - eta * f_fiber
    if denom <= 0:
        # Approaching percolation — use Voigt as fallback
        return voigt_bound([E_fiber, E_matrix], [f_fiber, 1.0 - f_fiber])

    return E_matrix * (1.0 + xi * eta * f_fiber) / denom


# ═══════════════════════════════════════════════════════════════════
# GIBSON-ASHBY FOAM MODEL
# ═══════════════════════════════════════════════════════════════════

def gibson_ashby_modulus(E_solid, relative_density, cell_type='open'):
    """Gibson-Ashby modulus for cellular/foam materials.

    E_foam = E_solid × C × (ρ_rel)ⁿ

    FIRST_PRINCIPLES: beam bending mechanics for open-cell foams,
    plate stretching for closed-cell foams.

    Args:
        E_solid: modulus of the solid phase (Pa)
        relative_density: ρ_foam/ρ_solid (0-1)
        cell_type: 'open' (bending) or 'closed' (stretching)

    Returns:
        Effective foam modulus (Pa).
    """
    rho_rel = max(0.0, min(1.0, relative_density))

    if rho_rel <= 0:
        return 0.0
    if rho_rel >= 1.0:
        return E_solid

    if cell_type == 'closed':
        C, n = 0.3, 3
    else:
        C, n = 1.0, 2

    return E_solid * C * rho_rel ** n


def gibson_ashby_strength(sigma_solid, relative_density, cell_type='open'):
    """Gibson-Ashby yield/crush strength for foam materials.

    σ_foam = σ_solid × C × (ρ_rel)^n

    For open-cell foams: n = 3/2 (plastic hinge formation)
    For closed-cell foams: n = 2 (membrane stretching)

    Args:
        sigma_solid: yield strength of solid phase (Pa)
        relative_density: ρ_foam/ρ_solid (0-1)
        cell_type: 'open' or 'closed'

    Returns:
        Effective foam strength (Pa).
    """
    rho_rel = max(0.0, min(1.0, relative_density))

    if rho_rel <= 0:
        return 0.0
    if rho_rel >= 1.0:
        return sigma_solid

    if cell_type == 'closed':
        C, n = 0.3, 2.0
    else:
        C, n = 0.3, 1.5

    return sigma_solid * C * rho_rel ** n


# ═══════════════════════════════════════════════════════════════════
# TRANSPORT PROPERTIES
# ═══════════════════════════════════════════════════════════════════

def density_rule_of_mixtures(densities, fractions):
    """Composite density from rule of mixtures (exact).

    ρ_c = Σ fᵢ × ρᵢ

    FIRST_PRINCIPLES: mass conservation. This is EXACT for any
    composite (no approximation involved).

    Args:
        densities: list of phase densities (kg/m³)
        fractions: list of volume fractions

    Returns:
        Composite density in kg/m³.
    """
    return sum(f * rho for f, rho in zip(fractions, densities))


def thermal_expansion_voigt(alphas, moduli, fractions):
    """Effective thermal expansion coefficient (Turner/Schapery model).

    α_c = Σ fᵢ × αᵢ × Eᵢ / Σ fᵢ × Eᵢ

    FIRST_PRINCIPLES: Turner (1946) — thermo-elastic weighted average.
    The stiffer phase dominates the expansion because it constrains
    the softer phase.

    For a simple rule of mixtures (no elastic weighting):
    use α_c = Σ fᵢ × αᵢ

    Args:
        alphas: list of thermal expansion coefficients (1/K)
        moduli: list of phase moduli (Pa)
        fractions: list of volume fractions

    Returns:
        Effective expansion coefficient (1/K).
    """
    numerator = sum(f * a * E for f, a, E in zip(fractions, alphas, moduli))
    denominator = sum(f * E for f, E in zip(fractions, moduli))

    if denominator <= 0:
        return sum(f * a for f, a in zip(fractions, alphas))

    return numerator / denominator


def thermal_conductivity_bounds(conductivities, fractions):
    """Voigt and Reuss bounds on thermal conductivity.

    Same mathematical structure as elastic modulus bounds.

    Args:
        conductivities: list of phase thermal conductivities (W/(m·K))
        fractions: list of volume fractions

    Returns:
        (k_lower, k_upper) tuple in W/(m·K).
    """
    k_upper = voigt_bound(conductivities, fractions)
    k_lower = reuss_bound(conductivities, fractions)
    return (k_lower, k_upper)


# ═══════════════════════════════════════════════════════════════════
# COMPOSITE DATABASE — Common Engineering Composites
# ═══════════════════════════════════════════════════════════════════
# Phase properties are MEASURED. Composite properties are DERIVED.
# Rule 9: every common composite type, all fields populated.

COMPOSITES = {
    'cfrp_unidirectional': {
        'name': 'Carbon Fiber / Epoxy (unidirectional)',
        'fiber': {'E_GPa': 230, 'rho_kg_m3': 1800, 'alpha_1_K': -0.4e-6},
        'matrix': {'E_GPa': 3.5, 'rho_kg_m3': 1200, 'alpha_1_K': 60e-6},
        'f_fiber': 0.60,
        'xi': 1e6,  # continuous fibers → xi → ∞
    },
    'gfrp_unidirectional': {
        'name': 'Glass Fiber / Epoxy (unidirectional)',
        'fiber': {'E_GPa': 73, 'rho_kg_m3': 2540, 'alpha_1_K': 5.0e-6},
        'matrix': {'E_GPa': 3.5, 'rho_kg_m3': 1200, 'alpha_1_K': 60e-6},
        'f_fiber': 0.55,
        'xi': 1e6,
    },
    'cfrp_chopped': {
        'name': 'Carbon Fiber / Epoxy (chopped, random)',
        'fiber': {'E_GPa': 230, 'rho_kg_m3': 1800, 'alpha_1_K': -0.4e-6},
        'matrix': {'E_GPa': 3.5, 'rho_kg_m3': 1200, 'alpha_1_K': 60e-6},
        'f_fiber': 0.30,
        'xi': 20,  # L/d ≈ 10, xi = 2L/d = 20
    },
    'concrete': {
        'name': 'Concrete (aggregate + cement paste)',
        'fiber': {'E_GPa': 50, 'rho_kg_m3': 2650, 'alpha_1_K': 10e-6},
        'matrix': {'E_GPa': 15, 'rho_kg_m3': 2000, 'alpha_1_K': 12e-6},
        'f_fiber': 0.70,  # aggregate fraction
        'xi': 2,  # roughly spherical aggregate
    },
    'alumina_zirconia': {
        'name': 'Alumina + Zirconia (ceramic composite)',
        'fiber': {'E_GPa': 210, 'rho_kg_m3': 6050, 'alpha_1_K': 10.3e-6},
        'matrix': {'E_GPa': 380, 'rho_kg_m3': 3980, 'alpha_1_K': 8.0e-6},
        'f_fiber': 0.20,  # zirconia particles in alumina matrix
        'xi': 2,
    },
    'wc_cobalt': {
        'name': 'WC-Co (cemented carbide / hardmetal)',
        'fiber': {'E_GPa': 700, 'rho_kg_m3': 15630, 'alpha_1_K': 5.2e-6},
        'matrix': {'E_GPa': 209, 'rho_kg_m3': 8900, 'alpha_1_K': 12.0e-6},
        'f_fiber': 0.88,
        'xi': 2,
    },
    'al_sic_particulate': {
        'name': 'Aluminum / SiC (metal matrix composite)',
        'fiber': {'E_GPa': 410, 'rho_kg_m3': 3210, 'alpha_1_K': 4.0e-6},
        'matrix': {'E_GPa': 70, 'rho_kg_m3': 2700, 'alpha_1_K': 23.0e-6},
        'f_fiber': 0.20,
        'xi': 2,
    },
    'polyurethane_foam': {
        'name': 'Polyurethane foam (open cell)',
        'fiber': {'E_GPa': 1.5, 'rho_kg_m3': 1200, 'alpha_1_K': 100e-6},
        'matrix': {'E_GPa': 1.5, 'rho_kg_m3': 1200, 'alpha_1_K': 100e-6},
        'f_fiber': 0.0,
        'xi': 2,
        'foam': True,
        'relative_density': 0.05,
        'cell_type': 'open',
    },
    'al_foam': {
        'name': 'Aluminum foam (closed cell, Alporas-type)',
        'fiber': {'E_GPa': 70, 'rho_kg_m3': 2700, 'alpha_1_K': 23e-6},
        'matrix': {'E_GPa': 70, 'rho_kg_m3': 2700, 'alpha_1_K': 23e-6},
        'f_fiber': 0.0,
        'xi': 2,
        'foam': True,
        'relative_density': 0.10,
        'cell_type': 'closed',
    },
}


# ═══════════════════════════════════════════════════════════════════
# COMPOSITE PROPERTY CALCULATOR
# ═══════════════════════════════════════════════════════════════════

def composite_modulus(composite_key):
    """Effective modulus for a named composite (Pa).

    Uses Halpin-Tsai for fiber/particle composites,
    Gibson-Ashby for foams.

    Args:
        composite_key: key into COMPOSITES dict

    Returns:
        Effective modulus in Pa.
    """
    comp = COMPOSITES[composite_key]

    if comp.get('foam'):
        E_solid = comp['fiber']['E_GPa'] * 1e9
        return gibson_ashby_modulus(
            E_solid, comp['relative_density'], comp.get('cell_type', 'open')
        )

    E_f = comp['fiber']['E_GPa'] * 1e9
    E_m = comp['matrix']['E_GPa'] * 1e9
    return halpin_tsai(E_f, E_m, comp['f_fiber'], comp.get('xi', 2))


def composite_density(composite_key):
    """Effective density for a named composite (kg/m³).

    EXACT: rule of mixtures.

    Args:
        composite_key: key into COMPOSITES dict

    Returns:
        Density in kg/m³.
    """
    comp = COMPOSITES[composite_key]

    if comp.get('foam'):
        rho_solid = comp['fiber']['rho_kg_m3']
        return rho_solid * comp['relative_density']

    rho_f = comp['fiber']['rho_kg_m3']
    rho_m = comp['matrix']['rho_kg_m3']
    f = comp['f_fiber']
    return density_rule_of_mixtures([rho_f, rho_m], [f, 1 - f])


def composite_expansion(composite_key):
    """Effective thermal expansion coefficient (1/K).

    Turner-weighted average.

    Args:
        composite_key: key into COMPOSITES dict

    Returns:
        CTE in 1/K.
    """
    comp = COMPOSITES[composite_key]

    if comp.get('foam'):
        return comp['fiber']['alpha_1_K']

    E_f = comp['fiber']['E_GPa'] * 1e9
    E_m = comp['matrix']['E_GPa'] * 1e9
    a_f = comp['fiber']['alpha_1_K']
    a_m = comp['matrix']['alpha_1_K']
    f = comp['f_fiber']

    return thermal_expansion_voigt(
        [a_f, a_m], [E_f, E_m], [f, 1 - f]
    )


def specific_stiffness(composite_key):
    """Specific modulus E/ρ (Pa·m³/kg = m²/s²).

    The figure of merit for lightweight structures.
    Higher specific stiffness = lighter for same stiffness.

    Args:
        composite_key: key into COMPOSITES dict

    Returns:
        Specific modulus in m²/s² (or equivalently Pa/(kg/m³)).
    """
    E = composite_modulus(composite_key)
    rho = composite_density(composite_key)

    if rho <= 0:
        return 0.0

    return E / rho


# ═══════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════

def composite_report(composite_key):
    """Complete property report for a named composite.

    Returns dict with all derived properties.
    """
    comp = COMPOSITES[composite_key]

    E_f = comp['fiber']['E_GPa'] * 1e9
    E_m = comp['matrix']['E_GPa'] * 1e9
    f = comp['f_fiber']

    report = {
        'name': comp['name'],
        'composite_key': composite_key,
        'f_fiber': f,
    }

    if comp.get('foam'):
        report['foam'] = True
        report['relative_density'] = comp['relative_density']
        report['cell_type'] = comp.get('cell_type', 'open')
    else:
        report['voigt_GPa'] = voigt_bound([E_f, E_m], [f, 1 - f]) / 1e9
        report['reuss_GPa'] = reuss_bound([E_f, E_m], [f, 1 - f]) / 1e9
        report['halpin_tsai_GPa'] = halpin_tsai(E_f, E_m, f, comp.get('xi', 2)) / 1e9

    report['effective_modulus_GPa'] = composite_modulus(composite_key) / 1e9
    report['density_kg_m3'] = composite_density(composite_key)
    report['CTE_1_K'] = composite_expansion(composite_key)
    report['specific_stiffness_MNm_kg'] = specific_stiffness(composite_key) / 1e6

    return report


def full_report():
    """Reports for ALL composites. Rule 9: if one, then all."""
    return {key: composite_report(key) for key in COMPOSITES}
