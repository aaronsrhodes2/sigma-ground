"""
Adhesion — interface binding between two materials.

When surface A touches surface B, three energies matter:
  γ_A: surface energy of A (energy to create a free A surface)
  γ_B: surface energy of B (energy to create a free B surface)
  γ_AB: interface energy (energy cost of the A-B boundary)

Work of adhesion (Dupré, 1869):
  W_AB = γ_A + γ_B − γ_AB

  FIRST_PRINCIPLES: thermodynamic energy balance.
  You had two free surfaces (cost γ_A + γ_B to create).
  Now they're in contact, forming an interface (cost γ_AB).
  Energy released = what you had − what you're left with.

Interface energy (Berthelot combining rule):
  γ_AB = γ_A + γ_B − 2√(γ_A × γ_B)

  APPROXIMATION: assumes cross-material bond energy is the
  geometric mean of same-material bond energies. This works
  well for similar metals, poorly for metal-ceramic or
  metal-polymer interfaces. We mark this honestly.

  Why geometric mean? If atom A has bond strength ε_A and atom B
  has ε_B, the cross-bond is ε_AB ≈ √(ε_A × ε_B). This is the
  Berthelot hypothesis — it says mixing follows a geometric mean
  rather than arithmetic or harmonic. It's not derived from
  first principles, but it has the right limiting behavior:
    - When A = B: γ_AB = 0 (correct: no interface)
    - When A ≠ B: γ_AB > 0 (correct: mismatched bonds cost energy)
    - More dissimilar → higher γ_AB (correct)

Contact angle (Young, 1805):
  cos θ = (γ_SV − γ_SL) / γ_LV

  FIRST_PRINCIPLES: force balance at the triple line where solid,
  liquid, and vapor meet. Three surface tension vectors pull on the
  contact line; equilibrium gives Young's equation.

  Combined with Dupré: cos θ = W_SL / γ_LV − 1
  (Young-Dupré equation)

σ-dependence:
  Inherited from surface.py. Both γ_A(σ) and γ_B(σ) carry QCD
  mass corrections, so W_AB(σ) and γ_AB(σ) shift accordingly.
  At Earth: negligible. At neutron stars: measurable.

Origin tags:
  - Dupré equation: FIRST_PRINCIPLES (energy balance)
  - Berthelot rule: APPROXIMATION (geometric mean combining)
  - Young's equation: FIRST_PRINCIPLES (force balance)
  - σ-dependence: CORE (through surface.py)
"""

import math
from .surface import (
    surface_energy, surface_energy_at_sigma,
    surface_energy_decomposition, MATERIALS,
)
from ..constants import SIGMA_HERE


# ── Interface Energy ──────────────────────────────────────────────

def interface_energy(material_1, material_2, sigma=SIGMA_HERE):
    """Interface energy γ₁₂ (J/m²) using Berthelot combining rule.

    γ₁₂ = γ₁ + γ₂ − 2√(γ₁ × γ₂)

    APPROXIMATION: geometric mean of cross-bond energies.

    Args:
        material_1: key into MATERIALS dict
        material_2: key into MATERIALS dict
        sigma: σ-field value (default 0)

    Returns:
        γ₁₂ in J/m²
    """
    g1 = surface_energy_at_sigma(material_1, sigma)
    g2 = surface_energy_at_sigma(material_2, sigma)

    # Berthelot: γ₁₂ = (√γ₁ − √γ₂)²
    # Equivalent to γ₁ + γ₂ − 2√(γ₁γ₂), but this form
    # makes it obvious that γ₁₂ ≥ 0 always.
    gamma_12 = (math.sqrt(g1) - math.sqrt(g2)) ** 2

    return gamma_12


# ── Work of Adhesion ──────────────────────────────────────────────

def work_of_adhesion(material_1, material_2, sigma=SIGMA_HERE):
    """Work of adhesion W₁₂ (J/m²) from Dupré equation.

    W₁₂ = γ₁ + γ₂ − γ₁₂

    FIRST_PRINCIPLES: thermodynamic energy balance.

    With Berthelot interface energy:
      W₁₂ = 2√(γ₁ × γ₂)

    For self-adhesion (material_1 = material_2):
      W_AA = 2γ_A (work of cohesion)

    Args:
        material_1: key into MATERIALS dict
        material_2: key into MATERIALS dict
        sigma: σ-field value (default 0)

    Returns:
        W₁₂ in J/m²
    """
    g1 = surface_energy_at_sigma(material_1, sigma)
    g2 = surface_energy_at_sigma(material_2, sigma)

    # Dupré with Berthelot: W = γ₁ + γ₂ − (√γ₁ − √γ₂)²
    #                        = 2√(γ₁γ₂)
    return 2.0 * math.sqrt(g1 * g2)


def work_of_adhesion_at_sigma(material_1, material_2, sigma=SIGMA_HERE):
    """Work of adhesion at arbitrary σ. Explicit sigma signature."""
    return work_of_adhesion(material_1, material_2, sigma=sigma)


# ── Decomposition ────────────────────────────────────────────────

def adhesion_decomposition(material_1, material_2, sigma=SIGMA_HERE):
    """Decompose work of adhesion into EM and QCD components.

    The decomposition inherits from surface energy decomposition:
      W₁₂ = 2√(γ₁ × γ₂)
      γ_i = γ_i_EM + γ_i_QCD

    We compute W at σ and at σ=0, and decompose:
      W_EM = contribution from EM-invariant parts of both γ
      W_QCD = total − W_EM

    Returns dict with em_component_j_m2, qcd_component_j_m2, total_j_m2.
    """
    # Total at this sigma
    W_total = work_of_adhesion(material_1, material_2, sigma=sigma)

    # EM-only: use surface energies with only EM component
    d1 = surface_energy_decomposition(material_1, sigma)
    d2 = surface_energy_decomposition(material_2, sigma)

    g1_em = d1['em_component_j_m2']
    g2_em = d2['em_component_j_m2']

    # W_EM = 2√(γ₁_EM × γ₂_EM)
    W_em = 2.0 * math.sqrt(g1_em * g2_em)

    return {
        'em_component_j_m2': W_em,
        'qcd_component_j_m2': W_total - W_em,
        'total_j_m2': W_total,
        'sigma': sigma,
    }


# ── Contact Angle ────────────────────────────────────────────────

def contact_angle(solid_material, liquid_material, gamma_lv,
                  sigma=SIGMA_HERE):
    """Contact angle θ (degrees) from Young-Dupré equation.

    cos θ = W_SL / γ_LV − 1

    FIRST_PRINCIPLES: force balance at triple line.

    Args:
        solid_material: key into MATERIALS dict (the substrate)
        liquid_material: key into MATERIALS dict (the liquid)
        gamma_lv: liquid-vapor surface tension (J/m²) — MEASURED
        sigma: σ-field value

    Returns:
        θ in degrees (0 = complete wetting, 180 = complete non-wetting)
        Returns 0 if cos θ ≥ 1 (complete wetting/spreading)
        Returns 180 if cos θ ≤ −1 (complete non-wetting)
        Returns None if gamma_lv ≤ 0 (undefined)
    """
    if gamma_lv <= 0:
        return None

    W = work_of_adhesion(solid_material, liquid_material, sigma=sigma)
    cos_theta = W / gamma_lv - 1.0

    # Clamp to physical range
    if cos_theta >= 1.0:
        return 0    # complete wetting (spreading)
    elif cos_theta <= -1.0:
        return 180  # complete non-wetting (beading)
    else:
        return math.degrees(math.acos(cos_theta))


# ── Wetting: liquid-on-solid contact angle ────────────────────────
#
# The contact_angle() above derives W_SL from the broken-bond SOLIDS
# database via Berthelot (2√(γ₁γ₂)). That is correct for a metal melt on
# a metal (both high-energy, both adhere strongly) but it cannot describe
# a real wetting phase — water, mercury, a molten solder — because those
# liquids are not in the solids DB, and the Berthelot mean of two metals
# is so large that cos θ = W/γ_LV − 1 ≥ 1 → it reports complete wetting
# (θ = 0) for systems that actually bead up.
#
# Real wetting is governed by how the liquid's surface tension splits into
# a DISPERSIVE (van der Waals) part and a POLAR (H-bond / dipole) part, and
# how those couple to the solid's two parts. The standard textbook model is
# the geometric-mean combining rule (Fowkes 1964 → Owens-Wendt-Rabel-Kaelble
# 1969):
#
#   W_SL = 2√(γ_S^d · γ_L^d) + 2√(γ_S^p · γ_L^p)        (Owens-Wendt)
#   cos θ = W_SL / γ_LV − 1                               (Young-Dupré)
#
#   FIRST_PRINCIPLES: Young-Dupré is a force balance at the triple line.
#   APPROXIMATION: the geometric-mean combining rule for the cross term.
#       It assumes the unlike-pair interaction is √(like·like) PER COMPONENT.
#       Works well for van der Waals + polar organics on solids; it is the
#       workhorse of surface science. We mark it honestly.
#
# Liquid metals (mercury, molten Pb/solder, gallium) are a known exception:
# their huge surface tension is metallic, not polar/dipolar, so the polar
# cross term has no physical counterpart against a dielectric solid. For a
# `metallic` liquid on a non-metallic solid we keep ONLY the dispersive
# (Fowkes) term — this is exactly how Fowkes (1964) treated mercury, and it
# is what makes mercury bead on glass (θ ≈ 130-140°) instead of wetting it.
#
# All γ values below are MEASURED surface tensions / surface energies in SI
# (J/m² = N/m), split into dispersive (γ^d) and polar (γ^p) components from
# the standard surface-science literature (Fowkes; Owens & Wendt; van Oss;
# CRC Handbook). γ_LV = γ^d + γ^p is the total liquid-vapor surface tension.

# ── Wetting liquids ───────────────────────────────────────────────
# γ_LV = γ^d + γ^p. `metallic` flags a liquid metal (dispersive cross term
# only against a dielectric solid). Sources: Fowkes 1964; Owens-Wendt 1969;
# CRC Handbook of Chemistry & Physics.
WETTING_LIQUIDS = {
    'water': {
        'name': 'Water',
        'gamma_d': 0.0218, 'gamma_p': 0.0510, 'gamma_lv': 0.0728,
        'metallic': False,
    },
    'glycerol': {
        'name': 'Glycerol',
        'gamma_d': 0.0340, 'gamma_p': 0.0300, 'gamma_lv': 0.0640,
        'metallic': False,
    },
    'ethylene_glycol': {
        'name': 'Ethylene glycol',
        'gamma_d': 0.0290, 'gamma_p': 0.0190, 'gamma_lv': 0.0480,
        'metallic': False,
    },
    'ethanol': {
        'name': 'Ethanol',
        'gamma_d': 0.0185, 'gamma_p': 0.0036, 'gamma_lv': 0.0221,
        'metallic': False,
    },
    'diiodomethane': {  # classic purely-dispersive probe liquid
        'name': 'Diiodomethane',
        'gamma_d': 0.0508, 'gamma_p': 0.0000, 'gamma_lv': 0.0508,
        'metallic': False,
    },
    'mercury': {
        'name': 'Mercury',
        'gamma_d': 0.200, 'gamma_p': 0.285, 'gamma_lv': 0.485,
        'metallic': True,
    },
    'solder_lead': {  # molten Pb / Pb-Sn solder near its melting point
        'name': 'Molten lead/solder',
        'gamma_d': 0.130, 'gamma_p': 0.320, 'gamma_lv': 0.450,
        'metallic': True,
    },
    'gallium': {  # liquid gallium (just above 30 °C)
        'name': 'Liquid gallium',
        'gamma_d': 0.180, 'gamma_p': 0.520, 'gamma_lv': 0.700,
        'metallic': True,
    },
}

# ── Wetting solids ────────────────────────────────────────────────
# Surface-energy components (J/m²) of common substrates. `metallic` flags a
# clean metal (polar cross term retained even with a metallic liquid).
# Sources: Owens-Wendt 1969; van Oss "Interfacial Forces in Aqueous Media".
# Glass is the clean (high-energy) soda-lime surface that water sheets on.
WETTING_SOLIDS = {
    'ptfe': {  # Teflon — the canonical low-energy, near-purely-dispersive solid
        'name': 'PTFE (Teflon)',
        'gamma_d': 0.0185, 'gamma_p': 0.0005,
    },
    'paraffin': {
        'name': 'Paraffin wax',
        'gamma_d': 0.0250, 'gamma_p': 0.0000,
    },
    'polyethylene': {
        'name': 'Polyethylene',
        'gamma_d': 0.0330, 'gamma_p': 0.0010,
    },
    'pmma': {
        'name': 'PMMA (acrylic)',
        'gamma_d': 0.0290, 'gamma_p': 0.0110,
    },
    'glass': {  # clean soda-lime glass: high-energy, strongly polar
        'name': 'Soda-lime glass (clean)',
        'gamma_d': 0.0290, 'gamma_p': 0.0510,
    },
    'silicon': {  # native-oxide silicon wafer
        'name': 'Silicon (native oxide)',
        'gamma_d': 0.0270, 'gamma_p': 0.0150,
    },
    'steel': {  # oxidised steel
        'name': 'Steel (oxidised)',
        'gamma_d': 0.0290, 'gamma_p': 0.0110,
    },
    'gold': {  # typical lab gold (some adsorbed carbon); see notes on cleanliness
        'name': 'Gold',
        'gamma_d': 0.0300, 'gamma_p': 0.0200,
        'metallic': True,
    },
}


def work_of_solid_liquid_adhesion(solid_key, liquid_key):
    """Work of adhesion W_SL (J/m²) between a solid and a wetting liquid.

    Owens-Wendt geometric-mean combining rule:
        W_SL = 2√(γ_S^d · γ_L^d) + 2√(γ_S^p · γ_L^p)

    FIRST_PRINCIPLES: Dupré energy balance (W_SL = γ_S + γ_L − γ_SL).
    APPROXIMATION: geometric-mean cross term, per component (Fowkes/OWRK).

    Metallic-liquid exception: for a `metallic` liquid (mercury, molten
    solder, gallium) on a non-metallic solid, the polar/metallic cross term
    has no geometric-mean counterpart, so only the dispersive (Fowkes) term
    contributes. This is what makes mercury non-wetting on glass.

    Args:
        solid_key: key into WETTING_SOLIDS
        liquid_key: key into WETTING_LIQUIDS

    Returns:
        W_SL in J/m²
    """
    s = WETTING_SOLIDS[solid_key]
    liq = WETTING_LIQUIDS[liquid_key]

    dispersive = 2.0 * math.sqrt(s['gamma_d'] * liq['gamma_d'])

    # Metallic liquid on a dielectric solid: dispersive (Fowkes) term only.
    if liq.get('metallic') and not s.get('metallic'):
        return dispersive

    polar = 2.0 * math.sqrt(s['gamma_p'] * liq['gamma_p'])
    return dispersive + polar


def wetting_contact_angle(solid_key, liquid_key, gamma_lv=None):
    """Equilibrium wetting contact angle θ (degrees), Young-Dupré + Owens-Wendt.

        cos θ = W_SL / γ_LV − 1

    FIRST_PRINCIPLES: force balance at the solid-liquid-vapor triple line.

    Unlike contact_angle() (which draws both phases from the broken-bond
    SOLIDS DB and therefore over-predicts wetting for liquid phases), this
    uses the dispersive/polar surface-energy split, so it reproduces textbook
    values: water on clean glass ≈ 0°, water on PTFE ≈ 108°, mercury on glass
    ≈ 130-140°.

    Args:
        solid_key: key into WETTING_SOLIDS (the substrate)
        liquid_key: key into WETTING_LIQUIDS (the wetting phase)
        gamma_lv: optional override of the liquid-vapor surface tension (J/m²);
            defaults to the tabulated total γ_LV for the liquid

    Returns:
        θ in degrees (0 = complete wetting, 180 = complete non-wetting).
        Returns 0.0 if cos θ ≥ 1 (spreads), 180.0 if cos θ ≤ −1 (beads),
        None if γ_LV ≤ 0 (undefined).
    """
    liq = WETTING_LIQUIDS[liquid_key]
    g_lv = liq['gamma_lv'] if gamma_lv is None else gamma_lv
    if g_lv <= 0:
        return None

    W = work_of_solid_liquid_adhesion(solid_key, liquid_key)
    cos_theta = W / g_lv - 1.0

    if cos_theta >= 1.0:
        return 0.0      # complete wetting (spreading)
    elif cos_theta <= -1.0:
        return 180.0    # complete non-wetting (beading)
    else:
        return math.degrees(math.acos(cos_theta))


def spreading_coefficient(solid_key, liquid_key):
    """Spreading coefficient S (J/m²) of a liquid on a solid.

        S = W_SL − W_LL = W_SL − 2 γ_LV

    S ≥ 0 → the liquid spreads spontaneously (complete wetting, θ = 0).
    S < 0 → a finite contact angle forms (partial or non-wetting).

    FIRST_PRINCIPLES: energy released by spreading vs. cohesion of the liquid.
    """
    liq = WETTING_LIQUIDS[liquid_key]
    W = work_of_solid_liquid_adhesion(solid_key, liquid_key)
    return W - 2.0 * liq['gamma_lv']


# ── Nagatha Integration ──────────────────────────────────────────

def material_adhesion_properties(material_1, material_2, sigma=SIGMA_HERE):
    """Export adhesion properties in Nagatha-compatible format.

    Returns a dict that can be merged into Nagatha's color.json.
    """
    W = work_of_adhesion(material_1, material_2, sigma=sigma)
    gamma_12 = interface_energy(material_1, material_2, sigma=sigma)
    dec = adhesion_decomposition(material_1, material_2, sigma=sigma)

    return {
        'work_of_adhesion_j_m2': W,
        'interface_energy_j_m2': gamma_12,
        'material_1': material_1,
        'material_2': material_2,
        'sigma': sigma,
        'em_fraction': dec['em_component_j_m2'] / dec['total_j_m2']
                       if dec['total_j_m2'] > 0 else 0,
        'origin_tag': (
            "FIRST_PRINCIPLES: Dupré equation (energy balance). "
            "APPROXIMATION: Berthelot combining rule (geometric mean "
            "of cross-bond energies — works well for similar metals, "
            "underestimates for dissimilar pairs). "
            "CORE: σ-dependence through nuclear mass correction."
        ),
    }
