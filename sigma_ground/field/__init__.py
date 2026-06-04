"""
sigma_ground.field — physics field library (standard, observation-anchored).

Fast path: constants, the (inert) scale factor, nucleon masses, binding energies.

    from sigma_ground.field.constants import HBAR, C, E_CHARGE
    from sigma_ground.field.nucleon import neutron_mass_mev
"""

from .constants import (
    LAMBDA_QCD_MEV,
    HBAR, C, G, E_CHARGE, EPS_0, MU_0, ALPHA, K_B,
    M_ELECTRON_MEV, M_ELECTRON_KG,
    L_PLANCK, M_PLANCK_KG,
    PROTON_TOTAL_MEV, NEUTRON_TOTAL_MEV,
)
from .scale import scale_ratio
from .nucleon import proton_mass_mev, neutron_mass_mev, nucleon_decomposition
from .binding import binding_energy_mev, binding_decomposition

__version__ = "1.1.1"
__all__ = [
    # Physical constants
    'LAMBDA_QCD_MEV',
    'HBAR', 'C', 'G', 'E_CHARGE', 'EPS_0', 'MU_0', 'ALPHA', 'K_B',
    'M_ELECTRON_MEV', 'M_ELECTRON_KG',
    'L_PLANCK', 'M_PLANCK_KG',
    'PROTON_TOTAL_MEV', 'NEUTRON_TOTAL_MEV',
    # Scale factor (inert; 1.0 at our scale)
    'scale_ratio',
    # Nucleon
    'proton_mass_mev', 'neutron_mass_mev', 'nucleon_decomposition',
    # Binding
    'binding_energy_mev', 'binding_decomposition',
]
