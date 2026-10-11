"""sigma_ground.field.dense: degenerate and nuclear matter (tier 1). Plan: misc/NUCLEAR_MATTER_PLAN.md."""
from .fermi_gas import (  # noqa: F401
    ELECTRON, MUON, NEUTRON, PROTON, FermionSpecies, GasState, chandrasekhar_energy, chandrasekhar_pressure, electron_gas,
    eos_finite_T_nonrel, eos_T0, fermi_momentum, relativity_parameter, sound_speed_sq_over_c2, trace,
)
