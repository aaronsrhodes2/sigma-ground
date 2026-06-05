"""Special-relativity & atomic-spectra analysis tools (standard physics).

Composite tools cascading through field.relativity and
field.interface.atomic_spectra.
"""
from __future__ import annotations

from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_EV_J = 1.602176634e-19


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        return None


def relativistic_energy_analysis(rest_mass_kg: float = 9.1093837015e-31,
                                 velocity_m_s: float = 2.6e8) -> dict[str, Any]:
    """Relativistic energy of a moving particle: rest energy (E0=m0 c^2),
    relativistic kinetic energy ((gamma-1) m0 c^2), and the energy-momentum
    invariant (m0 c^2)^2. Defaults: an electron at 0.867 c.
    e.g. relativistic_energy_analysis(9.109e-31, 2.6e8)."""
    from sigma_ground.field import relativity as R
    E0 = _safe(R.rest_energy, rest_mass_kg)
    KE = _safe(R.kinetic_energy_rel, rest_mass_kg, velocity_m_s)
    inv = _safe(R.energy_momentum_invariant, rest_mass_kg)
    results = {
        "rest_energy_J": E0,
        "rest_energy_MeV": (E0 / _EV_J / 1.0e6) if E0 is not None else None,
        "kinetic_energy_J": KE,
        "kinetic_energy_MeV": (KE / _EV_J / 1.0e6) if KE is not None else None,
        "energy_momentum_invariant_J2": inv,
    }
    return ToolResult(value=results, units="J, MeV, J^2",
                      source="sigma_ground.field.relativity",
                      provenance_tag="DERIVED",
                      formula="E0=m0 c^2; KE=(gamma-1) m0 c^2; E^2-(pc)^2=(m0 c^2)^2",
                      inputs={"rest_mass_kg": rest_mass_kg,
                              "velocity_m_s": velocity_m_s}).to_dict()


def zeeman_effect_analysis(total_angular_momentum_j: float = 1.0) -> dict[str, Any]:
    """Number of Zeeman sublevels a state of total angular momentum j splits
    into in a magnetic field (2j+1 values of m_j).
    e.g. zeeman_effect_analysis(1.0) -> 3 sublevels."""
    from sigma_ground.field.interface import atomic_spectra as A
    n = _safe(A.zeeman_splitting_count, total_angular_momentum_j)
    results = {"zeeman_sublevels": n}
    return ToolResult(value=results, units="dimensionless",
                      source="sigma_ground.field.interface.atomic_spectra",
                      provenance_tag="DERIVED",
                      formula="number of m_j values = 2j + 1",
                      inputs={"total_angular_momentum_j": total_angular_momentum_j}).to_dict()
