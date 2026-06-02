"""Nuclear physics — binding energy, mass defect, and the baryon-vs-mass gap.

These tools make the distinction between baryon COUNT and mass-ENERGY
explicit and computable. A nucleus is lighter than its constituent
nucleons by the binding energy (via E=mc²); the mass-defect FRACTION
varies by element (0 for ¹H, peaks ~0.8-0.9% near iron). That fraction
is exactly where a "gravity couples to baryon number" model would
diverge from "gravity couples to mass-energy."

Sources: proton/neutron masses from sigma_ground.field.constants
(PROTON_TOTAL_MEV, NEUTRON_TOTAL_MEV).
"""

from __future__ import annotations

from sigma_ground.mcp.provenance import ToolResult

_U_MEV = 931.49410242        # 1 atomic mass unit in MeV/c²
# Atomic-mass-unit values for the binding-energy bookkeeping. Tabulated
# nuclide masses (e.g. He-4 = 4.002602 u) are ATOMIC masses (they include
# the electrons), so the standard convention uses the hydrogen-ATOM mass
# for each proton; the Z electrons then cancel between the two sides.
_M_H_ATOM_U   = 1.00782503207   # ¹H atom (proton + electron)
_M_NEUTRON_U  = 1.00866491588   # free neutron


def nuclear_binding_energy(protons: int, neutrons: int,
                             measured_mass_u: float | None = None
                             ) -> ToolResult:
    """Total nuclear binding energy and mass defect for a nucleus.

    BE = [Z·m_p + N·m_n − M_nucleus] c².

    If `measured_mass_u` (the actual nuclide mass in atomic mass units) is
    given, the result is exact. If omitted, the binding energy is estimated
    from the semi-empirical mass formula (SEMF / Bethe-Weizsäcker) — good to
    a few percent for stable nuclei.

    Returns a dict: {binding_energy_MeV, binding_per_nucleon_MeV,
    mass_defect_u, mass_defect_fraction}.

    Examples:
      nuclear_binding_energy(2, 2, 4.002602)    # He-4, exact
      nuclear_binding_energy(26, 30)            # Fe-56, SEMF estimate
    """
    Z, N = int(protons), int(neutrons)
    A = Z + N
    if Z < 1 or N < 0 or A < 1:
        return ToolResult(value=None, source="invalid input",
                           inputs={"protons": Z, "neutrons": N})

    # Atomic convention: use ¹H-atom mass for each proton so the electrons
    # cancel against the atomic nuclide mass on the other side.
    constituent_MeV = (Z * _M_H_ATOM_U + N * _M_NEUTRON_U) * _U_MEV

    if measured_mass_u is not None:
        nucleus_MeV = float(measured_mass_u) * _U_MEV
        be_MeV = constituent_MeV - nucleus_MeV
        method = "exact (measured nuclide mass)"
    else:
        # Semi-empirical mass formula (Bethe-Weizsäcker), MeV
        a_v, a_s, a_c, a_a = 15.75, 17.8, 0.711, 23.7
        delta = 0.0
        if Z % 2 == 0 and N % 2 == 0:
            delta = 11.18 / (A ** 0.5)
        elif Z % 2 == 1 and N % 2 == 1:
            delta = -11.18 / (A ** 0.5)
        be_MeV = (a_v * A
                    - a_s * A ** (2.0 / 3.0)
                    - a_c * Z * (Z - 1) / A ** (1.0 / 3.0)
                    - a_a * (N - Z) ** 2 / A
                    + delta)
        nucleus_MeV = constituent_MeV - be_MeV
        method = "SEMF estimate (Bethe-Weizsäcker)"

    mass_defect_u = (constituent_MeV - nucleus_MeV) / _U_MEV
    defect_fraction = (constituent_MeV - nucleus_MeV) / constituent_MeV
    return ToolResult(
        value={
            "binding_energy_MeV": be_MeV,
            "binding_per_nucleon_MeV": be_MeV / A,
            "mass_defect_u": mass_defect_u,
            "mass_defect_fraction": defect_fraction,
        },
        units="",
        source=f"sigma-ground nuclear ({method})",
        formula="BE = [Z m_p + N m_n − M] c²",
        inputs={"protons": Z, "neutrons": N, "A": A,
                "measured_mass_u": measured_mass_u},
        notes=(f"A={A}. mass_defect_fraction is how much lighter the bound "
                f"nucleus is than its free nucleons — the quantity by which "
                f"baryon-count and mass-energy diverge. Peaks ~0.9% near iron."),
    )


def coulomb_force(charge1_c: float, charge2_c: float,
                    separation_m: float) -> ToolResult:
    """Coulomb's law: F = k q1 q2 / r², k = 1/(4π ε₀).

    Sign convention: positive = repulsive (like charges), negative =
    attractive (opposite charges). Charges in coulombs.

    Example (two protons 1 fm apart):
      coulomb_force(1.602e-19, 1.602e-19, 1e-15)
    """
    from sigma_ground.field.constants import EPS_0
    import math
    if separation_m <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"separation_m": separation_m})
    k = 1.0 / (4.0 * math.pi * EPS_0)
    F = k * charge1_c * charge2_c / (separation_m ** 2)
    return ToolResult(
        value=F, units="N",
        source="sigma-ground (Coulomb's law)",
        formula="F = q1 q2 / (4π ε₀ r²)",
        inputs={"charge1_c": charge1_c, "charge2_c": charge2_c,
                "separation_m": separation_m},
        notes="Positive = repulsive (like charges); negative = attractive.",
    )
