"""Fundamental physical constants used across the QuarkSum pipeline.

All values in SI units unless noted.

Provenance:
  - Defined constants (c, h, e, k_B, N_A): exact by 2019 SI redefinition.
  - Nucleon masses (m_p, m_n): derived from AME2020 mass excesses.
  - Electron mass (m_e): CODATA 2018 Penning-trap measurement.
  - Quark, lepton and boson masses: PDG 2026 central values (quarks are
    MS-bar current masses), adopted from the vendored snapshot
    inventory/data/particle_masses.json (distilled by tools/distill_pdg.py).
    THE ADOPTION CONTRACT: this file is the single Python root for these
    numbers; tests/test_pdg_drift_gate.py asserts it matches the JSON, so a
    future PDG edition bump fails loudly here instead of drifting silently.
  - Bohr magneton, vacuum permittivity, Bohr radius: CODATA 2018.
  - Rydberg energy: CODATA 2018.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalConstants:
    """Immutable collection of fundamental constants."""

    c: float = 2.99792458e8
    e: float = 1.602176634e-19
    h: float = 6.62607015e-34
    k_B: float = 1.380649e-23
    m_e: float = 9.1093837015e-31
    m_p: float = 1.6726219236278e-27
    m_n: float = 1.6749274980342e-27
    N_A: float = 6.02214076e23
    u: float = 1.66053906660e-27

    # Electromagnetic (CODATA 2018)
    mu_B: float = 9.2740100783e-24
    epsilon_0: float = 8.8541878128e-12
    a_0: float = 5.29177210903e-11

    # Atomic energy scale (CODATA 2018)
    E_rydberg_ev: float = 13.605693122994

    # Lepton masses (PDG 2026, kg — MeV values converted via e·1e6/c²)
    m_muon: float = 1.8835316270491e-28   # 105.6583755 MeV
    m_tau: float = 3.1676698428989e-27    # 1776.93246513409 MeV

    # Gauge boson masses (PDG 2026, kg)
    m_W: float = 1.4325916867682e-25      # 80362.5 MeV
    m_Z: float = 1.6255714944119e-25      # 91187.87329722135 MeV
    m_higgs: float = 2.2306616877982e-25  # 125130.9438281615 MeV

    # Light quark masses — MS-bar scheme (PDG 2026, MeV/c²)
    m_up_mev: float = 2.16
    m_down_mev: float = 4.7
    m_strange_mev: float = 92.9

    # Heavy quark masses — MS-bar scheme (PDG 2026, MeV/c²)
    m_charm_mev: float = 1272.9
    m_bottom_mev: float = 4186.0
    m_top_mev: float = 172603.5595874743

    @property
    def hbar(self) -> float:
        return self.h / (2.0 * math.pi)

    @property
    def c_squared(self) -> float:
        return self.c ** 2

    @property
    def MeV_to_J(self) -> float:
        return self.e * 1e6

    @property
    def MeV_to_kg(self) -> float:
        return self.e * 1e6 / self.c_squared

    @property
    def m_up_kg(self) -> float:
        return self.m_up_mev * self.MeV_to_kg

    @property
    def m_down_kg(self) -> float:
        return self.m_down_mev * self.MeV_to_kg


CONSTANTS = PhysicalConstants()


@dataclass(frozen=True)
class EarthDefaults:
    """Standard ambient conditions at sea level on Earth."""

    temperature: float = 293.15
    pressure: float = 101_325.0


EARTH = EarthDefaults()
