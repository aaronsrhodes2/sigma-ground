"""Nuclear matter from a Skyrme interaction: saturation properties, beta equilibrium (n, p, e, mu) and a neutron-star-core EOS.

The Skyrme energy-density functional for homogeneous matter (Vautherin & Brink 1972; Chabanat et al. 1998, Nucl. Phys. A635, 231), without
spin-orbit and Coulomb (both vanish in uniform matter at this level), with densities n_q in fm^-3 and energies in MeV:
    eps = (hbar^2/2m) (tau_n + tau_p)
        + (t0/4)  [(2 + x0) n^2 - (2 x0 + 1) (n_n^2 + n_p^2)]
        + (t3/24) n^sigma [(2 + x3) n^2 - (2 x3 + 1) (n_n^2 + n_p^2)]
        + (1/8) [t1 (2 + x1) + t2 (2 + x2)] tau n + (1/8) [t2 (2 x2 + 1) - t1 (2 x1 + 1)] (tau_n n_n + tau_p n_p)
with tau_q = (3/5) (3 pi^2)^(2/3) n_q^(5/3) (the Fermi-gas kinetic density), tau = tau_n + tau_p, n = n_n + n_p.
Chemical potentials mu_q = d eps/d n_q + m_q c^2 and the pressure P = sum_q mu_q n_q - (eps + rest-mass density) follow by thermodynamics.

Derivation chain (Golden Rule 1): FIRST_PRINCIPLES (the functional's form; thermodynamics; the free electron and muon gases from fermi_gas.py)
+ FITTED (the SLy4 parameter set below, fitted by Chabanat et al. 1998 to nuclear masses, radii and a neutron-matter EOS; copied, not refitted here).
The transcription of the parameters is CHECKED, not trusted: ``saturation_properties()`` must reproduce SLy4's published saturation point
(n_0 = 0.1595 fm^-3, E/A = -15.97 MeV, K = 229.9 MeV, J = 32.0 MeV, L = 45.9 MeV), and a wrong digit in any parameter moves at least one of them.

Golden Rule 2: ``regime`` is SAFE for n_b between 0.5 n_0 and 2.5 n_0 where the functional is anchored, EDGE up to 5 n_0 (extrapolated, Skyrme
sound speed can exceed c), WALL if c_s > c, BEYOND below 0.5 n_0 (no clusters or crust: homogeneous matter is mechanically unstable there).

PHYSICS_GAP: homogeneous matter only, so no crust (the star's outer ~1 km, which fixes radii but changes M_max by well under 1 %); neutron-star radii from
this EOS are NOT scored against data, only the maximum mass. PHYSICS_GAP: only one parametrization (SLy4); the experiment-based meta-model of the plan is not built yet.
PHYSICS_GAP: zero temperature; no hyperons, no deconfinement (quark matter is a separate module in the plan).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from sigma_ground.field.constants import C, HBAR, M_ELECTRON_KG, M_MUON_KG, M_NEUTRON_KG, M_PROTON_KG, MEV_TO_J
from sigma_ground.field.dense import fermi_gas as fg
from sigma_ground.field.dense.eos import TabulatedEOS

_HBARC_MEV_FM = HBAR * C / MEV_TO_J * 1e15                 # hbar c in MeV fm (derived from the SI constants)
_MC2_N = M_NEUTRON_KG * C * C / MEV_TO_J                   # MeV
_MC2_P = M_PROTON_KG * C * C / MEV_TO_J
_MC2_E = M_ELECTRON_KG * C * C / MEV_TO_J
_MC2_MU = M_MUON_KG * C * C / MEV_TO_J
_HB2_2M = _HBARC_MEV_FM ** 2 / (2.0 * 0.5 * (_MC2_N + _MC2_P))   # hbar^2/2m in MeV fm^2 with the mean nucleon mass (Chabanat's convention)
_CTAU = 0.6 * (3.0 * math.pi ** 2) ** (2.0 / 3.0)
_MEVFM3_TO_JM3 = MEV_TO_J * 1e45                            # MeV/fm^3 -> J/m^3
_FM3_TO_M3 = 1e45


@dataclass(frozen=True)
class Skyrme:
    name: str
    t0: float      # MeV fm^3
    t1: float      # MeV fm^5
    t2: float      # MeV fm^5
    t3: float      # MeV fm^(3+3 sigma)
    x0: float
    x1: float
    x2: float
    x3: float
    sigma: float

    def _pot(self, nn: float, npr: float) -> float:
        """Energy density without rest mass, MeV/fm^3."""
        n = nn + npr
        tn, tp = _CTAU * nn ** (5.0 / 3.0), _CTAU * npr ** (5.0 / 3.0)
        tau = tn + tp
        sq = nn * nn + npr * npr
        e = _HB2_2M * tau
        e += self.t0 / 4.0 * ((2.0 + self.x0) * n * n - (2.0 * self.x0 + 1.0) * sq)
        e += self.t3 / 24.0 * n ** self.sigma * ((2.0 + self.x3) * n * n - (2.0 * self.x3 + 1.0) * sq)
        e += (self.t1 * (2.0 + self.x1) + self.t2 * (2.0 + self.x2)) / 8.0 * tau * n
        e += (self.t2 * (2.0 * self.x2 + 1.0) - self.t1 * (2.0 * self.x1 + 1.0)) / 8.0 * (tn * nn + tp * npr)
        return e

    def energy_density(self, nn: float, npr: float) -> float:
        """Total energy density, rest mass included, MeV/fm^3."""
        return self._pot(nn, npr) + nn * _MC2_N + npr * _MC2_P

    def mu(self, nn: float, npr: float) -> tuple:
        """(mu_n, mu_p) in MeV including rest mass: the analytic derivatives d eps / d n_q of the functional (+ m_q c^2).

        Checked against central differences of ``_pot`` in the tests (``mu_numeric``).
        """
        n = nn + npr
        s = nn * nn + npr * npr
        a0, b0, a3, b3 = 2.0 + self.x0, 2.0 * self.x0 + 1.0, 2.0 + self.x3, 2.0 * self.x3 + 1.0
        c1 = (self.t1 * (2.0 + self.x1) + self.t2 * (2.0 + self.x2)) / 8.0
        c2 = (self.t2 * (2.0 * self.x2 + 1.0) - self.t1 * (2.0 * self.x1 + 1.0)) / 8.0
        tn, tp = _CTAU * nn ** (5.0 / 3.0), _CTAU * npr ** (5.0 / 3.0)
        tau = tn + tp
        out = []
        for nq, tq in ((nn, tn), (npr, tp)):
            dtq = (5.0 / 3.0) * _CTAU * nq ** (2.0 / 3.0)               # d tau_q / d n_q
            d = _HB2_2M * dtq
            d += self.t0 / 4.0 * (2.0 * a0 * n - 2.0 * b0 * nq)
            d += self.t3 / 24.0 * (self.sigma * n ** (self.sigma - 1.0) * (a3 * n * n - b3 * s)
                                   + n ** self.sigma * (2.0 * a3 * n - 2.0 * b3 * nq))
            d += c1 * (dtq * n + tau)
            d += c2 * (dtq * nq + tq)
            out.append(d)
        return out[0] + _MC2_N, out[1] + _MC2_P

    def mu_numeric(self, nn: float, npr: float) -> tuple:
        """(mu_n, mu_p) by central differences: an independent cross-check of ``mu``."""
        hn, hp = max(nn, 1e-12) * 1e-5, max(npr, 1e-12) * 1e-5
        mun = (self._pot(nn + hn, npr) - self._pot(nn - hn, npr)) / (2 * hn) + _MC2_N
        mup = (self._pot(nn, npr + hp) - self._pot(nn, npr - hp)) / (2 * hp) + _MC2_P
        return mun, mup

    def pressure(self, nn: float, npr: float) -> float:
        """Nucleon pressure in MeV/fm^3: P = sum mu_q n_q - eps (thermodynamic identity)."""
        mun, mup = self.mu(nn, npr)
        return mun * nn + mup * npr - self.energy_density(nn, npr)

    def energy_per_baryon(self, n: float, delta: float = 0.0) -> float:
        """E/A in MeV, rest mass subtracted (the nuclear-matter convention), asymmetry delta = (n_n - n_p)/n."""
        nn, npr = 0.5 * n * (1.0 + delta), 0.5 * n * (1.0 - delta)
        return self._pot(nn, npr) / n

    def saturation_properties(self) -> dict:
        """n_0, E/A, K, J, L of symmetric matter: the numbers the parameter set is quoted by."""
        def p_snm(n):
            return self.pressure(n / 2.0, n / 2.0)
        lo, hi = 0.08, 0.30
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if p_snm(mid) < 0.0:
                lo = mid
            else:
                hi = mid
        n0 = 0.5 * (lo + hi)
        h = 1e-4

        def ea(n, d=0.0):
            return self.energy_per_baryon(n, d)
        K = 9.0 * n0 * n0 * (ea(n0 + h) - 2.0 * ea(n0) + ea(n0 - h)) / (h * h)
        dd = 1e-3

        def sym(n):                                         # S(n) = 1/2 d^2(E/A)/d delta^2
            return 0.5 * (ea(n, dd) - 2.0 * ea(n, 0.0) + ea(n, -dd)) / (dd * dd)
        J = sym(n0)
        L = 3.0 * n0 * (sym(n0 + h) - sym(n0 - h)) / (2.0 * h)
        return {"n0_fm3": n0, "E_over_A_MeV": ea(n0), "K_MeV": K, "J_MeV": J, "L_MeV": L}


# SLy4: Chabanat, Bonche, Haensel, Meyer, Schaeffer 1998, Nucl. Phys. A635, 231, table 1 (the set behind Douchin & Haensel 2001).
SLY4 = Skyrme("SLy4", t0=-2488.91, t1=486.82, t2=-546.39, t3=13777.0, x0=0.834, x1=-0.344, x2=-1.0, x3=1.354, sigma=1.0 / 6.0)


@dataclass(frozen=True)
class BetaState:
    n_b_fm3: float
    n_n: float
    n_p: float
    n_e: float
    n_mu: float
    pressure_mev_fm3: float
    energy_density_mev_fm3: float          # rest mass included
    regime: str


def _lepton_density(mu_mev: float, mc2: float, species: fg.FermionSpecies) -> float:
    if mu_mev <= mc2:
        return 0.0
    x = math.sqrt((mu_mev / mc2) ** 2 - 1.0)
    return species.n0_m3 * x ** 3 / _FM3_TO_M3


def beta_equilibrium(n_b: float, model: Skyrme = SLY4, muons: bool = True) -> BetaState:
    """Charge-neutral beta-equilibrium composition at baryon density n_b (fm^-3): mu_n = mu_p + mu_e, mu_mu = mu_e, n_p = n_e + n_mu."""
    if n_b <= 0:
        raise ValueError("baryon density must be positive")

    def charge_imbalance(npr: float) -> float:
        mun, mup = model.mu(n_b - npr, npr)
        mue = mun - mup
        ne = _lepton_density(mue, _MC2_E, fg.ELECTRON)
        nm = _lepton_density(mue, _MC2_MU, fg.MUON) if muons else 0.0
        return npr - ne - nm

    lo, hi = 1e-12, 0.5 * n_b
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if charge_imbalance(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    npr = 0.5 * (lo + hi)
    nn = n_b - npr
    mun, mup = model.mu(nn, npr)
    mue = mun - mup
    ne = _lepton_density(mue, _MC2_E, fg.ELECTRON)
    nm = _lepton_density(mue, _MC2_MU, fg.MUON) if muons else 0.0
    # lepton pressure and energy density (fermi_gas in SI -> MeV/fm^3)
    P = model.pressure(nn, npr)
    eps = model.energy_density(nn, npr)
    for dens, mc2, sp in ((ne, _MC2_E, fg.ELECTRON), (nm, _MC2_MU, fg.MUON)):
        if dens > 0:
            x = (dens * _FM3_TO_M3 / sp.n0_m3) ** (1.0 / 3.0)
            P += fg.chandrasekhar_pressure(x, sp) / _MEVFM3_TO_JM3
            eps += fg.chandrasekhar_energy(x, sp) / _MEVFM3_TO_JM3
    n0 = 0.1595
    regime = "BEYOND" if n_b < 0.5 * n0 else ("SAFE" if n_b <= 2.5 * n0 else "EDGE")
    return BetaState(n_b, nn, npr, ne, nm, P, eps, regime)


def npe_mu_eos(model: Skyrme = SLY4, n_min: float = 0.08, n_max: float = 2.4, points: int = 600, muons: bool = True) -> TabulatedEOS:
    """Neutron-star-core EOS (n, p, e, mu in beta equilibrium) as a TabulatedEOS in SI units.

    Starts at ``n_min`` (default 0.5 n_0), where the pressure is already positive and increasing; below it the matter is not homogeneous (PHYSICS_GAP:
    no crust). The sound speed along the sequence is dP/d eps by a central difference of the model itself, not of the table.
    """
    P, E, N, CS = [], [], [], []
    for i in range(points):
        nb = math.exp(math.log(n_min) + (math.log(n_max) - math.log(n_min)) * i / (points - 1))
        s = beta_equilibrium(nb, model, muons)
        h = nb * 1e-5
        a, b = beta_equilibrium(nb - h, model, muons), beta_equilibrium(nb + h, model, muons)
        cs2 = (b.pressure_mev_fm3 - a.pressure_mev_fm3) / (b.energy_density_mev_fm3 - a.energy_density_mev_fm3)
        if s.pressure_mev_fm3 <= 0.0 or cs2 <= 0.0:
            continue
        if P and s.pressure_mev_fm3 * _MEVFM3_TO_JM3 <= P[-1]:
            continue
        P.append(s.pressure_mev_fm3 * _MEVFM3_TO_JM3)
        E.append(s.energy_density_mev_fm3 * _MEVFM3_TO_JM3)
        N.append(nb * _FM3_TO_M3)
        CS.append(cs2)
    return TabulatedEOS(f"{model.name}_npemu_core", P, E, N, CS, 0.5 * (M_NEUTRON_KG + M_PROTON_KG))
