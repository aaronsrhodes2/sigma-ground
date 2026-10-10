"""T5 v2 (repair pass) -- the torsion-balance window: sigma exclusion between lambda = 1 cm and 35 km.

Pre-registered in PREREG.txt (written before any bound was looked up or any number computed). The criterion is unchanged.

Model (misc/UNIFICATION_PROGRAM.md section 6; scripts/unification/sigma_exclusion.py):
  m_i(sigma) = m_i0 [1 + f_i sigma],  f_i = dln m_i/dsigma at 0  (repo inventory, inventory/core/sigma.py)
  static sigma(x) = (2 G xi / c^2) Int rho(x') e^{-|x-x'|/lambda}/|x-x'| d^3x'   (source = rho c^2, sigma charge per unit
  source mass s = 1)  =>  extra potential between body i and source mass M:  V = +2 xi f_i s G m_i M e^{-r/lambda}/r.
  Composition-dependent part (EP):   Delta a_12 = -2 xi (f_1 - f_2) s  x [G x Yukawa geometry]   (toward the attractor)
  Composition-independent part (ISL): alpha_ISL = -2 xi f s   (repulsive; UNIFICATION_PROGRAM.md section 6)
Published Yukawa form (Adelberger et al. 2009 eq. 5/7; Smith et al. 2000 eq. 2/26):
  Delta a_12 / g = alpha~ Delta(q/mu)_12 (q/mu)_A x [same geometry];  q = B panel.
  => xi_max(lambda) = alpha~_B,max(lambda) |Delta(B/mu)_12| (B/mu)_A / (2 |Delta f_12| s)          (pair mapping kappa)
     (B/mu)_A set to 1; the published 238U value 0.9962 would make every xi_max 0.4 % SMALLER, so 1 is conservative.

Data (all [LIT]; each paper verified by web search in the v1 run and by the physics skeptic):
  EW99  Smith et al. 2000, PRD 61, 022001: Cu vs "Pb" alloy toward a 3-t 238U attractor.
        Sec. II: "The Pb bodies were made from a 92% Pb/7.75% Sb/0.25% Sn alloy"  (smith2000.txt line 362)
        Eq. 22: Delta a = a_Cu - a_Pb = (1.0 +- 2.6 +- 0.9)e-13 cm/s^2 toward 238U; symmetric 95% |Delta a| < 5.7e-13.
        Eq. 4: g' = 9.2e-5 cm/s^2.  Table X: S(lambda) = Delta(q/mu) <(q/mu)_A I_A(lambda)>, q = N+Z, 1 cm .. 50 m.
        Table IX: I_E(lambda).  Table XI: B/mu and L/mu of Cu, the actual "Pb" alloy, 238U.
  EW08  Wagner PhD thesis (Be-Ti and Be-Al, Earth + local topography), as plotted in Adelberger et al. 2009, PPNP 62, 102,
        Fig. 5 lower-left (q = B) -- DIGITIZED (digitize_figures.py). Reading uncertainty: the digitizer reads 0.11 dex
        STRONGER than two independent anchors (Table X at 7 cm; the text's |alpha| = 1 at 56 um for the ISL figure), so
        every digitized curve is weakened by CONSERVATIVE_DEX = 0.15 dex.
        Order-of-magnitude cross-check from text numbers only: Adelberger 2009 Table 4 (local topography g_perp,max =
        7.2e-5 m/s^2, R_A = 5 m - 20 km) with Adelberger 2009 eq. 16 and Wagner et al. 2012 CQG 29 184002 Table 3.
  ISL   Adelberger et al. 2009, Figs. 9 (lambda <= 1 cm) and 10 (lambda >= 1 cm) -- DIGITIZED. Fig. 10 caption: laboratory
        constraints (about 1 cm - 1 m) and geophysical constraints taken from an earlier review (about 1 m - 35 km).
PLATINUM 21: nuclear binding energies from Mentat (nuclear_binding_energy, measured nuclide masses); isotope masses and
abundances from the repo inventory (inventory/data/atomic_weights.json, NIST); f_i from inventory/core/sigma.py;
MICROSCOPE / clock / white-dwarf curves, R_Earth and AU recomputed by importing scripts/unification/sigma_exclusion.py.
Run from the repo root:  python -X utf8 <this file>
"""
import json, math, os, re, sys
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

REPO = r"D:/Aaron/development/sigma-ground"
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "scripts", "unification"))
sys.path.insert(0, REPO)
from mentat_client import mentat
from sigma_ground.inventory.core.sigma import XI, three_measures_atom, _PROTON_BARE_MEV, _NEUTRON_BARE_MEV, _MEV_TO_KG
from sigma_ground.inventory.core.constants import CONSTANTS

SRC = {}
CACHE_F = os.path.join(OUT, "mentat_cache_v2.json")       # starts empty: every binding energy is a fresh Mentat call
CACHE = json.load(open(CACHE_F)) if os.path.exists(CACHE_F) else {}

def leaf(tool, args, label):
    key = tool + json.dumps(args, sort_keys=True)
    if key not in CACHE:
        v, s = mentat(tool, args)
        CACHE[key] = [v, s]
    v, s = CACHE[key]
    SRC[label] = s
    return v

# ------------------------------------------------------------------ sigma charge f and Yukawa charges per element
AW = json.load(open(os.path.join(REPO, "sigma_ground", "inventory", "data", "atomic_weights.json"), encoding="utf-8"))
SRC["isotope masses + abundances"] = "repo inventory: sigma_ground/inventory/data/atomic_weights.json (" + AW["_meta"]["source"] + ")"
SRC["f_i (sigma charge)"] = "repo: sigma_ground/inventory/core/sigma.py three_measures_atom, central difference d = 1e-6"
d = 1e-6

def f_isotope(Z, A, mass_u):
    be = leaf("nuclear_binding_energy", {"protons": Z, "neutrons": A - Z, "measured_mass_u": mass_u},
              f"BE Z={Z} A={A}")["binding_energy_MeV"]
    m = lambda s: three_measures_atom(Z, A - Z, be, s)["atom_stable_mass_kg"]
    f_repo = (math.log(m(d)) - math.log(m(-d))) / (2 * d)
    # sensitivity model D [DERIVED, not the repo's model]: nuclear Coulomb energy also scales with Lambda_QCD (radii ~ 1/Lambda,
    # alpha fixed), so only the Higgs-sourced bare-quark masses and the electron mass are sigma-invariant.
    higgs = (Z * _PROTON_BARE_MEV + (A - Z) * _NEUTRON_BARE_MEV) * _MEV_TO_KG + Z * CONSTANTS.m_e
    f_D = 1.0 - higgs / m(0.0)
    return f_repo, be, f_D

def element(sym):
    e = AW[sym]; Z = e["z"]
    isos = [(i["a"], i["mass_u"], i["abundance"]) for i in e["isotopes"] if i["abundance"]]
    mtot = sum(x * m for _, m, x in isos)
    fr = [f_isotope(Z, a, m) for a, m, x in isos]
    f = sum((x * m / mtot) * r[0] for (a, m, x), r in zip(isos, fr))        # mass-weighted
    fD = sum((x * m / mtot) * r[2] for (a, m, x), r in zip(isos, fr))
    B = sum(x * a for a, _, x in isos)
    return dict(symbol=sym, Z=Z, f=f, f_modelD=fD, B_mu=B / mtot, Z_mu=Z / mtot, N_mu=(B - Z) / mtot, mu=mtot,
                isotopes=[dict(A=a, mass_u=m, x=x) for a, m, x in isos])

ELEMS = {s: element(s) for s in ("Be", "Al", "Ti", "Cu", "Pb", "Pt", "W", "Mo", "Ta", "U", "Sb", "Sn")}
json.dump(CACHE, open(CACHE_F, "w"), indent=0)

# published charges (validation of the inventory against the papers' own tables)
PUB_BMU = {"Be": 0.99865, "Al": 1.00068, "Ti": 1.00108, "Cu": 1.00112, "Pt": 1.00018, "Pb": 1.00012}   # Adelberger 2009 Table 1
PUB_BMU_WAGNER = {"Be": 0.99865, "Ti": 1.00107, "Al": 1.00068, "Pt": 1.00015}                        # Wagner 2012 Table 1
TXI = {"Cu": dict(B_mu=1.0011166, L_mu=0.4563649), "Pb alloy": dict(B_mu=1.0001694, L_mu=0.3975718),
       "238U": dict(B_mu=0.9962151, L_mu=0.3850916)}                                                  # Smith 2000 Table XI
BMU_CU_SMITH, BMU_PB_ALLOY_SMITH = TXI["Cu"]["B_mu"], TXI["Pb alloy"]["B_mu"]
validation_bmu = {s: dict(repo=ELEMS[s]["B_mu"], published=PUB_BMU[s], diff=ELEMS[s]["B_mu"] - PUB_BMU[s]) for s in PUB_BMU}

# ------------------------------------------------------------------ the "Pb" test-body alloy of Smith et al. 2000 (PUBLISHED)
ALLOY_W = {"Pb": 0.92, "Sb": 0.0775, "Sn": 0.0025}     # [LIT] Smith 2000 Sec. II, mass fractions; Table XI footnote a
alloy = dict(mass_fractions=ALLOY_W,
             f=sum(w * ELEMS[s]["f"] for s, w in ALLOY_W.items()),
             f_modelD=sum(w * ELEMS[s]["f_modelD"] for s, w in ALLOY_W.items()),
             B_mu_computed=sum(w * ELEMS[s]["B_mu"] for s, w in ALLOY_W.items()),          # B/mu mixes by mass fraction
             L_mu_computed=sum(w * ELEMS[s]["Z_mu"] for s, w in ALLOY_W.items()),
             B_mu_published=TXI["Pb alloy"]["B_mu"], L_mu_published=TXI["Pb alloy"]["L_mu"])
_u238 = next(i for i in ELEMS["U"]["isotopes"] if i["A"] == 238)
validation_tableXI = {
    "Cu": dict(B_mu_repo=ELEMS["Cu"]["B_mu"], B_mu_pub=TXI["Cu"]["B_mu"], L_mu_repo=ELEMS["Cu"]["Z_mu"], L_mu_pub=TXI["Cu"]["L_mu"]),
    "Pb alloy (92/7.75/0.25)": dict(B_mu_repo=alloy["B_mu_computed"], B_mu_pub=alloy["B_mu_published"],
                                    L_mu_repo=alloy["L_mu_computed"], L_mu_pub=alloy["L_mu_published"]),
    "238U (inventory 238U mass)": dict(B_mu_repo=238 / _u238["mass_u"], L_mu_repo=92 / _u238["mass_u"],
                                       B_mu_pub=TXI["238U"]["B_mu"], L_mu_pub=TXI["238U"]["L_mu"]),
}
for v in validation_tableXI.values():
    v["ratio_B_pub_over_repo"] = v["B_mu_pub"] / v["B_mu_repo"]
    v["ratio_L_pub_over_repo"] = v["L_mu_pub"] / v["L_mu_repo"]

def pair(a, b, fa, fb, bmu_a, bmu_b):
    dBmu, df = bmu_a - bmu_b, fa - fb
    return dict(pair=f"{a}-{b}", delta_B_mu=dBmu, delta_f=df, kappa=abs(dBmu) / (2 * abs(df)),
                sign_relation="opposite" if dBmu * df < 0 else "same")

PAIRS = {
    "Be-Ti": pair("Be", "Ti", ELEMS["Be"]["f"], ELEMS["Ti"]["f"], PUB_BMU_WAGNER["Be"], PUB_BMU_WAGNER["Ti"]),
    "Be-Al": pair("Be", "Al", ELEMS["Be"]["f"], ELEMS["Al"]["f"], PUB_BMU_WAGNER["Be"], PUB_BMU_WAGNER["Al"]),
    "Cu-Pb alloy (published 92/7.75/0.25)": pair("Cu", "Pb alloy", ELEMS["Cu"]["f"], alloy["f"], BMU_CU_SMITH, BMU_PB_ALLOY_SMITH),
    "Cu-Pb (sensitivity: pure natural Pb f)": pair("Cu", "Pb", ELEMS["Cu"]["f"], ELEMS["Pb"]["f"], BMU_CU_SMITH, BMU_PB_ALLOY_SMITH),
}
KAPPA_CUPB = PAIRS["Cu-Pb alloy (published 92/7.75/0.25)"]["kappa"]
KAPPA_CUPB_PUREPB = PAIRS["Cu-Pb (sensitivity: pure natural Pb f)"]["kappa"]
kappa_ew08 = max(PAIRS["Be-Ti"]["kappa"], PAIRS["Be-Al"]["kappa"])      # pre-registered conservative pair choice
DF_CUPB = PAIRS["Cu-Pb alloy (published 92/7.75/0.25)"]["delta_f"]

# ------------------------------------------------------------------ EW99: Smith et al. 2000, tabulated
DA95 = 5.7e-13          # cm/s^2, symmetric 95% CL on |a_Cu - a_Pb|     [LIT Smith 2000 eq. 22 + text]
DA_MEAN, DA_STAT, DA_SYS = 1.0e-13, 2.6e-13, 0.9e-13                    # [LIT Smith 2000 eq. 22], 1 sigma each
GPRIME = 9.2e-5         # cm/s^2, gravitational acceleration toward the attractor [LIT Smith 2000 eq. 4]
TABX = [(0.010, 1.23e-8), (0.014, 1.75e-7), (0.020, 1.53e-6), (0.028, 7.46e-6), (0.05, 5.45e-5), (0.07, 1.25e-4),
        (0.10, 2.47e-4), (0.20, 5.61e-4), (0.50, 8.44e-4), (1.0, 9.19e-4), (2.0, 9.41e-4), (5.0, 9.48e-4),
        (10.0, 9.48e-4), (20.0, 9.49e-4), (50.0, 9.49e-4)]                       # [LIT Smith 2000 Table X, q = N+Z = B]
TABIX = [(1e1, 3.94e-7), (1e2, 4.31e-6), (1e3, 4.69e-5), (1e4, 9.81e-4), (1e5, 1.36e-2), (1e6, 1.72e-1),
         (1e7, 8.77e-1), (1e8, 9.98e-1)]                                          # [LIT Smith 2000 Table IX, I_E]
DBMU_CUPB = BMU_CU_SMITH - BMU_PB_ALLOY_SMITH
_TL = np.array([t[0] for t in TABX]); _TS = np.log([t[1] for t in TABX])

def S_B(lam, interp="inv"):
    """Table X. interp='inv': ln S linear in 1/lambda (Yukawa ~ e^{-r/lambda}); 'loglog': ln S linear in ln lambda.
    Saturated beyond 50 m (I_A -> 1). None below 1 cm (no published sensitivity)."""
    if lam < _TL[0] * (1 - 1e-12):
        return None
    if lam >= _TL[-1]:
        return float(np.exp(_TS[-1]))
    if interp == "loglog":
        return float(np.exp(np.interp(math.log(lam), np.log(_TL), _TS)))
    return float(np.exp(np.interp(1 / lam, (1 / _TL)[::-1], _TS[::-1])))

def I_E(lam):
    L = np.log10([t[0] for t in TABIX]); I = np.log10([t[1] for t in TABIX])
    if lam < 10:
        return 0.0
    return float(10 ** np.interp(math.log10(lam), L, I))

def alpha_B_ew99(lam, da=DA95, interp="inv"):
    s = S_B(lam, interp)
    if s is None:
        return None
    seff = s - DBMU_CUPB * I_E(lam)            # Smith eq. 26: [I_A (q/mu)_A - I_E (q/mu)_E], (B/mu)_E = 1
    return da / (GPRIME * seff)

def xi_ew99(lam, kappa=None, da=DA95, interp="inv"):
    a = alpha_B_ew99(lam, da, interp)
    return None if a is None else a * (KAPPA_CUPB if kappa is None else kappa)

# ------------------------------------------------------------------ sign of sigma's EP signal at the 1 cm edge
DA_SIG = math.hypot(DA_STAT, DA_SYS)
def _sym(p):   # symmetric region |da| < a holding probability p (Smith's convention); solved in units of 1e-13 cm/s^2
    u, sg = DA_MEAN / 1e-13, DA_SIG / 1e-13
    return 1e-13 * brentq(lambda a: norm.cdf(a, u, sg) - norm.cdf(-a, u, sg) - p, 1e-6, 100.0, xtol=1e-12)
a_sym95, a_sym68 = _sym(0.95), _sym(0.6827)
# sigma (repo sign, repulsive): a_Cu - a_Pb (toward 238U) = -2 xi (f_Cu - f_alloy) g' I_A
SIGMA_SIGN = -1 if DF_CUPB > 0 else +1
z2, z1 = norm.ppf(0.975), norm.ppf(0.95)
DA_BOUNDS = {
    "published, sign-agnostic symmetric |da| (pre-registered input)": DA95,
    "repo sign (repulsive), edge of the central 95% interval": abs(DA_MEAN - z2 * DA_SIG) if SIGMA_SIGN < 0 else DA_MEAN + z2 * DA_SIG,
    "repo sign (repulsive), one-sided 95%": abs(DA_MEAN - z1 * DA_SIG) if SIGMA_SIGN < 0 else DA_MEAN + z1 * DA_SIG,
    "attractive sign (if sigma's sign were flipped), edge of the central 95% interval": DA_MEAN + z2 * DA_SIG if SIGMA_SIGN < 0 else abs(DA_MEAN - z2 * DA_SIG),
}

# ------------------------------------------------------------------ digitized curves
DIG = json.load(open(os.path.join(OUT, "digitized", "digitized_bounds.json")))
CONSERVATIVE_DEX = 0.15      # digitizer reads ~0.11 dex too strong at two anchors (measured below); applied in the weakening direction

def curve(name):
    pts = np.array(DIG[name]["yellow_lower_boundary"], float)
    pts = pts[np.argsort(pts[:, 0])]
    return pts[:, 0], pts[:, 1]

x5, y5 = curve("fig5_B_panel"); x9, y9 = curve("fig9_ISL"); x10, y10 = curve("fig10_ISL")

def interp_log(xs, ys, lam):
    l = math.log10(lam)
    if l < xs[0] or l > xs[-1]:
        return None
    return 10 ** float(np.interp(l, xs, ys))

def alpha_env_fig5(lam):
    return interp_log(x5, y5, lam)

_xi = np.concatenate([x9[x9 <= -2.0], x10[x10 > -2.0]]); _yi = np.concatenate([y9[x9 <= -2.0], y10[x10 > -2.0]])
_o = np.argsort(_xi); xisl, yisl = _xi[_o], _yi[_o]
def alpha_isl(lam):
    return interp_log(xisl, yisl, lam)

F_ISL = min(ELEMS[s]["f"] for s in ("W", "Mo", "Ta", "Cu", "Pb", "Be", "Al", "Ti", "Pt", "U"))

def xi_isl(lam, dex=CONSERVATIVE_DEX):
    a = alpha_isl(lam)
    return None if a is None else a * 10 ** dex / (2 * F_ISL)

def a08_of(lam):
    """EW08 = the Fig. 5 envelope where it lies below EW99 by > 0.15 dex (otherwise the envelope is EW99 itself)."""
    a99 = alpha_B_ew99(lam); aenv = alpha_env_fig5(lam)
    return aenv if (aenv is not None and lam >= 0.5 and (a99 is None or aenv < a99 / 10 ** 0.15)) else None

def xi_ew08(lam, firm_only=False):
    a = a08_of(lam)
    if a is None or (firm_only and lam > 1.0e4):      # Adelberger 2009: 10 - 1000 km strength from "preliminary calculations"
        return None
    return a * 10 ** CONSERVATIVE_DEX * kappa_ew08

# ------------------------------------------------------------------ digitizer bias at independent anchors
ISL_ANCHOR_LAM, ISL_ANCHOR_ALPHA = 56e-6, 1.0     # [LIT] Adelberger 2009 text: |alpha| = 1 Yukawa must have lambda <= 56 um
dig_bias = dict(
    ISL_at_56um=dict(digitized_alpha=alpha_isl(ISL_ANCHOR_LAM), paper_text_alpha=ISL_ANCHOR_ALPHA,
                     dex=math.log10(alpha_isl(ISL_ANCHOR_LAM) / ISL_ANCHOR_ALPHA)),
    fig5_vs_tableX={f"{l:g} m": dict(fig5_digitized=alpha_env_fig5(l), tableX_computed=alpha_B_ew99(l),
                                     ratio=alpha_env_fig5(l) / alpha_B_ew99(l),
                                     dex=math.log10(alpha_env_fig5(l) / alpha_B_ew99(l)))
                    for l in (0.07, 0.1, 0.2, 0.5)},          # Fig. 5 digitized range starts at 5.2 cm
)
_g = np.logspace(-5, -4, 4001)
_v = np.array([alpha_isl(l) for l in _g])
_i = int(np.where(_v >= 1.0)[0].max())
dig_bias["ISL_lambda_where_digitized_alpha_eq_1_um"] = 1e6 * brentq(lambda l: alpha_isl(l) - 1.0, _g[_i], _g[_i + 1])
dig_bias["most_negative_dex"] = min([dig_bias["ISL_at_56um"]["dex"]] + [v["dex"] for v in dig_bias["fig5_vs_tableX"].values()])
dig_bias["applied_weakening_dex"] = CONSERVATIVE_DEX

# ------------------------------------------------------------------ continuous roots
def largest_allowed(fun, xi, lo, hi, n=6000):
    """Largest lambda in [lo, hi] with fun(lambda) >= xi (xi allowed): last crossing on a log grid, refined by brentq."""
    g = np.logspace(math.log10(lo), math.log10(hi), n)
    vals = [fun(l) for l in g]
    ok = [i for i, v in enumerate(vals) if v is not None and v >= xi]
    if not ok:
        return None
    i = max(ok)
    if i == n - 1:
        return hi
    if vals[i + 1] is None:
        return float(g[i])
    return float(brentq(lambda l: fun(l) - xi, g[i], g[i + 1], xtol=1e-15, rtol=1e-13))

XIS = {"0.13": 0.13, "0.1571": 0.1571, "0.1582": XI, "0.17": 0.17}

# EP-only sliver edges
sliver = {}
for k, xi in XIS.items():
    sliver[k] = dict(
        published_alloy_sign_agnostic_inv=largest_allowed(lambda l: xi_ew99(l), xi, 0.01, 0.014),
        published_alloy_sign_agnostic_loglog=largest_allowed(lambda l: xi_ew99(l, interp="loglog"), xi, 0.01, 0.014),
        pure_Pb_sensitivity=largest_allowed(lambda l: xi_ew99(l, KAPPA_CUPB_PUREPB), xi, 0.01, 0.014),
        v1_grid_method_same_inputs=(lambda g: max([l for l in g if xi_ew99(l) >= xi], default=None))(
            np.logspace(math.log10(0.01), math.log10(3.5e4), 4000)),          # v1's lam_cross, kept to show the grid artifact
    )
    for name, da in DA_BOUNDS.items():
        sliver[k][name] = largest_allowed(lambda l, da=da: xi_ew99(l, da=da), xi, 0.01, 0.014)

sign_analysis = dict(
    convention="Smith 2000 eq. 22: Delta a = a_Cu - a_Pb toward 238U = (1.0 +- 2.6 +- 0.9)e-13 cm/s^2 [LIT]",
    sigma_prediction="repo sigma is repulsive (UNIFICATION_PROGRAM.md section 6): a_Cu - a_Pb = -2 xi (f_Cu - f_alloy) g' I_A",
    delta_f_Cu_minus_alloy=DF_CUPB, predicted_sign=SIGMA_SIGN, measured_central=DA_MEAN, sigma_total=DA_SIG,
    symmetric_68_reproduced=a_sym68, symmetric_95_reproduced=a_sym95,
    bounds_cm_s2=DA_BOUNDS,
    xi_max_at_1cm={k: xi_ew99(0.01, da=v) for k, v in DA_BOUNDS.items()},
    xi_max_at_1cm_pure_Pb_sensitivity={k: xi_ew99(0.01, KAPPA_CUPB_PUREPB, da=v) for k, v in DA_BOUNDS.items()},
    literal_criterion_if_this_bound_were_used={k: ("PASS" if xi_ew99(0.01, da=v) < 0.13 else "FAIL") for k, v in DA_BOUNDS.items()},
    largest_xi_with_a_sliver={k: xi_ew99(0.01, da=v) for k, v in DA_BOUNDS.items()},
)

# ------------------------------------------------------------------ evaluate across the window and beyond
lams = np.unique(np.concatenate([np.logspace(-6, math.log10(3.5e4), 2400), [0.01, 0.0101, 0.0102, 0.0105, 1.0, 1e3, 1e4, 3.5e4]]))
rows = []
for lam in lams:
    a99 = alpha_B_ew99(lam)
    a08 = a08_of(lam)
    rows.append(dict(
        lam=float(lam),
        alpha_B_EW99=a99,
        xi_EW99=None if a99 is None else a99 * KAPPA_CUPB,
        xi_EW99_repo_sign=xi_ew99(lam, da=DA_BOUNDS["repo sign (repulsive), edge of the central 95% interval"]),
        xi_EW99_pure_Pb=None if a99 is None else a99 * KAPPA_CUPB_PUREPB,
        alpha_B_fig5_envelope=alpha_env_fig5(lam),
        alpha_B_EW08=a08,
        xi_EW08=xi_ew08(lam),
        xi_EW08_firm=xi_ew08(lam, firm_only=True),
        alpha_ISL=alpha_isl(lam),
        xi_ISL=xi_isl(lam),
    ))

def best(r, keys):
    v = [r[k] for k in keys if r[k] is not None]
    return min(v) if v else None

for r in rows:
    r["xi_max_EP"] = best(r, ("xi_EW99", "xi_EW08"))
    r["xi_max_EP_firm"] = best(r, ("xi_EW99", "xi_EW08_firm"))
    r["xi_max_all"] = best(r, ("xi_EW99", "xi_EW08", "xi_ISL"))

win = [r for r in rows if 0.01 * (1 - 1e-12) <= r["lam"] <= 3.5e4]
worst_EP = max(win, key=lambda r: r["xi_max_EP"] if r["xi_max_EP"] is not None else -1)
uncovered_EP = [r["lam"] for r in win if r["xi_max_EP"] is None]
worst_all = max(win, key=lambda r: r["xi_max_all"])
worst_isl = max(win, key=lambda r: r["xi_ISL"])

# ------------------------------------------------------------------ fixed points in the window (direct evaluation, no row lookup)
edge = {}
for lam in (0.010, 0.0105, 0.011, 0.012, 0.014, 0.020, 0.028, 0.05, 0.10, 1.0, 10.0, 50.0, 1e3, 1e4, 3.5e4):
    a99 = alpha_B_ew99(lam)
    edge[f"{lam:g} m"] = dict(alpha_B_EW99=a99, xi_max_EW99_published_alloy=a99 * KAPPA_CUPB,
                              xi_max_EW99_pure_Pb_sensitivity=a99 * KAPPA_CUPB_PUREPB,
                              xi_max_EW99_repo_sign=xi_ew99(lam, da=DA_BOUNDS["repo sign (repulsive), edge of the central 95% interval"]),
                              xi_max_EW08=xi_ew08(lam), xi_max_EW08_is_preliminary=bool(lam > 1e4 and xi_ew08(lam) is not None),
                              xi_max_ISL=xi_isl(lam), xi_max_ISL_nominal_reading=xi_isl(lam, 0.0),
                              exclusion_factor_EW99_xi_0p13=0.13 / (a99 * KAPPA_CUPB),
                              exclusion_factor_EW99_xi_0p1582=XI / (a99 * KAPPA_CUPB))

# ------------------------------------------------------------------ largest allowed lambda (continuous roots)
lam_allowed = {}
for k, xi in XIS.items():
    lam_allowed[k] = dict(
        EP_EW99_published_alloy=sliver[k]["published_alloy_sign_agnostic_inv"],
        EP_EW99_published_alloy_loglog_interp=sliver[k]["published_alloy_sign_agnostic_loglog"],
        EP_EW99_repo_sign=sliver[k]["repo sign (repulsive), edge of the central 95% interval"],
        ISL_padded_0p15dex=largest_allowed(lambda l: xi_isl(l), xi, 1e-6, 3.5e4),
        ISL_nominal_reading_biased_strong=largest_allowed(lambda l: xi_isl(l, 0.0), xi, 1e-6, 3.5e4),
    )

# ------------------------------------------------------------------ text-only order-of-magnitude check at the topography plateau
G_PERP_TOPO = 7.2e-5   # m/s^2 [LIT Adelberger 2009 Table 4, local topography, R_A = 5 m - 2e4 m]
TOPO_DATA = {   # 1e-15 m/s^2, (mean, 1 sigma) toward North and West  [LIT]
    "Be-Ti (Adelberger 2009 eq. 16 = Wagner 2012 Table 3)": ("Be-Ti", dict(N=(0.6, 3.1), W=(-2.5, 3.5))),
    "Be-Al (Adelberger 2009 eq. 16, behind the digitized Fig. 5)": ("Be-Al", dict(N=(-2.6, 2.5), W=(0.7, 2.5))),
    "Be-Al (Wagner 2012 Table 3, final)": ("Be-Al", dict(N=(-1.2, 2.2), W=(0.2, 2.4))),
}
topo = {}
for name, (p, comp) in TOPO_DATA.items():
    da95 = max(abs(m) + 1.96 * s for m, s in comp.values()) * 1e-15
    a_txt = da95 / (abs(PAIRS[p]["delta_B_mu"]) * G_PERP_TOPO)
    topo[name] = dict(delta_a_95_m_s2=da95, alpha_B_text_only=a_txt,
                      alpha_B_fig5_digitized_at_20km=alpha_env_fig5(2e4),
                      dex_digitized_minus_text=math.log10(alpha_env_fig5(2e4) / a_txt),
                      xi_max_text_only=da95 / (2 * abs(PAIRS[p]["delta_f"]) * G_PERP_TOPO),
                      xi_max_EW08_digitized_padded_at_20km=xi_ew08(2e4))
_dx = [t["dex_digitized_minus_text"] for t in topo.values()]
topo["_note"] = ("order-of-magnitude agreement only: g_perp,max applies to R_A = 5 m - 20 km; the digitized plateau at 20 km sits "
                 "%.2f to %.2f dex below the text-only alpha~ (per data set: see dex_digitized_minus_text); "
                 "the xi agreement is helped by the +0.15 dex padding and kappa_max" % (-max(_dx), -min(_dx)))

# ------------------------------------------------------------------ long-range curves (fig09a), via the existing script
import sigma_exclusion as SE
for k, v in SE.SRC.items():
    SRC["fig09a: " + k] = v
json.dump(CACHE, open(CACHE_F, "w"), indent=0)

pre_pass = (worst_EP["xi_max_EP"] is not None) and worst_EP["xi_max_EP"] < 0.13 and not uncovered_EP

# ------------------------------------------------------------------ sensitivity: charge model D (Coulomb scales too)
def pairD(fa, fb, bmu_a, bmu_b):
    return dict(delta_f=fa - fb, kappa=abs(bmu_a - bmu_b) / (2 * abs(fa - fb)))
modelD = dict(
    note="[DERIVED sensitivity, not the repo model] f_D = 1 - (bare-quark + electron mass)/m: nuclear Coulomb energy scales "
         "with Lambda_QCD as nuclear radii do. The repo (inventory/core/sigma.py) keeps nuclear Coulomb energy sigma-invariant.",
    pairs={"Be-Ti": pairD(ELEMS["Be"]["f_modelD"], ELEMS["Ti"]["f_modelD"], PUB_BMU_WAGNER["Be"], PUB_BMU_WAGNER["Ti"]),
           "Be-Al": pairD(ELEMS["Be"]["f_modelD"], ELEMS["Al"]["f_modelD"], PUB_BMU_WAGNER["Be"], PUB_BMU_WAGNER["Al"]),
           "Cu-Pb alloy": pairD(ELEMS["Cu"]["f_modelD"], alloy["f_modelD"], BMU_CU_SMITH, BMU_PB_ALLOY_SMITH)},
)
kD_cupb = modelD["pairs"]["Cu-Pb alloy"]["kappa"]
kD_ew08 = max(modelD["pairs"]["Be-Ti"]["kappa"], modelD["pairs"]["Be-Al"]["kappa"])
modelD["xi_max_EW99"] = {f"{l:g} m": alpha_B_ew99(l) * kD_cupb for l in (0.01, 0.0125, 0.014, 0.02, 0.05, 0.1, 1.0, 3.5e4)}
modelD["xi_max_EW08"] = {f"{l:g} m": alpha_env_fig5(l) * 10 ** CONSERVATIVE_DEX * kD_ew08 for l in (1.0, 1e3, 1e4, 3.5e4)}
modelD["EW99_largest_allowed_lambda_m"] = {k: largest_allowed(lambda l: xi_ew99(l, kD_cupb), xi, 0.01, 3.5e4) for k, xi in XIS.items()}
modelD["min_EP_exclusion_factor_xi_0p13_from_2cm"] = min(0.13 / (alpha_B_ew99(l) * kD_cupb) for l in np.logspace(math.log10(0.02), math.log10(3.5e4), 400))
modelD["EP_exclusion_factor_xi_0p13_at_1m"] = 0.13 / min(alpha_B_ew99(1.0) * kD_cupb, alpha_env_fig5(1.0) * 10 ** CONSERVATIVE_DEX * kD_ew08)
# fraction of Delta f carried by the sigma-invariant nuclear Coulomb energy (the only difference between the two models)
coulomb_share = {
    "Cu-Pb alloy": 1 - modelD["pairs"]["Cu-Pb alloy"]["delta_f"] / DF_CUPB,
    "Be-Ti": 1 - modelD["pairs"]["Be-Ti"]["delta_f"] / PAIRS["Be-Ti"]["delta_f"],
    "Be-Al": 1 - modelD["pairs"]["Be-Al"]["delta_f"] / PAIRS["Be-Al"]["delta_f"],
}

# ------------------------------------------------------------------ engine-defect audit (read-only): isotopes.json
ISO_F = os.path.join(REPO, "sigma_ground", "inventory", "data", "isotopes.json")
ISO = json.load(open(ISO_F, encoding="utf-8"))
_lines = open(ISO_F, encoding="utf-8").read().splitlines()
_ab_lines = [i + 1 for i, s in enumerate(_lines) if '"abundance_percent"' in s]
assert len(_ab_lines) == len(ISO)
stable = [(r, ln) for r, ln in zip(ISO, _ab_lines) if r.get("is_stable")]
zero = [(r, ln) for r, ln in stable if not r.get("abundance_percent")]
nist_ab = {(AW[s]["z"], i["a"]): i["abundance"] for s in AW if s != "_meta" and isinstance(AW[s], dict) and "isotopes" in AW[s]
           for i in AW[s]["isotopes"]}
def spin(r):
    return f'{r.get("nuclear_spin")} {r.get("nuclear_parity")}'
iso_audit = dict(
    file="sigma_ground/inventory/data/isotopes.json",
    n_stable=len(stable), n_stable_abundance_zero=len(zero),
    n_zero_rows_with_nonzero_NIST_abundance=sum(1 for r, _ in zero if nist_ab.get((r["Z"], r["A"]), 0) > 0),
    spin_parity_of_zero_rows={s: sum(1 for r, _ in zero if spin(r) == s) for s in sorted({spin(r) for r, _ in zero})},
    magnetic_moment_null_in_zero_rows=sum(1 for r, _ in zero if r.get("nuclear_magnetic_moment_mu_n") is None),
    spin_parity_of_nonzero_stable_rows_top5=dict(sorted({s: sum(1 for r, _ in stable if r.get("abundance_percent") and spin(r) == s)
                                                         for s in {spin(r) for r, _ in stable}}.items(), key=lambda kv: -kv[1])[:5]),
    examples={f'{r["symbol"]}-{r["A"]}': dict(abundance_percent_line=ln, abundance_percent=r["abundance_percent"],
                                             NIST_abundance=nist_ab.get((r["Z"], r["A"])), spin_parity=spin(r),
                                             magnetic_moment=r.get("nuclear_magnetic_moment_mu_n"))
              for r, ln in zero if (r["symbol"], r["A"]) in {("Be", 9), ("F", 19), ("Nb", 93), ("Au", 197), ("Pt", 195)}},
    zero_rows=[f'{r["symbol"]}-{r["A"]}' for r, _ in zero],
)
def _dist(rs):
    out = {}
    for r in rs:
        out[spin(r)] = out.get(spin(r), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
iso_audit["spin_parity_odd_A_zero_rows"] = _dist([r for r, _ in zero if r["A"] % 2 == 1])
iso_audit["spin_parity_even_even_zero_rows"] = _dist([r for r, _ in zero if r["Z"] % 2 == 0 and (r["A"] - r["Z"]) % 2 == 0])
iso_audit["spin_parity_odd_A_nonzero_rows"] = _dist([r for r, _ in stable if r.get("abundance_percent") and r["A"] % 2 == 1])

# ------------------------------------------------------------------ coverage of the window and the band verdict
coverage = [
    dict(range_m=[0.01, 50.0], experiment="EW99 Smith 2000 (Cu-Pb alloy, 238U attractor)", basis="[LIT] Table X tabulated, eq. 22"),
    dict(range_m=[50.0, 3.5e4], experiment="EW99 Smith 2000", basis="[DERIVED] I_A saturated (Table X flat to 0.1 % from 5 m to 50 m) "
         "with the Earth term I_E from Table IX; Fig. 21 caption: constraints valid for 1.0 m < lambda < 1e6 m"),
    dict(range_m=[0.5, 1.0e4], experiment="EW08 Wagner thesis (Be-Ti, Be-Al; Earth + topography)", basis="[LIT digitized] Adelberger 2009 Fig. 5, solid curve"),
    dict(range_m=[1.0e4, 3.5e4], experiment="EW08", basis="[LIT digitized] PRELIMINARY: Adelberger 2009 text, 'Preliminary calculations "
         "indicate that the attractor strength in the intermediate region is at least equal to that for lambda = 10 km'"),
    dict(range_m=[1e-6, 3.5e4], experiment="ISL (inverse-square-law) tests",
         basis="[LIT digitized] Adelberger 2009 Figs. 9 + 10; laboratory to ~1 m, geophysical (earlier review) from ~1 m to 35 km"),
]
band = {}
r1m, r1km, r10km, r35km = (next(r for r in rows if r["lam"] >= L * (1 - 1e-12)) for L in (1.0, 1e3, 1e4, 3.5e4))
for k, xi in XIS.items():
    lc = sliver[k]["published_alloy_sign_agnostic_inv"]
    band[k] = dict(
        EP_sign_agnostic_allowed_sliver_m=None if lc is None else [0.01, lc],
        EP_sign_agnostic_sliver_width_mm=None if lc is None else (lc - 0.01) * 1e3,
        EP_repo_sign_allowed_sliver_m=sliver[k]["repo sign (repulsive), edge of the central 95% interval"],
        EP_min_exclusion_factor_in_window_at_1cm=xi / worst_EP["xi_max_EP"],
        EP_exclusion_factor_at_2cm=xi / xi_ew99(0.02),
        EP_exclusion_factor_at_1m=xi / r1m["xi_max_EP"],
        EP_exclusion_factor_at_1km_EW08_digitized_solid=xi / r1km["xi_max_EP"],
        EP_exclusion_factor_10_to_35km_EW99_firm=[xi / r10km["xi_EW99"], xi / r35km["xi_EW99"]],
        EP_exclusion_factor_10_to_35km_EW08_preliminary=[xi / r10km["xi_EW08"], xi / r35km["xi_EW08"]],
        ISL_min_exclusion_factor_in_window=xi / worst_isl["xi_ISL"],
        ISL_weakest_point_lambda_m=worst_isl["lam"],
        combined_EP_ISL_min_exclusion_factor=xi / worst_all["xi_max_all"],
        ISL_largest_allowed_lambda_m=lam_allowed[k]["ISL_padded_0p15dex"],
    )
corrected_pass = all(r["xi_max_all"] is not None and r["xi_max_all"] < 0.13 for r in win)
result = dict(
    item="T5 torsion-balance window (v2 repair)",
    xi_repo=XI,
    prereg=open(os.path.join(OUT, "PREREG.txt"), encoding="utf-8").read(),
    sources=SRC,
    elements={s: {k: v for k, v in e.items() if k != "isotopes"} for s, e in ELEMS.items()},
    validation_B_mu_repo_vs_published=validation_bmu,
    validation_smith_tableXI=validation_tableXI,
    pb_alloy_published=alloy,
    pairs=PAIRS,
    kappa=dict(EW08_max=kappa_ew08, CuPb_published_alloy=KAPPA_CUPB, CuPb_pure_Pb_sensitivity=KAPPA_CUPB_PUREPB),
    f_ISL_used=F_ISL,
    published_inputs=dict(EW99_delta_a_95_cm_s2=DA95, EW99_eq22=[DA_MEAN, DA_STAT, DA_SYS], EW99_g_prime_cm_s2=GPRIME,
                          EW99_tableX_S_B=TABX, EW99_tableIX_I_E=TABIX, smith_tableXI=TXI, smith_alloy_mass_fractions=ALLOY_W,
                          topo_g_perp_max_m_s2=G_PERP_TOPO, topo_data=TOPO_DATA, ISL_anchor=[ISL_ANCHOR_LAM, ISL_ANCHOR_ALPHA]),
    digitizer_bias=dig_bias,
    sign_analysis=sign_analysis,
    sliver_edges_m=sliver,
    window_edges_and_points=edge,
    largest_allowed_lambda=lam_allowed,
    topography_text_only_crosscheck=topo,
    window_worst_case_EP=dict(lam_m=worst_EP["lam"], xi_max=worst_EP["xi_max_EP"]),
    window_worst_case_ISL=dict(lam_m=worst_isl["lam"], xi_max=worst_isl["xi_ISL"], alpha_nominal=worst_isl["alpha_ISL"]),
    window_worst_case_all=dict(lam_m=worst_all["lam"], xi_max=worst_all["xi_max_all"]),
    window_uncovered_by_EP_m=uncovered_EP,
    prereg_verdict_literal="PASS" if pre_pass else "FAIL (xi_max >= 0.13 somewhere in the window)",
    corrected_criterion_verdict=("PASS" if corrected_pass else "FAIL") + " (post hoc: EP or ISL, whichever is stronger, xi_max < 0.13 everywhere in [1 cm, 35 km])",
    band_verdicts=band,
    coverage=coverage,
    charge_model_D_sensitivity=modelD,
    coulomb_share_of_delta_f=coulomb_share,
    engine_defect_isotopes_audit=iso_audit,
    curves=[{k: r[k] for k in ("lam", "alpha_B_EW99", "xi_EW99", "xi_EW99_repo_sign", "alpha_B_EW08", "xi_EW08", "alpha_ISL",
                               "xi_ISL", "xi_max_EP", "xi_max_all")} for r in rows[::8]],
)
json.dump(result, open(os.path.join(OUT, "t5_results_v2.json"), "w"), indent=1, default=float)

if __name__ == "__main__":
    print(json.dumps({k: result[k] for k in ("validation_smith_tableXI", "pb_alloy_published", "pairs", "kappa", "f_ISL_used",
                                             "digitizer_bias", "sign_analysis", "sliver_edges_m", "window_edges_and_points",
                                             "largest_allowed_lambda", "topography_text_only_crosscheck", "window_worst_case_EP",
                                             "window_worst_case_ISL", "window_worst_case_all", "prereg_verdict_literal",
                                             "corrected_criterion_verdict", "band_verdicts", "charge_model_D_sensitivity",
                                             "coulomb_share_of_delta_f")}, indent=1, default=float))
    print("isotopes audit:", json.dumps({k: v for k, v in iso_audit.items() if k != "zero_rows"}, indent=1))
    print("uncovered by EP:", len(uncovered_EP))

    # ================================================================== figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.unicode_minus": True, "font.family": "DejaVu Sans"})
    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
    C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"
    lam_all = np.array([r["lam"] for r in rows])
    def col(key):
        return np.array([np.nan if r[key] is None else r[key] for r in rows], float)
    xi99, xi08, xiisl, xi08f = col("xi_EW99"), col("xi_EW08"), col("xi_ISL"), col("xi_EW08_firm")
    xi99_sign = col("xi_EW99_repo_sign")
    xi08_pre = np.where(lam_all >= 1e4, xi08, np.nan)
    LL = np.logspace(-16, 15, 700)
    ep = np.array([SE.xi_max_ep(l) for l in LL]); clk = np.array([SE.xi_max_clock(l) for l in LL])
    wd = np.array([min(SE.xi_max_wd(l, p, v, e) for _, p, v, e in SE.WD) for l in LL])
    TOP = 10.0
    b13, b17 = band["0.13"], band["0.17"]
    e1cm = edge["0.01 m"]
    sa = sign_analysis["xi_max_at_1cm"]
    K_PUB, K_REP, K_ONE, K_ATT = list(DA_BOUNDS.keys())
    fig = plt.figure(figsize=(19, 7.4), dpi=160)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.72])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    fig.patch.set_facecolor(SURFACE)
    # ---------------- (a) fig09a extended
    ax = axes[0]; ax.set_facecolor(SURFACE)
    ax.axvspan(0.01, 3.5e4, color=GRID, alpha=0.55, lw=0)
    ax.text(0.03, 1.5e-9, "the T5 window\n1 cm – 35 km", fontsize=8.5, color=INK2, va="bottom")
    for cv, ls, lab in ((ep, "solid", "MICROSCOPE free fall (fig09a)"), (clk, (0, (6, 3)), "Yb⁺/Cs clocks (fig09a)"),
                        (wd, (0, (2, 2)), "white-dwarf H₂ (fig09a)")):
        ax.plot(LL, np.clip(cv, 1e-20, TOP), color=INK2, lw=1.4, linestyle=ls, label=lab)
    for y, c, lab in ((xi99, C1, "torsion EP, Eöt-Wash Cu–Pb alloy / ²³⁸U (Smith 2000, tabulated)"),
                      (xi08, C2, "torsion EP, Eöt-Wash Be–Ti, Be–Al (EW08, digitized)"),
                      (xiisl, C3, "inverse-square-law tests (Adelberger 2009, digitized)")):
        yy = np.clip(y, 1e-20, TOP)
        ax.plot(lam_all, yy, color=c, lw=2.3, label=lab)
        ax.fill_between(lam_all, yy, TOP, where=~np.isnan(y), color=c, alpha=0.08, lw=0)
    ax.axhspan(0.13, 0.17, color=C4, alpha=0.30, lw=0, label="Captain's ξ = 0.13 – 0.17")
    ax.plot([SE.lam_qcd], [XI], marker="o", ms=9, color=C4, mec=INK, zorder=6)
    ax.annotate("repo paper: λ = 0.91 fm\n(allowed, invisible)", xy=(SE.lam_qcd, XI), xytext=(3e-14, 1e-5), fontsize=8.5,
                color=INK, arrowprops=dict(arrowstyle="->", color=INK2))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-16, 1e15); ax.set_ylim(1e-16, TOP)
    ax.set_xlabel("range of σ, λ = ħ / m_σ c   (m)")
    ax.set_ylabel("largest allowed coupling ξ  (95 % CL; above a curve = excluded)")
    ax.set_title("(a) Fig. 9a extended: laboratory and geophysical bounds now fill 1 µm – 35 km", fontsize=10, color=INK)
    for x, lab in ((1e-15, "nucleus"), (1e-2, "1 cm"), (SE.R_E, "Earth radius"), (SE.AU, "1 AU")):
        ax.axvline(x, color=GRID, lw=1); ax.text(x * 1.3, 2e-16, lab, fontsize=8, color=INK2, rotation=90, va="bottom")
    ax.grid(color=GRID, lw=0.6, which="major")
    ax.legend(loc="lower left", bbox_to_anchor=(0.035, 0.10), frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=7.6)
    # ---------------- (b) the window, zoomed
    ax = axes[1]; ax.set_facecolor(SURFACE)
    ax.axvspan(0.01, 3.5e4, color=GRID, alpha=0.55, lw=0)
    m = (LL >= 1e-5) & (LL <= 2e5)
    ax.plot(LL[m], np.clip(ep[m], 1e-20, 1e3), color=INK2, lw=1.4, label="MICROSCOPE (fig09a)")
    ax.plot(lam_all, xi99, color=C1, lw=2.3, label="torsion EP, Cu–Pb alloy / ²³⁸U (Smith 2000 Table X), published |Δa| bound")
    ax.plot(lam_all, xi99_sign, color=C1, lw=1.3, ls=(0, (4, 2)), label="same, with σ's own (repulsive) sign of Δa")
    ax.plot(lam_all, xi08f, color=C2, lw=2.3, label="torsion EP, Be–Ti / Be–Al (EW08, digitized, +0.15 dex)")
    ax.plot(lam_all, xi08_pre, color=C2, lw=2.0, ls=(0, (2, 1.5)), label="EW08 beyond 10 km: 'preliminary' plateau")
    ax.plot(lam_all, xiisl, color=C3, lw=2.3, label="inverse-square law (digitized, +0.15 dex); post hoc, a lookup")
    for i, (name, t) in enumerate((n, v) for n, v in topo.items() if not n.startswith("_")):
        ax.plot([2.0e4 / (1.25 ** i)], [t["xi_max_text_only"]], marker=("s", "D", "^")[i], ms=6.5, color=C2, mec=INK, ls="none",
                label="text-only order-of-magnitude check, Δa ÷ (2Δf · topography g⊥) at ≤ 20 km" if i == 0 else None)
    ax.axhspan(0.13, 0.17, color=C4, alpha=0.30, lw=0, label="Captain's ξ = 0.13 – 0.17")
    ax.text(12.0, 3e-5, "ISL bounds: ← laboratory | geophysical →", fontsize=8, color=C3, ha="center")
    ax.annotate("inverse-square-law tests: ξ = 0.13 – 0.17\nallowed only for λ ≲ %.0f – %.0f µm\n(2009 compilation; Lee 2020 is tighter)"
                % (b17["ISL_largest_allowed_lambda_m"] * 1e6, b13["ISL_largest_allowed_lambda_m"] * 1e6),
                xy=(b13["ISL_largest_allowed_lambda_m"], 0.13), xytext=(1.3e-5, 2e-6), fontsize=8.3, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK2))
    ax.text(0.012, 1.3e-9, "ξ = 0.1582 excluded by EP: ×%.2f at 1 cm, ×%.0f at 2 cm,\n×%.1e at 1 m, ×%.1e at 1 km (EW08),\n"
            "×%.1e at 35 km (firm EW99).\nISL alone ≥ ×%.0f (weakest near %.0f m); combined ≥ ×%.0f"
            % (band["0.1582"]["EP_min_exclusion_factor_in_window_at_1cm"], band["0.1582"]["EP_exclusion_factor_at_2cm"],
               band["0.1582"]["EP_exclusion_factor_at_1m"], band["0.1582"]["EP_exclusion_factor_at_1km_EW08_digitized_solid"],
               band["0.1582"]["EP_exclusion_factor_10_to_35km_EW99_firm"][1],
               math.floor(band["0.1582"]["ISL_min_exclusion_factor_in_window"]), band["0.1582"]["ISL_weakest_point_lambda_m"],
               math.floor(band["0.1582"]["combined_EP_ISL_min_exclusion_factor"])),
            fontsize=8.0, color=INK, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-5, 2e5); ax.set_ylim(1e-9, 30)
    ax.set_xlabel("range of σ, λ   (m)")
    ax.set_ylabel("largest allowed coupling ξ  (95 % CL)")
    ax.set_title("(b) The window: EP data exclude ξ = 0.13 – 0.17 at every λ above %.3f cm"
                 % (b13["EP_sign_agnostic_allowed_sliver_m"][1] * 100), fontsize=10, color=INK)
    ax.grid(color=GRID, lw=0.6, which="major"); ax.legend(loc="upper right", frameon=False, fontsize=7.2)
    # ---------------- (c) the 1 cm edge, linear lambda
    ax = axes[2]; ax.set_facecolor(SURFACE)
    lz = np.linspace(0.0100, 0.0125, 600)
    for da_key, c, ls, lab in ((K_PUB, C1, "solid", "published sign-agnostic |Δa| < 5.7×10⁻¹³ (pre-registered)"),
                               (K_REP, C2, "solid", "σ's repulsive sign: central 95 %% edge (ξ_max = %.3f)" % sa[K_REP]),
                               (K_ONE, C2, (0, (4, 2)), "σ's repulsive sign: one-sided 95 %% (ξ_max = %.3f)" % sa[K_ONE]),
                               (K_ATT, C3, "solid", "if σ were attractive (ξ_max = %.3f)" % sa[K_ATT])):
        ax.plot(lz * 100, [xi_ew99(l, da=DA_BOUNDS[da_key]) for l in lz], color=c, lw=2.0, ls=ls, label=lab)
    ax.plot(lz * 100, [xi_ew99(l, KAPPA_CUPB_PUREPB) for l in lz], color=INK2, lw=1.2, ls=(0, (1, 1.5)),
            label="sensitivity: pure natural Pb instead of the alloy")
    ax.axhspan(0.13, 0.17, color=C4, alpha=0.30, lw=0, label="Captain's ξ = 0.13 – 0.17")
    ls13 = b13["EP_sign_agnostic_allowed_sliver_m"][1]
    ax.annotate("ξ = 0.13 allowed for\nλ = 1.000 – %.4f cm\n(%.3f mm; %.4f cm with\nlog–log interpolation)"
                % (ls13 * 100, (ls13 - 0.01) * 1e3, sliver["0.13"]["published_alloy_sign_agnostic_loglog"] * 100),
                xy=(ls13 * 100, 0.13), xytext=(1.10, 0.085), fontsize=8.3, color=INK, arrowprops=dict(arrowstyle="->", color=INK2))
    ax.set_xlim(1.0, 1.25); ax.set_ylim(0.0, 0.30)
    ax.set_xlabel("range of σ, λ   (cm; Table X starts at 1.0 cm)")
    ax.set_ylabel("largest allowed coupling ξ  (95 % CL)")
    ax.set_title("(c) The 1 cm edge: the sliver exists only with the sign-agnostic bound", fontsize=10, color=INK)
    ax.grid(color=GRID, lw=0.6); ax.legend(loc="upper right", frameon=False, fontsize=7.4)
    fig.suptitle("T5 — Equivalence-principle torsion data exclude σ with ξ = 0.13 – 0.17 from 1 cm to 35 km except a %.2f mm sliver at "
                 "1.00 cm for ξ ≤ %.4f, which closes once σ's own repulsive sign is used;\ninverse-square-law bounds (added after "
                 "pre-registration; a lookup that could not fail) exclude the whole window by ≥ ×%.0f and put σ's range below "
                 "~0.1 mm (%.0f – %.0f µm, 2009 compilation)"
                 % ((ls13 - 0.01) * 1e3, e1cm["xi_max_EW99_published_alloy"], math.floor(b13["ISL_min_exclusion_factor_in_window"]),
                    b17["ISL_largest_allowed_lambda_m"] * 1e6, b13["ISL_largest_allowed_lambda_m"] * 1e6),
                 fontsize=11.2, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(OUT, "fig_T5_torsion_window_v2.png"), facecolor=SURFACE, bbox_inches="tight")
    print("figure written")
