"""T5 -- the torsion-balance window: sigma exclusion between lambda = 1 cm and 35 km.

Pre-registered in PREREG.txt (written before any bound was looked up or any number computed).

Model (misc/UNIFICATION_PROGRAM.md section 6; scripts/unification/sigma_exclusion.py):
  m_i(sigma) = m_i0 [1 + f_i sigma],  f_i = dln m_i/dsigma at 0  (repo inventory, inventory/core/sigma.py)
  static sigma(x) = (2 G xi / c^2) Int rho(x') e^{-|x-x'|/lambda}/|x-x'| d^3x'   (source = rho c^2, sigma charge per unit
  source mass s = 1)  =>  extra potential between body i and source mass M:  V = +2 xi f_i s G m_i M e^{-r/lambda}/r.
  Composition-dependent part (EP):   Delta a_12 = 2 xi (f_1 - f_2) s  x [G x Yukawa geometry]
  Composition-independent part (ISL): alpha_ISL = -2 xi f s   (repulsive)
Published Yukawa form (Adelberger et al. 2009 eq. 5/7; Smith et al. 2000 eq. 2/26):
  Delta a_12 / g = alpha~ Delta(q/mu)_12 (q/mu)_A x [same geometry];  q = B panel has (B/mu)_A = 1 for all attractors.
  => xi_max(lambda) = alpha~_B,max(lambda) |Delta(B/mu)_12| (B/mu)_A / (2 |Delta f_12| s)          (pair mapping kappa)

Data (all [LIT], each paper verified by web search 2026-10-10):
  EW99  Smith et al. 2000, PRD 61, 022001: Cu vs "Pb" alloy toward a 3-t 238U attractor.
        Eq. 22: Delta a = (1.0 +- 2.6 +- 0.9)e-13 cm/s^2, 95% |Delta a| < 5.7e-13 cm/s^2; g' = 9.2e-5 cm/s^2 (eq. 4).
        Table X: S(lambda) = Delta(q/mu) <(q/mu)_A I_A(lambda)> for q = N+Z, 1 cm .. 50 m.  Table IX: I_E(lambda).
        Table XI: B/mu Cu = 1.0011166, "Pb" alloy = 1.0001694.
  EW08  Wagner PhD thesis (Be-Ti and Be-Al, Earth + local topography), as plotted in Adelberger et al. 2009, PPNP 62, 102,
        Fig. 5 lower-left (q = B) -- DIGITIZED (digitize_figures.py; reading uncertainty +-0.05 dex, 0.1 dex used).
        Cross-check from text numbers only: Adelberger 2009 Table 4 (local topography g_perp,max = 7.2e-5 m/s^2) and
        Wagner et al. 2012, CQG 29, 184002, Table 3 (Delta a_N, Delta a_W for Be-Ti, Be-Al).
  ISL   Adelberger et al. 2009, Figs. 9 (lambda <= 1 cm) and 10 (lambda >= 1 cm) -- DIGITIZED.
PLATINUM 21: nuclear binding energies from Mentat (nuclear_binding_energy, measured nuclide masses); isotope masses and
abundances from the repo inventory (inventory/data/atomic_weights.json, NIST); f_i from inventory/core/sigma.py;
MICROSCOPE / clock / white-dwarf curves recomputed by importing scripts/unification/sigma_exclusion.py (Mentat leaves).
Run from the repo root:  python -X utf8 <this file>
"""
import json, math, os, sys
import numpy as np

REPO = r"D:/Aaron/development/sigma-ground"
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "scripts", "unification"))
sys.path.insert(0, REPO)
from mentat_client import mentat
from sigma_ground.inventory.core.sigma import XI, three_measures_atom, _PROTON_BARE_MEV, _NEUTRON_BARE_MEV, _MEV_TO_KG
from sigma_ground.inventory.core.constants import CONSTANTS

SRC = {}
CACHE_F = os.path.join(OUT, "mentat_cache.json")
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
    f = sum((x * m / mtot) * f_isotope(Z, a, m)[0] for a, m, x in isos)        # mass-weighted
    fD = sum((x * m / mtot) * f_isotope(Z, a, m)[2] for a, m, x in isos)
    B = sum(x * a for a, _, x in isos)
    return dict(symbol=sym, Z=Z, f=f, f_modelD=fD, B_mu=B / mtot, Z_mu=Z / mtot, N_mu=(B - Z) / mtot,
                isotopes=[dict(A=a, mass_u=m, x=x) for a, m, x in isos])

ELEMS = {s: element(s) for s in ("Be", "Al", "Ti", "Cu", "Pb", "Pt", "W", "Mo", "Ta", "U", "Sb", "Sn", "Ca")}

# published B/mu (validation of the inventory's charges against the papers' own tables)
PUB_BMU = {"Be": 0.99865, "Al": 1.00068, "Ti": 1.00108, "Cu": 1.00112, "Pt": 1.00018, "Pb": 1.00012}   # Adelberger 2009 Table 1
PUB_BMU_WAGNER = {"Be": 0.99865, "Ti": 1.00107, "Al": 1.00068, "Pt": 1.00015}                        # Wagner 2012 Table 1
BMU_CU_SMITH, BMU_PB_ALLOY_SMITH, BMU_U_SMITH = 1.0011166, 1.0001694, 0.9962151                       # Smith 2000 Table XI
validation_bmu = {s: dict(repo=ELEMS[s]["B_mu"], published=PUB_BMU[s], diff=ELEMS[s]["B_mu"] - PUB_BMU[s]) for s in PUB_BMU}
validation_bmu["U (Smith Table XI)"] = dict(repo=ELEMS["U"]["B_mu"], published=BMU_U_SMITH, diff=ELEMS["U"]["B_mu"] - BMU_U_SMITH)

# ------------------------------------------------------------------ the "Pb" test-body alloy of Smith et al. 2000
# Composition not stated; its published B/mu (1.0001694) exceeds pure natural Pb. Bracket it with the common Pb hardeners
# Sb, Sn, Ca (mass fraction w fixed so the alloy reproduces the published B/mu; B/mu mixes exactly by mass fraction).
alloy = {}
for X in ("Sb", "Sn", "Ca"):
    w = (BMU_PB_ALLOY_SMITH - ELEMS["Pb"]["B_mu"]) / (ELEMS[X]["B_mu"] - ELEMS["Pb"]["B_mu"])
    alloy[X] = dict(w=w, f=(1 - w) * ELEMS["Pb"]["f"] + w * ELEMS[X]["f"]) if 0 <= w <= 1 else dict(w=w, f=None)
f_pb_options = [ELEMS["Pb"]["f"]] + [a["f"] for a in alloy.values() if a["f"] is not None]

def pair(a, b, fa, fb, bmu_a, bmu_b):
    dBmu, df = bmu_a - bmu_b, fa - fb
    return dict(pair=f"{a}-{b}", delta_B_mu=dBmu, delta_f=df, kappa=abs(dBmu) / (2 * abs(df)),
                sign_relation="opposite" if dBmu * df < 0 else "same")

PAIRS = {
    "Be-Ti": pair("Be", "Ti", ELEMS["Be"]["f"], ELEMS["Ti"]["f"], PUB_BMU_WAGNER["Be"], PUB_BMU_WAGNER["Ti"]),
    "Be-Al": pair("Be", "Al", ELEMS["Be"]["f"], ELEMS["Al"]["f"], PUB_BMU_WAGNER["Be"], PUB_BMU_WAGNER["Al"]),
}
cupb = [pair("Cu", "Pb", ELEMS["Cu"]["f"], fpb, BMU_CU_SMITH, BMU_PB_ALLOY_SMITH) for fpb in f_pb_options]
PAIRS["Cu-Pb(pure Pb f)"] = cupb[0]
for X, c in zip([k for k, a in alloy.items() if a["f"] is not None], cupb[1:]):
    PAIRS[f"Cu-Pb({X}-hardened, matched B/mu)"] = c
kappa_cupb_max = max(c["kappa"] for c in cupb)          # conservative (largest xi_max)
kappa_cupb_min = min(c["kappa"] for c in cupb)
kappa_ew08 = max(PAIRS["Be-Ti"]["kappa"], PAIRS["Be-Al"]["kappa"])

# ------------------------------------------------------------------ EW99: Smith et al. 2000, tabulated
DA95 = 5.7e-13          # cm/s^2, 95% CL on |a_Cu - a_Pb|           [LIT Smith 2000 eq. 22 + text]
GPRIME = 9.2e-5         # cm/s^2, gravitational acceleration toward the attractor [LIT Smith 2000 eq. 4]
TABX = [(0.010, 1.23e-8), (0.014, 1.75e-7), (0.020, 1.53e-6), (0.028, 7.46e-6), (0.05, 5.45e-5), (0.07, 1.25e-4),
        (0.10, 2.47e-4), (0.20, 5.61e-4), (0.50, 8.44e-4), (1.0, 9.19e-4), (2.0, 9.41e-4), (5.0, 9.48e-4),
        (10.0, 9.48e-4), (20.0, 9.49e-4), (50.0, 9.49e-4)]                       # [LIT Smith 2000 Table X, q = N+Z = B]
TABIX = [(1e1, 3.94e-7), (1e2, 4.31e-6), (1e3, 4.69e-5), (1e4, 9.81e-4), (1e5, 1.36e-2), (1e6, 1.72e-1),
         (1e7, 8.77e-1), (1e8, 9.98e-1)]                                          # [LIT Smith 2000 Table IX, I_E]
DBMU_CUPB = BMU_CU_SMITH - BMU_PB_ALLOY_SMITH

def S_B(lam):
    """Table X, ln S interpolated linearly in 1/lambda (Yukawa ~ e^{-r/lambda}); saturated beyond 50 m (I_A -> 1)."""
    L = np.array([t[0] for t in TABX]); S = np.log([t[1] for t in TABX])
    if lam < L[0] * (1 - 1e-9):
        return None                                                  # no published sensitivity below 1 cm
    if lam >= L[-1]:
        return float(np.exp(S[-1]))
    return float(np.exp(np.interp(1 / lam, (1 / L)[::-1], S[::-1])))

def I_E(lam):
    L = np.log10([t[0] for t in TABIX]); I = np.log10([t[1] for t in TABIX])
    if lam < 10:
        return 0.0
    return float(10 ** np.interp(math.log10(lam), L, I))

def alpha_B_ew99(lam):
    s = S_B(lam)
    if s is None:
        return None
    seff = s - DBMU_CUPB * I_E(lam)            # Smith eq. 26: [I_A (q/mu)_A - I_E (q/mu)_E], (B/mu)_E = 1
    return DA95 / (GPRIME * seff)

def xi_ew99(lam, kappa):
    a = alpha_B_ew99(lam)
    return None if a is None else a * kappa

# ------------------------------------------------------------------ digitized curves
DIG = json.load(open(os.path.join(OUT, "digitized", "digitized_bounds.json")))
CONSERVATIVE_DEX = 0.15      # digitization reading uncertainty (max |dex| vs Table X was 0.11), applied in the weakening direction

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

# ISL: Fig. 9 (<= 1 cm) and Fig. 10 (>= 1 cm) merged into one curve; the 0.98-1.13 cm gap between the two images'
# digitized ranges is bridged by log-log interpolation (both figures read ~5e-4 there).
_xi = np.concatenate([x9[x9 <= -2.0], x10[x10 > -2.0]]); _yi = np.concatenate([y9[x9 <= -2.0], y10[x10 > -2.0]])
_o = np.argsort(_xi); xisl, yisl = _xi[_o], _yi[_o]
def alpha_isl(lam):
    return interp_log(xisl, yisl, lam)

F_ISL = min(ELEMS[s]["f"] for s in ("W", "Mo", "Ta", "Cu", "Pb", "Be", "Al", "Ti", "Pt", "U"))

# ------------------------------------------------------------------ evaluate across the window and beyond
lams = np.unique(np.concatenate([np.logspace(-6, math.log10(3.5e4), 1200), [0.01, 0.0101, 0.0102, 0.0105]]))
rows = []
for lam in lams:
    a99 = alpha_B_ew99(lam)
    aenv = alpha_env_fig5(lam)
    # EW08 = the Fig. 5 envelope where it lies below EW99 by > 0.15 dex (otherwise the envelope is EW99 itself)
    a08 = aenv if (aenv is not None and lam >= 0.5 and (a99 is None or aenv < a99 / 10 ** 0.15)) else None
    aisl = alpha_isl(lam)
    rows.append(dict(
        lam=lam,
        alpha_B_EW99=a99,
        xi_EW99=None if a99 is None else a99 * kappa_cupb_max,
        xi_EW99_min_kappa=None if a99 is None else a99 * kappa_cupb_min,
        alpha_B_fig5_envelope=aenv,
        alpha_B_EW08=a08,
        xi_EW08=None if a08 is None else a08 * 10 ** CONSERVATIVE_DEX * kappa_ew08,
        alpha_ISL=aisl,
        xi_ISL=None if aisl is None else aisl * 10 ** CONSERVATIVE_DEX / (2 * F_ISL),
    ))

def best(r, keys):
    v = [r[k] for k in keys if r[k] is not None]
    return min(v) if v else None

for r in rows:
    r["xi_max_EP"] = best(r, ("xi_EW99", "xi_EW08"))
    r["xi_max_all"] = best(r, ("xi_EW99", "xi_EW08", "xi_ISL"))

win = [r for r in rows if 0.01 * (1 - 1e-9) <= r["lam"] <= 3.5e4]
worst_EP = max(win, key=lambda r: r["xi_max_EP"] if r["xi_max_EP"] is not None else -1)
uncovered_EP = [r["lam"] for r in win if r["xi_max_EP"] is None]
worst_all = max(win, key=lambda r: r["xi_max_all"])

# exactly at the window edges
edge = {}
for lam in (0.010, 0.0105, 0.011, 0.012, 0.014, 0.020, 0.028, 0.05, 0.10, 1.0, 10.0, 50.0, 1e3, 1e4, 3.5e4):
    a99 = alpha_B_ew99(lam)
    edge[f"{lam:g} m"] = dict(alpha_B_EW99=a99, xi_max_EW99_conservative=a99 * kappa_cupb_max,
                              xi_max_EW99_other_alloys=[a99 * c["kappa"] for c in cupb],
                              exclusion_factor_xi_0p13=0.13 / (a99 * kappa_cupb_max),
                              exclusion_factor_xi_0p1582=XI / (a99 * kappa_cupb_max),
                              xi_max_ISL=next((r["xi_ISL"] for r in rows if abs(math.log10(r["lam"] / lam)) < 0.004), None))

# lambda at which the EW99 composition-dependent bound alone reaches each xi (solve on the 1/lambda interpolation)
def lam_cross(fun, xi, lo, hi):
    """Largest lambda in [lo, hi] (log grid) where fun(lam) >= xi (i.e. xi allowed); None if never."""
    g = np.logspace(math.log10(lo), math.log10(hi), 4000)
    ok = [l for l in g if fun(l) is not None and fun(l) >= xi]
    return max(ok) if ok else None

XIS = {"0.13": 0.13, "0.1571": 0.1571, "0.1582": XI, "0.17": 0.17}
lam_allowed = {}
for k, xi in XIS.items():
    lam_allowed[k] = dict(
        EW99_only_largest_allowed_lambda_m=lam_cross(lambda l: xi_ew99(l, kappa_cupb_max), xi, 0.01, 3.5e4),
        EW99_only_with_smallest_kappa=lam_cross(lambda l: xi_ew99(l, kappa_cupb_min), xi, 0.01, 3.5e4),
        ISL_largest_allowed_lambda_m=lam_cross(lambda l: (alpha_isl(l) * 10 ** CONSERVATIVE_DEX / (2 * F_ISL)) if alpha_isl(l) else None,
                                               xi, 1e-6, 3.5e4),
        ISL_largest_allowed_lambda_m_nominal_reading=lam_cross(lambda l: (alpha_isl(l) / (2 * F_ISL)) if alpha_isl(l) else None,
                                                               xi, 1e-6, 3.5e4),
    )

# ------------------------------------------------------------------ text-only cross-check at the topography plateau
G_PERP_TOPO = 7.2e-5   # m/s^2 [LIT Adelberger 2009 Table 4, local topography, R_A = 5 - 2e4 m]
WAGNER_T3 = {"Be-Ti": dict(N=(0.6, 3.1), W=(-2.5, 3.5)), "Be-Al": dict(N=(-1.2, 2.2), W=(0.2, 2.4))}   # 1e-15 m/s^2 [LIT]
topo = {}
for p, comp in WAGNER_T3.items():
    da95 = max(abs(m) + 1.96 * s for m, s in comp.values()) * 1e-15
    topo[p] = dict(delta_a_95_m_s2=da95, xi_max=da95 / (2 * abs(PAIRS[p]["delta_f"]) * G_PERP_TOPO),
                   alpha_B_max=da95 / (abs(PAIRS[p]["delta_B_mu"]) * G_PERP_TOPO))
digit_at = {f"{l:g} m": dict(alpha_B_fig5=alpha_env_fig5(l), xi_EW08=next((r["xi_EW08"] for r in rows if abs(math.log10(r["lam"] / l)) < 0.004), None))
            for l in (1.0, 10.0, 100.0, 1e3, 1e4, 2e4, 3.5e4)}
# digitization validation: the Fig. 5 envelope below 0.8 m is EW99, which Table X gives independently
dig_valid = {f"{l:g} m": dict(fig5_digitized=alpha_env_fig5(l), tableX_computed=alpha_B_ew99(l),
                              ratio=(alpha_env_fig5(l) / alpha_B_ew99(l)) if alpha_env_fig5(l) else None)
             for l in (0.05, 0.07, 0.1, 0.2, 0.5)}

# ------------------------------------------------------------------ long-range curves (fig09a), via the existing script
import sigma_exclusion as SE
for k, v in SE.SRC.items():
    SRC["fig09a: " + k] = v
json.dump(CACHE, open(CACHE_F, "w"), indent=0)

pre_pass = (worst_EP["xi_max_EP"] is not None) and worst_EP["xi_max_EP"] < 0.13 and not uncovered_EP

# ------------------------------------------------------------------ sensitivity: charge model D (Coulomb scales too)
def pairD(a, b, bmu_a, bmu_b, fa=None, fb=None):
    fa = ELEMS[a]["f_modelD"] if fa is None else fa
    fb = ELEMS[b]["f_modelD"] if fb is None else fb
    return dict(delta_f=fa - fb, kappa=abs(bmu_a - bmu_b) / (2 * abs(fa - fb)))
fD_pb_alloys = [ELEMS["Pb"]["f_modelD"]] + [(1 - alloy[X]["w"]) * ELEMS["Pb"]["f_modelD"] + alloy[X]["w"] * ELEMS[X]["f_modelD"]
                                            for X in alloy if alloy[X]["f"] is not None]
modelD = dict(
    note="[DERIVED sensitivity, not the repo model] f_D = 1 - (bare-quark + electron mass)/m: nuclear Coulomb energy scales "
         "with Lambda_QCD as nuclear radii do. The repo (inventory/core/sigma.py) keeps nuclear Coulomb energy sigma-invariant.",
    pairs={"Be-Ti": pairD("Be", "Ti", PUB_BMU_WAGNER["Be"], PUB_BMU_WAGNER["Ti"]),
           "Be-Al": pairD("Be", "Al", PUB_BMU_WAGNER["Be"], PUB_BMU_WAGNER["Al"]),
           "Cu-Pb": [pairD("Cu", "Pb", BMU_CU_SMITH, BMU_PB_ALLOY_SMITH, fb=fp) for fp in fD_pb_alloys]},
)
kD_cupb = max(c["kappa"] for c in modelD["pairs"]["Cu-Pb"])
kD_ew08 = max(modelD["pairs"]["Be-Ti"]["kappa"], modelD["pairs"]["Be-Al"]["kappa"])
modelD["xi_max_EW99"] = {f"{l:g} m": alpha_B_ew99(l) * kD_cupb for l in (0.01, 0.0125, 0.014, 0.02, 0.05, 0.1, 1.0, 3.5e4)}
modelD["xi_max_EW08"] = {f"{l:g} m": alpha_env_fig5(l) * 10 ** CONSERVATIVE_DEX * kD_ew08 for l in (1.0, 1e3, 1e4, 3.5e4)}
modelD["EW99_largest_allowed_lambda_m"] = {k: lam_cross(lambda l: xi_ew99(l, kD_cupb), xi, 0.01, 3.5e4) for k, xi in XIS.items()}

# ------------------------------------------------------------------ coverage of the window and the band verdict
coverage = [
    dict(range_m=[0.01, 50.0], experiment="EW99 Smith 2000 (Cu-Pb, 238U attractor)", basis="[LIT] Table X tabulated, eq. 22"),
    dict(range_m=[50.0, 3.5e4], experiment="EW99 Smith 2000", basis="[DERIVED] I_A saturated (Table X flat to 0.1 % from 5 m to 50 m; "
         "Yukawa factor 1 - (r/lambda)^2/2, r < 1 m) with the Earth term I_E from Table IX; paper states validity 1 m - 1e6 m"),
    dict(range_m=[0.5, 1.0e4], experiment="EW08 Wagner thesis (Be-Ti, Be-Al; Earth + topography)", basis="[LIT digitized] Adelberger 2009 Fig. 5, solid curve"),
    dict(range_m=[1.0e4, 3.5e4], experiment="EW08", basis="[LIT digitized] dashed 'preliminary' segment; text-only cross-check from Table 4 g_perp,max agrees"),
    dict(range_m=[1e-6, 3.5e4], experiment="ISL (inverse-square-law) tests", basis="[LIT digitized] Adelberger 2009 Figs. 9 + 10"),
]
band = {}
for k, xi in XIS.items():
    lc = lam_allowed[k]["EW99_only_largest_allowed_lambda_m"]
    band[k] = dict(
        EP_allowed_sliver_m=None if lc is None else [0.01, lc],
        EP_min_exclusion_factor_in_window=xi / worst_EP["xi_max_EP"],
        EP_exclusion_factor_at_1m=xi / next(r["xi_max_EP"] for r in rows if r["lam"] >= 1.0),
        EP_exclusion_factor_at_1km=xi / next(r["xi_max_EP"] for r in rows if r["lam"] >= 1e3),
        EP_exclusion_factor_at_35km=xi / rows[-1]["xi_max_EP"],
        ISL_min_exclusion_factor_in_window=xi / max(r["xi_ISL"] for r in win if r["xi_ISL"] is not None),
        ISL_largest_allowed_lambda_m=lam_allowed[k]["ISL_largest_allowed_lambda_m"],
    )
corrected_pass = all(r["xi_max_all"] is not None and r["xi_max_all"] < 0.13 for r in win)
result = dict(
    item="T5 torsion-balance window",
    xi_repo=XI,
    prereg=open(os.path.join(OUT, "PREREG.txt"), encoding="utf-8").read(),
    sources=SRC,
    elements={s: {k: v for k, v in e.items() if k != "isotopes"} for s, e in ELEMS.items()},
    validation_B_mu_repo_vs_published=validation_bmu,
    pb_alloy_brackets=alloy,
    pairs=PAIRS,
    kappa=dict(EW08_max=kappa_ew08, CuPb_max=kappa_cupb_max, CuPb_min=kappa_cupb_min),
    f_ISL_used=F_ISL,
    published_inputs=dict(EW99_delta_a_95_cm_s2=DA95, EW99_g_prime_cm_s2=GPRIME, EW99_tableX_S_B=TABX,
                          EW99_tableIX_I_E=TABIX, topo_g_perp_max_m_s2=G_PERP_TOPO, wagner2012_table3=WAGNER_T3,
                          fig5_plot_left_edge_m=10 ** DIG["fig5_B_panel"]["calibration"]["plot_left_edge_log10_lambda"]),
    window_edges_and_points=edge,
    largest_allowed_lambda=lam_allowed,
    topography_text_only_crosscheck=topo,
    EW08_digitized_points=digit_at,
    digitization_validation_vs_tableX=dig_valid,
    window_worst_case_EP=dict(lam_m=worst_EP["lam"], xi_max=worst_EP["xi_max_EP"]),
    window_worst_case_all=dict(lam_m=worst_all["lam"], xi_max=worst_all["xi_max_all"]),
    window_uncovered_by_EP_m=uncovered_EP,
    prereg_verdict_literal="PASS" if pre_pass else "FAIL (xi_max >= 0.13 somewhere in the window)",
    corrected_criterion_verdict=("PASS" if corrected_pass else "FAIL") + " (EP or ISL, whichever is stronger, xi_max < 0.13 everywhere in [1 cm, 35 km])",
    band_verdicts=band,
    coverage=coverage,
    charge_model_D_sensitivity=modelD,
    curves=[{k: r[k] for k in ("lam", "alpha_B_EW99", "xi_EW99", "alpha_B_EW08", "xi_EW08", "alpha_ISL", "xi_ISL", "xi_max_EP", "xi_max_all")}
            for r in rows[::6]],
)
json.dump(result, open(os.path.join(OUT, "t5_results.json"), "w"), indent=1, default=float)

if __name__ == "__main__":
    print(json.dumps({k: result[k] for k in ("elements", "validation_B_mu_repo_vs_published", "pb_alloy_brackets", "pairs", "kappa",
                                             "f_ISL_used", "window_edges_and_points", "largest_allowed_lambda",
                                             "topography_text_only_crosscheck", "EW08_digitized_points",
                                             "digitization_validation_vs_tableX", "window_worst_case_EP",
                                             "window_worst_case_all", "prereg_verdict_literal", "corrected_criterion_verdict",
                                             "band_verdicts", "charge_model_D_sensitivity")}, indent=1, default=float))
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
    xi99, xi08, xiisl = col("xi_EW99"), col("xi_EW08"), col("xi_ISL")
    xi99_lo = col("xi_EW99_min_kappa")
    LL = np.logspace(-16, 15, 700)
    ep = np.array([SE.xi_max_ep(l) for l in LL]); clk = np.array([SE.xi_max_clock(l) for l in LL])
    wd = np.array([min(SE.xi_max_wd(l, p, v, e) for _, p, v, e in SE.WD) for l in LL])
    TOP = 10.0
    b13, b17 = band["0.13"], band["0.17"]
    e1cm = edge["0.01 m"]
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.0), dpi=170, gridspec_kw=dict(width_ratios=[1.0, 1.0]))
    fig.patch.set_facecolor(SURFACE)
    # ---------------- (a) fig09a extended
    ax = axes[0]; ax.set_facecolor(SURFACE)
    ax.axvspan(0.01, 3.5e4, color=GRID, alpha=0.55, lw=0)
    ax.text(0.03, 1.5e-9, "the T5 window\n1 cm – 35 km", fontsize=8.5, color=INK2, va="bottom")
    for curve, ls, lab in ((ep, "solid", "MICROSCOPE free fall (fig09a)"), (clk, (0, (6, 3)), "Yb⁺/Cs clocks (fig09a)"),
                           (wd, (0, (2, 2)), "white-dwarf H₂ (fig09a)")):
        ax.plot(LL, np.clip(curve, 1e-20, TOP), color=INK2, lw=1.4, linestyle=ls, label=lab)
    for y, c, lab in ((xi99, C1, "torsion EP, Eöt-Wash Cu–Pb / ²³⁸U (Smith 2000, tabulated)"),
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
    ax.set_title("(a) Fig. 9a extended: laboratory bounds now fill 1 µm – 35 km", fontsize=10.5, color=INK)
    for x, lab in ((1e-15, "nucleus"), (1e-2, "1 cm"), (6.371e6, "Earth"), (1.496e11, "1 AU")):
        ax.axvline(x, color=GRID, lw=1); ax.text(x * 1.3, 2e-16, lab, fontsize=8, color=INK2, rotation=90, va="bottom")
    ax.grid(color=GRID, lw=0.6, which="major")
    ax.legend(loc="lower left", bbox_to_anchor=(0.035, 0.10), frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=7.8)
    # ---------------- (b) the window, zoomed
    ax = axes[1]; ax.set_facecolor(SURFACE)
    ax.axvspan(0.01, 3.5e4, color=GRID, alpha=0.55, lw=0)
    m = (LL >= 1e-5) & (LL <= 2e5)
    ax.plot(LL[m], np.clip(ep[m], 1e-20, 1e3), color=INK2, lw=1.4, label="MICROSCOPE (fig09a)")
    ax.plot(lam_all, xi99, color=C1, lw=2.3, label="torsion EP, Cu–Pb / ²³⁸U (Smith 2000 Table X)")
    ax.fill_between(lam_all, xi99_lo, xi99, color=C1, alpha=0.25, lw=0)
    ax.plot(lam_all, xi08, color=C2, lw=2.3, label="torsion EP, Be–Ti / Be–Al (EW08, digitized, +0.15 dex)")
    ax.plot(lam_all, xiisl, color=C3, lw=2.3, label="inverse-square law (digitized, +0.15 dex)")
    for p, mk in (("Be-Ti", "s"), ("Be-Al", "D")):
        ax.plot([2.6e4], [topo[p]["xi_max"]], marker=mk, ms=7, color=C2, mec=INK, ls="none",
                label="text-only check (■ Be–Ti, ◆ Be–Al): Δa ÷ (2Δf · topography g⊥)" if p == "Be-Ti" else None)
        if p == "Be-Al":
            ax.plot([2.6e4], [topo[p]["xi_max"]], marker=mk, ms=7, color=C2, mec=INK, ls="none")
    ax.axhspan(0.13, 0.17, color=C4, alpha=0.30, lw=0, label="Captain's ξ = 0.13 – 0.17")
    ax.annotate("λ = 1.00 cm: the EP bound alone is ξ < %.3f – %.3f\n(×%.2f on ξ = 0.13; ×%.2f on ξ = 0.1582)"
                % (min(e1cm["xi_max_EW99_other_alloys"]), max(e1cm["xi_max_EW99_other_alloys"]),
                   e1cm["exclusion_factor_xi_0p13"], e1cm["exclusion_factor_xi_0p1582"]),
                xy=(0.01, e1cm["xi_max_EW99_conservative"]), xytext=(3e-5, 1.5), fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK2))
    lisl = b13["ISL_largest_allowed_lambda_m"]
    ax.annotate("inverse-square-law tests: ξ = 0.13 – 0.17\nallowed only for λ ≲ %.0f – %.0f µm" % (b17["ISL_largest_allowed_lambda_m"] * 1e6, lisl * 1e6),
                xy=(lisl, 0.13), xytext=(2e-5, 3e-4), fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="->", color=INK2))
    ax.text(0.012, 1.3e-9, "window: EP excludes ξ = 0.1582 by ×%.1f at 1 cm,\n×%.0e at 1 m, ×%.0e at 1 km; ISL by ≥ ×%.0f"
            % (band["0.1582"]["EP_min_exclusion_factor_in_window"], band["0.1582"]["EP_exclusion_factor_at_1m"],
               band["0.1582"]["EP_exclusion_factor_at_1km"], band["0.1582"]["ISL_min_exclusion_factor_in_window"]),
            fontsize=8.5, color=INK, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-5, 2e5); ax.set_ylim(1e-9, 30)
    ax.set_xlabel("range of σ, λ   (m)")
    ax.set_ylabel("largest allowed coupling ξ  (95 % CL)")
    ax.set_title("(b) The window: every λ from 1 cm to 35 km is excluded for ξ = 0.13 – 0.17", fontsize=10.5, color=INK)
    ax.grid(color=GRID, lw=0.6, which="major"); ax.legend(loc="upper right", frameon=False, fontsize=7.8)
    fig.suptitle("T5 — Torsion balances close σ's 1 cm – 35 km window for ξ = 0.13 – 0.17: equivalence-principle tests exclude it "
                 "(only a %.2f mm sliver at 1.00 cm survives, and only for ξ ≤ %.2f),\nand the inverse-square-law tests σ must also pass "
                 "exclude it by ≥ ×80, so with ξ = 0.13 – 0.17 σ's range must be below ~0.1 mm (%.0f – %.0f µm)"
                 % ((b13["EP_allowed_sliver_m"][1] - 0.01) * 1e3 if b13["EP_allowed_sliver_m"] else 0.0,
                    max(e1cm["xi_max_EW99_other_alloys"]), b17["ISL_largest_allowed_lambda_m"] * 1e6, lisl * 1e6),
                 fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT, "fig_T5_torsion_window.png"), facecolor=SURFACE, bbox_inches="tight")
    print("figure written")
