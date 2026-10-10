"""What σ is NOT — the Captain's idea made mathematical, then cut against measurements.

The idea, in his words: "σ times the density of baryonic energy configures the speed of light and the scale of matter."
The mathematical form, using only what the repo already states:

  scale of matter   m_i(σ) = m_i⁰ [1 + f_i σ + O(σ²)],   f_i = ∂ln m_i/∂σ at σ = 0
                    (QCD part of every mass scales as e^σ; Higgs + EM parts are invariant — inventory/core/sigma.py)
  source            the trace of the stress-energy, i.e. the baryonic energy density ε_b = ρc² (minus 3P)
  field             (□ − 1/λ²) σ = −(8πGξ/c⁴)(ε_b − 3P)        λ = ħ/(m_σ c), the range of σ
  static solution   σ(x) = (2Gξ/c²) ∫ ρ(x') e^{−|x−x'|/λ} / |x−x'| d³x'
                      λ → ∞ : σ = −2ξΦ/c²                (σ tracks the potential — the field-equation reading, P6)
                      λ → 0 : σ = 8πGξλ²ρ/c²             (σ tracks the LOCAL baryonic energy density — the Captain's picture)

  observables, each measured:
   (1) atomic clocks:    (c²/μ) dμ/dΦ = −2ξ f_μ S(1+r/λ)  at r = 1 AU from the Sun           [Lange et al. 2021, PRL 126, 011102]
   (2) white-dwarf H₂:   Δμ/μ = 2ξ f_μ φ_WD S_surface                                         [Bagdonaite et al. 2014, PRL 113, 123002]
   (3) free fall (EP):   η(Ti,Pt) = 2ξ (f_TA6V − f_PtRh) S(1+r/λ)  at r = R_⊕ + 710 km       [Touboul et al. 2022, PRL 129, 121102]
   (S = Yukawa form factor of a uniform sphere: e^{−(r−R)/λ}·3[(x−1) + e^{−2x}(x+1)]/(2x³), x = R/λ; S → 1 as λ → ∞.)

  "configures the speed of light": a σ that rescales c, ħ and all masses together is a change of units (Duff 2002,
  hep-th/0208093); only dimensionless ratios are measurable. The repo's σ moves μ = m_p/m_e and leaves α fixed, so the
  μ observables above are the right ones. A σ that also moved α would face (c²/α)dα/dΦ = 14(11)×10⁻⁹ [Lange et al. 2021].

PLATINUM §21: G, c, ħ, eV, AU, solar and terrestrial masses/radii and nuclear binding energies come from Mentat
(tools/mentat_client.py); f_i come from the repo's own σ mass decomposition. [LIT] values are marked where used.
"""
import json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from mentat_client import mentat
from sigma_ground.inventory.core.sigma import XI, proton_mass_kg, three_measures_atom
from sigma_ground.inventory.core.constants import CONSTANTS
from sigma_ground.field.constants import LAMBDA_QCD_MEV, N0_FM3

SRC = {}
def leaf(tool, args, key=None, label=None):
    v, s = mentat(tool, args)
    if key is not None:
        v = v[key]
    SRC[label or f"{tool}{args}"] = s
    return v

G = leaf("lookup_constant", {"name": "G"}, label="G")
C = leaf("lookup_constant", {"name": "c"}, label="c")
HBAR = leaf("lookup_constant", {"name": "hbar"}, label="hbar")
EV = leaf("lookup_constant", {"name": "electron_volt"}, label="eV")
AU = leaf("lookup_constant", {"name": "au"}, label="au")
sun = leaf("solar_system_body", {"body_name": "sun"}, label="sun")
earth = leaf("solar_system_body", {"body_name": "earth"}, label="earth")
M_SUN, R_SUN = sun["mass_kg"], sun["radius_m"]
M_E, R_E = earth["mass_kg"], earth["radius_m"]

# ---------------- f_i from the repo's own σ mass decomposition ----------------
d = 1e-6
mp = lambda s: proton_mass_kg(s)
f_mu = (math.log(mp(d)) - math.log(mp(-d))) / (2 * d)            # m_e is σ-invariant, so ∂ln μ/∂σ = ∂ln m_p/∂σ

iso = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "sigma_ground", "inventory", "data", "isotopes.json"), encoding="utf-8"))
def isotope(Z, A):
    for r in iso:
        if r["Z"] == Z and r["A"] == A:
            return r
    raise KeyError((Z, A))

def f_atom(Z, A):
    r = isotope(Z, A)
    be = leaf("nuclear_binding_energy", {"protons": Z, "neutrons": A - Z, "measured_mass_u": r["atomic_mass_u"]},
              key="binding_energy_MeV", label=f"BE Z={Z} A={A}")
    m = lambda s: three_measures_atom(Z, A - Z, be, s)["atom_stable_mass_kg"]
    return (math.log(m(d)) - math.log(m(-d))) / (2 * d), be

# MICROSCOPE test masses: TA6V (Ti 90 %, Al 6 %, V 4 % by mass) and PtRh10 (Pt 90 %, Rh 10 %) [LIT: alloy compositions];
# one representative isotope per element (Ti-48, Al-27, V-51, Pt-195, Rh-103) [stated approximation].
ALLOY_TI = [(22, 48, 0.90), (13, 27, 0.06), (23, 51, 0.04)]
ALLOY_PT = [(78, 195, 0.90), (45, 103, 0.10)]
f_el = {}
for Z, A, _ in ALLOY_TI + ALLOY_PT:
    f_el[(Z, A)] = f_atom(Z, A)
f_ti = sum(w * f_el[(Z, A)][0] for Z, A, w in ALLOY_TI)
f_pt = sum(w * f_el[(Z, A)][0] for Z, A, w in ALLOY_PT)
df = abs(f_ti - f_pt)

# ---------------- Yukawa geometry ----------------
def S(R, r, lam):
    """e^{-r/λ}·F(R/λ) for a uniform sphere of radius R, evaluated at r ≥ R (stable form)."""
    x = R / lam
    if x < 1e-3:
        return math.exp(-r / lam) * (1 + x * x / 10)
    expo = -(r - R) / lam
    if expo < -700:
        return 0.0
    return math.exp(expo) * 3 * ((x - 1) + math.exp(-2 * x) * (x + 1)) / (2 * x ** 3)

def safe_div(a, b):
    return a / b if b > 0 else float("inf")

# ---------------- the three measurements (2σ two-sided, conservative) ----------------
K_MU_BOUND = abs(7e-8) + 2 * 45e-8                                     # Lange 2021: 7(45)e-8
WD = [("GD133", 1.2e-4, -2.7e-5, math.hypot(4.7e-5, 0.2e-5)),          # Bagdonaite 2014: φ in the photosphere
      ("G29-38", 1.9e-4, -5.8e-5, math.hypot(3.8e-5, 0.3e-5))]
M_WD = 0.6 * M_SUN                                                      # [LIT approx] typical DA white-dwarf mass
ETA_BOUND = abs(-1.5e-15) + 2 * math.hypot(2.3e-15, 1.5e-15)           # Touboul 2022
R_ORBIT = R_E + 710e3                                                   # MICROSCOPE altitude ~710 km [LIT]

def xi_max_clock(lam):
    return safe_div(K_MU_BOUND, 2 * f_mu * S(R_SUN, AU, lam) * (1 + AU / lam))
def xi_max_wd(lam, phi, val, err):
    R = G * M_WD / (phi * C * C)
    return safe_div(abs(val) + 2 * err, 2 * f_mu * phi * S(R, R, lam))
def xi_max_ep(lam):
    return safe_div(ETA_BOUND, 2 * df * S(R_E, R_ORBIT, lam) * (1 + R_ORBIT / lam))

# ---------------- the repo's own two readings ----------------
MEV = 1e6 * EV
lam_qcd = HBAR * C / (LAMBDA_QCD_MEV * MEV)                            # PAPER: m_σ ≈ Λ_QCD = 217 MeV  →  λ = ħc/Λ
rho_for = lambda sig, lam: sig * C * C / (8 * math.pi * G * XI * lam * lam)
rho_nuc = N0_FM3 * 1e45 * CONSTANTS.m_n                                 # nuclear saturation density
rho_planck = C ** 5 / (HBAR * G * G)

out = dict(
    sources=SRC, xi=XI, f_mu=f_mu,
    f_atoms={f"Z={Z},A={A}": dict(f=v[0], BE_MeV=v[1]) for (Z, A), v in f_el.items()},
    f_TA6V=f_ti, f_PtRh10=f_pt, delta_f=df,
    predictions_long_range=dict(
        k_mu_field_reading=-2 * XI * f_mu, k_mu_inventory_reading=-XI * f_mu, k_mu_bound_2sigma=K_MU_BOUND,
        clock_exclusion_factor_field=2 * XI * f_mu / K_MU_BOUND,
        eta_field_reading=2 * XI * df * S(R_E, R_ORBIT, 1e30) * (1 + R_ORBIT / 1e30), eta_bound_2sigma=ETA_BOUND,
        ep_exclusion_factor_field=2 * XI * df / ETA_BOUND,
        wd_predicted_dmu_field={n: 2 * XI * f_mu * p for n, p, _, _ in WD},
        wd_measured={n: (v, e) for n, _, v, e in WD},
        xi_max_long_range=dict(clock=xi_max_clock(1e30), ep=xi_max_ep(1e30), wd=min(xi_max_wd(1e30, p, v, e) for _, p, v, e in WD)),
    ),
    paper_point=dict(lambda_m=lam_qcd, lambda_fm=lam_qcd * 1e15,
                     sigma_at_nuclear_density=8 * math.pi * G * XI * lam_qcd ** 2 * rho_nuc / C ** 2,
                     rho_nuclear=rho_nuc, rho_planck=rho_planck,
                     rho_for_sigma={str(s): rho_for(s, lam_qcd) for s in (0.336, 0.405, 1.0, 1.844)},
                     xi_max_at_paper_lambda=dict(clock=xi_max_clock(lam_qcd), ep=xi_max_ep(lam_qcd))),
)

if __name__ == "__main__":
    import numpy as np, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
    lams = np.logspace(-16, 15, 600)
    clock = np.array([xi_max_clock(l) for l in lams]); ep = np.array([xi_max_ep(l) for l in lams])
    wd = np.array([min(xi_max_wd(l, p, v, e) for _, p, v, e in WD) for l in lams])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.6), dpi=180); fig.patch.set_facecolor(SURFACE)
    ax = axes[0]; ax.set_facecolor(SURFACE)
    top = 10.0
    for curve, col, ls, lab in ((ep, S1, "solid", "MICROSCOPE free fall, η(Ti,Pt) (Touboul 2022)"),
                                (clock, S2, (0, (6, 3)), "Yb⁺/Cs clocks, (c²/μ)dμ/dΦ (Lange 2021)"),
                                (wd, S3, (0, (2, 2)), "white-dwarf H₂, Δμ/μ (Bagdonaite 2014)")):
        cc = np.clip(curve, 1e-20, top)
        ax.plot(lams, cc, color=col, lw=2.2, linestyle=ls, label=lab)
        ax.fill_between(lams, cc, top, color=col, alpha=0.10)
    ax.axhline(XI, color=MUTED, lw=1, linestyle=":")
    ax.text(1.5e-16, XI * 1.35, "ξ = 0.1582 (the repo's coupling)", fontsize=8.5, color=INK2)
    ax.plot([lams[-1]], [XI], marker="X", ms=11, color=INK, clip_on=False, zorder=6)
    ax.annotate("field-equation reading\n(massless σ, λ → ∞): EXCLUDED\nclocks ×%.0e, free fall ×%.0e" % (2 * XI * f_mu / K_MU_BOUND, 2 * XI * df / ETA_BOUND),
                xy=(lams[-1], XI), xytext=(1e7, 3e-3), fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="->", color=INK2))
    ax.plot([lam_qcd], [XI], marker="o", ms=10, color=S4, mec=INK, zorder=6)
    ax.annotate("PAPER reading: m_σ ≈ Λ_QCD,\nλ = %.2f fm — allowed, and invisible" % (lam_qcd * 1e15),
                xy=(lam_qcd, XI), xytext=(3e-14, 1e-7), fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="->", color=INK2))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-16, 1e15); ax.set_ylim(1e-16, top)
    ax.set_xlabel("range of σ, λ = ħ / m_σ c   (m)"); ax.set_ylabel("coupling ξ  (σ sourced by (8πGξ/c⁴)·baryonic energy density)")
    ax.set_title("(a) Where σ can still live: shaded = excluded by measurement (2σ)", fontsize=10.5, color=INK)
    for x, lab in ((1e-15, "nucleus"), (R_E, "Earth"), (AU, "1 AU")):
        ax.axvline(x, color=GRID, lw=1); ax.text(x * 1.3, 2e-16, lab, fontsize=8, color=MUTED, rotation=90, va="bottom")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5); ax.grid(color=GRID, lw=0.6, which="major")
    ax = axes[1]; ax.set_facecolor(SURFACE)
    ladder = [("air", 1.2041), ("water", 999.97), ("iron", 7874.0), ("Sun core [LIT]", 1.5e5), ("white-dwarf centre [LIT]", 1e9),
              ("nuclear saturation", rho_nuc), ("neutron-star centre [LIT]", 1e18)]
    rr = np.logspace(0, 98, 400); sig = 8 * math.pi * G * XI * lam_qcd ** 2 * rr / C ** 2
    ax.plot(rr, sig, color=S4, lw=2.2, label="σ = 8πGξλ²ρ/c²  (PAPER range λ = %.2f fm)" % (lam_qcd * 1e15))
    for name, rho in ladder:
        s = 8 * math.pi * G * XI * lam_qcd ** 2 * rho / C ** 2
        ax.plot([rho], [s], marker="o", ms=6, color=S1, zorder=5); ax.text(rho * 2, s / 6, name, fontsize=7.6, color=INK2, rotation=-35)
    ax.axhspan(0.336, 0.405, color=S2, alpha=0.25); ax.text(1e60, 1e-5, "e^σ = 1.4–1.5\n(σ = 0.34–0.41)", fontsize=8.5, color=INK)
    ax.axhline(1.844, color=S2, lw=1.2, linestyle="--"); ax.text(1e60, 3e3, "σ_conv = −ln ξ = 1.844", fontsize=8.5, color=INK)
    ax.axvline(rho_planck, color=MUTED, lw=1, linestyle=":"); ax.text(rho_planck / 1e9, 1e-60, "Planck density", fontsize=8, color=MUTED, rotation=90)
    r14 = rho_for(0.336, lam_qcd)
    ax.axvline(r14, color=S2, lw=1, linestyle=":"); ax.text(r14 * 3, 1e-30, "σ reaches 0.34 at\nρ = %.1e kg/m³\n= %.0e × nuclear" % (r14, r14 / rho_nuc), fontsize=8.5, color=INK)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1, 1e98); ax.set_ylim(1e-70, 1e8)
    ax.set_xlabel("baryonic mass density ρ  (kg/m³)"); ax.set_ylabel("σ  (dimensionless)")
    ax.set_title("(b) In the one corner data allow, σ is O(1) only at ~10³⁹ × nuclear density", fontsize=10.5, color=INK)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5); ax.grid(color=GRID, lw=0.6)
    fig.suptitle("Fig. 9a — What σ is not: a long-range σ with ξ ≈ 0.16 is excluded by 10⁵–10¹¹; the short-range corner is open but silent",
                 fontsize=12, color=INK)
    fig.tight_layout(); os.makedirs("misc/unification", exist_ok=True)
    fig.savefig("misc/unification/fig09a_sigma_exclusion.png", bbox_inches="tight", facecolor=SURFACE)
    json.dump(out, open("misc/unification/sigma_exclusion.json", "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))
