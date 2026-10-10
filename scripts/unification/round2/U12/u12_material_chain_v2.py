"""U12 v2 (repair round) -- e^sigma through the material chain (sigma-ground / Mentat).

v2 keeps every v1 measurement, floor and criterion verbatim (u12_prereg.json is unchanged) and adds the
blocks the three skeptics required: I3 / T_m by-construction checks, the c_p gap decomposition, the extended
coupling audit (atom-level law, direct e^s couplings, E_coh law split), observability under the other E_coh law,
the coupling needed to reach any floor, the resonator row against named clocks, and the free-fall EP channel.
Writes u12_results_v2.json; the v1 files are left untouched.


Measures d ln X / d sigma at sigma = 0 by central differences for the iron property chain
(rho, K, G, E, v_L, v_T, v_D twice, Theta_D, T_melt, c_p) and dry-air sound speed, checks the
three chain identities, sweeps every material, records the sigma-coupling conventions found in
the code, and puts effect sizes next to metrology floors and the Lange+2021 clock bound.

Mentat flags, never fixes: the engine is only called, never patched. Consistent-chain values
are composed in-script from engine pieces and tagged [DERIVED].

Run from the repo root:
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -X utf8 <this file>
"""
import json, math, os, sys, time

REPO = r"D:/Aaron/development/sigma-ground"
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + "/tools")
from mentat_client import mentat  # noqa: E402

from sigma_ground.field.interface import (acoustics, thermal, mechanical, phase_transition,  # noqa: E402
                                          atmosphere, gas, statistical, atomic_spectra)
from sigma_ground.field.interface.surface import MATERIALS  # noqa: E402
from sigma_ground.field.interface.mechanical import MECHANICAL_DATA  # noqa: E402
from sigma_ground.field.interface.nbody import CelestialBody  # noqa: E402
from sigma_ground.field import relativity  # noqa: E402
from sigma_ground.field.constants import PROTON_QCD_FRACTION, SIGMA_HERE, N_AVOGADRO  # noqa: E402
from sigma_ground.field.scale import scale_ratio  # noqa: E402
from sigma_ground.inventory.core import sigma as inv_sigma  # noqa: E402

import numpy as np  # noqa: E402

T0 = time.time()
SRC = {}  # every Mentat leaf: name -> {value, source}


def M(key, tool, args):
    v, s = mentat(tool, args)
    SRC[key] = {"tool": tool, "args": args, "value": v, "source": s}
    return v


def Mnum(key, expr):
    v = M(key, "simplify_expr", {"expression": expr})
    return float(v)


# ---------------------------------------------------------------- Mentat leaves
G_ = float(M("G", "lookup_constant", {"name": "G"}))
c_ = float(M("c", "lookup_constant", {"name": "c"}))
AU = float(M("au", "lookup_constant", {"name": "au"}))
earth = M("earth", "solar_system_body", {"body_name": "earth"})
sun = M("sun", "solar_system_body", {"body_name": "sun"})
rs_sun = float(M("rs_sun", "schwarzschild_radius", {"mass_kg": sun["mass_kg"]}))
rs_earth = float(M("rs_earth", "schwarzschild_radius", {"mass_kg": earth["mass_kg"]}))
rho_fe_mentat = float(M("density_iron", "density", {"material": "iron"}))
tm_fe_mentat = float(M("melting_point_iron", "melting_point", {"material": "iron"}))
E_fe_mentat = float(M("youngs_modulus_iron", "youngs_modulus", {"material": "iron"}))
cs_air_mentat = float(M("cs_air_288K", "speed_of_sound_in_ideal_gas", {"temperature_k": 288.15}))
dlnrho_sym = M("dlnrho_symbolic", "differentiate_expr",
               {"expression": "log(1 - f + f*exp(s))", "variable": "s"})
dlnv_sym = M("dlnv_symbolic", "differentiate_expr",
             {"expression": "log(sqrt(K0*(1 + k*s)/(rho0*(1 - f + f*exp(s)))))", "variable": "s"})

dlnv_at0 = M("dlnv_symbolic_at_s0", "simplify_expr",
              {"expression": str(dlnv_sym).replace("exp(s)", "1").replace("k*s", "0")})
XI = inv_sigma.XI          # repo engine constant (Mentat): 0.1582
FQ = PROTON_QCD_FRACTION   # repo engine constant (Mentat): 0.99039
E_ORB = 0.01671022         # [LIT] NASA Earth fact sheet, J2000 mean elements (verified by web search)
a_earth = earth["semimajor_axis_au"] * AU
g_earth = earth["surface_g_ms2"]
R_earth = earth["radius_m"]

# sigma increments (field-equation normalisation sigma = xi r_s / r = -2 xi Phi/c^2)
dsig_km_field = Mnum("dsigma_per_km_field", f"2*{XI}*{g_earth}*1000/({c_}**2)")
dPhi_annual = Mnum("dPhi_over_c2_annual_amplitude", f"{E_ORB}*{rs_sun}/(2*{a_earth})")
dsig_ann_field = Mnum("dsigma_annual_field", f"2*{XI}*{dPhi_annual}")
sig_earth_self_field = Mnum("sigma_earth_self_field", f"{XI}*{rs_earth}/{R_earth}")
sig_sun_at_1au_field = Mnum("sigma_sun_at_earth_field", f"{XI}*{rs_sun}/{a_earth}")
# inventory normalisation (engine function sigma_from_potential = xi GM/(r c^2)) -- half
sig_earth_self_inv = inv_sigma.sigma_from_potential(R_earth, earth["mass_kg"])
sig_sun_at_1au_inv = inv_sigma.sigma_from_potential(a_earth, sun["mass_kg"])
NORM = {
    "field_eq": {"factor": 1.0, "dsig_km": dsig_km_field, "dsig_annual": dsig_ann_field,
                 "sigma_earth_self": sig_earth_self_field, "sigma_sun_at_1au": sig_sun_at_1au_field},
    "inventory": {"factor": 0.5, "dsig_km": 0.5 * dsig_km_field, "dsig_annual": 0.5 * dsig_ann_field,
                  "sigma_earth_self": sig_earth_self_inv, "sigma_sun_at_1au": sig_sun_at_1au_inv},
}

# ---------------------------------------------------------------- engine functions (iron, air)
MAT = "iron"
T_CP = 300.0
mass_ratio = lambda s: (1.0 - FQ) + FQ * scale_ratio(s)  # noqa: E731  (the code's own QCD-fraction law)

F = {
    "rho":      lambda s: acoustics.density_at_sigma(MAT, s),
    "K":        lambda s: mechanical.bulk_modulus(MAT, s),
    "G":        lambda s: mechanical.shear_modulus(MAT, s),
    "E":        lambda s: mechanical.youngs_modulus(MAT, s),
    "vL":       lambda s: acoustics.longitudinal_wave_speed(MAT, s),
    "vT":       lambda s: acoustics.transverse_wave_speed(MAT, s),
    "vD_acou":  lambda s: acoustics.debye_velocity(MAT, s),
    "vD_therm": lambda s: thermal.sound_velocity(MAT, s),
    "ThD":      lambda s: thermal.debye_temperature(MAT, s),
    "Tm":       lambda s: phase_transition.lindemann_melting_estimate(MAT, s),
    "cp":       lambda s: thermal.specific_heat_j_kg_K(MAT, T_CP, s),
    "Cv_vol":   lambda s: thermal.heat_capacity_volumetric(MAT, T_CP, s),
    "cs_air":   lambda s: atmosphere.speed_of_sound(288.15, s),
    "M_air":    lambda s: atmosphere.mean_molar_mass_kg_mol(s),
    "gamma_air": lambda s: atmosphere.air_gamma(288.15, s),
}

# consistent chain [DERIVED in-script from engine pieces; the engine is not changed]
x_Ar = atmosphere.DRY_AIR_COMPOSITION["Ar"]
m_Ar = atmosphere._AR_MASS_KG
n_fe = mechanical._number_density(MAT)


def ThD_cons(s):   # Theta_D with the Debye velocity that does carry rho(sigma)
    return F["ThD"](s) * F["vD_acou"](s) / F["vD_therm"](s)


def Tm_cons(s):    # Lindemann T_m = C M omega_D^2/(k_B n^(2/3)) with M(sigma) and consistent Theta_D
    return F["Tm"](s) * mass_ratio(s) * (ThD_cons(s) / F["ThD"](s)) ** 2


def cp_cons(s):    # Debye-Pade C_v with consistent Theta_D, per unit of rho(sigma)
    x_t, x_c = F["ThD"](s) / (4 * T_CP), ThD_cons(s) / (4 * T_CP)
    return F["cp"](s) * (1 + x_t ** 2) / (1 + x_c ** 2) * (MATERIALS[MAT]["density_kg_m3"] / F["rho"](s))


def Mair_cons(s):  # argon nucleus coupled like every other nucleus
    return F["M_air"](s) + x_Ar * m_Ar * N_AVOGADRO * FQ * (scale_ratio(s) - 1.0)


def cs_air_cons(s):
    return F["cs_air"](s) * math.sqrt(F["M_air"](s) / Mair_cons(s))


FC = {"ThD_cons": ThD_cons, "Tm_cons": Tm_cons, "cp_cons": cp_cons, "cs_air_cons": cs_air_cons,
      "M_air_cons": Mair_cons}


def dln(f, h):
    return (f(h) - f(-h)) / (2.0 * h * f(0.0))


HS = [1e-3, 1e-4, 1e-5]
H = 1e-4
D, DCONV = {}, {}
for k, f in {**F, **FC}.items():
    vals = [dln(f, h) for h in HS]
    D[k] = vals[1]
    DCONV[k] = {"h=1e-3": vals[0], "h=1e-4": vals[1], "h=1e-5": vals[2],
                "spread_rel": abs(vals[0] - vals[2]) / max(abs(vals[1]), 1e-300)}
LEVEL = {k: f(0.0) for k, f in {**F, **FC}.items()}
TM_SIGMA1 = phase_transition.lindemann_melting_estimate(MAT, 1.0) / phase_transition.lindemann_melting_estimate(MAT, 0.0) - 1.0
LEVEL_SIGMA_HERE = {k: F[k](SIGMA_HERE) for k in ("rho", "K", "vL", "vD_therm", "ThD", "Tm", "cs_air")}

# analytic expectations at sigma = 0 [DERIVED from the code's formulas]
mat = MATERIALS[MAT]
from sigma_ground.field.constants import K_B, EV_TO_J  # noqa: E402
f_zpe = (9.0 / 8.0) * K_B * LEVEL["ThD"] / (mat["cohesive_energy_ev"] * EV_TO_J)
EXPECT = {
    "rho": FQ,
    "K": f_zpe * 0.5 * FQ,                 # d/ds [f_zpe (1 - mr^-1/2)] at s = 0
    "vL": 0.5 * (f_zpe * 0.5 * FQ - FQ),
    "vD_therm": 0.5 * f_zpe * 0.5 * FQ,
    "vD_acou": 0.5 * (f_zpe * 0.5 * FQ - FQ),
}

# ---------------------------------------------------------------- chain identities (pre-registered)
TOL = 1e-6
I = {
    "I1_dlnrho_minus_fQCD": D["rho"] - FQ,
    "I2_dlnvL_minus_half(dlnK-dlnrho)": D["vL"] - 0.5 * (D["K"] - D["rho"]),
    "I2b_dlnvL_minus_half(dlnM_long-dlnrho)": D["vL"] - 0.5 * (
        dln(lambda s: F["K"](s) + 4.0 * F["G"](s) / 3.0, H) - D["rho"]),
    "I3_design_sketch_dlnvD_therm_minus_dlnvL": D["vD_therm"] - D["vL"],
    "I3_like_for_like_dlnvD_therm_minus_dlnvD_acou": D["vD_therm"] - D["vD_acou"],
}
VERDICT_I = {k: ("PASS" if abs(v) < TOL else "FAIL") for k, v in I.items()}
chain_pass = all(VERDICT_I[k] == "PASS" for k in ("I1_dlnrho_minus_fQCD",
                                                     "I2_dlnvL_minus_half(dlnK-dlnrho)",
                                                     "I3_like_for_like_dlnvD_therm_minus_dlnvD_acou"))
EXTRA = {
    "E1_dlnThD_minus_dlnvD_therm (n frozen)": D["ThD"] - D["vD_therm"],
    "E2_dlnTm_minus_2dlnThD (M, n frozen)": D["Tm"] - 2 * D["ThD"],
    "E3_dlnTm_minus_dlnK (physics: T_m prop. K)": D["Tm"] - D["K"],
    "E4_dlnTm_cons_minus_dlnK": D["Tm_cons"] - D["K"],
    "E5_dlnThD_minus_dlnThD_cons": D["ThD"] - D["ThD_cons"],
    "E6_dlncp_minus_dlncp_cons": D["cp"] - D["cp_cons"],
    "E7_dlnM_air_minus_fQCD": D["M_air"] - FQ,
    "E8_dlncs_air_minus_dlncs_air_cons": D["cs_air"] - D["cs_air_cons"],
    "E9_dlncs_air_minus(-0.5dlnM+0.5dlngamma)": D["cs_air"] - (-0.5 * D["M_air"] + 0.5 * D["gamma_air"]),
}

# ---------------------------------------------------------------- sweep over every material
SWEEP = {}
for key in MECHANICAL_DATA:
    try:
        d_r = dln(lambda s: acoustics.density_at_sigma(key, s), H)
        d_K = dln(lambda s: mechanical.bulk_modulus(key, s), H)
        d_vL = dln(lambda s: acoustics.longitudinal_wave_speed(key, s), H)
        d_vDa = dln(lambda s: acoustics.debye_velocity(key, s), H)
        d_vDt = dln(lambda s: thermal.sound_velocity(key, s), H)
        try:
            d_Tm = dln(lambda s: phase_transition.lindemann_melting_estimate(key, s), H)
        except Exception as e:  # noqa: BLE001
            d_Tm = None
        SWEEP[key] = {"dlnrho": d_r, "dlnK": d_K, "dlnvL": d_vL, "dlnvD_acou": d_vDa, "dlnvD_therm": d_vDt,
                      "dlnTm": d_Tm, "I1": d_r - FQ, "I2": d_vL - 0.5 * (d_K - d_r), "I3": d_vDt - d_vDa}
    except Exception as e:  # noqa: BLE001
        SWEEP[key] = {"error": repr(e)}
sw_ok = [v for v in SWEEP.values() if "error" not in v]
SWEEP_SUMMARY = {
    "n_materials": len(SWEEP), "n_evaluated": len(sw_ok),
    "max_abs_I1": max(abs(v["I1"]) for v in sw_ok), "max_abs_I2": max(abs(v["I2"]) for v in sw_ok),
    "min_abs_I3": min(abs(v["I3"]) for v in sw_ok), "max_abs_I3": max(abs(v["I3"]) for v in sw_ok),
    "dlnK_range": [min(v["dlnK"] for v in sw_ok), max(v["dlnK"] for v in sw_ok)],
    "n_dlnTm_positive": sum(1 for v in sw_ok if v["dlnTm"] is not None and v["dlnTm"] > 0),
}

# ---------------------------------------------------------------- the coupling conventions in the code
body = lambda s: CelestialBody(1.0e24, np.zeros(3), np.zeros(3), 1.0e6, 0.0, s)  # noqa: E731
CONV_F = {
    "QCD-fraction mass (acoustics/gas/mechanical/atomic_spectra): m = m0[(1-f)+f e^s]":
        lambda s: gas.molecular_mass_kg("N2", s),
    "proton mass, atomic_spectra._proton_mass_kg: m_bare + m_QCD e^s":
        lambda s: atomic_spectra._proton_mass_kg(s),
    "universal nbody GM_eff = GM e^s (CelestialBody.gm_m3_s2)": lambda s: body(s).gm_m3_s2,
    "e^s clock, relativity.sigma_time_dilation(s, 1) = e^s": lambda s: relativity.sigma_time_dilation(s, 1.0),
    "T e^s, statistical.sigma_partition_shift(s, 300) = T e^s": lambda s: statistical.sigma_partition_shift(s, 300.0),
    "argon mass in atmosphere.py (sigma-invariant constant)": lambda s: (
        atmosphere.mean_molecular_mass_kg(s) - sum(x * gas.molecular_mass_kg(sp, s)
                                                   for sp, x in atmosphere.DRY_AIR_COMPOSITION.items()
                                                   if sp in gas.MOLECULES)),
    "Lindemann atomic mass M = A m_u (sigma-invariant)": lambda s: (
        phase_transition.lindemann_melting_estimate(MAT, s) / thermal.debye_temperature(MAT, s) ** 2),
    "thermal.sound_velocity density (frozen rho_0)": lambda s: (
        thermal.sound_velocity(MAT, s) ** -2 * (mechanical.shear_modulus(MAT, s))),
}
CONV = {k: dln(f, H) for k, f in CONV_F.items()}

# ---------------------------------------------------------------- observability
FLOORS = {  # fixed in u12_prereg.json before this run
    "rho": (1.2e-8, "[LIT] Bartl+2017 Metrologia 54, 693 (XRCD N_A 1.2e-8)"),
    "K": (2e-7, "[DERIVED from LIT] rho v^2, pulse-echo 1e-7 (Pantea+2005 RSI 76, 114902)"),
    "G": (2e-7, "[DERIVED from LIT] as K"),
    "vL": (1e-7, "[LIT] Pantea+2005 RSI 76, 114902"),
    "vT": (1e-7, "[LIT] Pantea+2005 RSI 76, 114902"),
    "vD_therm": (1e-7, "[LIT] Pantea+2005 (v_D from v_L, v_T)"),
    "ThD": (1e-7, "[DERIVED from LIT] Theta_D prop. v_D"),
    "Tm": (1.3e-7, "[DERIVED from LIT] NIST Ga cells 0.04 mK / 302.9146 K"),
    "cp": (1e-3, "[LIT] adiabatic calorimetry 0.1 % (Stevens & Boerio-Goates 2004)"),
    "cs_air": (3.5e-7, "[DERIVED from LIT] AGT k_B 0.71 ppm (de Podesta+2013), c_s ~ sqrt(k_B)"),
}
CONS_KEY = {"rho": "rho", "K": "K", "G": "G", "vL": "vL", "vT": "vT", "vD_therm": "vD_acou",
            "ThD": "ThD_cons", "Tm": "Tm_cons", "cp": "cp_cons", "cs_air": "cs_air_cons"}
OBS = []
for prop, (floor, fsrc) in FLOORS.items():
    for chain, dkey in (("as_coded", prop), ("consistent", CONS_KEY[prop])):
        for nname, nz in NORM.items():
            for ch in ("dsig_annual", "dsig_km"):
                eff = abs(D[dkey]) * nz[ch]
                OBS.append({"property": prop, "chain": chain, "normalisation": nname, "channel": ch,
                            "dlnX_dsigma": D[dkey], "effect": eff, "floor": floor, "floor_source": fsrc,
                            "orders_below_floor": math.log10(floor / eff) if eff > 0 else float("inf")})


def crit(rows):
    return all(r["orders_below_floor"] >= 5.0 for r in rows)


CRIT = {}
for chain in ("as_coded", "consistent"):
    for nname in NORM:
        rows = [r for r in OBS if r["chain"] == chain and r["normalisation"] == nname]
        CRIT[f"{chain}|{nname}|both_channels"] = crit(rows)
        CRIT[f"{chain}|{nname}|annual_only"] = crit([r for r in rows if r["channel"] == "dsig_annual"])
        CRIT[f"{chain}|{nname}|per_km_only"] = crit([r for r in rows if r["channel"] == "dsig_km"])
        CRIT[f"{chain}|{nname}|min_orders"] = min(r["orders_below_floor"] for r in rows)
        CRIT[f"{chain}|{nname}|all_below_floor"] = all(r["orders_below_floor"] > 0 for r in rows)

# frequency-metrology rows (design correction; [LIT] floors)
LANGE_CENTRAL, LANGE_SIG = 7e-8, 45e-8          # [LIT] Lange+2021 PRL 126, 011102: (c^2/mu) dmu/dPhi = 7(45)e-8
lange_bound = abs(LANGE_CENTRAL) + 2 * LANGE_SIG
CLOCK = {}
for nname, nz in NORM.items():
    pred = -2.0 * XI * nz["factor"] * FQ          # (c^2/mu) d mu / d Phi = d ln mu/d sigma * d sigma/d(Phi/c^2)
    CLOCK[nname] = {"prediction_c2_dlnmu_dPhi": pred, "bound_2sigma": lange_bound,
                    "excluded_by": abs(pred) / lange_bound,
                    "annual_dlnmu_effect": FQ * nz["dsig_annual"],
                    "annual_dlnmu_floor": lange_bound * dPhi_annual}
# acoustic resonator referenced to an optical clock: f = v/(2L), L electromagnetic -> d ln f = d ln v_L
RES = {}
for nname, nz in NORM.items():
    eff = abs(D["vD_acou"]) * nz["dsig_annual"]
    RES[nname] = {"effect_annual": eff,
                  "floor_short_term_stability": 2.5e-14,
                  "floor_source": "[LIT] best room-temperature quartz BAW stability 2.5e-14 (cited in Lo+2016 PRX 6, 011018); "
                                  "annual-period aging floor NOT established here",
                  "orders_below_floor": math.log10(2.5e-14 / eff),
                  "literature_home": "Campbell+2021 PRL 126, 071301 (quartz BAW vs H-maser and CSO, |d_me - d_g|)"}

# =====================================================================================================
# v2 additions (repair round): every block below answers a skeptic's required change.
# The pre-registered criterion, floors and identities above are untouched.
# =====================================================================================================
from sigma_ground.field.interface import (plasticity, grain_structure, hysteresis, corrosion,  # noqa: E402
                                          fluid, diffusion, piezoelectricity)


def dln5(f, h):
    """5-point stencil log-derivative at sigma = 0 (truncation O(h^4))."""
    return (-f(2 * h) + 8 * f(h) - 8 * f(-h) + f(-2 * h)) / (12.0 * h * f(0.0))


# ---- (R1) I3 = f_QCD/2 in every material is BY CONSTRUCTION: v_D(thermal)/v_D(acoustics) = sqrt(rho(s)/rho0)
I3_BC = {"per_material": {}}
for key in MECHANICAL_DATA:
    try:
        ratio03 = thermal.sound_velocity(key, 0.3) / acoustics.debye_velocity(key, 0.3)
        i3_5 = dln5(lambda s: thermal.sound_velocity(key, s), 1e-3) - dln5(lambda s: acoustics.debye_velocity(key, s), 1e-3)
        I3_BC["per_material"][key] = {"ratio_at_sigma_0.3": ratio03, "I3_5pt": i3_5, "I3_5pt_minus_half_f": i3_5 - FQ / 2}
    except Exception as e:  # noqa: BLE001
        I3_BC["per_material"][key] = {"error": repr(e)}
_ok = [v for v in I3_BC["per_material"].values() if "error" not in v]
I3_BC["sqrt_mass_ratio_at_0.3"] = math.sqrt(mass_ratio(0.3))
I3_BC["max_abs_ratio_minus_sqrt_mass_ratio"] = max(abs(v["ratio_at_sigma_0.3"] - I3_BC["sqrt_mass_ratio_at_0.3"]) for v in _ok)
I3_BC["max_abs_I3_5pt_minus_half_f"] = max(abs(v["I3_5pt_minus_half_f"]) for v in _ok)
I3_BC["tag"] = ("[BY CONSTRUCTION] both modules take K(sigma), G(sigma) from mechanical and differ only by rho; "
                "the ratio is sqrt(mass_ratio(sigma)) for every material, so I3 = f_QCD/2 identically")
# T_m increases with sigma everywhere is also by construction: d ln T_m = d ln K = f_zpe f/2 > 0 (ZPE term positive)
TM_BC = {"max_abs_dlnTm_minus_dlnK_over_sweep": max(abs(v["dlnTm"] - v["dlnK"]) for v in sw_ok if v["dlnTm"] is not None),
         "tag": "[BY CONSTRUCTION] frozen M and frozen rho cancel, so d ln T_m = d ln K, which the ZPE term makes positive"}

# ---- (R2) c_p gap decomposition (consistent minus coded): per-mass term and Debye-Pade term
def _pade_ratio(s):
    x_t, x_c = F["ThD"](s) / (4 * T_CP), ThD_cons(s) / (4 * T_CP)
    return (1 + x_t ** 2) / (1 + x_c ** 2)


CP_DECOMP = {
    "dlncp_coded": D["cp"], "dlncp_consistent": D["cp_cons"], "gap": D["cp_cons"] - D["cp"],
    "per_mass_term_-dlnrho": -dln(lambda s: F["rho"](s) / F["rho"](0.0), H),
    "debye_pade_term": dln(_pade_ratio, H),
    "tag": "[DERIVED] from thermal.py:370-371 (C_v = 3 n k_B /(1 + x^2), x = Theta_D/4T) and thermal.py:390 (c_p = C_v/rho0)",
}
CP_DECOMP["sum_check"] = CP_DECOMP["per_mass_term_-dlnrho"] + CP_DECOMP["debye_pade_term"] - CP_DECOMP["gap"]

# ---- (R3) Coupling audit extended: atom-level inventory law, direct matter couplings, E_coh law split
be_fe56 = M("BE_Fe56", "nuclear_binding_energy", {"protons": 26, "neutrons": 30})
be_fe56 = float(be_fe56["binding_energy_MeV"]) if isinstance(be_fe56, dict) else float(be_fe56)


def _m_atom_fe56(s):
    return inv_sigma.three_measures_atom(26, 30, be_fe56, s)["atom_stable_mass_kg"]


hyst0 = hysteresis.sigma_hysteresis_shift(MAT, 0.0)
EXT_F = {
    "inventory atom-level law, Fe-56 (three_measures_atom: electrons sigma-invariant, SEMF binding x e^s)":
        (_m_atom_fe56, "inventory/core/sigma.py:264-285"),
    "yield stress, plasticity.yield_stress (sy x e^s)": (lambda s: plasticity.yield_stress(MAT, s), "plasticity.py:469-470"),
    "Hall-Petch slope, grain_structure.hall_petch_slope (k x e^s)":
        (lambda s: grain_structure.hall_petch_slope(MAT, s), "grain_structure.py:405-406"),
    "Curie T_C, hysteresis.sigma_hysteresis_shift (E_coh law 1 + f(e^s - 1))":
        (lambda s: hysteresis.sigma_hysteresis_shift(MAT, s), "hysteresis.py:617-622"),
    "oxidation rate, corrosion.sigma_corrosion_shift (Q prop. E_coh, 1 + f(e^s - 1), Arrhenius), 300 K":
        (lambda s: corrosion.sigma_corrosion_shift(MAT, s), "corrosion.py:567-575"),
    "grain growth K_g, grain_structure.grain_growth_rate_constant (Q x e^s, Arrhenius), 900 K":
        (lambda s: grain_structure.grain_growth_rate_constant(MAT, 900.0, s), "grain_structure.py:462-463"),
    "grain growth K_g, grain_structure.grain_growth_rate_constant (Q x e^s, Arrhenius), 300 K":
        (lambda s: grain_structure.grain_growth_rate_constant(MAT, 300.0, s), "grain_structure.py:462-463"),
    "E_coh, ZPE law with derived f_zpe (mechanical._effective_cohesive_energy_j)":
        (lambda s: mechanical._effective_cohesive_energy_j(MAT, s), "mechanical.py:317-322 (also surface.py:551/611)"),
    "E_coh, ZPE law with fixed f_zpe = 0.01 (fluid._cohesive_energy_j)":
        (lambda s: fluid._cohesive_energy_j(MAT, s), "fluid.py:191-193"),
    "diffusion activation energy, ZPE law with fixed f_zpe = 0.01 (diffusion.activation_energy_ev)":
        (lambda s: diffusion.activation_energy_ev(MAT, s), "diffusion.py:184-188"),
    "E_coh, full QCD-fraction law 1 + f(e^s - 1) (hysteresis ratio, same law as corrosion.py:569)":
        (lambda s: hysteresis.sigma_hysteresis_shift(MAT, s) / hyst0, "hysteresis.py:620, corrosion.py:569"),
    "quartz AT-cut resonator frequency, piezoelectricity.sigma_resonant_frequency_shift (f_zpe = 0.01)":
        (lambda s: piezoelectricity.sigma_resonant_frequency_shift("quartz", 1e-3, s)[1], "piezoelectricity.py:365-407"),
}
EXT = {}
for k, (f, where) in EXT_F.items():
    try:
        d3, d4 = dln(f, 1e-3), dln(f, 1e-4)
        EXT[k] = {"dln_dsigma": d4, "h=1e-3": d3, "where": where,
                  "continuity_f(+h)/f(0)-1": f(1e-4) / f(0.0) - 1.0, "continuity_f(-h)/f(0)-1": f(-1e-4) / f(0.0) - 1.0}
    except Exception as e:  # noqa: BLE001
        EXT[k] = {"error": repr(e), "where": where}
EXT_MAX_ABS = max(abs(v["dln_dsigma"]) for v in EXT.values() if "dln_dsigma" in v)
EXT_ATOM_GAP = EXT[next(k for k in EXT if k.startswith("inventory atom-level"))]["dln_dsigma"] - FQ
ECOH_RATIO = (EXT[next(k for k in EXT if k.startswith("E_coh, full"))]["dln_dsigma"]
              / EXT[next(k for k in EXT if k.startswith("E_coh, ZPE law with derived"))]["dln_dsigma"])

# ---- (R4) Observability under the engine's OTHER E_coh law (corrosion/hysteresis), consistent chain [DERIVED]
def _ecoh_alt(s):
    return hysteresis.sigma_hysteresis_shift(MAT, s) / hyst0


def K_alt(s):
    return F["K"](0.0) * _ecoh_alt(s)


def vL_alt(s):
    return F["vL"](0.0) * math.sqrt(_ecoh_alt(s) / (F["rho"](s) / F["rho"](0.0)))


def ThD_alt(s):
    return F["ThD"](0.0) * vL_alt(s) / vL_alt(0.0)


def Tm_alt(s):
    return F["Tm"](0.0) * mass_ratio(s) * (ThD_alt(s) / ThD_alt(0.0)) ** 2


def cp_alt(s):
    x0, xa = ThD_alt(0.0) / (4 * T_CP), ThD_alt(s) / (4 * T_CP)
    return F["cp"](0.0) * (1 + x0 ** 2) / (1 + xa ** 2) * (F["rho"](0.0) / F["rho"](s))


D_ALT = {"rho": D["rho"], "K": dln(K_alt, H), "G": dln(K_alt, H), "vL": dln(vL_alt, H), "vT": dln(vL_alt, H),
         "vD_therm": dln(vL_alt, H), "ThD": dln(ThD_alt, H), "Tm": dln(Tm_alt, H), "cp": dln(cp_alt, H),
         "cs_air": D["cs_air_cons"]}
OBS_ALT = []
for prop, (floor, fsrc) in FLOORS.items():
    for nname, nz in NORM.items():
        for ch in ("dsig_annual", "dsig_km"):
            eff = abs(D_ALT[prop]) * nz[ch]
            OBS_ALT.append({"property": prop, "normalisation": nname, "channel": ch, "dlnX_dsigma": D_ALT[prop],
                            "effect": eff, "floor": floor,
                            "orders_below_floor": math.log10(floor / eff) if eff > 1e-300 else float("inf")})
MIN_ALT = {f"{n}|{ch}": min(r["orders_below_floor"] for r in OBS_ALT if r["normalisation"] == n and r["channel"] == ch)
           for n in NORM for ch in ("dsig_annual", "dsig_km")}

# ---- (R5) Could the observability verdict have failed? |d ln X/d sigma| needed to reach each floor in one year
NEEDED = {prop: {"field_eq": floor / NORM["field_eq"]["dsig_annual"], "inventory": floor / NORM["inventory"]["dsig_annual"]}
          for prop, (floor, _) in FLOORS.items()}
NEEDED_MIN_FIELD = min(v["field_eq"] for v in NEEDED.values())
ALL_COUPLINGS = [abs(v) for v in D.values()] + [abs(v["dln_dsigma"]) for v in EXT.values() if "dln_dsigma" in v] + \
                [abs(v) for v in D_ALT.values()]
MAX_ENGINE_COUPLING = max(ALL_COUPLINGS)

# ---- (R6) Resonator row: engine quartz law, named reference clocks
d_quartz = EXT[next(k for k in EXT if k.startswith("quartz AT-cut"))]["dln_dsigma"]
d_mp = dln(atomic_spectra._proton_mass_kg, H)   # hyperfine (H maser, Cs) frequency prop. g alpha^2 R_inf m_e/m_p -> -d ln m_p
RES2 = {}
for nname, nz in NORM.items():
    rows = {}
    for ref, coef in (("optical clock (sigma-invariant: prop. m_e alpha^2)", d_quartz),
                      ("H maser or Cs (hyperfine prop. m_e/m_p)", d_quartz + d_mp)):
        eff = abs(coef) * nz["dsig_annual"]
        rows[ref] = {"dln_ratio_dsigma": coef, "effect_annual": eff,
                     "log10_floor_over_effect_short_term": math.log10(2.5e-14 / eff)}
    RES2[nname] = rows
RES2["floor_note"] = ("[LIT] 2.5e-14 is the short-term stability of room-temperature quartz BAW oscillators cited in Lo+2016 "
                      "PRX 6, 011018. No annual-period (aging, temperature) floor is cited, so whether a resonator can see the "
                      "annual signal is OPEN [SPECULATIVE]. Campbell+2021 PRL 126, 071301 is a 16-day (H maser vs OCXO) and "
                      "2-day (OCXO vs CSO) oscillating scalar-dark-matter search, not an annual-period measurement.")
RES2["dln_mp_dsigma"] = d_mp

# ---- (R7) Free-fall equivalence-principle channel (force -c^2 grad m(sigma)), [DERIVED]; MICROSCOPE [LIT]
m_e_ = float(M("m_e", "lookup_constant", {"name": "m_e"}))
m_u_ = float(M("m_u", "lookup_constant", {"name": "m_u"}))
ti = M("Ti", "element_atomic_data", {"element_symbol": "Ti"})
pt = M("Pt", "element_atomic_data", {"element_symbol": "Pt"})
df_e = Mnum("delta_f_electron_term_Ti_Pt",
            f"{FQ}*({ti['atomic_number']}/{ti['atomic_mass_amu']} - {pt['atomic_number']}/{pt['atomic_mass_amu']})*{m_e_}/{m_u_}")
# repo's own atom-level law applied to the pure elements (cross-check of S9a's alloy number)
be_ti48 = M("BE_Ti48", "nuclear_binding_energy", {"protons": 22, "neutrons": 26})
be_pt195 = M("BE_Pt195", "nuclear_binding_energy", {"protons": 78, "neutrons": 117})
be_ti48 = float(be_ti48["binding_energy_MeV"]) if isinstance(be_ti48, dict) else float(be_ti48)
be_pt195 = float(be_pt195["binding_energy_MeV"]) if isinstance(be_pt195, dict) else float(be_pt195)
f_ti48 = dln(lambda s: inv_sigma.three_measures_atom(22, 26, be_ti48, s)["atom_stable_mass_kg"], H)
f_pt195 = dln(lambda s: inv_sigma.three_measures_atom(78, 117, be_pt195, s)["atom_stable_mass_kg"], H)
S9A = json.load(open(REPO + "/misc/unification/sigma_exclusion.json", encoding="utf-8"))
df_s9a = S9A["delta_f"]
MIC_C, MIC_STAT, MIC_SYS = 1.5e-15, 2.3e-15, 1.5e-15   # [LIT] Touboul+2022 PRL 129, 121102: eta(Ti,Pt) = [-1.5 +- 2.3 +- 1.5]e-15
mic_bound = Mnum("microscope_bound_2sigma", f"{MIC_C} + 2*sqrt({MIC_STAT}**2 + {MIC_SYS}**2)")
EP = {"derivation": "[DERIVED] S = -int m(sigma) c^2 dtau gives a = -c^2 (d ln m/d sigma) grad sigma; with sigma = -2 xi Phi/c^2 "
                    "(field normalisation) the differential acceleration is eta = 2 xi |Delta f| (inventory normalisation: xi |Delta f|)",
      "note": "acoustics.density_at_sigma applies the same 0.990 to every material (eta = 0 under that law); a composition "
              "dependence exists only where electrons are sigma-invariant (atomic_spectra, inventory) and binding is scaled (inventory)",
      "bound_2sigma": mic_bound, "bound_source": "[LIT] Touboul+2022 PRL 129, 121102, doi:10.1103/PhysRevLett.129.121102, arXiv:2209.15487",
      "rows": {}}
for label, df in (("electron-mass term alone, pure Ti vs pure Pt", abs(df_e)),
                  ("repo inventory law, pure Ti-48 vs Pt-195 (three_measures_atom)", abs(f_ti48 - f_pt195)),
                  ("repo inventory law, TA6V vs PtRh10 alloys (S9a record misc/unification/sigma_exclusion.json)", abs(df_s9a))):
    for nname, nz in NORM.items():
        eta = 2.0 * XI * nz["factor"] * df
        EP["rows"][f"{label}|{nname}"] = {"delta_f": df, "eta": eta, "excluded_by": eta / mic_bound,
                                          "log10_floor_over_effect": math.log10(mic_bound / eta)}
EP["f_Ti48"], EP["f_Pt195"], EP["delta_f_electron_signed_Ti_minus_Pt"] = f_ti48, f_pt195, df_e
EP["electron_term_note"] = ("not a strict lower bound: under the inventory law the binding term (+) dominates the electron "
                            "term (-) and the two have opposite signs; it shows the size that survives with no binding coupling")

# ---- (R8) Delta Phi/c^2 to more digits (rounding fix)
DPHI_PRINT = f"{dPhi_annual:.6e}"

# ---------------------------------------------------------------- write JSON (v2)
prereg = json.load(open(os.path.join(OUT, "u12_prereg.json"), encoding="utf-8"))
res = {
    "id": "U12", "version": "v2 (repair round)", "runtime_s": None, "prereg": prereg,
    "mentat_sources": SRC,
    "engine_constants": {"XI": XI, "PROTON_QCD_FRACTION": FQ, "SIGMA_HERE": SIGMA_HERE,
                         "source": "sigma_ground.inventory.core.sigma.XI; sigma_ground.field.constants"},
    "lit": {"earth_eccentricity": [E_ORB, "NASA NSSDC Earth fact sheet J2000 mean elements"],
            "lange2021": "(c^2/mu) dmu/dPhi = 7(45)e-8, (c^2/alpha) dalpha/dPhi = 14(11)e-9; PRL 126, 011102; arXiv:2010.06620",
            "touboul2022": "eta(Ti,Pt) = [-1.5 +- 2.3 (stat) +- 1.5 (syst)]e-15; PRL 129, 121102; arXiv:2209.15487",
            "quartz_short_term_stability": "2.5e-14 (room-temperature BAW, cited in Lo+2016 PRX 6, 011018)"},
    "sigma_increments": NORM, "dPhi_over_c2_annual_amplitude": dPhi_annual, "dPhi_over_c2_annual_amplitude_6sf": DPHI_PRINT,
    "levels_sigma0": LEVEL, "Tm_shift_at_sigma1_iron": TM_SIGMA1, "dlnv_symbolic_at_s0": dlnv_at0, "levels_sigma_here": LEVEL_SIGMA_HERE,
    "anchors_mentat": {"rho_iron": rho_fe_mentat, "Tmelt_iron": tm_fe_mentat, "E_iron": E_fe_mentat,
                       "cs_air_288K": cs_air_mentat},
    "dlnX_dsigma": D, "convergence": DCONV, "analytic_expectation": EXPECT, "f_zpe_iron": f_zpe,
    "identities": I, "identity_verdicts": VERDICT_I, "chain_PASS_design": chain_pass,
    "extra_identities": EXTRA, "sweep_summary": SWEEP_SUMMARY, "sweep": SWEEP,
    "I3_by_construction": I3_BC, "Tm_positive_by_construction": TM_BC, "cp_gap_decomposition": CP_DECOMP,
    "coupling_conventions_dln_dsigma": CONV, "coupling_conventions_extended": EXT,
    "max_abs_extended_coupling": EXT_MAX_ABS, "fe56_atom_law_minus_fQCD": EXT_ATOM_GAP,
    "Ecoh_full_over_ZPE_ratio": ECOH_RATIO,
    "observability": OBS, "observability_criterion": CRIT,
    "observability_alt_Ecoh_law": {"dlnX_dsigma": D_ALT, "rows": OBS_ALT, "min_orders": MIN_ALT,
                                   "tag": "[DERIVED] K prop. E_coh with the corrosion/hysteresis law; rho(sigma), M(sigma) carried"},
    "needed_coupling_to_reach_floor_annual": NEEDED, "needed_min_field": NEEDED_MIN_FIELD,
    "max_engine_coupling_found": MAX_ENGINE_COUPLING,
    "clock_row": CLOCK, "acoustic_resonator_row": RES, "acoustic_resonator_row_v2": RES2, "ep_row": EP,
}
res["lit_refs"] = [
    "Lange et al. 2021, PRL 126, 011102, doi:10.1103/PhysRevLett.126.011102, arXiv:2010.06620: (c^2/mu)dmu/dPhi = 7(45)e-8",
    "Touboul et al. 2022, PRL 129, 121102, doi:10.1103/PhysRevLett.129.121102, arXiv:2209.15487: eta(Ti,Pt) = [-1.5 +- 2.3 +- 1.5]e-15",
    "Campbell et al. 2021, PRL 126, 071301, arXiv:2010.08107: 16 days H maser vs OCXO, 2 days OCXO vs CSO; |d_me - d_g| (oscillating scalar DM)",
    "Lo et al. 2016, PRX 6, 011018, doi:10.1103/PhysRevX.6.011018, arXiv:1412.2142: cites room-temperature BAW stability 2.5e-14",
    "Bartl et al. 2017, Metrologia 54, 693, doi:10.1088/1681-7575/aa7820: N_A by XRCD, 1.2e-8",
    "Pantea et al. 2005, Rev. Sci. Instrum. 76, 114902, doi:10.1063/1.2130715: digital pulse-echo precision better than 1e-7",
    "de Podesta et al. 2013, Metrologia 50, 354, doi:10.1088/0026-1394/50/4/354: k_B by AGT, u_r = 0.71e-6; later revised by -2.7e-6 "
    "(argon molar mass; de Podesta et al. 2015 Metrologia 52, S353 and 2017 Metrologia 54, 683)",
    "Stevens & Boerio-Goates 2004, J. Chem. Thermodyn. 36, 857, doi:10.1016/j.jct.2004.06.008: Cu c_p agreement 0.1 %",
    "Strouse (NIST), gallium TRIPLE-point cells: 13 cells span 0.04 mK (TP 302.9166 K; ITS-90 melting point 302.9146 K), nist.gov/node/735371",
    "NASA NSSDC Earth fact sheet: e = 0.01671022 (J2000 mean elements)",
]
res["findings"] = {
    "chain_identities": {"I1": I["I1_dlnrho_minus_fQCD"], "I2": I["I2_dlnvL_minus_half(dlnK-dlnrho)"],
                         "I3": I["I3_like_for_like_dlnvD_therm_minus_dlnvD_acou"],
                         "I1_equals_truncation_f_h2_over_6": FQ * H ** 2 / 6.0},
    "TODO_cancellation_claim_ratio_dlnrho_over_dlnK": D["rho"] / D["K"],
    "TODO_cancelled_fraction_of_dlnrho": D["K"] / D["rho"],
    "design_chain_criterion": "FAIL (predicted)",
    "design_observability_criterion_as_written": "FAIL" if not CRIT["consistent|field_eq|both_channels"] else "PASS",
    "design_criterion_per_km_only": "PASS" if CRIT["consistent|field_eq|per_km_only"] else "FAIL",
    "corrected_criterion_all_property_rows_below_floor": "PASS" if CRIT["consistent|field_eq|all_below_floor"] else "FAIL",
    "min_orders_below_floor_field": CRIT["consistent|field_eq|min_orders"],
    "min_orders_below_floor_alt_Ecoh_annual_field": MIN_ALT["field_eq|dsig_annual"],
    "clocks_only_channel_clause": "REFUTED as worded: free-fall EP tests reach grad sigma more strongly than clocks",
}
res["runtime_s"] = time.time() - T0
with open(os.path.join(OUT, "u12_results_v2.json"), "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1, default=float)

# ---------------------------------------------------------------- console summary
print(f"runtime {res['runtime_s']:.1f} s   dPhi/c2 = {DPHI_PRINT}")
for k in ("rho", "K", "vL", "vD_acou", "vD_therm", "ThD", "ThD_cons", "Tm", "Tm_cons", "cp", "cp_cons", "cs_air", "cs_air_cons"):
    print(f"  dln {k:12s}/dsigma = {D[k]:+.7e}")
for k, v in I.items():
    print(f"  {k:52s} {v:+.3e}  {VERDICT_I[k]}")
print("I3 by construction:", I3_BC["max_abs_ratio_minus_sqrt_mass_ratio"], I3_BC["max_abs_I3_5pt_minus_half_f"])
print("Tm by construction:", TM_BC["max_abs_dlnTm_minus_dlnK_over_sweep"])
print("cp decomposition:", json.dumps(CP_DECOMP, indent=0))
for k, v in EXT.items():
    print(f"  ext {v.get('dln_dsigma', v.get('error'))}  {k}")
print("Fe-56 atom law - fQCD", EXT_ATOM_GAP, "| E_coh full/ZPE", ECOH_RATIO)
print("alt E_coh dlnX", {k: f"{v:+.4e}" for k, v in D_ALT.items()})
print("alt E_coh min orders", MIN_ALT)
print("needed", {k: f"{v['field_eq']:.0f}" for k, v in NEEDED.items()}, "max engine coupling", MAX_ENGINE_COUPLING)
print("resonator v2", json.dumps(RES2["field_eq"], indent=0))
print("EP", json.dumps(EP["rows"], indent=0), "f_Ti48", f_ti48, "f_Pt195", f_pt195, "df_e", df_e)
for k, v in CRIT.items():
    print(" ", k, v)
print("clock", json.dumps(CLOCK, indent=0))
