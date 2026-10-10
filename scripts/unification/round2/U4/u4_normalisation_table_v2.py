#!/usr/bin/env python
"""U4 v2 - factor-2 and the two clock laws: sigma-normalisation decision table from Mentat leaves.

Repair pass after three skeptic reviews. The pre-registered table, metrics and criterion
(U4_prereg.json, unchanged) are recomputed exactly as in v1; the repairs add:
  R1  the xi/2-at-r_s attribution (weak-field formula evaluated at r_s, clamp inactive there),
  R2  the horizon row with correct limit signs, the regular branch's finite value, and gradient energy,
  R3  the coupling assumption made explicit: clock-only (pre-registered table) vs universal (metric)
      coupling, where GP-A is blind and Cassini gamma carries the exclusion,
  R4  [LIT] re-attribution (1.4e-4 = Vessot 1989 via Delva 2018 ref [14]) and margins vs every bound,
  R5  both route-(c) amplitude factors and the design inconsistency 2.5-2 vs 2.5-6(c),
  R6  C3 Omega values parsed from the sigma.py:40 docstring (no Mentat constant exists for them),
  R7  call-site scan for sigma_time_dilation / sigma_from_potential (is the composition coded?),
  R8  gr_basics.py:61 precision evidence at 1 km, and a Sun-mass leaf provenance flag.

Feeds decision C2 (misc/MENTAT_GAPS_UNIFICATION.md). Computes the table; decides nothing.
Run from the repo root:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -X utf8 <this file>
Read-only with respect to the repo.
"""
import sys, os, json, math, re, ast

REPO = r"D:/Aaron/development/sigma-ground"
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO + "/tools")
sys.path.insert(0, REPO)

from mentat_client import mentat                                   # noqa: E402
from sigma_ground.inventory.core.sigma import XI, sigma_from_potential   # noqa: E402
from sigma_ground.field.relativity import sigma_time_dilation      # noqa: E402
from sigma_ground.field.gr_basics import time_dilation_gr          # noqa: E402
from sigma_ground.field.gr_basics import gravitational_redshift as z_gr_engine  # noqa: E402
from sigma_ground.field.scale import scale_ratio                   # noqa: E402

LOG = []          # every Mentat call: tool, args, value, source


def M(tool, args):
    v, s = mentat(tool, args)
    LOG.append(dict(tool=tool, args=args, value=v, source=s))
    return v, s


def S(expr):
    """Arithmetic through Mentat's sympy (simplify_expr) -> float."""
    v, _ = M("simplify_expr", {"expression": expr})
    return float(v)


def num(v):
    return float(v["value"]) if isinstance(v, dict) else float(v)


# ---------------------------------------------------------------- literature bounds [LIT], all web-verified 2026-10-10
LIT = {
    "GPA_bound": dict(
        value=1.4e-4,
        what="GP-A (1976) |alpha| <= 1.4e-4 in z = (1+alpha) dU/c^2; a 1-sigma uncertainty, not a hard bound. "
             "PRE-REGISTERED criterion.",
        refs=["Vessot R.F.C. 1989, 'Clocks and space tests of relativistic gravitation', Adv. Space Res. 9(9), 21-28 "
              "(the source of 1.4e-4), as quoted by",
              "Delva P. et al. 2018, PRL 121, 231101, doi:10.1103/PhysRevLett.121.231101, arXiv:1812.03711 p.1: "
              "'reached an uncertainty of |alpha| <= 1.4 x 10^-4 [14]', ref [14] = Vessot, Adv. Space Res. 9, 21 (1989)"],
        correction_vs_v1="v1 and the design attributed 1.4e-4 to Vessot+1980; that paper's abstract gives the 70e-6 level.",
        tag="[LIT]"),
    "Vessot_1980": dict(
        value=7.0e-5,
        what="'agreement of the observed relativistic frequency shift with prediction is at the 70 x 10^-6 level'",
        refs=["Vessot R.F.C. et al. 1980, PRL 45, 2081, doi:10.1103/PhysRevLett.45.2081 (abstract, NASA NTRS 19810032669)"],
        tag="[LIT]"),
    "Will_2014": dict(
        value=2.0e-4,
        what="'Analysis of the data yielded a limit |alpha| < 2 x 10^-4' (GP-A)",
        refs=["Will C.M. 2014, Living Rev. Relativ. 17, 4, doi:10.12942/lrr-2014-4, arXiv:1403.7377, sec. 2.1.3"],
        tag="[LIT]"),
    "Galileo_2018": dict(
        value=2.48e-5,
        what="(0.19 +/- 2.48)e-5 at 1 sigma, eccentric Galileo satellites, 'improving the best previous test by a factor 5.6'",
        refs=["Delva P. et al. 2018, PRL 121, 231101, arXiv:1812.03711"],
        tag="[LIT]"),
    "Cassini_gamma": dict(
        value=2.3e-5,
        what="gamma - 1 = (2.1 +/- 2.3)e-5 (1 sigma), Doppler tracking of Cassini near conjunction, 2002",
        refs=["Bertotti B., Iess L., Tortora P. 2003, Nature 425, 374, doi:10.1038/nature01997",
              "quoted in Will 2014 LRR 17, 4 (arXiv:1403.7377)"],
        tag="[LIT]"),
    "GPA_apogee_m": dict(
        value=1.0e7,
        what="GP-A maser apogee ~10,000 km (Will 2014: 'an altitude of about 10,000 km')",
        refs=["Will C.M. 2014, LRR 17, 4"],
        tag="[LIT]"),
    "IAU2015_GM_sun": dict(
        value=1.3271244e20,
        what="nominal solar mass parameter (GM)^N_sun in m^3 s^-2 (IAU defines GM, not M)",
        refs=["Prsa A. et al. 2016, AJ 152, 41, IAU 2015 Resolution B3, arXiv:1510.07674"],
        tag="[LIT]"),
}
GPA = LIT["GPA_bound"]["value"]
BOUNDS = {"GPA_1p4e-4_prereg": GPA, "Will2014_2e-4": LIT["Will_2014"]["value"],
          "Vessot1980_7e-5": LIT["Vessot_1980"]["value"], "Galileo2018_2p48e-5": LIT["Galileo_2018"]["value"]}
CAS = LIT["Cassini_gamma"]["value"]

results = dict(id="U4", version="v2 (repair after three skeptic reviews)", xi_repo=XI,
               xi_source="sigma_ground.inventory.core.sigma.XI", lit=LIT,
               prereg_file="U4_prereg.json (unchanged; criterion not moved)")
results["lit_crosscheck_5p6_x_galileo"] = S("5.6*2.48e-5")

# ---------------------------------------------------------------- 1. field-equation normalisation, derived (unchanged)
dK, _ = M("differentiate_expr", {"expression": "K/r", "variable": "r"})
Ksol, _ = M("solve_equation", {"equation": f"4*pi*r**2*({dK}) = -8*pi*G*xi*M/c**2", "variable": "K"})
ratio_field_to_inventory, _ = M("simplify_expr", {"expression": f"({Ksol[0]})/(xi*G*M/c**2)"})
ratio_field_to_rs, _ = M("simplify_expr", {"expression": f"({Ksol[0]})/(xi*2*G*M/c**2)"})
results["field_equation_derivation"] = dict(
    statement="static weak field of box sigma = -xi R: lap sigma = -(8 pi G xi/c^2) rho  =>  sigma = K/r outside",
    dsigma_dr=dK, K=Ksol, K_over_inventory_coefficient=ratio_field_to_inventory, K_over_xi_rs=ratio_field_to_rs,
    reading=("Under the adopted reading (MTW sign, covariant box, R = 8 pi G rho/c^2 at weak field) the exterior "
             "solution is sigma = 2 xi GM/(r c^2) = xi r_s/r; sigma_from_potential codes xi GM/(r c^2), a factor "
             + str(ratio_field_to_inventory) + " apart. audit.py:145 says sigma_from_potential is 'Direct from "
             "box sigma = -xi R'. Which side is wrong is decision C2 (Captain's)."),
    tag="[DERIVED] under the adopted reading (design section 2.1)")

# ---------------------------------------------------------------- 2. analytic decision table (series in x = r_s/r)
ser_e, _ = M("series_expansion", {"expression": "exp(k*x/2) - 1", "variable": "x", "point": "0", "order": 3})
ser_g, _ = M("series_expansion", {"expression": "1/sqrt(1-x) - 1", "variable": "x", "point": "0", "order": 3})
ser_d, _ = M("series_expansion", {"expression": "exp(k*x/2) - 1/sqrt(1-x)", "variable": "x", "point": "0", "order": 3})
k_vals = {"field_eq": S(f"2*{XI}"), "inventory": float(XI), "docstring": 1.0}
xi_ceiling_extra_field = M("solve_equation", {"equation": f"2*xi = {GPA}", "variable": "xi"})[0]
xi_ceiling_extra_inv = M("solve_equation", {"equation": f"xi = {GPA}", "variable": "xi"})[0]
xi_window_notation_field = [M("solve_equation", {"equation": f"2*xi - 1 = {s}*{GPA}", "variable": "xi"})[0]
                            for s in (-1, 1)]
xi_window_notation_inv = [M("solve_equation", {"equation": f"xi - 1 = {s}*{GPA}", "variable": "xi"})[0]
                          for s in (-1, 1)]
results["analytic_table"] = dict(
    parametrisation="sigma = k r_s/(2r); k = 2 xi (field equation), xi (inventory), 1 (docstring / GR notation)",
    series_e_sigma_minus_1=ser_e, series_z_GR=ser_g, series_e_sigma_minus_GR_factor=ser_d,
    leading_order="extra-dilation ratio (e^sigma-1)/z_GR = k + O(x);  notation residual |e^sigma-(1+z)|/z_GR = |k-1| + O(x)",
    k=k_vals,
    xi_ceiling_extra_dilation_field_eq=xi_ceiling_extra_field,
    xi_ceiling_extra_dilation_inventory=xi_ceiling_extra_inv,
    xi_window_notation_field_eq=xi_window_notation_field,
    xi_window_notation_inventory=xi_window_notation_inv,
    scope_of_windows=("The notation windows (xi_field = 1/2 +/- 7e-5, xi_inv = 1 +/- 1.4e-4) apply ONLY to the "
                      "replacement-clock-law reading (clock rate set by e^sigma, orbits and hence the measured GM "
                      "following GR). Under universal (metric) notation xi is a rescaling of G and GP-A cannot see "
                      "it (section 3b)."),
    tag="[DERIVED] (Mentat sympy); the outcome is fixed by k and the bound [BY CONSTRUCTION]")

# ---------------------------------------------------------------- 3. numeric table (pre-registered; CLOCK-ONLY reading)
NORMS = [("field_eq", "sigma = xi r_s/R (field equation)"),
         ("inventory", "sigma_from_potential(R,M) = xi GM/Rc^2"),
         ("docstring", "sigma = r_s/2R (docstring 7e-10, no xi)")]
INTERP_LABEL = {
    "extra_dilation": "clock-only extra dilation: clocks tick e^sigma slower on top of GR; orbits (measured GM) follow GR",
    "notation": "clock-only replacement clock law: clock factor IS e^sigma; orbits (measured GM) follow GR "
                "(the pre-registered 'notation' row)"}
rows, leaves = [], {}
for body in ("earth", "sun"):
    bd, src_body = M("solar_system_body", {"body_name": body})
    Mk, R = float(bd["mass_kg"]), float(bd["radius_m"])
    rs, src_rs = M("schwarzschild_radius", {"mass_kg": Mk})
    rs = float(rs)
    zM, src_z = M("gravitational_redshift", {"mass_kg": Mk, "radius_m": R})
    dM, src_d = M("gravitational_time_dilation", {"mass_kg": Mk, "radius_m": R})
    zM, dM = float(zM), float(dM)
    x = rs / R
    dil_eng = time_dilation_gr(Mk, R)
    g_eng = 1.0 / dil_eng
    z_eng = z_gr_engine(Mk, R)
    z_stable = math.expm1(-0.5 * math.log1p(-x))
    leaves[body] = dict(mass_kg=Mk, radius_m=R, body_source=src_body, r_s_m=rs, r_s_source=src_rs,
                        x_rs_over_R=x, R_over_rs=R / rs,
                        z_GR_mentat=zM, z_GR_mentat_source=src_z, dilation_mentat=dM, dilation_mentat_source=src_d,
                        z_GR_engine=z_eng, one_over_dil_engine_minus_1=g_eng - 1.0, z_GR_stable=z_stable,
                        z_rel_diff_mentat_vs_stable=(zM - z_stable) / z_stable,
                        z_rel_diff_engine_vs_stable=(z_eng - z_stable) / z_stable)
    sig = {"field_eq": S(f"{XI}*{rs}/{R}"), "inventory": sigma_from_potential(R, Mk), "docstring": S(f"{rs}/(2*{R})")}
    inv_check = S(f"{XI}*{rs}/(2*{R})")
    leaves[body]["sigma_from_potential_vs_xi_rs_over_2R_reldiff"] = (sig["inventory"] - inv_check) / inv_check
    for key, label in NORMS:
        s = sig[key]
        e_eng = sigma_time_dilation(s, 1.0)
        assert e_eng == scale_ratio(s)
        extra_eng = (e_eng - 1.0) / z_eng
        extra_stab = math.expm1(s) / z_stable
        not_eng = abs(e_eng - g_eng) / z_eng
        not_stab = abs((1.0 + z_stable) * math.expm1(s + 0.5 * math.log1p(-x))) / z_stable
        for interp, m_eng, m_stab in (("extra_dilation", extra_eng, extra_stab), ("notation", not_eng, not_stab)):
            r = dict(body=body, normalisation=key, normalisation_label=label, interpretation=interp,
                     interpretation_label=INTERP_LABEL[interp], coupling="clock-only (pre-registered)",
                     sigma=s, e_sigma_minus_1_engine=e_eng - 1.0, z_GR=z_stable,
                     metric_engine_functions=m_eng, metric_stable=m_stab,
                     status="allowed" if m_stab < GPA else "excluded",
                     status_engine_path="allowed" if m_eng < GPA else "excluded",
                     engine_path_noise_limited=bool(abs(m_eng - m_stab) > 1e-3 * max(m_stab, 1e-300)),
                     tag="[VERIFIED numerics] from Mentat leaves + repo clock functions; bound [LIT]")
            for bname, bval in BOUNDS.items():
                r[f"orders_vs_{bname}"] = math.log10(m_stab / bval)
                r[f"status_vs_{bname}"] = "allowed" if m_stab < bval else "excluded"
            r["orders_vs_GPA"] = r["orders_vs_GPA_1p4e-4_prereg"]
            rows.append(r)
results["leaves"] = leaves
results["table"] = rows

# ---------------------------------------------------------------- 3a. GP-A differential geometry (ground -> 10,000 km), clock-only
bdE, _ = M("solar_system_body", {"body_name": "earth"})
MkE, RE = float(bdE["mass_kg"]), float(bdE["radius_m"])
rsE = float(M("schwarzschild_radius", {"mass_kg": MkE})[0])
h = LIT["GPA_apogee_m"]["value"]
Rh = RE + h
dil_top_M = float(M("gravitational_time_dilation", {"mass_kg": MkE, "radius_m": Rh})[0])


def zst(xx):
    return math.expm1(-0.5 * math.log1p(-xx))


xR, xT = rsE / RE, rsE / Rh
dG = zst(xR) - zst(xT)
diff_rows = []
for key, sfun in (("field_eq", lambda r: XI * rsE / r), ("inventory", lambda r: sigma_from_potential(r, MkE)),
                  ("docstring", lambda r: rsE / (2 * r))):
    dE = math.expm1(sfun(RE)) - math.expm1(sfun(Rh))
    diff_rows.append(dict(normalisation=key, extra_ratio=dE / dG, notation_residual=abs(dE - dG) / dG))
results["gpa_differential_check"] = dict(
    h_m=h, dilation_apogee_mentat=dil_top_M, dilation_apogee_engine=time_dilation_gr(MkE, Rh),
    gr_shift_ground_to_apogee=dG, rows=diff_rows,
    reading="clock-only ratios equal the surface ratios to O(r_s/R): the surface table is the GP-A observable",
    tag="[VERIFIED numerics]")

# ---------------------------------------------------------------- 3b. UNIVERSAL (metric) coupling: the assumption behind the table
# GP-A tests z = (1+alpha) dU/c^2 with dU from the orbit-measured GM. Mentat's Earth mass is that GM/G, so
# x_orb := r_s(Mentat)/r is the orbit-measured compactness. If e^sigma also sets g_tt for orbits, the bare
# compactness x_b differs from x_orb and GP-A compares against the orbit GM.
#   extra (universal):    g_tt = (1 - x_b) e^{-k x_b} = 1 - (1+k) x_b + ...  ->  x_orb = (1+k) x_b
#   notation (universal): g_tt = e^{-k x_b}           = 1 - k x_b + ...      ->  x_orb = k x_b   (G_eff = k G)
gtt_extra, _ = M("series_expansion", {"expression": "(1 - y)*exp(-k*y)", "variable": "y", "point": "0", "order": 2})
gtt_not, _ = M("series_expansion", {"expression": "exp(-k*y)", "variable": "y", "point": "0", "order": 2})
res_extra_ser, _ = M("series_expansion", {
    "expression": "((1 - y/(1+k))**(-1/2)*exp(k*y/(2*(1+k))) - (1 - y)**(-1/2))/((1 - y)**(-1/2) - 1)",
    "variable": "y", "point": "0", "order": 2})
res_not_ser, _ = M("series_expansion", {
    "expression": "(exp(y/2) - (1 - y)**(-1/2))/((1 - y)**(-1/2) - 1)",
    "variable": "y", "point": "0", "order": 2})
res_extra_ser_simpl, _ = M("simplify_expr", {"expression": str(res_extra_ser)})
gam_sol, _ = M("solve_equation", {"equation": "g*(1 + k) = 1", "variable": "g"})   # g = PPN gamma (sympy reserves gamma)
gam_dev, _ = M("simplify_expr", {"expression": f"1 - ({gam_sol[0]})"})
univ = dict(
    assumption=("GP-A measures alpha against the potential computed from the orbit-measured GM. The pre-registered "
                "table assumes sigma acts on clocks only (orbits follow GR). This block asks what happens if e^sigma "
                "also sets g_tt for orbits (universal, metric coupling), keeping g_rr = 1/(1 - x_b) (sigma leaves "
                "the spatial metric alone) for the PPN gamma estimate: g_rr = 1 + 2 gamma U_orb with "
                "U_orb = (1+k) x_b/2 gives gamma = 1/(1+k)."),
    g_tt_series_extra=gtt_extra, g_tt_series_notation=gtt_not,
    gpa_residual_series_extra_vs_orbit_GM=res_extra_ser,
    gpa_residual_series_extra_vs_orbit_GM_simplified=res_extra_ser_simpl,
    gpa_residual_series_notation_vs_orbit_GM=res_not_ser,
    ppn_gamma_extra=gam_sol, one_minus_gamma_extra=gam_dev,
    repo_coding=("The repo codes the clock-only reading: sigma_time_dilation (relativity.py:204-226) has no dynamical "
                 "counterpart, and the nbody hook (field/interface/nbody.py:233) multiplies a body's GM by "
                 "e^{sigma_field} of that body (default 0), which for sigma ~ 1e-10 is a 1 + O(1e-10) change, not "
                 "the 1 + k a metric coupling would need."),
    rows=[])
for key in ("field_eq", "inventory", "docstring"):
    k = k_vals[key]
    for body in ("earth", "sun"):
        xo = leaves[body]["x_rs_over_R"]
        xb = xo / (1.0 + k)
        z_obs = math.expm1(-0.5 * math.log1p(-xb) + 0.5 * k * xb)
        z_pred = zst(xo)
        res_e = abs(z_obs - z_pred) / z_pred
        xbn = xo / k
        z_obs_n = math.expm1(0.5 * k * xbn)
        res_n = abs(z_obs_n - z_pred) / z_pred
        gm1 = k / (1.0 + k)
        univ["rows"].append(dict(
            normalisation=key, body=body, k=k,
            extra_universal_gpa_residual_vs_orbit_GM=res_e,
            extra_universal_gpa_status="allowed (GP-A blind)" if res_e < GPA else "excluded",
            extra_universal_one_minus_gamma=gm1,
            extra_universal_orders_vs_Cassini=math.log10(gm1 / CAS),
            extra_universal_status_vs_Cassini="allowed" if gm1 < CAS else "excluded",
            notation_universal_gpa_residual_vs_orbit_GM=res_n,
            notation_universal_G_eff_over_G=k,
            notation_universal_reading=("xi is a rescaling of G (G_eff = k G); GP-A blind; if g_rr is also GR's with "
                                        "the orbit GM then gamma = 1 [BY CONSTRUCTION]: unobservable (design 2.5-6(c))"),
            tag="[DERIVED] leading PPN order + [VERIFIED numerics]; Cassini bound [LIT]"))
univ["gpa_differential_universal_extra"] = {}
for key in ("field_eq", "inventory"):
    k = k_vals[key]
    xbR, xbT = xR / (1 + k), xT / (1 + k)
    dO = (math.expm1(-0.5 * math.log1p(-xbR) + 0.5 * k * xbR)
          - math.expm1(-0.5 * math.log1p(-xbT) + 0.5 * k * xbT))
    univ["gpa_differential_universal_extra"][key] = abs(dO - dG) / dG
univ["reading"] = ("Under universal coupling GP-A is blind to both xi rows (residuals O(r_s/R) ~ 1e-10). The extra-"
                   "dilation rows are then excluded by Cassini's gamma instead (|gamma - 1| = k/(1+k)), assuming "
                   "sigma leaves g_rr alone. Under universal notation xi is a G rescaling and nothing tested here can "
                   "see it. So 'excluded' survives for the extra-dilation reading either way, but 'GP-A excludes' holds "
                   "only for clock-only coupling, which is what the repo codes.")
results["coupling_universal_vs_clock_only"] = univ

# ---------------------------------------------------------------- 4. horizon row (corrected signs; regular branch; gradient energy)
d1, _ = M("differentiate_expr", {"expression": "A + B*log(1 - rs/r)", "variable": "r"})
flux, _ = M("simplify_expr", {"expression": f"r**2*(1 - rs/r)*({d1})"})
box, _ = M("differentiate_expr", {"expression": str(flux), "variable": "r"})
far, _ = M("compute_limit", {"expression": "r*(B*log(1 - rs/r))", "variable": "r", "point": "oo"})
at_h_Bpos, _ = M("compute_limit", {"expression": "A + log(1 - rs/r)", "variable": "r", "point": "rs", "direction": "+"})
at_h_Bneg, _ = M("compute_limit", {"expression": "A - log(1 - rs/r)", "variable": "r", "point": "rs", "direction": "+"})
at_h_B0, _ = M("compute_limit", {"expression": "A", "variable": "r", "point": "rs", "direction": "+"})
grad_energy, _ = M("simplify_expr", {"expression": f"(1 - rs/r)*({d1})**2"})
grad_unitB, _ = M("simplify_expr", {"expression": f"(({grad_energy}))/B**2"})
grad_at_h, _ = M("compute_limit", {"expression": str(grad_unitB), "variable": "r", "point": "rs", "direction": "+"})
gr_id, _ = M("simplify_expr", {"expression": "exp(-log(1 - rs/r)/2)*sqrt(1 - rs/r)"})
compact_sym, _ = M("simplify_expr", {"expression": "G*Mass/((2*G*Mass/c**2)*c**2)"})
bds, src_sun = M("solar_system_body", {"body_name": "sun"})
Msun = float(bds["mass_kg"])
rs_sun = float(M("schwarzschild_radius", {"mass_kg": Msun})[0])
Gv_raw, Gsrc = M("lookup_constant", {"name": "G"})
cv_raw, csrc = M("lookup_constant", {"name": "c"})
Gv, cv = num(Gv_raw), num(cv_raw)
compact_num_at_rs = S(f"{Gv}*{Msun}/({rs_sun}*{cv}**2)")
unclamped_at_rs = S(f"{XI}*{Gv}*{Msun}/({rs_sun}*{cv}**2)")
eng = {repr(u): sigma_from_potential(u * rs_sun, Msun) for u in (1.001, 1.000001, 1.0, 0.5, 0.1)}
results["horizon_row"] = dict(
    schwarzschild_static_operator="box sigma = (1/r^2) d/dr [ r^2 (1 - r_s/r) dsigma/dr ]  (vacuum: R = 0, so box sigma = 0)",
    trial="sigma = A + B ln(1 - r_s/r)",
    dsigma_dr=d1, r2_g_rr_dsigma_dr=flux, box_sigma_times_r2=box,
    far_field_limit_r_times_sigma_minus_A=far,
    limit_at_horizon={"B = +1": at_h_Bpos, "B = -1": at_h_Bneg, "B = 0 (regular branch)": at_h_B0,
                      "general": "lim sigma (r -> r_s+) = -oo * sign(B) for B != 0; = A for B = 0"},
    gradient_energy_g_rr_sigma_prime_sq=grad_energy, gradient_energy_over_B2=grad_unitB,
    gradient_energy_over_B2_limit_at_horizon=grad_at_h,
    B_matching={"field_eq": "B = -xi (sigma ~ -B r_s/r = xi r_s/r)", "inventory": "B = -xi/2", "notation": "B = -1/2"},
    exact_GR_identity_B_minus_half=gr_id,
    regular_solution=("B = 0: sigma = A = sigma_infinity, finite at r_s with zero gradient energy, independent of M "
                      "and xi (no scalar hair). Every B != 0 branch, including the notation branch B = -1/2, diverges "
                      "logarithmically at r_s and has divergent gradient energy."),
    xi_over_2_attribution=dict(
        compactness_at_r_s_symbolic=compact_sym, compactness_at_r_s_numeric_mentat_G_c=compact_num_at_rs,
        unclamped_xi_GM_over_rc2_at_r_s=unclamped_at_rs,
        engine_sigma_from_potential_at_r_over_rs=eng,
        reading=("At r = r_s the compactness GM/(r c^2) is exactly 1/2, so the clamp at sigma.py:301-302 "
                 "(compactness > 0.5) is inactive there; sigma_from_potential(r_s) = xi/2 = 0.0791 is the weak-field "
                 "1/r formula xi GM/(r c^2) evaluated at r_s, outside its domain (audit.py:161 says exactly this). "
                 "It joins continuously from outside. The clamp only caps sigma at xi/2 for r < r_s (0.0791 at "
                 "0.5 r_s and 0.1 r_s). Neither step follows from box sigma = -xi R.")),
    star_R_over_rs={b: leaves[b]["R_over_rs"] for b in leaves},
    reading=("Static vacuum solutions of box sigma = 0 on Schwarzschild are A + B ln(1 - r_s/r) exactly. The only "
             "solution regular at a horizon is sigma = const = sigma_infinity (no M- or xi-dependence), so box sigma "
             "= -xi R gives no nonzero, xi-dependent horizon value, and 'sigma = xi/2 at any horizon' "
             "(sigma.py:293, relativity.py:214, audit.py:159 'always') has no derivation: withdraw it. For a star "
             "R/r_s >> 1 and the horizon is never reached."),
    side_result=("With B = -1/2, e^sigma = (1 - r_s/r)^(-1/2) exactly: the ln branch that reproduces GR's clock "
                 "factor is an exact vacuum solution, carries no xi, and is the irregular (horizon-divergent) branch."),
    sign_correction_vs_v1=("v1 took the limits of -(sigma - A) and reported +B r_s and +oo*sign(B); for the stated "
                           "trial the limits are -B r_s and -oo*sign(B). The B-matching was already right."),
    tag="[DERIVED] (Mentat sympy) + [VERIFIED numerics] (engine)")

# ---------------------------------------------------------------- 5. amplitude factors; design inconsistency; C3 sensitivity
results["amplitude_factor_for_U5_U6_U12"] = dict(
    reference="U5/U6 code sigma = XI*r_s/r (field-equation reading); U12 uses Delta sigma per km / annual",
    factor={"route (a)/(b), field equation (C2 = 2 xi GM/rc^2)": 1.0,
            "route (a)/(b), inventory (C2 = xi GM/rc^2, sigma_from_potential)": S(f"({XI}/2)/{XI}"),
            "route (c) as design 2.5-6(c) writes it (sigma = -2 xi Phi/c^2 = xi r_s/r)": S(f"(2*{XI}/2)/{XI}"),
            "route (c) as xi-free GR notation (design 2.5-2: sigma = GM/rc^2, no xi)": S(f"1/(2*{XI})")},
    design_inconsistency=("design_synthesis 2.5-2 says the surviving notation reading means 'sigma carries no xi'; "
                          "2.5-6(c) defines route (c) as sigma = -2 xi Phi/c^2 (universal, non-dynamical, "
                          "unobservable), which keeps 2 xi. Both factors are emitted; the choice is the Captain's."),
    note=("Under xi-free notation the QCD-mass coupling must be dropped and sigma is not dynamical; under 2.5-6(c) "
          "the xi stays but, being universal, is a G rescaling (section 3b) and unobservable."))
sig_path = os.path.join(REPO, "sigma_ground", "inventory", "core", "sigma.py")
src = open(sig_path, encoding="utf8").read()
tree = ast.parse(src)
omega_b = omega_c = None
doc_line = None
for i, node in enumerate(tree.body):
    tgt = node.target if isinstance(node, ast.AnnAssign) else (node.targets[0] if isinstance(node, ast.Assign) else None)
    if isinstance(tgt, ast.Name) and tgt.id == "XI" and i + 1 < len(tree.body):
        nxt = tree.body[i + 1]
        if isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Constant) and isinstance(nxt.value.value, str):
            m = re.search(r"Ω_b h² = (\d+\.\d+),\s*Ω_c h² = (\d+\.\d+)", nxt.value.value)
            omega_b, omega_c, doc_line = m.group(1), m.group(2), nxt.lineno
xi_doc = S(f"{omega_b}/({omega_b} + {omega_c})")
results["C3_sensitivity"] = dict(
    omega_b_h2=omega_b, omega_c_h2=omega_c,
    omega_source=(f"parsed with ast from sigma_ground/inventory/core/sigma.py:{doc_line} (the XI docstring, quoting "
                  "Planck 2018); Mentat has no Omega_b h^2 / Omega_c h^2 constant (list_constants 'OMEGA' returns only "
                  "OMEGA_M, OMEGA_LAMBDA). Docstring values, not Mentat-verified."),
    xi_docstring_omegas=xi_doc,
    extra_ratio_field_eq=S(f"2*{xi_doc}"), extra_ratio_inventory=xi_doc,
    reading="0.7 % change in xi; no row changes status")


# ---------------------------------------------------------------- 6. is the composition coded? call-site scan (read-only)
def scan(names, roots):
    hits = {n: [] for n in names}
    for root in roots:
        for dp, dn, fn in os.walk(os.path.join(REPO, root)):
            dn[:] = [d for d in dn if d not in ("__pycache__", ".git", ".claude")]
            for f in fn:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dp, f)
                try:
                    lines = open(p, encoding="utf8", errors="replace").read().splitlines()
                except OSError:
                    continue
                for ln, L in enumerate(lines, 1):
                    for n in names:
                        if re.search(rf"\b{n}\s*\(", L) and not re.search(rf"def\s+{n}\b", L):
                            hits[n].append(f"{os.path.relpath(p, REPO).replace(os.sep, '/')}:{ln}")
    return hits


calls = scan(["sigma_time_dilation", "sigma_from_potential"], ["sigma_ground", "scripts", "tools"])
results["composition_scan"] = dict(
    roots_scanned=["sigma_ground", "scripts", "tools"], call_sites=calls,
    hit_notes=("sigma_ground/field/relativity.py:12 is the module docstring line 'See sigma_time_dilation().', not a "
               "call; scripts/unification/pilot_p6_sigma_bridge.py:59 is an analysis script, not engine code. Neither "
               "function has a production caller."),
    reading=("No engine code composes sigma_time_dilation(sigma_from_potential(r, M)); the tested clock law is the "
             "analyst's composition of two functions, which the relativity.py:214 docstring endorses ('At the event "
             "horizon of a solar-mass BH: sigma ~ xi/2 ~ 0.079, factor ~ 1.08'). sigma_ground/mcp/tools/relativity.py:7-9 "
             "states sigma_time_dilation is deliberately NOT exposed through MCP."),
    tag="[VERIFIED] by source scan")


# ---------------------------------------------------------------- 7. verdict vs pre-registration (criterion unchanged)
def row(body, norm, interp):
    return next(r for r in rows if r["body"] == body and r["normalisation"] == norm and r["interpretation"] == interp)


pred_extra = all(row(b, n, "extra_dilation")["orders_vs_GPA"] >= 3.0 for b in ("earth", "sun") for n in ("field_eq", "inventory"))
survivors = sorted({(r["normalisation"], r["interpretation"]) for r in rows if r["status"] == "allowed"})
only_notation_survives = all(i == "notation" for _, i in survivors) and len(survivors) > 0
margins = {bn: {n: min(row(b, n, "extra_dilation")[f"orders_vs_{bn}"] for b in ("earth", "sun"))
                for n in ("field_eq", "inventory")} for bn in BOUNDS}
results["verdict"] = dict(
    criterion=("pre-registered: allowed iff metric < 1.4e-4 (GP-A); prediction: both xi extra-dilation rows excluded "
               "by >= 3 orders; only notation rows survive"),
    outcome="PASS against the pre-registered criterion",
    prediction_extra_rows_excluded_ge_3_orders=pred_extra,
    survivors=survivors, only_notation_survives=only_notation_survives,
    survivor_is_xi_free_only=all(n == "docstring" for n, _ in survivors),
    min_orders_extra_xi_rows=min(row(b, n, "extra_dilation")["orders_vs_GPA"] for b in ("earth", "sun") for n in ("field_eq", "inventory")),
    margins_extra_xi_rows_by_bound=margins,
    margin_note=("The '>= 3 orders' wording holds against the pre-registered 1.4e-4 (inventory 3.05). Against Will 2014's "
                 "|alpha| < 2e-4 the inventory row is 2.90 orders: still excluded, but under the '>= 3 orders' wording. "
                 "The PASS stands only against the pre-registered 1.4e-4."),
    against_original_design_metric=("applying |e^sigma - 1|/z_GR literally to every row: nothing survives, so 'only "
                                    "notation survives' fails; the notation metric |e^sigma - (1+z)|/z_GR was declared "
                                    "in U4_prereg.json before the run"),
    coupling_qualifier=("'GP-A excludes' holds for clock-only coupling (what the repo codes). Under universal coupling "
                        "GP-A is blind; the extra-dilation rows are then excluded by Cassini gamma instead and the "
                        "notation rows are a G rescaling (section 3b)."),
    by_construction=("Given xi = 0.1582 and the [LIT] bound, the extra ratio is k + O(r_s/R) and the outcome could not "
                     "differ [BY CONSTRUCTION]; it is a consistency check of the repo's formulas, not evidence about "
                     "the unification thesis."),
    code_implements_notation=False,
    code_note=("sigma_time_dilation (relativity.py:204-226) multiplies by e^sigma with sigma supplied by the caller; "
               "time_dilation_gr (gr_basics.py:64-80) is sqrt(1 - r_s/r); no function maps one to the other, no code "
               "composes sigma_time_dilation with sigma_from_potential, and sigma_from_potential carries xi."))


# ---------------------------------------------------------------- 8. flags: engine redshift precision; Sun-mass leaf provenance
def st(r):
    return math.expm1(-0.5 * math.log1p(-rsE / r))


prec = []
for hh in (1.0, 0.01, 1000.0):
    e = z_gr_engine(MkE, RE) - z_gr_engine(MkE, RE + hh)
    s_ = st(RE) - st(RE + hh)
    prec.append(dict(height_m=hh, engine_delta_z=e, stable_delta_z=s_, rel_error=(e - s_) / s_))
results["flag_redshift_precision"] = dict(
    file="sigma_ground/field/gr_basics.py", line=61, code="return 1.0 / math.sqrt(1.0 - rs / r) - 1.0",
    surface_z_rel_error_earth=leaves["earth"]["z_rel_diff_engine_vs_stable"],
    height_differences=prec,
    reading=("catastrophic cancellation at weak field: Earth-surface z carries a 9e-8 relative error; the clock shift "
             "between heights 1 m or 1 cm apart is exactly 0; at 1 km the error is ~1.7e-4 relative. Does not "
             "affect the U4 table (stable path used); matters for U12 clock rows. Mentat flags, never fixes."),
    tag="[VERIFIED numerics]")
GM_N = LIT["IAU2015_GM_sun"]["value"]
G_implied = S(f"{GM_N}/{Msun}")
M_from_GMN = S(f"{GM_N}/{Gv}")
results["flag_sun_mass_leaf"] = dict(
    file="sigma_ground/mcp/tools/astronomy.py", line=20,
    mentat_mass_kg=Msun, mentat_source=src_sun,
    iau2015_GM_N=GM_N, G_mentat=Gv, G_source=Gsrc,
    G_implied_by_mentat_mass=G_implied, M_from_GM_N_and_mentat_G=M_from_GMN,
    rel_diff=(Msun - M_from_GMN) / M_from_GMN,
    reading=("The Sun entry is tagged 'IAU 2015 nominal solar values' (astronomy.py:24), but IAU 2015 B3 defines GM, "
             "not M. The kg value equals GM_N / G_implied with an older G, and differs from GM_N / G_Mentat by "
             "rel_diff. No effect on U4 (ratios depend on M only at O(r_s/R)). Flag only."),
    tag="[VERIFIED numerics] + [LIT]")

results["mentat_calls"] = LOG
results["n_mentat_calls"] = len(LOG)
with open(os.path.join(OUT, "U4_results_v2.json"), "w", encoding="utf8") as f:
    json.dump(results, f, indent=1, ensure_ascii=False, default=str)

# ---------------------------------------------------------------- console summary
print(f"xi = {XI}; field-eq K = {Ksol}; K/inventory = {ratio_field_to_inventory}")
for r in rows:
    print(f"{r['body']:6s} {r['normalisation']:10s} {r['interpretation']:15s} metric={r['metric_stable']:.4e} "
          f"(engine {r['metric_engine_functions']:.4e}) orders: GPA {r['orders_vs_GPA']:+.3f} "
          f"Will {r['orders_vs_Will2014_2e-4']:+.3f} V80 {r['orders_vs_Vessot1980_7e-5']:+.3f} "
          f"Gal {r['orders_vs_Galileo2018_2p48e-5']:+.3f} {r['status']}")
print("GP-A differential:", diff_rows)
print("universal:", gtt_extra, "|", gtt_not, "|", res_extra_ser_simpl, "|", res_not_ser, "|", gam_sol, gam_dev)
for u in univ["rows"]:
    print("  ", u["normalisation"], u["body"], f"extra res {u['extra_universal_gpa_residual_vs_orbit_GM']:.3e}",
          f"1-gamma {u['extra_universal_one_minus_gamma']:.4f} ({u['extra_universal_orders_vs_Cassini']:+.2f})",
          f"notation res {u['notation_universal_gpa_residual_vs_orbit_GM']:.3e}")
print("  diff universal:", univ["gpa_differential_universal_extra"])
hr = results["horizon_row"]
print("horizon:", hr["dsigma_dr"], "|", hr["r2_g_rr_dsigma_dr"], "|", hr["box_sigma_times_r2"], "|", far, "|",
      hr["limit_at_horizon"], "|", grad_energy, "|", grad_unitB, grad_at_h, "|", gr_id)
print("xi/2 attribution:", compact_sym, compact_num_at_rs, unclamped_at_rs, eng)
print("amplitude:", results["amplitude_factor_for_U5_U6_U12"]["factor"])
print("C3:", omega_b, omega_c, doc_line, xi_doc)
print("calls:", calls)
print("verdict:", {k: v for k, v in results["verdict"].items() if k in (
    "prediction_extra_rows_excluded_ge_3_orders", "survivors", "only_notation_survives",
    "min_orders_extra_xi_rows", "margins_extra_xi_rows_by_bound")})
print("precision:", prec)
print("sun flag:", G_implied, M_from_GMN, results["flag_sun_mass_leaf"]["rel_diff"])
print("n calls", len(LOG))
