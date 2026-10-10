#!/usr/bin/env python
"""U4 - factor-2 and the two clock laws: sigma-normalisation decision table from Mentat leaves.

Feeds decision C2 (misc/MENTAT_GAPS_UNIFICATION.md). Computes the table; decides nothing.
Pre-registration: U4_prereg.json in this folder (written before this script was run).

Run from the repo root:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -X utf8 <this file>

Read-only with respect to the repo. Every physical leaf is a Mentat call (logged with its source)
or a repo engine function (sigma_ground.*, which is Mentat too). Literature numbers are tagged [LIT].
"""
import sys, os, json, math

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


# ---------------------------------------------------------------- literature bounds [LIT]
LIT = {
    "GPA_bound": dict(
        value=1.4e-4,
        what="fractional deviation of the gravitational redshift from GR, Gravity Probe A (1976 flight)",
        refs=["Vessot R.F.C. et al. 1980, PRL 45, 2081, doi:10.1103/PhysRevLett.45.2081",
              "Delva P. et al. 2018, PRL 121, 231101, doi:10.1103/PhysRevLett.121.231101, arXiv:1812.03711 "
              "(states GP-A verified the redshift to 1.4e-4 and improves it by a factor 5.6)"],
        tag="[LIT]"),
    "Galileo_2018": dict(
        value=2.48e-5,
        what="(0.19 +/- 2.48)e-5, 1 sigma, eccentric Galileo satellites (secondary column, not the criterion)",
        refs=["Delva P. et al. 2018, PRL 121, 231101, arXiv:1812.03711"],
        tag="[LIT]"),
    "GPA_apogee_m": dict(
        value=1.0e7,
        what="GP-A maser apogee ~10,000 km (used only to illustrate the differential redshift; ratio is h-independent at leading order)",
        refs=["Vessot et al. 1980 (as summarised in arXiv:1009.2485 and the PRL record)"],
        tag="[LIT]"),
}
GPA = LIT["GPA_bound"]["value"]
GAL = LIT["Galileo_2018"]["value"]

results = dict(id="U4", xi_repo=XI, xi_source="sigma_ground.inventory.core.sigma.XI", lit=LIT)

# cross-check of the [LIT] pair: 5.6 x 2.48e-5 should reproduce ~1.4e-4
results["lit_crosscheck_5p6_x_galileo"] = S("5.6*2.48e-5")

# ---------------------------------------------------------------- 1. field-equation normalisation, derived
# Static weak field of box(sigma) = -xi R (MTW sign, R = 8 pi G rho / c^2):  lap(sigma) = -(8 pi G xi / c^2) rho.
# Exterior sigma = K/r; Gauss: 4 pi r^2 dsigma/dr = -(8 pi G xi / c^2) M.
dK, srcd = M("differentiate_expr", {"expression": "K/r", "variable": "r"})
Ksol, srcs = M("solve_equation", {"equation": f"4*pi*r**2*({dK}) = -8*pi*G*xi*M/c**2", "variable": "K"})
ratio_field_to_inventory, _ = M("simplify_expr", {"expression": f"({Ksol[0]})/(xi*G*M/c**2)"})
ratio_field_to_rs, _ = M("simplify_expr", {"expression": f"({Ksol[0]})/(xi*2*G*M/c**2)"})
results["field_equation_derivation"] = dict(
    statement="static weak field of box sigma = -xi R: lap sigma = -(8 pi G xi/c^2) rho  =>  sigma = K/r outside",
    dsigma_dr=dK, K=Ksol, K_over_inventory_coefficient=ratio_field_to_inventory,
    K_over_xi_rs=ratio_field_to_rs,
    reading="sigma_field = 2 xi G M/(r c^2) = xi r_s/r; sigma_from_potential codes xi G M/(r c^2): factor "
            + str(ratio_field_to_inventory),
    tag="[DERIVED] under the adopted reading (MTW sign, R = 8 pi G rho/c^2 at weak field; design section 2.1)")

# ---------------------------------------------------------------- 2. analytic decision table (series in x = r_s/r)
# Write every normalisation as sigma = k * r_s/(2r):  k = 2 xi (field eq), xi (inventory), 1 (docstring).
ser_e, s1 = M("series_expansion", {"expression": "exp(k*x/2) - 1", "variable": "x", "point": "0", "order": 3})
ser_g, s2 = M("series_expansion", {"expression": "1/sqrt(1-x) - 1", "variable": "x", "point": "0", "order": 3})
ser_d, s3 = M("series_expansion", {"expression": "exp(k*x/2) - 1/sqrt(1-x)", "variable": "x", "point": "0", "order": 3})
k_vals = {"field_eq": S(f"2*{XI}"), "inventory": float(XI), "docstring": 1.0}
# xi ceilings / windows implied by GP-A at leading order
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
    tag="[DERIVED] (Mentat sympy)")

# ---------------------------------------------------------------- 3. numeric table from Mentat leaves
NORMS = [
    ("field_eq", "sigma = xi r_s/R (field equation)"),
    ("inventory", "sigma_from_potential(R,M) = xi GM/Rc^2"),
    ("docstring", "sigma = r_s/2R (docstring 7e-10, no xi)"),
]
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
    # engine GR clock
    dil_eng = time_dilation_gr(Mk, R)
    g_eng = 1.0 / dil_eng                       # coordinate time per unit proper time (= 1 + z)
    z_eng = z_gr_engine(Mk, R)
    z_stable = math.expm1(-0.5 * math.log1p(-x))  # numerically stable 1/sqrt(1-x) - 1
    leaves[body] = dict(mass_kg=Mk, radius_m=R, body_source=src_body, r_s_m=rs, r_s_source=src_rs,
                        x_rs_over_R=x, R_over_rs=R / rs,
                        z_GR_mentat=zM, z_GR_mentat_source=src_z,
                        dilation_mentat=dM, dilation_mentat_source=src_d,
                        z_GR_engine=z_eng, one_over_dil_engine_minus_1=g_eng - 1.0,
                        z_GR_stable=z_stable,
                        z_rel_diff_mentat_vs_stable=(zM - z_stable) / z_stable,
                        z_rel_diff_engine_vs_stable=(z_eng - z_stable) / z_stable)
    sig = {
        "field_eq": S(f"{XI}*{rs}/{R}"),
        "inventory": sigma_from_potential(R, Mk),
        "docstring": S(f"{rs}/(2*{R})"),
    }
    inv_check = S(f"{XI}*{rs}/(2*{R})")
    leaves[body]["sigma_from_potential_vs_xi_rs_over_2R_reldiff"] = (sig["inventory"] - inv_check) / inv_check
    for key, label in NORMS:
        s = sig[key]
        e_eng = sigma_time_dilation(s, 1.0)     # = t0 e^sigma, t0 = 1
        assert e_eng == scale_ratio(s)
        # extra dilation on top of GR
        extra_eng = (e_eng - 1.0) / z_eng
        extra_stab = math.expm1(s) / z_stable
        # e^sigma IS the GR factor (notation)
        not_eng = abs(e_eng - g_eng) / z_eng
        not_stab = abs((1.0 + z_stable) * math.expm1(s + 0.5 * math.log1p(-x))) / z_stable
        for interp, m_eng, m_stab in (("extra_dilation", extra_eng, extra_stab),
                                       ("notation", not_eng, not_stab)):
            rows.append(dict(
                body=body, normalisation=key, normalisation_label=label, interpretation=interp,
                sigma=s, e_sigma_minus_1_engine=e_eng - 1.0, z_GR=z_stable,
                metric_engine_functions=m_eng, metric_stable=m_stab,
                status="allowed" if m_stab < GPA else "excluded",
                status_engine_path="allowed" if m_eng < GPA else "excluded",
                orders_vs_GPA=math.log10(m_stab / GPA),
                orders_vs_Galileo2018=math.log10(m_stab / GAL),
                status_vs_Galileo2018="allowed" if m_stab < GAL else "excluded",
                engine_path_noise_limited=bool(abs(m_eng - m_stab) > 1e-3 * max(m_stab, 1e-300)),
                tag="[VERIFIED numerics] from Mentat leaves + repo clock functions; bound [LIT]"))

results["leaves"] = leaves
results["table"] = rows

# ---------------------------------------------------------------- 4. GP-A is a differential measurement: check ratio is r-independent
bd, _ = M("solar_system_body", {"body_name": "earth"})
Mk, R = float(bd["mass_kg"]), float(bd["radius_m"])
rs = float(M("schwarzschild_radius", {"mass_kg": Mk})[0])
h = LIT["GPA_apogee_m"]["value"]
Rh = R + h
dil_top_M = float(M("gravitational_time_dilation", {"mass_kg": Mk, "radius_m": Rh})[0])
xR, xT = rs / R, rs / Rh
dG = math.expm1(-0.5 * math.log1p(-xR)) - math.expm1(-0.5 * math.log1p(-xT))   # GR shift ground-apogee
diff_rows = []
for key, sfun in (("field_eq", lambda r: XI * rs / r),
                  ("inventory", lambda r: sigma_from_potential(r, Mk)),
                  ("docstring", lambda r: rs / (2 * r))):
    dE = math.expm1(sfun(R)) - math.expm1(sfun(Rh))
    diff_rows.append(dict(normalisation=key, extra_ratio=dE / dG, notation_residual=abs(dE - dG) / dG))
results["gpa_differential_check"] = dict(
    h_m=h, dilation_apogee_mentat=dil_top_M, dilation_apogee_engine=time_dilation_gr(Mk, Rh),
    gr_shift_ground_to_apogee=dG, rows=diff_rows,
    reading="ratios equal the surface ratios to O(r_s/R): the surface table is the GP-A observable",
    tag="[VERIFIED numerics]")

# ---------------------------------------------------------------- 5. horizon row
d1, _ = M("differentiate_expr", {"expression": "B*log(1 - rs/r)", "variable": "r"})
flux, _ = M("simplify_expr", {"expression": f"r**2*(1 - rs/r)*({d1})"})
box, _ = M("differentiate_expr", {"expression": str(flux), "variable": "r"})
far, _ = M("compute_limit", {"expression": "r*(-B*log(1 - rs/r))", "variable": "r", "point": "oo"})
at_h, _ = M("compute_limit", {"expression": "-B*log(1 - rs/r)", "variable": "r", "point": "rs", "direction": "+"})
gr_id, _ = M("simplify_expr", {"expression": "exp(-log(1 - rs/r)/2)*sqrt(1 - rs/r)"})
Msun = float(M("solar_system_body", {"body_name": "sun"})[0]["mass_kg"])
rs_sun = float(M("schwarzschild_radius", {"mass_kg": Msun})[0])
results["horizon_row"] = dict(
    schwarzschild_static_operator="box sigma = (1/r^2) d/dr [ r^2 (1 - r_s/r) dsigma/dr ]  (vacuum: R = 0, so box sigma = 0)",
    trial="sigma = A + B ln(1 - r_s/r)",
    dsigma_dr=d1, r2_g_rr_dsigma_dr=flux, box_sigma_times_r2=box,
    far_field_limit_r_sigma=far, limit_at_horizon=at_h,
    B_matching={"field_eq": "B = -xi (sigma ~ xi r_s/r)", "inventory": "B = -xi/2", "notation": "B = -1/2"},
    exact_GR_identity_B_minus_half=gr_id,
    linear_formula_at_r_eq_rs={"field_eq": XI, "inventory": sigma_from_potential(rs_sun, Msun),
                               "inventory_at_half_rs_clamped": sigma_from_potential(0.5 * rs_sun, Msun),
                               "docstring": S(f"{rs_sun}/(2*{rs_sun})")},
    star_R_over_rs={b: leaves[b]["R_over_rs"] for b in leaves},
    reading=("Static vacuum solutions of box sigma = 0 on Schwarzschild are A + B ln(1 - r_s/r) exactly. "
             "Regular at the horizon => B = 0 => sigma = const (no scalar hair). B != 0 diverges logarithmically "
             "at r_s. For a star R/r_s >> 1 and the horizon is never reached. The values xi, xi/2, 1/2 at r = r_s "
             "come only from evaluating the weak-field 1/r formula outside its domain; the inventory's xi/2 is "
             "enforced by the compactness clamp at 0.5 (sigma.py:301-302), not derived. 'sigma = xi/2 at any "
             "horizon' has no derivation from box sigma = -xi R."),
    side_result=("With B = -1/2, e^sigma = (1 - r_s/r)^(-1/2) exactly: the 'notation' reading is an exact vacuum "
                 "solution and carries no xi."),
    tag="[DERIVED] (Mentat sympy)")

# ---------------------------------------------------------------- 6. amplitude factor handed to U5/U6/U12, C3 sensitivity
xi_doc = S("0.02237/(0.02237 + 0.1200)")    # the sigma.py docstring's own Omega_b h^2, Omega_c h^2
results["amplitude_factor_for_U5_U6_U12"] = dict(
    reference="U5/U6 code sigma = XI*r_s/r (field-equation reading); U12 uses Delta sigma per km / annual",
    factor={"field_eq (C2 = 2 xi GM/rc^2)": 1.0,
            "inventory (C2 = xi GM/rc^2, sigma_from_potential)": S(f"({XI}/2)/{XI}"),
            "notation (sigma = GM/rc^2, no xi; route (c))": S(f"1/(2*{XI})")},
    note="Under 'notation' sigma carries no xi, the QCD-mass coupling must be dropped, and sigma is not dynamical.")
results["C3_sensitivity"] = dict(
    xi_docstring_omegas=xi_doc,
    extra_ratio_field_eq=S(f"2*{xi_doc}"), extra_ratio_inventory=xi_doc,
    reading="0.7 % change in xi; no row changes status")

# ---------------------------------------------------------------- 7. verdict vs pre-registration
def row(body, norm, interp):
    return next(r for r in rows if r["body"] == body and r["normalisation"] == norm and r["interpretation"] == interp)

pred_extra = all(row(b, n, "extra_dilation")["orders_vs_GPA"] >= 3.0
                 for b in ("earth", "sun") for n in ("field_eq", "inventory"))
survivors = sorted({(r["normalisation"], r["interpretation"]) for r in rows if r["status"] == "allowed"})
only_notation_survives = all(i == "notation" for _, i in survivors) and len(survivors) > 0
results["verdict"] = dict(
    prediction_extra_rows_excluded_ge_3_orders=pred_extra,
    survivors=survivors,
    only_notation_survives=only_notation_survives,
    min_orders_extra_xi_rows=min(row(b, n, "extra_dilation")["orders_vs_GPA"]
                                 for b in ("earth", "sun") for n in ("field_eq", "inventory")),
    code_implements_notation=False,
    code_note=("sigma_time_dilation (relativity.py:204-226) multiplies by e^sigma with sigma supplied by the "
               "caller; time_dilation_gr (gr_basics.py:64-80) is sqrt(1 - r_s/r); no function maps one to the other "
               "and sigma_from_potential carries xi, so the only surviving reading is not coded."))
# ---------------------------------------------------------------- 8. flag: weak-field precision of the engine redshift
bd, _ = M("solar_system_body", {"body_name": "earth"})
Mk, R = float(bd["mass_kg"]), float(bd["radius_m"])
rs = float(M("schwarzschild_radius", {"mass_kg": Mk})[0])
st = lambda r: math.expm1(-0.5 * math.log1p(-rs / r))
prec = []
for hh in (1.0, 0.01):
    e = z_gr_engine(Mk, R) - z_gr_engine(Mk, R + hh)
    s_ = st(R) - st(R + hh)
    prec.append(dict(height_m=hh, engine_delta_z=e, stable_delta_z=s_, rel_error=(e - s_) / s_))
results["flag_redshift_precision"] = dict(
    file="sigma_ground/field/gr_basics.py", line=61,
    code="return 1.0 / math.sqrt(1.0 - rs / r) - 1.0",
    surface_z_rel_error_earth=leaves["earth"]["z_rel_diff_engine_vs_stable"],
    height_differences=prec,
    reading=("catastrophic cancellation at weak field: z(Earth surface) carries a 9e-8 relative error, and the "
             "clock shift between two heights 1 m or 1 cm apart comes out exactly 0 (true 1.09e-16 / 1.09e-18). "
             "Does not affect the U4 table (stable path used for decisions); matters for any optical-clock "
             "height comparison (U12 clock row). Mentat flags, never fixes."),
    tag="[VERIFIED numerics]")

results["mentat_calls"] = LOG
results["n_mentat_calls"] = len(LOG)

with open(os.path.join(OUT, "U4_results.json"), "w", encoding="utf8") as f:
    json.dump(results, f, indent=1, ensure_ascii=False, default=str)

# ---------------------------------------------------------------- console summary
print(f"xi = {XI}; field-eq K = {Ksol}; K/inventory = {ratio_field_to_inventory}")
print("series:", ser_d)
for r in rows:
    print(f"{r['body']:6s} {r['normalisation']:10s} {r['interpretation']:15s} sigma={r['sigma']:.4e} "
          f"metric={r['metric_stable']:.4e} (engine {r['metric_engine_functions']:.4e}) "
          f"orders={r['orders_vs_GPA']:+.2f} {r['status']}")
print("GP-A differential:", diff_rows)
print("horizon:", results["horizon_row"]["box_sigma_times_r2"], far, at_h, gr_id,
      results["horizon_row"]["linear_formula_at_r_eq_rs"])
print("verdict:", results["verdict"])
print("amplitude:", results["amplitude_factor_for_U5_U6_U12"]["factor"])
