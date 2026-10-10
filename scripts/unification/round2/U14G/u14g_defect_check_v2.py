"""U14G v2 -- repair run of the defect check of two suspected defects in sigma_ground/dynamics/continuum.py.

D1  _buoyancy_force uses the LINEAR expansion coefficient alpha_L where the
    Boussinesq density perturbation needs the VOLUMETRIC beta = 3 alpha_L.
D2  _heat_conduction divides the SPH Fourier sum by rho_i * c_p although the sum
    is already du_i/dt (W/kg): an extra 1/rho_i.

v2 re-runs every pre-registered measurement of v1 unchanged (same criteria, prereg.txt) and adds
the checks the three skeptics asked for:
  D1-LIT  the comparison that CAN fail: engine buoyancy / (3 alpha_L[LIT] dT g), alpha_L from the
          literature (Wikipedia 'Thermal expansion' table at 20 C; Engineering ToolBox as second source).
  D1-CF   counterfactual control, IN-PROCESS ONLY (no file is edited): the cascade function is replaced
          in memory by 3x itself (a 'volumetric' cascade). The v1 ratio R1 stays at 1/3 (so R1 is by
          construction) while the LIT ratio moves to ~1 (so the LIT test could have failed).
  D1-EOS  the Boussinesq term is also applied under eos='ideal' (lone parcel, carrier material's alpha_L).
  D2      copper lattice; two-parcel test in both orientations; two-parcel end to end through
          continuum_step; the k h convergence with the INDEPENDENT eq.-17 reference at every k h;
          the long-wavelength lattice moment with an independent kernel; realised diffusivities.

Read-only against the repo. Run from the repo root:
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -X utf8 <this file>
"""
import sys, os, math, json, time

REPO = r"D:/Aaron/development/sigma-ground"
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO + "/tools")
sys.path.insert(0, REPO)

from mentat_client import mentat                                    # PLATINUM 21
from sigma_ground.dynamics.vec import Vec3
from sigma_ground.dynamics.continuum import (
    ContinuumParcel, ContinuumScene, continuum_step,
    _buoyancy_force, _heat_conduction, _sph_density_sum, material_properties,
)
from sigma_ground.dynamics.fluid.kernel import smoothing_length
import sigma_ground.field.interface.thermal_expansion as te
from sigma_ground.field.interface.thermal_expansion import (
    linear_expansion_coefficient, volumetric_expansion_coefficient,
)
import sigma_ground.field.interface.gas as gasmod

t_start = time.time()
sources = {}

# ── Mentat leaves ──────────────────────────────────────────────────────────
earth, src = mentat("solar_system_body", {"body_name": "earth"})
g = float(earth["surface_g_ms2"]); sources["g_surface_earth"] = {"value": g, "source": src,
                                                                  "tool": "solar_system_body(earth).surface_g_ms2",
                                                                  "note": "9.80665 is standard gravity; the label says NASA fact sheet. "
                                                                          "Every buoyancy ratio here is independent of g."}
for m in ("iron", "copper"):
    v, s = mentat("density", {"material": m})
    sources[f"rho_{m}_mentat"] = {"value": v, "source": s, "tool": f"density({m})"}
fe_atom, s = mentat("element_atomic_data", {"element_symbol": "Fe"})
sources["Fe_atomic_mass"] = {"value": fe_atom["atomic_mass_amu"], "source": s, "tool": "element_atomic_data(Fe)"}
sources["engine_cascade"] = {
    "source": "sigma_ground.dynamics.continuum.material_properties -> field.interface "
              "(thermal_expansion.expansion_coefficient_at_T, thermal.thermal_conductivity, "
              "thermal.specific_heat_j_kg_K, acoustics.density_at_sigma); field.interface.gas "
              "(molecular_mass_kg, heat_capacity_ratio); repo HEAD 339c4f4",
    "note": "repo engines are Mentat (run rule 3)"}

# ── [LIT] leaves (verified by web fetch 2026-10-10) ────────────────────────
LIT = {
    "alpha_L_wikipedia_1e-6_per_K": {"iron": 11.8, "copper": 17, "aluminum": 23.1, "gold": 14, "silicon": 2.56,
                                     "tungsten": 4.5, "nickel": 13, "silver": 18, "lead": 29, "platinum": 9},
    "alpha_V_wikipedia_1e-6_per_K": {"iron": 35.4, "copper": 51, "aluminum": 69, "gold": 42, "silicon": 9,
                                     "tungsten": 13.5, "nickel": 39, "silver": 54, "lead": 87, "platinum": 27},
    "alpha_L_engineeringtoolbox_1e-6_per_K": {"iron": [12.0, 12.0], "copper": [16.0, 16.7], "aluminum": [23.0, 23.0],
                                              "gold": [14.2, 14.2], "silicon": [3.0, 5.0], "tungsten": [4.5, 4.5],
                                              "nickel": [13.0, 13.0], "silver": [19.0, 19.7], "lead": [29.0, 29.0],
                                              "platinum": [9.0, 9.0]},
    "kappa_iron_300K_W_mK": [80.2, 80.4],
    "kappa_iron_400K_W_mK_engineeringtoolbox": 70.0,
    "kappa_iron_300K_W_mK_engineeringtoolbox": 80.0,
    "D_iron_m2_s": 2.3e-5,
    "citations": {
        "alpha": "Wikipedia 'Thermal expansion', table 'Thermal expansion coefficients at 20 C' (linear alpha and "
                 "volumetric alpha_V columns), https://en.wikipedia.org/wiki/Thermal_expansion, fetched 2026-10-10. "
                 "Second source: Engineering ToolBox 'Linear Thermal Expansion Coefficients' (mostly 25 C), "
                 "https://www.engineeringtoolbox.com/linear-expansion-coefficients-d_95.html, fetched 2026-10-10.",
        "kappa_iron": "Wikipedia 'Thermal conductivities of the elements (data page)', Fe: 0.802 W/(cm K), 80.4 and 80 "
                      "W/(m K) (CRC / Lange / Ho-Powell-Liley), fetched 2026-10-10; Engineering ToolBox 'Iron - density, "
                      "specific heats and thermal conductivities vs temperature': 80 W/m/K at 300 K, 70 at 400 K.",
        "D_iron": "Wikipedia 'Thermal diffusivity' table: iron 23 mm^2/s (citing J. Wilson, 'Materials Data', "
                  "Electronics Cooling, Aug 2007), fetched 2026-10-10.",
    },
}

T0 = 300.0
GVEC = Vec3(0.0, -g, 0.0)

# ══════════════════════════════════════════════════════════════════════════
# D1 -- Boussinesq buoyancy
# ══════════════════════════════════════════════════════════════════════════
CUBIC = ["iron", "copper", "aluminum", "gold", "silicon", "tungsten",
         "nickel", "silver", "lead", "platinum"]
DTS = [1.0, 10.0, -10.0, 100.0]
ORIG_ALPHA = te.expansion_coefficient_at_T


def int_alpha(mat, Ta, Tb, n=400):
    """Composite Simpson of the CURRENT cascade alpha(T) (looked up on the module, so the
    in-process counterfactual is seen) from Ta to Tb (n even)."""
    f = te.expansion_coefficient_at_T
    hstep = (Tb - Ta) / n
    s = f(mat, Ta) + f(mat, Tb)
    for k in range(1, n):
        s += (4 if k % 2 else 2) * f(mat, Ta + k * hstep)
    return s * hstep / 3.0


def lone_step_accel_y(mat, T, dt=1e-3, **scene_kw):
    p = ContinuumParcel(mat, radius=0.01, temperature=T, position=Vec3(0, 0, 0))
    sc = ContinuumScene([p], gravity=GVEC, ground=False, reference_temperature=T0, **scene_kw)
    continuum_step(sc, dt)
    return p.velocity.y / dt, p.temperature


def d1_table(tag):
    rows = []
    for mat in CUBIC:
        vol_ratio_repo = volumetric_expansion_coefficient(mat) / linear_expansion_coefficient(mat)
        a_neutral, _ = lone_step_accel_y(mat, T0)
        aL = LIT["alpha_L_wikipedia_1e-6_per_K"][mat] * 1e-6
        aV = LIT["alpha_V_wikipedia_1e-6_per_K"][mat] * 1e-6
        etb = LIT["alpha_L_engineeringtoolbox_1e-6_per_K"][mat]
        for dT in DTS:
            T = T0 + dT
            p = ContinuumParcel(mat, radius=0.01, temperature=T)
            p.update_properties()
            alpha_T = p.thermal_expansion
            a_direct = _buoyancy_force(p, p.props["density_kg_m3"], T0, GVEC).y
            I = int_alpha(mat, T0, T)
            rho_ratio = math.exp(-3.0 * I)                      # rho(T)/rho(T0) from 3 x cascade alpha
            a_exact = -g * (1.0 - 1.0 / rho_ratio)
            a_bous = 3.0 * alpha_T * dT * g
            a_hot, T_after = lone_step_accel_y(mat, T)
            a_e2e = a_hot - a_neutral
            rows.append({
                "cascade": tag, "material": mat, "dT_K": dT, "alpha_at_T_1_K": alpha_T,
                "repo_volumetric_over_linear": vol_ratio_repo,
                "a_engine_direct_m_s2": a_direct, "a_engine_end_to_end_m_s2": a_e2e,
                "a_exact_archimedes_from_3x_cascade_m_s2": a_exact, "a_boussinesq_3x_cascade_m_s2": a_bous,
                "R1_direct_over_exact": a_direct / a_exact,
                "R1_e2e_over_exact": a_e2e / a_exact,
                "R1_direct_over_boussinesq3a": a_direct / a_bous,
                "engine_matches_minus_alpha_dT_g": abs(a_direct - alpha_T * dT * g) <= 1e-12 * abs(alpha_T * dT * g) + 1e-300,
                # the comparisons that can fail: literature coefficients
                "R_LIT_e2e_over_3alphaL_wiki": a_e2e / (3.0 * aL * dT * g),
                "R_LIT_e2e_over_alphaV_wiki": a_e2e / (aV * dT * g),
                "R_LIT_e2e_over_3alphaL_etb_range": sorted([a_e2e / (3.0 * etb[0] * 1e-6 * dT * g),
                                                            a_e2e / (3.0 * etb[1] * 1e-6 * dT * g)]),
                "cascade_alpha300_over_LIT_alphaL_wiki": te.expansion_coefficient_at_T(mat, T0) / aL,
                "T_after_step_K": T_after,
            })
    return rows


def rng(vals):
    return [min(vals), max(vals)]


d1_rows = d1_table("as-coded")

# -- original pre-registered criterion (unchanged) --
small = [r for r in d1_rows if abs(r["dT_K"]) <= 10.0]
R1s = [r["R1_direct_over_exact"] for r in small] + [r["R1_e2e_over_exact"] for r in small]
if all(0.300 <= x <= 0.367 for x in R1s):
    d1_verdict_orig = "DEFECT-CONFIRMED"
elif all(0.9 <= x <= 1.1 for x in R1s):
    d1_verdict_orig = "DEFECT-REFUTED"
else:
    d1_verdict_orig = "INCONCLUSIVE"

# -- corrected criterion (post hoc, same windows): engine / (3 alpha_L[LIT] dT g) --
RL = [r["R_LIT_e2e_over_3alphaL_wiki"] for r in small]
if all(0.300 <= x <= 0.367 for x in RL):
    d1_verdict_lit = "DEFECT-CONFIRMED"
elif all(0.9 <= x <= 1.1 for x in RL):
    d1_verdict_lit = "DEFECT-REFUTED"
else:
    d1_verdict_lit = "INCONCLUSIVE"
print(f"[D1] original criterion R1 (|dT|<=10 K) in [{min(R1s):.6f}, {max(R1s):.6f}] -> {d1_verdict_orig}")
print(f"[D1] corrected criterion engine/(3 alpha_L[LIT] dT g) in [{min(RL):.4f}, {max(RL):.4f}] -> {d1_verdict_lit}")

# -- counterfactual control: cascade replaced IN MEMORY by 3x itself (no file is touched) --
te.expansion_coefficient_at_T = lambda m, T, sigma=0.0: 3.0 * ORIG_ALPHA(m, T, sigma)
try:
    cf_rows = d1_table("counterfactual-volumetric-cascade(3x, in-process)")
finally:
    te.expansion_coefficient_at_T = ORIG_ALPHA
assert te.expansion_coefficient_at_T is ORIG_ALPHA
cf_small = [r for r in cf_rows if abs(r["dT_K"]) <= 10.0]
cf_R1 = [r["R1_direct_over_exact"] for r in cf_small] + [r["R1_e2e_over_exact"] for r in cf_small]
cf_RL = [r["R_LIT_e2e_over_3alphaL_wiki"] for r in cf_small]
print(f"[D1-CF] counterfactual: R1 in [{min(cf_R1):.6f}, {max(cf_R1):.6f}] (unchanged -> by construction); "
      f"LIT ratio in [{min(cf_RL):.4f}, {max(cf_RL):.4f}]")
# sanity: patch removed -> as-coded value reproduced
chk, _ = lone_step_accel_y("iron", T0 + 10.0)
chk0, _ = lone_step_accel_y("iron", T0)
assert abs((chk - chk0) - [r for r in d1_rows if r["material"] == "iron" and r["dT_K"] == 10.0][0]["a_engine_end_to_end_m_s2"]) < 1e-15

# -- D1-EOS: under eos='ideal' the Boussinesq term is still applied (lone parcel) --
eos_rows = []
M_N2 = gasmod.molecular_mass_kg("N2") * gasmod.N_AVOGADRO
gam_N2 = gasmod.heat_capacity_ratio("N2")
M_Fe = fe_atom["atomic_mass_amu"] * 1e-3                         # amu -> kg/mol (unit conversion)
for carrier, M, gam, lab in (("water_ice", M_N2, gam_N2, "N2 (repo gas module), water_ice carrier as in test_continuum_energy.py"),
                             ("iron", M_Fe, 5.0 / 3.0, "Fe atomic mass (Mentat), gamma 5/3 monatomic [DERIVED]")):
    kw = dict(eos="ideal", ideal_molar_mass_kg_mol=M, ideal_gamma=gam)
    a0, _ = lone_step_accel_y(carrier, T0, **kw)
    a1, _ = lone_step_accel_y(carrier, T0 + 10.0, **kw)
    pc = ContinuumParcel(carrier, radius=0.01, temperature=T0 + 10.0); pc.update_properties()
    eos_rows.append({"carrier_material": carrier, "gas": lab, "M_kg_mol": M, "gamma": gam, "dT_K": 10.0,
                     "a_buoyancy_e2e_m_s2": a1 - a0, "carrier_alpha_L_1_K": pc.thermal_expansion,
                     "ratio_to_alpha_L_dT_g": (a1 - a0) / (pc.thermal_expansion * 10.0 * g),
                     "ideal_gas_beta_1_over_T0_1_K": 1.0 / T0,
                     "ratio_to_ideal_gas_beta_dT_g": (a1 - a0) / (10.0 / T0 * g)})
print("[D1-EOS]", json.dumps([{k: r[k] for k in ("carrier_material", "ratio_to_alpha_L_dT_g",
                                                  "ratio_to_ideal_gas_beta_dT_g")} for r in eos_rows]))

d1 = {"rows_as_coded": d1_rows, "rows_counterfactual_volumetric_cascade": cf_rows, "ideal_eos_lone_parcel": eos_rows,
      "original_criterion": {"R1_small_dT_range": rng(R1s),
                             "R1_all_range": rng([min(r["R1_direct_over_exact"], r["R1_e2e_over_exact"]) for r in d1_rows]
                                                 + [max(r["R1_direct_over_exact"], r["R1_e2e_over_exact"]) for r in d1_rows]),
                             "verdict": d1_verdict_orig,
                             "status": "BY CONSTRUCTION with respect to linear-vs-volumetric: the counterfactual "
                                       "volumetric cascade leaves it at 1/3"},
      "corrected_criterion": {"definition": "engine end-to-end buoyancy / (3 alpha_L[LIT, Wikipedia 20 C] dT g), all "
                                            "|dT| <= 10 K cases; CONFIRMED if all in [0.300, 0.367], REFUTED if all in "
                                            "[0.9, 1.1] (same windows as the pre-registration; set after the skeptic review)",
                              "range_as_coded": rng(RL), "verdict": d1_verdict_lit,
                              "range_counterfactual_volumetric_cascade": rng(cf_RL),
                              "range_vs_alphaV_wiki_as_coded": rng([r["R_LIT_e2e_over_alphaV_wiki"] for r in small]),
                              "range_vs_3alphaL_engineeringtoolbox_as_coded": rng(
                                  [x for r in small for x in r["R_LIT_e2e_over_3alphaL_etb_range"]]),
                              "cascade_alpha300_over_LIT_alphaL_range": rng(
                                  [r["cascade_alpha300_over_LIT_alphaL_wiki"] for r in d1_rows])},
      "counterfactual_R1_small_dT_range": rng(cf_R1)}

# ══════════════════════════════════════════════════════════════════════════
# D2 -- SPH heat conduction
# ══════════════════════════════════════════════════════════════════════════
NY, NZ = 7, 7
DX = 0.01
DTAMP = 1.0
R_PARCEL = DX * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)        # (4/3) pi r^3 = dx^3
H = smoothing_length(DX ** 3)                                   # engine default, 1.2 dx


def build_lattice(mat, nx, kw, mass_scale=1.0):
    ps, idx = [], {}
    for ix in range(nx):
        for iy in range(NY):
            for iz in range(NZ):
                x = ix * DX
                p = ContinuumParcel(mat, radius=R_PARCEL,
                                    temperature=T0 + DTAMP * math.sin(kw * x),
                                    position=Vec3(x, iy * DX, iz * DX))
                p.mass *= mass_scale
                idx[(ix, iy, iz)] = len(ps)
                ps.append(p)
    for p in ps:
        p.update_properties()
    return ps, idx


# independent reference: cubic spline (Monaghan & Lattanzio 1985) written here, not imported
_N3 = 3.0 / (2.0 * math.pi)


def my_W(r, h):
    q = r / h
    if q < 1.0:
        return _N3 / h ** 3 * (2.0 / 3.0 - q * q + 0.5 * q ** 3)
    if q < 2.0:
        return _N3 / h ** 3 * (2.0 - q) ** 3 / 6.0
    return 0.0


def my_dWdr(r, h):
    q = r / h
    if q < 1.0:
        return _N3 / h ** 4 * (-2.0 * q + 1.5 * q * q)
    if q < 2.0:
        return _N3 / h ** 4 * (-0.5 * (2.0 - q) ** 2)
    return 0.0


def reference_dTdt(ps, i, h, rho):
    """Jubelgas, Springel & Dolag 2004 eq. (17) [Brookshaw 1985; Monaghan 1992], arithmetic kappa
    mean and the engine's eta^2 = 0.01 h^2, divided by c_p only."""
    pi_ = ps[i]
    s = 0.0
    for j, pj in enumerate(ps):
        if j == i:
            continue
        dx = pi_.position.x - pj.position.x
        dy = pi_.position.y - pj.position.y
        dz = pi_.position.z - pj.position.z
        r2 = dx * dx + dy * dy + dz * dz
        r = math.sqrt(r2)
        if r >= 2.0 * h:
            continue
        rdotgrad = r * my_dWdr(r, h)
        s += (pj.mass / (rho[i] * rho[j]) * (pi_.thermal_conductivity + pj.thermal_conductivity)
              * (pi_.temperature - pj.temperature) * rdotgrad / (r2 + 0.01 * h * h))
    return s / pi_.specific_heat


def my_density_for(ps, h, which):
    """Independent kernel density at the parcels in `which` (set of indices)."""
    out = {}
    for i in which:
        pi_ = ps[i]
        s = 0.0
        for pj in ps:
            dx = pi_.position.x - pj.position.x
            dy = pi_.position.y - pj.position.y
            dz = pi_.position.z - pj.position.z
            s += pj.mass * my_W(math.sqrt(dx * dx + dy * dy + dz * dz), h)
        out[i] = s
    return out


def lattice_case(mat, nlam=8, mass_scale=1.0, with_reference=True):
    lam = nlam * DX
    kw = 2 * math.pi / lam
    nx = max(16, nlam + 8)
    ps, idx = build_lattice(mat, nx, kw, mass_scale)
    _sph_density_sum(ps, H)
    eng = _heat_conduction(ps, H)
    props = material_properties(mat, T0)
    kappa0, cp0, rho_phys = props["thermal_conductivity_W_mK"], props["specific_heat_J_kgK"], props["density_kg_m3"]
    interior = [(ix, 3, 3) for ix in range(3, nx - 3)]
    rows = []
    if with_reference:
        need = set()
        for key in interior:
            i = idx[key]
            need.add(i)
            for j, pj in enumerate(ps):
                ddx = ps[i].position.x - pj.position.x
                ddy = ps[i].position.y - pj.position.y
                ddz = ps[i].position.z - pj.position.z
                if ddx * ddx + ddy * ddy + ddz * ddz < 4.0 * H * H:
                    need.add(j)
        rho_my = my_density_for(ps, H, need)
    for key in interior:
        i = idx[key]
        x = ps[i].position.x
        rho_m = rho_phys * mass_scale
        analytic = -(kappa0 / (rho_m * cp0)) * kw * kw * DTAMP * math.sin(kw * x)
        row = {"ix": key[0], "x_m": x, "rho_sph": ps[i].sph_density, "engine_dTdt_K_s": eng[i],
               "analytic_dTdt_K_s": analytic}
        if with_reference:
            ref = reference_dTdt(ps, i, H, rho_my)
            row.update({"reference_dTdt_K_s": ref, "rho_sph_independent": rho_my[i],
                        "engine_times_rho_over_reference": (eng[i] * ps[i].sph_density / ref) if ref != 0 else None})
        rows.append(row)
    s2 = sum(r["analytic_dTdt_K_s"] ** 2 for r in rows)
    out = {"material": mat, "lambda_over_dx": nlam, "kh": kw * H, "nx": nx, "mass_scale": mass_scale,
           "Q_engine": sum(r["engine_dTdt_K_s"] * r["analytic_dTdt_K_s"] for r in rows) / s2,
           "rho_sph_interior_mean": sum(r["rho_sph"] for r in rows) / len(rows),
           "rho_mass_per_volume": rho_phys * mass_scale,
           "kappa_W_mK": kappa0, "cp_J_kgK": cp0, "D_cascade_m2_s": kappa0 / (rho_phys * cp0)}
    ssin = sum(math.sin(kw * r["x_m"]) ** 2 for r in rows)
    out["engine_amp_K_s"] = sum(r["engine_dTdt_K_s"] * math.sin(kw * r["x_m"]) for r in rows) / ssin
    out["Q_engine_times_rho_sph"] = out["Q_engine"] * out["rho_sph_interior_mean"]
    if with_reference:
        out["Q_reference_independent"] = sum(r["reference_dTdt_K_s"] * r["analytic_dTdt_K_s"] for r in rows) / s2
        out["reference_amp_K_s"] = sum(r["reference_dTdt_K_s"] * math.sin(kw * r["x_m"]) for r in rows) / ssin
        rel = [abs(r["engine_times_rho_over_reference"] - 1.0) for r in rows
               if r["engine_times_rho_over_reference"] is not None]
        out["max_rel_dev_engine_times_rho_vs_reference"] = max(rel)
        out["max_rel_dev_density_engine_vs_independent"] = max(
            abs(r["rho_sph"] / r["rho_sph_independent"] - 1.0) for r in rows)
    out["rows"] = rows
    return out, ps, idx


# (a) operator test, iron, k h = 0.94 (pre-registered) + copper (new, same geometry)
t_a = time.time()
case1, ps1, idx1 = lattice_case("iron", 8, 1.0, True)
print(f"[D2a] iron: Q_engine = {case1['Q_engine']:.4e}  Q_reference = {case1['Q_reference_independent']:.4f}  "
      f"rho_sph = {case1['rho_sph_interior_mean']:.1f}  engine*rho/ref max dev = "
      f"{case1['max_rel_dev_engine_times_rho_vs_reference']:.2e}  ({time.time()-t_a:.0f}s)")
d2a_ok = (0.8 <= case1["Q_reference_independent"] <= 1.2 and case1["Q_engine"] < 1e-2
          and case1["max_rel_dev_engine_times_rho_vs_reference"] <= 1e-9)
d2a_ref = 0.8 <= case1["Q_engine"] <= 1.2
case_cu, _, _ = lattice_case("copper", 8, 1.0, True)
print(f"[D2a] copper: Q_engine = {case_cu['Q_engine']:.4e}  Q_reference = {case_cu['Q_reference_independent']:.4f}  "
      f"rho_sph = {case_cu['rho_sph_interior_mean']:.1f}  Q_engine*rho = {case_cu['Q_engine_times_rho_sph']:.5f}")

# (b) density scaling (pre-registered)
scal = []
for s in (0.25, 0.5, 1.0, 2.0, 4.0):
    c, _, _ = lattice_case("iron", 8, s, with_reference=(s in (0.25, 1.0, 4.0)))
    entry = {"mass_scale": s, "rho_sph_interior_mean": c["rho_sph_interior_mean"],
             "engine_amp_K_s": c["engine_amp_K_s"], "Q_engine": c["Q_engine"]}
    if "reference_amp_K_s" in c:
        entry["reference_amp_K_s"] = c["reference_amp_K_s"]
    scal.append(entry)
    print(f"[D2b] s={s}: rho_sph={c['rho_sph_interior_mean']:.1f} engine amp={c['engine_amp_K_s']:.4e}")


def slope(xs, ys):
    lx = [math.log(x) for x in xs]; ly = [math.log(abs(y)) for y in ys]
    mx = sum(lx) / len(lx); my = sum(ly) / len(ly)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)


slope_eng = slope([e["rho_sph_interior_mean"] for e in scal], [e["engine_amp_K_s"] for e in scal])
refpts = [e for e in scal if "reference_amp_K_s" in e]
slope_ref = slope([e["rho_sph_interior_mean"] for e in refpts], [e["reference_amp_K_s"] for e in refpts])
print(f"[D2b] slope engine = {slope_eng:.4f}  slope reference = {slope_ref:.4f}")
d2b_ok = -2.05 <= slope_eng <= -1.95
d2b_ref = -1.05 <= slope_eng <= -0.95


# (c) two-parcel energy balance, unequal masses, both orientations
def two_parcel(T_small, T_big):
    p1 = ContinuumParcel("iron", radius=0.010, temperature=T_small, position=Vec3(0, 0, 0))
    p2 = ContinuumParcel("iron", radius=0.0125, temperature=T_big, position=Vec3(0.015, 0, 0))
    for p in (p1, p2):
        p.update_properties()
    h2 = 0.02
    _sph_density_sum([p1, p2], h2)
    eng2 = _heat_conduction([p1, p2], h2)
    rho_my2 = my_density_for([p1, p2], h2, {0, 1})
    ref2 = [reference_dTdt([p1, p2], i, h2, rho_my2) for i in (0, 1)]
    q_eng = [p.mass * p.specific_heat * eng2[i] for i, p in enumerate((p1, p2))]
    q_ref = [p.mass * p.specific_heat * ref2[i] for i, p in enumerate((p1, p2))]
    I_eng = abs(sum(q_eng)) / sum(abs(q) for q in q_eng)
    I_ref = abs(sum(q_ref)) / sum(abs(q) for q in q_ref)
    I_pred = abs(p1.sph_density - p2.sph_density) / (p1.sph_density + p2.sph_density)
    hot = 0 if T_small > T_big else 1
    cold = 1 - hot
    rho_s = [p1.sph_density, p2.sph_density]
    return {"T_small_parcel_K": T_small, "T_big_parcel_K": T_big, "m_kg": [p1.mass, p2.mass], "rho_sph": rho_s,
            "dTdt_engine": [eng2[0], eng2[1]], "dTdt_reference": ref2,
            "heat_rate_engine_W": q_eng, "heat_rate_reference_W": q_ref,
            "net_heat_engine_W": sum(q_eng), "net_heat_reference_W": sum(q_ref),
            "gain_over_loss_engine": q_eng[cold] / -q_eng[hot], "gain_over_loss_reference": q_ref[cold] / -q_ref[hot],
            "rho_hot_over_rho_cold": rho_s[hot] / rho_s[cold],
            "I_engine": I_eng, "I_predicted": I_pred, "I_reference": I_ref}


two_a = two_parcel(1000.0, 300.0)      # pre-registered orientation: hot parcel is the smaller, less dense one
two_b = two_parcel(300.0, 1000.0)      # swapped: hot parcel is the denser one
print(f"[D2c] hot less dense: I_engine = {two_a['I_engine']:.6e} (pred {two_a['I_predicted']:.6e}) "
      f"net {two_a['net_heat_engine_W']:+.4f} W gain/loss {two_a['gain_over_loss_engine']:.6f}; "
      f"I_ref = {two_a['I_reference']:.1e}")
print(f"[D2c] hot denser    : net {two_b['net_heat_engine_W']:+.4f} W gain/loss {two_b['gain_over_loss_engine']:.6f}; "
      f"I_ref = {two_b['I_reference']:.1e}")
d2c_ok = abs(two_a["I_engine"] / two_a["I_predicted"] - 1.0) <= 1e-6 and two_a["I_engine"] > 1e-3
d2c_ref = two_a["I_engine"] < 1e-12


# (c2) two-parcel end to end through continuum_step (one 1 ms step, frozen positions)
def two_parcel_e2e(T_small, T_big, dt=1e-3):
    p1 = ContinuumParcel("iron", radius=0.010, temperature=T_small, position=Vec3(0, 0, 0))
    p2 = ContinuumParcel("iron", radius=0.0125, temperature=T_big, position=Vec3(0.015, 0, 0))
    for p in (p1, p2):
        p.update_properties()
    sc = ContinuumScene([p1, p2], gravity=Vec3(0, 0, 0), ground=False, smoothing_h=0.02,
                        reference_temperature=T0, eos="isothermal", sound_speed_m_s=1e-3)
    sc.energy_equation = False
    cp_before = [p1.specific_heat, p2.specific_heat]
    T_before = [p1.temperature, p2.temperature]
    continuum_step(sc, dt)
    dQ = [p.mass * cp_before[i] * (p.temperature - T_before[i]) for i, p in enumerate((p1, p2))]
    hot = 0 if T_small > T_big else 1
    return {"dQ_J": dQ, "net_J": sum(dQ), "net_W": sum(dQ) / dt, "gain_over_loss": dQ[1 - hot] / -dQ[hot]}


e2e_two = {"hot_less_dense": two_parcel_e2e(1000.0, 300.0), "hot_denser": two_parcel_e2e(300.0, 1000.0)}
print("[D2c2] continuum_step:", json.dumps(e2e_two))

# (d) end-to-end lattice through continuum_step (pre-registered)
KW8 = 2 * math.pi / (8 * DX)
ps, idx = build_lattice("iron", 16, KW8)
sc = ContinuumScene(ps, gravity=Vec3(0, 0, 0), ground=False, smoothing_h=H,
                    reference_temperature=T0, eos="isothermal", sound_speed_m_s=1e-3)
sc.energy_equation = False
_sph_density_sum(ps, H)
pre = _heat_conduction(ps, H)
T_before = [p.temperature for p in ps]
DT_STEP = 0.01
continuum_step(sc, DT_STEP)
e2e_rows = []
for key in [(ix, 3, 3) for ix in range(3, 13)]:
    i = idx[key]
    meas = (ps[i].temperature - T_before[i]) / DT_STEP
    e2e_rows.append({"ix": key[0], "measured_dTdt_K_s": meas, "operator_dTdt_K_s": pre[i],
                     "rel_dev": abs(meas / pre[i] - 1.0) if pre[i] != 0 else None})
max_dev_e2e = max(r["rel_dev"] for r in e2e_rows if r["rel_dev"] is not None)
full_amp = [r["rel_dev"] for r in e2e_rows if abs(math.sin(KW8 * r["ix"] * DX)) > 0.5]
max_disp = max(math.sqrt((p.position.x - (k[0] * DX)) ** 2 + (p.position.y - k[1] * DX) ** 2
                         + (p.position.z - k[2] * DX) ** 2) for k, p in ((k, ps[i]) for k, i in idx.items()))
print(f"[D2d] continuum_step dT/dt vs operator: max rel dev {max_dev_e2e:.2e} (full-amplitude parcels "
      f"{max(full_amp):.1e}); max displacement {max_disp:.2e} m")
d2d_ok = max_dev_e2e <= 1e-3

# (e) supplementary, not pre-registered: k h convergence with the INDEPENDENT reference at every k h
conv = []
for nlam in (8, 12, 16, 24):
    c, _, _ = lattice_case("iron", nlam, 1.0, True)
    conv.append({k: c[k] for k in ("lambda_over_dx", "kh", "nx", "Q_engine", "Q_reference_independent",
                                   "Q_engine_times_rho_sph", "rho_sph_interior_mean",
                                   "max_rel_dev_engine_times_rho_vs_reference")})
    print(f"[D2e] lambda={nlam}dx kh={c['kh']:.3f}: Q_ref(indep)={c['Q_reference_independent']:.4f} "
          f"Q_eng*rho={c['Q_engine_times_rho_sph']:.4f} Q_eng={c['Q_engine']:.4e}")

# long-wavelength lattice moment with the independent kernel (k h -> 0 limit of the reference)
n = 4
m_par = ps1[0].mass
rho_inf = sum(m_par * my_W(math.sqrt((a * DX) ** 2 + (b * DX) ** 2 + (c_ * DX) ** 2), H)
              for a in range(-n, n + 1) for b in range(-n, n + 1) for c_ in range(-n, n + 1))
M_mom = 0.0
for a in range(-n, n + 1):
    for b in range(-n, n + 1):
        for c_ in range(-n, n + 1):
            if a == b == c_ == 0:
                continue
            x, y, z = a * DX, b * DX, c_ * DX
            r = math.sqrt(x * x + y * y + z * z)
            M_mom += (m_par / rho_inf) * x * x * (-r * my_dWdr(r, H)) / (r * r + 0.01 * H * H)
print(f"[D2e] lattice moment M = {M_mom:.5f} (independent kernel), rho_inf = {rho_inf:.3f}")

if d2a_ok and d2b_ok and d2c_ok:
    d2_verdict = "DEFECT-CONFIRMED"
elif d2a_ref and d2b_ref:
    d2_verdict = "DEFECT-REFUTED"
else:
    d2_verdict = "INCONCLUSIVE"

# realised diffusivities (iron)
D_cas = case1["D_cascade_m2_s"]
rho_s = case1["rho_sph_interior_mean"]
realised = {
    "D_cascade_kappa_over_rho_cp_m2_s": D_cas,
    "cascade_kappa_iron_300K_W_mK": case1["kappa_W_mK"], "cascade_cp_iron_300K_J_kgK": case1["cp_J_kgK"],
    "D_LIT_iron_m2_s": LIT["D_iron_m2_s"],
    "D_engine_at_kh_0.94_m2_s": D_cas * case1["Q_engine"],
    "ratio_D_cascade_over_D_engine_at_kh_0.94": 1.0 / case1["Q_engine"],
    "decomposition": f"1/Q_engine = rho_sph / Q_ref = {rho_s:.2f} / {case1['Q_reference_independent']:.4f}",
    "D_defect_only_D_cascade_over_rho_sph_m2_s": D_cas / rho_s,
    "D_engine_long_wavelength_D_cascade_x_M_over_rho_sph_m2_s": D_cas * M_mom / rho_s,
    "clean_factor_1_over_rho_sph": 1.0 / rho_s,
    "D_engine_long_wavelength_over_D_LIT": D_cas * M_mom / rho_s / LIT["D_iron_m2_s"],
}
side = {"cascade_kappa_iron_W_mK": {"300K": material_properties("iron", 300.0)["thermal_conductivity_W_mK"],
                                    "400K": material_properties("iron", 400.0)["thermal_conductivity_W_mK"]},
        "LIT_kappa_iron_W_mK": {"300K": LIT["kappa_iron_300K_W_mK"], "400K_engineeringtoolbox": 70.0}}
for m in ("glass", "granite", "concrete", "rubber", "plastic_abs"):
    pr = material_properties(m, 300.0)
    side[m] = {"kappa_W_mK": pr["thermal_conductivity_W_mK"], "melting_point_K": pr.get("melting_point_K")}

consequence = {
    "Ra_note": "Ra = g beta dT d^3/(nu D). As coded beta -> beta/3 and D -> D/rho_sph[SI] (x lattice moment). "
               "If Ra is computed from the cascade D, Ra_eff/Ra = rho_sph/3 (about 2.6e3 for iron). If Ra is computed "
               "from a MEASURED kappa_eff (as the U14 design sketch does, 1-D bar decay), the conduction defect is "
               "absorbed into kappa_eff and the as-coded onset appears about 3x too high in dT, from buoyancy alone.",
    "Ra_eff_over_Ra_with_cascade_D_iron": rho_s / 3.0 / M_mom,
}

result = {
    "id": "U14G", "version": "v2 (repair run after three skeptic reviews)",
    "engine_file": "sigma_ground/dynamics/continuum.py",
    "engine_lines": {"_buoyancy_force": "621-637 (alpha at 630, return 633-637)",
                     "buoyancy applied for every eos": "763 (inside _forces; eos='ideal' pressure at 742-747)",
                     "_heat_conduction": "640-684 (extra 1/rho_i at 682: dTdt[i] = rate / (rho_i * cp_i); docstrings "
                                         "45 and 643 carry the same extra factor; 646 credits Cleary & Monaghan 1999 "
                                         "for what is the arithmetic-mean Brookshaw/Monaghan form)",
                     "thermal_expansion property": "239-242 (docstring: 'Linear thermal expansion')",
                     "cascade": "field/interface/thermal_expansion.py:74-182 EXPANSION_DATA 'alpha_linear_per_K' (CRC), "
                                ":513 expansion_coefficient_at_T ('linear'), :326-343 volumetric = 3 x linear"},
    "mentat_sources": sources, "literature": LIT,
    "D1": d1,
    "D2": {"a_operator_lattice_iron": {k: v for k, v in case1.items()},
           "a_operator_lattice_copper": {k: v for k, v in case_cu.items() if k != "rows"},
           "a_criteria": {"pass_confirm": d2a_ok, "pass_refute": d2a_ref},
           "b_density_scaling": {"points": scal, "slope_engine": slope_eng, "slope_reference": slope_ref,
                                 "pass_confirm": d2b_ok, "pass_refute": d2b_ref,
                                 "status": "VERIFIED numerics: algebraic consequence of the 1/rho_i factor at line 682"},
           "c_two_parcel": {"hot_less_dense_preregistered": dict(two_a, pass_confirm=d2c_ok, pass_refute=d2c_ref),
                            "hot_denser_new": two_b, "end_to_end_continuum_step": e2e_two},
           "d_end_to_end": {"rows": e2e_rows, "max_rel_dev": max_dev_e2e, "max_rel_dev_full_amplitude": max(full_amp),
                            "max_displacement_m": max_disp, "dt_s": DT_STEP, "pass": d2d_ok},
           "e_convergence_supplementary": {"rows": conv, "lattice_moment_M_independent_kernel": M_mom,
                                           "rho_sph_infinite_lattice": rho_inf,
                                           "note": "not pre-registered; v2 computes Q_reference with the independent "
                                                   "eq.-17 implementation at every k h (v1 used engine x rho_sph "
                                                   "for k h < 0.94)"},
           "verdict": d2_verdict},
    "realised_diffusivity_iron": realised, "side_observations": side, "consequence": consequence,
    "verdicts": {"D1_original_criterion": d1_verdict_orig, "D1_corrected_LIT_criterion": d1_verdict_lit,
                 "D2": d2_verdict},
}
d1_v = d1_verdict_lit
result["overall"] = ("DEFECT-CONFIRMED" if d1_v == d2_verdict == "DEFECT-CONFIRMED" else
                     "DEFECT-REFUTED" if d1_v == d2_verdict == "DEFECT-REFUTED" else
                     "MIXED" if {d1_v, d2_verdict} == {"DEFECT-CONFIRMED", "DEFECT-REFUTED"} else "INCONCLUSIVE")
result["overall_under_original_criteria"] = (
    "DEFECT-CONFIRMED" if d1_verdict_orig == d2_verdict == "DEFECT-CONFIRMED" else "see verdicts")
result["runtime_s"] = time.time() - t_start
with open(os.path.join(OUT, "u14g_results_v2.json"), "w", encoding="utf8") as f:
    json.dump(result, f, indent=1)
print("overall:", result["overall"], "| original criteria:", result["overall_under_original_criteria"],
      f"runtime {result['runtime_s']:.0f}s")
