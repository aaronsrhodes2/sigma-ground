"""U14G -- defect check of two suspected engine defects in sigma_ground/dynamics/continuum.py.

D1  _buoyancy_force uses the LINEAR expansion coefficient alpha_L where the
    Boussinesq density perturbation needs the VOLUMETRIC beta = 3 alpha_L.
D2  _heat_conduction divides the SPH Fourier sum by rho_i * c_p although the sum
    is already du_i/dt (W/kg): an extra 1/rho_i.

Read-only against the repo: imports the engine, never edits it.  Pre-registration
in prereg.txt (same folder), written before this script was run.

Run from the repo root:
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
from sigma_ground.field.interface.thermal_expansion import (
    expansion_coefficient_at_T, linear_expansion_coefficient,
    volumetric_expansion_coefficient,
)

t_start = time.time()
sources = {}

# ── Mentat leaves ──────────────────────────────────────────────────────────
earth, src = mentat("solar_system_body", {"body_name": "earth"})
g = float(earth["surface_g_ms2"]); sources["g_surface_earth"] = {"value": g, "source": src,
                                                                  "tool": "solar_system_body(earth).surface_g_ms2"}
rho_fe_mentat, src = mentat("density", {"material": "iron"})
sources["rho_iron_mentat"] = {"value": rho_fe_mentat, "source": src, "tool": "density(iron)"}
sources["engine_cascade"] = {
    "source": "sigma_ground.dynamics.continuum.material_properties -> field.interface "
              "(thermal_expansion.expansion_coefficient_at_T, thermal.thermal_conductivity, "
              "thermal.specific_heat_j_kg_K, acoustics.density_at_sigma); repo HEAD 339c4f4",
    "note": "repo engines are Mentat (run rule 3)"}

T0 = 300.0
GVEC = Vec3(0.0, -g, 0.0)

# ══════════════════════════════════════════════════════════════════════════
# D1 -- Boussinesq buoyancy
# ══════════════════════════════════════════════════════════════════════════
CUBIC = ["iron", "copper", "aluminum", "gold", "silicon", "tungsten",
         "nickel", "silver", "lead", "platinum"]
DTS = [1.0, 10.0, -10.0, 100.0]


def int_alpha(mat, Ta, Tb, n=400):
    """Composite Simpson of the cascade alpha_L(T) from Ta to Tb (n even)."""
    hstep = (Tb - Ta) / n
    s = expansion_coefficient_at_T(mat, Ta) + expansion_coefficient_at_T(mat, Tb)
    for k in range(1, n):
        s += (4 if k % 2 else 2) * expansion_coefficient_at_T(mat, Ta + k * hstep)
    return s * hstep / 3.0


def lone_step_accel_y(mat, T, dt=1e-3):
    p = ContinuumParcel(mat, radius=0.01, temperature=T, position=Vec3(0, 0, 0))
    sc = ContinuumScene([p], gravity=GVEC, ground=False, reference_temperature=T0)
    continuum_step(sc, dt)
    return p.velocity.y / dt, p.temperature


d1_rows = []
for mat in CUBIC:
    vol_ratio_repo = volumetric_expansion_coefficient(mat) / linear_expansion_coefficient(mat)
    a_neutral, _ = lone_step_accel_y(mat, T0)
    for dT in DTS:
        T = T0 + dT
        p = ContinuumParcel(mat, radius=0.01, temperature=T)
        p.update_properties()
        alpha_T = p.thermal_expansion
        a_direct = _buoyancy_force(p, p.props["density_kg_m3"], T0, GVEC).y
        I = int_alpha(mat, T0, T)
        rho_ratio = math.exp(-3.0 * I)                      # rho(T)/rho(T0), isotropic expansion
        a_exact = -g * (1.0 - 1.0 / rho_ratio)              # y-comp of g_vec (1 - rho0/rho), g_vec = -g y
        a_bous = 3.0 * alpha_T * dT * g                     # -beta dT g_vec, beta = 3 alpha_L
        a_hot, T_after = lone_step_accel_y(mat, T)
        a_e2e = a_hot - a_neutral
        d1_rows.append({
            "material": mat, "dT_K": dT, "alpha_L_at_T_1_K": alpha_T,
            "repo_volumetric_over_linear": vol_ratio_repo,
            "a_engine_direct_m_s2": a_direct, "a_engine_end_to_end_m_s2": a_e2e,
            "a_exact_archimedes_m_s2": a_exact, "a_boussinesq_3alpha_m_s2": a_bous,
            "R1_direct_over_exact": a_direct / a_exact,
            "R1_e2e_over_exact": a_e2e / a_exact,
            "R1_direct_over_boussinesq3a": a_direct / a_bous,
            "engine_matches_minus_alpha_dT_g": abs(a_direct - alpha_T * dT * g) <= 1e-12 * abs(alpha_T * dT * g) + 1e-300,
            "T_after_step_K": T_after,
        })

small = [r for r in d1_rows if abs(r["dT_K"]) <= 10.0]
R1s = [r["R1_direct_over_exact"] for r in small] + [r["R1_e2e_over_exact"] for r in small]
if all(0.300 <= x <= 0.367 for x in R1s):
    d1_verdict = "DEFECT-CONFIRMED"
elif all(0.9 <= x <= 1.1 for x in R1s):
    d1_verdict = "DEFECT-REFUTED"
else:
    d1_verdict = "INCONCLUSIVE"
d1 = {"rows": d1_rows, "R1_small_dT_min": min(R1s), "R1_small_dT_max": max(R1s),
      "R1_all_min": min(min(r["R1_direct_over_exact"], r["R1_e2e_over_exact"]) for r in d1_rows),
      "R1_all_max": max(max(r["R1_direct_over_exact"], r["R1_e2e_over_exact"]) for r in d1_rows),
      "verdict": d1_verdict}
print(f"[D1] R1 (|dT|<=10 K) in [{d1['R1_small_dT_min']:.6f}, {d1['R1_small_dT_max']:.6f}] -> {d1_verdict}")

# ══════════════════════════════════════════════════════════════════════════
# D2 -- SPH heat conduction
# ══════════════════════════════════════════════════════════════════════════
MAT = "iron"
NX, NY, NZ = 16, 7, 7
DX = 0.01
LAM = 8 * DX
KW = 2 * math.pi / LAM
DTAMP = 1.0
R_PARCEL = DX * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)        # (4/3) pi r^3 = dx^3
H = smoothing_length(DX ** 3)                                   # engine default, 1.2 dx
INTERIOR = [(ix, 3, 3) for ix in range(3, 13)]


def build_lattice(mass_scale=1.0):
    ps, idx = [], {}
    for ix in range(NX):
        for iy in range(NY):
            for iz in range(NZ):
                x = ix * DX
                p = ContinuumParcel(MAT, radius=R_PARCEL,
                                    temperature=T0 + DTAMP * math.sin(KW * x),
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
    """Jubelgas, Springel & Dolag 2004 eq. (17) [Brookshaw 1985; Cleary & Monaghan 1999 family],
    arithmetic kappa mean and the engine's eta^2 = 0.01 h^2, divided by c_p only."""
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
        rdotgrad = r * my_dWdr(r, h)                         # r_ij . grad_i W_ij
        s += (pj.mass / (rho[i] * rho[j]) * (pi_.thermal_conductivity + pj.thermal_conductivity)
              * (pi_.temperature - pj.temperature) * rdotgrad / (r2 + 0.01 * h * h))
    return s / pi_.specific_heat


def my_density(ps, h):
    out = []
    for i, pi_ in enumerate(ps):
        s = 0.0
        for pj in ps:
            dx = pi_.position.x - pj.position.x
            dy = pi_.position.y - pj.position.y
            dz = pi_.position.z - pj.position.z
            s += pj.mass * my_W(math.sqrt(dx * dx + dy * dy + dz * dz), h)
        out.append(s)
    return out


props0 = material_properties(MAT, T0)
kappa0, cp0 = props0["thermal_conductivity_W_mK"], props0["specific_heat_J_kgK"]
rho_phys = props0["density_kg_m3"]
D0 = kappa0 / (rho_phys * cp0)


def lattice_case(mass_scale=1.0, with_reference=True):
    ps, idx = build_lattice(mass_scale)
    _sph_density_sum(ps, H)
    eng = _heat_conduction(ps, H)
    rho_eng = [p.sph_density for p in ps]
    rows = []
    ref_vals = {}
    if with_reference:
        rho_my = my_density(ps, H)
    for key in INTERIOR:
        i = idx[key]
        x = ps[i].position.x
        rho_m = rho_phys * mass_scale
        analytic = -(kappa0 / (rho_m * cp0)) * KW * KW * DTAMP * math.sin(KW * x)
        row = {"ix": key[0], "x_m": x, "rho_sph": rho_eng[i], "engine_dTdt_K_s": eng[i],
               "analytic_dTdt_K_s": analytic}
        if with_reference:
            ref = reference_dTdt(ps, i, H, rho_my)
            row.update({"reference_dTdt_K_s": ref, "rho_sph_independent": rho_my[i],
                        "engine_over_reference": (eng[i] / ref) if ref != 0 else None,
                        "engine_times_rho_over_reference":
                            (eng[i] * rho_eng[i] / ref) if ref != 0 else None})
        rows.append(row)
    s2 = sum(r["analytic_dTdt_K_s"] ** 2 for r in rows)
    A_eng = sum(r["engine_dTdt_K_s"] * r["analytic_dTdt_K_s"] for r in rows) / s2
    out = {"mass_scale": mass_scale, "rows": rows, "Q_engine": A_eng,
           "rho_sph_interior_mean": sum(r["rho_sph"] for r in rows) / len(rows),
           "rho_mass_per_volume": rho_phys * mass_scale}
    # amplitude of the engine field itself (for the slope test): fit to sin(kx)
    ssin = sum(math.sin(KW * r["x_m"]) ** 2 for r in rows)
    out["engine_amp_K_s"] = sum(r["engine_dTdt_K_s"] * math.sin(KW * r["x_m"]) for r in rows) / ssin
    if with_reference:
        out["Q_reference"] = sum(r["reference_dTdt_K_s"] * r["analytic_dTdt_K_s"] for r in rows) / s2
        out["reference_amp_K_s"] = sum(r["reference_dTdt_K_s"] * math.sin(KW * r["x_m"]) for r in rows) / ssin
        rel = [abs(r["engine_times_rho_over_reference"] - 1.0) for r in rows
               if r["engine_times_rho_over_reference"] is not None]
        out["max_rel_dev_engine_times_rho_vs_reference"] = max(rel)
        out["max_rel_dev_density_engine_vs_independent"] = max(
            abs(r["rho_sph"] / r["rho_sph_independent"] - 1.0) for r in rows)
    return out, ps, idx


# (a) operator test
t_a = time.time()
case1, ps1, idx1 = lattice_case(1.0, True)
print(f"[D2a] Q_engine = {case1['Q_engine']:.4e}  Q_reference = {case1['Q_reference']:.4f}  "
      f"rho_sph = {case1['rho_sph_interior_mean']:.1f}  engine*rho/ref max dev = "
      f"{case1['max_rel_dev_engine_times_rho_vs_reference']:.2e}  ({time.time()-t_a:.0f}s)")
d2a_ok = (0.8 <= case1["Q_reference"] <= 1.2 and case1["Q_engine"] < 1e-2
          and case1["max_rel_dev_engine_times_rho_vs_reference"] <= 1e-9)
d2a_ref = 0.8 <= case1["Q_engine"] <= 1.2

# (b) density scaling
scal = []
for s in (0.25, 0.5, 1.0, 2.0, 4.0):
    c, _, _ = lattice_case(s, with_reference=(s in (0.25, 4.0)))
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

# (c) two-parcel energy balance, unequal masses
p1 = ContinuumParcel(MAT, radius=0.010, temperature=1000.0, position=Vec3(0, 0, 0))
p2 = ContinuumParcel(MAT, radius=0.0125, temperature=300.0, position=Vec3(0.015, 0, 0))
for p in (p1, p2):
    p.update_properties()
h2 = 0.02
_sph_density_sum([p1, p2], h2)
eng2 = _heat_conduction([p1, p2], h2)
rho_my2 = my_density([p1, p2], h2)
ref2 = [reference_dTdt([p1, p2], i, h2, rho_my2) for i in (0, 1)]
q_eng = [p.mass * p.specific_heat * eng2[i] for i, p in enumerate((p1, p2))]
q_ref = [p.mass * p.specific_heat * ref2[i] for i, p in enumerate((p1, p2))]
I_eng = abs(sum(q_eng)) / sum(abs(q) for q in q_eng)
I_ref = abs(sum(q_ref)) / sum(abs(q) for q in q_ref)
I_pred = abs(p1.sph_density - p2.sph_density) / (p1.sph_density + p2.sph_density)
print(f"[D2c] I_engine = {I_eng:.6e} (pred {I_pred:.6e})  I_reference = {I_ref:.2e}")
d2c_ok = abs(I_eng / I_pred - 1.0) <= 1e-6 and I_eng > 1e-3
d2c_ref = I_eng < 1e-12
two = {"m1_kg": p1.mass, "m2_kg": p2.mass, "rho_sph_1": p1.sph_density, "rho_sph_2": p2.sph_density,
       "dTdt_engine": [eng2[0], eng2[1]], "dTdt_reference": ref2,
       "heat_rate_engine_W": q_eng, "heat_rate_reference_W": q_ref,
       "engine_over_reference": [eng2[0] / ref2[0], eng2[1] / ref2[1]],
       "one_over_rho_sph": [1 / p1.sph_density, 1 / p2.sph_density],
       "I_engine": I_eng, "I_predicted": I_pred, "I_reference": I_ref,
       "net_heat_created_by_engine_W": sum(q_eng)}

# (d) end-to-end through continuum_step
ps, idx = build_lattice(1.0)
sc = ContinuumScene(ps, gravity=Vec3(0, 0, 0), ground=False, smoothing_h=H,
                    reference_temperature=T0, eos="isothermal", sound_speed_m_s=1e-3)
sc.energy_equation = False
_sph_density_sum(ps, H)
pre = _heat_conduction(ps, H)
T_before = [p.temperature for p in ps]
DT_STEP = 0.01
continuum_step(sc, DT_STEP)
e2e_rows = []
for key in INTERIOR:
    i = idx[key]
    meas = (ps[i].temperature - T_before[i]) / DT_STEP
    e2e_rows.append({"ix": key[0], "measured_dTdt_K_s": meas, "operator_dTdt_K_s": pre[i],
                     "rel_dev": abs(meas / pre[i] - 1.0) if pre[i] != 0 else None})
max_dev_e2e = max(r["rel_dev"] for r in e2e_rows if r["rel_dev"] is not None)
max_disp = max(math.sqrt((p.position.x - (k[0] * DX)) ** 2 + (p.position.y - k[1] * DX) ** 2
                         + (p.position.z - k[2] * DX) ** 2) for k, p in ((k, ps[i]) for k, i in idx.items()))
print(f"[D2d] continuum_step dT/dt vs operator: max rel dev {max_dev_e2e:.2e}; max displacement {max_disp:.2e} m")
d2d_ok = max_dev_e2e <= 1e-3

if d2a_ok and d2b_ok and d2c_ok:
    d2_verdict = "DEFECT-CONFIRMED"
elif d2a_ref and d2b_ref:
    d2_verdict = "DEFECT-REFUTED"
else:
    d2_verdict = "INCONCLUSIVE"

# consequence numbers: effective diffusivity the engine realises (iron) and as-coded Rayleigh factor
D_eff_engine = D0 * case1["Q_engine"]
consequence = {
    "D_physical_iron_m2_s": D0, "D_engine_realised_m2_s": D_eff_engine,
    "conduction_slowdown_factor": 1.0 / case1["Q_engine"] * case1["Q_reference"],
    "buoyancy_factor_engine_over_physical": d1_rows[1]["R1_direct_over_exact"],
    "note": "Ra = g beta dT d^3/(nu D): as coded beta -> beta/3 and D -> D/rho_sph[SI], so the engine's "
            "effective Ra is (rho_sph/3) x the physical one for the same set-up (unit-dependent: the 1/rho_i "
            "factor is numerically 1/7874 in SI for iron, 1/7.874 in cgs)."}

result = {
    "id": "U14G",
    "engine_file": "sigma_ground/dynamics/continuum.py",
    "engine_lines": {"_buoyancy_force": "621-637 (alpha at 630, return 633-637)",
                     "_heat_conduction": "640-684 (extra 1/rho_i at 682: dTdt[i] = rate / (rho_i * cp_i); docstrings 45 and 643 carry the same extra factor)",
                     "thermal_expansion property": "239-242 (docstring: 'Linear thermal expansion')",
                     "cascade": "field/interface/thermal_expansion.py:513 expansion_coefficient_at_T "
                                "('linear expansion coefficient'), :326 volumetric = 3 x linear"},
    "mentat_sources": sources,
    "D1": d1,
    "D2": {"a_operator_lattice": {k: v for k, v in case1.items()},
           "a_criteria": {"pass_confirm": d2a_ok, "pass_refute": d2a_ref},
           "b_density_scaling": {"points": scal, "slope_engine": slope_eng, "slope_reference": slope_ref,
                                 "pass_confirm": d2b_ok, "pass_refute": d2b_ref},
           "c_two_parcel": dict(two, pass_confirm=d2c_ok, pass_refute=d2c_ref),
           "d_end_to_end": {"rows": e2e_rows, "max_rel_dev": max_dev_e2e, "max_displacement_m": max_disp,
                            "dt_s": DT_STEP, "pass": d2d_ok},
           "lattice": {"NX": NX, "NY": NY, "NZ": NZ, "dx_m": DX, "h_m": H, "lambda_m": LAM,
                       "kh": KW * H, "dT_amp_K": DTAMP, "kappa_W_mK": kappa0, "cp_J_kgK": cp0,
                       "rho_kg_m3": rho_phys, "D_m2_s": D0},
           "verdict": d2_verdict},
    "consequence": consequence,
    "overall": ("DEFECT-CONFIRMED" if d1_verdict == d2_verdict == "DEFECT-CONFIRMED" else
                "DEFECT-REFUTED" if d1_verdict == d2_verdict == "DEFECT-REFUTED" else
                "MIXED" if {d1_verdict, d2_verdict} == {"DEFECT-CONFIRMED", "DEFECT-REFUTED"} else
                "INCONCLUSIVE"),
    "runtime_s": time.time() - t_start,
}
with open(os.path.join(OUT, "u14g_results.json"), "w", encoding="utf8") as f:
    json.dump(result, f, indent=1)
print("overall:", result["overall"], f"runtime {result['runtime_s']:.0f}s")
