"""U7 — assemble v2 (repair pass after three skeptic reviews). Writes u7_results_v2.json and u7_arrival_exponents_v2.png.

Changes against u7_assemble.py (kept unchanged):
  * every SPH arrival statistic is reported with every discriminator (u7_sph_stats_v2.py), not only the favourable one;
  * post-hoc verdicts are formed from SAME-statistic pairs (heat half-peak with SPH half-peak, heat time-to-peak with
    SPH pulse-peak), and the pre-registered statistic on the plane source is reported;
  * new engine runs (all with the unmodified repo engine via u7_sph.py): half-amplitude plane source (linearity of the
    plane-source statistics) and wider free bars 27x15x15 and 27x21x21 (does the 1.24 c_Tait excess depend on the
    cross-section?);
  * verifier re-implementation results are cited with their files and tagged as such;
  * traceable repair numbers from u7_repairs_v2.json (7^3 density deficit, IC mass range, Lindemann flag, heat literal).
"""
import json, math, os, sys
import numpy as np
REPO = r"D:/Aaron/development/sigma-ground"
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + "/tools")
OUT = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U7"
sys.path.insert(0, OUT)
from mentat_client import mentat
from sigma_ground.dynamics.continuum import material_properties
from sigma_ground.field.interface.thermal import sound_velocity
import u7_sph_analyse as SA
import u7_sph_stats_v2 as ST

MINUS = "\u2212"
def fm(x, nd=2):
    return f"{x:.{nd}f}".replace("-", MINUS)

def ols(x, y):
    A = np.vstack([np.log(x), np.ones_like(x)]).T
    return float(np.linalg.lstsq(A, np.log(y), rcond=None)[0][0])

# ── Mentat leaves (PLATINUM 21) ──
leaves = {}
for key, tool, args in (("c", "lookup_constant", {"name": "c"}), ("G", "lookup_constant", {"name": "G"}),
                        ("au", "lookup_constant", {"name": "au"}),
                        ("c_s_air_293K", "speed_of_sound_in_ideal_gas", {"temperature_k": 293.15}),
                        ("T_melt_water", "melting_point", {"material": "water"}),
                        ("T_melt_iron", "melting_point", {"material": "iron"}),
                        ("rho_iron", "density", {"material": "iron"})):
    v, src = mentat(tool, args)
    leaves[key] = dict(value=v, source=src, tool=tool, args=args)
sun, sun_src = mentat("solar_system_body", {"body_name": "Sun"})
leaves["M_sun"] = dict(value=sun["mass_kg"], source=sun_src, tool="solar_system_body", args={"body_name": "Sun"})
c = leaves["c"]["value"]
lt, lt_src = mentat("light_travel_time", {"distance_m": 16 * leaves["au"]["value"]})
leaves["light_travel_time_16au"] = dict(value=lt, source=lt_src, tool="light_travel_time", args={"distance_m": 16 * leaves["au"]["value"]})
casc = {}
for mat in ("water_ice", "iron"):
    pr = material_properties(mat, 300.0)
    casc[mat] = dict(rho0=pr["density_kg_m3"], K=pr["bulk_modulus_Pa"], G=pr["shear_modulus_Pa"],
                     c_tait=math.sqrt(pr["bulk_modulus_Pa"] / pr["density_kg_m3"]), c_L=pr["sound_speed_m_s"],
                     v_D=sound_velocity(mat), T_melt_cascade=pr["melting_point_K"],
                     source="sigma_ground.dynamics.continuum.material_properties (field.interface cascade); "
                            "v_D = field.interface.thermal.sound_velocity")
ratios = {f"c/c_tait({m})": c / casc[m]["c_tait"] for m in casc}
ratios.update({f"c/c_L({m})": c / casc[m]["c_L"] for m in casc})
ratios["c/c_s(air, 293.15 K, Mentat)"] = c / leaves["c_s_air_293K"]["value"]
ratios["c_s at the design's lower bound c/2e5 (m/s)"] = c / 2e5
ratios["design_range_note"] = ("design range 2e5-1e8 was stated for fluids; Mentat air (8.74e5) lies inside it; "
                               "only the cascade solids (5.0e4-8.5e4) fall below it; c/2e5 = 1499 m/s (about liquid water)")

# ── heat ──
H = json.load(open(f"{OUT}/heat_results.json"))
HR = json.load(open(f"{OUT}/heat_refine_results.json"))
RP = json.load(open(f"{OUT}/u7_repairs_v2.json"))
def heat_models(rows, key):
    dd = np.arange(2, 11, dtype=float)
    return SA.model_compare(dd, np.array([rows[str(d)][key] for d in range(2, 11)]))
heat = dict(
    literal=dict(p=H["literal"]["p"], arrivals=H["literal"]["arrivals"], distances_crossing=H["literal"]["distances_crossing"],
                 continuum_expectation=RP["E_heat_literal"]),
    corrected_prereg=dict(p_half_dx1=H["corrected"]["p_half"], p_peak_dx1=H["corrected"]["p_peak"]),
    refinement=dict(p_half_dx1=HR["dx=1"]["p_half"], p_half_dx05=HR["dx=0.5"]["p_half"],
                    p_peak_dx1=HR["dx=1"]["p_peak"], p_peak_dx05=HR["dx=0.5"]["p_peak"],
                    t_half_d10_dx1=HR["dx=1"]["rows"]["10"]["t_half"], t_half_d10_dx05=HR["dx=0.5"]["rows"]["10"]["t_half"],
                    t_half_d10_continuum=HR["continuum"]["10"]["t_half"],
                    models_half_dx05=heat_models(HR["dx=0.5"]["rows"], "t_half"),
                    models_half_dx1=heat_models(HR["dx=1"]["rows"], "t_half"),
                    fit_improvement_factor_dx05=1.0 / heat_models(HR["dx=0.5"]["rows"], "t_half")["rms_ratio_parabolic_over_hyperbolic"]),
    threshold_sweep={k: v["p"] for k, v in H["threshold_sweep"].items()},
    threshold_sweep_note=("continuum absolute-threshold exponent is p = 2/(1 - 3/(2L)), L = d^2/(4Dt) (Mentat-derived, "
                          "u7_repairs_v2.json E): always > 2 and diverging near the reach limit, so 4.6 at 1e-3 is continuum "
                          "behaviour plus an early discrete near field at d = 2; values below 2 are the explicit stencil; the "
                          "plateau at 1.34 for 1e-20..1e-80 is an artefact of linear interpolation from a zero sample; "
                          "p = 1.00 is the first-nonzero (one cell per substep) light cone only"),
    explicit_lightcone_p=ols(np.arange(2, 11, dtype=float),
                             np.array([H["explicit_lightcone_first_nonzero"][str(d)] for d in range(2, 11)])),
    energy_conservation=H["corrected"]["energy_conservation"])
th_dx05 = {d: HR["dx=0.5"]["rows"][str(d)]["t_half"] for d in range(2, 11)}
heat["increment_trend_half_dx05_d4_10"] = (th_dx05[10] - th_dx05[7]) / (th_dx05[7] - th_dx05[4])
tp_dx05 = {d: HR["dx=0.5"]["rows"][str(d)]["t_peak"] for d in range(2, 11)}
heat["increment_trend_peak_dx05_d4_10"] = (tp_dx05[10] - tp_dx05[7]) / (tp_dx05[7] - tp_dx05[4])
heat["increment_trend_parabolic_expectation"] = ST.PARABOLIC_TREND

# ── SPH: literal 7^3 ──
lit_r = np.load(f"{OUT}/sph_lit7_ref.npz"); lit_p = np.load(f"{OUT}/sph_lit7_pert.npz")
ijk = [tuple(int(v) for v in x) for x in lit_p["ijk"]]; cc = tuple(int(v) for v in lit_p["centre"])
Vp = np.linalg.norm(lit_p["V"], axis=2); v0 = Vp[:, ijk.index(cc)].max()
lit_steps = {d: int(np.nonzero(Vp[:, ijk.index((cc[0] + d, cc[1], cc[2]))] > 1e-3 * v0)[0][0]) for d in (1, 2, 3)}
lit_meta = json.load(open(f"{OUT}/sph_lit7_pert.json"))
sph_lit = dict(arrival_step=lit_steps,
               p=ols(np.array([1., 2., 3.]), np.array([lit_steps[d] * lit_meta["dt"] for d in (1, 2, 3)])),
               ref_max_speed=float(np.linalg.norm(lit_r["V"], axis=2).max()), source_peak_speed=float(v0),
               t0_density_deficit_uniform_mass=RP["A_lit7_uniform_mass_t0"])

# ── SPH: bar runs, every statistic ──
pairs = {"point_a1_11x11": ("bar_w_ref", "bar_w_pert"), "point_a0_11x11": ("bar_w_ref_a0", "bar_w_pert_a0"),
         "plane_a1_11x11": ("bar_w_ref", "bar_w_plane"), "plane_a0_11x11": ("bar_w_ref_a0", "bar_w_plane_a0"),
         "plane_a1_11x11_dt2": ("bar_w_ref_a1_dt2", "bar_w_plane_a1_dt2"), "iron_plane_a1_11x11": ("bar_fe_ref", "bar_fe_plane"),
         "plane_a1_11x11_half_amplitude_v2": ("bar_w_ref", "v2_bar_w_plane_half"),
         "plane_a1_15x15_v2": ("v2_bar15_ref", "v2_bar15_plane"), "plane_a1_21x21_v2": ("v2_bar21_ref", "v2_bar21_plane")}
sph = {}
for name, (r_, p_) in pairs.items():
    if not (os.path.exists(f"{OUT}/sph_{r_}.npz") and os.path.exists(f"{OUT}/sph_{p_}.npz")):
        sph[name] = dict(missing=True, ref=r_, pert=p_)
        continue
    sph[name] = ST.run_table(r_, p_)
    if sph[name]["source"] == "plane":
        try:
            sph[name]["spatial_peak_speed_over_c_tait"] = SA.track_spatial_peak(r_, p_)["speed_over_c_tait"]
        except Exception as e:      # noqa: BLE001 - diagnostic only
            sph[name]["spatial_peak_speed_over_c_tait"] = None
            sph[name]["spatial_peak_error"] = repr(e)
    sph[name]["wall_seconds"] = json.load(open(f"{OUT}/sph_{p_}.json"))["wall_seconds"]
lin_point = SA.linearity("bar_w_ref", "bar_w_pert", "bar_w_half")
lin_plane = SA.linearity("bar_w_ref", "bar_w_plane", "v2_bar_w_plane_half")
sph_op = json.load(open(f"{OUT}/sph_operator_check.json"))
sph_disp = json.load(open(f"{OUT}/sph_dispersion.json"))
lin1d = ST.linear_1d_table()

# ── verifier results cited (not re-run here; their files are named) ──
VP = json.load(open(f"{OUT}/verify_physics/vp_bar_results.json"))
VC = json.load(open(f"{OUT}/verify_physics/vp_chain_results.json"))
SKR = json.load(open(f"{OUT}/verify_skeptic/sk_rows.json"))
SKO = json.load(open(f"{OUT}/verify_skeptic/sk_operator_rows.json"))
verifier = dict(
    note="[DERIVED numerics by the physics verifier] numpy re-implementation of continuum_step's force path, validated "
         "against the engine runs of this item to 1e-12 m/s (verify_physics/vp_bar.py, vp_bar_results.json); chain model "
         "verify_physics/vp_chain.py; per-row statistics verify_skeptic/sk_rows.py, sk_operator_rows.py",
    reimplementation_validation=VP["validation"],
    bar_11_free_a0=VP["mine_bar_11_free_a0"], bar_21_free_a1=VP["mine_bar_21_free_a1"],
    bar_periodicYZ_a1=VP["mine_bar_periodicYZ_a1"], bar_periodicYZ_a1_ref_max_speed=VP["mine_bar_periodicYZ_a1_ref_max"],
    periodic_excluded_reason=("reference run of the lateral-periodic toggle is not quiescent: max reference speed "
                              f"{VP['mine_bar_periodicYZ_a1_ref_max']:.0f} m/s (11x11 free bar: 36 m/s; signal 3-8 m/s), "
                              "because the x-end slabs reuse masses solved for a free-sided bar; its 0.883 c_Tait is not "
                              "used as evidence here"),
    chain_a1_rectified_d4_10=VC["water_ice_a1_rectified_dt1"]["fits"]["4-10"],
    chain_a1_rectified_d100_150=VC["water_ice_a1_rectified_dt1"]["fits"]["100-150"],
    chain_a0_d4_10=VC["water_ice_a0_none_dt1"]["fits"]["4-10"],
    chain_a0_d100_150=VC["water_ice_a0_none_dt1"]["fits"]["100-150"],
    per_row_dynamic_pulse_peak_speed_ice={k: v["speed_mean"] for k, v in SKR["ice_plane_a1"]["by_lateral_depth"].items()},
    per_row_static_operator_speed={k: v["vphase_mean"] for k, v in SKO["by_lateral_depth"].items()})

# ── N-body ──
NB = json.load(open(f"{OUT}/nbody_results.json"))
p_n = NB["summary"]["p_nbody"]

# ── verdicts ──
def band(x, centre, tol):
    return abs(x - centre) <= tol

def verdict(label, p_h, p_s, speed_ratio, statistic):
    sep = band(p_h, 2, 0.2) and band(p_s, 1, 0.1) and band(speed_ratio, 1, 0.10) and abs(p_n) < 1e-9
    ps = [p_h, p_s, p_n]
    same = (max(ps) - min(ps)) <= 0.4
    return dict(label=label, statistic=statistic,
                verdict="SEPARATE" if sep else ("SAME" if same else "MIXED (neither band set met)"),
                heat_in_band=band(p_h, 2, 0.2), sph_p_in_band=band(p_s, 1, 0.1),
                sph_speed_in_band=(band(speed_ratio, 1, 0.10) if speed_ratio == speed_ratio else None),
                nbody_zero_by_construction=abs(p_n) < 1e-9, p=[p_h, p_s, p_n], sph_speed_over_c_tait=speed_ratio,
                heat_minus_sph=p_h - p_s, same_window_full_width=0.4)

S = lambda run, key: sph[run]["stats"][key]
verdicts = dict(
    literal=verdict("literal design", heat["literal"]["p"], sph_lit["p"], float("nan"), "1e-3 thresholds"),
    prereg_corrected=verdict("pre-registered corrected (heat dx = 1, SPH point source)", heat["corrected_prereg"]["p_half_dx1"],
                             S("point_a1_11x11", "t_0.5")["p_4_10"], S("point_a1_11x11", "t_0.5")["speed_over_c_tait"], "half-peak"),
    posthoc_half_peak=verdict("post hoc: heat dx = 0.5, SPH plane source, SAME pre-registered half-peak statistic",
                              heat["refinement"]["p_half_dx05"], S("plane_a1_11x11", "t_0.5")["p_4_10"],
                              S("plane_a1_11x11", "t_0.5")["speed_over_c_tait"], "half-peak"),
    posthoc_time_to_peak=verdict("post hoc: heat dx = 0.5 time-to-peak, SPH plane source pulse-peak",
                                 heat["refinement"]["p_peak_dx05"], S("plane_a1_11x11", "t_peak")["p_4_10"],
                                 S("plane_a1_11x11", "t_peak")["speed_over_c_tait"], "time-to-peak / pulse-peak"))
if not sph["plane_a1_21x21_v2"].get("missing"):
    verdicts["posthoc_wide_bar_time_to_peak"] = verdict(
        "post hoc: heat dx = 0.5 time-to-peak, SPH plane source pulse-peak in the 27x21x21 bar (three post-hoc changes)",
        heat["refinement"]["p_peak_dx05"], S("plane_a1_21x21_v2", "t_peak")["p_4_10"],
        S("plane_a1_21x21_v2", "t_peak")["speed_over_c_tait"], "time-to-peak / pulse-peak")
    verdicts["posthoc_wide_bar_half_peak"] = verdict(
        "post hoc: heat dx = 0.5 half-peak, SPH plane source half-peak in the 27x21x21 bar",
        heat["refinement"]["p_half_dx05"], S("plane_a1_21x21_v2", "t_0.5")["p_4_10"],
        S("plane_a1_21x21_v2", "t_0.5")["speed_over_c_tait"], "half-peak")
verdicts["operator_longwave_speed_over_c_tait_BY_CONSTRUCTION"] = sph_op["0.2"]["vphase_engine_over_c_tait"]

out = dict(id="U7", version="v2 (repair pass)", mentat_leaves=leaves, cascade=casc, speed_ratios=ratios,
           lit=dict(GW170817="Abbott et al. 2017, ApJL 848, L13, doi:10.3847/2041-8213/aa920c, arXiv:1710.05834: "
                             "-3e-15 < (v_GW - v_EM)/v_EM < +7e-16 from the 1.74 +/- 0.05 s GW-GRB delay "
                             "[LIT, verified by web search 2026-10-10 by the runner; re-checked by the physics verifier by "
                             "direct fetch of the arXiv abstract page]"),
           heat=heat,
           sph=dict(literal_7cubed=sph_lit, bar=sph,
                    linearity_ratio_full_over_half=dict(point_source=lin_point, plane_source_v2=lin_plane),
                    operator_check_BY_CONSTRUCTION=sph_op,
                    dispersion=dict(water_ice={k: v for k, v in sph_disp["water_ice"].items() if not isinstance(v, list)}),
                    linear_1d_model=lin1d,
                    ic_masses=RP["B_ic_masses"],
                    background_27x11x11=RP["F_background_27x11x11_a1"]),
           verifier_cited=verifier,
           nbody=NB["summary"],
           nbody_slopes=dict(longitudinal=NB["results"]["newton"]["longitudinal"]["slope"],
                             transverse=NB["results"]["newton"]["transverse"]["slope"],
                             ratio_to_closed_form_longitudinal=NB["results"]["newton"]["longitudinal"]["ratio_to_closed_form"]),
           engine_defect_evidence=dict(lindemann=RP["C_lindemann_water_ice"], shear_free_speed=RP["D_shear_free_speed"]),
           verdicts=verdicts)
json.dump(out, open(f"{OUT}/u7_results_v2.json", "w"), indent=1, default=float)

# ── figure ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"
plt.rcParams.update({"axes.unicode_minus": True, "font.size": 9.5, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": INK2})
fig, axs = plt.subplots(2, 2, figsize=(14.0, 11.0), dpi=160)
(ax, bx), (cx, dx_) = axs
fig.patch.set_facecolor(SURF)
for a in axs.flat:
    a.set_facecolor(SURF)
    a.grid(color=GRID, lw=0.6, which="both")

# (a) arrival time against distance
ax.set_xscale("log"); ax.set_yscale("log")
d_h = np.arange(2, 11)
th05 = np.array([HR["dx=0.5"]["rows"][str(d)]["t_half"] for d in d_h]); th1 = np.array([HR["dx=1"]["rows"][str(d)]["t_half"] for d in d_h])
ax.plot(d_h / 2, th1 / th1[0], "o--", color=C1, lw=1.2, ms=4, mfc=SURF,
        label=f"heat, dx = 1, half-peak (pre-registered): p = {fm(heat['corrected_prereg']['p_half_dx1'])}")
ax.plot(d_h / 2, th05 / th05[0], "o-", color=C1, lw=2, ms=5,
        label=f"heat, dx = 0.5, half-peak (post hoc): p = {fm(heat['refinement']['p_half_dx05'])}")
d_s = np.arange(4, 11)
def tser(run, key):
    return np.array([S(run, key)["t_us"][d] for d in d_s])
for run, key, sty, col, lab in (
        ("point_a1_11x11", "t_0.5", "s--", C4, "SPH point source, half-peak (pre-registered)"),
        ("plane_a1_11x11", "t_0.5", "s--", C2, "SPH plane source, half-peak (post hoc)"),
        ("plane_a1_11x11", "t_peak", "s-", C2, "SPH plane source, pulse-peak (post hoc)")):
    t = tser(run, key)
    ax.plot(d_s / 4, t / t[0], sty, color=col, lw=2 if sty == "s-" else 1.3, ms=5 if sty == "s-" else 4,
            mfc=col if sty == "s-" else SURF, label=f"{lab}: p = {fm(S(run, key)['p_4_10'])}")
d_n = np.array([1, 2, 4, 8, 16]); tn = np.array(NB["summary"]["t_arr_by_dt"]["86400s"])
ax.plot(d_n, tn / tn[0], "^-", color=C3, lw=2, ms=6, label="N-body: every distance in the first step, p = 0 (by construction)")
xg = np.array([1, 16.0])
for slope, xt in ((2, 6.2), (1, 12.0), (0, 12.0)):
    ax.plot(xg, xg ** slope, ":", color=INK2, lw=0.9)
    ax.text(xt, min(xt ** slope, 34) * (0.62 if slope == 2 else 1.12), f"slope {slope}", color=INK2, fontsize=8)
ax.set_xlim(0.9, 18); ax.set_ylim(0.22, 40)
ax.set_xlabel("distance / first distance of the window  (heat 2\u201310 dx, SPH 4\u201310 spacings, N-body 1\u201316 au)")
ax.set_ylabel("arrival time / arrival time at the first distance")
ax.set_title("(a) arrival time against distance", fontsize=10, color=INK, loc="left")
ax.legend(frameon=True, facecolor=SURF, edgecolor="none", framealpha=0.92, fontsize=7.4, loc="lower right",
          bbox_to_anchor=(0.995, 0.005))

# (b) SPH exponent by statistic
labels = ["10 % of peak", "25 % of peak", "half-peak\n(pre-registered)", "pulse peak"]
keys = ["t_0.1", "t_0.25", "t_0.5", "t_peak"]
xs = np.arange(4)
bx.axhspan(0.9, 1.1, color=GRID, alpha=0.9, lw=0)
bx.text(3.35, 1.0, "SPH band\n1 \u00b1 0.1", color=INK2, fontsize=8, va="center", ha="left")
bx.axhspan(1.8, 2.2, color=GRID, alpha=0.45, lw=0)
bx.text(3.35, 2.0, "heat band\n2 \u00b1 0.2", color=INK2, fontsize=8, va="center", ha="left")
series_b = [("point_a1_11x11", C4, "s--", "engine, point source, 27\u00d711\u00d711 bar"),
            ("plane_a1_11x11", C2, "s-", "engine, plane source, 27\u00d711\u00d711 bar"),
            ("plane_a1_21x21_v2", C1, "o-", "engine, plane source, 27\u00d721\u00d721 bar (v2)")]
for run, col, sty, lab in series_b:
    if sph[run].get("missing"):
        continue
    bx.plot(xs, [S(run, k)["p_4_10"] for k in keys], sty, color=col, lw=1.8, ms=6, mfc=col if "-" == sty[1:] else SURF, label=lab)
bx.plot(xs, [lin1d["stats"][k]["p_4_10"] for k in keys], "D:", color=C3, lw=1.4, ms=5,
        label="1-D linear chain, no surfaces or viscosity (control)")
bx.set_xticks(xs); bx.set_xticklabels(labels, fontsize=8.5)
bx.set_xlim(-0.3, 4.1); bx.set_ylim(0.4, 3.3)
bx.set_ylabel("SPH arrival exponent p over 4\u201310 spacings")
bx.set_title("(b) the SPH exponent depends on which arrival statistic is used", fontsize=10, color=INK, loc="left")
bx.legend(frameon=False, fontsize=7.6, loc="upper right")

# (c) SPH speed against bar width
widths, vpk, vhf = [], [], []
for run, w in (("plane_a1_11x11", 11), ("plane_a1_15x15_v2", 15), ("plane_a1_21x21_v2", 21)):
    if sph[run].get("missing"):
        continue
    widths.append(w); vpk.append(S(run, "t_peak")["speed_over_c_tait"]); vhf.append(S(run, "t_0.5")["speed_over_c_tait"])
cx.axhspan(0.9, 1.1, color=GRID, alpha=0.9, lw=0)
cx.plot(widths, vpk, "s-", color=C2, lw=2, ms=6, label="engine, pulse-peak arrival")
cx.plot(widths, vhf, "s--", color=C4, lw=1.5, ms=5, mfc=SURF, label="engine, half-peak arrival (pre-registered statistic)")
x_inf = 28
ch = verifier["chain_a1_rectified_d4_10"]
cx.plot([x_inf], [ch["t_peak"]["speed_over_c_tait"]], "D", color=C2, ms=7, mfc=SURF)
cx.plot([x_inf], [ch["t_0.5"]["speed_over_c_tait"]], "D", color=C4, ms=6, mfc=SURF)
cx.text(x_inf, ch["t_peak"]["speed_over_c_tait"] - 0.06, "laterally infinite\n(verifier's chain model,\nengine viscosity)",
        color=INK2, fontsize=7.6, ha="center", va="top")
cx.plot([21], [VP["mine_bar_21_free_a1"]["t_peak"]["speed_over_c_tait"]], "o", color=C1, ms=9, mfc="none", mew=1.4,
        label="verifier's re-implementation, 27\u00d721\u00d721")
cx.axhline(sph_op["0.2"]["vphase_engine_over_c_tait"], color=INK2, lw=0.9, ls=":")
cx.text(10.3, sph_op["0.2"]["vphase_engine_over_c_tait"] - 0.035, "engine pressure operator, long waves (0.99)", color=INK2, fontsize=7.8)
cx.set_xlim(9.5, 31); cx.set_ylim(0.8, 1.5)
cx.set_xticks([11, 15, 21, x_inf]); cx.set_xticklabels(["11", "15", "21", "\u221e"])
cx.set_xlabel("bar cross-section (lattice spacings per side), plane source, \u03b1 = 1")
cx.set_ylabel("SPH arrival speed / c_Tait  (fit over 4\u201310 spacings)")
cx.set_title("(c) the 1.24 c_Tait excess shrinks as the free side surfaces move away", fontsize=10, color=INK, loc="left")
cx.legend(frameon=False, fontsize=7.6, loc="upper right")

# (d) dt test
dx_.set_xscale("log"); dx_.set_yscale("log")
dts_n = np.array([86400, 43200, 600, 60.0])
tn16 = np.array([NB["summary"]["t_arr_by_dt"][f"{int(x)}s"][-1] for x in dts_n])
dx_.plot(dts_n / dts_n[0], tn16 / tn16[0], "^-", color=C3, lw=2, ms=6, label="N-body at 16 au: arrival = one step, \u221d dt")
sp1 = S("plane_a1_11x11", "t_peak")["t_us"][10]; sp2 = S("plane_a1_11x11_dt2", "t_peak")["t_us"][10]
dx_.plot([1, 0.5], [1, sp2 / sp1], "s-", color=C2, lw=2, ms=6,
         label=f"SPH pulse peak at 10 spacings: dt \u2192 dt/2 moves it {fm(100 * (sp2 / sp1 - 1))} %")
h1 = HR["dx=1"]["rows"]["10"]["t_half"]; h2 = HR["dx=0.5"]["rows"]["10"]["t_half"]
dx_.plot([1, HR["dx=0.5"]["dt_sub"] / HR["dx=1"]["dt_sub"]], [1, h2 / h1], "o-", color=C1, lw=2, ms=6,
         label=f"heat half-peak at 10 dx: dx \u2192 dx/2 (dt \u2192 dt/4) moves it +{fm(100 * (h2 / h1 - 1), 1)} %")
light = lt / tn16[0]
dx_.axhline(light, color=INK2, lw=0.9, ls=":")
dx_.text(7e-4, light * 1.3, "light-travel time to 16 au (Mentat light_travel_time)", color=INK2, fontsize=8)
dx_.set_xlim(5e-4, 2); dx_.set_ylim(3e-4, 3)
dx_.set_xlabel("time step / coarsest time step")
dx_.set_ylabel("arrival time / arrival time at the coarsest step")
dx_.set_title("(d) is the arrival set by the physics or by the step?", fontsize=10, color=INK, loc="left")
dx_.legend(frameon=False, fontsize=7.6, loc="lower right")

vp = verdicts["prereg_corrected"]
pk11 = S("plane_a1_11x11", "t_peak")
line3 = ""
if not sph["plane_a1_21x21_v2"].get("missing"):
    pk21 = S("plane_a1_21x21_v2", "t_peak")
    line3 = (f"The SPH pulse-peak speed falls from {fm(pk11['speed_over_c_tait'])} c_Tait in the 11\u00d711 bar to "
             f"{fm(pk21['speed_over_c_tait'])} in the 21\u00d721 bar: the excess shrinks as the free sides move away; viscosity is not the cause "
             f"(\u03b1 = 0 gives {fm(S('plane_a0_11x11', 't_peak')['speed_over_c_tait'])}).")
fig.suptitle("U7 (repaired): heat arrives as d\u00b2, SPH as a finite-speed pulse, N-body everywhere in one step (by construction); "
             "the pre-registered bands are missed\n"
             f"Pre-registered statistics: heat p = {fm(vp['p'][0])} (dx = 1), SPH p = {fm(vp['p'][1])} at "
             f"{fm(vp['sph_speed_over_c_tait'])} c_Tait, so MIXED. SPH reaches p \u2248 1 only with two post-hoc changes "
             f"(plane source and pulse-peak statistic: {fm(pk11['p_4_10'])});\nwith the pre-registered half-peak statistic the plane source "
             f"still gives {fm(S('plane_a1_11x11', 't_0.5')['p_4_10'])}. " + line3,
             fontsize=9.8, color=INK, x=0.01, y=0.995, ha="left", va="top")
fig.subplots_adjust(left=0.06, right=0.985, bottom=0.06, top=0.895, wspace=0.2, hspace=0.27)
fig.savefig(f"{OUT}/u7_arrival_exponents_v2.png", facecolor=SURF, bbox_inches="tight")

summary = dict(verdicts=verdicts,
               sph={k: ({kk: dict(p=vv["p_4_10"], p_5_10=vv["p_5_10"], p_7_10=vv["p_7_10"], v=vv["speed_over_c_tait"],
                                  rms=vv["rms_ratio_parabolic_over_hyperbolic"], trend=vv["increment_trend"], mono=vv["monotone"])
                              for kk, vv in v["stats"].items()} if not v.get("missing") else "missing")
                    for k, v in sph.items()},
               spatial={k: v.get("spatial_peak_speed_over_c_tait") for k, v in sph.items() if not v.get("missing")},
               refmax={k: (v.get("ref_max_speed_axis"), v.get("ref_max_speed_any")) for k, v in sph.items() if not v.get("missing")},
               lin_plane=lin_plane, lin1d={k: (v["p_4_10"], v["speed_over_c_tait"], v["increment_trend"], v["monotone"]) for k, v in lin1d["stats"].items()},
               heat_trends=(heat["increment_trend_half_dx05_d4_10"], heat["increment_trend_peak_dx05_d4_10"]),
               fit_factor=heat["refinement"]["fit_improvement_factor_dx05"])
print(json.dumps(summary, indent=1, default=float))
