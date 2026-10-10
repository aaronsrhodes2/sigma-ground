"""U7 — assemble: Mentat leaves (with sources), the three solver legs, pre-registered and corrected verdicts, figure."""
import json, math, sys
import numpy as np
sys.path.insert(0, r"D:/Aaron/development/sigma-ground")
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
OUT = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U7"
sys.path.insert(0, OUT)
from mentat_client import mentat
from sigma_ground.dynamics.continuum import material_properties
from sigma_ground.field.interface.thermal import sound_velocity
import u7_sph_analyse as SA

MINUS = "\u2212"


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

# ── heat ──
H = json.load(open(f"{OUT}/heat_results.json"))
HR = json.load(open(f"{OUT}/heat_refine_results.json"))


def heat_models(rows, key):
    dd = np.arange(2, 11, dtype=float)
    tt = np.array([rows[str(d)][key] for d in range(2, 11)])
    return SA.model_compare(dd, tt)


heat = dict(
    literal=dict(p=H["literal"]["p"], arrivals=H["literal"]["arrivals"], distances_crossing=H["literal"]["distances_crossing"]),
    corrected_prereg=dict(p_half_dx1=H["corrected"]["p_half"], p_peak_dx1=H["corrected"]["p_peak"]),
    refinement=dict(p_half_dx1=HR["dx=1"]["p_half"], p_half_dx05=HR["dx=0.5"]["p_half"],
                    p_peak_dx1=HR["dx=1"]["p_peak"], p_peak_dx05=HR["dx=0.5"]["p_peak"],
                    t_half_d10_dx1=HR["dx=1"]["rows"]["10"]["t_half"], t_half_d10_dx05=HR["dx=0.5"]["rows"]["10"]["t_half"],
                    t_half_d10_continuum=HR["continuum"]["10"]["t_half"],
                    models_half_dx05=heat_models(HR["dx=0.5"]["rows"], "t_half"),
                    models_half_dx1=heat_models(HR["dx=1"]["rows"], "t_half")),
    threshold_sweep={k: v["p"] for k, v in H["threshold_sweep"].items()},
    explicit_lightcone_p=ols(np.arange(2, 11, dtype=float),
                             np.array([H["explicit_lightcone_first_nonzero"][str(d)] for d in range(2, 11)])),
    energy_conservation=H["corrected"]["energy_conservation"])

# ── SPH ──
lit_r = np.load(f"{OUT}/sph_lit7_ref.npz")
lit_p = np.load(f"{OUT}/sph_lit7_pert.npz")
ijk = [tuple(int(v) for v in x) for x in lit_p["ijk"]]
cc = tuple(int(v) for v in lit_p["centre"])
Vp = np.linalg.norm(lit_p["V"], axis=2)
v0 = Vp[:, ijk.index(cc)].max()
lit_steps = {d: int(np.nonzero(Vp[:, ijk.index((cc[0] + d, cc[1], cc[2]))] > 1e-3 * v0)[0][0]) for d in (1, 2, 3)}
lit_meta = json.load(open(f"{OUT}/sph_lit7_pert.json"))
sph_lit = dict(arrival_step=lit_steps,
               p=ols(np.array([1., 2., 3.]), np.array([lit_steps[d] * lit_meta["dt"] for d in (1, 2, 3)])),
               ref_max_speed=float(np.linalg.norm(lit_r["V"], axis=2).max()), source_peak_speed=float(v0))
runs = {"point_a1": ("bar_w_ref", "bar_w_pert"), "point_a0": ("bar_w_ref_a0", "bar_w_pert_a0"),
        "plane_a1": ("bar_w_ref", "bar_w_plane"), "plane_a0": ("bar_w_ref_a0", "bar_w_plane_a0"),
        "plane_a1_dt2": ("bar_w_ref_a1_dt2", "bar_w_plane_a1_dt2"), "iron_plane_a1": ("bar_fe_ref", "bar_fe_plane")}
sph = {}
for name, (r_, p_) in runs.items():
    o = SA.analyse(r_, p_)
    sph[name] = dict(material=o["material"], source=o["source"], alpha_visc=o["alpha_visc"], dt=o["dt"],
                     fits={k: dict(p_loglog=v["p_loglog"], speed_over_c_tait=v["speed_over_c_tait"],
                                   offset_d0_over_s=v["offset_d0_over_s"],
                                   rms_ratio_parabolic_over_hyperbolic=v["models"]["rms_ratio_parabolic_over_hyperbolic"])
                           for k, v in o["fits"].items()},
                     t_peak={d: o["rows"][d]["t_peak"] for d in range(4, 11)},
                     t_half={d: o["rows"][d]["t_0.5"] for d in range(4, 11)},
                     ref_max_speed_axis=o["ref_max_speed_axis"], v0_source_peak=o["v0_source_peak"])
    if o["source"] == "plane":
        sph[name]["spatial_peak_speed_over_c_tait"] = SA.track_spatial_peak(r_, p_)["speed_over_c_tait"]
sph_lin = SA.linearity("bar_w_ref", "bar_w_pert", "bar_w_half")
sph_op = json.load(open(f"{OUT}/sph_operator_check.json"))
sph_disp = json.load(open(f"{OUT}/sph_dispersion.json"))
sph_1d = json.load(open(f"{OUT}/sph_1d_linear.json"))
masses_info = json.load(open(f"{OUT}/masses_bar27_water_ice.json"))

# ── N-body ──
NB = json.load(open(f"{OUT}/nbody_results.json"))


# ── verdicts ──
def band(x, centre, tol):
    return abs(x - centre) <= tol


def verdict(p_h, p_s, speed_ratio, p_n):
    sep = band(p_h, 2, 0.2) and band(p_s, 1, 0.1) and band(speed_ratio, 1, 0.10) and abs(p_n) < 1e-9
    ps = [p_h, p_s, p_n]
    same = all(abs(p - np.mean(ps)) <= 0.2 for p in ps)
    return ("SEPARATE" if sep else ("SAME" if same else "MIXED (neither band set met)"),
            dict(heat_in_band=band(p_h, 2, 0.2), sph_p_in_band=band(p_s, 1, 0.1),
                 sph_speed_in_band=band(speed_ratio, 1, 0.10) if speed_ratio == speed_ratio else None,
                 nbody_zero=abs(p_n) < 1e-9, p=[p_h, p_s, p_n], sph_speed_over_c_tait=speed_ratio))


p_n = NB["summary"]["p_nbody"]
v_literal = verdict(heat["literal"]["p"], sph_lit["p"], float("nan"), p_n)
v_prereg = verdict(heat["corrected_prereg"]["p_half_dx1"], sph["point_a1"]["fits"]["t_0.5"]["p_loglog"],
                   sph["point_a1"]["fits"]["t_0.5"]["speed_over_c_tait"], p_n)
v_posthoc = verdict(heat["refinement"]["p_half_dx05"], sph["plane_a1"]["fits"]["t_peak"]["p_loglog"],
                    sph["plane_a1"]["fits"]["t_peak"]["speed_over_c_tait"], p_n)
v_posthoc_speed_operator = sph_op["0.2"]["vphase_engine_over_c_tait"]


def increment_trend(ds, ts):
    """[post-hoc] inverse local speed dt/dd: mean over the far half of the window / mean over the near half.
    Hyperbolic (finite speed): 1. Parabolic (t ~ d^2): ratio of the mean distances of the two halves."""
    ds = np.asarray(ds, float); ts = np.asarray(ts, float)
    inc = np.diff(ts) / np.diff(ds); mid = 0.5 * (ds[1:] + ds[:-1]); n = len(inc) // 2
    return dict(ratio=float(inc[-n:].mean() / inc[:n].mean()), parabolic_expectation=float(mid[-n:].mean() / mid[:n].mean()),
                hyperbolic_expectation=1.0)


trends = dict(
    heat_half_dx05=increment_trend(range(2, 11), [HR["dx=0.5"]["rows"][str(d)]["t_half"] for d in range(2, 11)]),
    heat_half_dx05_window_4_10=increment_trend(range(4, 11), [HR["dx=0.5"]["rows"][str(d)]["t_half"] for d in range(4, 11)]),
    sph_plane_a1_peak=increment_trend(range(4, 11), [sph["plane_a1"]["t_peak"][d] for d in range(4, 11)]),
    sph_plane_a1_dt2_peak=increment_trend(range(4, 11), [sph["plane_a1_dt2"]["t_peak"][d] for d in range(4, 11)]),
    sph_iron_plane_peak=increment_trend(range(4, 11), [sph["iron_plane_a1"]["t_peak"][d] for d in range(4, 11)]),
    sph_point_a1_half=increment_trend(range(4, 11), [sph["point_a1"]["t_half"][d] for d in range(4, 11)]))

out = dict(id="U7", mentat_leaves=leaves, cascade=casc, speed_ratios=ratios,
           lit=dict(GW170817="Abbott et al. 2017, ApJL 848, L13, doi:10.3847/2041-8213/aa920c, arXiv:1710.05834: "
                             "-3e-15 < (v_gw - c)/c < +7e-16 from the 1.74 +/- 0.05 s GW-GRB delay "
                             "[LIT, verified by web search 2026-10-10]"),
           heat=heat,
           sph=dict(literal_7cubed=sph_lit, bar=sph, linearity_ratio_full_over_half=sph_lin,
                    operator_check=sph_op,
                    dispersion=dict(water_ice={k: v for k, v in sph_disp["water_ice"].items() if not isinstance(v, list)}),
                    linear_1d_model=sph_1d["fits"],
                    ic_masses=dict(res_max=masses_info["res_max"], res_rms=masses_info["res_rms"],
                                   uniform_mass_res_max=masses_info["uniform_mass_res_max"])),
           nbody=NB["summary"],
           nbody_slopes=dict(longitudinal=NB["results"]["newton"]["longitudinal"]["slope"],
                             transverse=NB["results"]["newton"]["transverse"]["slope"],
                             ratio_to_closed_form_longitudinal=NB["results"]["newton"]["longitudinal"]["ratio_to_closed_form"]),
           increment_trends_posthoc=trends,
           verdicts=dict(literal=v_literal, prereg_corrected=v_prereg, posthoc_refined_plane=v_posthoc,
                         posthoc_engine_longwave_speed_over_c_tait=v_posthoc_speed_operator))
json.dump(out, open(f"{OUT}/u7_results.json", "w"), indent=1, default=float)

# ── figure ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"
plt.rcParams.update({"axes.unicode_minus": True, "font.size": 9.5, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": INK2})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.4, 6.0), dpi=170, gridspec_kw=dict(width_ratios=[1.3, 1]))
fig.patch.set_facecolor(SURF)
for a in (ax, bx):
    a.set_facecolor(SURF)
    a.grid(color=GRID, lw=0.6, which="both")
    a.set_xscale("log")
    a.set_yscale("log")
# (a) arrival time vs distance, each series normalised to its first point
d_h = np.arange(2, 11)
th05 = np.array([HR["dx=0.5"]["rows"][str(d)]["t_half"] for d in d_h])
th1 = np.array([HR["dx=1"]["rows"][str(d)]["t_half"] for d in d_h])
ax.plot(d_h / 2, th05 / th05[0], "o-", color=C1, lw=2, ms=5,
        label=f"heat FVM, dx = 0.5: half-peak arrival, p = {heat['refinement']['p_half_dx05']:.2f}")
ax.plot(d_h / 2, th1 / th1[0], "o--", color=C1, lw=1.2, ms=4, mfc=SURF,
        label=f"heat FVM, dx = 1 (pre-registered): p = {heat['corrected_prereg']['p_half_dx1']:.2f}")
d_s = np.arange(4, 11)
tpk = np.array([sph["plane_a1"]["t_peak"][d] for d in d_s])
thf = np.array([sph["point_a1"]["t_half"][d] for d in d_s])
ax.plot(d_s / 4, tpk / tpk[0], "s-", color=C2, lw=2, ms=5,
        label=f"SPH, plane source: pulse-peak arrival, p = {sph['plane_a1']['fits']['t_peak']['p_loglog']:.2f}")
ax.plot(d_s / 4, thf / thf[0], "s--", color=C4, lw=1.2, ms=4, mfc=SURF,
        label=f"SPH, point source: half-peak (pre-registered), p = {sph['point_a1']['fits']['t_0.5']['p_loglog']:.2f}")
d_n = np.array([1, 2, 4, 8, 16])
tn = np.array(NB["summary"]["t_arr_by_dt"]["86400s"])
ax.plot(d_n, tn / tn[0], "^-", color=C3, lw=2, ms=6, label="N-body: every distance in the first force evaluation, p = 0")
xg = np.array([1, 16.0])
for slope, xt in ((2, 6.2), (1, 12.0), (0, 12.0)):
    ax.plot(xg, xg ** slope, ":", color=INK2, lw=0.9)
    ax.text(xt, min(xt ** slope, 34) * (0.62 if slope == 2 else 1.12), f"slope {slope}", color=INK2, fontsize=8)
ax.set_xlim(0.9, 18)
ax.set_ylim(0.6, 40)
ax.set_xlabel("distance / first distance of the window   (heat 2\u201310 dx, SPH 4\u201310 lattice spacings, N-body 1\u201316 au)")
ax.set_ylabel("arrival time / arrival time at the first distance")
ax.set_title("(a) arrival time against distance", fontsize=10, color=INK, loc="left")
ax.legend(frameon=False, fontsize=7.6, loc="upper left")
# (b) arrival time at the farthest distance against the time step
dts_n = np.array([86400, 43200, 600, 60.0])
tn16 = np.array([NB["summary"]["t_arr_by_dt"][f"{int(x)}s"][-1] for x in dts_n])
bx.plot(dts_n / dts_n[0], tn16 / tn16[0], "^-", color=C3, lw=2, ms=6, label="N-body at 16 au: arrival = one step, \u221d dt")
sp1 = sph["plane_a1"]["t_peak"][10]
sp2 = sph["plane_a1_dt2"]["t_peak"][10]
bx.plot([1, 0.5], [1, sp2 / sp1], "s-", color=C2, lw=2, ms=6,
        label=f"SPH at 10 lattice spacings: dt \u2192 dt/2 moves it {100 * (sp2 / sp1 - 1):+.1f} %".replace("-", MINUS))
h1 = HR["dx=1"]["rows"]["10"]["t_half"]
h2 = HR["dx=0.5"]["rows"]["10"]["t_half"]
bx.plot([1, HR["dx=0.5"]["dt_sub"] / HR["dx=1"]["dt_sub"]], [1, h2 / h1], "o-", color=C1, lw=2, ms=6,
        label=f"heat at 10 dx: dx \u2192 dx/2 (dt \u2192 dt/4) moves it {100 * (h2 / h1 - 1):+.1f} %".replace("-", MINUS))
light = leaves["au"]["value"] * 16 / c / tn16[0]
bx.axhline(light, color=INK2, lw=0.9, ls=":")
bx.text(7e-4, light * 1.3, "light-travel time to 16 au (Mentat c, au)", color=INK2, fontsize=8)
bx.set_xlim(5e-4, 2)
bx.set_ylim(3e-4, 3)
bx.set_xlabel("time step / coarsest time step")
bx.set_ylabel("arrival time / arrival time at the coarsest step")
bx.set_title("(b) is the arrival set by the physics or by the step?", fontsize=10, color=INK, loc="left")
bx.legend(frameon=False, fontsize=7.6, loc="lower right")
fig.suptitle("U7 — three solvers, three clocks: heat arrives as d², SPH as a finite-speed front, N-body everywhere in one step\n"
             f"Pre-registered exponent bands missed with the design's statistics (heat {heat['corrected_prereg']['p_half_dx1']:.2f} at dx = 1, "
             f"SPH {sph['point_a1']['fits']['t_0.5']['p_loglog']:.2f} from a point source); met after grid refinement "
             f"({heat['refinement']['p_half_dx05']:.2f}) and with a plane source ({sph['plane_a1']['fits']['t_peak']['p_loglog']:.2f}).\n"
             f"SPH arrival speed {sph['plane_a1']['fits']['t_peak']['speed_over_c_tait']:.2f} c_Tait (the 10 % band fails); "
             f"the engine's own pressure operator carries long waves at {v_posthoc_speed_operator:.2f} c_Tait",
             fontsize=9.6, color=INK, x=0.01, y=0.995, ha="left", va="top")
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.10, top=0.83, wspace=0.17)
fig.savefig(f"{OUT}/u7_arrival_exponents.png", facecolor=SURF, bbox_inches="tight")
print(json.dumps(dict(verdicts=out["verdicts"], ratios=ratios, heat=heat["refinement"], sph_lit=sph_lit,
                      sph_summary={k: dict(p_peak=v["fits"]["t_peak"]["p_loglog"], v_peak=v["fits"]["t_peak"]["speed_over_c_tait"],
                                           p_half=v["fits"]["t_0.5"]["p_loglog"], v_half=v["fits"]["t_0.5"]["speed_over_c_tait"],
                                           v_spatial=v.get("spatial_peak_speed_over_c_tait"), ref_axis=v["ref_max_speed_axis"])
                                   for k, v in sph.items()},
                      lin=sph_lin, trends=trends), indent=1, default=float))
