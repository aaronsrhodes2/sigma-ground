"""U7 repair pass (v2) — every SPH arrival statistic reported with every discriminator, no cherry-picking.

For each run pair (reference, perturbed) and each statistic (t_0.1, t_0.25, t_0.5 = pre-registered half-peak, t_peak):
  p over the windows d = 4-10 (pre-registered), 5-10, 7-10; speed (slope of d vs t) / c_Tait; RMS ratio of the
  parabolic model t0 + d^2/D over the hyperbolic model t0 + d/c; increment trend (t10 - t7)/(t7 - t4)
  (hyperbolic 1, parabolic 51/33 = 1.545); monotonicity of t(d).
Uses u7_sph_analyse.analyse (unchanged) for the arrival rows. Also the same table for the 1-D linearised chain model
(sph_1d_linear.json) so the control is judged by the same rules.
"""
import ast, json, sys
import numpy as np
OUT = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U7"
sys.path.insert(0, OUT)
import u7_sph_analyse as SA

STATS = ("t_0.1", "t_0.25", "t_0.5", "t_peak")
PARABOLIC_TREND = (100 - 49) / (49 - 16)


def table_from_rows(rows, s, c_tait):
    out = {}
    for key in STATS:
        t = {d: rows[d][key] for d in range(4, 11)}
        tt = np.array([t[d] for d in range(4, 11)])
        dd = np.arange(4, 11) * s
        e = dict(p_4_10=SA.ols_loglog(dd, tt),
                 p_5_10=SA.ols_loglog(dd[1:], tt[1:]),
                 p_7_10=SA.ols_loglog(dd[3:], tt[3:]),
                 speed_over_c_tait=float(np.polyfit(tt, dd, 1)[0] / c_tait),
                 rms_ratio_parabolic_over_hyperbolic=SA.model_compare(dd, tt)["rms_ratio_parabolic_over_hyperbolic"],
                 increment_trend=float((t[10] - t[7]) / (t[7] - t[4])),
                 monotone=bool(np.all(np.diff(tt) > 0)),
                 t_us={d: float(t[d] * 1e6) for d in range(4, 11)})
        e["p_in_band_1pm0.1"] = abs(e["p_4_10"] - 1) <= 0.1
        e["speed_in_band_1pm0.1"] = abs(e["speed_over_c_tait"] - 1) <= 0.10
        out[key] = e
    return out


def run_table(ref, pert):
    o = SA.analyse(ref, pert)
    tab = table_from_rows(o["rows"], o["s"], o["c_tait"])
    meta = json.load(open(f"{OUT}/sph_{pert}.json"))
    return dict(ref=ref, pert=pert, dims=meta["dims"], material=o["material"], source=o["source"], alpha_visc=o["alpha_visc"],
                dt=o["dt"], ref_max_speed_axis=o["ref_max_speed_axis"], ref_max_speed_any=o["ref_max_speed_any"],
                v0_source_peak=o["v0_source_peak"],
                signal_first_peak_axis_m_s={d: o["rows"][d]["peak"] for d in range(1, 11)},
                stats=tab)


def linear_1d_table():
    j = json.load(open(f"{OUT}/sph_1d_linear.json"))
    rows = j["rows"]
    if isinstance(rows, str):
        rows = ast.literal_eval(rows)
    rows = {int(k): v for k, v in rows.items()}
    return dict(model="1-D linearised engine pressure operator, no viscosity, no surfaces (u7_sph_1d_linear.py) [DERIVED numerics]",
                stats=table_from_rows(rows, 0.01, j["c_tait"]))


if __name__ == "__main__":
    print(json.dumps(run_table(sys.argv[1], sys.argv[2]), indent=1, default=float))
