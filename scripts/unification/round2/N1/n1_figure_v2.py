"""N1 figure v2: (a) where pressure is an order-one share of the trace (corrected: GW170817 and Chandrasekhar values,
P6 rows flagged), (b) the SEMF defect on the whole AME2020 table, (c) what the plan's own reference EOS (SLy) will score
against the current NICER / J0740 anchors, (d) the work order. Reads n1_survey_v2.json; writes n1_plan_figure_v2.png."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
d = json.load(open(HERE / "n1_survey_v2.json", encoding="utf8"))
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"]
plt.rcParams.update({"axes.unicode_minus": True, "font.size": 10, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": INK2, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, axs = plt.subplots(2, 2, figsize=(17, 12.5), gridspec_kw={"width_ratios": [1.0, 1.15]})
(a, b), (cax, dax) = axs

# ---------------------------------------------------------------- (a) pressure share of the trace source
p6 = d["p6_rows_3P_over_rho_c2"]
wd = d["white_dwarf_pressure_share"]["3P_over_rho_c2_chandra_by_rho"]
gw = d["gw170817_pressure_share"]["2 rho_nuc (Abbott 2018 convention, rho_nuc = 2.8e14 g/cm3) [LIT]"]["3P_over_rho_c2"]
sly = d["sly_expectation_scratch"]
a.plot([0], [p6["Sun core"]["3P_over_rho_c2"]], "o", color=S[0], ms=8, zorder=4)
wv = list(wd.values())
a.plot([1] * len(wv), wv, "o-", color=S[0], ms=5, lw=2, zorder=4)
a.text(1.08, wv[0], "ρ_c = 10⁸", fontsize=8, color=INK2, va="center"); a.text(1.08, wv[-1], "ρ_c = 10¹³ kg/m³", fontsize=8, color=INK2, va="center")
a.errorbar([2], [gw["central"]], yerr=[[gw["central"] - gw["low_90"]], [gw["high_90"] - gw["central"]]], fmt="o", color=S[0],
           ms=8, capsize=5, lw=2, zorder=4)
a.text(2.1, gw["central"] * 0.62, "GW170817, 90 %:\n0.11 to 0.37", fontsize=8.5, color=INK2)
a.plot([1], [p6["white dwarf centre"]["3P_over_rho_c2"]], "o", mfc="none", mec=S[1], mew=2, ms=10, zorder=5)
a.annotate("P6 row: 3.8× the\nChandrasekhar pressure", (1, p6["white dwarf centre"]["3P_over_rho_c2"]), xytext=(-118, 8),
           textcoords="offset points", fontsize=8.5, color=S[1])
a.plot([2], [p6["neutron star centre (~2 n0)"]["3P_over_rho_c2"]], "o", mfc="none", mec=S[1], mew=2, ms=10, zorder=5)
a.annotate("P6 row 0.67: 1.6× above\nGW170817's bound, withdrawn", (2, p6["neutron star centre (~2 n0)"]["3P_over_rho_c2"]),
           xytext=(-175, -34), textcoords="offset points", fontsize=8.5, color=S[1])
cw = sly["central_3P_over_eps"]
pts = [(1.0, cw["1.0"]), (1.4, cw["1.4"]), (1.8, cw["1.8"]), (2.0, cw["2.0"]), (sly["M_max_Msun"], sly["w_c_at_Mmax"])]
ylab = [0.20, 0.42, 0.85, 1.9, 4.2]   # label positions (log axis), spread so they do not collide
for i, (m, w) in enumerate(pts):
    a.plot([3], [w], "D", color=S[2], ms=7, zorder=4)
    a.annotate(f"{m:.2f} M☉: {w:.2f}", (3, w), xytext=(3.25, ylab[i]), fontsize=8.5, va="center", color=INK2,
               arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8))
a.axhline(1.0, color=INK2, ls="--", lw=1, zorder=2)
a.text(-0.4, 1.35, "3P = ε: the trace ε − 3P changes sign", color=INK2, fontsize=8.5)
a.set_yscale("log"); a.set_ylim(1e-6, 12); a.set_xlim(-0.5, 4.1)
a.set_xticks(range(4)); a.set_xticklabels(["Sun core\n(P6 [LIT] row)", "white-dwarf\ncentres", "2 ρ_nuc\n(GW170817)", "neutron-star\ncentres (SLy)"])
a.set_ylabel("3P / ε  (pressure share of the trace; ε ≈ ρc² below 2 ρ_nuc)")
a.legend(handles=[Line2D([], [], marker="o", color=S[0], ls="none", label="Mentat constants: Chandrasekhar EOS (μ_e = 2), GW170817 band"),
                  Line2D([], [], marker="o", mfc="none", mec=S[1], mew=2, ls="none", label="P6 [LIT] rows that fail a check (flagged)"),
                  Line2D([], [], marker="D", color=S[2], ls="none", label="SLy central value: scratch TOV on HP2004 fit, not Mentat")],
         fontsize=8, frameon=False, loc="lower right")
a.set_title("(a) pressure reaches order one only in neutron-star cores:\n0.21 at 2 ρ_nuc (GW170817), 0.46 at a 1.4 M☉ SLy centre, flip near 1.94 M☉",
            fontsize=10, loc="left")
a.grid(axis="y", color=GRID, zorder=0)

# ---------------------------------------------------------------- (b) SEMF defect over the whole table
ft = d["semf_full_table"]; pts_b = ft["points"]
b.plot([p["NmZ"] for p in pts_b], [p["r_asis"] for p in pts_b], "o", color=S[1], ms=5, label="element.py as written: a_a (A − 2Z)²/(4A)", zorder=3)
b.plot([p["NmZ"] for p in pts_b], [p["r_cons"] for p in pts_b], "s", color=S[2], ms=4.5, label="same a_a in the (A − 2Z)²/A form its stable_mass_number assumes", zorder=3)
r10 = d["semf_comparison"]["rows"]
b.plot([x["A"] - 2 * x["Z"] for x in r10 if x["A"] >= 12], [x["resid_mcp_SEMF_pct"] for x in r10 if x["A"] >= 12], "^", color=S[0], ms=7,
       label="Mentat nuclear_binding_energy (second coefficient set), 9 nuclides", zorder=4)
for x in r10:
    if (x["Z"], x["A"]) in [(82, 208), (92, 238)]:
        b.annotate({(82, 208): "²⁰⁸Pb", (92, 238): "²³⁸U"}[(x["Z"], x["A"])], (x["A"] - 2 * x["Z"], x["resid_element_py_pct"]),
                   textcoords="offset points", xytext=(-22, 6), fontsize=9, color=INK2)
b.axhline(0, color=INK2, lw=0.8)
b.set_xlabel("neutron excess N − Z"); b.set_ylabel("binding energy: SEMF − AME2020 (%)")
b.set_title(f"(b) element.py:{d['semf_comparison']['element_py_lines']['asymmetry_term']} asymmetry term is 4× too weak: rms "
            f"{ft['rms_pct_A_ge_12']['as_written']:.1f} % → {ft['rms_pct_A_ge_12']['consistent']:.1f} %\nover {ft['n_A_ge_12']} AME2020 nuclides "
            f"(A ≥ 12); best divisor {ft['best_divisor_k_A12']['k']:.2f}, code uses 4 (flagged, not patched)", fontsize=10, loc="left")
b.legend(fontsize=8, frameon=False, loc="upper left"); b.grid(color=GRID, zorder=0)

# ---------------------------------------------------------------- (c) SLy against the anchors the plan pre-registers
seq = sly["sequence"]; imax = max(range(len(seq)), key=lambda i: seq[i]["M_Msun"])
cax.plot([s["R_km"] for s in seq[:imax + 1]], [s["M_Msun"] for s in seq[:imax + 1]], "-", color=S[0], lw=2.4, label="SLy (HP2004 fit), stable branch", zorder=5)
cax.plot([s["R_km"] for s in seq[imax:]], [s["M_Msun"] for s in seq[imax:]], ":", color=S[0], lw=1.4, zorder=5)
rows = d["nicer_compactness"]["rows"]
for k, col, ls, fill in [("J0740+6620 Salmi+2024", S[1], "-", True), ("J0437-4715 Choudhury+2024", S[1], "-", True),
                         ("J0740+6620 Riley+2021", INK2, "--", False), ("J0030+0451 Riley+2019", INK2, "--", False),
                         ("J0030+0451 Miller+2019", INK2, "--", False)]:
    r = rows[k]; (r0, r1), (m0, m1) = r["R_68"], r["M_68"]
    cax.add_patch(Rectangle((r0, m0), r1 - r0, m1 - m0, fill=fill, alpha=0.18 if fill else 1, fc=col if fill else "none",
                            ec=col, ls=ls, lw=1.6 if fill else 1.1, zorder=3))
    lab = {"J0740+6620 Salmi+2024": "J0740 Salmi 2024", "J0437-4715 Choudhury+2024": "J0437 Choudhury 2024",
           "J0740+6620 Riley+2021": "J0740 Riley 2021", "J0030+0451 Riley+2019": "J0030 Riley 2019",
           "J0030+0451 Miller+2019": "J0030 Miller 2019"}[k]
    if k == "J0740+6620 Riley+2021":
        cax.text(r0, m1 + 0.02, lab, fontsize=7.8, color=col, va="bottom")
    else:
        cax.text(r1 + 0.05, (m0 + m1) / 2, lab, fontsize=7.8, color=col, va="center")
for k in ["J0030+0451 Vinciguerra+2024 mode A (approx)", "J0030+0451 Vinciguerra+2024 mode B (approx)"]:
    r = rows[k]; cax.plot([r["R_km_LIT"]], [r["M_Msun_LIT"]], "x", color=INK2, ms=8, mew=2, zorder=4)
cax.text(14.0, 1.75, "J0030 2024 modes (≈)", fontsize=7.8, color=INK2)
cax.axhline(1.94, color=S[2], lw=1.4, ls="-.", zorder=2)
cax.text(8.1, 1.955, "J0740 mass floor 2.08 − 2×0.07 = 1.94 M☉", fontsize=8, color=S[2])
GMc2_km = d["constants"]["G"] * d["constants"]["M_sun"] / d["constants"]["c"] ** 2 / 1e3
for Cc in (0.245, 0.273):
    cax.plot([8, 16], [Cc * 8 / GMc2_km, Cc * 16 / GMc2_km], color=S[3], lw=1.1, ls="--", zorder=2)
fl = sly["central_trace_flip"]
cax.plot([fl["M_Msun"] * GMc2_km / fl["C"]], [fl["M_Msun"]], "*", color=S[3], ms=14, zorder=6)
cax.text(8.1, 2.33, "PMP trace-flip band C = 0.245 to 0.273 (model spread)", fontsize=8, color=S[3])
cax.text(fl["M_Msun"] * GMc2_km / fl["C"] + 0.15, fl["M_Msun"] - 0.07, f"SLy centre flips: C = {fl['C']:.3f}", fontsize=8, color=S[3])
cax.set_xlim(8, 16); cax.set_ylim(0.9, 2.45)
cax.set_xlabel("radius R (km)"); cax.set_ylabel("gravitational mass M (M☉)")
cax.legend(handles=[Line2D([], [], color=S[0], lw=2.4, label="SLy (HP2004 fit), scratch TOV with Mentat G, c, M☉"),
                    Patch(fc=S[1], alpha=0.3, ec=S[1], label="current NICER 68 % boxes (2024)"),
                    Patch(fc="none", ec=INK2, ls="--", label="2019 / 2021 boxes, revisited or superseded"),
                    Line2D([], [], color=S[3], ls="--", label="lines of constant C = GM/(Rc²)")],
           fontsize=8, frameon=False, loc="lower right")
cax.set_title("(c) the plan's own reference EOS fails two radius windows it pre-registered\n(R(2.0 M☉) = 10.6 km vs J0740's 11.6 km lower edge): those tests score EOSs, not code",
              fontsize=10, loc="left")
cax.grid(color=GRID, zorder=0)

# ---------------------------------------------------------------- (d) work order (v2)
phases = [("P0 one adaptive ODE (kernel/ode.py; sg_solve_ivp delegates), cited constants", 0.75, 0),
          ("P1 M1 Fermi gas (e, μ, n, p; T = 0 exact, NR finite T)", 1.25, 0),
          ("P2 M2 Lane–Emden + white dwarfs", 1.0, 1), ("P3 M5 EOS protocol + TOV", 1.5, 1),
          ("P4 M8 trace diagnostics; P7 on the TOV metric", 1.0, 1), ("P5 M3 one SEMF fitted to AME2020", 0.75, 2),
          ("P6 M4 nuclear matter + β-eq.; NS with provisional crust", 2.5, 2),
          ("P7 M6 crust, drip, layers; NICER scored on the unified EOS", 2.0, 2),
          ("P8 M7 bag model, deconfinement, hybrids", 1.25, 2), ("P9 MCP sweep, docs, push, redeploy Mentat", 1.0, 3)]
t = 0.0
for i, (name, dur, g) in enumerate(phases):
    dax.barh(i, dur, left=t, color=S[g], height=0.62, zorder=3)
    dax.text(t + dur + 0.12, i, name, va="center", fontsize=8.6)
    t += dur
dax.set_yticks([]); dax.set_xlim(0, 24); dax.set_ylim(9.7, -1.7); dax.set_xlabel("working days (rough, one agent)")
for x, lab, y in [(4.5, "MS1", -1.1), (5.5, "MS2", -1.1), (8.75, "MS3", -1.1), (13.0, "MS5: live in Mentat", -0.4)]:
    dax.axvline(x, color=INK2, ls=":", lw=1, zorder=2)
    dax.text(x + 0.08, y, lab, fontsize=8, color=INK2)
dax.legend(handles=[Patch(color=S[0], label="foundations"), Patch(color=S[1], label="degenerate stars + GR"),
                    Patch(color=S[2], label="nuclear and quark matter"), Patch(color=S[3], label="integration")],
           fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4)
dax.set_title("(d) the plan (pending Captain decisions D1–D10): 10 phases, ≈ 13 days;\nkernel/ode.py plus 8 modules in field/dense/, tests land with each", fontsize=10, loc="left")
dax.grid(axis="x", color=GRID, zorder=0)
for ax_ in axs.flat:
    for s_ in ["top", "right"]:
        ax_.spines[s_].set_visible(False)

fig.suptitle("Pressure is an order-one share of the σ source only in neutron-star cores, and Mentat cannot build one yet;\n"
             "the plan's own reference EOS already sits below two pre-registered radius windows, so those tests judge the EOS, not the code",
             fontsize=12.5, x=0.01, ha="left", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.965))
fig.savefig(HERE / "n1_plan_figure_v2.png", dpi=140)
print("wrote", HERE / "n1_plan_figure_v2.png")
