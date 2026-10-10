"""Figure 4 — Stage 4 (L2, same equation): the virial identity on the repo's SPH engine, exact t = 0 sums
(misc/unification/stage4_virial.json from stage4_virial.py)."""
import json, statistics as st, matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
d = json.load(open("misc/unification/stage4_virial.json"))
rows = [r for r in d["rows"] if not r.get("h_scan")]
hs = [r for r in d["rows"] if r.get("h_scan")]
Ns = sorted(set(r["N"] for r in rows))
mean = lambda k, N: st.mean(r[k] for r in rows if r["N"] == N and r[k] is not None)
sd = lambda k, N: (st.stdev([r[k] for r in rows if r["N"] == N]) if sum(1 for r in rows if r["N"] == N) > 1 else 0.0)
x = [mean("x_measured", N) for N in Ns]; xe = [sd("x_measured", N) for N in Ns]
xi = [(N, mean("x_interior", N)) for N in Ns if any(r["x_interior"] is not None for r in rows if r["N"] == N)]
g = [mean("gravity_over_Usoft", N) for N in Ns]
x_inf = rows[0]["predicted_uniform"]

fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), dpi=180); fig.patch.set_facecolor(SURFACE)
ax = axes[0]; ax.set_facecolor(SURFACE)
ax.axhline(1.0, color=INK, lw=1, linestyle=":"); ax.text(70, 1.006, "continuum virial: x = 1", fontsize=8.5, color=INK2)
ax.axhline(x_inf, color=S2, lw=1.4, linestyle="--"); ax.text(70, x_inf + 0.006, "self-term limit 1/(1+32/3N_nb) = %.3f (h = 1.5·spacing, N_nb = 113)" % x_inf, fontsize=8.5, color=INK2)
ax.errorbar(Ns, x, yerr=xe, color=S1, marker="o", ms=6, lw=2, capsize=3, label="pressure virial x, whole cloud (exact t = 0 sums; ±1 sd over seeds)")
ax.plot([n for n, _ in xi], [v for _, v in xi], color=S3, marker="s", ms=6, lw=1.6, label="interior particles only (r < R − 2h)")
ax.plot(Ns, g, color=MUTED, marker="^", ms=5, lw=1, label="gravity virial / |U_softened|")
ax.set_xscale("log", base=2); ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
ax.set_ylim(0.66, 1.03); ax.set_xlabel("particles N"); ax.set_ylabel("delivered fraction of the continuum virial term")
ax.set_title("(a) gravity's term arrives whole; pressure's is short —\nand it converges to 0.914, not to 1", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.2, loc="lower right"); ax.grid(color=GRID, lw=0.6)
ax = axes[1]; ax.set_facecolor(SURFACE)
nnb = [r["full_support_neighbours"] for r in hs]
nn = np.logspace(np.log10(20), np.log10(700), 100)
ax.plot(nn, 1 / (1 + 32 / (3 * nn)), color=S2, lw=2, linestyle="--", label="prediction: x_interior = 1/(1 + 32/(3 N_nb))  (no free parameter)")
ax.plot(nnb, [r["x_measured"] for r in hs], color=S1, marker="o", ms=7, lw=2, label="whole cloud, N = 512")
ok = [(n, r["x_interior"], r["n_interior"]) for n, r in zip(nnb, hs) if r["x_interior"] is not None]
ax.plot([o[0] for o in ok], [o[1] for o in ok], color=S3, marker="s", ms=7, lw=1.6, label="interior particles, N = 512")
for n, v, k in ok:
    ax.text(n * 1.06, v + 0.012, f"{k} interior", fontsize=7.5, color=MUTED)
for n, r in zip(nnb, hs):
    ax.text(n * 0.96, r["x_measured"] - 0.03, f"h = {r['h_mult']:g}×", fontsize=7.5, color=INK2, ha="right")
ax.set_xscale("log"); ax.set_xlabel("full-support neighbour count N_nb = n·(4/3)π(2h)³"); ax.set_ylabel("pressure-virial fraction x")
ax.set_title("(b) the cause: the particle counts itself in its own density\n(ρ_i includes m W(0); the pressure gradient cannot)", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.2, loc="lower right"); ax.grid(color=GRID, lw=0.6); ax.set_ylim(0.6, 1.15)
fig.suptitle("Fig. 4 — L2, same equation: one virial identity holds exactly on the engine (x = Σm D/ρ / 3M to 10⁻¹⁶); the 'missing pressure' is an SPH estimator bias",
             fontsize=11.5, color=INK)
fig.tight_layout(); fig.savefig("misc/unification/fig04_same_equation.png", bbox_inches="tight", facecolor=SURFACE); print("saved")
