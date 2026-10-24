"""The n-body shoot-out figures and their data tables (PNG + JSON), from the stored runs. Palette: the validated reference categorical slots 1-3 (blue, orange, aqua) and the blue sequential ramp.

Figures (all errors are position errors in km against DE440, DE440 start states; see the fixtures README for wording):
  F1  headline dumbbell: r2 baseline -> restated model, +5 yr, every body, converged step (0.0125 d), same three windows
  F2  inner system error against lead (r2, control, restated), small multiples, medians over windows
  F3  what each addition bought for the outer planets at +20 yr (7 asteroids -> 343 -> ring -> 7 Kuiper-belt objects)
  F4  the fixture repair: giants' planet-centre error at +5 yr, old fixture against r4
  F5  moon step convergence heatmap (23 moons x 4 steps)
  F6  comets: the ladder at +5 yr, by comet group

usage: python scripts/make_shootout_figures.py [--out DIR] [--full-dir DIR] [--candidate NAME]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import shootout_full_restate_report as R  # noqa: E402

FX = ROOT / "sigma_ground/field/interface/fixtures"
RES = FX / "v2physics_restated"
# reference palette (light), validated: slots 1-3 and the sequential blue ramp
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a9a8a2"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]          # sequential blue steps 250, 400, 550, 700 (ordinal: lightest step still >= 2:1)

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.edgecolor": GRID, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def clean(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fmt_km(x):
    return f"{x:,.0f}" if x >= 100 else (f"{x:.1f}" if x >= 1 else f"{x:.2f}")


def title(fig, main, sub):
    """Finding as the title, a wrapped one-or-two-line subtitle; the figure's axes sit below y = 0.80 (callers set top accordingly)."""
    import textwrap
    width = int(fig.get_size_inches()[0] * 12.2)
    fig.text(0.04, 0.975, main, fontsize=14, fontweight="bold", va="top")
    fig.text(0.04, 0.925, textwrap.fill(sub, width), fontsize=9.5, color=INK2, va="top", linespacing=1.35)


def legend_row(fig, handles, labels, y=0.835, ncol=None):
    """Legend between the subtitle and the axes: it can never sit on the data."""
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.04, y), ncol=ncol or len(handles), frameon=False, fontsize=9, handlelength=1.6, columnspacing=1.6)


def save(fig, path: Path, data: dict) -> None:
    fig.savefig(path, dpi=150)
    path.with_suffix(".json").write_text(json.dumps(data, indent=1))
    plt.close(fig)
    print("wrote", path.name)


BODIES = ["Mercury", "Venus", "Earth", "Moon", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
W3 = ["j2008", "j2014", "j2020"]


def f1_headline(out: Path, r2, fine_r4, cand):
    rows = []
    for b in BODIES:
        a, c = R.med(R.cells(r2, W3, b, 5)), R.med(R.cells(fine_r4[cand], W3, b, 5))
        rows.append((b, a, c))
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    fig.subplots_adjust(left=0.14, right=0.93, top=0.76, bottom=0.12)
    y = np.arange(len(rows))[::-1]
    for yi, (b, a, c) in zip(y, rows):
        ax.plot([c, a], [yi, yi], color=GRAY, linewidth=2, zorder=1)
        ax.scatter([a], [yi], s=80, color=ORANGE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.scatter([c], [yi], s=80, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.text(a * 1.18, yi, fmt_km(a), va="center", fontsize=9, color=INK2)
        ax.text(c / 1.18, yi, fmt_km(c), va="center", ha="right", fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0.02, 2500)
    ax.set_xlabel("position error at +5 yr against DE440 (km, log scale, median of three windows)")
    clean(ax, "x")
    from matplotlib.lines import Line2D
    legend_row(fig, [Line2D([], [], marker="o", color="none", markerfacecolor=ORANGE, markersize=9), Line2D([], [], marker="o", color="none", markerfacecolor=BLUE, markersize=9)],
               ["r2 baseline (step 0.05 d; no asteroids, ring or lunar figure)", "restated (step 0.0125 d; + 343 asteroids + ring + the Moon's C22)"], ncol=1)
    title(fig, "Every body is closer to DE440 after the restatement", "Same DE440 start states and the same three windows (J2008, J2014, J2020); converged step; the giants' systems on the corrected r4 fixture.")
    save(fig, out / "F1_headline_before_after.png", {"windows": W3, "lead_yr": 5, "rows": [{"body": b, "r2_km": a, "restated_km": c} for b, a, c in rows]})


def f2_inner(out: Path, r2, ctrl, cand_d):
    leads = [1, 3, 5, 10, 20]
    fig, axes = plt.subplots(1, 5, figsize=(14, 4.2), sharey=False)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.74, bottom=0.17, wspace=0.32)
    data = {}
    for ax, b in zip(axes, ["Mercury", "Venus", "Earth", "Moon", "Mars"]):
        def series(d, wins_short, leads_use):
            vals = []
            for l in leads_use:
                w = R.LONG20 if l > 5 else wins_short
                v = R.med(R.cells(d, w, b, l))
                vals.append(v)
            return vals
        s_r2 = series(r2, R.SHORT5, leads)
        s_ct = series(ctrl, R.CONTROL_W, [1, 3, 5])
        s_cd = series(cand_d, R.SHORT5, leads)
        xs = np.arange(len(leads))
        ax.plot(xs, s_r2, color=ORANGE, linewidth=2, marker="o", markersize=6)
        ax.plot(xs[:3], s_ct, color=AQUA, linewidth=2, marker="s", markersize=6)
        ax.plot(xs, s_cd, color=BLUE, linewidth=2.4, marker="o", markersize=6)
        ax.set_yscale("log")
        ax.set_xticks(xs, [f"+{l}" for l in leads])
        ax.set_title(b)
        clean(ax)
        if b == "Mercury":
            ax.set_ylabel("km")
        data[b] = {"leads_yr": leads, "r2": s_r2, "control_0.025d": s_ct, "restated": s_cd}
    from matplotlib.lines import Line2D
    legend_row(fig, [Line2D([], [], color=ORANGE, marker="o", linewidth=2, markersize=7), Line2D([], [], color=AQUA, marker="s", linewidth=2, markersize=7), Line2D([], [], color=BLUE, marker="o", linewidth=2.4, markersize=7)],
               ["r2 baseline (step 0.05 d)", "same stack at the finer step only (0.025 d; 6 windows, to +5 yr)", "restated model (+ asteroids, ring, 7 KBOs, lunar figure; r4 fixture)"], y=0.87)
    fig.text(0.5, 0.03, "lead time (years)", ha="center", color=INK2)
    title(fig, "Inner system error against lead time", "Median over windows (5-yr leads: 16 windows; 10 and 20 yr: 7 windows), km against DE440 on a log axis.")
    save(fig, out / "F2_inner_system_vs_lead.png", data)


def f3_ladder(out: Path, restate_runs, ring_runs, kbo_runs):
    def load(f):
        d = {}
        for ln in f.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                d.setdefault(r["vintage"].split("|", 1)[1], {})[r["start"]] = {b: {int(k): v["eO"] for k, v in x.items()} for b, x in r["bodies"].items()}
        return d
    rs, rg, kb = load(restate_runs), load(ring_runs), load(kbo_runs)
    stages = [("7 heaviest asteroids", rs["cand_long"]), ("all 343 asteroids", rg["ast343_c22"]), ("+ Kuiper-belt ring", rg["ast343_ring_c22"]), ("+ 7 Kuiper-belt objects", kb["ast343_ring_kbo7_c22"])]
    bodies = ["Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
    vals = {s: [statistics.mean(d[w][b][20] for w in ("j2000", "j2003")) for b in bodies] for s, d in stages}
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    fig.subplots_adjust(left=0.13, right=0.94, top=0.77, bottom=0.12)
    h, gap = 0.17, 0.02
    for gi, b in enumerate(bodies[::-1]):
        bi = bodies.index(b)
        for si, (s, _) in enumerate(stages):
            y = gi + (1.5 - si) * (h + gap)                      # stage order reads top to bottom, like the legend
            ax.barh(y, vals[s][bi], height=h, color=RAMP[si], zorder=2)
            if si in (0, 3):
                ax.text(vals[s][bi] * 1.12, y, fmt_km(vals[s][bi]), va="center", fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(0.15, 90)
    ax.set_yticks(range(len(bodies)), bodies[::-1])
    ax.set_xlabel("system-barycentre error at +20 yr against DE440 (km, log scale, mean of J2000 and J2003)")
    clean(ax, "x")
    from matplotlib.patches import Patch
    legend_row(fig, [Patch(facecolor=RAMP[si]) for si in range(4)], [s for s, _ in stages], y=0.845)
    title(fig, "What each addition bought for the outer planets", "Reduced model, step 0.025 d, 20 yr from DE440 starts; each bar adds to the one above it. The ring removes most of the Uranus, Neptune and Pluto error and seven identified Kuiper-belt objects take Pluto from 4.1 to 1.5 km.")
    save(fig, out / "F3_outer_planet_ladder.png", {"lead_yr": 20, "windows": ["j2000", "j2003"], "bodies": bodies, "stages": {s: dict(zip(bodies, v)) for s, v in vals.items()}})


def f4_fixture(out: Path, old_fine, r4_fine, cand):
    bodies = ["Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
    a = [R.med(R.cells(old_fine[cand], W3, b, 5)) for b in bodies]
    c = [R.med(R.cells(r4_fine[cand], W3, b, 5)) for b in bodies]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.76, bottom=0.12)
    x = np.arange(len(bodies))
    w = 0.34
    ax.bar(x - w / 2 - 0.01, a, w, color=ORANGE, zorder=2, label="old fixture (Horizons planet-centre states)")
    ax.bar(x + w / 2 + 0.01, c, w, color=BLUE, zorder=2, label="r4 fixture (each planet system moved onto DE440's barycentre)")
    for xi, va, vc in zip(x, a, c):
        ax.text(xi - w / 2 - 0.01, va * 1.12, fmt_km(va), ha="center", fontsize=9, color=INK2)
        ax.text(xi + w / 2 + 0.01, vc * 1.12, fmt_km(vc), ha="center", fontsize=9, color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 3000)
    ax.set_xticks(x, bodies)
    ax.set_ylabel("planet-centre error at +5 yr (km, log scale)")
    clean(ax)
    from matplotlib.patches import Patch
    legend_row(fig, [Patch(facecolor=ORANGE), Patch(facecolor=BLUE)], ["old fixture (Horizons planet-centre states)", "r4 fixture (each planet system moved onto DE440's barycentre)"], y=0.845)
    title(fig, "The planet-centre floors were the fixture's start states", "Full 33-body system, converged step 0.0125 d, restated model, median of J2008, J2014 and J2020. Velocity mismatch in the old fixture: Uranus 5.3, Neptune 2.9, Pluto 0.3, Jupiter 0.02 mm/s.")
    save(fig, out / "F4_fixture_repair.png", {"lead_yr": 5, "windows": W3, "bodies": bodies, "old_fixture_km": a, "r4_fixture_km": c})


def f5_moons(out: Path):
    dts = ["0.05", "0.025", "0.0125", "0.00625"]
    res = {dt: json.loads((RES / f"moon_conv_dt{dt}.json").read_text()) for dt in dts}
    order = ["Phobos", "Deimos", "Io", "Europa", "Ganymede", "Callisto", "Mimas", "Enceladus", "Tethys", "Dione", "Rhea", "Titan", "Hyperion", "Iapetus", "Phoebe",
             "Miranda", "Ariel", "Umbriel", "Titania", "Oberon", "Triton", "Charon", "Moon"]
    M = np.array([[res[dt]["error_km"][b]["1.0"] for dt in dts] for b in order])
    fig, ax = plt.subplots(figsize=(8.2, 8.6))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.85, bottom=0.07)
    L = np.log10(M)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue_ramp", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    ax.imshow(L, aspect="auto", cmap=cmap, vmin=0, vmax=5.6)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, fmt_km(v) if v < 1e5 else f"{v / 1000:,.0f}k", ha="center", va="center", fontsize=8.5, color=(SURFACE if L[i, j] > 3.2 else INK))
    ax.set_xticks(range(len(dts)), [f"{d} d" for d in dts])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(order)), order)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    title(fig, "Most moons are step-limited; the fast ones never converge", "Position error at +1 yr (km) from J2010, headline stack, full 33-body system; darker = larger. Errors fall ~16-fold per halving of the step until a model or fixture floor of 100-2,000 km.")
    save(fig, out / "F5_moon_step_convergence.png", {"window": "j2010", "lead_yr": 1, "steps_d": dts, "rows": {b: dict(zip(dts, M[i].tolist())) for i, b in enumerate(order)}})


def f6_comets(out: Path):
    rows = json.loads((FX / "horizons_comets" / "ladder_full.json").read_text())
    groups = {"all comets": rows, "far (> 10 au)": [r for r in rows if r["r0_au"] > 10], "near, no time delay": [r for r in rows if r["r0_au"] <= 10 and not r["DT"]],
              "near, with time delay": [r for r in rows if r["r0_au"] <= 10 and r["DT"]]}
    rungs = [("R0", "gravity only"), ("R2", "+ non-gravitational push"), ("R3", "+ push time delay"), ("R4", "+ 343 asteroids")]
    vals = {g: {k: statistics.median(r["err_km"][k][4] for r in rs) for k, _ in rungs} | {"R3n": statistics.median(r["err_km"]["R3n"][4] for r in rs)} for g, rs in groups.items()}
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.74, bottom=0.14)
    x = np.arange(len(groups))
    w = 0.17
    for si, (k, label) in enumerate(rungs):
        xs = x + (si - 1.5) * (w + 0.015)
        v = [vals[g][k] for g in groups]
        ax.bar(xs, v, w, color=RAMP[si], zorder=2, label=label)
        if si in (0, 3):
            for xi, vi in zip(xs, v):
                ax.text(xi, vi * 1.15, fmt_km(vi), ha="center", fontsize=8.5, color=INK2)
    for xi, g in zip(x, groups):
        ax.scatter([xi + 2.15 * (w + 0.015)], [vals[g]["R3n"]], marker="D", s=40, facecolor=SURFACE, edgecolor=ORANGE, linewidth=2, zorder=4)      # beside the last bar, clear of its label
    ax.set_yscale("log")
    ax.set_ylim(0.03, 1.5e5)
    ax.set_xticks(x, [f"{g}\n({len(groups[g])} runs)" for g in groups])
    ax.set_ylabel("position error at +5 yr (km, log scale)")
    clean(ax)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_row(fig, [Patch(facecolor=RAMP[si]) for si in range(4)] + [Line2D([], [], marker="D", color="none", markerfacecolor=SURFACE, markeredgecolor=ORANGE, markeredgewidth=2, markersize=7)],
               [label for _, label in rungs] + ["R3 without the Sun's relativistic term"], y=0.845, ncol=3)
    title(fig, "Comets: each addition earns its place", "60 runs from 20 periodic comets against JPL's own solutions (median). The push, its time delay, the Sun's relativity and the 343 asteroids all matter near the Sun; far comets need nothing beyond gravity.")
    save(fig, out / "F6_comet_ladder.png", {"lead_yr": 5, "groups": {g: len(rs) for g, rs in groups.items()}, "median_km": vals})


def f7_asteroids(out: Path):
    rows = json.loads((FX / "horizons_asteroids" / "asteroid_ladder.json").read_text())
    by = {}
    for r in rows:
        by.setdefault((r["asteroid"], r["name"].split("(")[0].strip()), []).append(r)
    items = []
    for (n, name), rs in by.items():
        items.append((name, {k: statistics.median(r["err_km"][k][4] for r in rs) for k in ("R0", "R1", "R4")}))
    items.sort(key=lambda t: t[1]["R0"])
    fig, ax = plt.subplots(figsize=(9.5, 9.0))
    fig.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.07)
    y = np.arange(len(items))
    for yi, (name, v) in zip(y, items):
        ax.plot([v["R4"], v["R0"]], [yi, yi], color=GRAY, linewidth=1.8, zorder=1)
        ax.scatter([v["R0"]], [yi], s=46, color=ORANGE, zorder=3, edgecolor=SURFACE, linewidth=1)
        ax.scatter([v["R1"]], [yi], s=34, marker="D", facecolor=SURFACE, edgecolor=AQUA, linewidth=1.8, zorder=4)
        ax.scatter([v["R4"]], [yi], s=46, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(y, [t[0] for t in items], fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("position error at +5 yr against JPL's own solution (km, log scale, median of three windows)")
    clean(ax, "x")
    from matplotlib.lines import Line2D
    legend_row(fig, [Line2D([], [], marker="o", color="none", markerfacecolor=ORANGE, markersize=8), Line2D([], [], marker="D", color="none", markerfacecolor=SURFACE, markeredgecolor=AQUA, markeredgewidth=2, markersize=7),
                     Line2D([], [], marker="o", color="none", markerfacecolor=BLUE, markersize=8)], ["gravity only", "+ the Sun's relativity", "+ the pull of the other asteroids"], y=0.865)
    title(fig, "Asteroids as scored bodies: 61 km to 2.6 km", "The 30 heaviest of DE440's 343 asteroids, 90 runs (3 windows), our propagation from JPL's state against JPL's later states. The other asteroids' pull helps 89 of 90 runs.")
    save(fig, out / "F7_asteroids_scored.png", {"lead_yr": 5, "windows": ["j2005", "j2010", "j2015"], "median_km_by_asteroid": {n: v for n, v in items}})


def f8_slow_moons(out: Path):
    sc = json.loads((RES / "MOON_SCORES_R5.json").read_text())["bodies"]
    names = ["Himalia", "Nereid", "Helene"]
    cols = {"Himalia": BLUE, "Nereid": ORANGE, "Helene": AQUA}
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.79, bottom=0.12)
    leads = np.arange(1, 6)
    for nm in names:
        for w, v in sorted(sc[nm].items()):
            ax.plot(leads, [v[str(k)] for k in leads], color=cols[nm], linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1, linestyle="-" if float(w) < 2456000 else "--")
        last = max(max(v.values()) for v in sc[nm].values())
        ax.annotate(nm, (5, last), xytext=(5, 0), textcoords="offset points", fontsize=9, color=INK, va="center", ha="left")
    ax.set_yscale("log")
    ax.set_xticks(leads, [f"+{k} yr" for k in leads])
    ax.set_xlim(0.8, 5.7)
    ax.set_ylabel("position error against DE440 (km, log scale)")
    clean(ax, "y")
    from matplotlib.lines import Line2D
    legend_row(fig, [Line2D([], [], color=INK2, linewidth=2), Line2D([], [], color=INK2, linewidth=2, linestyle="--")], ["window J2010", "window J2015"], y=0.835)
    title(fig, "Himalia is scored to 1-24 km; Nereid and Helene drift by thousands",
          "Three slow moons added to the corrected fixture (r5), step 0.0125 d, candidate stack. Nereid and Helene grow almost linearly with lead time, the signature of a wrong orbital period. The cause is not yet tested.")
    save(fig, out / "F8_slow_moons.png", {"dt_days": 0.0125, "windows_jd": [2455197.5, 2457023.8], "err_km": sc})


def f9_fast_moons(out: Path):
    rows = json.loads((RES / "SYSTEM_LOCAL_MOONS.json").read_text())
    items = [r for r in rows if r["control_e5"] is not None and r["period_d"] < 600]
    items.sort(key=lambda r: r["control_e5"])
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    fig.subplots_adjust(left=0.2, right=0.95, top=0.82, bottom=0.09)
    y = np.arange(len(items))
    for yi, r in zip(y, items):
        ax.plot([r["e5"], r["control_e5"]], [yi, yi], color=GRAY, linewidth=1.8, zorder=1)
        ax.scatter([r["control_e5"]], [yi], s=46, color=ORANGE, zorder=3, edgecolor=SURFACE, linewidth=1)
        ax.scatter([r["e5"]], [yi], s=46, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(y, [r["moon"] + "  (" + format(r["period_d"], ".2f") + " d)" for r in items], fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("position error relative to the planet at +5 yr (km, log scale, median of two windows)")
    clean(ax, "x")
    from matplotlib.lines import Line2D
    legend_row(fig, [Line2D([], [], marker="o", color="none", markerfacecolor=ORANGE, markersize=8), Line2D([], [], marker="o", color="none", markerfacecolor=BLUE, markersize=8)],
               ["table zonals (Jupiter J2 14736e-6, Saturn J2 16298e-6)", "measured zonals (Juno, Cassini)"], y=0.865)
    title(fig, "Measured planet zonals: Io 8,588 km to 159 km, Mimas 22,667 to 1,963",
          "Moons of Jupiter and Saturn in system-local models (Sun + planet + all its moons) with the step set by the fastest orbit (300 steps per orbit), so the integrator is not the limit. The six small ring moons at the top lose their orbital phase within a year under either set of zonals (not explained).")
    save(fig, out / "F9_fast_moons_zonals.png", {"unit": "km", "lead_yr": 5, "windows": ["j2010", "j2015"], "rows": [{k: r[k] for k in ("system", "moon", "period_d", "e5", "control_e5")} for r in items]})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=RES / "figures")
    ap.add_argument("--full-dir", type=Path, default=None, help="directory with rolling_shootout_short.json / _long.json of the full 21-window run (default: the stored 0.025 d run)")
    ap.add_argument("--full-short", type=Path, default=None, help="explicit short-run file (overrides --full-dir)")
    ap.add_argument("--full-long", type=Path, default=None, help="explicit long-run file (overrides --full-dir)")
    ap.add_argument("--candidate", default="restated_ast_ring_c22")
    ap.add_argument("--fine-candidate", default="restated_ast_ring_c22")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    r2 = R.load([R.V2 / "rolling_shootout_short_v2physics_r2.json", R.V2 / "rolling_shootout_long_v2physics_r2.json"], {"de440_stack"})["de440_stack"]
    full_short = (a.full_dir / "rolling_shootout_short.json") if a.full_dir else RES / "rolling_shootout_full_short_0.025.json"
    full_long = (a.full_dir / "rolling_shootout_long.json") if a.full_dir else RES / "rolling_shootout_full_long_0.025.json"
    full_short, full_long = a.full_short or full_short, a.full_long or full_long
    full = R.load([full_short, full_long])
    fine_r4 = R.load([RES / "rolling_shootout_r4_fine_0.0125.json"])
    fine_old = R.load([RES / "rolling_shootout_fine_0.0125.json"])
    f1_headline(a.out, r2, fine_r4, a.fine_candidate)
    f2_inner(a.out, r2, full["de440_stack"], full[a.candidate])
    f3_ladder(a.out, RES / "shootout_restated_runs.jsonl", RES / "ring_reduced_runs.jsonl", RES / "kbo_reduced_runs.jsonl")
    f4_fixture(a.out, fine_old, fine_r4, a.fine_candidate)
    f5_moons(a.out)
    f6_comets(a.out)
    f7_asteroids(a.out)
    f8_slow_moons(a.out)
    f9_fast_moons(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
