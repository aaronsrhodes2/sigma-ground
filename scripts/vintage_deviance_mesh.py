"""The deviance mesh: how far JPL's own past ephemerides were from the best current one, by target date and by lead time.

Each JPL ephemeris is a snapshot of what JPL believed at its release date. Scored against the most recent one (DE440, 2020) at dates AFTER its
release, a vintage's deviation is its forecast error (what JPL "predicted at that time"); before its release it is a hindcast (fitted data).
Laying all vintages out by target date gives the mesh:
  * deviation that grows with lead time in every vintage is the limit of prediction from the data of that era;
  * a feature at one target date shared by several vintages (a vertical stripe) is an event, or a data change, at that date.
The same stripe score as scripts/shootout_fabric.py: z = ln(deviation / median over vintages at that lead), median over vintages reaching the date (>= 3).

Vintage release dates are the ones written in each kernel's own comments (NAIF generic_kernels/spk/planets, downloaded to D:/datasets/jpl/vintages).
Truth here is DE440, which is itself a fit to measurements, so this is "JPL's past vs JPL's present", not yet "vs the sky". Targets stop at 2020: beyond the end
of DE440's own fit data it is forecasting too, and a 'stripe' there would only mean DE440 differs from every older vintage.

IMPORTANT for the comparison with ours: our shoot-out starts each forecast from the DE440 state AT the issue date (perfect initial conditions); a JPL vintage had to
determine its initial conditions from the data of its day. So 'ours' measures integration error alone, JPL's includes orbit-determination error: a fair contest needs
our own fit from data available at the issue date (the observation-based contest).

Optional comparison: ``--ours <rolling_shootout.json>`` adds our own N-body forecasts (heliocentric error vs lead) beside JPL's vintages at the same leads.

usage: python scripts/vintage_deviance_mesh.py --out DIR [--ours rolling_shootout_short.json --ours-label v2physics_r1]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

V_DIR = Path("D:/datasets/jpl/vintages")
TRUTH = Path("D:/datasets/jpl/de440s.bsp")
J2000 = 2451545.0
# (name, file, release date as stated in the kernel's own comments)
VINTAGES = [
    ("DE403", "de403s.bsp", dt.date(1995, 11, 14)),
    ("DE405", "de405.bsp", dt.date(1998, 8, 26)),
    ("DE410", "de410s.bsp", dt.date(2003, 4, 24)),
    ("DE413", "de413.bsp", dt.date(2004, 11, 4)),
    ("DE418", "de418.bsp", dt.date(2007, 8, 3)),
    ("DE421", "de421.bsp", dt.date(2008, 2, 13)),
    ("DE432", "de432s.bsp", dt.date(2014, 4, 30)),
]
BODIES = {"Mercury": ((0, 1), (1, 199)), "Venus": ((0, 2), (2, 299)), "Mars": ((0, 4),), "Jupiter": ((0, 5),),
          "Saturn": ((0, 6),), "Uranus": ((0, 7),), "Neptune": ((0, 8),), "Earth": ((0, 3), (3, 399))}


def jd_of(d: dt.date) -> float:
    return d.toordinal() + 1721424.5


def year_of(jd: float) -> float:
    return 2000.0 + (jd - J2000) / 365.25


def helio(k, chain, jd):
    pos = 0.0
    for c, t in chain:
        pos = pos + k[c, t].compute(jd)
    return pos - k[0, 10].compute(jd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ours", type=Path, default=None, help="rolling shoot-out json to compare against at the same leads")
    ap.add_argument("--ours-label", default="ours")
    ap.add_argument("--predictor", default="de440_stack")
    a = ap.parse_args()
    from jplephem.spk import SPK
    truth = SPK.open(str(TRUTH))
    TRUTH_END = dt.date(2020, 1, 1)      # DE440's fit data end about here; later dates are DE440's own forecast, not truth (to be verified against its documentation)
    jd = np.arange(jd_of(dt.date(1996, 1, 1)), jd_of(TRUTH_END), 30.4375)
    truth_pos = {b: helio(truth, ch, jd) for b, ch in BODIES.items()}
    mesh = {}                                  # vintage -> body -> deviation km (heliocentric position)
    for name, f, rel in VINTAGES:
        k = SPK.open(str(V_DIR / f))
        lo, hi = k[0, 3].start_jd, k[0, 3].end_jd
        ok = (jd >= lo) & (jd <= hi)
        mesh[name] = {"release_jd": jd_of(rel), "bodies": {}}
        for b, ch in BODIES.items():
            dev = np.full(len(jd), np.nan)
            try:
                dev[ok] = np.linalg.norm(helio(k, ch, jd[ok]) - truth_pos[b][:, ok], axis=0)
            except Exception:
                pass
            mesh[name]["bodies"][b] = dev
    a.out.mkdir(parents=True, exist_ok=True)
    # ---- lead-time table: forecast deviation (target after release), by whole-year lead ------------------------------------
    by_lead = {b: defaultdict(list) for b in BODIES}
    for name, m in mesh.items():
        for b in BODIES:
            for j, e in zip(jd, m["bodies"][b]):
                lead = (j - m["release_jd"]) / 365.25
                if lead > 0 and not math.isnan(e):
                    by_lead[b][int(math.floor(lead)) + 1].append(e)         # lead bin 1 = first year after release
    lead_table = {b: {str(k): {"median_km": float(np.median(v)), "n": len(v)} for k, v in sorted(d.items()) if k <= 12 and len(v) >= 6}
                  for b, d in by_lead.items()}
    # ---- stripe test across vintages ---------------------------------------------------------------------------------------
    # Each vintage is first divided by its OWN typical forecast deviation (median over its forecast years): an old vintage that is simply worse than a
    # newer one must not look like an event at the dates only it covers. What remains is the date-to-date shape inside each vintage; a stripe is a date where
    # several vintages are above their own typical level together.
    stripes = {}
    for b in BODIES:
        own = {}
        for name, m in mesh.items():
            vals = [e for j, e in zip(jd, m["bodies"][b]) if (j - m["release_jd"]) > 0 and not math.isnan(e) and e > 0]
            own[name] = float(np.median(vals)) if vals else None
        norm_by_lead = defaultdict(list)
        for name, m in mesh.items():
            if own[name] is None:
                continue
            for j, e in zip(jd, m["bodies"][b]):
                lead = (j - m["release_jd"]) / 365.25
                if lead > 0 and not math.isnan(e) and e > 0:
                    norm_by_lead[int(math.floor(lead)) + 1].append(e / own[name])
        base = {k: float(np.median(v)) for k, v in norm_by_lead.items() if len(v) >= 6}
        per_year = defaultdict(list)
        for name, m in mesh.items():
            if own[name] is None:
                continue
            for j, e in zip(jd, m["bodies"][b]):
                lead = (j - m["release_jd"]) / 365.25
                bin_ = int(math.floor(lead)) + 1
                if lead > 0 and not math.isnan(e) and e > 0 and bin_ in base:
                    per_year[int(round(year_of(j)))].append((name, math.log(e / own[name] / base[bin_])))
        st = {}
        for y, zs in per_year.items():
            if len({n for n, _ in zs}) >= 3:
                st[y] = (float(np.median([z for _, z in zs])), len({n for n, _ in zs}))
        vals = np.array([v[0] for v in st.values()]) if st else np.array([0.0])
        med = float(np.median(vals))
        sig = 1.4826 * float(np.median(np.abs(vals - med))) or 1e-9
        stripes[b] = {"median": med, "robust_sigma": sig, "scores": {str(y): v[0] for y, v in sorted(st.items())},
                      "flagged": {str(y): {"score": v[0], "sigmas": (v[0] - med) / sig, "vintages": v[1]} for y, v in sorted(st.items()) if (v[0] - med) / sig > 3.0}}
    out = {"truth": "DE440s (2020)", "vintages": [{"name": n, "release": r.isoformat(), "file": f} for n, f, r in VINTAGES],
           "lead_table_median_km": lead_table, "stripes": stripes, "target_sampling_days": 30.4375}
    # ---- comparison with our forecasts at the same leads -------------------------------------------------------------------
    if a.ours:
        d = json.loads(a.ours.read_text())
        AU = 149597870.7
        ours = {b: defaultdict(list) for b in BODIES}
        for s in d["samples"]:
            if s["predictor"] == a.predictor and s["body"] in BODIES and s.get("error_au") is not None:
                k = int(round((s["sample_jd"] - s["window_start_jd"]) / 365.25))
                if k >= 1:
                    ours[s["body"]][k].append(s["error_au"] * AU)
        cmp = {}
        for b in BODIES:
            cmp[b] = {}
            for k in range(1, 6):
                o = ours[b].get(k, [])
                j = [e for v_name, m in mesh.items() for jj, e in zip(jd, m["bodies"][b]) if 0 < (jj - m["release_jd"]) / 365.25 and int(math.floor((jj - m["release_jd"]) / 365.25)) + 1 == k and not math.isnan(e)]
                cmp[b][str(k)] = {"ours_median_km": float(np.median(o)) if o else None, "jpl_vintages_median_km": float(np.median(j)) if j else None}
        out["comparison_with_ours"] = {"label": a.ours_label, "predictor": a.predictor, "per_body_by_lead_year": cmp,
                                       "note": "ours = heliocentric error vs DE440 at the same lead; JPL = vintage deviation from DE440 at that lead after its release. DE440 is the truth for both."}
    (a.out / "DEVIANCE_MESH.json").write_text(json.dumps(out, indent=1))
    # ---- figures -----------------------------------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(BODIES)
    fig, axes = plt.subplots(3, 3, figsize=(15, 9.5))
    cmap = plt.get_cmap("plasma")
    xs = [year_of(j) for j in jd]
    for ax, b in zip(axes.ravel(), names):
        for i, (name, m) in enumerate(mesh.items()):
            y = np.array(m["bodies"][b])
            ax.plot(xs, np.where(y > 0, y, np.nan), color=cmap(i / len(mesh)), lw=1.2, label=name)
            ax.axvline(year_of(m["release_jd"]), color=cmap(i / len(mesh)), lw=0.6, ls=":")
        ax.set_yscale("log")
        ax.set_title(b, fontsize=10)
        ax.set_xlabel("target date")
        ax.set_ylabel("deviation from DE440 (km)")
    axes.ravel()[0].legend(fontsize=7, ncol=2)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Deviance mesh: each JPL vintage vs DE440 by target date (dotted = its release date; right of it is its forecast)", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out / "DEVIANCE_MESH_lines.png", dpi=110)
    plt.close(fig)
    for b in names:
        f = stripes[b]["flagged"]
        print(f"{b:8s} stripes >3 robust sigma: " + (", ".join(f"{y} ({v['sigmas']:.1f}s)" for y, v in f.items()) or "none"))
    print("\nmedian forecast deviation by lead year (km), JPL vintages" + (" | ours" if a.ours else ""))
    for b in names:
        row = []
        for k in range(1, 6):
            jv = lead_table[b].get(str(k), {}).get("median_km")
            ov = out.get("comparison_with_ours", {}).get("per_body_by_lead_year", {}).get(b, {}).get(str(k), {}).get("ours_median_km") if a.ours else None
            row.append(f"{jv:9.1f}" + (f"|{ov:8.1f}" if ov is not None else "") if jv is not None else "    n/a")
        print(f"{b:8s}", "  ".join(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
