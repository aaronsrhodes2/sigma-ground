"""Stage 10 — Quarksum, the storage fixture: what σ would act on, for every structure in the engine's own universe.

Quarksum (sigma_ground/inventory) stores matter as a Structure tree — material → molecule → atom → nucleon → quark /
gluon / electron — with a resolved mass at every node (anchor-anywhere resolver), measured seeds in inventory/data/*.json
(isotopes, particle masses, materials), and checksums that prove mass closure at every level. Its quark_chain checksum
rebuilds each structure's mass bottom-up:
    predicted = bare quarks + electrons + QCD binding/c² − nuclear binding/c² − chemical binding/c²
That decomposition is exactly what the σ-bridge needs: in SSBM only the QCD-generated part of mass scales with e^σ
(inventory/core/sigma.py, checksum/sigma_chain.py); bare quark (Higgs) masses, electrons and chemistry do not. So the
fraction of a structure's mass that σ would move is
    f = ∂ln M/∂σ ≈ (QCD binding − nuclear binding)/(c² · M)
(the nuclear binding is treated as all-strong here; the Coulomb refinement, which matters at the 10⁻³ level and drives
the free-fall test, is in inventory.core.sigma and was used for the MICROSCOPE alloys in sigma_exclusion.py).
This script runs quark_chain on every built-in structure and reports f, the mass closure, and the particle counts.
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from mentat_client import mentat
from sigma_ground.inventory import list_structures, load_structure, quark_chain

C, C_src = mentat("lookup_constant", {"name": "c"})
rows = []
for meta in list_structures():
    name = meta.get("name") if isinstance(meta, dict) else str(meta)
    key = meta.get("key") or meta.get("id") or meta.get("file") if isinstance(meta, dict) else None
    s = None
    for cand in (key, name):
        if cand:
            try:
                s = load_structure(cand)
            except Exception:
                s = None
            if s is not None:
                break
    if s is None:
        rows.append(dict(name=name, error="could not load")); continue
    q = quark_chain(s)
    M = q["predicted_mass_kg"]
    qcd = q["qcd_binding_joules"] / C ** 2; nuc = q["nuclear_binding_joules"] / C ** 2
    rows.append(dict(name=name, stated_mass_kg=q["stated_mass_kg"], predicted_mass_kg=M, closure_percent=q["mass_defect_percent"],
                     f_sigma=(qcd - nuc) / M, bare_quark_fraction=q["bare_quark_mass_kg"] / M,
                     electron_fraction=q["electron_mass_kg"] / M, nuclear_binding_fraction=nuc / M,
                     protons=q["proton_count"], neutrons=q["neutron_count"], electrons=q["electron_count"]))
ok = [r for r in rows if "f_sigma" in r]
out = dict(c_source=C_src, rows=rows, f_min=min(r["f_sigma"] for r in ok), f_max=max(r["f_sigma"] for r in ok))
os.makedirs("misc/unification", exist_ok=True)
json.dump(out, open("misc/unification/stage10_quarksum.json", "w"), indent=1, default=str)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
ok.sort(key=lambda r: r["predicted_mass_kg"])
fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=180, gridspec_kw=dict(width_ratios=[1.4, 1])); fig.patch.set_facecolor(SURFACE)
ax = axes[0]; ax.set_facecolor(SURFACE)
y = range(len(ok))
ax.barh(list(y), [r["f_sigma"] for r in ok], color=S1, height=0.6)
for i, r in enumerate(ok):
    ax.text(r["f_sigma"] + 0.0001, i, "%.4f   (M = %.1e kg)" % (r["f_sigma"], r["predicted_mass_kg"]), va="center", fontsize=8, color=INK2)
ax.set_yticks(list(y)); ax.set_yticklabels([r["name"] for r in ok], fontsize=8.5)
lo = min(r["f_sigma"] for r in ok); hi = max(r["f_sigma"] for r in ok)
ax.set_xlim(lo - 0.003, hi + 0.004)
ax.set_xlabel("f = fraction of the structure's mass σ would scale (QCD binding share, Quarksum quark_chain)")
ax.set_title("(a) σ acts on ~%.1f %% of everything: from a hydrogen atom to the observable universe,\nthe spread in f is only %.1e — the differences free-fall experiments can see" % (100 * (lo + hi) / 2, hi - lo),
             fontsize=10, color=INK)
ax.grid(color=GRID, lw=0.6, axis="x")
ax = axes[1]; ax.set_facecolor(SURFACE)
comp = [("QCD binding (scales with e^σ)", [r["f_sigma"] + r["nuclear_binding_fraction"] for r in ok], S1),
        ("bare quark (Higgs) mass", [r["bare_quark_fraction"] for r in ok], S3),
        ("electrons", [r["electron_fraction"] for r in ok], S2)]
pick = [ok[0], ok[len(ok) // 2], ok[-1]]
labels = [p["name"] for p in pick]
import numpy as np
xs = np.arange(len(pick))
bottom = np.zeros(len(pick))
for lab, vals, col in comp:
    v = np.array([vals[ok.index(p)] for p in pick]) * 100
    ax.bar(xs, v, bottom=bottom, color=col, width=0.55, label=lab, edgecolor=SURFACE, linewidth=2)
    bottom += v
nb = np.array([p["nuclear_binding_fraction"] for p in pick]) * 100
ax.bar(xs, -nb, bottom=bottom, color=MUTED, width=0.55, label="− nuclear binding", edgecolor=SURFACE, linewidth=2)
for i, p in enumerate(pick):
    ax.text(i, 101.2, "closure %+.3f %%" % p["closure_percent"], ha="center", fontsize=8.5, color=INK2)
ax.set_xticks(xs); ax.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=8.5)
ax.set_ylim(96.5, 102); ax.set_ylabel("percent of the structure's mass")
ax.set_title("(b) the ledger σ reads from: every structure rebuilt bottom-up\nfrom quarks, electrons and binding energies, and checked", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8, loc="lower center"); ax.grid(color=GRID, lw=0.6, axis="y")
fig.suptitle("Fig. 10 — Quarksum as the storage fixture: the inventory is the ledger the σ equation acts on", fontsize=12, color=INK)
fig.tight_layout(); fig.savefig("misc/unification/fig10_quarksum.png", bbox_inches="tight", facecolor=SURFACE)
for r in rows:
    print(r.get("name"), "|", ("f=%.5f closure=%+.4f%%" % (r["f_sigma"], r["closure_percent"])) if "f_sigma" in r else r.get("error"))
