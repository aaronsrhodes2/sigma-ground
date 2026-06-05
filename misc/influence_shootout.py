"""Influence-coupling n-body experiment.

Tests whether gravity coupling to BARYON COUNT (entanglement influence)
rather than MASS-ENERGY closes the DE440 residual gap.

Each body's gravitational source is scaled by e^σ where
    σ_j = alpha * (bf_j - bf_Sun)          [variant 'diff', Sun-referenced]
    σ_j = alpha * bf_j                      [variant 'all']
and bf_j is the body's nuclear-binding mass-deficit fraction (composition-
derived). alpha=0 recovers pure mass gravity (the DE440 baseline). alpha=+1
means gravity tracks baryon count exactly. The scan finds the alpha that
MINIMIZES the heliocentric residual vs DE440 — the data decides whether an
influence coupling helps, hurts, or does nothing.

Run:  python misc/influence_shootout.py [horizon_years] [variant] [a0 a1 ...]
"""
import sys, time
sys.path.insert(0, r"D:\Aaron\development\sigma-ground")
sys.path = [p for p in sys.path if "nostalgic-kapitsa" not in p and p not in ("", ".")]
import dataclasses
import numpy as np
from sigma_ground.field.interface import rolling_shootout as rs

# Per-body nuclear-binding mass-deficit fraction (composition-derived; see
# influence bf computation). gas giants ~0.0019, ice giants ~0.0058,
# rocky ~0.0089, icy moons ~0.0077.
BF = {
    "Sun": 0.00202, "Mercury": 0.00900, "Venus": 0.00885, "Earth": 0.00887,
    "Moon": 0.00860, "Mars": 0.00885, "Phobos": 0.00850, "Deimos": 0.00850,
    "Jupiter": 0.00189, "Io": 0.00860, "Europa": 0.00773, "Ganymede": 0.00773,
    "Callisto": 0.00773, "Saturn": 0.00197, "Enceladus": 0.00773,
    "Titan": 0.00773, "Uranus": 0.00577, "Miranda": 0.00773, "Ariel": 0.00773,
    "Umbriel": 0.00773, "Titania": 0.00773, "Oberon": 0.00773,
    "Neptune": 0.00577, "Triton": 0.00773, "Pluto": 0.00800, "Charon": 0.00780,
    "Mimas": 0.00773, "Tethys": 0.00773, "Dione": 0.00773, "Rhea": 0.00773,
}
BF_SUN = BF["Sun"]
PLANETS = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn",
           "Uranus", "Neptune"]


def run_alpha(alpha, start_key, sample_keys, variant="diff"):
    de440 = rs._load_de440()
    pred = [p for p in rs.PREDICTORS if p.name == "jpl_de440"][0]
    start_jd = de440["snapshots"][start_key]["epoch"]["jd_tdb"]
    sample_jds = [de440["snapshots"][k]["epoch"]["jd_tdb"] for k in sample_keys]
    bodies, names, sun_idx, parents = rs._build_bodies_at_snapshot(
        start_key, de440, pred)
    newb = []
    for b, nm in zip(bodies, names):
        bf = BF.get(nm, 0.00850)
        sig = alpha * (bf - BF_SUN) if variant == "diff" else alpha * bf
        sig = max(0.0, sig)   # SSBM domain: σ >= 0 (gravity can only ADD)
        newb.append(dataclasses.replace(b, sigma_field=sig))
    res = rs._integrate_nbody(pred, newb, names, sun_idx, start_jd,
                               sample_jds, parents)
    out = {}
    for nm in PLANETS:
        ref = rs._de440_heliocentric_km(de440, sample_keys[-1], nm)
        if ref is not None and nm in res:
            out[nm] = float(np.linalg.norm(res[nm][-1] - ref) / rs.AU_KM)
    return out


def main():
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    variant = sys.argv[2] if len(sys.argv) > 2 else "diff"
    alphas = ([float(a) for a in sys.argv[3:]] if len(sys.argv) > 3
              else [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    sample_keys = [f"j{2000 + i}" for i in range(horizon + 1)]
    print(f"horizon={horizon}yr  variant={variant}  alphas={alphas}\n")
    rows = []
    for a in alphas:
        t0 = time.perf_counter()
        r = run_alpha(a, "j2000", sample_keys, variant)
        rms = (sum(v * v for v in r.values()) / len(r)) ** 0.5
        rows.append((a, rms, r))
        dt = time.perf_counter() - t0
        print(f"alpha={a:+5.2f}  RMS={rms:.5e} AU  "
              f"Earth={r['Earth']:.3e}  Mars={r['Mars']:.3e}  "
              f"Jup={r['Jupiter']:.3e}  ({dt:.0f}s)")
    best = min(rows, key=lambda x: x[1])
    base = [r for r in rows if r[0] == 0.0]
    print(f"\nBEST alpha = {best[0]:+.2f}  (RMS {best[1]:.5e} AU)")
    if base:
        b = base[0][1]
        print(f"baseline (alpha=0) RMS = {b:.5e} AU")
        delta = (best[1] - b) / b * 100
        verdict = ("influence HELPS" if best[0] != 0 and best[1] < b
                   else "MASS WINS (no influence improvement)")
        print(f"best vs baseline: {delta:+.2f}%   -> {verdict}")


if __name__ == "__main__":
    main()
