"""Generate barycentric states of the heaviest asteroids at every DE440-fixture snapshot epoch (J2000..J2026), for use as massive perturbers in the N-body engine.

Elements and GM: the JPL SBDB fixture (fixtures/sbdb_asteroids_54.json; Ceres, Vesta, Pallas, Hygiea, Interamnia, Psyche, Kalliope carry a measured GM).
Planets and Sun: the DE440s kernel (D:/datasets/jpl/de440s.bsp) for positions, the DE440 fixture for GM values (planet systems = planet + moons).
Method and limits: sigma_ground/field/interface/asteroid_perturbers.py (each asteroid integrated once from its element epoch as a massless test particle).

Output: fixtures/asteroid_perturber_states_v2physics.json  (km and km/day, barycentric ICRF equatorial; label v2physics_2026-10-12).

usage: python scripts/make_asteroid_states.py [--out PATH]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from sigma_ground.field.interface import asteroid_perturbers as ap   # noqa: E402

KERNEL = Path("D:/datasets/jpl/de440s.bsp")
FIXTURE = _ROOT / "sigma_ground/field/interface/fixtures/de440_state_vectors_v2physics.json"
OUT_DEFAULT = _ROOT / "sigma_ground/field/interface/fixtures/asteroid_perturber_states_v2physics.json"
SYSTEM_BARYCENTRES = {"Mercury": (0, 1), "Venus": (0, 2), "EMB": (0, 3), "Mars": (0, 4), "Jupiter": (0, 5), "Saturn": (0, 6), "Uranus": (0, 7),
                      "Neptune": (0, 8), "Pluto": (0, 9)}


def system_gms(snap: dict) -> dict:
    gm = {b["name"]: b["gm_km3_s2"] for b in snap["bodies"]}
    anchor = {b["name"]: b.get("gravitational_anchor") for b in snap["bodies"]}
    out = {"Sun": gm["Sun"]}
    for n in ("Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"):
        out[n] = gm[n] + sum(g for m, g in gm.items() if anchor.get(m) == n)
    out["EMB"] = gm["Earth"] + gm["Moon"]
    return out


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--out", type=Path, default=OUT_DEFAULT)
    a = ap_.parse_args()
    from jplephem.spk import SPK
    k = SPK.open(str(KERNEL))
    de = json.loads(FIXTURE.read_text())
    keys = sorted((x for x in de["snapshots"] if x.startswith("j")), key=lambda x: int(x[1:]))
    epochs = {x: de["snapshots"][x]["epoch"]["jd_tdb"] for x in keys}
    gms = system_gms(de["snapshots"]["j2010"])
    gm_sun = gms["Sun"]

    def attractors(jd):
        out = [(gms["Sun"], np.asarray(k[0, 10].compute(jd), dtype=float))]
        for n, (c, t) in SYSTEM_BARYCENTRES.items():
            out.append((gms[n], np.asarray(k[c, t].compute(jd), dtype=float)))
        return out

    def sun_state(jd):
        p, v = k[0, 10].compute_and_differentiate(jd)
        return np.asarray(p, dtype=float), np.asarray(v, dtype=float)

    result = {"metadata": {"label": "v2physics_2026-10-12", "generated": dt.datetime.now().isoformat(timespec="seconds"), "frame": "barycentric ICRF equatorial",
                           "units": "km and km/day", "elements_source": "JPL SBDB (sbdb_asteroids_54.json), osculating at each body's epoch",
                           "planets": "DE440s kernel positions; GM from the DE440 fixture (planet systems include their moons)",
                           "method": "massless test particle integrated from the element epoch (asteroid_perturbers.py); asteroids do not perturb each other here",
                           "generator": "scripts/make_asteroid_states.py", "snapshot_epochs_jd_tdb": epochs}, "asteroids": {}}
    t0 = time.time()
    for p in ap.load_perturbers():
        jds = list(epochs.values())
        out = ap.propagate_to(p, jds, attractors, sun_state, gm_sun)
        result["asteroids"][p.name] = {"gm_km3_s2": p.gm_km3_s2, "radius_km": p.radius_km, "element_epoch_jd": p.epoch_jd,
                                       "states": {key: [float(x) for x in out[jd]] for key, jd in epochs.items()}}
        r = np.array(out[epochs["j2010"]][:3]) - np.asarray(k[0, 10].compute(epochs["j2010"]), dtype=float)
        print(f"{p.name:11s} GM {p.gm_km3_s2:9.4f}  heliocentric distance at J2010 {np.linalg.norm(r) / ap.AU_KM:.3f} AU  ({time.time() - t0:.0f}s)", flush=True)
    a.out.write_text(json.dumps(result))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
