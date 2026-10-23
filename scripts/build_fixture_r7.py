"""Build the r7 fixture: the ORIGINAL fixture plus the 23 Horizons moons in their own (true) frame, then each planet system (planet + ALL its moons, the added ones included) moved rigidly onto DE440's barycentre.

Why (2026-10-23). The r4 rigid shift repaired a 2.9 mm/s (Neptune) disagreement between the fixture's planet centre and DE440's system barycentre. Putting Neptune's four small inner moons (Proteus, Larissa, Galatea,
Despina; GM 3.14 km^3/s^2) into the barycentre closes most of it (2.87 -> 0.52 mm/s): the "defect" was mostly missing mass. With the small moons present in the model, the Neptune system also keeps the true
Triton (Triton +1 yr 302 and 246 km -> 39 and 11 km in the Sun+Neptune system). The shift is recomputed over all members, so what is left (Neptune 0.5 mm/s, Uranus 5 mm/s: moons with no published mass) is still
absorbed rigidly, and relative states inside a system are those of Horizons.

Nothing existing is changed: r4 and r5/r6 stay as they are.

usage: python scripts/build_fixture_r7.py [--out PATH]
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
import extend_fixture_with_moons as ex  # noqa: E402
import shift_giant_systems as sg  # noqa: E402

G = ex.G


def add_moons(snapshots: dict, moons: list[str]) -> dict:
    """Append each Horizons moon to every yearly snapshot, in the Horizons frame (no offset). Returns {moon: {system, gm}}."""
    from sigma_ground.field.interface.comet_truth import parse_horizons_comet
    keys = sorted((k for k in snapshots if k.startswith("j")), key=lambda k: int(k[1:]))
    added = {}
    for name in moons:
        f = next((ex.FIX / "horizons_moons").glob(f"horizons_moon_*_{name}.json"))
        t = parse_horizons_comet(f)
        res = json.loads(f.read_text())["result"]
        gm = ex.parse_gm(res)
        hid = int(re.search(r"horizons_moon_(\d+)_", f.name).group(1))
        for i, key in enumerate(keys):
            assert abs(t["jd"][i] - snapshots[key]["epoch"]["jd_tdb"]) < 1e-3, (name, key)
            s = t["state_km_kmday"][i]
            pos, vel = s[:3], s[3:] / 86400.0
            snapshots[key]["bodies"].append({"name": name, "sol_complex_name": name, "horizons_id": hid, "gm_km3_s2": gm or 0.0,
                                             "gm_source": "Horizons header" if gm is not None else "massless test body", "mass_kg": (gm or 0.0) * 1e9 / G,
                                             "gravitational_anchor": ex.SYSTEM_OF[name],
                                             "state_vector": {"x_km": float(pos[0]), "y_km": float(pos[1]), "z_km": float(pos[2]), "vx_km_s": float(vel[0]), "vy_km_s": float(vel[1]), "vz_km_s": float(vel[2])}})
        added[name] = {"system": ex.SYSTEM_OF[name], "gm_km3_s2": gm or 0.0}
    return added


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ex.FIX / "de440_state_vectors_v2physics_r7.json")
    a = ap.parse_args()
    from jplephem.spk import SPK
    kernel = SPK.open("D:/datasets/jpl/de440s.bsp")
    new = copy.deepcopy(json.loads((ex.FIX / "de440_state_vectors_v2physics.json").read_text()))
    added = add_moons(new["snapshots"], list(ex.SYSTEM_OF))
    shifts = {}
    for key, snap in new["snapshots"].items():
        if key.startswith("j"):
            for p, v in sg.shift_snapshot(snap, kernel).items():
                shifts.setdefault(p, []).append(v)
    new.setdefault("metadata", {})["r7"] = {"added": added, "note": "original fixture + 23 Horizons moons in the Horizons frame, each planet system (all members) moved rigidly onto DE440's barycentre (scripts/build_fixture_r7.py)"}
    a.out.write_text(json.dumps(new))
    for p, v in shifts.items():
        print(f"{p:8s} median rigid shift over {len(v)} snapshots: position {np.median([x[0] for x in v]):9.2f} km, velocity {np.median([x[1] for x in v]):8.3f} mm/s")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
