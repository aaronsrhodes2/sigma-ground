"""Build the r8 fixture: r7 with Neptune's and Triton's tracks replaced by the CURRENT Horizons ones (the same vintage as the 23 moons), then each planet system moved rigidly onto DE440's barycentre.

Why (2026-10-23). The fixture's Neptune and Triton rows came from an older Horizons vintage: against a fresh fetch (D:/datasets/jpl/horizons_centres, fetched 2026-10-23 with the Captain's approval) the Neptune centre differs by 159 km
(median, 59-336 km) and Triton by 209 km and 1.3 m/s, while Saturn, Titan, Uranus and Titania are identical to 0.000 km. A moon fetched now (Proteus, Despina, Nereid, ...) minus the OLD Neptune centre therefore has a starting
relative state wrong by about 160 km, which is the constant mean-motion error found in scripts/phase_slope.py. r8 makes Neptune, Triton and the added Neptune moons one consistent vintage. Nothing existing is changed.

usage: python scripts/build_fixture_r8.py [--out PATH]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
import build_fixture_r7 as b7  # noqa: E402
import extend_fixture_with_moons as ex  # noqa: E402
import shift_giant_systems as sg  # noqa: E402

CENTRES = Path("D:/datasets/jpl/horizons_centres")
REPLACE = {"Neptune": "horizons_centre_899_Neptune.json", "Triton": "horizons_centre_801_Triton.json"}


def replace_tracks(snapshots: dict) -> None:
    from sigma_ground.field.interface.comet_truth import parse_horizons_comet
    keys = sorted((k for k in snapshots if k.startswith("j")), key=lambda k: int(k[1:]))
    for name, fname in REPLACE.items():
        t = parse_horizons_comet(CENTRES / fname)
        for i, key in enumerate(keys):
            assert abs(t["jd"][i] - snapshots[key]["epoch"]["jd_tdb"]) < 1e-3, (name, key)
            s = t["state_km_kmday"][i]
            b = next(x for x in snapshots[key]["bodies"] if x["name"] == name)
            b["state_vector"] = {"x_km": float(s[0]), "y_km": float(s[1]), "z_km": float(s[2]), "vx_km_s": float(s[3] / 86400.0), "vy_km_s": float(s[4] / 86400.0), "vz_km_s": float(s[5] / 86400.0)}
            b["state_source"] = "Horizons 2026-10-23 (same vintage as the moons)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ex.FIX / "de440_state_vectors_v2physics_r8.json")
    a = ap.parse_args()
    from jplephem.spk import SPK
    kernel = SPK.open("D:/datasets/jpl/de440s.bsp")
    new = copy.deepcopy(json.loads((ex.FIX / "de440_state_vectors_v2physics.json").read_text()))
    replace_tracks(new["snapshots"])
    added = b7.add_moons(new["snapshots"], list(ex.SYSTEM_OF))
    shifts = {}
    for key, snap in new["snapshots"].items():
        if key.startswith("j"):
            for p, v in sg.shift_snapshot(snap, kernel).items():
                shifts.setdefault(p, []).append(v)
    new.setdefault("metadata", {})["r8"] = {"added": added, "replaced": REPLACE, "note": "r7 with Neptune and Triton from the 2026-10-23 Horizons fetch; each system (all members) moved rigidly onto DE440's barycentre (scripts/build_fixture_r8.py)"}
    a.out.write_text(json.dumps(new))
    for p, v in shifts.items():
        print(f"{p:8s} median rigid shift over {len(v)} snapshots: position {np.median([x[0] for x in v]):9.2f} km, velocity {np.median([x[1] for x in v]):8.3f} mm/s")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
