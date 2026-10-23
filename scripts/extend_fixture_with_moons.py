"""Add moons fetched from Horizons (fixtures/horizons_moons/) to the r4 fixture as extra bodies, written to a NEW fixture (r5). Nothing existing is changed.

Each new moon sits in a planet system that r4 moved rigidly onto DE440's barycentre (scripts/shift_giant_systems.py). The same offset (position and velocity, per snapshot) is applied to the new moon,
so its state relative to its planet is exactly the Horizons one and the system stays consistent. GM comes from the Horizons header where given; otherwise the moon is a massless test body
(its pull on the planet is below 1e-9 of the planet's; a massless body feels everything and pulls on nothing).

usage: python scripts/extend_fixture_with_moons.py --moons Himalia Nereid Helene [--out PATH]
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
FIX = _ROOT / "sigma_ground/field/interface/fixtures"
G = 6.6743e-11
SYSTEM_OF = {"Amalthea": "Jupiter", "Thebe": "Jupiter", "Himalia": "Jupiter", "Janus": "Saturn", "Epimetheus": "Saturn", "Prometheus": "Saturn", "Pandora": "Saturn", "Atlas": "Saturn",
             "Pan": "Saturn", "Helene": "Saturn", "Puck": "Uranus", "Portia": "Uranus", "Juliet": "Uranus", "Belinda": "Uranus", "Proteus": "Neptune", "Nereid": "Neptune", "Larissa": "Neptune",
             "Galatea": "Neptune", "Despina": "Neptune", "Nix": "Pluto", "Hydra": "Pluto", "Kerberos": "Pluto", "Styx": "Pluto"}


_GM_RE = re.compile(r"GM[^=\n]*=\s*([-+]?\d+\.?\d*(?:[Ee][-+]?\d+)?)")


def parse_gm(text: str) -> float | None:
    """The first 'GM ... = value' of a Horizons header, km^3/s^2; the value is read up to any '+-' uncertainty ('GM= 0.001496+-0.0005' gives 0.001496)."""
    m = _GM_RE.search(text)
    return float(m.group(1)) if m else None


def system_offset(snap_orig: dict, planet: str, kernel, seg: int):
    """(dX km, dV km/s) that shift_giant_systems.py applied to the system of `planet` in this snapshot (computed from the ORIGINAL fixture)."""
    POS, VEL = ("x_km", "y_km", "z_km"), ("vx_km_s", "vy_km_s", "vz_km_s")
    by = {b["name"]: b for b in snap_orig["bodies"]}
    anchor = {b["name"]: b.get("gravitational_anchor") for b in snap_orig["bodies"]}
    members = [planet] + [m for m, an in anchor.items() if an == planet]
    gm = np.array([by[n]["gm_km3_s2"] for n in members])
    X = np.array([[by[n]["state_vector"][c] for c in POS] for n in members])
    V = np.array([[by[n]["state_vector"][c] for c in VEL] for n in members])
    pb, vb = kernel[0, seg].compute_and_differentiate(snap_orig["epoch"]["jd_tdb"])
    return np.asarray(pb, dtype=float) - (gm[:, None] * X).sum(0) / gm.sum(), np.asarray(vb, dtype=float) / 86400.0 - (gm[:, None] * V).sum(0) / gm.sum()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--moons", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=FIX / "de440_state_vectors_v2physics_r5.json")
    a = ap.parse_args()
    from jplephem.spk import SPK
    from sigma_ground.field.interface.comet_truth import parse_horizons_comet
    kernel = SPK.open("D:/datasets/jpl/de440s.bsp")
    orig = json.loads((FIX / "de440_state_vectors_v2physics.json").read_text())["snapshots"]
    new = json.loads((FIX / "de440_state_vectors_v2physics_r4.json").read_text())
    seg = {"Jupiter": 5, "Saturn": 6, "Uranus": 7, "Neptune": 8, "Pluto": 9}
    keys = sorted((k for k in new["snapshots"] if k.startswith("j")), key=lambda k: int(k[1:]))
    added = {}
    for name in a.moons:
        f = next((FIX / "horizons_moons").glob(f"horizons_moon_*_{name}.json"))
        t = parse_horizons_comet(f)
        res = json.loads(f.read_text())["result"]
        g = parse_gm(res)
        m = g is not None
        gm = g if m else 0.0
        hid = int(re.search(r"horizons_moon_(\d+)_", f.name).group(1))
        planet = SYSTEM_OF[name]
        for i, key in enumerate(keys):
            assert abs(t["jd"][i] - new["snapshots"][key]["epoch"]["jd_tdb"]) < 1e-3, (name, key)
            dX, dV = system_offset(orig[key], planet, kernel, seg[planet])
            s = t["state_km_kmday"][i]
            pos, vel = s[:3] + dX, s[3:] / 86400.0 + dV
            new["snapshots"][key]["bodies"].append({"name": name, "sol_complex_name": name, "horizons_id": hid, "gm_km3_s2": gm, "gm_source": ("Horizons header" if m else "massless test body"),
                                                    "mass_kg": gm * 1e9 / G, "gravitational_anchor": planet,
                                                    "state_vector": {"x_km": float(pos[0]), "y_km": float(pos[1]), "z_km": float(pos[2]), "vx_km_s": float(vel[0]), "vy_km_s": float(vel[1]), "vz_km_s": float(vel[2])}})
        added[name] = {"system": planet, "gm_km3_s2": gm}
    new.setdefault("metadata", {})["r5_moons"] = {"added": added, "note": "r4 plus Horizons moons shifted with their systems (scripts/extend_fixture_with_moons.py); gm 0 = massless test body"}
    a.out.write_text(json.dumps(new))
    print("added", added, "->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
