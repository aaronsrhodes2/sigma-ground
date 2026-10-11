"""Extend de440_state_vectors.json with Iapetus, Hyperion and Phoebe (the Saturn moons the fixture lacked).

Why: the shoot-out's Saturn system sat 11.7 km from the DE440 barycentre because these three were missing (Iapetus alone has
GM 120.5 km^3/s^2, enough to perturb Titan and Rhea). Found by unification round 2 / the 2026-10-10 shoot-out fixes (Track F).
Approved by the Captain 2026-10-10: 81 Horizons requests (3 bodies x 27 annual epochs), 0.5 s apart.

Request format (as in extend_de440_fixture_saturn_moons.py): GET https://ssd.jpl.nasa.gov/api/horizons.api with
COMMAND='<NAIF id>', CENTER='500@0', EPHEM_TYPE='VECTORS', VEC_TABLE='2', REF_SYSTEM='ICRF', TLIST='<JD TDB>'.
NAIF codes: Iapetus 608, Hyperion 607, Phoebe 609.
GM (km^3/s^2), SAT441 (Jacobson 2022, AJ 164:199), read from https://ssd.jpl.nasa.gov/sats/phys_par/ on 2026-10-10:
Iapetus 120.51511 (+-0.00242), Hyperion 0.37049 (+-0.00005), Phoebe 0.55479 (+-0.00108).

Usage: python scripts/extend_de440_fixture_saturn_outer_moons.py [--input PATH] [--output PATH] [--dry-run] [--throttle-s F]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


_FIXTURE = (
    Path(__file__).parent.parent
    / "sigma_ground" / "field" / "interface" / "fixtures"
    / "de440_state_vectors.json"
)

# SAT441 GM values (JPL Saturn Satellite Ephemeris)
# Matches the existing fixture's "gm_source": "SAT441" tag.
_SATURN_MOONS = [
    {"name": "Iapetus",  "naif_id": 608, "gm_km3_s2": 120.51511, "anchor": "Saturn"},
    {"name": "Hyperion", "naif_id": 607, "gm_km3_s2": 0.37049,   "anchor": "Saturn"},
    {"name": "Phoebe",   "naif_id": 609, "gm_km3_s2": 0.55479,   "anchor": "Saturn"},
]

# CODATA 2018 G, same as the existing fixture metadata uses
_G_M3_KG_S2 = 6.67430e-11

# Pattern that matches a single state-vector row inside $$SOE / $$EOE
# Example:
#   2457023.750000000 = A.D. 2015-Jan-01 06:00:00.0000 TDB
#    X = ...  Y = ...  Z = ...
#    VX= ...  VY= ...  VZ= ...
_NUM_RE = re.compile(r"[-+]?\d+\.\d+E[+-]\d+")


def _query_horizons_state_vector(naif_id: int, jd_tdb: float,
                                   timeout_s: float = 30.0) -> dict:
    """Hit Horizons for the SSB-centred ICRF state vector at jd_tdb.

    Returns dict with x_km, y_km, z_km, vx_km_s, vy_km_s, vz_km_s.
    Raises RuntimeError if the response can't be parsed.
    """
    params = {
        "format":      "text",
        "COMMAND":     f"'{naif_id}'",
        "CENTER":      "'500@0'",     # Solar System Barycentre
        "EPHEM_TYPE":  "'VECTORS'",
        "OUT_UNITS":   "'KM-S'",
        "VEC_TABLE":   "'2'",         # position + velocity, no light-time
        "REF_PLANE":   "'FRAME'",     # ICRF (J2000 equatorial)
        "REF_SYSTEM":  "'ICRF'",
        "CSV_FORMAT":  "'NO'",
        "TLIST":       f"'{jd_tdb}'",
        "TIME_DIGITS": "'FRACSEC'",
    }
    url = ("https://ssd.jpl.nasa.gov/api/horizons.api?"
           + urllib.parse.urlencode(params))
    req = urllib.request.Request(url, headers={"User-Agent": "sigma-ground/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        text = r.read().decode()

    soe, eoe = text.find("$$SOE"), text.find("$$EOE")
    if soe < 0 or eoe < 0:
        raise RuntimeError(
            f"Horizons response for body {naif_id} at JD {jd_tdb} "
            f"lacks $$SOE/$$EOE markers.\nFirst 500 chars: {text[:500]!r}"
        )

    block = text[soe:eoe]
    nums = _NUM_RE.findall(block)
    if len(nums) < 6:
        raise RuntimeError(
            f"Horizons block for body {naif_id} parsed to {len(nums)} "
            f"numbers, expected >=6.\nBlock:\n{block}"
        )
    x, y, z, vx, vy, vz = (float(n) for n in nums[:6])
    return {
        "x_km":     x,
        "y_km":     y,
        "z_km":     z,
        "vx_km_s":  vx,
        "vy_km_s":  vy,
        "vz_km_s":  vz,
    }


def _build_body_record(moon: dict, sv: dict) -> dict:
    """Match the schema the existing fixture uses for satellite bodies."""
    return {
        "name":                  moon["name"],
        "sol_complex_name":      moon["name"],
        "horizons_id":           moon["naif_id"],
        "gm_km3_s2":             moon["gm_km3_s2"],
        "gm_source":             "SAT441",
        "mass_kg":               moon["gm_km3_s2"] * 1e9 / _G_M3_KG_S2,
        "gravitational_anchor":  moon["anchor"],
        "state_vector":          sv,
    }


def extend_fixture(input_path: Path, output_path: Path,
                    dry_run: bool, throttle_s: float) -> None:
    print(f"Loading {input_path.name}")
    with input_path.open() as f:
        fixture = json.load(f)

    snapshots = fixture["snapshots"]
    snap_keys = sorted(snapshots.keys())
    print(f"Found {len(snap_keys)} snapshots: {snap_keys[0]} .. {snap_keys[-1]}")

    # Skip snapshots that already contain all four bodies (idempotent reruns)
    needed_names = {m["name"] for m in _SATURN_MOONS}

    total_calls = 0
    for snap_key in snap_keys:
        snap     = snapshots[snap_key]
        jd_tdb   = snap["epoch"]["jd_tdb"]
        existing = {b["name"] for b in snap["bodies"]}
        missing  = needed_names - existing
        if not missing:
            print(f"  {snap_key}: already complete, skipping")
            continue

        print(f"  {snap_key} (JD={jd_tdb}): fetching {len(missing)} moons")
        for moon in _SATURN_MOONS:
            if moon["name"] not in missing:
                continue
            try:
                sv = _query_horizons_state_vector(moon["naif_id"], jd_tdb)
            except Exception as e:
                print(f"    !! {moon['name']:<10} FAILED: {e}", file=sys.stderr)
                raise
            total_calls += 1
            print(f"    {moon['name']:<10} ok  "
                  f"(x={sv['x_km']:+.3e}, vx={sv['vx_km_s']:+.3f})")
            if not dry_run:
                snap["bodies"].append(_build_body_record(moon, sv))
            time.sleep(throttle_s)

    print(f"\nTotal Horizons calls: {total_calls}")

    if dry_run:
        print("Dry run -- not writing output")
        return

    # Re-sort each snapshot's bodies for stable diffs (existing order: by
    # gravitational_anchor + name -- but the fixture isn't strictly sorted,
    # so we just append. That keeps diffs clean.)

    # Update top-level sources/notes
    fixture.setdefault("notes", {})
    extension_note = (
        "Saturn moons Iapetus (608), Hyperion (607) and Phoebe (609) added 2026-10-10 from JPL Horizons "
        "(81 requests, SSB-centred ICRF vectors at the 27 annual epochs); GM from SAT441 (Jacobson 2022, AJ 164:199) "
        "via ssd.jpl.nasa.gov/sats/phys_par. Removes the 11.7 km offset between the fixture's Saturn system "
        "centre and the DE440 barycentre. See scripts/extend_de440_fixture_saturn_outer_moons.py."
    )
    fixture["notes"]["saturn_outer_moons_extension_2026_10_10"] = extension_note

    print(f"Writing {output_path.name}")
    with output_path.open("w") as f:
        json.dump(fixture, f, indent=2)
    print(f"Done. File size: {output_path.stat().st_size:,} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",  type=Path, default=_FIXTURE)
    parser.add_argument("--output", type=Path, default=_FIXTURE)
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--throttle-s", type=float, default=0.5)
    args = parser.parse_args()
    extend_fixture(args.input, args.output, args.dry_run, args.throttle_s)


if __name__ == "__main__":
    main()
