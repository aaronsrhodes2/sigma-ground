"""Fold the JPL DE440 header constants into local data: sigma_ground/field/interface/fixtures/de440_header_constants.json.

Source: the DE440 ASCII header (header.440, ssd.jpl.nasa.gov/ftp/eph/planets/ascii/de440/), the constants JPL's own integration used (Park et al. 2021, AJ 161:105). GM values in the header are au^3/day^2; they are
converted with the header's own AU (149597870.7 km) to km^3/s^2. Earth and Moon GM are split from the Earth-Moon barycentre GM (GMB) with the header's EMRAT.

The file has: `raw` (every constant, header units), `gm_km3_s2` (Sun, the planets, the Earth-Moon pair), `sun`, `earth`, `moon` (the gravity-field and tide constants with their units and the name they have in the
header), and `other_sources` (values for planets whose field the DE header does not carry: Jupiter from Juno, Saturn from Cassini, each with its citation and what was and was not read from the cited page).

usage: python scripts/make_de440_constants.py [--header PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
import make_perturber_catalog as mp  # noqa: E402

OUT = _ROOT / "sigma_ground/field/interface/fixtures/de440_header_constants.json"


def build(header: Path) -> dict:
    c = mp.parse_header(header)
    au_km = c["AU"]
    conv = au_km ** 3 / 86400.0 ** 2                      # au^3/day^2 -> km^3/s^2
    emrat = c["EMRAT"]
    gmb = c["GMB"] * conv
    gm = {"Sun": c["GMS"] * conv, "Mercury": c["GM1"] * conv, "Venus": c["GM2"] * conv, "Earth": gmb * emrat / (1.0 + emrat), "Moon": gmb / (1.0 + emrat),
          "Earth-Moon barycentre": gmb, "Mars system": c["GM4"] * conv, "Jupiter system": c["GM5"] * conv, "Saturn system": c["GM6"] * conv, "Uranus system": c["GM7"] * conv,
          "Neptune system": c["GM8"] * conv, "Pluto system": c["GM9"] * conv}
    moon_keys = sorted(k for k in c if re.fullmatch(r"(J[2-6]|[CS][2-6][0-6])M", k))
    return {
        "metadata": {"source": "JPL DE440 ASCII header header.440 (Park et al. 2021, AJ 161:105)", "header_file": str(header), "au_km": au_km, "emrat": emrat, "units": "GM km^3/s^2 (converted from au^3/day^2 with the header's AU); radii km; J, C, S dimensionless; tau days",
                     "generator": "scripts/make_de440_constants.py", "constants_in_header": len(c)},
        "gm_km3_s2": gm,
        "sun": {"gm_km3_s2": gm["Sun"], "j2": c["J2SUN"], "j3": c["J3SUN"], "j4": c["J4SUN"], "reference_radius_km": c["ASUN"], "gm_rate_per_yr": c["GMSDOT"],
                "header_names": {"j2": "J2SUN", "reference_radius_km": "ASUN", "gm_rate_per_yr": "GMSDOT"}},
        "earth": {"gm_km3_s2": gm["Earth"], "reference_radius_km": c["RE"], "j2": c["J2E"], "j3": c["J3E"], "j4": c["J4E"], "j5": c["J5E"], "j2_rate_per_yr": c["J2EDOT"], "j2_t2": c["J2ET2"],
                  "k2": [c["K2E0"], c["K2E1"], c["K2E2"]], "tidal_time_delay_days": [c["TAUE0"], c["TAUE1"], c["TAUE2"]],
                  "header_names": {"reference_radius_km": "RE", "j2": "J2E", "j3": "J3E", "j4": "J4E", "j5": "J5E", "k2": "K2E0..2", "tidal_time_delay_days": "TAUE0..2"}},
        "moon": {"gm_km3_s2": gm["Moon"], "reference_radius_km": c["AM"], "k2": c["K2M"], "tidal_time_delay_days": c["TAUM"], "libration_lbet": c["LBET"], "libration_lgam": c["LGAM"], "core_oblateness": c["COBLAT"],
                 "gravity": {k: c[k] for k in moon_keys}, "header_names": {"reference_radius_km": "AM", "gravity": ", ".join(moon_keys), "k2": "K2M", "tidal_time_delay_days": "TAUM"}},
        "other_sources": {
            "Jupiter": {"reference_radius_km": 71492.0, "j2": 14696.514e-6, "j2_uncertainty": 0.272e-6,
                        "citation": "Folkner et al. 2017 (Juno), as tabulated in arXiv:1708.06177 Table 1; radius 71492 km from its Section 2 and Appendix A",
                        "read_2026-10-23": "J2 and radius printed in that table and text"},
            "Saturn": {"reference_radius_km": 60330.0, "j2": 16290.573e-6, "j4": -935.314e-6,
                       "citation": "Iess et al. 2019 (Cassini Grand Finale), as tabulated in arXiv:2304.09215 Table IV.1",
                       "read_2026-10-23": "J2 and J4 printed; the reference radius was NOT printed on the page read. 60330 km is the radius for which J2 R^2 equals the 16324.108e-6 on 60268 km that a second page attributed to the same measurement (arithmetic, 2e-5)"},
        },
        "raw": c,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--header", type=Path, default=mp.HEADER)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    doc = build(a.header)
    a.out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print("wrote", a.out, f"({doc['metadata']['constants_in_header']} header constants)")
    for k in ("Sun", "Earth", "Moon"):
        print(f"  GM {k:8s} {doc['gm_km3_s2'][k]:.6f} km^3/s^2")
    print("  moon gravity keys:", ", ".join(doc["moon"]["gravity"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
