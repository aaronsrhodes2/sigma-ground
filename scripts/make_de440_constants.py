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
    """The header's constants exactly as JPL wrote them, plus the cited-paper values. NOTHING derived is stored: GM in km^3/s^2, the Earth/Moon split and the field blocks are calculated by
    sigma_ground/field/interface/de440_constants.py when asked (the Captain, 2026-10-26: keep calculations as calculations)."""
    c = mp.parse_header(header)
    return {
        "metadata": {"source": "JPL DE440 ASCII header header.440 (Park et al. 2021, AJ 161:105)", "header_file": str(header), "units": "raw: exactly as in the header (GM in au^3/day^2, AU in km, rates per year, radii km); other_sources: J dimensionless, radii km",
                     "generator": "scripts/make_de440_constants.py", "constants_in_header": len(c)},
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
    from sigma_ground.field.interface import de440_constants as dc
    dc.load.cache_clear()
    for k in ("Sun", "Earth", "Moon"):
        print(f"  GM {k:8s} {dc.gm_km3_s2(k):.6f} km^3/s^2 (calculated from the header values)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
