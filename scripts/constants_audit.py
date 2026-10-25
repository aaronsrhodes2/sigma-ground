"""Audit: the constants our predictor table (rolling_shootout._BODY_PARAMS) and Mentat's body lookup (mcp/tools/astronomy.py) carried, against DE440's header and the Juno/Cassini papers (de440_constants.py),
and WHY each one differs.

Every difference is classified: convention (a mean radius used where the coefficient is defined on an equatorial or reference radius), rounding (digits dropped), vintage (an older measurement), or unresolved.
The reason text is from the code's own comments and from the radius conventions; where no source for the old value could be found it says so. Nothing here changes any stored result.

usage: python scripts/constants_audit.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
RES = _ROOT / "sigma_ground/field/interface/fixtures/v2physics_restated"

# (body, quantity) -> (class, why). The 'why' is what the evidence shows, with its limits.
WHY = {
    ("Sun", "radius_km"): ("convention", "ours is the IAU 2015 nominal solar radius (695,700 km, the Mentat table has the same); DE440's J2 is defined on its own reference radius ASUN = 696,000 km"),
    ("Sun", "j2"): ("unresolved", "the code comment calls 2.1106e-7 'the DE440 fitted value (Park et al. 2021 Table 3)'; the header that JPL's integration read has 2.19614e-7. Table 3 was not re-read, so the origin of 2.1106e-7 is not established; the header is what was integrated"),
    ("Sun", "j4"): ("vintage", "ours (-9e-7) is not in DE440, whose header sets J3 = J4 = 0 for the Sun; no source for -9e-7 was found in the code"),
    ("Earth", "radius_km"): ("convention", "6371.0 is the Fact Sheet's volumetric MEAN radius; Earth's J2, J3, J4 are defined on the EQUATORIAL radius 6378.1366 km. Using the mean radius in the force makes J2 R^2 0.22 % low"),
    ("Earth", "j2"): ("rounding", "1.0826e-3 is 1.08262539e-3 cut to five digits"),
    ("Earth", "j3"): ("rounding", "-2.5e-6 is -2.53241e-6 cut to two digits"),
    ("Earth", "j4"): ("rounding", "-1.6e-6 is -1.619898e-6 cut to two digits"),
    ("Moon", "radius_km"): ("convention", "1737.4 is the Fact Sheet's mean radius; DE440's lunar harmonics (J2M ... C66M, C22M) are defined on AM = 1738.0 km"),
    ("Moon", "j2"): ("vintage", "2.034e-4 against the header's 2.03214e-4 (0.09 %); the same size as the difference between lunar gravity solutions of different eras, source of ours not found in the code"),
    ("Moon", "j3"): ("rounding", "8.5e-6 is 8.45965e-6 cut to two digits"),
    ("Moon", "j4"): ("vintage", "-1.2e-5 against the header's -9.704e-6 (24 %): an older lunar gravity field; DE440 uses the GRAIL-era value. Source of ours not found in the code"),
    ("Jupiter", "j2"): ("vintage", "1.4736e-2 is the pre-Juno value; Juno measured 14696.514e-6 +- 0.272e-6 (Folkner et al. 2017, arXiv:1708.06177 Table 1). Already replaced in restated_zon"),
    ("Saturn", "j2"): ("vintage", "1.6298e-2 is the pre-Cassini-Grand-Finale value; Cassini gives 16290.573e-6 (Iess et al. 2019, arXiv:2304.09215 Table IV.1). Already replaced in restated_zon"),
    ("Saturn", "radius_km"): ("convention", "60268 km is the Fact Sheet's equatorial (1 bar) radius; the Cassini field is normalised on 60330 km (arithmetic from a second source, not printed on the page read)"),
    ("Saturn", "j4"): ("vintage", "-9.15e-4 on 60268 km against Cassini's -935.314e-6 on 60330 km (-939.5e-6 on 60268 km): 2.6 % after converting the radius"),
}
NOTE_UNCHECKED = {"Mars": "Mars (R 3389.5 km mean, J2 1.9606e-3) has no DE440-header or cited-paper value in the local data; the radius pattern is the same as Earth's and the Moon's (mean radius with an equatorial coefficient) but that is NOT verified here",
                  "Uranus": "Uranus (R 25559, J2 3.5107e-3) and Neptune (R 24764, J2 3.539e-3): J2 R^2 agrees with the satellite-ephemeris papers' values to 1e-3 (checked by arithmetic against remembered values: Uranus 3341.29e-6 on 26200 km, Neptune 3408.43e-6 on 25225 km; not re-read)"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=RES)
    a = ap.parse_args()
    from sigma_ground.field import constants as fc
    from sigma_ground.field.interface import de440_constants as dc
    from sigma_ground.field.interface import rolling_shootout as rs
    from sigma_ground.mcp.tools import astronomy as ast
    rows = []
    for name in ("Sun", "Earth", "Moon", "Jupiter", "Saturn"):
        ours = rs._BODY_PARAMS[name]
        theirs = dc.body(name)
        mentat = ast.solar_system_body(name).to_dict()["value"]
        for q, mine, de in (("radius_km", ours.radius_km, theirs["reference_radius_km"]), ("j2", ours.j2, theirs.get("j2")), ("j3", ours.j3, theirs.get("j3")), ("j4", ours.j4, theirs.get("j4"))):
            if de is None or (mine == de):
                continue
            rel = (mine - de) / de if de else 0.0
            cls, why = WHY.get((name, q), ("unclassified", "no reason recorded"))
            rows.append({"body": name, "quantity": q, "ours": mine, "de440_or_paper": de, "relative_difference": rel, "class": cls, "why": why,
                         "source_of_new": theirs["source"], "mentat_radius_km": mentat["radius_m"] / 1e3 if q == "radius_km" else None})
    gm_nominal = fc.GM_SUN_M3_S2 / 1e9
    gm_de = dc.gm_km3_s2("Sun")
    rows.append({"body": "Sun", "quantity": "gm_km3_s2", "ours": gm_nominal, "de440_or_paper": gm_de, "relative_difference": (gm_nominal - gm_de) / gm_de, "class": "rounding",
                 "why": "field.constants uses the IAU 2015 nominal GM (1.3271244e20 m^3/s^2, exact by definition of the nominal); DE440's fitted value has more digits. Irrelevant to the shoot-out, which takes GM from the fixture", "source_of_new": "JPL DE440 header", "mentat_radius_km": None})
    lines = ["# Constants: ours against DE440 (and why they differ)", "",
             "Ours = the predictor table `rolling_shootout._BODY_PARAMS` (and, for radius, Mentat's `solar_system_body` before 2026-10-24, which reads the Fact Sheet's mean radii). Theirs = `fixtures/de440_header_constants.json` (DE440 header; Jupiter, Saturn from the Juno and Cassini papers). "
             "Both Mentat and the predictor now read that file through `de440_constants.py`, so they cannot disagree.", "",
             "| body | quantity | ours | DE440 / paper | difference | class | why |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        diff = "n/a (DE440 = 0)" if r["de440_or_paper"] == 0 else f"{100 * r['relative_difference']:+.3g} %"
        lines.append(f"| {r['body']} | {r['quantity']} | {r['ours']:.9g} | {r['de440_or_paper']:.9g} | {diff} | {r['class']} | {r['why']} |")
    lines += ["", "## Not compared", ""] + [f"* {v}" for v in NOTE_UNCHECKED.values()]
    lines += ["", "## Effect", "", "None of these changes any stored result. The predictor label `restated_hdr` applies the Sun, Earth and Moon header values (plus the Jupiter and Saturn ones of `restated_zon`); its effect on the answer key is a separate run."]
    text = "\n".join(lines)
    print(text)
    (a.out / "CONSTANTS_AUDIT.md").write_text(text + "\n", encoding="utf-8")
    (a.out / "CONSTANTS_AUDIT.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
