"""Distill Particle Data Group masses into a cited JSON aggregate.

Source: the PDG bulk SQLite database, D:/datasets/pdg/pdg-<edition>.sqlite —
the authoritative, annually-reviewed particle-physics reference (the
particle-physics analog of IAU nominal values / JPL ephemeris this project
already vendors for celestial bodies and isotopes). Fetched via the official
`pdg` Python package (pip install pdg), which bundles the current edition's
SQLite file directly -- no separate download URL needed. The package itself
is NOT a runtime dependency of sigma-ground; it is used only to extract this
JSON, matching the project's "download once, distill, vendor a
dependency-free table" doctrine (same as isotopes.json <- AME2020).

Output: sigma_ground/inventory/data/particle_masses.json -- MeV/c^2 central
values + asymmetric errors for the six quarks (MS-bar current mass scheme),
three charged leptons, W/Z/Higgs bosons, and proton/neutron (cross-check
against the AME2020-derived values already in core/constants.py).

NON-DERIVED (audit): this file records what PDG PUBLISHES. Whether/how those
values get adopted into core/constants.py's PhysicalConstants (which some
other code path may have hand-transcribed from an earlier PDG edition) is a
separate, deliberate decision -- this script never writes to constants.py.

Run:  python tools/distill_pdg.py [path/to/pdg.sqlite]
      (defaults to D:/datasets/pdg/pdg-2026.0.sqlite)
"""
import json
import os
import sqlite3
import sys

_OUT = os.path.join(os.path.dirname(__file__), "..", "sigma_ground",
                    "inventory", "data", "particle_masses.json")
_DEFAULT_DB = "D:/datasets/pdg/pdg-2026.0.sqlite"

# name -> (pdg short name or mcid, is_mcid, category)
_TARGETS = [
    ("up",      "u",  False, "quark"),
    ("down",    "d",  False, "quark"),
    ("strange", "s",  False, "quark"),
    ("charm",   "c",  False, "quark"),
    ("bottom",  "b",  False, "quark"),
    ("top",     "t",  False, "quark"),
    ("electron", 11,  True,  "lepton"),
    ("muon",     13,  True,  "lepton"),
    ("tau",      15,  True,  "lepton"),
    ("W_boson",  24,  True,  "boson"),
    ("Z_boson",  23,  True,  "boson"),
    ("Higgs",    25,  True,  "boson"),
    ("proton",   2212, True, "nucleon"),
    ("neutron",  2112, True, "nucleon"),
]


def _best_mass(masses):
    """Prefer a MeV-native entry; fall back to GeV converted to MeV."""
    best = None
    for m in masses:
        if m.units == "MeV":
            return m, m.value
        if best is None and m.units == "GeV":
            best = m
    if best is not None:
        return best, best.value * 1000.0
    return None, None


def distill(db_path=None):
    import pdg  # local import -- NOT a sigma-ground runtime dependency

    db_path = db_path or _DEFAULT_DB
    api = pdg.connect(f"sqlite:///{db_path}")

    out = {
        "_source": "Particle Data Group (PDG), Review of Particle Physics",
        "_source_url": "https://pdg.lbl.gov/",
        "_edition": str(api.edition),
        "_license_note": "PDG data -- freely usable with citation; see "
                         "https://pdg.lbl.gov/2025/reviews/rpp2024-rev-intro.pdf "
                         "§ 'Citing the Review'",
        "_units": "MeV/c^2 (mass, mass_error_positive, mass_error_negative)",
        "_note": "Quark masses are MS-bar current (Lagrangian) masses, NOT "
                 "constituent (QCD-dressed) masses -- matches the "
                 "'bare_mass_mev' field in inventory/models/quark.py.",
        "particles": {},
    }

    for label, key, by_mcid, category in _TARGETS:
        p = api.get_particle_by_mcid(key) if by_mcid else api.get_particle_by_name(key)
        m, mev = _best_mass(list(p.masses()))
        if m is None:
            out["particles"][label] = {"category": category, "pdg_name": p.name,
                                       "pdgid": p.pdgid, "mass_mev": None,
                                       "note": "no mass entry found"}
            continue
        out["particles"][label] = {
            "category": category,
            "pdg_name": p.name,
            "pdgid": p.pdgid,
            "mass_mev": mev,
            "mass_error_positive_mev": (m.error_positive * 1000.0
                                        if m.units == "GeV" else m.error_positive),
            "mass_error_negative_mev": (m.error_negative * 1000.0
                                        if m.units == "GeV" else m.error_negative),
        }

    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    with open(_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=False)
        f.write("\n")
    return out


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else None
    result = distill(db)
    print(f"PDG edition {result['_edition']} -> {_OUT}")
    for label, entry in result["particles"].items():
        mev = entry.get("mass_mev")
        print(f"  {label:10s} {entry['pdg_name']:6s} "
              f"{'%.6f MeV' % mev if mev is not None else 'N/A':>18s}")
