"""Distill measured n,k optical constants from the refractiveindex.info database.

Input: the locally-cloned polyanskiy/refractiveindex.info-database (open data
compiled from published papers; the yml files name each paper). Output: a small
cited JSON aggregate — for every material in our render catalog, the complex
refractive index (n, k) linearly interpolated at the renderer's three RGB
wavelengths (650/550/450 nm), with the source paper named per material.

This REPLACES hand-smoothed approximations with the actual tabulations (the
session's spot-check found our mercury values were rounded versions of Inagaki —
right reflectance, inexact n,k). Only tabulated-nk/-n references are used;
formula-based (Sellmeier) glass entries are out of scope here (glass n comes
from the existing Cauchy tables). Raw database stays local; this aggregate is
the committable artifact.

Run:  python tools/distill_optics_nk.py
"""
import json
import os

import yaml

_DB = "D:/datasets/optics/refractiveindex.info-database/database/data"
_OUT = os.path.join(os.path.dirname(__file__), "..", "sigma_ground",
                    "inventory", "data", "optics_nk.json")
_WAVELENGTHS_NM = (650.0, 550.0, 450.0)

# our material name -> (db path under data/, human citation)
# References chosen: the standard visible-range tabulated measurements.
MATERIALS = {
    "gold":      ("main/Au/nk/Johnson.yml",  "Johnson & Christy 1972, PRB 6, 4370"),
    "silver":    ("main/Ag/nk/Johnson.yml",  "Johnson & Christy 1972, PRB 6, 4370"),
    "copper":    ("main/Cu/nk/Johnson.yml",  "Johnson & Christy 1972, PRB 6, 4370"),
    "aluminum":  ("main/Al/nk/Rakic.yml",    "Rakic 1995, Appl. Opt. 34, 4755"),
    "iron":      ("main/Fe/nk/Johnson.yml",  "Johnson & Christy 1974, PRB 9, 5056"),
    "titanium":  ("main/Ti/nk/Johnson.yml",  "Johnson & Christy 1974, PRB 9, 5056"),
    "nickel":    ("main/Ni/nk/Johnson.yml",  "Johnson & Christy 1974, PRB 9, 5056"),
    "chromium":  ("main/Cr/nk/Johnson.yml",  "Johnson & Christy 1974, PRB 9, 5056"),
    "tungsten":  ("main/W/nk/Werner.yml",    "Werner et al. 2009, JPCRD 38, 1013"),
    "lead":      ("main/Pb/nk/Werner.yml",   "Werner et al. 2009, JPCRD 38, 1013"),
    "platinum":  ("main/Pt/nk/Werner.yml",   "Werner et al. 2009, JPCRD 38, 1013"),
    "zinc":      ("main/Zn/nk/Werner.yml",   "Werner et al. 2009, JPCRD 38, 1013"),
    "mercury":   ("main/Hg/nk/Inagaki.yml",  "Inagaki et al. 1981, PRB 23, 5246"),
    "silicon":   ("main/Si/nk/Aspnes.yml",   "Aspnes & Studna 1983, PRB 27, 985"),
    "germanium": ("main/Ge/nk/Aspnes.yml",   "Aspnes & Studna 1983, PRB 27, 985"),
    "water":     ("main/H2O/nk/Hale.yml",    "Hale & Querry 1973, Appl. Opt. 12, 555"),
}


def _tabulated(yml_path):
    """Parse a rii yml → sorted [(lambda_um, n, k)] from its tabulated blocks.

    Handles 'tabulated nk' (3 cols) and separate 'tabulated n'/'tabulated k'
    blocks (2 cols each, merged on the n-grid with k interpolated)."""
    with open(yml_path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    n_rows, k_rows, nk_rows = [], [], []
    for block in doc.get("DATA", []):
        btype = (block.get("type") or "").strip()
        raw = block.get("data") or ""
        rows = []
        for line in raw.splitlines():
            parts = line.split()
            if parts and all(_is_num(p) for p in parts):
                rows.append([float(p) for p in parts])
        if btype == "tabulated nk":
            nk_rows += [(r[0], r[1], r[2]) for r in rows if len(r) >= 3]
        elif btype == "tabulated n":
            n_rows += [(r[0], r[1]) for r in rows if len(r) >= 2]
        elif btype == "tabulated k":
            k_rows += [(r[0], r[1]) for r in rows if len(r) >= 2]
    if nk_rows:
        return sorted(nk_rows)
    if n_rows:
        n_rows.sort()
        k_rows.sort()
        return [(l, n, _interp1(k_rows, l) or 0.0) for l, n in n_rows]
    return []


def _is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _interp1(rows, x):
    """Linear interpolation on [(x, y)] rows; None outside the range."""
    for (x0, y0), (x1, y1) in zip(rows, rows[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
            return y0 + t * (y1 - y0)
    return None


def _interp3(rows, lam_um):
    for (l0, n0, k0), (l1, n1, k1) in zip(rows, rows[1:]):
        if l0 <= lam_um <= l1:
            t = (lam_um - l0) / (l1 - l0) if l1 > l0 else 0.0
            return n0 + t * (n1 - n0), k0 + t * (k1 - k0)
    return None


def main() -> None:
    out = {"_meta": {
        "source": "refractiveindex.info database (open data; papers cited per entry)",
        "method": "linear interpolation of the published tabulation at the "
                  "renderer RGB wavelengths",
        "wavelengths_nm": list(_WAVELENGTHS_NM),
        "generated_by": "tools/distill_optics_nk.py",
    }}
    for name, (rel, cite) in sorted(MATERIALS.items()):
        path = os.path.join(_DB, rel)
        if not os.path.exists(path):
            print(f"  ! {name}: {rel} missing — skipped")
            continue
        rows = _tabulated(path)
        if not rows:
            print(f"  ! {name}: no tabulated data — skipped")
            continue
        entry = {"ref": rel, "citation": cite, "nk_by_nm": {}}
        for nm in _WAVELENGTHS_NM:
            got = _interp3(rows, nm / 1000.0)
            if got is None:
                print(f"  ! {name}: {nm:.0f}nm outside tabulated range "
                      f"({rows[0][0]:.3f}-{rows[-1][0]:.3f} um)")
                continue
            entry["nk_by_nm"][f"{nm:.0f}"] = [round(got[0], 4), round(got[1], 4)]
        if entry["nk_by_nm"]:
            out[name] = entry
            trip = "  ".join(f"{w}nm n={v[0]:.3f} k={v[1]:.3f}"
                             for w, v in entry["nk_by_nm"].items())
            print(f"  + {name:10s} {trip}")
    dest = os.path.abspath(_OUT)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(f"\nwrote {dest} ({len(out) - 1} materials)")


if __name__ == "__main__":
    main()
