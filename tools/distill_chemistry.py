"""Distill the chemistry datasets into cited JSON aggregates.

Three public-domain / open sources fetched to D:/datasets/chemistry/:

  nist_atomic_weights.txt — NIST "Atomic Weights and Isotopic Compositions"
      (linearized blocks inside the CGI page): per-isotope relative atomic mass
      + isotopic composition + the standard atomic weight per element.
  burcat.thr              — the Burcat/ATcT Third-Millennium ideal-gas
      thermochemical database (NASA-7 polynomials) — the format engine-
      combustion simulation actually consumes.
  codata_allascii.txt     — the full CODATA recommended constants table.

Outputs (committable distilled aggregates, sources cited inside):
  sigma_ground/inventory/data/atomic_weights.json    (all elements + isotopes)
  sigma_ground/inventory/data/combustion_thermo.json (target species, NASA-7)
  sigma_ground/inventory/data/codata_constants.json  (core constants registry)

Verification printed: Hf°(298) recomputed FROM the polynomials vs textbook
(CO2 −393.5, H2O(g) −241.8, n-octane(g) ≈ −208.5 kJ/mol); mercury standard
atomic weight 200.592.

Run:  python tools/distill_chemistry.py
"""
import json
import os
import re

_SRC = "D:/datasets/chemistry"
_OUT = os.path.join(os.path.dirname(__file__), "..", "sigma_ground",
                    "inventory", "data")
_R = 8.31446261815324          # J/(mol K), CODATA exact-adjacent


# ── NIST atomic weights ──────────────────────────────────────────────────
def _num(s):
    """Parse NIST's '200.592(3)' / '171.00353(33#)' / '[97]' style numbers."""
    s = (s or "").strip().replace("&nbsp;", "")
    if not s:
        return None
    m = re.match(r"\[?([\d.]+)", s)
    return float(m.group(1)) if m else None


def distill_atomic_weights():
    txt = open(os.path.join(_SRC, "nist_atomic_weights.txt"), encoding="utf-8",
               errors="replace").read()
    elements = {}
    for block in txt.split("Atomic Number = ")[1:]:
        lines = block.splitlines()
        z = int(lines[0].strip())
        f = {}
        for ln in lines[1:]:
            if " = " in ln:
                k, v = ln.split(" = ", 1)
                f[k.strip()] = v.strip()
            elif not ln.strip():
                break
        sym = f.get("Atomic Symbol", "")
        el = elements.setdefault(z, {
            "symbol": sym, "z": z,
            "standard_atomic_weight": _num(f.get("Standard Atomic Weight")),
            "isotopes": []})
        if not el["standard_atomic_weight"]:
            el["standard_atomic_weight"] = _num(f.get("Standard Atomic Weight"))
        a = f.get("Mass Number")
        mass = _num(f.get("Relative Atomic Mass"))
        comp = _num(f.get("Isotopic Composition"))
        if a and mass:
            el["isotopes"].append({"a": int(a), "mass_u": mass,
                                   "abundance": comp})
    out = {"_meta": {
        "source": "NIST Atomic Weights and Isotopic Compositions "
                  "(physics.nist.gov/Compositions; NIST SP — public domain, "
                  "US government work)",
        "generated_by": "tools/distill_chemistry.py"}}
    for z in sorted(elements):
        out[elements[z]["symbol"]] = elements[z]
    dest = os.path.join(_OUT, "atomic_weights.json")
    json.dump(out, open(dest, "w", encoding="utf-8"), indent=1)
    n_iso = sum(len(e["isotopes"]) for e in elements.values())
    print(f"atomic_weights.json: {len(elements)} elements, {n_iso} isotopes")
    hg = elements[80]
    print(f"  verify Hg standard weight = {hg['standard_atomic_weight']} (expect 200.592)")
    return dest


# ── Burcat NASA-7 polynomials ────────────────────────────────────────────
# Species keys as they appear at the start of a Burcat entry's line 1.
_SPECIES = {
    "n-octane":   "C8H18,n-octane",
    "iso-octane": "C8H18,isooctane",
    "O2":  "O2 REF ELEMENT",
    "N2":  "N2  REF ELEMENT",
    "CO2": "CO2",
    "CO":  "CO ",
    "H2O": "H2O ",
    "H2":  "H2 REF ELEMENT",
    "CH4": "CH4 ",
    "OH":  "OH ",
    "NO":  "NO ",
}


def _nasa_h(coeffs, T):
    """H(T) in J/mol from NASA-7 low-T coefficients (a1..a6 used)."""
    a1, a2, a3, a4, a5, a6, _a7 = coeffs
    return _R * (a1 * T + a2 / 2 * T**2 + a3 / 3 * T**3
                 + a4 / 4 * T**4 + a5 / 5 * T**5 + a6)


def distill_burcat():
    lines = open(os.path.join(_SRC, "burcat.thr"), encoding="utf-8",
                 errors="replace").read().splitlines()
    def find_block(key):
        for i, ln in enumerate(lines):
            # a NASA line 1 ends with '1' in col 80 and carries the T range
            if ln.startswith(key) and len(ln) >= 80 and ln[79] == "1":
                return lines[i:i + 4]
        return None

    def coeffs_of(block):
        raw = "".join(ln[:75] for ln in block[1:4])
        vals = [float(raw[j:j + 15]) for j in range(0, 15 * 14, 15)
                if raw[j:j + 15].strip()]
        return vals[:7], vals[7:14]        # high-T a1..a7, low-T a1..a7

    out = {"_meta": {
        "source": "Burcat & Ruscic, Third Millennium Ideal Gas and Condensed "
                  "Phase Thermochemical Database (ATcT-based; garfield.chem."
                  "elte.hu/Burcat — free for scientific use)",
        "format": "NASA-7 polynomials; low range typically 200-1000 K, "
                  "high 1000-6000 K; units J, mol, K",
        "generated_by": "tools/distill_chemistry.py"}}
    for name, key in _SPECIES.items():
        blk = find_block(key)
        if blk is None:
            print(f"  ! {name}: '{key}' not found — skipped")
            continue
        hi, lo = coeffs_of(blk)
        hf298 = _nasa_h(lo, 298.15) / 1000.0        # kJ/mol
        out[name] = {"burcat_line1": blk[0][:60].strip(),
                     "nasa7_low": lo, "nasa7_high": hi,
                     "hf298_kj_mol": round(hf298, 2)}
        print(f"  + {name:10s} Hf298 = {hf298:8.1f} kJ/mol")
    dest = os.path.join(_OUT, "combustion_thermo.json")
    json.dump(out, open(dest, "w", encoding="utf-8"), indent=1)
    print(f"combustion_thermo.json: {len(out) - 1} species")
    return dest


# ── CODATA core constants ────────────────────────────────────────────────
_CONSTANTS = ["speed of light in vacuum", "Planck constant",
              "Boltzmann constant", "Avogadro constant", "molar gas constant",
              "elementary charge", "Newtonian constant of gravitation",
              "Stefan-Boltzmann constant", "electron mass", "proton mass",
              "fine-structure constant", "vacuum electric permittivity"]


def distill_codata():
    out = {"_meta": {
        "source": "CODATA recommended values (physics.nist.gov/cuu/Constants "
                  "allascii table — public domain)",
        "generated_by": "tools/distill_chemistry.py"}}
    for ln in open(os.path.join(_SRC, "codata_allascii.txt"),
                   encoding="utf-8", errors="replace"):
        for want in _CONSTANTS:
            if ln.lower().startswith(want.lower()):
                # fixed columns: name [0:60], value [60:85], uncertainty, unit
                val = ln[60:85].replace(" ", "").replace("...", "")
                unc = ln[85:110].replace(" ", "")
                unit = ln[110:].strip()
                try:
                    out[want] = {"value": float(val.replace("e", "E")),
                                 "uncertainty": (0.0 if "exact" in unc
                                                 else float(unc or 0.0)),
                                 "unit": unit}
                except ValueError:
                    pass
    dest = os.path.join(_OUT, "codata_constants.json")
    json.dump(out, open(dest, "w", encoding="utf-8"), indent=1)
    print(f"codata_constants.json: {len(out) - 1} constants")
    c = out.get("speed of light in vacuum", {})
    print(f"  verify c = {c.get('value')} {c.get('unit')} (expect 299792458 m s^-1)")
    return dest


if __name__ == "__main__":
    distill_atomic_weights()
    distill_burcat()
    distill_codata()
