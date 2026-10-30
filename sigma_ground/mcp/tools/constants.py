"""Physical constant lookup.

Three sources, in priority order:
  1. sigma_ground.field.constants -- our tagged constants with full
     provenance (CODATA-sourced fundamental constants, plus
     SSBM-specific values where they exist).
  2. scipy.constants -- broader CODATA 2018 / 2022 coverage for
     constants we haven't curated.
  3. Empty result with a 'not found' note if neither source has it.

The LLM front-end picks the right name. Common synonyms ('hbar' vs
'reduced_planck_constant') are mapped to the canonical scipy.constants
key.
"""

from __future__ import annotations

import dataclasses
import re

from sigma_ground.mcp.provenance import ToolResult


def _lookup_sigma_ground(name: str) -> ToolResult | None:
    """Try sigma_ground.field.constants. Returns None if not found."""
    try:
        import sigma_ground.field.constants as sgc
    except ImportError:
        return None

    # Constants in sigma_ground.field.constants are module-level uppercase
    # names. We allow case-insensitive lookup.
    canonical = name.upper().replace(" ", "_")
    if not hasattr(sgc, canonical):
        return None

    value = getattr(sgc, canonical)
    if value is None or callable(value):
        return None

    # Extract provenance tag from the constant's inline comment, if present.
    # Constants are tagged via [VERIFIED] / [DERIVED] / [EMPIRICAL-INPUT]
    # / [SPECULATIVE-PENDING] / [REJECTED] in their inline comments.
    tag = _extract_tag_for_constant(canonical)

    result = ToolResult(
        value=value,
        units=_guess_units(canonical),
        source="sigma_ground.field.constants",
        provenance_tag=tag,
        inputs={"name": name, "canonical": canonical},
    )
    # A curated fundamental constant carries CODATA's uncertainty (audit
    # 2026-10-27, finding 11: uncertainty was null on every constant). Only
    # when the stored value IS the CODATA value; a drifted copy is flagged.
    key = _CURATED_TO_CODATA.get(canonical)
    if key is not None:
        codata = _codata_result(key, name)
        if codata is not None:
            if abs(value - codata.value) <= 1e-9 * abs(codata.value):
                result = dataclasses.replace(result, uncertainty=codata.uncertainty, provenance_tag=tag or "VERIFIED",
                                             source=result.source + f"; CODATA: {key}", notes=codata.notes or result.notes)
            else:
                result = dataclasses.replace(result, notes=f"Differs from CODATA ({key} = {codata.value}) by {value - codata.value:+.3e}.")
    return result


# curated field.constants name -> CODATA row (uncertainty source). Only exact
# fundamental constants, never derived or σ-dependent ones.
_CURATED_TO_CODATA: dict[str, str] = {
    "G": "Newtonian constant of gravitation",
    "C": "speed of light in vacuum",
    "HBAR": "reduced Planck constant",
    "E_CHARGE": "elementary charge",
    "K_B": "Boltzmann constant",
    "MU_0": "vacuum mag. permeability",
    "EPS_0": "vacuum electric permittivity",
    "N_AVOGADRO": "Avogadro constant",
}


# scipy module attribute -> its key in scipy.constants.physical_constants, the
# CODATA table that carries units and uncertainty. The attributes themselves
# are bare floats; looking one up by attribute used to return units "" and no
# uncertainty (audit 2026-10-27). Several attributes share a value with other
# CODATA rows ("electron volt" = "elementary charge"), so the map is explicit,
# never by value. Attributes not listed (angstrom, calorie, day, ...) are unit
# conversion factors, not CODATA constants, and keep the attribute path.
_SCIPY_ATTR_TO_CODATA: dict[str, str] = {
    "c": "speed of light in vacuum", "speed_of_light": "speed of light in vacuum",
    "h": "Planck constant", "Planck": "Planck constant",
    "hbar": "reduced Planck constant",
    "G": "Newtonian constant of gravitation", "gravitational_constant": "Newtonian constant of gravitation",
    "g": "standard acceleration of gravity",
    "e": "elementary charge", "elementary_charge": "elementary charge",
    "eV": "electron volt", "electron_volt": "electron volt",
    "k": "Boltzmann constant", "Boltzmann": "Boltzmann constant",
    "R": "molar gas constant", "gas_constant": "molar gas constant",
    "N_A": "Avogadro constant", "Avogadro": "Avogadro constant",
    "alpha": "fine-structure constant", "fine_structure": "fine-structure constant",
    "sigma": "Stefan-Boltzmann constant", "Stefan_Boltzmann": "Stefan-Boltzmann constant",
    "Rydberg": "Rydberg constant",
    "Wien": "Wien wavelength displacement law constant",
    "epsilon_0": "vacuum electric permittivity",
    "mu_0": "vacuum mag. permeability",
    "m_e": "electron mass", "electron_mass": "electron mass",
    "m_p": "proton mass", "proton_mass": "proton mass",
    "m_n": "neutron mass", "neutron_mass": "neutron mass",
    "m_u": "atomic mass constant", "u": "atomic mass constant", "atomic_mass": "atomic mass constant",
    "atm": "standard atmosphere", "atmosphere": "standard atmosphere",
}

# CODATA writes units as "m s^-1", "J Hz^-1", "kg"; the contract's unit
# normaliser and pint want "m / s", "J / Hz".
_NEG_EXP = re.compile(r"\s*([A-Za-z]+)\^-(\d+)")


def _codata_units(unit_text: str) -> str:
    """'m^3 kg^-1 s^-2' -> 'm^3 / (kg s^2)'; 'm s^-1' -> 'm / s'; 'kg' -> 'kg'."""
    denom = []
    def take(m: re.Match) -> str:
        sym, exp = m.group(1), m.group(2)
        denom.append(sym if exp == "1" else f"{sym}^{exp}")
        return ""
    numer = _NEG_EXP.sub(take, unit_text).strip()
    if not denom:
        return numer
    numer = numer or "1"
    return f"{numer} / {denom[0]}" if len(denom) == 1 else f"{numer} / ({' '.join(denom)})"


def _codata_result(key: str, name: str) -> ToolResult | None:
    """The CODATA row `key` as a ToolResult with value, units and uncertainty. None if scipy lacks the row."""
    from scipy import constants as spc
    try:
        value, units, uncertainty = spc.physical_constants[key]
    except KeyError:
        return None
    return ToolResult(
        value=float(value),
        units=_codata_units(units),
        source=f"scipy.constants (CODATA: {key})",
        uncertainty=abs(uncertainty) if uncertainty else None,
        provenance_tag="VERIFIED",
        inputs={"name": name, "codata_key": key},
        notes="" if uncertainty else "Exact by definition in the SI (uncertainty 0).",
    )


def _lookup_scipy(name: str) -> ToolResult | None:
    """Try scipy.constants. Returns None if not found."""
    try:
        from scipy import constants as spc
    except ImportError:
        return None

    # The CODATA table first: it is the only scipy source with units and
    # uncertainty. Reached by the table's own key ("speed of light in vacuum")
    # or by the module attribute that aliases it ("speed_of_light", "c").
    result = _codata_result(name, name)
    if result is not None:
        return result
    canonical_module = name.lower().replace(" ", "_")
    for attr in (name, canonical_module):
        key = _SCIPY_ATTR_TO_CODATA.get(attr)
        if key is not None:
            return _codata_result(key, name)

    # Remaining module attributes are unit conversion factors in SI
    # (angstrom, calorie, day, ...): exact, no CODATA row.
    if hasattr(spc, canonical_module):
        attr_value = getattr(spc, canonical_module)
        if isinstance(attr_value, (int, float)):
            return ToolResult(
                value=float(attr_value),
                units="",
                source="scipy.constants (SI conversion factor)",
                inputs={"name": name},
                notes=f"The SI value of one {name}; scipy stores conversion factors without unit annotations.",
            )
    return None


_BODY_CONSTANT_PATTERNS = (r"^(gm|mass)[ _]of[ _](?:the[ _])?(?P<b>.+)$", r"^(?P<b>.+?)[ _](?P<w>gm|mass)$", r"^(?P<w>gm|mass|m)[ _](?P<b>.+)$")


def _lookup_body_constant(name: str) -> ToolResult | None:
    """'GM_earth', 'earth_mass', 'mass of the Moon', 'M_sun', 'gm ceres': the GM or mass of a body from the n-body catalogue (field/interface/body_data.py). SI units.

    Tried last, after the curated and CODATA constants, so it never shadows one.
    """
    import re
    from sigma_ground.field.interface import body_data
    text = name.strip().lower().replace("-", "_")
    for pat in _BODY_CONSTANT_PATTERNS:
        m = re.match(pat, text)
        if not m:
            continue
        what = (m.groupdict().get("w") or m.group(1)).lower()
        what = "mass" if what in ("m", "mass") else "gm"
        entry, _ = body_data.resolve(m.group("b"))
        if entry is None:
            continue
        d = body_data.describe(entry)
        if "gm_km3_s2" not in entry:                          # a known body with no published mass: say so, do not fall through to "constant not found"
            return ToolResult(value=None, source="body catalogue", notes=f"{entry['name']}: " + " ".join(d["notes"]), inputs={"name": name, "requested_name": name, "body": entry["name"]})
        gm = entry["gm_km3_s2"]
        src = d["sources"].get("gm_km3_s2", "")
        if what == "gm":
            return ToolResult(value=gm * 1e9, units="m^3/s^2", source=f"{src} (body catalogue: {entry['name']})", provenance_tag="VERIFIED",
                              notes=f"GM of {entry['name']} = {gm} km^3/s^2, exact to its published digits; it does not depend on G. " + " ".join(d["notes"]),
                              inputs={"name": name, "requested_name": name, "resolved_via": "body catalogue", "body": entry["name"]})
        return ToolResult(value=d["mass_kg"]["value"], units="kg", source=f"{src} (body catalogue: {entry['name']}), divided by G", uncertainty=d["mass_kg"]["uncertainty"],
                          provenance_tag="DERIVED",
                          notes=f"mass = GM / G; the GM ({gm} km^3/s^2) is exact to its published digits, the uncertainty is G's (relative 2.2e-5). Ask for GM_{m.group('b')} for the exact quantity. " + " ".join(d["notes"]),
                          inputs={"name": name, "requested_name": name, "resolved_via": "body catalogue", "body": entry["name"]})
    return None


def lookup_constant(name: str) -> ToolResult:
    """Look up a physical constant by name.

    Resolution order: alias map -> sigma_ground curated -> scipy.constants CODATA.

    Accepts symbol ("G", "c", "ℏ", "α"), informal name ("Newton's
    constant", "speed of light", "h-bar", "fine structure"), and
    canonical name ("gravitational_constant"). See aliases.py for the
    full map.

    Parameters
    ----------
    name : str
        Constant name in any of the accepted forms above.

    Returns
    -------
    ToolResult with `value`, `units`, `source`, and (when curated)
    `provenance_tag`. `inputs` echoes the requested name and the
    canonical name that was actually resolved.
    """
    from sigma_ground.mcp.tools.aliases import CONSTANT_ALIASES, resolve
    canonical = resolve(name, CONSTANT_ALIASES)
    # Try original name first (covers exact matches like 'speed_of_light'
    # that aren't in the alias map but already are canonical). Then try
    # the resolved canonical form.
    for query in (name, canonical):
        result = _lookup_sigma_ground(query)
        if result is not None:
            result.inputs["requested_name"] = name
            result.inputs["resolved_via"] = "alias" if query != name else "direct"
            return result
        result = _lookup_scipy(query)
        if result is not None:
            result.inputs["requested_name"] = name
            result.inputs["resolved_via"] = "alias" if query != name else "direct"
            return result
    result = _lookup_body_constant(name)
    if result is not None:
        return result
    return ToolResult(
        value=None,
        source="not found",
        notes=f"Constant '{name}' (resolved to '{canonical}') not found in "
               f"sigma_ground.field.constants or scipy.constants. Try a "
               f"more standard name or use list_constants() to see what's "
               f"available.",
        inputs={"name": name, "resolved_to": canonical},
    )


def list_constants(category: str | None = None,
                    contains: str | None = None,
                    limit: int = 50) -> ToolResult:
    """List available constants, optionally filtered.

    Parameters
    ----------
    category : str | None
        One of: 'sigma_ground', 'codata', 'all'. Default 'all'.
    contains : str | None
        Substring filter on the constant name (case-insensitive).
    limit : int
        Maximum number of results to return (default 50).

    Returns
    -------
    ToolResult whose `value` is a list of constant names available
    via lookup_constant().
    """
    names: list[str] = []
    if category in (None, "all", "sigma_ground"):
        try:
            import sigma_ground.field.constants as sgc
            sgc_names = [
                n for n in dir(sgc)
                if n.isupper() and not n.startswith("_")
                and not callable(getattr(sgc, n, None))
            ]
            names.extend(f"sg:{n}" for n in sgc_names)
        except ImportError:
            pass
    if category in (None, "all", "codata"):
        try:
            from scipy import constants as spc
            for n in spc.physical_constants:
                names.append(f"codata:{n}")
        except ImportError:
            pass

    if contains:
        c = contains.lower()
        names = [n for n in names if c in n.lower()]

    return ToolResult(
        value=names[:limit],
        source="sigma_ground.field.constants + scipy.constants",
        inputs={"category": category, "contains": contains, "limit": limit},
        notes=f"Truncated to {limit}. Total matching: {len(names)}. "
               f"Use lookup_constant(name) to get value+units+source.",
    )


# ── private helpers ───────────────────────────────────────────────────

_TAG_PATTERN = re.compile(
    r"\[(VERIFIED|DERIVED|EMPIRICAL-INPUT|SPECULATIVE-PENDING|REJECTED)\b"
)


def _extract_tag_for_constant(name: str) -> str | None:
    """Walk constants.py source to find the provenance tag for a name."""
    try:
        import sigma_ground.field.constants as sgc
        source_path = sgc.__file__
    except (ImportError, AttributeError):
        return None
    try:
        with open(source_path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return None
    target = f"{name} ="
    for i, line in enumerate(lines):
        if not line.lstrip().startswith(target.split()[0]):
            continue
        if target not in line:
            continue
        # Search up to 6 lines (the line itself + 5 above) for a tag.
        for j in range(max(0, i - 5), i + 1):
            m = _TAG_PATTERN.search(lines[j])
            if m:
                return m.group(1)
        return None
    return None


# Rough unit mapping for sigma_ground constants where the name implies units.
# Not exhaustive; the LLM can ask for clarification or pull from PROVENANCE.md.
_UNIT_HINTS: dict[str, str] = {
    "G":            "m^3 / (kg s^2)",
    "C":            "m/s",
    "HBAR":         "J s",
    "K_B":          "J/K",
    "ALPHA":        "dimensionless",
    "M_PLANCK_KG":  "kg",
    "L_PLANCK":     "m",
    "M_ELECTRON_KG":"kg",
    "M_PROTON_KG":  "kg",
    "M_NEUTRON_KG": "kg",
    "M_SUN_KG":     "kg",
    "AU":           "m",
    "L_SUN_W":      "W",
    "H0":           "1/s",
    "ETA":          "dimensionless",
    "XI":           "dimensionless",
    "SIGMA_CONV":   "dimensionless",
    "PHI":          "dimensionless",
    "BOHR_RADIUS":  "m",
    "MU_BOHR":      "J/T",
    "E_CHARGE":     "C",
    "EPS_0":        "F/m",
    "MU_0":         "H/m",
    "N_AVOGADRO":   "1/mol",
}


def _guess_units(name: str) -> str:
    """Best-effort unit lookup by name. Empty string if unknown."""
    if name in _UNIT_HINTS:
        return _UNIT_HINTS[name]
    # Suffix hints
    if name.endswith("_KG"): return "kg"
    if name.endswith("_M"):  return "m"
    if name.endswith("_S"):  return "s"
    if name.endswith("_J"):  return "J"
    if name.endswith("_W"):  return "W"
    if name.endswith("_K"):  return "K"
    if name.endswith("_C"):  return "C"
    if name.endswith("_HZ"): return "Hz"
    if name.endswith("_MEV"): return "MeV"
    if name.endswith("_EV"):  return "eV"
    if name.endswith("_FM"):  return "fm"
    if name.endswith("_PCT"): return "percent"
    return ""
