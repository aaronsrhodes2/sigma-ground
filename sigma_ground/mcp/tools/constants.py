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

    return ToolResult(
        value=value,
        units=_guess_units(canonical),
        source="sigma_ground.field.constants",
        provenance_tag=tag,
        inputs={"name": name, "canonical": canonical},
    )


def _lookup_scipy(name: str) -> ToolResult | None:
    """Try scipy.constants. Returns None if not found."""
    try:
        from scipy import constants as spc
    except ImportError:
        return None

    # scipy.constants exposes both module-level attributes and a `value()`
    # function that queries the CODATA dictionary by name.
    canonical_module = name.lower().replace(" ", "_")
    if hasattr(spc, canonical_module):
        attr = getattr(spc, canonical_module)
        if isinstance(attr, (int, float)):
            return ToolResult(
                value=float(attr),
                units="",  # scipy module attrs are dimensional but unlabelled
                source="scipy.constants (CODATA)",
                inputs={"name": name},
                notes="Look up units from the CODATA reference; scipy stores "
                       "module attrs without unit annotations.",
            )

    # Try the CODATA dict via spc.value/unit/precision.
    try:
        value = spc.value(name)
        units = spc.unit(name)
        uncertainty = spc.precision(name) * value
        return ToolResult(
            value=value,
            units=units,
            source="scipy.constants (CODATA via value())",
            uncertainty=abs(uncertainty) if uncertainty != 0.0 else None,
            inputs={"name": name},
        )
    except (KeyError, AttributeError):
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
