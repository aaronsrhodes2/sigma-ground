"""
Materials Project API client -- DFT-computed materials properties, cited.

Scope (read this before reaching for this module):

  The Materials Project (next-gen.materialsproject.org) is a database of
  ~154,000 computed materials, ~30,000+ of them "stable" (on or near the
  convex hull), built from high-throughput Density Functional Theory (DFT)
  calculations. It is NOT an experimental database and it is NOT a
  mechanical-testing database.

  WHAT IT COVERS (confirmed against the live API schema, 2026-07):
    - Elastic tensors / elastic moduli: bulk modulus, shear modulus, Young's
      modulus, Poisson's ratio, universal anisotropy, sound velocity,
      thermal conductivity, Debye temperature -- for ~6,000+ materials via
      the `/materials/elasticity/` endpoint (`mpr.materials.elasticity`).
    - Dielectric tensors: total, ionic, and electronic-only dielectric
      constants plus refractive index, via `/materials/dielectric/`
      (`mpr.materials.dielectric`) -- DFPT-computed, narrower coverage than
      elasticity (mostly insulators/semiconductors with a nonzero band gap;
      metals are largely absent because a metal's static dielectric
      response diverges).
    - Also present (not wired up by this module, but available the same
      way if a future gap needs it): piezoelectric tensors, phonon data,
      magnetism, surface energies, band structure/DOS, thermodynamic
      stability (formation energy, energy above hull), XAS spectra.

  WHAT IT DOES NOT COVER -- confirmed absent from the API schema:
    - Fatigue data: S-N curves, fatigue strength coefficients, Basquin
      exponents. NOT PRESENT. No endpoint, no field, anywhere in the
      schema. Fatigue life is a mechanical-testing (cyclic loading)
      measurement, not something DFT computes.
    - Creep data: creep activation energy, stress exponent, pre-exponential
      (Dorn power-law creep). NOT PRESENT, same reason -- creep is a
      thermally-activated, time-dependent mechanical-test measurement.
    - Paris-law fatigue-crack-growth constants (C, m in da/dN = C*(dK)^m).
      NOT PRESENT. This is exactly the shape of gap sigma-ground hit for
      zinc in `sigma_ground/field/interface/stress.py` (STRESS_FATIGUE_DATA
      entry for 'zinc' has C_paris/m_paris marked ESTIMATED, interpolated
      from lead/silver rather than independently measured) -- Materials
      Project cannot close that gap. A real fix needs a source like the
      NIST fatigue database, ASM Handbook Vol. 19 (Fatigue and Fracture),
      or a materials-testing standard (ASTM E466/E647), none of which
      have a comparable free structured API as far as this survey found.

  eps_inf caveat (dielectric.py convention): this codebase's `eps_inf`
  (see DIELECTRIC_DATA in dielectric.py) is typically a MEASURED
  high-frequency/optical dielectric constant (often derived from a
  measured refractive index via the Maxwell relation eps_inf = n^2).
  Materials Project's `e_electronic` field is a DIFFERENT (related but not
  identical) quantity: a DFPT, zero-frequency, ionic-positions-frozen
  electronic response. The two often agree to within the DFT elastic/
  dielectric error bars for simple materials, but they are not defined the
  same way. Treat `e_electronic` as a citable cross-check, not a drop-in
  replacement, before writing it into DIELECTRIC_DATA.

Authentication:

  Uses the official `mp-api` Python client (`MPRester` class), which
  handles the HTTP transport and auth header internally -- this module
  never hand-rolls a raw HTTP request against the Materials Project REST
  API, precisely so it doesn't have to guess at the wire format.

  The API key is read from `MATERIALS_PROJECT_API_KEY` in the dev-root
  `.env` (`D:\\Aaron\\development\\.env`), following the same loading
  pattern as the existing benchmark runners
  (`sigma_ground/mcp/benchmark/__init__.py::load_env_from_dev_root`).
  This module intentionally does NOT import from `sigma_ground.mcp` (the
  `field/` layer doesn't depend on the `mcp/` layer anywhere else in this
  codebase) -- the loader below is a self-contained copy of the same
  walk-up-and-load logic.

  Registration is NOT something this module or sigma-ground can do for
  you: get a free API key at https://next-gen.materialsproject.org
  (register -> API dashboard -> copy key), then add it to the dev-root
  `.env`. See `elastic_properties`/`dielectric_properties` for the exact
  error message and instructions if the key is missing.

Status: this module has NOT been exercised against the live API (no key
was available while writing it -- see PLATINUM_RULES.md sec8: registering
an account is the Captain's action, not an agent's). Every code path that
touches the network is covered only by offline/mocked tests
(test_materials_project_client.py). Field names for the elasticity and
dielectric documents were confirmed against the live OpenAPI schema at
https://api.materialsproject.org/openapi.json, but exact units per field
(e.g. GPa vs Pa for young_modulus) were NOT independently re-verified
against a live response and should be spot-checked the first time this
runs for real.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

MATERIALS_PROJECT_API_KEY_ENV = "MATERIALS_PROJECT_API_KEY"
_REGISTER_URL = "https://next-gen.materialsproject.org"
_DEV_ROOT_ENV_HINT = r"D:\Aaron\development\.env"


# ── Errors ────────────────────────────────────────────────────────────
# Every failure mode this module can hit gets its own exception type and
# an honest, actionable message -- never a bare crash, matching the
# "not in lookup, available: [...]" refusal style used throughout
# sigma_ground/mcp/tools/ (e.g. atomic.py::first_ionization_energy).

class MaterialsProjectError(RuntimeError):
    """Base class for all errors raised by this client."""


class MaterialsProjectNotConfigured(MaterialsProjectError):
    """Raised when MATERIALS_PROJECT_API_KEY is not set anywhere findable."""


class MaterialsProjectNotFound(MaterialsProjectError):
    """Raised when the query is well-formed but no matching document exists."""


# ── Dev-root .env loading ────────────────────────────────────────────
# Mirrors sigma_ground/mcp/benchmark/__init__.py::load_env_from_dev_root.
# Kept as a local copy (not an import) to avoid field/ depending on mcp/,
# a boundary the rest of this codebase also respects.

def _load_dev_root_env() -> Path | None:
    """Find and load the dev-root .env, walking up from CWD and from this
    file's own location. Existing env vars are NOT overridden -- shell
    values win. Returns the path loaded (shallowest match), or None."""
    candidates: list[Path] = []
    here = Path.cwd().resolve()
    for ancestor in [here, *here.parents]:
        candidates.append(ancestor / ".env")
    pkg_root = Path(__file__).resolve()
    for ancestor in pkg_root.parents:
        candidates.append(ancestor / ".env")

    seen: set[Path] = set()
    existing: list[Path] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            existing.append(path)
    existing.sort(key=lambda p: len(p.parts))  # shallowest (dev-root) first

    if not existing:
        return None
    for path in existing:
        _load_env_file(path)
    return existing[0]


def _load_env_file(path: Path) -> None:
    """Load one .env file. Prefer python-dotenv; fall back to a minimal
    manual KEY=VALUE parser if python-dotenv isn't installed."""
    try:
        from dotenv import load_dotenv
        load_dotenv(str(path), override=False)
        return
    except ImportError:
        pass
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def _get_api_key() -> str | None:
    _load_dev_root_env()
    return os.environ.get(MATERIALS_PROJECT_API_KEY_ENV) or None


def is_configured() -> bool:
    """True if MATERIALS_PROJECT_API_KEY is set (after loading the
    dev-root .env). Does not validate the key -- only that one is present."""
    return _get_api_key() is not None


def _not_configured_error() -> MaterialsProjectNotConfigured:
    return MaterialsProjectNotConfigured(
        f"{MATERIALS_PROJECT_API_KEY_ENV} is not set -- sigma-ground cannot "
        "query the Materials Project API until you provide a key. This is "
        "expected, not a bug: sigma-ground never registers accounts or "
        "generates API keys on your behalf.\n\n"
        "To fix this:\n"
        f"  1. Register a free account at {_REGISTER_URL}\n"
        "  2. Copy your API key from the account dashboard\n"
        f"  3. Add this line to {_DEV_ROOT_ENV_HINT}:\n"
        f"       {MATERIALS_PROJECT_API_KEY_ENV}=<your key>\n"
        "  4. Re-run the query."
    )


# ── mp-api import (optional dependency) ──────────────────────────────

def _import_mprester():
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise MaterialsProjectError(
            "The 'mp-api' package is not installed. It is an optional "
            "dependency of sigma-ground -- install it with:\n"
            "    pip install sigma-ground[materials_project]\n"
            "or directly:\n"
            "    pip install mp-api\n"
            "The core sigma_ground library has zero required dependencies; "
            "this client is opt-in, same as the [mcp] and [benchmark] "
            "extras."
        ) from exc
    return MPRester


# ── Response parsing helpers ─────────────────────────────────────────
# mp-api returns pydantic-model documents (attribute access) from a real
# query. Offline tests hand-construct plain dicts shaped like those
# documents. Both are supported so the parsing logic is exercised the
# same way whether the object came from the live API or a test fixture.

def _get_field(doc: Any, *names: str, default: Any = None) -> Any:
    """Fetch the first present, non-None field from `doc` by trying each
    name in `names` in order. Works for dict-like or attribute-like docs."""
    for name in names:
        if isinstance(doc, dict):
            if name in doc and doc[name] is not None:
                return doc[name]
        elif hasattr(doc, name):
            val = getattr(doc, name)
            if val is not None:
                return val
    return default


# ── Elastic / mechanical properties ──────────────────────────────────

def elastic_properties(formula_or_mp_id: str) -> dict[str, Any]:
    """Query DFT-computed elastic/mechanical properties for a material.

    Parameters
    ----------
    formula_or_mp_id : str
        Either a chemical formula (e.g. "Zn", "Fe2O3") or a Materials
        Project ID (e.g. "mp-79"). A leading "mp-" (case-insensitive)
        is treated as an ID; anything else is treated as a formula.

    Returns
    -------
    dict shaped consistently with this codebase's curated tables (see
    `poisson_ratio` in mechanical.py's MECHANICAL_DATA, which this dict
    reuses the same key name for) --

        {
            "name": "Zn",
            "material_id": "mp-79",
            "formula_pretty": "Zn",
            "bulk_modulus_vrh": <float or None>,
            "shear_modulus_vrh": <float or None>,
            "young_modulus": <float or None>,
            "poisson_ratio": <float or None>,        # homogeneous_poisson
            "universal_anisotropy": <float or None>,
            "source": "Materials Project (DFT-computed elastic tensor, mp-79)",
            "provenance_tag": "DFT-COMPUTED",
            "notes": "...",
        }

    Does NOT return fatigue, creep, or Paris-law constants -- see the
    module docstring; Materials Project has no such data.

    Raises
    ------
    MaterialsProjectNotConfigured
        MATERIALS_PROJECT_API_KEY is not set in the dev-root .env.
    MaterialsProjectError
        mp-api isn't installed, or the API call itself failed (network,
        auth, malformed query).
    MaterialsProjectNotFound
        The query was well-formed but no elasticity document exists for
        this material (common -- only ~6,000 of 30,000+ stable materials
        have computed elastic tensors).
    """
    query = _validate_query(formula_or_mp_id)
    api_key = _get_api_key()
    if not api_key:
        raise _not_configured_error()

    MPRester = _import_mprester()
    is_mp_id = query.lower().startswith("mp-")

    try:
        with MPRester(api_key) as mpr:
            if is_mp_id:
                docs = mpr.materials.elasticity.search(material_ids=[query])
            else:
                docs = mpr.materials.elasticity.search(formula=query)
    except MaterialsProjectError:
        raise
    except Exception as exc:  # mp-api's own exception hierarchy varies by version
        raise MaterialsProjectError(
            f"Materials Project API call failed for '{query}': {exc}"
        ) from exc

    if not docs:
        raise MaterialsProjectNotFound(
            f"No elasticity data found in Materials Project for '{query}'. "
            f"Checked {'material_id' if is_mp_id else 'formula'}: {query}. "
            "Materials Project has computed elastic tensors for ~6,000 of "
            "its 30,000+ stable materials -- not every material has this "
            "data. Try a specific Materials Project ID (e.g. 'mp-79') if "
            "you have one, or search next-gen.materialsproject.org "
            "directly to confirm coverage before assuming this is a bug."
        )

    return _parse_elasticity_doc(docs[0], query)


def _parse_elasticity_doc(doc: Any, query: str) -> dict[str, Any]:
    material_id = _get_field(doc, "material_id", "task_id")
    formula = _get_field(doc, "formula_pretty", "pretty_formula", "formula")

    bulk = _get_field(doc, "bulk_modulus")
    shear = _get_field(doc, "shear_modulus")
    bulk_vrh = _get_field(bulk, "vrh") if bulk is not None else None
    shear_vrh = _get_field(shear, "vrh") if shear is not None else None

    cite_id = material_id or query
    return {
        "name": formula,
        "material_id": str(material_id) if material_id else None,
        "formula_pretty": formula,
        "bulk_modulus_vrh": bulk_vrh,
        "shear_modulus_vrh": shear_vrh,
        "young_modulus": _get_field(doc, "young_modulus"),
        "poisson_ratio": _get_field(doc, "homogeneous_poisson"),
        "universal_anisotropy": _get_field(doc, "universal_anisotropy"),
        "source": f"Materials Project (DFT-computed elastic tensor, {cite_id})",
        "provenance_tag": "DFT-COMPUTED",
        "notes": (
            "Elastic moduli from DFT (Voigt-Reuss-Hill averaged), NOT "
            "experimentally measured. Units as returned by the Materials "
            "Project API (bulk/shear modulus are documented in GPa; "
            "young_modulus's exact unit was not independently re-verified "
            "against a live response -- spot-check before use, see module "
            "docstring). Does NOT include fatigue, creep, or Paris-law "
            "fatigue-crack-growth constants -- Materials Project has no "
            "such data; those remain MEASURED/ESTIMATED entries in "
            "sigma_ground/field/interface/stress.py."
        ),
    }


# ── Dielectric properties ────────────────────────────────────────────

def dielectric_properties(formula_or_mp_id: str) -> dict[str, Any]:
    """Query DFPT-computed dielectric properties for a material.

    Parameters
    ----------
    formula_or_mp_id : str
        Chemical formula (e.g. "Si") or Materials Project ID (e.g. "mp-149").

    Returns
    -------
    dict:
        {
            "name": "Si",
            "material_id": "mp-149",
            "formula_pretty": "Si",
            "eps_electronic": <float or None>,   # e_electronic (DFPT)
            "eps_ionic": <float or None>,         # e_ionic
            "eps_total": <float or None>,         # e_total = electronic + ionic
            "refractive_index_n": <float or None>,
            "source": "Materials Project (DFPT-computed dielectric tensor, mp-149)",
            "provenance_tag": "DFT-COMPUTED",
            "notes": "...",   # includes the eps_inf caveat, see module docstring
        }

    IMPORTANT: `eps_electronic` is related to but not identical to this
    codebase's `eps_inf` convention (dielectric.py) -- see the module
    docstring's "eps_inf caveat" section before treating this as a
    drop-in replacement for a DIELECTRIC_DATA entry.

    Raises
    ------
    MaterialsProjectNotConfigured, MaterialsProjectError,
    MaterialsProjectNotFound
        Same semantics as `elastic_properties`.
    """
    query = _validate_query(formula_or_mp_id)
    api_key = _get_api_key()
    if not api_key:
        raise _not_configured_error()

    MPRester = _import_mprester()
    is_mp_id = query.lower().startswith("mp-")

    try:
        with MPRester(api_key) as mpr:
            if is_mp_id:
                docs = mpr.materials.dielectric.search(material_ids=[query])
            else:
                docs = mpr.materials.dielectric.search(formula=query)
    except MaterialsProjectError:
        raise
    except Exception as exc:
        raise MaterialsProjectError(
            f"Materials Project API call failed for '{query}': {exc}"
        ) from exc

    if not docs:
        raise MaterialsProjectNotFound(
            f"No dielectric data found in Materials Project for '{query}'. "
            f"Checked {'material_id' if is_mp_id else 'formula'}: {query}. "
            "The dielectric endpoint is DFPT-computed and has narrower "
            "coverage than elasticity -- mostly insulators/semiconductors "
            "with a nonzero band gap; metals are usually absent because a "
            "metal's static dielectric response diverges. Try a specific "
            "Materials Project ID if you have one, or confirm coverage at "
            "next-gen.materialsproject.org before assuming this is a bug."
        )

    return _parse_dielectric_doc(docs[0], query)


def _parse_dielectric_doc(doc: Any, query: str) -> dict[str, Any]:
    material_id = _get_field(doc, "material_id", "task_id")
    formula = _get_field(doc, "formula_pretty", "pretty_formula", "formula")
    cite_id = material_id or query

    return {
        "name": formula,
        "material_id": str(material_id) if material_id else None,
        "formula_pretty": formula,
        "eps_electronic": _get_field(doc, "e_electronic"),
        "eps_ionic": _get_field(doc, "e_ionic"),
        "eps_total": _get_field(doc, "e_total"),
        "refractive_index_n": _get_field(doc, "n"),
        "source": f"Materials Project (DFPT-computed dielectric tensor, {cite_id})",
        "provenance_tag": "DFT-COMPUTED",
        "notes": (
            "eps_electronic is the DFPT electronic-only (ionic-clamped) "
            "dielectric response -- the closest Materials Project analog "
            "to this codebase's eps_inf convention (dielectric.py), but "
            "it is NOT the same quantity: eps_inf here is typically "
            "MEASURED (often via refractive index, eps_inf = n^2), while "
            "eps_electronic is a zero-frequency DFT electronic response "
            "with ions frozen at equilibrium. Cross-check, don't "
            "silently substitute, before writing into DIELECTRIC_DATA. "
            "eps_total (electronic + ionic) is the static dielectric "
            "constant. Units/tensor-averaging as returned by the API -- "
            "not independently re-verified against a live response, see "
            "module docstring."
        ),
    }


def _validate_query(formula_or_mp_id: str) -> str:
    if not formula_or_mp_id or not str(formula_or_mp_id).strip():
        raise ValueError("formula_or_mp_id must be a non-empty string")
    return str(formula_or_mp_id).strip()


__all__ = [
    "MaterialsProjectError",
    "MaterialsProjectNotConfigured",
    "MaterialsProjectNotFound",
    "MATERIALS_PROJECT_API_KEY_ENV",
    "is_configured",
    "elastic_properties",
    "dielectric_properties",
]
