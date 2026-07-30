"""Name → material PROPERTIES resolver (the materials half of the handshake).

A user (or Deckard, emitting a part's constitutive-material name) names a
material; this returns a bundle of provenance-tagged properties so no layer
ever invents material data. The rule: prefer the DERIVED first-principles value
(surface/mechanical/stress/optics/fluid), fall back to a CITED reference value
(inventory/data/materials_engineering.json), and only then a flagged estimate —
never a fabricated number.

Two real external material-property databases are wired in ahead of that
chain: NASA TPSX (tier 0a, real per-material citations, see tpsx_lookup.py
— highest priority, only fires for its curated ~13-material alias set) and
Materials Project (tier 2b, cached DFT elastic moduli — deliberately placed
BELOW mechanical.py's own measured-Poisson-anchored E/G/K, not above, after
live verification (2026-07-29) found MP's automatic lowest-energy-polymorph
selection returns physically unrepresentative elastic tensors for several
of our metals, e.g. aluminum +68% vs textbook E, gold -53% — see the 2b)
block's own comment for the full story and the numbers).

Tiering: this lives in field/interface (tier 1) and imports only its tier-1
siblings + stdlib. It deliberately does NOT import deckard's ``Fact`` (tier 2);
that would be an upward import and break the layering contract. Instead it
returns ``MaterialFact`` — the same ``value/source/license/confidence/.estimated``
shape, so a caller can consume it identically or convert to a ``Fact``.

Property keys (those that apply to the material): ``density`` (kg/m³), ``E``,
``G``, ``K``, ``poisson_ratio`` (unitless) (Pa for E/G/K), ``K_Ic`` (Pa√m),
``sigma_UTS`` (Pa), ``is_ductile`` (bool), ``n_refractive``, ``reflectance``
(0..1), ``emissivity`` (0..1), ``viscosity`` (Pa·s), ``surface_tension``
(N/m), ``color`` ((r,g,b)), plus TPSX-sourced ``melt_temperature_k`` (K),
``thermal_conductivity_w_mk`` (W/m·K), ``specific_heat_j_kgk`` (J/kg·K).
Plus meta: ``name``, ``canonical``, ``material_class``, ``identified``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .surface import MATERIALS
from .mechanical import MECHANICAL_DATA, youngs_modulus, shear_modulus, bulk_modulus
from .stress import STRESS_DATA, fracture_toughness
from .optics import (
    MEASURED_NK, CAUCHY_COEFFICIENTS, ORGANIC_SPECTRA,
    metal_reflectance, metal_rgb, cauchy_n, dielectric_surface_reflectance,
    dielectric_color_rgb, organic_rgb, LAMBDA_G,
)
from .thermal_emission import THERMAL_EMISSION_MATERIALS, emissivity
from .tpsx_lookup import tpsx_property, TPSX_RESOLVE_KEYS

_ESTIMATED = "estimated"


@dataclass
class MaterialFact:
    """One resolved property value with its provenance (Fact-shaped, tier 1)."""
    value: Any
    source: str = _ESTIMATED
    license: str = ""
    confidence: float = 0.5

    @property
    def estimated(self) -> bool:
        return (not self.source) or self.source == _ESTIMATED

    def to_dict(self) -> dict:
        return {"value": self.value, "source": self.source,
                "license": self.license, "confidence": self.confidence}


# ── Aliases → canonical token (covering the physics-module keys) ─────────────
# Canonical tokens use underscores to match the physics tables. Anything not
# here falls through to norm.replace(' ', '_') and then the JSON alias map.
_ALIAS = {
    # glasses
    "glass": "glass", "window glass": "glass", "plate glass": "glass",
    "soda lime glass": "glass_soda_lime", "soda-lime glass": "glass_soda_lime",
    "borosilicate": "glass_borosilicate", "borosilicate glass": "glass_borosilicate",
    "pyrex": "glass_borosilicate", "bk7": "glass_borosilicate",
    "tempered glass": "glass_tempered", "toughened glass": "glass_tempered",
    "lead glass": "glass_lead", "crystal glass": "glass_lead", "lead crystal": "glass_lead",
    # liquids / metals
    "mercury": "mercury", "quicksilver": "mercury", "hydrargyrum": "mercury",
    "water": "water", "h2o": "water", "sea water": "seawater",
    "alcohol": "ethanol", "glycerine": "glycerol", "glycerin": "glycerol",
    "steel": "steel_mild", "mild steel": "steel_mild", "carbon steel": "steel_mild",
    "aluminium": "aluminum", "depleted uranium": "depleted_uranium", "uranium": "depleted_uranium",
    # solids in the physics tables
    "ice": "water_ice", "water ice": "water_ice", "oak": "wood_oak", "wood": "wood_oak",
    "carbon fiber": "carbon_fiber", "carbon fibre": "carbon_fiber", "cfrp": "carbon_fiber",
    "alumina": "ceramic_alumina",
    # generic category names (e.g. the ones ShapeNetSem/Deckard emit) → a
    # representative member so they resolve to a CITED density rather than an
    # estimate. "wood"→oak above; the rest here.
    "metal": "steel_mild", "metallic": "steel_mild", "steel alloy": "steel_mild",
    "plastic": "plastic_abs", "polymer": "plastic_abs",
    "fabric": "cotton", "textile": "cotton", "cloth": "cotton", "carpet": "wool",
    "foam": "foam_polyurethane",
}

# Glass/solid → the dielectric optics key that carries its refractive index.
_CAUCHY_OVERRIDE = {
    "glass": "crown_glass", "glass_soda_lime": "crown_glass", "glass_tempered": "crown_glass",
    "glass_borosilicate": "borosilicate", "glass_lead": "flint_glass",
}

# Brittle solids (everything else metallic/polymeric defaults ductile). Liquids
# get is_ductile omitted (the concept doesn't apply).
_BRITTLE = {
    "glass", "glass_soda_lime", "glass_borosilicate", "glass_tempered", "glass_lead",
    "silicon", "ceramic_alumina", "concrete", "granite", "water_ice", "carbon_fiber",
}
_LIQUID_CLASS_NAMES = {"water", "seawater", "ethanol", "glycerol", "mercury"}


# ── cited engineering catalog (the supplement) ──────────────────────────────
_JSON_PATH = Path(__file__).resolve().parents[2] / "inventory" / "data" / "materials_engineering.json"


def _load_catalog():
    try:
        doc = json.loads(_JSON_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, {}, {}
    mats = doc.get("materials", {})
    sources = doc.get("_sources", {})
    lic = doc.get("_license", "")
    alias = {}
    for key, rec in mats.items():
        alias[key] = key
        for a in rec.get("aliases", []):
            alias[_normalize(a)] = key
            alias[a] = key
    return mats, alias, {"_sources": sources, "_license": lic}


def _normalize(name: str) -> str:
    s = (name or "").strip().lower()
    for ch in ("-", "_", "/"):
        s = s.replace(ch, " ")
    return " ".join(s.split())


_CATALOG, _JSON_ALIAS, _CAT_META = _load_catalog()

# ── Materials Project cache (DFT-computed elastic moduli, cited) ───────────
_MP_JSON_PATH = Path(__file__).resolve().parents[2] / "inventory" / "data" / "materials_project_cache.json"


def _load_mp_cache() -> dict:
    try:
        doc = json.loads(_MP_JSON_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    doc.pop("_meta", None)
    return doc


_MP_CACHE = _load_mp_cache()


def _mp_elastic_pa(entry: dict) -> dict:
    """K, G in Pa from the cached GPa values; E is DERIVED via the isotropic
    identity E = 9KG/(3K+G) rather than trusting entry['young_modulus'] --
    that field was empirically None in every live MP response checked this
    session (see materials_project_client.py's own docstring caveat), so
    deriving E from K+G (both consistently populated) is the path that
    actually fires in practice."""
    K_gpa, G_gpa = entry.get("bulk_modulus_vrh"), entry.get("shear_modulus_vrh")
    out: dict[str, float] = {}
    if K_gpa is not None and K_gpa > 0:
        out["K"] = K_gpa * 1e9
    if G_gpa is not None and G_gpa > 0:
        out["G"] = G_gpa * 1e9
    if "K" in out and "G" in out and (3 * out["K"] + out["G"]) > 0:
        out["E"] = 9.0 * out["K"] * out["G"] / (3.0 * out["K"] + out["G"])
    return out


def _cat_source(code: str) -> str:
    return _CAT_META.get("_sources", {}).get(code, code)


def _fact(value, source=_ESTIMATED, license="", confidence=0.5):
    return MaterialFact(value, source, license, confidence)


def _add(props, key, value, source, license="", confidence=0.5):
    if value is not None and key not in props:
        props[key] = _fact(value, source, license, confidence)


# ─────────────────────────────────────────────────────────────────────────────
def material_profile(name: str) -> dict:
    """Resolve a material NAME to a dict of provenance-tagged properties.

    Returns ``{property: MaterialFact, ..., "name", "canonical",
    "material_class", "identified"}``. ``identified`` is False when nothing but
    a flagged estimate could be produced.
    """
    norm = _normalize(name)
    canon = _ALIAS.get(norm) or norm.replace(" ", "_")
    props: dict[str, Any] = {}
    mat_class = None

    _CRYSTAL_LIC = "CRC Handbook (crystal density), facts cited by reference"

    # 0a) NASA TPSX -- real measured/predicted materials data (tier 1, the
    # highest-priority tier). Only fires for the curated ~13-material alias
    # set in tpsx_lookup.py -- tpsx_property() returns None for every other
    # canon token, so this section is a pure no-op for every material NOT
    # in that table, and for every liquid/catalog-only material. Verified
    # 2026-07-29: for the 13 aliased metals this supplies density + thermal
    # properties only (its mechanical rows belong to aerospace TPS
    # composites not in this alias set) -- sigma_UTS/E/G/poisson_ratio stay
    # inert today but are kept wired for forward compatibility.
    for _rkey in TPSX_RESOLVE_KEYS:
        # NOTE: deliberately named _tpsx_fact, not _fact -- the module-level
        # _fact() helper (see below) is called later in this same function
        # (props["density"] = _fact(...), props["is_ductile"] = _fact(...)).
        # A same-named loop variable here would shadow it for the REST of
        # material_profile()'s scope (Python locals are whole-function, not
        # per-block) and break every one of those later call sites with a
        # "'NoneType'/'TPSXFact' object is not callable" TypeError -- this
        # was caught by a real test run against an unaliased material
        # ("flarbleglomp") during implementation, not just reasoned about.
        _tpsx_fact = tpsx_property(canon, _rkey)
        if _tpsx_fact is not None:
            _add(props, _rkey, _tpsx_fact.value, _tpsx_fact.source,
                _tpsx_fact.license, _tpsx_fact.confidence)

    # 1) Liquids — fluid.py (density, viscosity, surface tension, bulk modulus)
    from .fluid import KNOWN_LIQUIDS, liquid_properties
    if canon in KNOWN_LIQUIDS:
        mat_class = "liquid"
        lp = liquid_properties(canon)
        _add(props, "density", lp["density_kg_m3"], "fluid.KNOWN_LIQUIDS (measured)", "", 0.9)
        _add(props, "viscosity", lp["viscosity_pa_s"], "fluid.liquid_properties (Eyring/Arrhenius)", "", 0.8)
        _add(props, "surface_tension", lp["surface_tension_n_m"], "fluid.KNOWN_LIQUIDS (measured)", "", 0.85)
        _add(props, "K", lp["bulk_modulus_pa"], "fluid.KNOWN_LIQUIDS (measured)", "", 0.85)

    # 2) Solids — derived elastic moduli (mechanical.py) + crystal density
    if canon in MECHANICAL_DATA and canon in MATERIALS:
        mat_class = mat_class or "solid"
        _add(props, "E", youngs_modulus(canon), "mechanical.youngs_modulus (E=3K(1-2ν), derived)", "", 0.7)
        _add(props, "G", shear_modulus(canon), "mechanical.shear_modulus (derived)", "", 0.7)
        _add(props, "K", bulk_modulus(canon), "mechanical.bulk_modulus (harmonic approx, ±50%)", "", 0.6)
        _add(props, "density", MATERIALS[canon].get("density_kg_m3"), "surface.MATERIALS", _CRYSTAL_LIC, 0.9)
        _add(props, "poisson_ratio", MECHANICAL_DATA[canon].get("poisson_ratio"),
             "mechanical.MECHANICAL_DATA (poisson_ratio, measured)", "", 0.8)

    # 2b) Materials Project — cached DFT elastic moduli (E/G/K/poisson_ratio).
    # DELIBERATELY placed AFTER mechanical.py's own tier above, not before it
    # (a deviation from the original wiring design -- see note). _add()'s
    # "not already in props" guard means this only ever fills a genuine gap:
    # a CURATED_FORMULAS material with MP cache data but no MECHANICAL_DATA
    # entry. For all 14 materials currently in CURATED_FORMULAS that gap
    # doesn't exist (all 14 already have MECHANICAL_DATA coverage), so this
    # section is a verified no-op today -- same "inert but forward-compatible"
    # shape as the TPSX mechanical properties above.
    #
    # WHY the reorder: MP's elasticity search picks a formula's lowest-
    # energy_above_hull polymorph that has a POSITIVE VRH bulk+shear tensor
    # (materials_project_client.py::_pick_mechanically_stable). That filter
    # catches obviously-corrupted negative moduli (its own documented Zn
    # case) but not "positive but physically unrepresentative polymorph."
    # Live-verified 2026-07-29 against textbook E: aluminum's ground-state
    # entry (mp-134) has a NEGATIVE shear modulus in MP today and gets
    # skipped in favor of mp-998860, whose derived E (9KG/(3K+G)) comes out
    # ~118 GPa vs aluminum's real ~70 GPa (+68%); gold similarly falls
    # through to a polymorph giving E ~37 GPa vs real ~79 GPa (-53%); lead
    # +92%; titanium +36%. mechanical.py's own poisson-ratio-based E/G
    # (added above) is measured-Poisson-anchored and, while itself only a
    # ±25-30% approximation, is not subject to this per-material polymorph
    # lottery. Overriding it with MP's DFT numbers (as originally designed)
    # would have made material_profile()'s default E answer WORSE for
    # several of our highest-value metals. MP's numbers remain fully
    # available (cited, cached, dataset_minder-tracked) as the fallback tier
    # here and directly via resolve._MP_CACHE for any caller that wants the
    # raw DFT record regardless of this ordering.
    _mp = _MP_CACHE.get(canon)
    if _mp:
        for _key, _val in _mp_elastic_pa(_mp).items():
            _add(props, _key, _val, _mp.get("source", "Materials Project (DFT)"), "", 0.75)
        if _mp.get("poisson_ratio") is not None:
            _add(props, "poisson_ratio", _mp["poisson_ratio"],
                _mp.get("source", "Materials Project (DFT)"), "", 0.75)

    # 3) Fracture — stress.py (σ_UTS, K_Ic, ductility)
    if canon in STRESS_DATA:
        mat_class = mat_class or "solid"
        _add(props, "sigma_UTS", STRESS_DATA[canon].get("sigma_UTS_Pa"),
             "stress.STRESS_DATA (ASM/Callister/Hertzberg, measured)", "", 0.8)
        _add(props, "K_Ic", fracture_toughness(canon),
             "stress.fracture_toughness (measured K_Ic, else Griffith)", "", 0.75)

    # 4) Metals — measured optics (reflectance + color), Fresnel from n+ik
    if canon in MEASURED_NK:
        mat_class = mat_class or "metal"
        _add(props, "reflectance", metal_reflectance(canon, LAMBDA_G),
             "optics.metal_reflectance (Fresnel from Palik/JC72 n+ik)", "", 0.8)
        _add(props, "color", metal_rgb(canon),
             "optics.metal_rgb (Drude/measured n+ik → Fresnel RGB)", "", 0.7)

    # 5) Dielectrics — refractive index + Fresnel reflectance + color
    cauchy_key = _CAUCHY_OVERRIDE.get(canon, canon)
    if cauchy_key in CAUCHY_COEFFICIENTS:
        mat_class = mat_class or "dielectric"
        n = cauchy_n(cauchy_key, LAMBDA_G)
        _add(props, "n_refractive", n, "optics.cauchy_n (Cauchy fit, Schott/Palik)", "", 0.85)
        _add(props, "reflectance", dielectric_surface_reflectance(n),
             "optics.dielectric_surface_reflectance (Fresnel)", "", 0.8)
        _add(props, "color", dielectric_color_rgb(cauchy_key),
             "optics.dielectric_color_rgb (Fresnel + Beer-Lambert)", "", 0.6)

    # 6) Thermal emissivity — Kirchhoff ε = 1 − R (metals + blackbody table)
    if canon in THERMAL_EMISSION_MATERIALS:
        _add(props, "emissivity", emissivity(canon, LAMBDA_G),
             "thermal_emission.emissivity (Kirchhoff ε=1−R)", "", 0.75)

    # 7) Cited engineering catalog — fill everything still missing
    cat_key = _JSON_ALIAS.get(canon) or _JSON_ALIAS.get(norm)
    if cat_key:
        rec = _CATALOG[cat_key]
        mat_class = mat_class or rec.get("class")
        _cat_prop(props, rec, "density", "density_kg_m3")
        _cat_prop(props, rec, "E", "youngs_modulus_pa")
        _cat_prop(props, rec, "sigma_UTS", "sigma_UTS_pa")
        _cat_prop(props, rec, "K_Ic", "K_Ic_pa_sqrtm")
        _cat_prop(props, rec, "n_refractive", "n_refractive")
        _cat_prop(props, rec, "reflectance", "reflectance")
        _cat_prop(props, rec, "viscosity", "viscosity_pa_s")
        _cat_prop(props, rec, "surface_tension", "surface_tension_n_m")
        if "is_ductile" in rec and "is_ductile" not in props:
            props["is_ductile"] = _fact(bool(rec["is_ductile"]), "materials_engineering.json", "", 0.7)

    # 8) Cheap derived fallbacks (only when a real input exists)
    if "reflectance" in props and "emissivity" not in props:
        _add(props, "emissivity", 1.0 - props["reflectance"].value,
             "Kirchhoff ε=1−R (derived from resolved reflectance)", "", 0.6)
    if "reflectance" not in props and "n_refractive" in props:
        n = props["n_refractive"].value
        _add(props, "reflectance", ((n - 1.0) / (n + 1.0)) ** 2,
             "Fresnel R from resolved n (derived)", "", 0.6)

    # 9) Ductility for solids if still unset
    if "is_ductile" not in props and mat_class in ("solid", "metal", "dielectric"):
        props["is_ductile"] = _fact(canon not in _BRITTLE,
                                    "derived from material class (brittle-set)", "", 0.6)

    identified = "density" in props
    if not identified:
        # nothing real — return a flagged density estimate so callers degrade gracefully
        props["density"] = _fact(1000.0, _ESTIMATED, "", 0.2)
        mat_class = mat_class or "unknown"

    # report the catalog key as canonical when the match came from the catalog
    # (not a physics table) — e.g. "balsa" → "wood_balsa"
    physics_hit = (canon in MECHANICAL_DATA or canon in STRESS_DATA or canon in MEASURED_NK
                   or canon in KNOWN_LIQUIDS or cauchy_key in CAUCHY_COEFFICIENTS)
    props["name"] = name
    props["canonical"] = canon if physics_hit else (cat_key or canon)
    props["material_class"] = mat_class or "unknown"
    props["identified"] = identified
    return props


def _cat_prop(props, rec, prop_key, json_key):
    """Add a cited catalog value (with its real source) if not already resolved."""
    if prop_key in props or json_key not in rec:
        return
    cell = rec[json_key]
    if not isinstance(cell, dict):
        return
    props[prop_key] = _fact(cell.get("v"), _cat_source(cell.get("src", _ESTIMATED)),
                            _CAT_META.get("_license", ""), float(cell.get("c", 0.5)))


__all__ = ["MaterialFact", "material_profile"]
