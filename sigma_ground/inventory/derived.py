"""Per-body parameters DERIVED from inventory composition data.

Aaron's perfect-library principle: minimize the per-body free inputs of
the physics engines. For each parameter that the n-body engine currently
reads from external tables (J2, J3, J4, Love number k2, moment of inertia
factor C/MR²), we want a derivation pipeline that computes it from
composition + rotation + measured radius/mass.

This module implements that pipeline. Current status:

  derive_moi_factor()           IMPLEMENTED  -- moment of inertia C/MR²
  derive_j2_hydrostatic()       SCAFFOLD     -- via Clairaut + (2/3)f - q/3
  derive_j2_darwin_radau()      SCAFFOLD     -- closed-form approximation
  derive_love_number_k2()       SCAFFOLD     -- via Radau equation + elastic moduli
  derive_j3_via_asymmetry()     NOT STARTED  -- needs north-south density asymmetry
  derive_j4_extended_clairaut() NOT STARTED  -- second-order Clairaut theory

Each function takes inputs that are themselves either:
  - measured externally (mass, radius, rotation rate)
  - in sigma_ground/inventory (composition profiles, material densities)
  - derived from upstream values in this module

The shell-integration core is general -- once a body has a layered density
profile in `inventory/samples/`, ANY zonal harmonic falls out of the same
machinery applied with the right polynomial.

References for future implementation:
  Clairaut 1743 / Murray & Dermott 1999 "Solar System Dynamics" ch. 4
  Darwin 1899 / Radau 1885 -- closed-form approximation for J₂
  Love 1911 -- elastic Love numbers k₂ from internal rigidity
  Wahr 1981 -- nutation theory connecting C/MR² to observation
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_INV = Path(__file__).parent
_MATERIALS_JSON = _INV / "data" / "materials.json"
_SAMPLES_DIR    = _INV / "samples"


# ── Material density lookup ─────────────────────────────────────────────

def _load_material_densities() -> dict[str, float]:
    """name -> density (kg/m³) from materials.json. Only materials with
    a numeric `density` field are returned."""
    return {
        m["name"]: m["density"]
        for m in json.loads(_MATERIALS_JSON.read_text())
        if isinstance(m.get("density"), (int, float))
    }


def _layer_density(layer: dict, materials_db: dict[str, float]) -> float:
    """Effective density (kg/m³) of a layer from material ratios.

    layer["materials"] = [{"material": name, "ratio": fraction}, ...]
    """
    total = 0.0
    for entry in layer["materials"]:
        name  = entry["material"]
        ratio = entry["ratio"]
        rho   = materials_db.get(name)
        if rho is None:
            raise KeyError(
                f"material {name!r} has no density in materials.json"
            )
        total += ratio * rho
    return total


# ── Result types ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ShellContribution:
    """One spherical shell's contribution to the planet integration."""
    r_in_m:    float
    r_out_m:   float
    rho_kgm3:  float
    mass_kg:   float
    moi_kgm2:  float   # moment of inertia of this shell about any diameter


@dataclass(frozen=True)
class DerivedBodyParameters:
    """All derivable per-body parameters for a planet with a known layer
    composition. Fields are None when their derivation is not yet
    implemented in this module.
    """
    planet_radius_m:  float
    total_mass_kg:    float
    moi_C_kgm2:       float
    c_over_mr2:       float
    # Future fields (will populate as derivations land):
    flattening_hyd:   float | None = None
    j2_hyd:           float | None = None
    j3:               float | None = None
    j4:               float | None = None
    love_k2:          float | None = None
    shells:           tuple[ShellContribution, ...] = ()
    scale_km_per_unit: float = 0.0


# ── The core derivation: C/MR² from layered composition ────────────────

def derive_moi_factor(
    sample_path:      Path | str,
    planet_radius_km: float,
    skip_materials:   Iterable[str] = ("Interstellar Medium (ISM)", "Air"),
    *,
    materials_db:     dict[str, float] | None = None,
) -> DerivedBodyParameters:
    """Derive C/MR² of a layered planet from inventory composition.

    For each spherical shell with uniform density ρ:
      M_shell = ρ × (4/3)π × (r_out³ − r_in³)
      I_shell = ρ × (8/15)π × (r_out⁵ − r_in⁵)       (about any diameter)

    Then C/MR² = ΣI_shell / (ΣM_shell × R²).

    Reference values for cross-checking:
      Uniform sphere C/MR² = 0.400
      Earth          C/MR² = 0.3307  (NASA Planetary Fact Sheet)
      Moon           C/MR² = 0.3931
      Mars           C/MR² = 0.3662
      Jupiter        C/MR² = 0.254

    Parameters
    ----------
    sample_path      : path to a layered-planet JSON in inventory/samples/,
                       or just the sample name (e.g. "earths_layers")
    planet_radius_km : the radius the planet's solid+ocean envelope should
                       scale to; the layer "thickness" units in the JSON
                       are dimensionless proportions that get scaled to fit
                       this radius
    skip_materials   : material names whose layers are dropped from the
                       integration (e.g. ISM, atmosphere -- low-density
                       layers that don't contribute meaningfully to MoI)
    materials_db     : optional density override map; if None, loaded from
                       inventory/data/materials.json

    Returns
    -------
    DerivedBodyParameters with c_over_mr2, total_mass_kg, moi_C_kgm2, and
    per-shell breakdown populated.
    """
    if isinstance(sample_path, str):
        if not sample_path.endswith(".json"):
            sample_path = sample_path + ".json"
        sample_path = _SAMPLES_DIR / sample_path

    if materials_db is None:
        materials_db = _load_material_densities()

    sample = json.loads(Path(sample_path).read_text())
    layers_in = sample["children"]
    skip_set  = set(skip_materials)

    # Filter to layers that count toward the solid+ocean envelope
    used = []
    for layer in layers_in:
        material_names = {m["material"] for m in layer["materials"]}
        if material_names & skip_set:
            continue
        used.append(layer)

    if not used:
        raise ValueError(
            f"no usable layers in {sample_path} after filtering "
            f"{skip_set}"
        )

    # Scale layer thicknesses so the total radius equals planet_radius_km
    total_thickness_units = sum(layer["thickness"] for layer in used)
    R_m       = planet_radius_km * 1000.0
    scale_m   = R_m / total_thickness_units

    r_in     = 0.0
    total_M  = 0.0
    total_C  = 0.0
    shells   = []

    for layer in used:
        thickness_m = layer["thickness"] * scale_m
        r_out       = r_in + thickness_m
        rho         = _layer_density(layer, materials_db)

        V_shell = (4.0 / 3.0) * math.pi * (r_out ** 3 - r_in ** 3)
        M_shell = rho * V_shell
        # Moment of inertia of shell about a diameter:
        #   I = (8π/15) × ρ × (r_out⁵ − r_in⁵)
        I_shell = (8.0 / 15.0) * math.pi * rho * (r_out ** 5 - r_in ** 5)

        total_M += M_shell
        total_C += I_shell

        shells.append(ShellContribution(
            r_in_m=r_in, r_out_m=r_out, rho_kgm3=rho,
            mass_kg=M_shell, moi_kgm2=I_shell,
        ))
        r_in = r_out

    c_over_mr2 = total_C / (total_M * R_m * R_m)

    return DerivedBodyParameters(
        planet_radius_m   = R_m,
        total_mass_kg     = total_M,
        moi_C_kgm2        = total_C,
        c_over_mr2        = c_over_mr2,
        shells            = tuple(shells),
        scale_km_per_unit = scale_m / 1000.0,
    )


# ── Future-work scaffolds ───────────────────────────────────────────────



