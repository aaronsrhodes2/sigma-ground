"""Derive a planet's moment-of-inertia factor C/MR2 from a layered model.

For each spherical shell with uniform density rho:
  V_shell = (4/3)π × (r_out³ − r_in³)
  M_shell = rho × V_shell
  I_shell = (8π/15) × rho × (r_out⁵ − r_in⁵)     (about a diameter)

Total:
  M = Σ M_shell
  C = Σ I_shell                  (moment of inertia about any diameter --
                                   spherically symmetric body has 3 equal axes)
  C/MR2 = the dimensionless concentration factor

Uniform density sphere: C/MR2 = 2/5 = 0.400
Earth (measured):       C/MR2 = 0.3307  (iron-rich core concentrates mass inward)
Moon (measured):        C/MR2 = 0.3931  (nearly uniform)
Mars (measured):        C/MR2 = 0.3662
Jupiter (measured):     C/MR2 = 0.254

This script reads a layered-planet sample from sigma_ground/inventory/samples/,
looks up material densities from sigma_ground/inventory/data/materials.json,
and computes C/MR2. First demonstration that per-body parameters previously
read from NASA tables can be DERIVED from internal-composition data already
in the library.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

_INV = Path(__file__).parent.parent / "sigma_ground" / "inventory"
_MATERIALS_JSON = _INV / "data" / "materials.json"
_SAMPLES = _INV / "samples"


def load_material_densities() -> dict[str, float]:
    """Material name -> density (kg/m^3)."""
    return {
        m["name"]: m["density"]
        for m in json.loads(_MATERIALS_JSON.read_text())
        if m.get("density") is not None
    }


def layer_density(layer: dict, materials_db: dict[str, float]) -> float:
    """Compute layer's effective density (kg/m^3) from material ratios."""
    total = 0.0
    for entry in layer["materials"]:
        name  = entry["material"]
        ratio = entry["ratio"]
        rho   = materials_db.get(name)
        if rho is None:
            raise KeyError(f"material {name!r} has no density in materials.json")
        total += ratio * rho
    return total


def derive_moi_factor(
    planet_sample_path: Path,
    planet_radius_km:   float,
    skip_layer_names:   set[str] = frozenset({"Interstellar Medium (ISM)", "Air"}),
) -> dict:
    """Compute C/MR2 from a layered planet model.

    Parameters
    ----------
    planet_sample_path : path to JSON in sigma_ground/inventory/samples/
    planet_radius_km   : the radius the planet's solid+ocean envelope should
                          scale to. Atmosphere and ISM are skipped from the
                          shell integration (they're so low-density they
                          contribute nothing to MoI anyway).
    skip_layer_names   : material names to exclude from the integration

    Returns
    -------
    dict with keys: total_mass_kg, planet_radius_m, moi_C_kgm2, c_over_mr2,
    layers (list of per-shell contributions), and provenance metadata.
    """
    materials_db = load_material_densities()
    sample = json.loads(planet_sample_path.read_text())
    layers_in = sample["children"]

    # Identify which layers count toward the solid+ocean envelope.
    used = []
    for layer in layers_in:
        material_names = {m["material"] for m in layer["materials"]}
        if material_names & skip_layer_names:
            continue
        used.append(layer)

    # Scale layer thicknesses so the total radius equals planet_radius_km.
    total_thickness_units = sum(layer["thickness"] for layer in used)
    R_m       = planet_radius_km * 1000.0
    scale_m   = R_m / total_thickness_units

    r_in = 0.0
    total_M = 0.0
    total_C = 0.0
    shells: list[dict] = []
    for layer in used:
        thickness_m = layer["thickness"] * scale_m
        r_out       = r_in + thickness_m
        rho         = layer_density(layer, materials_db)

        # Volume of spherical shell
        V_shell = (4.0 / 3.0) * math.pi * (r_out**3 - r_in**3)
        M_shell = rho * V_shell

        # Moment of inertia of shell about a diameter
        I_shell = (8.0 / 15.0) * math.pi * rho * (r_out**5 - r_in**5)

        total_M += M_shell
        total_C += I_shell

        shells.append({
            "r_in_km":  r_in / 1000.0,
            "r_out_km": r_out / 1000.0,
            "rho_kgm3": rho,
            "mass_kg":  M_shell,
            "moi_kgm2": I_shell,
            "materials": layer["materials"],
        })
        r_in = r_out

    c_over_mr2 = total_C / (total_M * R_m * R_m)

    return {
        "planet_radius_m":  R_m,
        "planet_radius_km": planet_radius_km,
        "total_mass_kg":    total_M,
        "moi_C_kgm2":       total_C,
        "c_over_mr2":       c_over_mr2,
        "shells":           shells,
        "scale_km_per_unit": scale_m / 1000.0,
        "n_shells":         len(shells),
        "skipped_materials": list(skip_layer_names),
    }


def main():
    print("=" * 70)
    print(" Earth -- moment of inertia factor from layered composition")
    print("=" * 70)

    earth = derive_moi_factor(
        _SAMPLES / "earths_layers.json",
        planet_radius_km=6371.0,
    )
    print(f"\nScale factor: 1 thickness unit = {earth['scale_km_per_unit']:.2f} km")
    print(f"\n  {'r_in':>10s} {'r_out':>10s} {'rho':>10s} {'M_shell':>14s} {'I_shell':>14s}")
    print(f"  {'(km)':>10s} {'(km)':>10s} {'(kg/m^3)':>10s} {'(10^24 kg)':>14s} {'(10^38 kgm2)':>14s}")
    print(f"  {'-' * 65}")
    for s in earth["shells"]:
        print(f"  {s['r_in_km']:>10.0f} {s['r_out_km']:>10.0f} {s['rho_kgm3']:>10.1f}"
                f"  {s['mass_kg']/1e24:>12.3f}  {s['moi_kgm2']/1e38:>13.3f}")
    print(f"  {'-' * 65}")
    print(f"\n  Derived planet mass:        {earth['total_mass_kg']:.3e} kg")
    print(f"  Measured Earth mass:         5.972e+24 kg")
    print(f"  Ratio (derived/measured):   {earth['total_mass_kg']/5.972e24:.3f}")
    print(f"")
    print(f"  Derived C:                  {earth['moi_C_kgm2']:.3e} kg m^2")
    print(f"  Derived C/MR2:              {earth['c_over_mr2']:.4f}")
    print(f"  Measured Earth C/MR2:        0.3307")
    print(f"  Discrepancy:                {(earth['c_over_mr2']/0.3307 - 1)*100:+.1f}%")
    print(f"")
    print(f"  Uniform sphere C/MR2:        0.4000  (reference)")

    print(f"\nThis is a first concrete derivation of a per-body parameter from")
    print(f"sigma_ground.inventory composition data. The earths_layers.json sample")
    print(f"is a simplified 7-shell model -- a precise PREM-based derivation would")
    print(f"need finer radial discretization but the framework is now in place.")


if __name__ == "__main__":
    main()
