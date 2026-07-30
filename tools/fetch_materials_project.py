"""Fetch + cache Materials Project DFT elastic moduli for our curated
metals (the Captain's second real materials-property source, alongside
NASA TPSX).

Materials Project only covers single, ordered, crystalline compounds/
elements -- no alloys, no amorphous glasses, no organic composites, no
mineral aggregates. CURATED_FORMULAS below is scoped to sigma-ground's
~25-material vocabulary (field/interface/surface.py::MATERIALS) restricted
to what MP can actually answer: pure elements + one clean binary ceramic
(Al2O3). Everything else in MATERIALS (steel_mild, glass, concrete,
granite, carbon_fiber, kevlar, rubber, plastic_abs, water_ice, wood_oak,
bone) is an alloy/composite/organic/amorphous/aggregate and is deliberately
NOT queried.

Calls the EXISTING, already-verified
field.interface.materials_project_client.elastic_properties(formula) --
never reimplements query logic, matching fetch_tpsx.py's own convention.

Gracefully no-ops (prints a message, exits 0) when
MATERIALS_PROJECT_API_KEY is not configured -- a missing key on some future
machine must never fail this tool or the dataset_minder cron that calls it.

Output: sigma_ground/inventory/data/materials_project_cache.json -- same
_meta + per-material-entry shape as tpsx_materials.json.

Run:  python tools/fetch_materials_project.py
"""
from __future__ import annotations

import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sigma_ground.field.interface.materials_project_client import (  # noqa: E402
    elastic_properties, is_configured, MaterialsProjectError, MaterialsProjectNotFound,
)

_OUT = os.path.join(os.path.dirname(__file__), "..", "sigma_ground",
                    "inventory", "data", "materials_project_cache.json")

# canon (surface.py::MATERIALS key) -> chemical formula. Pure elements + one
# clean ceramic compound only -- see module docstring for exclusions.
CURATED_FORMULAS = {
    "aluminum": "Al", "copper": "Cu", "gold": "Au", "iron": "Fe",
    "nickel": "Ni", "silicon": "Si", "silver": "Ag", "tungsten": "W",
    "titanium": "Ti", "zinc": "Zn", "lead": "Pb", "platinum": "Pt",
    "depleted_uranium": "U", "ceramic_alumina": "Al2O3",
}


def main() -> None:
    if not is_configured():
        print("MATERIALS_PROJECT_API_KEY not set -- not configured, skipping "
              "(expected on a machine without a key; see "
              "materials_project_client.py's module docstring to register).")
        return                                    # exit 0 -- not a failure

    out = {"_meta": {
        "source": "Materials Project (next-gen.materialsproject.org) -- "
                  "DFT-computed elastic tensors, Voigt-Reuss-Hill averaged",
        "generated_by": "tools/fetch_materials_project.py",
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
                        .isoformat(timespec="seconds"),
        "note": "NOT experimentally measured -- DFT-computed. No fatigue, "
                "creep, or Paris-law data; see materials_project_client.py "
                "module docstring.",
    }}
    ok, missing, failed = 0, [], []
    for canon, formula in CURATED_FORMULAS.items():
        try:
            data = elastic_properties(formula)
        except MaterialsProjectNotFound:
            missing.append(canon)
            continue
        except MaterialsProjectError as exc:
            failed.append((canon, str(exc)))
            continue
        out[canon] = {**data, "formula_queried": formula}
        ok += 1
        print(f"  + {canon:20s} <- {formula:6s} "
              f"K={data.get('bulk_modulus_vrh')} G={data.get('shear_modulus_vrh')} GPa")

    dest = os.path.abspath(_OUT)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {dest}: {ok} materials, {len(missing)} no elasticity data, "
          f"{len(failed)} query failures")
    if missing:
        print(f"  no elasticity data in MP for: {', '.join(missing)}")
    for canon, msg in failed:
        print(f"  FAILED {canon}: {msg[:150]}")


if __name__ == "__main__":
    main()
