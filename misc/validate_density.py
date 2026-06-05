"""Dry-run: does crystallographic density reproduce reality?

Two derivations, compared to the tabulated density:
  cell  = rho from the MEASURED crystal cell:  n_atoms * (A*u) / V_cell
  fromZ = element.predict_density_kg_m3(Z): cell PREDICTED from Slater radii

If `cell` matches the tabulated value to ~1-2%, density is accurately emergent
from composition + the measured cell — no stored density needed. `fromZ` is the
fully-from-Z version (rougher) that already exists.
"""
import math
import sys

sys.path.insert(0, r"D:\Aaron\development\sigma-ground")

from sigma_ground.field.interface.surface import MATERIALS
from sigma_ground.field.interface.element import predict_density_kg_m3

try:
    from sigma_ground.field.constants import AMU_KG
except Exception:
    AMU_KG = 1.66053906660e-27   # CODATA atomic mass unit (kg)

_ATOMS_PER_CELL = {"fcc": 4, "bcc": 2, "hcp": 2, "diamond": 8, "diamond_cubic": 8}


def _v_cell_m3(structure, a_m):
    if structure == "hcp":
        c = a_m * math.sqrt(8.0 / 3.0)            # ideal c/a
        return a_m * a_m * c * math.sqrt(3.0) / 2.0
    return a_m ** 3                               # cubic (fcc/bcc/diamond)


def crystal_density(structure, a_angstrom, A):
    a_m = a_angstrom * 1e-10
    n = _ATOMS_PER_CELL.get(structure, 1)
    return n * A * AMU_KG / _v_cell_m3(structure, a_m)


print(f"{'metal':17s}{'struct':8s}{'tab kg/m3':>10s}{'cell':>9s}{'err%':>7s}{'fromZ':>9s}{'err%':>7s}")
print("-" * 67)
cell_errs = []
for k, m in MATERIALS.items():
    if m.get("material_type") != "metal":
        continue
    a, s = m.get("lattice_param_angstrom"), m.get("crystal_structure")
    rho, A, Z = m.get("density_kg_m3"), m.get("A"), m.get("Z")
    if not (a and s and rho and A and Z):
        continue
    dc = crystal_density(s, a, A)
    try:
        dz = predict_density_kg_m3(Z)
    except Exception:
        dz = float("nan")
    ec = 100.0 * (dc - rho) / rho
    ez = 100.0 * (dz - rho) / rho if dz == dz else float("nan")
    cell_errs.append(abs(ec))
    print(f"{k:17s}{s:8s}{rho:10.0f}{dc:9.0f}{ec:7.1f}{dz:9.0f}{ez:7.1f}")

if cell_errs:
    print("-" * 67)
    print(f"cell-derived mean |err| = {sum(cell_errs)/len(cell_errs):.2f}%   "
          f"max |err| = {max(cell_errs):.2f}%   (n={len(cell_errs)})")
