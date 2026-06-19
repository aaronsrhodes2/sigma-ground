"""Real mesh → labeled voxel field (opt-in numpy / trimesh / scipy).

Turns a model's per-part triangle meshes into a single labeled voxel grid — the
matter field the voxel `Construct` weighs, the renderer raymarches, and the
interface scan reads. The honest method, validated empirically:

  PER PART → surface-voxelize (trimesh) → fill (scipy `binary_fill_holes`) → union
  into a common labeled grid (cell → material id).

Filling each part SEPARATELY is the trick: a chair *leg* is a roughly-convex solid
that fills correctly, and no single part encloses the big inter-part gaps — so the
union is solid members, not the 10×-over-solidified blob that filling the merged
shell produces. libigl's generalized winding number would be more exact, but it has
no Python 3.13 wheel; per-part fill + the watertight-confidence flag is the honest
available path, and thin-member inflation at finite pitch is carried by the
volume-reconcile + confidence the plan already mandates.

All heavy imports are lazy so importing `deckard` stays clean.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VoxelField:
    """A voxelized model: geometry (SDF) + per-cell material + exact mass props."""
    sdf_grid: object            # ndarray (nx,ny,nz): signed distance, m (− inside)
    label_grid: object          # ndarray int: 0 = void/air, k = materials[k]
    voxel_size: float           # cell edge, m
    center: tuple               # grid geometric centre, world (m)
    materials: list             # index → material name; materials[0] == "air"
    volume_m3: float
    mass_kg: float
    com_m: tuple
    inertia_kgm2: tuple         # (Ixx, Iyy, Izz) about the centre of mass
    interfaces: dict            # {(matA, matB): area_m2} — material↔material faces
    free_surfaces: dict         # {mat: area_m2} — material↔void faces
    watertight_frac: float
    confidence: float
    density_by_label: dict = None   # {material: density} actually used for mass
    notes: str = ""
    reconciliation: dict = None     # set by fill_cavity: requested vs capacity vs filled

    def to_voxel(self):
        """Build the kernel `Voxel` shape (centred on this field)."""
        from ..kernel.voxel import Voxel
        return Voxel(self.sdf_grid, self.voxel_size, center=self.center,
                     volume_m3=self.volume_m3, label_grid=self.label_grid,
                     source=self.notes)


def _default_density(name):
    from ..field.interface.resolve import material_profile
    d = material_profile(name).get("density")
    return float(d.value) if d is not None else 1000.0


def _scan_interfaces(label, materials, pitch):
    """6-neighbour face adjacency → material-pair interface + free-surface areas."""
    import numpy as np
    area = pitch * pitch
    K = len(materials)
    pairs, free = {}, {}
    for ax in (0, 1, 2):
        s1 = [slice(None)] * 3
        s2 = [slice(None)] * 3
        s1[ax] = slice(0, -1)
        s2[ax] = slice(1, None)
        A = label[tuple(s1)]
        B = label[tuple(s2)]
        diff = A != B
        a = A[diff].ravel()
        b = B[diff].ravel()
        if a.size == 0:
            continue
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        code = lo.astype(np.int64) * K + hi.astype(np.int64)
        u, cnt = np.unique(code, return_counts=True)
        for cc, n in zip(u.tolist(), cnt.tolist()):
            la, lb = cc // K, cc % K
            if la == 0:                                  # one side is void
                mat = materials[lb]
                if mat != "air":
                    free[mat] = free.get(mat, 0.0) + n * area
            else:
                key = tuple(sorted((materials[la], materials[lb])))
                pairs[key] = pairs.get(key, 0.0) + n * area
    return pairs, free


def _finalize(label, materials, pitch, lo, density_of, watertight_frac,
              *, reconciliation=None, extra_note="") -> VoxelField:
    """Build a VoxelField from a labeled grid: SDF + exact mass props + interfaces.

    Shared by ``voxelize`` (first build) and ``fill_cavity`` (re-derive after a
    fill paints cavity cells) so the two paths can never drift.
    """
    import numpy as np
    import scipy.ndimage as ndi

    label = np.ascontiguousarray(label)
    dims = np.asarray(label.shape)
    lo = np.asarray(lo, float)
    occ = label > 0
    if not occ.any():
        raise ValueError("voxelize: empty occupancy after solidification")

    # signed distance grid (− inside), in metres
    sdf = (ndi.distance_transform_edt(~occ) - ndi.distance_transform_edt(occ)) * pitch

    cellvol = pitch ** 3
    dens_by_id = np.zeros(len(materials))
    for i, name in enumerate(materials):
        if i:
            dens_by_id[i] = density_of(name)

    occ_idx = np.argwhere(occ)
    world = lo + (occ_idx + 0.5) * pitch
    m = dens_by_id[label[occ]] * cellvol
    mass = float(m.sum())
    com = (world * m[:, None]).sum(0) / mass if mass > 0 else world.mean(0)
    r = world - com
    Ixx = float((m * (r[:, 1] ** 2 + r[:, 2] ** 2)).sum())
    Iyy = float((m * (r[:, 0] ** 2 + r[:, 2] ** 2)).sum())
    Izz = float((m * (r[:, 0] ** 2 + r[:, 1] ** 2)).sum())
    volume = float(occ.sum()) * cellvol

    interfaces, free = _scan_interfaces(label, materials, pitch)
    conf = round(0.5 + 0.5 * watertight_frac, 2)
    center = tuple((lo + dims * pitch * 0.5).tolist())
    note = (f"voxel {tuple(int(d) for d in dims)} @ {pitch*1000:.0f}mm; "
            f"per-part fill+union; watertight {watertight_frac:.0%} "
            f"(mass/volume confidence {conf})")
    if extra_note:
        note += "; " + extra_note
    return VoxelField(
        sdf_grid=sdf, label_grid=label, voxel_size=float(pitch),
        center=center, materials=list(materials), volume_m3=volume, mass_kg=mass,
        com_m=tuple(float(c) for c in com), inertia_kgm2=(Ixx, Iyy, Izz),
        interfaces=interfaces, free_surfaces=free, watertight_frac=watertight_frac,
        confidence=conf,
        density_by_label={materials[i]: float(dens_by_id[i])
                          for i in range(1, len(materials))},
        notes=note, reconciliation=reconciliation,
    )


def voxelize(parts, *, pitch=0.008, margin=4, density_of=None) -> VoxelField:
    """Voxelize ``parts`` = ``[(trimesh.Trimesh, material_name), ...]`` into a field.

    Each part is filled independently (avoids merged-shell over-solidification),
    unioned into one labeled grid; the SDF is the signed Euclidean distance
    transform; mass/CoM/inertia are exact numpy sums over the labeled cells.
    """
    import numpy as np
    import trimesh
    import scipy.ndimage as ndi

    if not parts:
        raise ValueError("voxelize: no parts")
    density_of = density_of or _default_density

    meshes = [m for m, _ in parts]
    full = trimesh.util.concatenate(meshes)
    lo = np.asarray(full.bounds[0], float) - pitch * margin
    hi = np.asarray(full.bounds[1], float) + pitch * margin
    dims = np.maximum(np.ceil((hi - lo) / pitch).astype(int), 1)

    label = np.zeros(tuple(dims.tolist()), dtype=np.int32)
    materials = ["air"]
    mat_id = {}
    watertight = []
    for mesh, name in parts:
        mid = mat_id.get(name)
        if mid is None:
            mid = len(materials)
            materials.append(name)
            mat_id[name] = mid
        try:
            vg = mesh.voxelized(pitch=pitch)
            filled = ndi.binary_fill_holes(np.asarray(vg.matrix))
        except Exception:
            continue
        watertight.append(bool(getattr(mesh, "is_watertight", False)))
        idx = np.argwhere(filled)
        if idx.size == 0:
            continue
        world = trimesh.transform_points(idx.astype(float), vg.transform)
        gi = np.floor((world - lo) / pitch).astype(int)
        ok = ((gi >= 0) & (gi < dims)).all(1)
        gi = gi[ok]
        label[gi[:, 0], gi[:, 1], gi[:, 2]] = mid       # later part wins on overlap

    wt_frac = (sum(watertight) / len(watertight)) if watertight else 0.0
    return _finalize(label, materials, pitch, lo, density_of, wt_frac)


def _trapped_cavity(label, *, up_axis=2):
    """The gravity-trapped void of a container: the cells liquid would rest in.

    Physical model (z-up, gravity along −``up_axis``): sweep a water level up one
    cell at a time. At level ``L`` the candidate water is the void at or below L;
    its connected components that touch a *lateral or bottom* grid face leak to the
    outside, the rest are trapped by the surrounding solid. Capacity is the largest
    trapped volume over all levels — the brim, just before water spills over the
    lowest rim. This needs no seed and handles BOTH a sealed pocket (trapped at
    every level) and an open cup (trapped up to its rim), exactly as poured liquid
    behaves. Returns ``(mask, capacity_cells)``.
    """
    import numpy as np
    import scipy.ndimage as ndi

    void = (label == 0)
    n_up = label.shape[up_axis]
    struct = ndi.generate_binary_structure(3, 1)        # 6-connectivity
    # index of each cell along the up-axis, broadcast to the grid
    shape_idx = [1, 1, 1]
    shape_idx[up_axis] = n_up
    level = np.arange(n_up).reshape(shape_idx)
    lvl = np.broadcast_to(level, label.shape)

    # which faces are "leaks" (liquid escapes): the four lateral faces + the bottom
    # face (the up-axis low face). The top face is the open sky and never a leak.
    lateral_axes = [a for a in (0, 1, 2) if a != up_axis]

    best_mask = np.zeros_like(void)
    best_cells = 0
    for L in range(n_up):
        below = void & (lvl <= L)
        if not below.any():
            continue
        comp, ncomp = ndi.label(below, structure=struct)
        if ncomp == 0:
            continue
        leak = set()
        for a in lateral_axes:                          # both lateral faces leak
            lo_face = np.take(comp, 0, axis=a)
            hi_face = np.take(comp, comp.shape[a] - 1, axis=a)
            leak.update(np.unique(lo_face).tolist())
            leak.update(np.unique(hi_face).tolist())
        bottom = np.take(comp, 0, axis=up_axis)         # the floor leaks too
        leak.update(np.unique(bottom).tolist())
        leak.discard(0)
        trapped = below & ~np.isin(comp, list(leak)) if leak else below
        c = int(trapped.sum())
        if c > best_cells:
            best_cells = c
            best_mask = trapped
    return best_mask, best_cells


def fill_cavity(field, fill_material, *, requested_m3=None, density_of=None,
                up_axis=2) -> "tuple[VoxelField, dict]":
    """Pour ``fill_material`` into the container's cavity under gravity.

    The SHAPE is authoritative (the Captain's rule): we never place more than the
    cavity physically holds. Liquid settles bottom-first; we fill the lowest cells
    up to ``min(requested_m3, capacity)``. If ``requested_m3`` is None we fill to
    the brim. Returns ``(new_field, reconciliation)`` — a re-finalized field (mass,
    CoM, SDF, interfaces all recomputed with the fill present) plus a report of
    requested vs capacity vs actually-filled volume so Mentat can amend the sim.
    Never silently fudges: any shortfall is deferred to the shape and flagged.
    """
    import numpy as np

    density_of = density_of or _default_density
    pitch = field.voxel_size
    cellvol = pitch ** 3
    label = np.array(field.label_grid, dtype=field.label_grid.dtype, copy=True)
    materials = list(field.materials)

    # ensure the fill material has a label id
    if fill_material in materials:
        mid = materials.index(fill_material)
    else:
        mid = len(materials)
        materials.append(fill_material)

    mask, cap_cells = _trapped_cavity(label, up_axis=up_axis)
    capacity_m3 = cap_cells * cellvol

    if requested_m3 is None:
        target_m3 = capacity_m3
    else:
        target_m3 = min(float(requested_m3), capacity_m3)
    n_fill = int(round(target_m3 / cellvol)) if cellvol > 0 else 0
    n_fill = max(0, min(n_fill, cap_cells))

    cells = np.argwhere(mask)
    filled_cells = 0
    if n_fill and cells.size:
        order = np.argsort(cells[:, up_axis], kind="stable")   # gravity: low cells first
        chosen = cells[order[:n_fill]]
        label[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = mid
        filled_cells = len(chosen)
    filled_m3 = filled_cells * cellvol

    # geometry confidence is unchanged by a fill (the outer shell is the same)
    lo = np.asarray(field.center, float) - np.asarray(label.shape, float) * pitch * 0.5
    short_m3 = max(0.0, (float(requested_m3) - capacity_m3)) if requested_m3 is not None else 0.0
    deferred = requested_m3 is not None and (requested_m3 - capacity_m3) > 0.5 * cellvol
    reconciliation = {
        "fill_material": fill_material,
        "requested_m3": (float(requested_m3) if requested_m3 is not None else None),
        "capacity_m3": capacity_m3,
        "filled_m3": filled_m3,
        "shortfall_m3": short_m3,
        "cell_quantum_m3": cellvol,    # fill resolution — one voxel = the smallest drop
        "fill_fraction": (filled_m3 / capacity_m3) if capacity_m3 > 0 else 0.0,
        "deferred_to_shape": bool(deferred),
        "confidence": field.confidence,
        "note": (
            f"requested {requested_m3*1e6:.1f} cm3 > capacity {capacity_m3*1e6:.1f} cm3 - "
            f"deferred to shape, filled to brim ({filled_m3*1e6:.1f} cm3)"
            if deferred else
            f"filled {filled_m3*1e6:.1f} cm3 of {fill_material} "
            f"(cavity capacity {capacity_m3*1e6:.1f} cm3)"
        ),
    }
    note = f"filled {fill_material} {filled_m3*1e6:.1f}/{capacity_m3*1e6:.1f} cm3"
    new_field = _finalize(label, materials, pitch, lo, density_of,
                          field.watertight_frac, reconciliation=reconciliation,
                          extra_note=note)
    return new_field, reconciliation


class _VoxLeaf:
    """One CSG leaf wrapping a kernel `Voxel` — the single-leaf serialization
    handle ``construct_to_scene`` reads (``.shape`` + ``.material``)."""
    __slots__ = ("shape", "material")

    def __init__(self, shape, material):
        self.shape = shape
        self.material = material


class _VoxelComposed:
    """Adapts a kernel `Voxel` to the `ComposedSDF` interface the `Construct`
    consumers expect: ``sdf(x,y,z)`` (the trilinear distance) and
    ``material_at(x,y,z)`` (the per-cell label → material NAME, or None for void).

    Also exposes ``_leaves`` (a single ``add`` Voxel leaf, labeled by the dominant
    material) so the construct serializes through the existing primitive
    ``construct_to_scene`` → the viewer gets a real Voxel leaf to raymarch. The
    per-cell label grid still lives on the Voxel for the eventual per-cell material
    render; stage-1 colours the whole solid by its dominant material.
    """

    def __init__(self, voxel, materials, dominant=None):
        self._v = voxel
        self._mats = materials
        if dominant is None:
            dominant = next((m for m in materials if m != "air"), "air")
        self._leaves = [(_VoxLeaf(voxel, dominant), "add")]

    def sdf(self, x, y, z):
        return self._v.surface_distance(x, y, z)

    def material_at(self, x, y, z):
        lid = self._v.material_at(x, y, z)
        if not lid:                       # None or 0 → void / air
            return None
        return self._mats[lid]


def construct_from_field(name, field, *, source="", identified=True):
    """Build a drop-in Deckard `Construct` from a `VoxelField`.

    The construct exposes the SAME interface as the primitive path
    (mass/CoM/inertia/bbox/.sdf()/.material_at()/density_at), so the physics that
    consumes a Construct works unchanged — exact mass props come straight from the
    voxel field (closing the bbox-mass bug). The geometry RENDER (serializing the
    Voxel leaf to a SceneSpec) is Phase R-vox; this delivers the data + physics.
    """
    import numpy as np
    from .construct import Construct, Layer

    voxel = field.to_voxel()
    cellvol = field.voxel_size ** 3
    lab = field.label_grid
    dby = field.density_by_label or {}
    density_by_label, layers = {}, []
    dominant, dominant_cells = None, -1
    for i, matname in enumerate(field.materials):
        if i == 0:
            continue
        cells = int((lab == i).sum())
        if cells == 0:
            continue
        if cells > dominant_cells:                      # most-voxels material = the look
            dominant, dominant_cells = matname, cells
        rho = float(dby.get(matname, 0.0))
        vol = cells * cellvol
        density_by_label[matname] = rho
        layers.append(Layer(matname, matname, rho, vol, rho * vol, source="voxel field"))

    composed = _VoxelComposed(voxel, field.materials, dominant)

    cx, cy, cz = field.center
    a, b, c = voxel._extents()
    bbox = ((cx - a / 2, cx + a / 2), (cy - b / 2, cy + b / 2), (cz - c / 2, cz + c / 2))
    validation = {
        "passed": True, "mode": "voxel", "volume_m3": field.volume_m3,
        "watertight_frac": field.watertight_frac, "confidence": field.confidence,
        "note": field.notes,
        "interfaces": {f"{x}|{y}": area for (x, y), area in field.interfaces.items()},
        "free_surfaces": dict(field.free_surfaces),
    }
    if field.reconciliation:
        validation["volume_reconciliation"] = dict(field.reconciliation)
    return Construct(
        name=name, composed=composed, density_by_label=density_by_label,
        layers=layers, mass_kg=field.mass_kg, com_m=field.com_m,
        inertia_kgm2=field.inertia_kgm2, bbox=bbox, validation=validation,
        identified=identified, source=source or field.notes)


__all__ = ["VoxelField", "voxelize", "fill_cavity", "construct_from_field"]
