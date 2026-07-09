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
    """A voxelized model: geometry (SDF) + per-cell material + exact mass props.

    PART IDENTITY (actuation epic, Lane 1): a label id is one REGION — one
    (part, material) pair. ``materials[k]`` KEEPS meaning "material name of
    label k" (duplicate names allowed), so every material-keyed consumer
    (interface scan, heat grids, colour bake) is unchanged; the parallel
    tables below carry the part view. All three default None — a field built
    from legacy 2-tuple input behaves exactly as before.
    """
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
    # ── part tables (None = unsegmented legacy field) ──
    parts: list = None          # [{part_id, name, material, labels, mass_kg,
                                #   com_m, inertia_kgm2, volume_m3, flags}]
    part_of_label: list = None  # label id → part id (index 0 → -1 = void)
    part_interfaces: dict = None  # {(pid_a, pid_b): {area_m2, centroid_m,
                                  #   principal_dir, elongation}} — the A-graph substrate

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


def _scan_part_interfaces(label, part_of_label, pitch, grid_lo):
    """6-neighbour face adjacency between PARTS → per-pair contact GEOMETRY.

    The A-graph substrate: for every touching part pair, the contact area plus
    deterministic axis cues computed from the face-midpoint cloud — centroid
    (a joint's anchor guess), principal direction (the dominant axis of the
    contact patch, e.g. a hinge line), and elongation (λ₁/λ₂ of the patch's
    2-D spread: a long thin patch ≈ a hinge; a round patch ≈ a pivot/weld).
    All streamed as moment sums — no face-point storage.
    """
    import numpy as np
    area = pitch * pitch
    pol = np.asarray(part_of_label, dtype=np.int64)
    P = int(pol.max()) + 2 if pol.size else 1
    acc = {}                        # pair code → [n, Σx(3), Σxxᵀ(3x3)]
    lo_w = np.asarray(grid_lo, float)
    for ax in (0, 1, 2):
        s1 = [slice(None)] * 3
        s2 = [slice(None)] * 3
        s1[ax] = slice(0, -1)
        s2[ax] = slice(1, None)
        A = label[tuple(s1)]
        B = label[tuple(s2)]
        pa = pol[A]
        pb = pol[B]
        m = (pa >= 0) & (pb >= 0) & (pa != pb)          # solid↔solid, different parts
        if not m.any():
            continue
        idx = np.argwhere(m).astype(float)              # A-side cell index
        mid = lo_w + (idx + 0.5) * pitch                # face midpoint world pos:
        mid[:, ax] += 0.5 * pitch                       #   half a cell along the axis
        qa, qb = pa[m], pb[m]
        code = np.minimum(qa, qb) * P + np.maximum(qa, qb)
        for cc in np.unique(code):
            sel = code == cc
            pts = mid[sel]
            n = int(sel.sum())
            s = pts.sum(0)
            ss = pts.T @ pts
            if cc in acc:
                acc[cc][0] += n
                acc[cc][1] += s
                acc[cc][2] += ss
            else:
                acc[cc] = [n, s, ss]
    out = {}
    for cc, (n, s, ss) in acc.items():
        pa, pb = int(cc // P), int(cc % P)
        centroid = s / n
        cov = ss / n - np.outer(centroid, centroid)
        w, v = np.linalg.eigh(cov)                       # ascending eigenvalues
        principal = v[:, 2]
        elong = float(w[2] / w[1]) if w[1] > 1e-18 else float("inf")
        out[(pa, pb)] = {
            "area_m2": n * area,
            "centroid_m": tuple(float(c) for c in centroid),
            "principal_dir": tuple(float(c) for c in principal),
            "elongation": round(elong, 3),
        }
    return out


def _finalize(label, materials, pitch, lo, density_of, watertight_frac,
              *, reconciliation=None, extra_note="", regions=None) -> VoxelField:
    """Build a VoxelField from a labeled grid: SDF + exact mass props + interfaces.

    Shared by ``voxelize`` (first build) and ``fill_cavity`` (re-derive after a
    fill paints cavity cells) so the two paths can never drift.

    ``regions``: the part table — ``[(part_name, material, [label ids], flags)]``
    in part-id order, or None for an unsegmented legacy field. Per-part
    mass/CoM/inertia come from the SAME sums as the totals, partitioned by
    region mask — exact by construction, and asserted so.
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
    lab_occ = label[occ]
    m = dens_by_id[lab_occ] * cellvol
    mass = float(m.sum())
    com = (world * m[:, None]).sum(0) / mass if mass > 0 else world.mean(0)
    r = world - com
    Ixx = float((m * (r[:, 1] ** 2 + r[:, 2] ** 2)).sum())
    Iyy = float((m * (r[:, 0] ** 2 + r[:, 2] ** 2)).sum())
    Izz = float((m * (r[:, 0] ** 2 + r[:, 1] ** 2)).sum())
    volume = float(occ.sum()) * cellvol

    interfaces, free = _scan_interfaces(label, materials, pitch)

    # ── part tables: the same sums, partitioned by region mask ──
    parts = part_of_label = part_interfaces = None
    if regions:
        part_of_label = [-1] * len(materials)
        parts = []
        mass_check = 0.0
        for pid, (pname, pmat, lids, flags) in enumerate(regions):
            for lid in lids:
                part_of_label[lid] = pid
            sel = np.isin(lab_occ, lids)
            pm = m[sel]
            pmass = float(pm.sum())
            mass_check += pmass
            if pmass > 0:
                pw = world[sel]
                pcom = (pw * pm[:, None]).sum(0) / pmass
                pr = pw - pcom
                pI = (float((pm * (pr[:, 1] ** 2 + pr[:, 2] ** 2)).sum()),
                      float((pm * (pr[:, 0] ** 2 + pr[:, 2] ** 2)).sum()),
                      float((pm * (pr[:, 0] ** 2 + pr[:, 1] ** 2)).sum()))
            else:                                       # a part that voxelized to nothing
                pcom, pI = com, (0.0, 0.0, 0.0)
                flags = tuple(flags) + ("empty",)
            parts.append({
                "part_id": pid, "name": pname, "material": pmat,
                "labels": tuple(int(x) for x in lids),
                "mass_kg": pmass,
                "com_m": tuple(float(c) for c in pcom),
                "inertia_kgm2": pI,                    # about the PART's own CoM, world axes
                "volume_m3": float(sel.sum()) * cellvol,
                "flags": tuple(flags),
            })
        assert abs(mass_check - mass) <= max(1e-9 * mass, 1e-15), (
            f"part masses {mass_check} != total {mass} — region partition broken")
        part_interfaces = _scan_part_interfaces(label, part_of_label, pitch, lo)

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
        parts=parts, part_of_label=part_of_label, part_interfaces=part_interfaces,
    )


def voxelize(parts, *, pitch=0.008, margin=4, density_of=None) -> VoxelField:
    """Voxelize ``parts`` into a field.

    ``parts`` entries are ``(trimesh.Trimesh, material_name)`` — the LEGACY
    form: one region per material, label grids byte-identical to the
    pre-part-identity behavior, part table mirroring the materials flagged
    ``("unsegmented",)`` — or ``(mesh, material_name, part_name)``: one region
    per (part, material), so parts survive as first-class bodies. Meshes
    sharing the same (part, material) share one region (a chair back's three
    OBJ files are ONE part). Part names must be unique per part instance
    (callers disambiguate repeats: "arm", "arm_2").

    Each mesh is filled independently (avoids merged-shell over-solidification),
    unioned into one labeled grid; the SDF is the signed Euclidean distance
    transform; mass/CoM/inertia are exact numpy sums over the labeled cells —
    totals AND per part, from the same sums.
    """
    import numpy as np
    import trimesh
    import scipy.ndimage as ndi

    if not parts:
        raise ValueError("voxelize: no parts")
    density_of = density_of or _default_density
    norm = []                                            # (mesh, material, part|None)
    for entry in parts:
        if len(entry) == 2:
            norm.append((entry[0], entry[1], None))
        else:
            norm.append((entry[0], entry[1], entry[2]))
    segmented = any(p is not None for _, _, p in norm)

    meshes = [m for m, _, _ in norm]
    full = trimesh.util.concatenate(meshes)
    lo = np.asarray(full.bounds[0], float) - pitch * margin
    hi = np.asarray(full.bounds[1], float) + pitch * margin
    dims = np.maximum(np.ceil((hi - lo) / pitch).astype(int), 1)

    label = np.zeros(tuple(dims.tolist()), dtype=np.int32)
    materials = ["air"]
    region_id = {}                                       # (part|None, material) → label id
    region_order = []                                    # region keys in id order
    watertight = []
    for mesh, name, part in norm:
        key = (part, name)
        mid = region_id.get(key)
        if mid is None:
            mid = len(materials)
            materials.append(name)                       # duplicates allowed: label→material
            region_id[key] = mid
            region_order.append(key)
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
        label[gi[:, 0], gi[:, 1], gi[:, 2]] = mid       # later mesh wins on overlap

    # the part table: one region per (part, material); legacy fields get one
    # part per material, honestly flagged unsegmented
    regions = []
    for part, name in region_order:
        lid = region_id[(part, name)]
        if part is None:
            regions.append((name, name, [lid], ("unsegmented",)))
        else:
            regions.append((part, name, [lid], ()))

    wt_frac = (sum(watertight) / len(watertight)) if watertight else 0.0
    return _finalize(label, materials, pitch, lo, density_of, wt_frac,
                     regions=regions,
                     extra_note=("" if segmented else "unsegmented (legacy input)"))


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

    # the fill's label id: on a PART-AWARE field the fill is always its OWN new
    # region (duplicate material names are legal now — merging the fill into an
    # existing part would corrupt per-part masses); legacy fields keep the old
    # reuse-the-material-id behavior exactly.
    if field.parts is not None:
        mid = len(materials)
        materials.append(fill_material)
    elif fill_material in materials:
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
    regions = None
    if field.parts is not None:
        # carry the part table forward + the fill as its own flagged part.
        # The liquid is APPROXIMATED AS RIGID (welded to its container) until
        # the fluid lane exists — flagged, never hidden.
        regions = [(p["name"], p["material"], list(p["labels"]), tuple(p["flags"]))
                   for p in field.parts]
        if filled_cells:
            regions.append((f"fill:{fill_material}", fill_material, [mid],
                            ("fill", "fluid_approximated_as_rigid")))
    new_field = _finalize(label, materials, pitch, lo, density_of,
                          field.watertight_frac, reconciliation=reconciliation,
                          extra_note=note, regions=regions)
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

    def __init__(self, voxel, materials, dominant=None, part_of_label=None):
        self._v = voxel
        self._mats = materials
        self._pol = part_of_label         # label id → part id (None = unsegmented)
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

    def part_at(self, x, y, z):
        """Part id at a point (−1 = void), or None when the field is unsegmented."""
        if self._pol is None:
            return None
        lid = self._v.material_at(x, y, z)
        if not lid:
            return -1
        return self._pol[lid]


def construct_from_field(name, field, *, source="", identified=True,
                         anno_id=None):
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
    if field.parts:
        # Layer-per-PART: the layer is named by the part, carries its material
        # (the existing name→material colour bake keeps working — layer.material
        # is what it reads), and its exact per-part mass/volume.
        mat_cells: dict = {}
        for p in field.parts:
            rho = float(dby.get(p["material"], 0.0))
            density_by_label[p["material"]] = rho
            layers.append(Layer(p["name"], p["material"], rho,
                                p["volume_m3"], p["mass_kg"],
                                source="voxel field (part)"))
            n = int(round(p["volume_m3"] / cellvol))
            mat_cells[p["material"]] = mat_cells.get(p["material"], 0) + n
        dominant = max(mat_cells, key=mat_cells.get) if mat_cells else None
    else:
        for i, matname in enumerate(field.materials):
            if i == 0:
                continue
            cells = int((lab == i).sum())
            if cells == 0:
                continue
            if cells > dominant_cells:                  # most-voxels material = the look
                dominant, dominant_cells = matname, cells
            rho = float(dby.get(matname, 0.0))
            vol = cells * cellvol
            density_by_label[matname] = rho
            layers.append(Layer(matname, matname, rho, vol, rho * vol,
                                source="voxel field"))

    composed = _VoxelComposed(voxel, field.materials, dominant,
                              part_of_label=field.part_of_label)

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
    if field.parts:
        validation["parts"] = [
            {"name": p["name"], "material": p["material"],
             "mass_kg": round(p["mass_kg"], 9), "flags": list(p["flags"])}
            for p in field.parts]
    if field.reconciliation:
        validation["volume_reconciliation"] = dict(field.reconciliation)
    articulation = None
    if field.parts:
        from .articulate import acquire
        articulation = acquire(anno_id, field.parts, field.part_interfaces,
                               source_note=source or field.notes)
    construct = Construct(
        name=name, composed=composed, density_by_label=density_by_label,
        layers=layers, mass_kg=field.mass_kg, com_m=field.com_m,
        inertia_kgm2=field.inertia_kgm2, bbox=bbox, validation=validation,
        identified=identified, source=source or field.notes,
        articulation=articulation)
    # the raw part view also rides the construct (tests + consumers that want
    # the field tables directly)
    construct.parts = field.parts
    construct.part_interfaces = field.part_interfaces
    construct.part_id_at = composed.part_at
    return construct


__all__ = ["VoxelField", "voxelize", "fill_cavity", "construct_from_field"]
