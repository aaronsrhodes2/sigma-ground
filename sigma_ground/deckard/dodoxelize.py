"""Dodoxel field — the FCC-lattice (rhombic-dodecahedral cell) analogue of
``voxelize.py``'s ``VoxelField``, built as a SIBLING module, not a
modification: the existing cubic ``VoxelField`` stays fully intact for
materials that don't warrant FCC treatment (amorphous solids per
field/interface/surface.py's ``bulk_coordination()`` — wood, plastic,
glass, concrete keep 6-neighbor cubic cells; only FCC/HCP metals are
candidates for dodoxel treatment, and even then only once a real caller
opts in — this module builds the CAPABILITY, wiring it into the real
mesh-voxelization entry point is explicitly a later, separate decision).

LATTICE REPRESENTATION: FCC sites are the even-parity sublattice of a
single dense cubic integer array (sites where i+j+k is even) — verified in
kernel/rhombic_dodecahedron.py to be the same lattice as the standard
4-sublattice conventional-cell basis, just re-indexed onto one array
instead of four. The array's own cell spacing is ``g = pitch / sqrt(2)``
(pitch = the FCC nearest-neighbor distance, matching voxelize.py's own use
of "pitch" for its cubic grid's edge length) — chosen so that the 12
integer neighbor offsets (permutations of (+-1,+-1,0), magnitude sqrt(2) in
array-index units) correspond to exactly ``pitch`` metres in world space.
Odd-parity array cells are simply never used (not a site, not occupied) --
roughly half the dense array's storage goes to unused cells, an accepted
simplification for a first implementation (a true sparse/half-density
structure is a possible future optimization, not required for correctness).

SDF: unlike voxelize.py's cubic VoxelField (which has no per-cell analytic
SDF available and so falls back to ``scipy.ndimage.distance_transform_edt``
on a boolean occupancy raster), each dodoxel CELL has an exact closed-form
SDF (RhombicDodecahedron.surface_distance, Phase 0). This is a genuine and
deliberate DEVIATION from the plan's originally-stated "rasterize each
cell's footprint then reuse distance_transform_edt" approach: computing the
field's SDF as ``min over nearby occupied sites of that site's own exact
analytic SDF`` is strictly MORE accurate (exact near the surface, not
capped by an extra rasterization resolution) and needs no extra rasterize-
to-fine-grid step at all -- it was only planned as "reuse EDT" because that
was the closest existing precedent at plan-writing time, before this
simpler and better approach was apparent once actually implementing it.

Mass/CoM/inertia: the SAME moment-sum pattern as voxelize.py's
``_finalize`` (cellvol x density at cell-center positions, summed) --
verified in the original plan to generalize cleanly to a different
cellvol/site-position formula, confirmed here by direct implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..kernel.rhombic_dodecahedron import RhombicDodecahedron, FCC_NEIGHBOR_OFFSETS
import math

_SQRT2 = math.sqrt(2.0)


@dataclass
class DodoxelField:
    """A dodoxel-lattice model: geometry (SDF) + per-site material + exact
    mass props. Mirrors VoxelField's field names 1:1 where the concept
    carries over (mass_kg, com_m, inertia_kgm2, volume_m3, ...); fields
    that are cubic-grid-specific (interfaces/free_surfaces from
    ``_scan_interfaces``'s 6-neighbor axis-slicing) are NOT part of this
    Phase 1 dataclass -- the 12-neighbor equivalent is Phase 2's job.
    """
    sdf_grid: object            # ndarray (nx,ny,nz): signed distance, m (- inside)
    occ_grid: object            # ndarray bool (nx,ny,nz): True at occupied EVEN-parity sites
    pitch: float                 # FCC nearest-neighbor distance, m
    cell_spacing: float          # underlying array spacing g = pitch/sqrt(2), m
    center: tuple                # grid geometric centre, world (m)
    material: str
    volume_m3: float
    mass_kg: float
    com_m: tuple
    inertia_kgm2: tuple          # (Ixx, Iyy, Izz) about the centre of mass
    site_count: int
    notes: str = ""
    # ── part tables (None = single-part field, the Phase 1 shape) ──
    # Mirrors VoxelField's own part-table convention (voxelize.py):
    parts: list = None           # [{part_id, name, material, mass_kg, com_m,
                                 #   inertia_kgm2, volume_m3, site_count}]
    part_grid: object = None     # ndarray int (nx,ny,nz): -1 = empty,
                                 #   else part_id at that occupied site
    part_interfaces: dict = None  # {(pid_a, pid_b): {...}} — Phase B's scan

    def to_voxel(self):
        """Build the kernel `Voxel` shape -- the render bridge (Phase 3).
        The sdf_grid here is ALREADY a plain regular-spacing cubic sample
        array (spacing=cell_spacing), so this needs no extra resampling:
        Voxel doesn't know or care that the samples originated from a
        rhombic-dodecahedral lattice rather than a cubic one."""
        from ..kernel.voxel import Voxel
        return Voxel(self.sdf_grid, self.cell_spacing, center=self.center,
                     volume_m3=self.volume_m3, source=self.notes)


def _site_world_pos(i, j, k, lo, g):
    """World position of lattice array index (i,j,k)."""
    return (lo[0] + i * g, lo[1] + j * g, lo[2] + k * g)


def dodoxelize_region(inside_test, pitch, bounds_lo, bounds_hi, material,
                      density_of, *, sdf_margin_cells=3):
    """Build a DodoxelField filling the region where ``inside_test(x,y,z)``
    is True, within the world-space box [bounds_lo, bounds_hi].

    This is the Phase 1 entry point -- a simple region predicate, not a
    real mesh. Real-mesh dodoxelization (surface-voxelize + fill, mirroring
    voxelize.py's actual ``voxelize()``) is explicitly out of scope for
    this phase (flagged as a later, separate decision in the approved
    plan) -- this function exists to prove the SDF/mass/CoM/inertia math
    is correct given SOME occupancy specification, exactly as voxelize.py's
    own ``_finalize`` doesn't do mesh-to-label conversion itself either.

    Args:
        inside_test: callable(x, y, z) -> bool, world-frame.
        pitch: FCC nearest-neighbor distance (m).
        bounds_lo, bounds_hi: (x,y,z) world-space box to search for occupied
            sites within.
        material: material name (single-material field; multi-material is
            future work, matching voxelize.py's own per-part precedent).
        density_of: callable(material_name) -> density (kg/m^3).
        sdf_margin_cells: how many extra lattice spacings of empty margin
            to pad the SDF output grid by, so boundary samples are safely
            positive (mirrors voxelize.py's own documented padding
            convention).
    """
    g = pitch / _SQRT2
    lo = list(bounds_lo)
    hi = list(bounds_hi)
    dims = [max(1, int(math.ceil((hi[a] - lo[a]) / g)) + 1) for a in range(3)]

    occ = {}   # (i,j,k) -> True, only even-parity occupied sites
    for i in range(dims[0]):
        x = lo[0] + i * g
        for j in range(dims[1]):
            y = lo[1] + j * g
            for k in range(dims[2]):
                if (i + j + k) % 2 != 0:
                    continue           # not an FCC lattice site
                z = lo[2] + k * g
                if inside_test(x, y, z):
                    occ[(i, j, k)] = True

    if not occ:
        raise ValueError("dodoxelize_region: empty occupancy — inside_test "
                         "matched no lattice site in the given bounds")

    return _field_from_occupancy(occ, pitch, g, lo, material, density_of,
                                 sdf_margin_cells)


def _field_from_occupancy(occ, pitch, g, lo, material, density_of,
                          sdf_margin_cells, extra_note="", part_of=None,
                          parts_meta=None):
    """Shared by ``dodoxelize_region`` (first build), ``find_disconnected``
    (re-derive per fragment after a cut) and ``dodoxelize_parts`` (multi-
    part build) so the paths can never drift -- same discipline as
    voxelize.py's own ``_finalize`` docstring states for its callers. The
    delicate origin/padding/fencepost logic lives HERE and only here (this
    session's own (dims-1)/2 fencepost bug is exactly the class of mistake
    that duplicate copies of this logic would eventually reintroduce).

    ``occ``: dict of (i,j,k) -> True, indices relative to ``lo``, even
    parity only (an FCC lattice site).
    ``part_of``: optional dict of (i,j,k) -> part_id. When given,
    ``parts_meta`` must be a list of {"name", "material"} in part-id order;
    per-site density comes from each site's own part's material, per-part
    mass/CoM/inertia tables are computed from the SAME sums (partitioned),
    and the partition is asserted to reproduce the totals exactly --
    mirroring voxelize._finalize's own mass_check discipline.
    """
    cell = RhombicDodecahedron(pitch)
    cellvol = cell.volume()

    if part_of is None:
        dens_of_site = {None: float(density_of(material))}

        def site_density(key):
            return dens_of_site[None]
    else:
        part_density = [float(density_of(pm["material"])) for pm in parts_meta]

        def site_density(key):
            return part_density[part_of[key]]

    # ── mass / CoM: same moment-sum pattern as voxelize._finalize, just
    # with the dodoxel cell's own volume/site-position formulas ──
    mass = 0.0
    sx = sy = sz = 0.0
    sites_world = {}
    for key in occ:
        wx, wy, wz = _site_world_pos(*key, lo, g)
        sites_world[key] = (wx, wy, wz)
        m_cell = site_density(key) * cellvol
        mass += m_cell
        sx += wx * m_cell
        sy += wy * m_cell
        sz += wz * m_cell
    com = (sx / mass, sy / mass, sz / mass)

    # Deliberate departure from voxelize.py's _finalize, noted explicitly:
    # _finalize treats each cell as a bare point mass (m*r^2, no per-cell
    # self-inertia) because its cells are a FINE subdivision of the real
    # object -- self-inertia is negligible next to the macroscopic spread.
    # A dodoxel field's cells are typically much coarser (each cell IS a
    # whole lattice site, not a fine subdivision), so the per-cell self-
    # inertia (isotropic, exact per Phase 0: I/m = pitch^2/8) is NOT
    # negligible here and is included via the parallel-axis theorem.
    i_self_factor = cell.inertia_factor()
    Ixx = Iyy = Izz = 0.0
    for key, (wx, wy, wz) in sites_world.items():
        m_cell = site_density(key) * cellvol
        i_self = i_self_factor * m_cell
        rx, ry, rz = wx - com[0], wy - com[1], wz - com[2]
        Ixx += m_cell * (ry * ry + rz * rz) + i_self
        Iyy += m_cell * (rx * rx + rz * rz) + i_self
        Izz += m_cell * (rx * rx + ry * ry) + i_self

    volume = len(occ) * cellvol

    # ── part tables: the same sums, partitioned by part id ──
    parts = None
    if part_of is not None:
        parts = []
        mass_check = 0.0
        for pid, pm in enumerate(parts_meta):
            keys = [k for k in occ if part_of[k] == pid]
            p_dens = float(density_of(pm["material"]))
            p_m_cell = p_dens * cellvol
            p_mass = p_m_cell * len(keys)
            mass_check += p_mass
            if p_mass > 0.0:
                pcx = sum(sites_world[k][0] for k in keys) / len(keys)
                pcy = sum(sites_world[k][1] for k in keys) / len(keys)
                pcz = sum(sites_world[k][2] for k in keys) / len(keys)
                p_i_self = i_self_factor * p_m_cell
                pI = [0.0, 0.0, 0.0]
                for k in keys:
                    wx, wy, wz = sites_world[k]
                    rx, ry, rz = wx - pcx, wy - pcy, wz - pcz
                    pI[0] += p_m_cell * (ry * ry + rz * rz) + p_i_self
                    pI[1] += p_m_cell * (rx * rx + rz * rz) + p_i_self
                    pI[2] += p_m_cell * (rx * rx + ry * ry) + p_i_self
                p_com = (pcx, pcy, pcz)
            else:
                p_com, pI = com, [0.0, 0.0, 0.0]
            parts.append({
                "part_id": pid, "name": pm["name"], "material": pm["material"],
                "mass_kg": p_mass, "com_m": p_com,
                "inertia_kgm2": tuple(pI),     # about the PART's own CoM, world axes
                "volume_m3": len(keys) * cellvol,
                "site_count": len(keys),
            })
        assert abs(mass_check - mass) <= max(1e-9 * mass, 1e-15), (
            f"part masses {mass_check} != total {mass} — partition broken")

    # ── SDF grid: min over nearby occupied sites' exact analytic SDF ──
    # `occ`'s keys are indexed relative to `lo` (the unpadded fill loop);
    # the SDF/occ output arrays are indexed relative to `sdf_lo` (padded
    # by `margin` on the low side). Re-key ONCE into the padded frame so
    # both grid-builders below share one consistent index space -- an
    # earlier version of this function called _sdf_grid_from_sites with
    # the un-re-keyed dict directly, which silently found zero neighbor
    # candidates everywhere (a real bug, caught by test_sdf_sign_correct_
    # at_occupied_site_and_far_outside asserting a false "outside" at the
    # fill's own center).
    margin = sdf_margin_cells
    dims = [max(k[a] for k in occ) - min(k[a] for k in occ) + 1 for a in range(3)]
    idx_lo = [min(k[a] for k in occ) for a in range(3)]
    sdf_dims = [dims[a] + 2 * margin for a in range(3)]
    sdf_lo = [lo[0] + (idx_lo[0] - margin) * g,
             lo[1] + (idx_lo[1] - margin) * g,
             lo[2] + (idx_lo[2] - margin) * g]
    occ_padded = {(i - idx_lo[0] + margin, j - idx_lo[1] + margin,
                  k - idx_lo[2] + margin): True for (i, j, k) in occ}
    sdf_grid = _sdf_grid_from_sites(occ_padded, pitch, g, sdf_lo, sdf_dims)
    occ_grid = _occ_grid_from_sites(occ_padded, sdf_dims)

    part_grid = part_interfaces = None
    if part_of is not None:
        import numpy as np
        part_grid = np.full(tuple(sdf_dims), -1, dtype=np.int32)
        for (i, j, k), pid in part_of.items():
            part_grid[i - idx_lo[0] + margin, j - idx_lo[1] + margin,
                      k - idx_lo[2] + margin] = pid
        part_interfaces = _scan_dodoxel_part_interfaces(part_grid, pitch, g,
                                                        sdf_lo)

    # kernel/voxel.py's Voxel places cell (i,j,k) at
    # center + (i-(nx-1)/2)*voxel_size -- i.e. index 0 is (nx-1)/2 cells
    # BELOW center, not nx/2 (N cells span (N-1) gaps, not N gaps; a fence-
    # post distinction). An earlier version of this line used
    # sdf_dims[a]*g*0.5, which is off by g/2 per axis -- silently wrong by
    # a small amount that didn't fail the loose Phase 1 sign-only checks,
    # caught only by test_dodoxel_field_to_parcel_sdf_local_is_correctly_
    # offset_from_com's EXACT value comparison against a known-occupied
    # cell. Fixed to match Voxel's own documented convention exactly.
    center = tuple(sdf_lo[a] + (sdf_dims[a] - 1) * g * 0.5 for a in range(3))
    note = (f"dodoxel {tuple(sdf_dims)} @ pitch={pitch*1000:.2f}mm "
            f"(cell_spacing={g*1000:.2f}mm); {len(occ)} occupied FCC sites; "
            f"material={material}")
    if extra_note:
        note += "; " + extra_note

    return DodoxelField(
        sdf_grid=sdf_grid, occ_grid=occ_grid, pitch=float(pitch),
        cell_spacing=float(g), center=center, material=material,
        volume_m3=volume, mass_kg=mass, com_m=com,
        inertia_kgm2=(Ixx, Iyy, Izz), site_count=len(occ), notes=note,
        parts=parts, part_grid=part_grid, part_interfaces=part_interfaces,
    )


def _scan_dodoxel_part_interfaces(part_grid, pitch, g, grid_lo):
    """12-neighbor part adjacency -> per-pair contact GEOMETRY, the dodoxel
    twin of voxelize._scan_part_interfaces -- with one capability the cubic
    scan can never have: each contact's normal is EXACT, not estimated.
    A rhombic dodecahedron's face toward a neighbor is, by construction,
    perpendicular to that neighbor's FCC direction (Voronoi-cell fact,
    verified in Phase 0's tessellation gate), so the discrete direction of
    each touching pair IS the true face normal -- no PCA needed for
    normals. PCA still supplies the contact patch's principal axis (the
    hinge-line cue) and elongation, same as the cubic scan.

    Per contact, the shared face area is exact: surface_area/12 =
    sqrt(2)/4 * pitch**2 (Phase 0's closed-form surface area over the 12
    identical rhombic faces).

    Returns {(pid_a, pid_b): {"area_m2", "n_contacts", "centroid_m",
    "principal_dir", "elongation", "contact_normals"}} with pid_a < pid_b
    and contact_normals = {unit-direction-tuple: count} oriented FROM
    pid_a TOWARD pid_b.
    """
    import numpy as np

    # one canonical representative per +- offset pair (6 of the 12), so a
    # touching site pair is counted exactly once
    pos_offsets = [(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
                  (0, 1, 1), (0, 1, -1)]
    face_area = 3.0 * _SQRT2 * pitch * pitch / 12.0
    nx, ny, nz = part_grid.shape
    acc = {}
    for idx in np.argwhere(part_grid >= 0):
        i, j, k = (int(v) for v in idx)
        pa = int(part_grid[i, j, k])
        for (di, dj, dk) in pos_offsets:
            i2, j2, k2 = i + di, j + dj, k + dk
            if not (0 <= i2 < nx and 0 <= j2 < ny and 0 <= k2 < nz):
                continue
            pb = int(part_grid[i2, j2, k2])
            if pb < 0 or pb == pa:
                continue
            # canonical pair (lo, hi); normal oriented lo -> hi
            if pa < pb:
                key, sign = (pa, pb), 1.0
            else:
                key, sign = (pb, pa), -1.0
            n_unit = (sign * di / _SQRT2, sign * dj / _SQRT2,
                      sign * dk / _SQRT2)
            n_unit = tuple(round(c, 9) for c in n_unit)
            mid = ((grid_lo[0] + (i + i2) * 0.5 * g),
                   (grid_lo[1] + (j + j2) * 0.5 * g),
                   (grid_lo[2] + (k + k2) * 0.5 * g))
            rec = acc.setdefault(key, {
                "n": 0, "s": np.zeros(3), "ss": np.zeros((3, 3)),
                "normals": {}})
            rec["n"] += 1
            m = np.asarray(mid)
            rec["s"] += m
            rec["ss"] += np.outer(m, m)
            rec["normals"][n_unit] = rec["normals"].get(n_unit, 0) + 1

    out = {}
    for key, rec in acc.items():
        n = rec["n"]
        centroid = rec["s"] / n
        cov = rec["ss"] / n - np.outer(centroid, centroid)
        w, v = np.linalg.eigh(cov)                    # ascending eigenvalues
        principal = v[:, 2]
        elong = float(w[2] / w[1]) if w[1] > 1e-18 else float("inf")
        out[key] = {
            "area_m2": n * face_area,
            "n_contacts": n,
            "centroid_m": tuple(float(c) for c in centroid),
            "principal_dir": tuple(float(c) for c in principal),
            "elongation": round(elong, 3),
            "contact_normals": dict(rec["normals"]),
        }
    return out


def dodoxelize_parts(parts_spec, pitch, bounds_lo, bounds_hi, density_of,
                     *, sdf_margin_cells=3):
    """Multi-part dodoxel build: like ``dodoxelize_region`` but with one
    region predicate PER PART, producing a single connected field carrying
    per-part mass/CoM/inertia tables -- the substrate the interface scan
    and joint inference (Phases B/C) read, mirroring how voxelize.py's
    ``VoxelField`` carries ``parts``/``part_of_label`` for the cubic case.

    Args:
        parts_spec: list of (name, material, inside_test) in PRIORITY
            order -- where two predicates overlap, the earlier part wins
            (a deterministic tie-break, stated rather than silent).
        pitch, bounds_lo, bounds_hi, density_of, sdf_margin_cells: as in
            ``dodoxelize_region``.
    """
    g = pitch / _SQRT2
    lo = list(bounds_lo)
    hi = list(bounds_hi)
    dims = [max(1, int(math.ceil((hi[a] - lo[a]) / g)) + 1) for a in range(3)]

    occ = {}
    part_of = {}
    for i in range(dims[0]):
        x = lo[0] + i * g
        for j in range(dims[1]):
            y = lo[1] + j * g
            for k in range(dims[2]):
                if (i + j + k) % 2 != 0:
                    continue
                z = lo[2] + k * g
                for pid, (pname, pmat, inside_test) in enumerate(parts_spec):
                    if inside_test(x, y, z):
                        occ[(i, j, k)] = True
                        part_of[(i, j, k)] = pid
                        break                       # priority: first wins

    if not occ:
        raise ValueError("dodoxelize_parts: empty occupancy — no predicate "
                         "matched any lattice site in the given bounds")

    parts_meta = [{"name": pname, "material": pmat}
                 for (pname, pmat, _) in parts_spec]
    label = "+".join(pm["material"] for pm in parts_meta)
    return _field_from_occupancy(occ, pitch, g, lo, label, density_of,
                                 sdf_margin_cells,
                                 extra_note=f"{len(parts_spec)} parts",
                                 part_of=part_of, parts_meta=parts_meta)


def find_disconnected(field, cut_test, density_of, anchor_test=None,
                      sdf_margin_cells=3):
    """Given a DodoxelField and a predicate marking cells to REMOVE (a cut
    event), return the resulting connected components as fresh
    DodoxelFields -- Teardown's "sever from anchor -> new rigid body"
    mechanic, built on real 12-neighbor FCC connectivity (verified against
    scipy's own ``ndimage.label`` conventions before being trusted: a
    hand-built two-site case sharing an FCC offset labels as ONE component
    under this structure and TWO under the default 6-connectivity
    structure; a non-FCC diagonal like (1,1,1) correctly does NOT connect).
    Reuses ``_field_from_occupancy`` per fragment so this can't drift from
    ``dodoxelize_region``'s own mass/CoM/inertia math.

    Args:
        field: the DodoxelField to cut.
        cut_test: callable(x, y, z) -> bool, world-frame; True cells are
            REMOVED before connectivity is re-evaluated.
        density_of: callable(material_name) -> density (kg/m^3), same
            convention as ``dodoxelize_region``.
        anchor_test: optional callable(x, y, z) -> bool, world-frame. When
            given, each returned fragment is tagged ``is_anchored`` (True
            if it contains at least one anchor cell) -- the actual "did
            this fall off" test a caller uses to decide which fragments
            need spinning off into new PhysicsParcels.

    Returns:
        list of (DodoxelField, is_anchored) tuples, largest-first by cell
        count. ``is_anchored`` is always False if ``anchor_test`` is None.
    """
    import numpy as np
    import scipy.ndimage as ndi
    from ..kernel.rhombic_dodecahedron import FCC_NEIGHBOR_OFFSETS

    nx, ny, nz = field.occ_grid.shape
    g = field.cell_spacing
    # same convention as Voxel/_field_from_occupancy: index 0 sits
    # (dims-1)/2 cells below center, not dims/2 (see _field_from_occupancy's
    # own comment on this exact fencepost bug, caught in this phase).
    lo = (field.center[0] - (nx - 1) * g / 2.0,
         field.center[1] - (ny - 1) * g / 2.0,
         field.center[2] - (nz - 1) * g / 2.0)

    occ = np.array(field.occ_grid, copy=True)
    for (i, j, k) in np.argwhere(occ):
        x, y, z = lo[0] + i * g, lo[1] + j * g, lo[2] + k * g
        if cut_test(x, y, z):
            occ[i, j, k] = False

    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1, 1, 1] = True
    for (di, dj, dk) in FCC_NEIGHBOR_OFFSETS:
        structure[1 + di, 1 + dj, 1 + dk] = True
    labeled, n_components = ndi.label(occ, structure=structure)
    if n_components == 0:
        raise ValueError("find_disconnected: cut removed all occupied cells")

    results = []
    for comp_id in range(1, n_components + 1):
        comp_occ = {}
        is_anchored = False
        for (i, j, k) in np.argwhere(labeled == comp_id):
            i, j, k = int(i), int(j), int(k)
            comp_occ[(i, j, k)] = True
            if anchor_test is not None and not is_anchored:
                x, y, z = lo[0] + i * g, lo[1] + j * g, lo[2] + k * g
                if anchor_test(x, y, z):
                    is_anchored = True
        frag = _field_from_occupancy(
            comp_occ, field.pitch, g, lo, field.material, density_of,
            sdf_margin_cells,
            extra_note=f"fragment {comp_id}/{n_components} of a cut field")
        results.append((frag, is_anchored))
    results.sort(key=lambda pair: pair[0].site_count, reverse=True)
    return results


def dodoxel_field_to_parcel(field, material, *, velocity=None,
                            orientation=None, label=""):
    """Bridge a DodoxelField to a live PhysicsParcel -- the first real
    consumer of PhysicsParcel's declared-but-never-populated ``sdf_local``
    narrow-phase collision callback (dynamics/parcel.py). This is what
    makes a fragment (from ``find_disconnected``) actually exercise the
    constraint solver rather than being a rendering-only trick.

    FRAME CONVENTION (this codebase's dynamics layer never previously had
    a real ``sdf_local`` consumer to check against -- documented precisely
    here since it's now load-bearing): ``sdf_local`` must be evaluated in
    the body's own LOCAL frame (parcel.py's own docstring: "optional local
    SDF callback for body-body narrow phase"), i.e. relative to the body's
    tracked CoM and orientation, not world position. ``field.to_voxel()``
    builds a ``Voxel`` centered at the FIELD's own geometric grid center
    (``field.center``), which is generally NOT the same point as the
    field's mass-weighted center of mass (``field.com_m``) -- since the
    physics engine tracks ``parcel.position`` as the body's CoM, the
    voxel's own ``.center`` is re-expressed here as the (fixed, rotation-
    invariant) offset from the CoM to the grid center, so
    ``sdf_local(lx, ly, lz)`` is correctly evaluated relative to the
    body's own local origin regardless of how the body has since moved.

    Args:
        field: the DodoxelField (or a fragment from find_disconnected).
        material: a Material object matching every other PhysicsParcel
            construction site's convention (density_kg_m3/restitution/
            density_at_sigma).
        velocity: initial velocity (vx,vy,vz), default rest.
        orientation: initial orientation quaternion, default identity.
        label: parcel label for debugging; defaults to field.material.
    """
    from ..dynamics.parcel import PhysicsParcel
    from ..dynamics.vec import Vec3

    voxel = field.to_voxel()
    voxel.center = tuple(c - m for c, m in zip(field.center, field.com_m))

    pos = Vec3(*field.com_m)
    vel = Vec3(*velocity) if velocity is not None else None
    return PhysicsParcel(
        voxel, material, position=pos, velocity=vel,
        mass=field.mass_kg, inertia_body=field.inertia_kgm2,
        orientation=orientation, sdf_local=voxel.surface_distance,
        label=label or field.material,
    )


def _neighbor_site_window(i, j, k):
    """Occupied-site candidates whose RhombicDodecahedron could plausibly
    reach a query point near array index (i,j,k) -- generous +-2 window in
    each axis (a cell's own bounding_radius is < 1 cell_spacing, so +-2 is
    comfortably conservative), filtered to even parity by the caller."""
    for di in range(-2, 3):
        for dj in range(-2, 3):
            for dk in range(-2, 3):
                yield (i + di, j + dj, k + dk)


def _sdf_grid_from_sites(occ, pitch, g, sdf_lo, sdf_dims):
    """Field SDF: exact near the surface, a TRUE conservative bound far away.

    Two-band construction, each band a mathematically valid (never-
    overestimating) lower bound on the true distance to the nearest cell
    surface -- the property sphere tracing requires:

      NEAR BAND (a window candidate exists): exact min over the +-2-index
        window of each site's closed-form SDF, clamped at 2g. The clamp is
        what makes it VALID: a site outside the window has its center >= 3
        index units away, hence its surface >= 3g - R = 2g away (the cell
        circumradius R = pitch/sqrt(2) EQUALS the array spacing g), so any
        window-exact value <= 2g cannot be beaten by an unseen site and is
        truly exact; values above 2g are capped to 2g, conservative.
        An earlier version had NO clamp and capped no-candidate cells at a
        flat 4*pitch -- an OVERESTIMATE wherever the nearest site sat just
        outside the window (true distance can be as low as 2g ~= 1.41*pitch
        < 4*pitch), i.e. a latent tunneling bug for thin dodoxel walls,
        caught by the validity gate in tests/test_dodoxel_sdf_farfield.py
        (grid value must never exceed the brute-force true distance).

      FAR BAND (no window candidate): distance to the nearest occupied
        site CENTER (scipy.ndimage.distance_transform_edt on the site
        raster, x g -- exact Euclidean center distance) minus R. Valid by
        the triangle inequality (every site's surface is at least
        center_distance - R away), automatically >= 2g here (no site
        within the window means center distance >= 3g), and UNBOUNDED far
        from the solid -- which is the Teardown-inspired empty-space skip:
        a marching ray in open space now steps by the true distance toward
        the field instead of creeping in flat 4*pitch hops. (The plan
        sketched this as a coarse-mip lookup structure; building the exact
        EDT bound directly into the grid is simpler, tighter, and reuses
        the same scipy EDT machinery voxelize.py's _finalize already
        depends on -- a deliberate, documented deviation, same as the
        min-over-analytic-SDFs choice above.)
    """
    import numpy as np
    import scipy.ndimage as ndi

    cell = RhombicDodecahedron(pitch)
    R = cell.bounding_radius()             # == g, see class docstring
    near_cap = 3.0 * g - R                 # = 2g: the window-exactness bound
    nx, ny, nz = sdf_dims

    occ_arr = np.zeros((nx, ny, nz), dtype=bool)
    for (i, j, k) in occ:
        occ_arr[i, j, k] = True
    # exact Euclidean distance (in index units) to the nearest occupied
    # site center, for every grid node; x g -> metres
    center_dist = ndi.distance_transform_edt(~occ_arr) * g

    grid = np.empty((nx, ny, nz), dtype=float)
    for i in range(nx):
        x = sdf_lo[0] + i * g
        for j in range(ny):
            y = sdf_lo[1] + j * g
            for k in range(nz):
                z = sdf_lo[2] + k * g
                best = float("inf")
                for cand in _neighbor_site_window(i, j, k):
                    if cand not in occ:
                        continue
                    cx, cy, cz = _site_world_pos(*cand, sdf_lo, g)
                    cell.center = (cx, cy, cz)
                    d = cell.surface_distance(x, y, z)
                    if d < best:
                        best = d
                if math.isinf(best):
                    grid[i, j, k] = center_dist[i, j, k] - R
                else:
                    grid[i, j, k] = min(best, near_cap)
    return grid


def _occ_grid_from_sites(occ, sdf_dims):
    import numpy as np
    nx, ny, nz = sdf_dims
    grid = np.zeros((nx, ny, nz), dtype=bool)
    for (i, j, k) in occ:
        grid[i, j, k] = True
    return grid


def part_subfield(field, part_id, density_of, sdf_margin_cells=3):
    """Extract one part of a multi-part DodoxelField as its own standalone
    single-material field -- reusing ``_field_from_occupancy`` (the one
    origin-math implementation) so a part's mass/CoM/inertia can never
    drift from what the parent field's own partition tables say."""
    import numpy as np
    if field.part_grid is None:
        raise ValueError("part_subfield needs a multi-part field")
    g = field.cell_spacing
    nx, ny, nz = field.part_grid.shape
    lo = tuple(field.center[a] - (dims - 1) * g * 0.5
              for a, dims in enumerate((nx, ny, nz)))
    occ = {}
    for idx in np.argwhere(field.part_grid == part_id):
        occ[tuple(int(v) for v in idx)] = True
    if not occ:
        raise ValueError(f"part {part_id} has no occupied sites")
    meta = field.parts[part_id]
    return _field_from_occupancy(occ, field.pitch, g, lo, meta["material"],
                                 density_of, sdf_margin_cells,
                                 extra_note=f"part '{meta['name']}' of a "
                                            "multi-part field")


def dodoxel_parts_to_scene(field, material_obj_of, density_of, *,
                           gravity=None, elongation_threshold=None):
    """The full bridge, closing the dormant inference-to-solver gap: a
    multi-part DodoxelField becomes a LIVE PhysicsScene -- one PhysicsParcel
    per part (each with its exact partitioned mass/CoM/inertia and its own
    narrow-phase SDF) connected by real dynamics/joints.py constraints
    typed by ``infer_dodoxel_joints``'s contact-geometry reasoning:
    weld -> WeldJoint, revolute -> RevoluteJoint about the lattice-snapped
    hinge axis through the interface centroid. This is the first time
    ANYTHING in the repo turns an inferred articulation into live solver
    constraints (deckard/articulate.py's cubic Joint graph has only ever
    round-tripped to scene_export for inspection).

    Args:
        field: multi-part DodoxelField (from ``dodoxelize_parts``).
        material_obj_of: callable(material_name) -> Material object for
            PhysicsParcel (kept as a callable so deckard never imports the
            materia layer above it -- same inversion-avoidance as
            ``density_of``).
        density_of: callable(material_name) -> density (kg/m^3).
        gravity: passed to PhysicsScene (None = engine default).
        elongation_threshold: forwarded to ``infer_dodoxel_joints``.

    Returns {"scene", "parcels" (part-id order), "joints" (live objects,
    same order as inference), "inference" (the raw proposal dict, choices
    included)}.
    """
    from ..dynamics.scene import PhysicsScene
    from ..dynamics.joints import WeldJoint, RevoluteJoint
    from ..dynamics.vec import Vec3
    from .dodoxel_articulate import infer_dodoxel_joints

    name_to_pid = {p["name"]: p["part_id"] for p in field.parts}
    parcels = []
    for p in field.parts:
        sub = part_subfield(field, p["part_id"], density_of)
        parcel = dodoxel_field_to_parcel(
            sub, material_obj_of(p["material"]), label=p["name"])
        # Jointed parts TOUCH by construction, so their full bounding
        # spheres overlap and the sphere narrow phase brakes the very
        # motion the joints permit (measured: an 11% spin loss over 0.5s
        # on a free hinge, iteration-count-independent, vanishing exactly
        # when the spheres stop pairing). Same resolution as record_clock's
        # fully-jointed arbors (trajectory.py): the narrow phase has no
        # legitimate job BETWEEN members of one jointed assembly, so each
        # member's collision radius is shrunk well below any inter-part
        # spacing. Honest scope note: this also disables the assembly's
        # sphere-phase contact with ground/other bodies -- fragments spun
        # off by find_disconnected keep their full radius and collide
        # normally.
        parcel.radius = 0.25 * field.pitch
        parcels.append(parcel)

    inference = infer_dodoxel_joints(field,
                                     elongation_threshold=elongation_threshold)
    live_joints = []
    for j in inference["joints"]:
        a = parcels[name_to_pid[j["part_a"]]]
        b = parcels[name_to_pid[j["part_b"]]]
        anchor = Vec3(*j["anchor_m"])
        if j["type"] == "revolute":
            live_joints.append(RevoluteJoint(a, b, anchor, Vec3(*j["axis"])))
        else:
            live_joints.append(WeldJoint(a, b, anchor))

    kw = {"gravity": gravity} if gravity is not None else {}
    scene = PhysicsScene(parcels, ground=False, constraints=live_joints, **kw)
    return {"scene": scene, "parcels": parcels, "joints": live_joints,
            "inference": inference}


__all__ = ["DodoxelField", "dodoxelize_region", "dodoxelize_parts",
          "find_disconnected", "dodoxel_field_to_parcel", "part_subfield",
          "dodoxel_parts_to_scene"]
