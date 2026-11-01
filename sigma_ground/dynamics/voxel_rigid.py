"""
Rigid voxel bodies that rotate, touch, slide, topple and break (SKIPPYCRAFT_PLAN 10b, P3).

A body is a set of cubic cells of one material, each with its own mass and temperature, moving as one rigid piece:
centre of mass, velocity, orientation (a quaternion) and angular velocity, with the inertia tensor summed from its
cells (each a solid cube). Bodies are drawn back onto the grid one cell per cell (``raster``), so what is drawn always
holds exactly the body's mass, whatever its angle.

Contact (``RigidWorld.step``): the classic voxmap/point-shell test (McNeely, Neff and Troy, SIGGRAPH 1999): the corners
of a body's exposed faces are tested against the signed distance field of the static walls and of every other body
(trilinear, in cell units); every corner inside gives a contact point with its depth and normal. The contacts are solved
by sequential impulses (Catto's projected Gauss-Seidel): non-penetration with the coefficient of restitution, Coulomb
friction in a circular cone (static coefficient below 1 mm/s of sliding, kinetic above), and penetration removed by
split impulses, which move the bodies without giving them speed (so correcting overlap adds no kinetic energy).

Every number comes from the materials: the friction from ``friction.contact_friction`` (measured first), the restitution
from ``impact.coefficient_of_restitution`` (Johnson-Thornton, from yield stress and modulus, at the impact speed).

Strength (``sections``): each body is cut by every plane between two layers of its cells, along its own three axes. The
force and moment carried across a cut are what one side needs to move as the rigid body moves (its weight, the contact
forces on it, less its mass times its acceleration). A ductile cut fails when that moment exceeds the cut's fully plastic
moment, every face at its own cell's yield stress at its own temperature (measured for carbon steel,
``plasticity.steel_yield_at``; else Johnson-Cook's thermal softening, ``plasticity.johnson_cook_stress``), less what the shear takes (von Mises); a brittle one when the elastic tensile stress
on any face exceeds its tensile strength. A ductile failure is a PLASTIC HINGE: the body splits in two, joined at the cut
by a ball joint that resists bending with no more than the plastic moment at the hinge cells' current temperature, so a
hot beam sags, slowly, faster as it heats; the hinge tears when its bend strains the outer fibre past the material's
elongation at fracture (hinge length = the section's depth: strain = angle / 2). The plastic work at the hinge, and the
kinetic energy contact takes away, are returned as heat in the cells where they happen.

Honest limits (``gaps``): elastic deflection before yield is not drawn (micrometres here); the hinge resists torsion with
its bending capacity; brittle compression and the strength of brittle materials when hot are not modelled (room
temperature tensile strength); the static walls are infinitely strong; contact conducts heat perfectly (the grid's cells
are simply neighbours).
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
except ImportError:                                                     # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not a or not callable(a[0]) else a[0]

FACES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])


# ── rotations ────────────────────────────────────────────────────────────────────────────────────────────────────────
def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_rotvec(rv):
    a = float(np.linalg.norm(rv))
    if a < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    s = math.sin(a / 2) / a
    return np.array([math.cos(a / 2), rv[0] * s, rv[1] * s, rv[2] * s])


def quat_angle(q):
    return 2 * math.acos(min(1.0, abs(float(q[0]))))


# ── signed distance fields (cell units, negative inside) ───────────────────────────────────────────────────────────
def signed_distance(occ):
    """At each cell centre: distance to the solid's surface in cells, negative inside. ``occ`` must have empty and
    solid cells both (pad it). On a flat face the interpolated zero lies exactly on the face."""
    from scipy.ndimage import distance_transform_edt as edt
    occ = np.asarray(occ, bool)
    return np.where(occ, -(edt(occ) - 0.5), edt(~occ) - 0.5)


def sample_sdf(phi, p):
    """Trilinear value and gradient of ``phi`` at points ``p`` (N,3) in index space (cell centre i at i). Outside the
    array the edge value is held."""
    hi = np.array(phi.shape) - 1
    p = np.clip(p, 0, hi - 1e-9)
    i0 = np.floor(p).astype(int)
    i0 = np.minimum(i0, hi - 1)
    f = p - i0
    val = np.zeros(len(p))
    grad = np.zeros((len(p), 3))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                c = phi[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
                wx = f[:, 0] if dx else 1 - f[:, 0]
                wy = f[:, 1] if dy else 1 - f[:, 1]
                wz = f[:, 2] if dz else 1 - f[:, 2]
                val += c * wx * wy * wz
                grad[:, 0] += c * (1 if dx else -1) * wy * wz
                grad[:, 1] += c * wx * (1 if dy else -1) * wz
                grad[:, 2] += c * wx * wy * (1 if dz else -1)
    return val, grad


def voxel_penetration(occ, phi, g):
    """Points at ``g`` (N,3), in a padded occupancy grid's continuous coordinates (cell i spans [i, i+1)): which are
    inside a solid cell, how deep (cells) and the outward normal. Exact for cubes: the depth is the distance to the
    nearest face of the point's cell that is open (its neighbour empty), the normal that face's; a point with no open
    face nearby (deep inside) takes the distance field's depth and gradient. (A distance field alone rounds every
    edge by a third of a cell, so a block flush on a beam of its own width would stand on a knife edge.)"""
    hi = np.array(occ.shape) - 1
    cell = np.clip(np.floor(g).astype(int), 0, hi)
    inside = occ[cell[:, 0], cell[:, 1], cell[:, 2]]
    depth = np.zeros(len(g))
    normal = np.zeros((len(g), 3))
    if not inside.any():
        return inside, depth, normal
    c = cell[inside]
    fr = g[inside] - c
    best = np.full(len(c), np.inf)
    nrm = np.zeros((len(c), 3))
    for f in FACES:
        ax = int(np.nonzero(f)[0][0])
        nb = np.clip(c + f, 0, hi)
        open_ = ~occ[nb[:, 0], nb[:, 1], nb[:, 2]]
        d = (1 - fr[:, ax]) if f[ax] > 0 else fr[:, ax]
        take = open_ & (d < best)
        best[take] = d[take]
        nrm[take] = f
    deep = ~np.isfinite(best)
    if deep.any():
        val, grad = sample_sdf(phi, g[inside][deep] - 0.5)
        best[deep] = -val
        nrm[deep] = grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-12)
    depth[inside] = best
    normal[inside] = nrm
    return inside, depth, normal


# ── material numbers a body needs ──────────────────────────────────────────────────────────────────────────────────
_STRENGTH = {}


def strength_table(name):
    """``{"T", "yield", "ductile", "uts", "elongation", "source"}`` for a material: the yield stress over temperature
    (each entry a sigma-ground call): measured for carbon steel (Eurocode 3's fire table), else Johnson-Cook's thermal
    softening at zero plastic strain, quasi-static; the tensile strength for a brittle one; the elongation at fracture."""
    if name in _STRENGTH:
        return _STRENGTH[name]
    from ..field.interface.block import block
    from ..field.interface.plasticity import (PLASTICITY_DATA, STEEL_FIRE_KY_SOURCE, johnson_cook_stress,
                                              steel_yield_at, total_elongation)
    b = block(name)
    key, p = b["canonical"], b["properties"]
    T = np.arange(100.0, 4000.0 + 1, 25.0)
    ductile = bool(p.get("is_ductile", {}).get("value", False))
    out = {"T": T, "ductile": ductile, "uts": p.get("sigma_UTS", {}).get("value"), "elongation": None, "flags": []}
    if ductile and steel_yield_at(key, 293.15) is not None:
        out["yield"] = np.array([steel_yield_at(key, float(t)) for t in T])
        out["elongation"] = total_elongation(key)
        out["source"] = f"plasticity.steel_yield_at ({STEEL_FIRE_KY_SOURCE}), plasticity.total_elongation"
    elif ductile and key in PLASTICITY_DATA:
        out["yield"] = np.array([johnson_cook_stress(key, 0.0, 1.0, float(t)) for t in T])
        out["elongation"] = total_elongation(key)
        out["source"] = "plasticity.johnson_cook_stress (thermal softening), plasticity.total_elongation"
    elif ductile:
        out["ductile"] = False
        out["flags"].append(f"{name}: no Johnson-Cook data; treated as brittle at its tensile strength")
    if not out["ductile"]:
        out["yield"] = None
        out["source"] = p.get("sigma_UTS", {}).get("source", "none")
        out["flags"].append(f"{name}: brittle strength at room temperature (no law for it hot)")
        if out["uts"] is None:
            out["flags"].append(f"{name}: no tensile strength in sigma-ground; it never breaks")
    _STRENGTH[name] = out
    return out


def contact_numbers(mat_a, mat_b):
    """(static mu, kinetic mu, flags) for a pair of materials (block names)."""
    from ..field.interface.block import block
    from ..field.interface.friction import contact_friction
    ka, kb = block(mat_a)["canonical"], block(mat_b)["canonical"]
    f = contact_friction(ka, kb)
    if f is None:
        return 0.0, 0.0, [f"{mat_a}/{mat_b}: no friction data at all; frictionless"]
    return f["static"], f["kinetic"], f["flags"]


_VY = {}


def _yield_onset(m, radius_m):
    if (m, radius_m) not in _VY:
        from ..field.interface.block import block
        from ..field.interface.impact import yield_onset_velocity
        from ..field.interface.plasticity import PLASTICITY_DATA
        k = block(m)["canonical"]
        _VY[(m, radius_m)] = yield_onset_velocity(k, radius_m) if k in PLASTICITY_DATA else None
    return _VY[(m, radius_m)]


def restitution(mat_a, mat_b, speed, radius_m):
    """Johnson-Thornton restitution at this impact speed; for two materials the lower of the two (e = (v_y/v)^(1/4) above
    the yield-onset speed v_y, 1 below it; v_y cached per material)."""
    es = []
    for m in (mat_a, mat_b):
        vy = _yield_onset(m, radius_m)
        if vy is not None:
            es.append(1.0 if speed <= vy else (vy / speed) ** 0.25)
    return min(es) if es else 0.0


# ── a body ──────────────────────────────────────────────────────────────────────────────────────────────────────────
class VoxelBody:
    """``cells``: (N,3) grid cells it occupies now (axis-aligned, as built). ``cell_mass`` (N,) kg; ``T``, ``melt``,
    ``boil`` per cell. The body's own frame is its cells' offsets (cell i spans offsets[i]*h .. +h)."""

    def __init__(self, material, index, cells, pitch, cell_mass, T, melt=None, boil=None, *, frame=None):
        self.material, self.index, self.h = material, int(index), float(pitch)
        cells = np.asarray(cells, int)
        n = len(cells)
        self.cell_mass = np.asarray(cell_mass, float).copy()
        self.T = np.asarray(T, float).copy()
        self.melt = np.zeros(n) if melt is None else np.asarray(melt, float).copy()
        self.boil = np.zeros(n) if boil is None else np.asarray(boil, float).copy()
        self.v, self.w = np.zeros(3), np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.asleep = 0
        if frame is None:                                    # as built: the frame's origin is the grid's
            self.offsets = cells - cells.min(0)
            self._shape()
            self.x = cells.min(0) * self.h + self.r0
            self.grid = cells.copy()
        else:                                                # a piece of another body: its offsets, pose and motion
            parent, keep = frame
            self.offsets = parent.offsets[keep]
            self._shape()
            R = parent.R
            d = R @ (self.r0 - parent.r0)
            self.x = parent.x + d
            self.q = parent.q.copy()
            self.v = parent.v + np.cross(parent.w, d)
            self.w = parent.w.copy()
            self.grid = parent.grid[keep].copy()

    # geometry ----------------------------------------------------------------------------------------------------
    def _shape(self):
        h, m = self.h, self.cell_mass
        self.mass = float(m.sum())
        centres = (self.offsets + 0.5) * h
        self.r0 = (m[:, None] * centres).sum(0) / self.mass
        self.r = centres - self.r0
        r = self.r
        I = np.eye(3) * float((m * (r * r).sum(1)).sum()) - (m[:, None, None] * r[:, :, None] * r[:, None, :]).sum(0)
        I += np.eye(3) * self.mass * h * h / 6.0
        self.I_body = I
        self.I_inv_body = np.linalg.inv(I)
        # voxmap: occupancy and cell ids, padded by one empty layer
        dims = self.offsets.max(0) + 3
        occ = np.zeros(dims, bool)
        ids = np.full(dims, -1, int)
        o = self.offsets + 1
        occ[tuple(o.T)] = True
        ids[tuple(o.T)] = np.arange(len(o))
        self.occ, self.ids = occ, ids
        self.phi = signed_distance(occ)
        from scipy.ndimage import distance_transform_edt as edt
        _, near = edt(~occ, return_indices=True)
        self.nearest = ids[near[0], near[1], near[2]]            # nearest cell of the body to any voxmap cell
        # point shell: corners of exposed faces, each with the cell it belongs to
        pts, own = [], []
        corners = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])
        for f in FACES:
            nb = o + f
            exposed = ~occ[tuple(nb.T)]
            ax = int(np.nonzero(f)[0][0])
            on = corners[corners[:, ax] == (1 if f[ax] > 0 else 0)]
            for c in on:
                pts.append(self.offsets[exposed] + c)
                own.append(np.nonzero(exposed)[0])
        pts = np.concatenate(pts)
        own = np.concatenate(own)
        key, first = np.unique(pts, axis=0, return_index=True)
        self.shell = key * h - self.r0                        # in the body frame, about the centre of mass
        self.shell_owner = own[first]
        self.radius = float(np.sqrt((self.shell ** 2).sum(1)).max())

    @property
    def R(self):
        return quat_to_R(self.q)

    def I_inv(self):
        R = self.R
        return R @ self.I_inv_body @ R.T

    def world(self, local_about_com):
        return self.x + local_about_com @ self.R.T

    def to_local_index(self, p):
        """World points to this body's voxmap index space (cell centre i at i)."""
        local = (p - self.x) @ self.R + self.r0
        return local / self.h - 0.5 + 1

    def cell_at(self, p):
        """The body's cell nearest each world point."""
        idx = np.clip(np.round(self.to_local_index(p)).astype(int), 0, np.array(self.occ.shape) - 1)
        return self.nearest[idx[:, 0], idx[:, 1], idx[:, 2]]

    def point_velocity(self, p):
        return self.v + np.cross(self.w, p - self.x)

    def kinetic_J(self):
        return 0.5 * self.mass * float(self.v @ self.v) + 0.5 * float(self.w @ (self.R @ self.I_body @ self.R.T) @ self.w)

    def centres(self):
        return self.world(self.r)

    def piece(self, keep):
        """A new body of the cells in ``keep`` (bool mask), with this body's pose and motion."""
        return VoxelBody(self.material, self.index, self.grid[keep], self.h, self.cell_mass[keep], self.T[keep],
                         self.melt[keep], self.boil[keep], frame=(self, keep))

    def components(self, keep=None):
        """Face-connected groups of the cells (all, or those in ``keep``): a list of bool masks."""
        from scipy.ndimage import label
        keep = np.ones(len(self.offsets), bool) if keep is None else keep
        occ = np.zeros(self.occ.shape, bool)
        o = self.offsets[keep] + 1
        occ[tuple(o.T)] = True
        lab, n = label(occ)
        which = lab[tuple((self.offsets + 1).T)]
        return [(which == k) & keep for k in range(1, n + 1)]


# ── the contact solver (sequential impulses) ───────────────────────────────────────────────────────────────────────
@njit(cache=True)
def _cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


@njit(cache=True)
def _solve(inv_m, inv_I, v, w, vp, wp, ca, cb, ra, rb, n, t1, t2, depth, mu_s, mu_k, e_bias, slip0,
           ja, jb, jra, jrb, jC, jtau, dt, iters, beta, slop, lam_n0, lam_t0):
    """Velocities in place. Returns per-contact normal and tangential impulses, per-joint linear and angular ones.
    ``lam_n0``/``lam_t0``: last step's impulses at the same contacts (warm start)."""
    nc = len(ca)
    nj = len(ja)
    lam_n = lam_n0.copy()
    lam_t = lam_t0.copy()
    lam_p = np.zeros(nc)
    jl = np.zeros((nj, 3))
    jang = np.zeros((nj, 3))
    kn = np.zeros(nc)
    kt = np.zeros((nc, 2))
    for c in range(nc):
        a, b = ca[c], cb[c]
        for k in range(3):
            d = n[c] if k == 0 else (t1[c] if k == 1 else t2[c])
            ka = 0.0
            if a >= 0:
                rn = _cross(ra[c], d)
                ka += inv_m[a] + rn @ (inv_I[a] @ rn)
            if b >= 0:
                rn = _cross(rb[c], d)
                ka += inv_m[b] + rn @ (inv_I[b] @ rn)
            if k == 0:
                kn[c] = ka
            else:
                kt[c, k - 1] = ka
    # joint effective masses
    Kl = np.zeros((nj, 3, 3))
    Ka = np.zeros((nj, 3, 3))
    for j in range(nj):
        a, b = ja[j], jb[j]
        K = np.zeros((3, 3))
        A = np.zeros((3, 3))
        for s in range(2):
            i = a if s == 0 else b
            r = jra[j] if s == 0 else jrb[j]
            if i < 0:
                continue
            Rx = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])
            K += np.eye(3) * inv_m[i] - Rx @ inv_I[i] @ Rx
            A += inv_I[i]
        Kl[j] = np.linalg.inv(K)
        Ka[j] = np.linalg.inv(A + np.eye(3) * 1e-30)
    for c in range(nc):                                          # warm start
        a, b = ca[c], cb[c]
        P = n[c] * lam_n[c] + t1[c] * lam_t[c, 0] + t2[c] * lam_t[c, 1]
        if a >= 0:
            v[a] += inv_m[a] * P
            w[a] += inv_I[a] @ _cross(ra[c], P)
        if b >= 0:
            v[b] -= inv_m[b] * P
            w[b] -= inv_I[b] @ _cross(rb[c], P)
    for it in range(iters):
        for j in range(nj):
            a, b = ja[j], jb[j]
            dv = np.zeros(3)
            dw = np.zeros(3)
            if a >= 0:
                dv += v[a] + _cross(w[a], jra[j])
                dw += w[a]
            if b >= 0:
                dv -= v[b] + _cross(w[b], jrb[j])
                dw -= w[b]
            # bending/torsion: resisted up to the plastic moment
            P = -(Ka[j] @ dw)
            old = jang[j].copy()
            nw = old + P
            mag = math.sqrt(nw @ nw)
            cap = jtau[j] * dt
            if mag > cap:
                nw = nw * (cap / mag)
            P = nw - old
            jang[j] = nw
            if a >= 0:
                w[a] += inv_I[a] @ P
            if b >= 0:
                w[b] -= inv_I[b] @ P
            # the ball joint
            dv = np.zeros(3)
            if a >= 0:
                dv += v[a] + _cross(w[a], jra[j])
            if b >= 0:
                dv -= v[b] + _cross(w[b], jrb[j])
            L = -(Kl[j] @ dv)
            jl[j] += L
            if a >= 0:
                v[a] += inv_m[a] * L
                w[a] += inv_I[a] @ _cross(jra[j], L)
            if b >= 0:
                v[b] -= inv_m[b] * L
                w[b] -= inv_I[b] @ _cross(jrb[j], L)
        for c in range(nc):
            a, b = ca[c], cb[c]
            vr = np.zeros(3)
            if a >= 0:
                vr += v[a] + _cross(w[a], ra[c])
            if b >= 0:
                vr -= v[b] + _cross(w[b], rb[c])
            vn = vr @ n[c]
            dl = -(vn - e_bias[c]) / kn[c]
            new = max(lam_n[c] + dl, 0.0)
            dl = new - lam_n[c]
            lam_n[c] = new
            Pn = n[c] * dl
            # friction, circular cone: static below the slip speed the contact started at, kinetic above
            mu = mu_s[c] if slip0[c] < 1e-3 else mu_k[c]
            vt1 = vr @ t1[c]
            vt2 = vr @ t2[c]
            o1, o2 = lam_t[c, 0], lam_t[c, 1]
            n1 = o1 - vt1 / kt[c, 0]
            n2 = o2 - vt2 / kt[c, 1]
            cap = mu * lam_n[c]
            m2 = math.sqrt(n1 * n1 + n2 * n2)
            if m2 > cap:
                n1 *= cap / m2
                n2 *= cap / m2
            lam_t[c, 0], lam_t[c, 1] = n1, n2
            P = Pn + t1[c] * (n1 - o1) + t2[c] * (n2 - o2)
            if a >= 0:
                v[a] += inv_m[a] * P
                w[a] += inv_I[a] @ _cross(ra[c], P)
            if b >= 0:
                v[b] -= inv_m[b] * P
                w[b] -= inv_I[b] @ _cross(rb[c], P)
    # split impulses: penetration and joint drift removed by pseudo-velocities (positions only, no kinetic energy)
    for it in range(iters):
        for j in range(nj):
            a, b = ja[j], jb[j]
            dv = np.zeros(3)
            if a >= 0:
                dv += vp[a] + _cross(wp[a], jra[j])
            if b >= 0:
                dv -= vp[b] + _cross(wp[b], jrb[j])
            L = -(Kl[j] @ (dv + beta * jC[j] / dt))
            if a >= 0:
                vp[a] += inv_m[a] * L
                wp[a] += inv_I[a] @ _cross(jra[j], L)
            if b >= 0:
                vp[b] -= inv_m[b] * L
                wp[b] -= inv_I[b] @ _cross(jrb[j], L)
        for c in range(nc):
            a, b = ca[c], cb[c]
            vr = np.zeros(3)
            if a >= 0:
                vr += vp[a] + _cross(wp[a], ra[c])
            if b >= 0:
                vr -= vp[b] + _cross(wp[b], rb[c])
            target = beta * max(depth[c] - slop, 0.0) / dt
            dl = (target - vr @ n[c]) / kn[c]
            new = max(lam_p[c] + dl, 0.0)
            dl = new - lam_p[c]
            lam_p[c] = new
            P = n[c] * dl
            if a >= 0:
                vp[a] += inv_m[a] * P
                wp[a] += inv_I[a] @ _cross(ra[c], P)
            if b >= 0:
                vp[b] -= inv_m[b] * P
                wp[b] -= inv_I[b] @ _cross(rb[c], P)
    return lam_n, lam_t, jl, jang


# ── the world of bodies ────────────────────────────────────────────────────────────────────────────────────────────
class RigidWorld:
    """Bodies among static walls. ``static``: bool grid of wall cells (refresh with ``set_static``). ``material_of``:
    a function grid cell -> material name for the walls (for friction). Grid cell i spans i*h .. (i+1)*h."""

    def __init__(self, shape, pitch, static, static_material, gravity, *, iters=24, max_dt=1 / 400, beta=0.2):
        self.shape, self.h = tuple(shape), float(pitch)
        self.g = float(gravity)
        self.iters, self.max_dt, self.beta = int(iters), float(max_dt), float(beta)
        self.slop = 0.02 * self.h
        self.bodies, self.joints = [], []
        self.static_material = static_material
        self.set_static(static)
        self.ledger = {"contact_heat_J": 0.0, "plastic_heat_J": 0.0, "integration_J": 0.0, "hinges": 0, "tears": 0,
                       "fractures": 0}
        self.heat = []                        # (body or None, cell index or grid cell, joules) since last collected
        self.flags = set()
        self._mu = {}
        self._impulse = {}                   # id(body) -> (N,3) impulse per cell this tick
        self._warm = {}                      # contact key -> last step's (normal, tangential) impulses
        self._elapsed = 0.0

    def set_static(self, static):
        static = np.asarray(static, bool)
        key = hash(static.tobytes())
        if getattr(self, "_static_key", None) == key:
            return
        self._static_key = key
        self.static = static
        pad = np.pad(static, 1, constant_values=True)          # outside the grid is a wall
        self.occ_static = pad
        self.phi_static = signed_distance(pad)

    # pairs --------------------------------------------------------------------------------------------------------
    def _friction(self, ma, mb):
        k = (ma, mb) if ma <= mb else (mb, ma)
        if k not in self._mu:
            s, kk, flags = contact_numbers(*k)
            self._mu[k] = (s, kk)
            self.flags.update(flags)
        return self._mu[k]

    def _contacts(self):
        """Contact points: (a, b, p, n, depth, cell_a, cell_b) with b = -1 for a wall (cell_b then a flat grid index)."""
        out = []
        h = self.h
        shells = [b.world(b.shell) for b in self.bodies]
        for i, b in enumerate(self.bodies):
            P = shells[i]
            hit, d, nrm = voxel_penetration(self.occ_static, self.phi_static, P / h + 1)
            if hit.any():
                p = P[hit]
                wc = np.clip(np.floor(p / h - 0.5 * nrm[hit]).astype(int), 0, np.array(self.shape) - 1)
                out.append((i, -1, p, nrm[hit], d[hit] * h, b.shell_owner[hit], np.ravel_multi_index(wc.T, self.shape),
                            np.nonzero(hit)[0]))
        for i in range(len(self.bodies)):
            for j in range(i + 1, len(self.bodies)):
                A, B = self.bodies[i], self.bodies[j]
                if np.linalg.norm(A.x - B.x) > A.radius + B.radius + h:
                    continue
                for (s, o, P, own) in ((i, j, shells[i], A.shell_owner), (j, i, shells[j], B.shell_owner)):
                    O = self.bodies[o]
                    hit, d, nrm = voxel_penetration(O.occ, O.phi, O.to_local_index(P) + 0.5)
                    if not hit.any():
                        continue
                    p = P[hit]
                    out.append((s, o, p, nrm[hit] @ O.R.T, d[hit] * h, own[hit], O.cell_at(p), np.nonzero(hit)[0]))
        return out

    # stepping -----------------------------------------------------------------------------------------------------
    def max_speed(self):
        return max([float(np.linalg.norm(b.v)) + float(np.linalg.norm(b.w)) * b.radius for b in self.bodies] + [0.0])

    def mechanical_J(self):
        E = 0.0
        for b in self.bodies:
            E += b.kinetic_J() + b.mass * self.g * float(b.x[2])
        return E

    def step(self, dt, external=None):
        """Advance ``dt``: gravity (and ``external[i]`` = (force, torque) on body i, e.g. a liquid's pressure), contact
        and the hinges, in sub-steps of at most ``max_dt`` and a quarter cell of motion."""
        left = float(dt)
        while left > 1e-12:
            sub = min(left, self.max_dt, 0.25 * self.h / max(self.max_speed(), 1e-9))
            self._substep(sub, external)
            left -= sub

    def _substep(self, dt, external):
        bodies = self.bodies
        nb = len(bodies)
        if not nb:
            return
        E0 = self.mechanical_J()
        inv_m = np.array([1.0 / b.mass for b in bodies])
        inv_I = np.array([b.I_inv() for b in bodies])
        v = np.array([b.v for b in bodies])
        w = np.array([b.w for b in bodies])
        for i, b in enumerate(bodies):
            F = np.array([0.0, 0.0, -self.g * b.mass])
            tau = np.zeros(3)
            if external is not None and external[i] is not None:
                F = F + external[i][0]
                tau = tau + external[i][1]
            Iw = b.R @ b.I_body @ b.R.T @ w[i]
            v[i] += F * inv_m[i] * dt
            w[i] += inv_I[i] @ (tau - np.cross(w[i], Iw)) * dt
        con = self._contacts()
        if con:
            ca = np.concatenate([np.full(len(c[2]), c[0]) for c in con])
            cb = np.concatenate([np.full(len(c[2]), c[1]) for c in con])
            p = np.concatenate([c[2] for c in con])
            n = np.concatenate([c[3] for c in con])
            depth = np.concatenate([c[4] for c in con])
            cell_a = np.concatenate([c[5] for c in con])
            cell_b = np.concatenate([c[6] for c in con])
            keys = [(id(bodies[c[0]]), id(bodies[c[1]]) if c[1] >= 0 else -1, int(k)) for c in con for k in c[7]]
        else:
            keys = []
            ca = cb = cell_a = cell_b = np.zeros(0, int)
            p = n = np.zeros((0, 3))
            depth = np.zeros(0)
        nc = len(ca)
        X = np.array([b.x for b in bodies])
        ra = p - X[ca] if nc else np.zeros((0, 3))
        rb = np.where((cb >= 0)[:, None], p - X[np.maximum(cb, 0)], 0.0) if nc else np.zeros((0, 3))
        # tangents
        t1 = np.cross(n, np.where(np.abs(n[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])) if nc else np.zeros((0, 3))
        t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-12) if nc else 1
        t2 = np.cross(n, t1)
        mu_s, mu_k, e_bias, slip0 = np.zeros(nc), np.zeros(nc), np.zeros(nc), np.zeros(nc)
        for c in range(nc):
            a, b = ca[c], cb[c]
            ma = bodies[a].material
            mb = bodies[b].material if b >= 0 else self.static_material(cell_b[c])
            mu_s[c], mu_k[c] = self._friction(ma, mb)
            vr = v[a] + np.cross(w[a], ra[c])
            if b >= 0:
                vr = vr - (v[b] + np.cross(w[b], rb[c]))
            vn = float(vr @ n[c])
            slip0[c] = float(np.linalg.norm(vr - vn * n[c]))
            if vn < -2 * self.g * dt:                       # an impact, not resting contact: it bounces
                e = restitution(ma, mb, -vn, 0.5 * self.h)
                e_bias[c] = -e * vn
        # joints
        nj = len(self.joints)
        ja = np.array([bodies.index(J["a"]) for J in self.joints], dtype=np.int64)
        jb = np.array([bodies.index(J["b"]) for J in self.joints], dtype=np.int64)
        jra = np.zeros((nj, 3))
        jrb = np.zeros((nj, 3))
        jC = np.zeros((nj, 3))
        jtau = np.zeros(nj)
        for k, J in enumerate(self.joints):
            pa = J["a"].world(J["pa"][None])[0]
            pb = J["b"].world(J["pb"][None])[0]
            jra[k], jrb[k], jC[k] = pa - J["a"].x, pb - J["b"].x, pa - pb
            jtau[k] = J["Mp"]
        vp, wp = np.zeros_like(v), np.zeros_like(w)
        lam_n0, lam_t0 = np.zeros(nc), np.zeros((nc, 2))
        for c, k in enumerate(keys):
            old = self._warm.get(k)
            if old is not None:
                lam_n0[c] = old[0]
                lam_t0[c] = old[1]
        lam_n, lam_t, jl, jang = _solve(inv_m, inv_I, v, w, vp, wp, ca.astype(np.int64), cb.astype(np.int64), ra, rb,
                                        n, t1, t2, depth, mu_s, mu_k, e_bias, slip0, ja, jb, jra, jrb, jC, jtau,
                                        dt, self.iters, self.beta, self.slop, lam_n0, lam_t0)
        self._warm = {k: (lam_n[c], lam_t[c].copy()) for c, k in enumerate(keys)}
        # move
        for i, b in enumerate(bodies):
            b.v, b.w = v[i], w[i]
            b.x = b.x + (v[i] + vp[i]) * dt
            b.q = quat_mul(quat_rotvec((w[i] + wp[i]) * dt), b.q)
            b.q /= np.linalg.norm(b.q)
        # where the impulses landed (for the strength check) and where the lost energy goes (heat)
        J = n * lam_n[:, None] + t1 * lam_t[:, :1] + t2 * lam_t[:, 1:] if nc else np.zeros((0, 3))
        for c in range(nc):
            self._impulse.setdefault(id(bodies[ca[c]]), np.zeros((len(bodies[ca[c]].r), 3)))[cell_a[c]] += J[c]
            if cb[c] >= 0:
                self._impulse.setdefault(id(bodies[cb[c]]), np.zeros((len(bodies[cb[c]].r), 3)))[cell_b[c]] -= J[c]
        for k, Jt in enumerate(self.joints):
            for side, sign in (("a", 1.0), ("b", -1.0)):
                B = Jt[side]
                imp = self._impulse.setdefault(id(B), np.zeros((len(B.r), 3)))
                cells = Jt["cells_" + side]
                imp[cells] += sign * jl[k] / len(cells)
        self._elapsed += dt
        # plastic work at the hinges: torque times the bend it allowed this step
        plastic = 0.0
        for k, Jt in enumerate(self.joints):
            wr = Jt["a"].w - Jt["b"].w
            work = abs(float(jang[k] @ wr))
            if work > 0:
                plastic += work
                half = 0.5 * work
                self.heat.append((Jt["a"], Jt["cells_a"], half))
                self.heat.append((Jt["b"], Jt["cells_b"], half))
        self.ledger["plastic_heat_J"] += plastic
        lost = E0 - self.mechanical_J() - plastic
        if lost > 0:
            self.ledger["contact_heat_J"] += lost
            if nc:
                wgt = lam_n + np.linalg.norm(lam_t, axis=1)
                wgt = wgt / wgt.sum() if wgt.sum() > 0 else np.full(nc, 1.0 / nc)
                for c in range(nc):
                    if wgt[c] <= 0:
                        continue
                    share = lost * wgt[c]
                    self.heat.append((bodies[ca[c]], np.array([cell_a[c]]), 0.5 * share))
                    if cb[c] >= 0:
                        self.heat.append((bodies[cb[c]], np.array([cell_b[c]]), 0.5 * share))
                    else:
                        self.heat.append((None, np.array([cell_b[c]]), 0.5 * share))
            else:
                self.ledger["contact_heat_J"] -= lost         # nothing touched: the integrator's own error, not heat
                self.ledger["integration_J"] += lost
        elif lost < 0:
            self.ledger["integration_J"] += lost             # energy the step gained (penetration pushed out upward)
        for k, Jt in enumerate(self.joints):
            Jt["bend"] = quat_angle(quat_mul(Jt["q0"], quat_mul(_conj(Jt["b"].q), Jt["a"].q)))

    def collect_heat(self):
        out, self.heat = self.heat, []
        return out

    # drawing ------------------------------------------------------------------------------------------------------
    def raster(self, blocked):
        """Each body's cells onto the grid, one grid cell per body cell (its centre's cell; a collision goes to the
        nearest free neighbour), so the drawing holds exactly each body's mass. ``blocked``: cells already taken."""
        taken = np.asarray(blocked, bool).copy()
        hi = np.array(self.shape) - 1
        nbr = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)])
        for b in self.bodies:
            C = b.centres() / self.h
            idx = np.clip(np.floor(C).astype(int), 0, hi)
            flat = np.ravel_multi_index(idx.T, self.shape)
            order = np.argsort(np.linalg.norm(C - idx - 0.5, axis=1))     # cells nearest their centre's cell first
            grid = idx.copy()
            _, first = np.unique(flat[order], return_index=True)
            clean = np.zeros(len(flat), bool)
            clean[order[first]] = True
            clean &= ~taken.ravel()[flat]
            taken.ravel()[flat[clean]] = True
            for k in order[~clean[order]]:                                # the few that collide
                cand = np.clip(idx[k] + nbr, 0, hi)
                free = ~taken[tuple(cand.T)]
                if free.any():
                    cand = cand[free]
                    grid[k] = cand[np.argmin(np.linalg.norm(cand + 0.5 - C[k], axis=1))]
                taken[tuple(grid[k])] = True                              # nowhere free: overlap (rare, transient)
            b.grid = grid
        return taken

    # strength -----------------------------------------------------------------------------------------------------
    def sections(self, body):
        """Every cut through ``body``: list of dicts (axis, layer, utilisation, faces (i, j cell pairs), centroid,
        bending direction, ductile). Forces: the impulses since the last check, plus weight, less mass x acceleration."""
        imp = self._impulse.get(id(body))
        dt = max(self._elapsed, 1e-12)
        f = np.zeros((len(body.r), 3)) if imp is None else imp / dt
        f = f + body.cell_mass[:, None] * np.array([0.0, 0.0, -self.g])
        R = body.R
        f = f @ R                                                   # into the body frame
        r = body.r
        M = body.mass
        Ftot = f.sum(0)
        a = Ftot / M
        tau = np.cross(r, f).sum(0)
        alpha = body.I_inv_body @ tau
        f = f - body.cell_mass[:, None] * (a + np.cross(alpha, r))   # balanced: what the cuts must carry
        st = strength_table(body.material)
        T = body.T
        if st["ductile"]:
            sy = np.interp(T, st["T"], st["yield"]) * np.clip(1 - body.melt, 0, 1)
        else:
            sy = np.full(len(T), np.inf if st["uts"] is None else st["uts"]) * np.clip(1 - body.melt, 0, 1)
        h = self.h
        A = h * h
        out = []
        o = body.offsets
        for ax in range(3):
            nb = o + np.eye(3, dtype=int)[ax] + 1
            j = body.ids[tuple(nb.T)]
            bond = j >= 0
            if not bond.any():
                continue
            bi, bj = np.nonzero(bond)[0], j[bond]
            layers = o[:, ax]
            # side S of the cut after layer k: cells with layer <= k
            for k in np.unique(o[bi, ax]):
                S = layers <= k
                fs = f[S].sum(0)
                faces = o[bi, ax] == k
                fi, fj = bi[faces], bj[faces]
                pos = 0.5 * (r[fi] + r[fj])                         # face centres
                c = pos.mean(0)
                ms = np.cross(r[S] - c, f[S]).sum(0)
                F_int, M_int = -fs, -ms
                e = np.eye(3)[ax]
                N = float(F_int @ e)
                V = F_int - N * e
                Tq = float(M_int @ e)
                Mb = M_int - Tq * e
                rho = pos - c
                s_face = np.minimum(sy[fi], sy[fj])
                nf = len(fi)
                Atot = nf * A
                J = float(A * (rho * rho).sum()) + nf * A * h * h / 6
                rmax = float(np.sqrt((rho * rho).sum(1)).max()) + 0.5 * h
                tau_s = float(np.linalg.norm(V)) / Atot + abs(Tq) * rmax / J
                if st["ductile"]:
                    red = np.sqrt(np.clip(1 - 3 * (tau_s / np.maximum(s_face, 1e-30)) ** 2, 0, 1))
                    cap = s_face * red * A
                    mb = float(np.linalg.norm(Mb))
                    C = float(cap.sum())
                    if abs(N) >= C or C <= 0:
                        U = np.inf
                        mhat = Mb / mb if mb > 0 else np.cross(e, [1.0, 0, 0]) if ax else np.array([0, 1.0, 0])
                    elif mb <= 1e-12:
                        U = abs(N) / C
                        mhat = np.cross(e, np.eye(3)[(ax + 1) % 3])
                    else:
                        mhat = Mb / mb
                        Mp = _plastic_moment(*_subfaces(rho @ np.cross(e, mhat), cap, h), N)
                        U = mb / Mp if Mp > 0 else np.inf
                else:
                    # elastic: sigma = s0 + s . rho (in plane); N and the bending moment
                    u = np.cross(e, np.eye(3)[(ax + 1) % 3])
                    wv = np.cross(e, u)
                    pu, pw = rho @ u, rho @ wv
                    G = np.array([[nf, pu.sum(), pw.sum()],
                                  [pu.sum(), (pu * pu).sum() + nf * h * h / 12, (pu * pw).sum()],
                                  [pw.sum(), (pu * pw).sum(), (pw * pw).sum() + nf * h * h / 12]]) * A
                    # the stresses' moment about the centroid is sum sigma A (rho x e), with u x w = e:
                    # its u part is sum sigma A pw, its w part -sum sigma A pu
                    rhs = np.array([N, -float(Mb @ wv), float(Mb @ u)])
                    s = np.linalg.solve(G, rhs)
                    sig = s[0] + s[1] * pu + s[2] * pw
                    sig1 = 0.5 * sig + np.sqrt(0.25 * sig * sig + tau_s * tau_s)
                    U = float((sig1 / s_face).max())
                    mhat = Mb / (np.linalg.norm(Mb) + 1e-30)
                out.append({"axis": ax, "layer": int(k), "U": float(U), "fi": fi, "fj": fj, "centroid": c,
                            "mhat": mhat, "ductile": st["ductile"]})
        return out

    def check_strength(self):
        """Once a tick: the weakest cut of each body, if it has failed, becomes a hinge (ductile) or a break (brittle);
        a hinge past its fracture strain tears. Then the impulses start over."""
        changed = False
        for Jt in list(self.joints):
            st = strength_table(Jt["a"].material)
            Jt["Mp"] = self._hinge_moment(Jt)
            if st["elongation"] is not None and Jt["bend"] / 2 >= st["elongation"]:
                self.joints.remove(Jt)
                self.ledger["tears"] += 1
                changed = True
        for body in list(self.bodies):
            if len(body.r) < 2:
                continue
            cuts = self.sections(body)
            if not cuts:
                continue
            worst = max(cuts, key=lambda s: s["U"])
            if worst["U"] < 1:
                continue
            ax, k = worst["axis"], worst["layer"]
            S = body.offsets[:, ax] <= k
            pieces = [m for side in (S, ~S) for m in body.components(side)]
            new = [body.piece(m) for m in pieces]
            i = self.bodies.index(body)
            self.bodies[i:i + 1] = new
            for Jt in self.joints:                               # joints that held this body move to its pieces
                for side in ("a", "b"):
                    if Jt[side] is body:
                        cells = Jt["cells_" + side]
                        for m, nb in zip(pieces, new):
                            if m[cells].all():
                                remap = np.cumsum(m) - 1
                                Jt[side], Jt["cells_" + side] = nb, remap[cells]
                                Jt["p" + side] = Jt["p" + side] + body.r0 - nb.r0
            if worst["ductile"]:
                self.ledger["hinges"] += 1
                for ma, A in zip(pieces, new):
                    for mb, B in zip(pieces, new):
                        if A is B or not (S[ma].all() and (~S)[mb].all()):
                            continue
                        link = ma[worst["fi"]] & mb[worst["fj"]]
                        if not link.any():
                            continue
                        ra_, rb_ = np.cumsum(ma) - 1, np.cumsum(mb) - 1
                        anchor = 0.5 * (body.r[worst["fi"][link]] + body.r[worst["fj"][link]]).mean(0) + body.r0
                        Jt = {"a": A, "b": B, "pa": anchor - A.r0, "pb": anchor - B.r0,
                              "cells_a": ra_[worst["fi"][link]], "cells_b": rb_[worst["fj"][link]],
                              "axis": ax, "mhat_a": worst["mhat"], "bend": 0.0,
                              "q0": quat_mul(_conj(A.q), B.q)}
                        Jt["Mp"] = self._hinge_moment(Jt)
                        self.joints.append(Jt)
            else:
                self.ledger["fractures"] += 1
            changed = True
        self._impulse = {}
        self._elapsed = 0.0
        return changed

    def _hinge_moment(self, Jt):
        """The plastic moment the hinge carries now: its faces at their cells' current yield stress."""
        A, B = Jt["a"], Jt["b"]
        st = strength_table(A.material)
        if not st["ductile"]:
            return 0.0
        sa = np.interp(A.T[Jt["cells_a"]], st["T"], st["yield"]) * np.clip(1 - A.melt[Jt["cells_a"]], 0, 1)
        sb = np.interp(B.T[Jt["cells_b"]], st["T"], st["yield"]) * np.clip(1 - B.melt[Jt["cells_b"]], 0, 1)
        s = np.minimum(sa, sb)
        pos = A.r[Jt["cells_a"]] @ np.eye(3)                      # A's frame
        e = np.eye(3)[Jt["axis"]]
        rho = pos - pos.mean(0)
        d = rho @ np.cross(e, Jt["mhat_a"])
        return _plastic_moment(*_subfaces(d, s * self.h * self.h, self.h), 0.0)

    def gaps(self):
        g = ["elastic deflection before yield is not drawn; a cut's failure is judged rigid-body (statics of each side)",
             "a hinge resists torsion with its bending capacity; brittle materials fail in tension only, at room "
             "temperature strength", "static walls are infinitely strong",
             "the liquid's pressure moves bodies but is not in the strength check"]
        return g + sorted(self.flags) + sorted(f for b in self.bodies for f in strength_table(b.material)["flags"])


def _conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _subfaces(d, cap, h, k=8):
    """Each face as k strips across the lever direction, so a single cell's own depth carries a moment (a 1-cell bar's
    plastic moment is sigma_y h^3 / 4, not zero). Exact for a lever along a face edge."""
    off = ((np.arange(k) + 0.5) / k - 0.5) * h
    return (d[:, None] + off[None]).ravel(), np.repeat(cap / k, k)


def _plastic_moment(d, cap, N):
    """Fully plastic moment of faces at lever ``d`` (along the bending's tension direction) each carrying ``cap`` (N):
    the faces furthest out in tension, the rest in compression, split so the net is ``N``."""
    order = np.argsort(-d)
    d, cap = d[order], cap[order]
    C = cap.sum()
    want = 0.5 * (N + C)                                      # force carried in tension
    cum = np.cumsum(cap)
    k = int(np.searchsorted(cum, want))
    sign = np.where(np.arange(len(d)) < k, 1.0, -1.0)
    if k < len(d):
        frac = (want - (cum[k - 1] if k else 0.0)) / cap[k] if cap[k] > 0 else 0.0
        sign[k] = 2 * frac - 1
    return float((sign * cap * d).sum())
