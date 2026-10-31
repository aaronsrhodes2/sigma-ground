"""
FLIP: an incompressible liquid with a free surface, as particles that carry momentum through a voxel grid.

The liquid is a cloud of particles, each with a position and a velocity (its share of the liquid's mass and momentum).
Every step:

  1. particles → grid: their momentum is spread onto a staggered (MAC) grid, one velocity component per cell face;
  2. body forces: gravity, and the liquid's own viscosity (explicit diffusion, the measured viscosity);
  3. walls: every face touching a solid takes that solid's velocity (zero for a wall, the paddle's for a paddle);
  4. projection: one pressure per liquid cell, solved so the velocity has no divergence (the liquid neither squeezes
     nor stretches; this is the incompressible limit, exact, no artificial sound speed). Air cells hold zero pressure,
     so the free surface is wherever the liquid is: waves, splashes and droplets are emergent;
  5. grid → particles: each particle takes the grid's CHANGE of velocity (FLIP, low numerical damping) blended with a
     little of the grid's velocity itself (PIC, for stability);
  6. particles move through the new velocity field (midpoint rule) and are pushed out of any solid they ended up in.

Pressure is in pascals (ρ is the liquid's density), so the walls' and paddle's forces, the work the paddle does and
the hydrostatic pressure at the floor are all readable. Mass is exact (particles are never made or lost). The axes are
sigma-ground's: z is up, gravity is −z.

References: Zhu & Bridson, "Animating Sand as a Fluid", SIGGRAPH 2005 (FLIP for fluids); Bridson, *Fluid Simulation for
Computer Graphics*, 2nd ed. (2015), ch. 4-7 (MAC grid, pressure projection, free surface, extrapolation).
"""
from __future__ import annotations

import numpy as np

try:                                                    # compiled transfers when numba is installed, else NumPy
    from . import _flip_numba as _fast
except ImportError:
    _fast = None

_OFFSETS = (np.array([0.0, 0.5, 0.5]), np.array([0.5, 0.0, 0.5]), np.array([0.5, 0.5, 0.0]))


class Flip:
    """A liquid on a grid of ``shape`` cells of side ``h`` m.

    ``density`` kg/m³ and ``viscosity`` Pa·s are the liquid's (from ``field.interface.block``); ``gravity`` m/s² points
    down −z. ``solid`` is a bool grid of walls; ``liquid`` a bool grid of the cells that start full (``ppc`` particles
    each, jittered). ``flip_ratio``: how much of each particle's velocity update is FLIP (the rest PIC).
    """

    def __init__(self, shape, h, *, density, viscosity, gravity, solid, liquid, ppc=8, flip_ratio=0.97, seed=1):
        self.shape, self.h = tuple(int(n) for n in shape), float(h)
        self.rho, self.mu, self.g = float(density), float(viscosity), float(gravity)
        self.flip_ratio = float(flip_ratio)
        self.solid = np.asarray(solid, bool).copy()
        self.solid_vel = np.zeros(self.shape + (3,))
        self.t = 0.0
        rng = np.random.default_rng(seed)
        cells = np.argwhere(np.asarray(liquid, bool) & ~self.solid)
        per = round(ppc ** (1 / 3))                                   # stratified: per³ sub-cells per cell, one each
        sub = (np.stack(np.meshgrid(*[np.arange(per)] * 3, indexing="ij"), -1).reshape(-1, 3) + 0.5) / per
        jitter = (rng.random((len(cells) * len(sub), 3)) - 0.5) / per
        self.pos = ((cells[:, None, :] + sub[None]).reshape(-1, 3) + jitter) * self.h
        self.vel = np.zeros_like(self.pos)
        self.ppc = len(sub)
        self.particle_mass = self.rho * self.h ** 3 / self.ppc
        self.pressure = np.zeros(self.shape)
        self.ledger = {"gravity_impulse_Ns": np.zeros(3), "solid_force_N": np.zeros(3), "paddle_work_J": 0.0,
                       "steps": 0, "cg_iterations": 0}
        self._last_p = None

    # ── public ─────────────────────────────────────────────────────────────────────────────────────────────────────
    def set_solid(self, solid, velocity=None):
        """Move the solids: a new wall mask and each solid cell's velocity (m/s, shape + (3,)); None = all at rest."""
        self.solid = np.asarray(solid, bool).copy()
        self.solid_vel = np.zeros(self.shape + (3,)) if velocity is None else np.asarray(velocity, float)

    def step(self, dt, cfl=1.0):
        """Advance ``dt`` seconds, in as many sub-steps as the CFL limit (``cfl`` cells per sub-step) needs."""
        left = float(dt)
        while left > 1e-12:
            vmax = float(np.sqrt((self.vel ** 2).sum(1)).max()) if len(self.vel) else 0.0
            vmax = max(vmax, float(np.abs(self.solid_vel).max()), np.sqrt(2 * self.g * self.h))
            sub = min(left, cfl * self.h / vmax)
            self._substep(sub)
            left -= sub

    def momentum(self):
        """Total linear momentum of the liquid, kg·m/s."""
        return self.particle_mass * self.vel.sum(0)

    def force_on(self, mask):
        """The liquid's pressure force (N) on the solid cells in ``mask``, from the last step's pressure: pressure x face
        area on every face between one of them and a liquid cell, pushing into the solid. The pressure is carried from the
        liquid cell's centre to the face itself (linearly, from the next liquid cell out), since the half cell between them
        matters: on a cube under still water this is exactly Archimedes' rho g V."""
        F = np.zeros(3)
        liquid = getattr(self, "_liquid", None)
        if liquid is None:
            return F
        P, M = self.pressure, np.asarray(mask, bool)

        def shift(a, step, ax, fill):                                 # value of the neighbour at index + step
            out = np.full_like(a, fill)
            src = [slice(None)] * 3
            dst = [slice(None)] * 3
            if step > 0:
                src[ax], dst[ax] = slice(1, None), slice(None, -1)
            else:
                src[ax], dst[ax] = slice(None, -1), slice(1, None)
            out[tuple(dst)] = a[tuple(src)]
            return out

        for c in range(3):
            up_l, dn_l = shift(liquid, 1, c, False), shift(liquid, -1, c, False)
            p_low = np.where(up_l, 1.5 * P - 0.5 * shift(P, 1, c, 0.0), P)     # at the cell's face toward -c
            p_high = np.where(dn_l, 1.5 * P - 0.5 * shift(P, -1, c, 0.0), P)   # at its face toward +c
            body_above, body_below = shift(M, 1, c, False), shift(M, -1, c, False)
            F[c] += self.h * self.h * float((p_high * (liquid & body_above)).sum() - (p_low * (liquid & body_below)).sum())
        return F

    def mass_kg(self):
        return self.particle_mass * len(self.pos)

    def kinetic_J(self):
        return 0.5 * self.particle_mass * float((self.vel ** 2).sum())

    def potential_J(self):
        return self.particle_mass * self.g * float(self.pos[:, 2].sum())

    def cell_counts(self):
        """Particles in each cell."""
        idx = np.clip((self.pos / self.h).astype(int), 0, np.array(self.shape) - 1)
        return np.bincount(np.ravel_multi_index(idx.T, self.shape), minlength=int(np.prod(self.shape))).reshape(self.shape)

    def fill_fraction(self):
        """How full each cell is (its particles over the particles a full cell holds)."""
        return self.cell_counts() / self.ppc

    # ── one sub-step ───────────────────────────────────────────────────────────────────────────────────────────────
    def _substep(self, dt):
        counts = self.cell_counts()
        liquid = (counts > 0) & ~self.solid
        grid = self._to_grid()
        old = [u.copy() for u in grid]
        grid[2] -= self.g * dt                                        # gravity, −z
        if self.mu > 0:
            grid = self._viscosity(grid, liquid, dt)
        grid = self._walls(grid)
        grid = self._project(grid, liquid, dt)
        valid = self._valid(liquid)
        for c in range(3):
            grid[c] = self._extrapolate(grid[c], valid[c])
            old[c] = self._extrapolate(old[c], valid[c])
        grid = self._walls(grid)
        # grid → particles: FLIP (the grid's change) blended with PIC (the grid's velocity)
        pic = self._sample(grid, self.pos)
        delta = pic - self._sample(old, self.pos)
        self.vel = self.flip_ratio * (self.vel + delta) + (1 - self.flip_ratio) * pic
        # move: midpoint rule through the new field, then out of the solids
        mid = self.pos + 0.5 * dt * pic
        self.pos = self.pos + dt * self._sample(grid, mid)
        self._push_out()
        self._spread()
        self.t += dt
        gi = np.array([0.0, 0.0, -self.g * self.mass_kg() * dt])
        self.ledger["gravity_impulse_Ns"] += gi
        self.ledger["steps"] += 1

    # ── particles ↔ grid ───────────────────────────────────────────────────────────────────────────────────────────
    def _face_shape(self, c):
        s = list(self.shape)
        s[c] += 1
        return tuple(s)

    def _to_grid(self):
        if _fast is not None:
            return [_fast.p2g(self.pos, np.ascontiguousarray(self.vel[:, c]), self.h, _OFFSETS[c], *self._face_shape(c))
                    for c in range(3)]
        return self._p2g(self._stencils(self.pos))

    def _sample(self, grid, pos):
        if _fast is not None:
            return np.stack([_fast.g2p(pos, grid[c], self.h, _OFFSETS[c]) for c in range(3)], 1)
        return self._g2p_all(grid, self._stencils(pos))

    def _stencil(self, pos, c):
        """The 8 faces of component ``c`` around each position: flat indices and trilinear weights, ``(N, 8)`` each.
        Positions past the last face use the edge face (constant extrapolation)."""
        sx, sy, sz = self._face_shape(c)
        g = pos / self.h - _OFFSETS[c]
        i0 = np.floor(g)
        f = g - i0
        i0 = i0.astype(np.int32)
        two = np.array([0, 1], np.int32)
        ix, iy, iz = (np.clip(i0[:, a, None] + two, 0, n - 1) for a, n in enumerate((sx, sy, sz)))
        wx, wy, wz = (np.stack([1 - f[:, a], f[:, a]], 1) for a in range(3))
        idx = (ix[:, :, None, None] * sy + iy[:, None, :, None]) * sz + iz[:, None, None, :]
        w = wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]
        return idx.reshape(-1, 8), w.reshape(-1, 8)

    def _stencils(self, pos):
        return [self._stencil(pos, c) for c in range(3)]

    def _p2g(self, st):
        grid = []
        for c, (idx, w) in enumerate(st):
            size = int(np.prod(self._face_shape(c)))
            num = np.bincount(idx.ravel(), (w * self.vel[:, c:c + 1]).ravel(), size)
            den = np.bincount(idx.ravel(), w.ravel(), size)
            grid.append(np.where(den > 0, num / np.maximum(den, 1e-30), 0.0).reshape(self._face_shape(c)))
        return grid

    def _g2p_all(self, grid, st):
        return np.stack([(grid[c].ravel()[idx] * w).sum(1) for c, (idx, w) in enumerate(st)], 1)

    # ── faces: which are walls, which are liquid ───────────────────────────────────────────────────────────────────
    def _neighbours(self, cells, c, pad):
        """The two cells either side of each face along axis ``c`` (outside the grid = ``pad``)."""
        p = [(0, 0)] * 3
        p[c] = (1, 1)
        padded = np.pad(cells, p + [(0, 0)] * (cells.ndim - 3), constant_values=pad)
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[c], hi[c] = slice(0, -1), slice(1, None)
        return padded[tuple(lo)], padded[tuple(hi)]

    def _walls(self, grid):
        out = []
        for c in range(3):
            a, b = self._neighbours(self.solid, c, True)
            va, vb = self._neighbours(self.solid_vel[..., c], c, 0.0)
            wall = a | b
            sv = np.where(a & b, 0.5 * (va + vb), np.where(a, va, vb))
            out.append(np.where(wall, sv, grid[c]))
        return out

    def _valid(self, liquid):
        valid = []
        for c in range(3):
            la, lb = self._neighbours(liquid, c, False)
            sa, sb = self._neighbours(self.solid, c, True)
            valid.append((la | lb) & ~(sa | sb))
        return valid

    def _viscosity(self, grid, liquid, dt):
        """Explicit diffusion of each component, ν = μ/ρ, on faces between two liquid cells."""
        nu = self.mu / self.rho
        n = max(1, int(np.ceil(nu * dt / self.h ** 2 / 0.15)))       # explicit stability: ν dt / h² ≤ 1/6
        out = []
        for c in range(3):
            u = grid[c]
            la, lb = self._neighbours(liquid, c, False)
            inner = la & lb
            for _ in range(n):
                pad = np.pad(u, 1, mode="edge")
                lap = (pad[2:, 1:-1, 1:-1] + pad[:-2, 1:-1, 1:-1] + pad[1:-1, 2:, 1:-1] + pad[1:-1, :-2, 1:-1] +
                       pad[1:-1, 1:-1, 2:] + pad[1:-1, 1:-1, :-2] - 6 * u) / self.h ** 2
                u = np.where(inner, u + (dt / n) * nu * lap, u)
            out.append(u)
        return out

    # ── pressure ───────────────────────────────────────────────────────────────────────────────────────────────────
    def _project(self, grid, liquid, dt):
        h, rho = self.h, self.rho
        cells = np.argwhere(liquid)
        n = len(cells)
        self.pressure = np.zeros(self.shape)
        if n == 0:
            return grid
        div = sum(np.diff(grid[c], axis=c) for c in range(3)) / h
        self.pressure = self._poisson(liquid, -(rho * h * h / dt) * div, self._last_p)
        self._liquid = liquid
        self._last_p = self.pressure.copy()
        # subtract the pressure gradient on every face with liquid on a side and no wall
        out = []
        force = np.zeros(3)
        work = 0.0
        for c in range(3):
            pa, pb = self._neighbours(self.pressure, c, 0.0)
            la, lb = self._neighbours(liquid, c, False)
            sa, sb = self._neighbours(self.solid, c, True)
            open_ = (la | lb) & ~(sa | sb)
            out.append(np.where(open_, grid[c] - dt / rho * (pb - pa) / h, grid[c]))
            # a wall face's push on the liquid: pressure × area, along the face's normal into the liquid
            into_b = sa & lb & ~sb                                    # wall on the low side, liquid above: pushes +c
            into_a = sb & la & ~sa                                    # wall on the high side: pushes −c
            f_b, f_a = pb * h * h * into_b, pa * h * h * into_a
            force[c] += f_b.sum() - f_a.sum()
            va, vb = self._neighbours(self.solid_vel[..., c], c, 0.0)
            work += float((f_b * va).sum() - (f_a * vb).sum()) * dt   # the solid's velocity along its push: power
        self.ledger["solid_force_N"] = force
        self.ledger["paddle_work_J"] += work
        return out

    def _poisson(self, liquid, rhs, x0=None):
        """Solve A x = rhs on the liquid cells, A = −h²∇² with x = 0 in air and no flux through walls. Compiled MIC(0)-PCG
        over the liquid's bounding box when numba is there, else SciPy CG (Jacobi-preconditioned) on a sparse matrix."""
        open_ = np.pad(~self.solid, 1, constant_values=False)
        diag = np.zeros(self.shape)
        for c in range(3):
            for sh in (-1, 1):
                sl = [slice(1, -1)] * 3
                sl[c] = slice(1 + sh, open_.shape[c] - 1 + sh)
                diag += open_[tuple(sl)]
        cell = liquid & (diag > 0)                                    # a cell walled in on all six sides: no pressure
        out = np.zeros(self.shape)
        if not cell.any():
            return out
        b = np.where(cell, rhs, 0.0)
        x0 = np.where(cell, x0, 0.0) if x0 is not None else np.zeros(self.shape)
        occ = np.argwhere(cell)
        box = tuple(slice(lo, hi + 1) for lo, hi in zip(occ.min(0), occ.max(0)))
        if _fast is not None:
            x, iters = _fast.pcg(np.ascontiguousarray(cell[box]), np.ascontiguousarray(diag[box]),
                                 np.ascontiguousarray(b[box]), np.ascontiguousarray(x0[box]), 1e-5, 2000)
            out[box] = np.where(cell[box], x, 0.0)
        else:
            x, iters = self._sparse_pcg(cell, diag, b, x0)
            out[cell] = x
        self.ledger["cg_iterations"] += iters
        return out

    def _sparse_pcg(self, cell, diag, b, x0):
        import scipy.sparse as sp
        from scipy.sparse.linalg import cg

        cells = np.argwhere(cell)
        n = len(cells)
        index = -np.ones(self.shape, int)
        index[cell] = np.arange(n)            # argwhere and boolean indexing share C order
        rows, cols = [], []
        for c in range(3):
            for step in (-1, 1):
                nb = cells.copy()
                nb[:, c] += step
                inside = (nb[:, c] >= 0) & (nb[:, c] < self.shape[c])
                nbi = np.where(inside, index[tuple(np.clip(nb, 0, np.array(self.shape) - 1).T)], -1)
                link = nbi >= 0
                rows.append(np.nonzero(link)[0])
                cols.append(nbi[link])
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        A = sp.csr_matrix((np.concatenate([diag[cell], -np.ones(len(rows))]),
                           (np.concatenate([np.arange(n), rows]), np.concatenate([np.arange(n), cols]))), shape=(n, n))
        iters = [0]
        x, _ = cg(A, b[cell], x0=x0[cell], rtol=1e-5, maxiter=2000, M=sp.diags(1.0 / diag[cell]),
                  callback=lambda _: iters.__setitem__(0, iters[0] + 1))
        return x, iters[0]

    # ── keeping the particles evenly spread (position only) ────────────────────────────────────────────────────────
    def _spread(self):
        """Move particles from over-full cells to under-full ones, without touching any velocity: a displacement d = −∇q
        with −h²∇²q = h²·(fill − 1) (Kugelstadt et al., "Implicit density projection for volume conserving liquids", IEEE TVCG
        2019). FLIP's particles drift and bunch (a moving paddle shoves a whole layer into the next cell); the velocity
        projection keeps the FLOW incompressible but never spreads particles back, so the drawn volume would shrink.
        This is a correction of the particle sampling, not a force: momentum is unchanged."""
        counts = self.cell_counts()
        liquid = (counts > 0) & ~self.solid
        # over-full cells push out everywhere; under-full cells pull in only inside the liquid (every face neighbour
        # liquid or wall), so the surface is free to move
        closed = liquid.copy()
        for c in range(3):
            for sh in (-1, 1):
                nb = np.roll(liquid | self.solid, sh, axis=c)
                edge = [slice(None)] * 3
                edge[c] = 0 if sh == 1 else -1
                nb[tuple(edge)] = True                                # outside the grid is a wall
                closed &= nb
        excess = counts / self.ppc - 1.0
        excess = np.where(liquid & ((excess > 0) | closed), excess, 0.0)
        if not excess.any():
            return
        q = self._poisson(liquid, self.h * self.h * excess)
        d = []
        for c in range(3):
            qa, qb = self._neighbours(q, c, 0.0)
            sa, sb = self._neighbours(self.solid, c, True)
            d.append(np.where(sa | sb, 0.0, -(qb - qa) / self.h))
        move = self._sample(d, self.pos)
        limit = 0.5 * self.h
        self.pos = self.pos + np.clip(move, -limit, limit)
        self._push_out()

    # ── velocity beyond the liquid ─────────────────────────────────────────────────────────────────────────────────
    def _extrapolate(self, u, valid, layers=3):
        if _fast is not None:
            return _fast.extrapolate(np.ascontiguousarray(u, dtype=float).copy(), np.ascontiguousarray(valid), layers)
        u, valid = u.copy(), valid.copy()
        for _ in range(layers):
            s = np.zeros_like(u)
            n = np.zeros_like(u)
            for ax in range(3):
                for sh in (-1, 1):
                    vu = np.roll(u * valid, sh, ax)
                    vv = np.roll(valid, sh, ax).astype(float)
                    edge = [slice(None)] * 3
                    edge[ax] = 0 if sh == 1 else -1                   # no wrap-around
                    vu[tuple(edge)] = 0
                    vv[tuple(edge)] = 0
                    s += vu
                    n += vv
            grow = ~valid & (n > 0)
            u[grow] = s[grow] / n[grow]
            valid |= grow
        return u

    # ── particles stay in the liquid's domain ──────────────────────────────────────────────────────────────────────
    def _push_out(self):
        from scipy.ndimage import distance_transform_edt

        top = np.array(self.shape) * self.h
        eps = 1e-6 * self.h
        self.pos = np.clip(self.pos, eps, top - eps)
        idx = (self.pos / self.h).astype(int)
        inside = self.solid[idx[:, 0], idx[:, 1], idx[:, 2]]
        if not inside.any():
            return
        _, near = distance_transform_edt(self.solid, return_indices=True)
        stuck = idx[inside]
        target = np.stack([near[a][tuple(stuck.T)] for a in range(3)], 1)
        lo = target * self.h + 0.01 * self.h
        hi = (target + 1) * self.h - 0.01 * self.h
        self.pos[inside] = np.clip(self.pos[inside], lo, hi)
        # it now moves as the solid that pushed it does (along the push)
        sv = self.solid_vel[tuple(stuck.T)]
        self.vel[inside] = np.where(np.abs(sv) > 0, sv, self.vel[inside])
