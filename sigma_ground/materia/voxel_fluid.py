"""
FluidVoxelWorld: a voxel build whose liquid MOVES, with momentum, driven by a wave machine, with solid bodies that fall,
splash, sink or float.

The liquid's cells become ``dynamics.fluid.flip.Flip`` particles (incompressible, free surface, momentum carried); every
other material is a wall. One material may be a PADDLE: its cells are a rigid piston that moves back and forth along one
axis, ``x(t) = A·sin(2πt/T)``, pushing the liquid with its own velocity (the wave machine). Each tick the cells are drawn
again from where the particles are: as many liquid blocks as the liquid's volume, in its fullest cells.

Materials named in ``bodies`` are rigid BODIES (each connected piece one body): a mass (its ``block()`` density times its
volume), a velocity, gravity, and the liquid's pressure on every wetted face (``Flip.force_on``), so buoyancy, drag and the
splash's push-back are all one force, not three models. The liquid in turn sees each body as a wall moving with the body's
velocity, so a falling block shoves water aside: that is the splash. Gold sinks and oak floats because of the pressure on
them, not because of a density rule.

Numbers: the liquid's density and viscosity from ``field.interface.block`` (measured first), gravity the standard value
(``fluid_surface.STANDARD_GRAVITY``), and the wave machine's default period is the tank's own first sloshing mode
(``fluid_surface.sloshing_angular_frequency``: finite depth, walls as antinodes), so it drives the tank at resonance.

Honest limits (reported in ``gaps``):
  - no heat or phase change here: the liquid keeps its starting temperature (``voxel_world`` does heat, without motion);
  - surface tension is not in the solver (it matters below the capillary length, ~2.7 mm for water, under a cell here);
  - the paddle is drawn on whole cells (it moves in steps of one cell), while the liquid feels its smooth velocity;
  - walls are fixed and infinitely strong; the paddle's motion is prescribed (the liquid does not slow it);
  - bodies translate but do not rotate, and they stop dead on walls and on each other (perfectly inelastic contact,
    no friction model); they are drawn on whole cells (their smooth position rounded);
  - body and liquid are coupled one sub-step apart (the force on the body is last step's pressure), with an added-mass
    term that keeps light bodies stable (the method's, not a physical constant).
"""
from __future__ import annotations

import math

import numpy as np



class FluidVoxelWorld:
    """``field``: a Deckard VoxelField. ``liquid``: the moving material. ``paddle``: ``{"material", "axis",
    "amplitude_m", "period_s"}`` (period None = the tank's first sloshing mode), or None."""

    def __init__(self, field, *, liquid="water", temperature_k=293.15, paddle=None, bodies=(), ppc=8, seed=1, headroom=8,
                 cfl=2.0):
        from ..dynamics.fluid.flip import Flip
        from ..field.interface.block import block
        from ..field.interface.fluid_surface import STANDARD_GRAVITY

        self.pitch = float(field.voxel_size)
        self.cfl = float(cfl)
        self.materials = list(field.materials)
        self.label = np.array(field.label_grid, dtype=np.int32)
        self.liquid_name = liquid
        self.li = self.materials.index(liquid)
        self.temperature_k = float(temperature_k)
        props = block(liquid, self.temperature_k)["properties"]
        self.sources = {k: props[k]["source"] for k in ("density", "viscosity", "surface_tension") if k in props}
        rho, mu = props["density"]["value"], props["viscosity"]["value"]
        self.rho = rho
        self.surface_tension = props.get("surface_tension", {}).get("value")
        self.gravity = STANDARD_GRAVITY
        liquid_cells = self.label == self.li

        self.paddle = None
        static = (self.label > 0) & ~liquid_cells
        if paddle:
            pi = self.materials.index(paddle["material"])
            home = self.label == pi
            static &= ~home
            self.paddle = dict(paddle, index=pi, home=home)
            self.paddle["resonant_s"] = self.resonant_period(paddle["axis"], liquid_cells)
            if paddle.get("period_s") is None:
                self.paddle["period_s"] = self.paddle["resonant_s"]
        self.bodies = []
        for name in bodies:
            from scipy.ndimage import label as components
            bi = self.materials.index(name)
            cells = self.label == bi
            static &= ~cells
            rho_b = block(name, self.temperature_k)["properties"]["density"]["value"]
            pieces, n = components(cells)
            for k in range(1, n + 1):
                idx = np.argwhere(pieces == k)
                anchor = idx.min(0)
                self.bodies.append({"material": name, "index": bi, "offsets": idx - anchor,
                                    "pos": anchor.astype(float), "cell": anchor.copy(), "vel": np.zeros(3),
                                    "acc": np.zeros(3), "force_N": np.zeros(3),
                                    "mass_kg": rho_b * len(idx) * self.pitch ** 3, "density": rho_b})
        self.static = static
        self.base = np.where(static, self.label, 0)                  # walls: never change
        # the liquid moves only inside the build's bounding box (plus headroom for splashes); its edges are walls
        occ = np.argwhere(self.label > 0)
        lo, hi = occ.min(0), occ.max(0) + 1
        hi[2] = min(self.label.shape[2], hi[2] + headroom)
        self.box = tuple(slice(a, b) for a, b in zip(lo, hi))
        self.flip = Flip(self.label[self.box].shape, self.pitch, density=rho, viscosity=mu, gravity=self.gravity,
                         solid=(static | (self.paddle["home"] if self.paddle else False) | self._bodies_mask())[self.box],
                         liquid=liquid_cells[self.box], ppc=ppc, seed=seed)
        self.t = 0.0
        self.shift = 0
        self._update_solids()

    # ── the wave machine ───────────────────────────────────────────────────────────────────────────────────────────
    def resonant_period(self, axis, liquid_cells=None):
        """The first sloshing mode of the liquid as built: its length along ``axis``, its depth (s)."""
        from ..field.interface.fluid_surface import sloshing_angular_frequency
        cells = np.argwhere(self.label == self.li if liquid_cells is None else liquid_cells)
        length = (cells[:, axis].max() - cells[:, axis].min() + 1) * self.pitch
        depth = (cells[:, 2].max() - cells[:, 2].min() + 1) * self.pitch
        w = sloshing_angular_frequency(length, depth, 1, self.surface_tension or 0.0, self.rho, self.gravity)
        return 2 * math.pi / w

    def set_period(self, period_s):
        """Change the drive's period, keeping the paddle where it is (its phase carries on smoothly)."""
        p = self.paddle
        phase = 2 * math.pi * self.t / p["period_s"] + p.get("phase0", 0.0)
        p["period_s"] = float(period_s)
        p["phase0"] = phase - 2 * math.pi * self.t / p["period_s"]

    def stop_paddle(self):
        if self.paddle:
            self.paddle["amplitude_m"] = 0.0

    def _paddle_state(self):
        p = self.paddle
        w = 2 * math.pi / p["period_s"]
        ph = w * self.t + p.get("phase0", 0.0)
        return p["amplitude_m"] * math.sin(ph), p["amplitude_m"] * w * math.cos(ph)

    def _update_solids(self):
        vel = np.zeros(self.label.shape + (3,))
        solid = self.static.copy()
        if self.paddle:
            p = self.paddle
            x, v = self._paddle_state()
            self.shift = int(round(x / self.pitch))
            mask = np.roll(p["home"], self.shift, axis=p["axis"])     # home has empty margin around it: no wrap
            vel[mask, p["axis"]] = v
            self.paddle_mask = mask
            solid |= mask
        for b in self.bodies:
            m = self._body_mask(b)
            vel[m] = b["vel"]
            solid |= m
        self.flip.set_solid(solid[self.box], vel[self.box])

    # ── bodies ─────────────────────────────────────────────────────────────────────────────────────────────────────
    def _body_cells(self, b, cell=None):
        return (b["cell"] if cell is None else cell) + b["offsets"]

    def _body_mask(self, b, cell=None):
        m = np.zeros(self.label.shape, bool)
        m[tuple(self._body_cells(b, cell).T)] = True
        return m

    def _bodies_mask(self, skip=None):
        m = np.zeros(self.label.shape, bool)
        for b in self.bodies:
            if b is not skip:
                m |= self._body_mask(b)
        return m

    def _fits(self, b, cell, blocked):
        idx = self._body_cells(b, cell)
        lo, hi = np.array([s.start for s in self.box]), np.array([s.stop for s in self.box])
        if (idx < lo).any() or (idx >= hi).any():
            return False
        return not blocked[tuple(idx.T)].any()

    def _wet_fraction(self, b):
        """How much of the body's surface has liquid against it (0 in air, 1 under water)."""
        liquid = getattr(self.flip, "_liquid", None)
        if liquid is None:
            return 0.0
        full = np.zeros(self.label.shape, bool)
        full[self.box] = liquid
        mask = self._body_mask(b)
        shell = np.zeros_like(mask)
        for ax in range(3):
            for sh in (-1, 1):
                shell |= np.roll(mask, sh, axis=ax)
        shell &= ~mask & ~self.static
        return float(full[shell].mean()) if shell.any() else 0.0

    def _film_force(self, b):
        """Pressure on the faces where a body touches a wall. Between them is a film of liquid thinner than a cell, which
        the grid cannot hold, so without this a body resting on the floor would feel no pressure beneath it (a perfect
        suction seal: an oak block driven to the floor would stay there). The film carries the pressure of the liquid
        beside it: the contact faces take the mean pressure of the liquid around the body's contact layer."""
        P = np.zeros(self.label.shape)
        P[self.box] = self.flip.pressure
        liquid = np.zeros(self.label.shape, bool)
        if getattr(self.flip, "_liquid", None) is None:
            return np.zeros(3)
        liquid[self.box] = self.flip._liquid
        cells = self._body_cells(b)
        shape = np.array(self.label.shape)
        F = np.zeros(3)
        for c in range(3):
            for s in (-1, 1):
                nb = cells.copy()
                nb[:, c] += s
                ok = np.all((nb >= 0) & (nb < shape), axis=1)
                touching = np.zeros(len(cells), bool)
                touching[ok] = self.static[tuple(nb[ok].T)]
                if not touching.any():
                    continue
                # the liquid beside the body's contact layer (its cells level with the touching ones along c)
                layer = cells[np.isin(cells[:, c], cells[touching][:, c])]
                side = []
                for c2 in range(3):
                    if c2 == c:
                        continue
                    for s2 in (-1, 1):
                        q = layer.copy()
                        q[:, c2] += s2
                        inside = np.all((q >= 0) & (q < shape), axis=1)
                        q = q[inside]
                        wet = liquid[tuple(q.T)]
                        side.extend(P[tuple(q[wet].T)].tolist())
                if side:
                    F[c] -= s * float(np.mean(side)) * self.pitch ** 2 * int(touching.sum())   # away from the wall
        return F

    def _liquid_velocity_around(self, b):
        """Mean velocity of the liquid particles in the cells next to the body, or None when it is dry."""
        mask = self._body_mask(b)
        shell = np.zeros_like(mask)
        for ax in range(3):
            for sh in (-1, 1):
                shell |= np.roll(mask, sh, axis=ax)
        shell &= ~mask
        f = self.flip
        idx = np.clip((f.pos / f.h).astype(int), 0, np.array(f.shape) - 1)
        near = shell[self.box][idx[:, 0], idx[:, 1], idx[:, 2]]
        return f.vel[near].mean(0) if near.any() else None

    def _move_bodies(self, dt):
        """Gravity and the liquid's pressure, then move; a body that would enter a wall, the paddle or another body
        stops along that axis (inelastic)."""
        for b in self.bodies:
            F = self.flip.force_on(self._body_mask(b)[self.box]) + self._film_force(b)
            # added mass, the partitioned-coupling stabiliser: the liquid pushed by the body pushes back one step late;
            # weighting by (m + m_a) and adding m_a times the last acceleration removes that lag's oscillation for light
            # bodies and changes nothing once the acceleration settles. m_a = half the displaced liquid (a cube's is
            # about 0.6 of it), scaled by how wet the body is.
            m_a = 0.5 * self.rho * len(b["offsets"]) * self.pitch ** 3 * self._wet_fraction(b)
            gravity = b["mass_kg"] * np.array([0.0, 0.0, -self.gravity])
            acc = (F + gravity + m_a * b["acc"]) / (b["mass_kg"] + m_a)
            b["acc"], b["force_N"] = acc, F
            v_new = b["vel"] + acc * dt
            u = self._liquid_velocity_around(b)
            if u is not None:
                # one step of the liquid's pressure can bring the body to the liquid's speed, not past it, and can
                # speed it away from the liquid no faster than buoyancy can (rho_liquid / rho_body times g): the coupling
                # one sub-step apart otherwise lets a pressure spike throw a light body out of the water
                rel0, rel1 = b["vel"] - u, v_new - u
                cap = np.abs(rel0) + self.gravity * max(1.0, self.rho / b["density"]) * dt
                rel1 = np.where(rel0 * rel1 < 0, 0.0, np.clip(rel1, -cap, cap))
                v_new = u + rel1
            b["vel"] = v_new
            blocked = self.static | self._bodies_mask(skip=b)
            if self.paddle:
                blocked |= self.paddle_mask
            pos = b["pos"].copy()
            for ax in range(3):
                trial = pos.copy()
                trial[ax] += b["vel"][ax] * dt / self.pitch
                # in contact: moving past its cell's centre toward a blocked neighbour cell
                toward = np.round(pos).astype(int)
                toward[ax] += int(np.sign(b["vel"][ax]))
                past = (trial[ax] - round(pos[ax])) * np.sign(b["vel"][ax]) > 0
                if not self._fits(b, np.round(trial).astype(int), blocked) or                         (past and b["vel"][ax] and not self._fits(b, toward, blocked)):
                    b["vel"][ax] = 0.0
                    pos[ax] = float(round(pos[ax]))
                else:
                    pos = trial
            b["pos"] = pos
            b["cell"] = np.round(pos).astype(int)

    # ── stepping ───────────────────────────────────────────────────────────────────────────────────────────────────
    def step(self, dt):
        """Advance ``dt`` s: the paddle moves (in sub-steps the solver's CFL asks for), the liquid follows."""
        left = float(dt)
        while left > 1e-12:
            sub = min(left, self.cfl * self.pitch / max(1e-9, self._max_speed()))
            self._update_solids()
            self.flip.step(sub, cfl=self.cfl)
            self._move_bodies(sub)
            self.t += sub
            left -= sub
        self._update_solids()
        self._draw()

    def _max_speed(self):
        v = float(np.sqrt((self.flip.vel ** 2).sum(1)).max()) if len(self.flip.vel) else 0.0
        if self.paddle:
            v = max(v, abs(self._paddle_state()[1]))
        for b in self.bodies:
            v = max(v, float(np.abs(b["vel"]).max()))
        return max(v, math.sqrt(2 * self.gravity * self.pitch))

    def _draw(self):
        lab = self.base.copy()
        if self.paddle:
            lab[self.paddle_mask] = self.paddle["index"]
        for b in self.bodies:
            lab[self._body_mask(b)] = b["index"]
        # exactly as many liquid blocks as the particles' volume, in the fullest cells (the mass-keeping rule of
        # deckard.voxel_resample): a fill threshold would grow the drawn water with every half-full surface cell
        # ranked by fill averaged over each cell's 3×3×3 neighbourhood, so sampling noise does not punch holes in the body
        from scipy.ndimage import uniform_filter
        fill = np.zeros(self.label.shape)
        fill[self.box] = uniform_filter(self.flip.fill_fraction(), size=3, mode="constant") + 1e-6 * self.flip.fill_fraction()
        fill[lab > 0] = 0.0
        quota = int(round(len(self.flip.pos) / self.flip.ppc))
        order = np.argsort(-fill, axis=None, kind="stable")[:quota]
        order = order[fill.ravel()[order] > 0]
        lab.ravel()[order] = self.li
        self.label = lab

    # ── the books ──────────────────────────────────────────────────────────────────────────────────────────────────
    def mass_kg(self):
        return self.flip.mass_kg()

    def energy_J(self):
        """The liquid's kinetic plus gravitational potential energy (z from the grid's floor)."""
        return self.flip.kinetic_J() + self.flip.potential_J()

    @property
    def ledger(self):
        f = self.flip
        return {"momentum_Ns": f.momentum(), "kinetic_J": f.kinetic_J(), "paddle_work_J": f.ledger["paddle_work_J"],
                "wall_force_N": f.ledger["solid_force_N"], "gravity_impulse_Ns": f.ledger["gravity_impulse_Ns"]}

    def gaps(self):
        g = ["no heat or phase change in the moving liquid (it stays at its starting temperature)",
             "the paddle's motion is prescribed; the liquid does not slow it"]
        if self.bodies:
            g.append("bodies translate only (no rotation) and stop dead on contact (no restitution or friction model)")
            g.append("body and liquid update one sub-step apart (no implicit coupling): each step's pressure may bring a "
                     "body to the liquid's speed but not past it, and no faster than buoyancy, which stands in for the solve")
            g.append("the free surface's zero pressure sits at the first air cell's centre, half a cell above the surface "
                     "(no ghost-fluid boundary), so a floating body rides about half a cell high")
        if self.surface_tension and self.pitch > 2.7e-3:
            g.append("surface tension is not in the solver (cells are larger than water's capillary length)")
        return g

    def readout(self):
        L = self.ledger
        px = L["momentum_Ns"]
        drive = ""
        if self.paddle:
            drive = f" | drive {self.paddle['period_s']:.2f} s, A {self.paddle['amplitude_m'] * 100:.1f} cm, " \
                    f"work {L['paddle_work_J']:.3g} J"
        return (f"t {self.t:.2f} s | {self.mass_kg():.4g} kg water | p ({px[0]:+.3f}, {px[1]:+.3f}, {px[2]:+.3f}) N s | "
                f"KE {L['kinetic_J']:.3g} J{drive}" +
                "".join(f" | {b['material']} z {b['pos'][2] * self.pitch:.3f} m v {b['vel'][2]:+.2f} m/s" for b in self.bodies))
