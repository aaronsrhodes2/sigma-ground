"""
FluidVoxelWorld: a voxel build whose liquid MOVES, with momentum, driven by a wave machine.

The liquid's cells become ``dynamics.fluid.flip.Flip`` particles (incompressible, free surface, momentum carried); every
other material is a wall. One material may be a PADDLE: its cells are a rigid piston that moves back and forth along one
axis, ``x(t) = A·sin(2πt/T)``, pushing the liquid with its own velocity (the wave machine). Each tick the cells are drawn
again from where the particles are: a cell at least half full of liquid is liquid (``voxel_world.MELTED``'s rule).

Numbers: the liquid's density and viscosity from ``field.interface.block`` (measured first), gravity the standard value
(``fluid_surface.STANDARD_GRAVITY``), and the wave machine's default period is the tank's own first sloshing mode
(``fluid_surface.sloshing_angular_frequency``: finite depth, walls as antinodes), so it drives the tank at resonance.

Honest limits (reported in ``gaps``):
  - no heat or phase change here: the liquid keeps its starting temperature (``voxel_world`` does heat, without motion);
  - surface tension is not in the solver (it matters below the capillary length, ~2.7 mm for water, under a cell here);
  - the paddle is drawn on whole cells (it moves in steps of one cell), while the liquid feels its smooth velocity;
  - walls are fixed and infinitely strong; the paddle's motion is prescribed (the liquid does not slow it).
"""
from __future__ import annotations

import math

import numpy as np

from .voxel_world import MELTED


class FluidVoxelWorld:
    """``field``: a Deckard VoxelField. ``liquid``: the moving material. ``paddle``: ``{"material", "axis",
    "amplitude_m", "period_s"}`` (period None = the tank's first sloshing mode), or None."""

    def __init__(self, field, *, liquid="water", temperature_k=293.15, paddle=None, ppc=8, seed=1, headroom=8,
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
        self.static = static
        self.base = np.where(static, self.label, 0)                  # walls: never change
        # the liquid moves only inside the build's bounding box (plus headroom for splashes); its edges are walls
        occ = np.argwhere(self.label > 0)
        lo, hi = occ.min(0), occ.max(0) + 1
        hi[2] = min(self.label.shape[2], hi[2] + headroom)
        self.box = tuple(slice(a, b) for a, b in zip(lo, hi))
        self.flip = Flip(self.label[self.box].shape, self.pitch, density=rho, viscosity=mu, gravity=self.gravity,
                         solid=(static | (self.paddle["home"] if self.paddle else False))[self.box],
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
        if not self.paddle:
            self.flip.set_solid(self.static[self.box])
            return
        p = self.paddle
        x, v = self._paddle_state()
        self.shift = int(round(x / self.pitch))
        mask = np.roll(p["home"], self.shift, axis=p["axis"])         # home has empty margin around it: no wrap
        vel = np.zeros(self.label.shape + (3,))
        vel[mask, p["axis"]] = v
        self.paddle_mask = mask
        self.flip.set_solid((self.static | mask)[self.box], vel[self.box])

    # ── stepping ───────────────────────────────────────────────────────────────────────────────────────────────────
    def step(self, dt):
        """Advance ``dt`` s: the paddle moves (in sub-steps the solver's CFL asks for), the liquid follows."""
        left = float(dt)
        while left > 1e-12:
            sub = min(left, self.cfl * self.pitch / max(1e-9, self._max_speed()))
            self._update_solids()
            self.flip.step(sub, cfl=self.cfl)
            self.t += sub
            left -= sub
        self._update_solids()
        self._draw()

    def _max_speed(self):
        v = float(np.sqrt((self.flip.vel ** 2).sum(1)).max()) if len(self.flip.vel) else 0.0
        if self.paddle:
            v = max(v, abs(self._paddle_state()[1]))
        return max(v, math.sqrt(2 * self.gravity * self.pitch))

    def _draw(self):
        lab = self.base.copy()
        if self.paddle:
            lab[self.paddle_mask] = self.paddle["index"]
        full = np.zeros(self.label.shape, bool)
        full[self.box] = self.flip.fill_fraction() >= MELTED
        lab[full & (lab == 0)] = self.li
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
                f"KE {L['kinetic_J']:.3g} J{drive}")
