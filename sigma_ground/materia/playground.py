"""
PlaygroundWorld: one world where heat, phase change, liquid momentum and rigid bodies all run on one clock
(misc/SKIPPYCRAFT_PLAN.md 10b, P2). A hot gold bar sinks through a lead bath that its plate melted; the melt flows with
momentum, carries its heat with it, and freezes where it cools.

Three kinds of matter, every property from ``field.interface.block`` at its own temperature (the live cards of
``voxel_world.live_table``):

  walls   static solid cells: temperature, mass, melt and boil fractions per cell;
  bodies  rigid bodies with momentum (``FluidVoxelWorld``'s coupling: gravity, the liquid's pressure, added mass,
          inelastic contact); every body cell carries its own temperature;
  liquid  FLIP particles (``dynamics.fluid.flip``) of ONE material, each carrying its temperature, its liquid fraction
          (frozen 0 .. liquid 1) and its boiled fraction.

Every tick (``step``): (1) heat: walls, bodies and liquid are put on one grid (a liquid cell's temperature is its particles'
mean, its heat capacity the mass actually in it) and conduct and radiate (``dynamics.fields.heat.diffuse_fvm``); each
cell's energy change goes back through the enthalpy h(T) = integral of cp dT, to the cell or shared equally among its
particles; (2) phase: a wall or body cell of the liquid's material that has taken its whole latent heat of fusion becomes
particles (its mass, temperature and velocity kept); particles below the melting point give their latent heat up, and a
cell whose particles have all frozen becomes a wall again with exactly their mass; particles past the boiling point take
the latent heat of vaporisation and then vent (mass and energy to the ledger); (3) the liquid flows and the bodies move
(FLIP sub-steps at its CFL limit).

The books: mass (walls + bodies + particles + vented) is exact. Thermal energy (enthalpy + latent heat held) closes
against what was radiated, vented and lost to an air bath. Mechanical energy is reported apart: FLIP's numerical damping
and bodies stopping dead on contact take kinetic energy out of the world (``dissipated_J``), where a real liquid would
turn it into heat; it is shown, not hidden.

Honest limits (``gaps()``): one liquid material (another material that melts does so in place, as a wall); the melt is
inviscid (sigma-ground's melt viscosity, ``fluid.eyring_viscosity``, is 10^3-10^6 too high: KNOWN_GAPS); bodies translate
only and stop dead on
contact; a body of the liquid's material that melts loses those cells but is not split in two; steam is not modelled;
the melt flows at its solid's density (no expansion on melting) so a cell is exactly its particles and the books are exact.
"""
from __future__ import annotations

import math

import numpy as np

from .voxel_fluid import FluidVoxelWorld
from .voxel_world import live_table


def _per_kg(name):
    """The live table per kilogram: ``{"T", "H": h(T) J/kg from the table's floor, "rho_cp": cp}`` and the rest."""
    lv = dict(live_table(name))
    cp = lv["cp"]
    lv["rho_cp"] = cp
    lv["H"] = np.concatenate([[0.0], np.cumsum(0.5 * (cp[1:] + cp[:-1]) * np.diff(lv["T"]))])
    return lv


def _h(lv, T):
    from .voxel_world import _h_of
    return _h_of(lv, T)


def _T(lv, H):
    from .voxel_world import _t_of
    return _t_of(lv, H)


class PlaygroundWorld(FluidVoxelWorld):
    """``field``: a Deckard VoxelField. ``temperature_k``: scalar or a grid like the field. ``liquid``: the one material
    that flows when molten (its cells start as walls or bodies unless they are already past its melting point).
    ``bodies``: materials whose connected pieces are rigid bodies. ``env``: "ISM" (vacuum, radiating) or "STP" (an air
    bath at its temperature)."""

    def __init__(self, field, *, liquid, temperature_k=293.15, bodies=(), env="ISM", ppc=8, seed=1, headroom=8, cfl=2.0):
        from scipy.ndimage import label as components

        from ..dynamics.fluid.flip import Flip
        from ..field.interface.atmosphere import atmosphere_preset
        from ..field.interface.block import block
        from ..field.interface.fluid_surface import STANDARD_GRAVITY
        from ..field.interface.phase_transition import latent_heat_J_kg

        self.env = atmosphere_preset(env) if isinstance(env, str) else dict(env)
        self.ambient_T = float(self.env.get("temperature_k", 293.15))
        self.pitch = float(field.voxel_size)
        self.V = self.pitch ** 3
        self.cfl = float(cfl)
        self.materials = list(field.materials)
        self.label = np.array(field.label_grid, dtype=np.int32)
        self.gravity = STANDARD_GRAVITY
        self.paddle = None
        T0 = np.broadcast_to(np.asarray(temperature_k, dtype=float), self.label.shape).copy()

        # per material: build density (a cell's mass), per-kg live table, phase points and latent heats
        dens = field.density_by_label or {}
        self.mat = [None]
        for name in self.materials[1:]:
            p = block(name)["properties"]
            data_name = {"water": "water_ice"}.get(block(name)["canonical"], block(name)["canonical"])
            m = {"name": name, "rho": float(dens.get(name) or p["density"]["value"]), "lv": _per_kg(name),
                 "T_melt": p.get("melting_point", {}).get("value"), "T_boil": p.get("boiling_point", {}).get("value"),
                 "L_fus": latent_heat_J_kg(data_name, "fus") if "melting_point" in p else None,
                 "L_vap": latent_heat_J_kg(data_name, "vap") if "boiling_point" in p else None}
            if m["L_fus"] is None or m["L_vap"] is None:
                m["T_melt"] = m["T_boil"] = None                    # keeps its phase (block() names the gap)
            self.mat.append(m)

        # the liquid: inviscid, at the material's build density so a cell is exactly ppc particles (mass and energy exact)
        self.liquid_name = liquid
        self.li = self.materials.index(liquid) if liquid in self.materials else None     # None: nothing here can flow
        L = self.mat[self.li] if self.li is not None else {"rho": 1.0, "T_melt": float("inf")}
        if L["T_melt"] is None:
            raise ValueError(f"{liquid!r} has no melting point and latent heats in sigma-ground: it cannot melt")
        self.rho = L["rho"]
        self.surface_tension = None
        self.sources = {"density": f"{liquid}'s build density (the solid's): a melted cell is exactly its particles"}

        # walls and bodies
        molten = (self.label == self.li) & (T0 > L["T_melt"])
        self.Ts = np.where(self.label > 0, T0, self.ambient_T)                   # wall temperature
        self.Ms = np.zeros(self.label.shape)                                      # wall mass (kg)
        for k in range(1, len(self.materials)):
            self.Ms[self.label == k] = self.mat[k]["rho"] * self.V
        self.melt = np.zeros(self.label.shape)
        self.boil = np.zeros(self.label.shape)
        self.bodies = []
        body_mask = np.zeros(self.label.shape, bool)
        for name in bodies:
            bi = self.materials.index(name)
            cells = (self.label == bi) & ~molten
            pieces, n = components(cells)
            for k in range(1, n + 1):
                idx = np.argwhere(pieces == k)
                anchor = idx.min(0)
                body_mask[tuple(idx.T)] = True
                self.bodies.append({"material": name, "index": bi, "offsets": idx - anchor, "pos": anchor.astype(float),
                                    "cell": anchor.copy(), "vel": np.zeros(3), "acc": np.zeros(3), "force_N": np.zeros(3),
                                    "density": self.mat[bi]["rho"], "mass_kg": self.mat[bi]["rho"] * len(idx) * self.V,
                                    "T": T0[tuple(idx.T)].copy(), "melt": np.zeros(len(idx)), "boil": np.zeros(len(idx))})
        self.static = (self.label > 0) & ~body_mask & ~molten
        for a in (self.Ms, self.melt, self.boil):
            a[~self.static] = 0.0
        self.Ts[~self.static] = self.ambient_T
        self.base = np.where(self.static, self.label, 0)

        # the liquid's domain: the build's bounding box plus headroom
        occ = np.argwhere(self.label > 0)
        lo, hi = occ.min(0), occ.max(0) + 1
        hi[2] = min(self.label.shape[2], hi[2] + headroom)
        self.box = tuple(slice(a, b) for a, b in zip(lo, hi))
        self.lo = lo
        self.flip = Flip(self.label[self.box].shape, self.pitch, density=self.rho, viscosity=0.0, gravity=self.gravity,
                         solid=(self.static | body_mask)[self.box], liquid=molten[self.box], ppc=ppc, seed=seed)
        n = len(self.flip.pos)
        idx = self._particle_cells()
        self.pT = T0[tuple(idx.T)] if n else np.zeros(0)
        self.pL = np.ones(n)                                                      # liquid fraction
        self.pB = np.zeros(n)                                                     # boiled fraction
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.shift = 0
        self.ledger_ = {"radiated_J": 0.0, "to_surroundings_J": 0.0, "vented_kg": 0.0, "vented_J": 0.0,
                        "dissipated_J": 0.0, "melted_cells": 0, "frozen_cells": 0, "vented_particles": 0}
        self._update_solids()
        self._draw()

    # ── where things are ──────────────────────────────────────────────────────────────────────────────────────────
    def _particle_cells(self, pos=None):
        """Each particle's cell in the FULL grid."""
        f = self.flip
        pos = f.pos if pos is None else pos
        return np.clip((pos / f.h).astype(int), 0, np.array(f.shape) - 1) + self.lo

    def _body_cell_list(self, b):
        return b["cell"] + b["offsets"]

    # ── the books ─────────────────────────────────────────────────────────────────────────────────────────────────
    def mass_kg(self):
        bodies = sum(b["mass_kg"] for b in self.bodies)
        return float(self.Ms.sum()) + bodies + len(self.flip.pos) * self.flip.particle_mass

    def thermal_J(self):
        """Enthalpy (from the tables' floor) plus latent heat held, of everything in the world."""
        E = 0.0
        for k in range(1, len(self.materials)):
            m, c = self.mat[k], self.static & (self.label == k)
            if c.any():
                E += float((self.Ms[c] * (_h(m["lv"], self.Ts[c]) + self.melt[c] * (m["L_fus"] or 0)
                                          + self.boil[c] * (m["L_vap"] or 0))).sum())
        for b in self.bodies:
            m = self.mat[b["index"]]
            per = b["mass_kg"] / len(b["offsets"])
            E += float((per * (_h(m["lv"], b["T"]) + b["melt"] * (m["L_fus"] or 0) + b["boil"] * (m["L_vap"] or 0))).sum())
        if len(self.pT):
            L = self.mat[self.li]
            E += float((self.flip.particle_mass * (_h(L["lv"], self.pT) + self.pL * L["L_fus"] + self.pB * L["L_vap"])).sum())
        return E

    def mechanical_J(self):
        """Kinetic + gravitational potential energy (height from the grid's floor) of the liquid and the bodies."""
        E = self.flip.kinetic_J() + self.flip.potential_J() + float(self.lo[2] * self.pitch) * self.flip.mass_kg() * self.gravity
        for b in self.bodies:
            z = (b["pos"][2] + b["offsets"][:, 2].mean() + 0.5) * self.pitch
            E += 0.5 * b["mass_kg"] * float(b["vel"] @ b["vel"]) + b["mass_kg"] * self.gravity * z
        return E

    @property
    def ledger(self):
        return dict(self.ledger_, momentum_Ns=self.flip.momentum() + sum(b["mass_kg"] * b["vel"] for b in self.bodies))

    def gaps(self):
        return ["one liquid material: another that melts does so in place, as a wall",
                "the melt is inviscid (sigma-ground's melt viscosity is unusable: KNOWN_GAPS) and flows at the solid's "
                "density (it does not expand on melting: lead is 3.5% less dense molten), so mass and energy stay exact",
                "bodies translate only and stop dead on contact; a body that loses cells to melting is not split",
                "kinetic energy lost to FLIP's damping and to inelastic contact leaves the world (dissipated_J), not as heat",
                "steam is not modelled: boiled particles vent"]

    # ── one tick ──────────────────────────────────────────────────────────────────────────────────────────────────
    def step(self, dt):
        self._heat(dt)
        self._phase()
        E_mech0 = self.mechanical_J()
        left = float(dt)
        while left > 1e-12:
            sub = min(left, self.cfl * self.pitch / max(1e-9, self._max_speed()))
            self._update_solids()
            if len(self.flip.pos):
                n0 = len(self.flip.pos)
                self.flip.step(sub, cfl=self.cfl)
                assert len(self.flip.pos) == n0
            self._move_bodies(sub)
            self.t += sub
            left -= sub
        # gravity's work is in the mechanical energy itself; what is missing after the flow is what the step dissipated
        self.ledger_["dissipated_J"] += max(0.0, E_mech0 - self.mechanical_J())
        self._update_solids()
        self._draw()

    def run(self, n, dt):
        for _ in range(n):
            self.step(dt)

    # ── heat ──────────────────────────────────────────────────────────────────────────────────────────────────────
    def _assemble(self):
        """One grid of everything: material index, temperature, mass per cell; plus each particle's cell."""
        lab = np.where(self.static, self.label, 0)
        T = np.where(self.static, self.Ts, self.ambient_T)
        M = np.where(self.static, self.Ms, 0.0)
        for b in self.bodies:
            cells = self._body_cell_list(b)
            lab[tuple(cells.T)] = b["index"]
            T[tuple(cells.T)] = b["T"]
            M[tuple(cells.T)] = b["mass_kg"] / len(b["offsets"])
        pc = self._particle_cells()
        if len(pc):
            flat = np.ravel_multi_index(pc.T, self.label.shape)
            free = lab.ravel()[flat] == 0                                          # a particle pushed into a solid: none
            n = np.bincount(flat[free], minlength=lab.size).reshape(lab.shape)
            sT = np.bincount(flat[free], self.pT[free], minlength=lab.size).reshape(lab.shape)
            liq = n > 0
            lab[liq] = self.li
            T[liq] = sT[liq] / n[liq]
            M[liq] = n[liq] * self.flip.particle_mass
        else:
            flat, free, n = np.zeros(0, int), np.zeros(0, bool), np.zeros(lab.shape)
        return lab, T, M, flat, free, n

    def _prop(self, lab, T, key):
        out = np.zeros(lab.shape)
        for k in range(1, len(self.materials)):
            c = lab == k
            if c.any() and self.mat[k]["lv"].get(key) is not None:
                out[c] = np.interp(T[c], self.mat[k]["lv"]["T"], self.mat[k]["lv"][key])
        return out

    def _heat(self, dt):
        from ..dynamics.fields.heat import diffuse_fvm
        from .thermal_field import boundary_from_env

        lab, T, M, flat, free, n = self._assemble()
        occ = lab > 0
        if not occ.any():
            return
        cp, k = self._prop(lab, T, "cp"), self._prop(lab, T, "k")
        rho_cp = M / self.V * cp
        bc = boundary_from_env(self.env, occ, emissivity_grid=None)
        if "radiative" in bc:
            bc["radiative"]["emissivity"] = self._prop(lab, T, "emissivity") * occ
        T1, info = diffuse_fvm(T, k, rho_cp, self.pitch, total_time=dt, **bc)
        dE = rho_cp * self.V * (T1 - T)                                           # joules into each cell this step
        self.ledger_["radiated_J"] += info["radiated_J"]
        self.ledger_["to_surroundings_J"] += -float(dE.sum()) - info["radiated_J"]
        # back to the walls, the body cells and the particles, through their enthalpy
        for kk in range(1, len(self.materials)):
            c = self.static & (self.label == kk)
            if c.any():
                lv = self.mat[kk]["lv"]
                self.Ts[c] = _T(lv, _h(lv, self.Ts[c]) + dE[c] / self.Ms[c])
        for b in self.bodies:
            lv, cells = self.mat[b["index"]]["lv"], self._body_cell_list(b)
            per = b["mass_kg"] / len(b["offsets"])
            b["T"] = _T(lv, _h(lv, b["T"]) + dE[tuple(cells.T)] / per)
        if len(self.pT):
            lv = self.mat[self.li]["lv"]
            share = np.zeros(len(self.pT))
            share[free] = dE.ravel()[flat[free]] / (n.ravel()[flat[free]] * self.flip.particle_mass)
            self.pT = _T(lv, _h(lv, self.pT) + share)

    # ── phase ─────────────────────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _latent(lv, T, frac, T_line, L, heating):
        """Enthalpy method on arrays: past T_line, energy goes into ``frac`` (0..1) first. Returns (T, frac)."""
        if heating:
            m = (T > T_line) & (frac < 1)
        else:
            m = (T < T_line) & (frac > 0)
        if not m.any():
            return T, frac
        T, frac = T.copy(), frac.copy()
        E = np.abs(_h(lv, T[m]) - _h(lv, T_line))                                # J/kg past the line
        room = (1 - frac[m]) * L if heating else frac[m] * L
        done = E >= room
        sign = 1.0 if heating else -1.0
        frac[m] = np.where(done, 1.0 if heating else 0.0, frac[m] + sign * E / L)
        T[m] = np.where(done, _T(lv, _h(lv, T_line) + sign * (E - room)), T_line)
        return T, frac

    def _phase(self):
        L = self.mat[self.li] if self.li is not None else None
        # walls: melt / freeze / boil in place; the liquid's material, fully melted, becomes particles
        for kk in range(1, len(self.materials)):
            m = self.mat[kk]
            c = self.static & (self.label == kk)
            if not c.any() or m["T_melt"] is None:
                continue
            for heating in (True, False):
                T, f = self._latent(m["lv"], self.Ts[c], self.melt[c], m["T_melt"], m["L_fus"], heating)
                self.Ts[c], self.melt[c] = T, f
            if kk != self.li:
                for heating in (True, False):
                    full = c & (self.melt >= 1)
                    T, f = self._latent(m["lv"], self.Ts[full], self.boil[full], m["T_boil"], m["L_vap"], heating)
                    self.Ts[full], self.boil[full] = T, f
                gone = c & (self.boil >= 1)
                if gone.any():
                    self._vent_cells(gone, m)
        molten = self.static & (self.label == self.li) & (self.melt >= 1) if self.li is not None else None
        if molten is not None and molten.any():
            cells = np.argwhere(molten)
            self._spawn(cells, self.Ts[molten], self.Ms[molten], np.zeros((len(cells), 3)))
            self.static[molten] = False
            self.base[molten] = 0
            for a in (self.Ms, self.melt, self.boil):
                a[molten] = 0.0
            self.ledger_["melted_cells"] += len(cells)
        # body cells
        for b in list(self.bodies):
            m = self.mat[b["index"]]
            if m["T_melt"] is None:
                continue
            for heating in (True, False):
                b["T"], b["melt"] = self._latent(m["lv"], b["T"], b["melt"], m["T_melt"], m["L_fus"], heating)
            if b["index"] == self.li and (b["melt"] >= 1).any():
                go = b["melt"] >= 1
                per = b["mass_kg"] / len(b["offsets"])
                cells = self._body_cell_list(b)[go]
                self._spawn(cells, b["T"][go], np.full(go.sum(), per), np.tile(b["vel"], (go.sum(), 1)))
                keep = ~go
                b["offsets"], b["T"], b["melt"], b["boil"] = b["offsets"][keep], b["T"][keep], b["melt"][keep], b["boil"][keep]
                b["mass_kg"] = per * keep.sum()
                self.ledger_["melted_cells"] += int(go.sum())
                if not keep.any():
                    self.bodies.remove(b)
        # particles: freeze below T_melt, boil past T_boil, vent
        if len(self.pT):
            self.pT, self.pL = self._latent(L["lv"], self.pT, self.pL, L["T_melt"], L["L_fus"], False)
            liquid = self.pL >= 1
            T, f = self._latent(L["lv"], self.pT[liquid], self.pB[liquid], L["T_boil"], L["L_vap"], True)
            self.pT[liquid], self.pB[liquid] = T, f
            T, f = self._latent(L["lv"], self.pT, self.pB, L["T_boil"], L["L_vap"], False)
            self.pT, self.pB = T, f
            # a liquid particle reheated past T_melt takes its fusion heat back first (handled by the heating branch)
            self.pT, self.pL = self._latent(L["lv"], self.pT, self.pL, L["T_melt"], L["L_fus"], True)
            gone = self.pB >= 1
            if gone.any():
                pm = self.flip.particle_mass
                self.ledger_["vented_kg"] += pm * int(gone.sum())
                self.ledger_["vented_J"] += float((pm * (_h(L["lv"], self.pT[gone]) + L["L_fus"] + L["L_vap"])).sum())
                self.ledger_["vented_particles"] += int(gone.sum())
                self._drop(gone)
            self._freeze_cells()

    def _vent_cells(self, gone, m):
        self.ledger_["vented_kg"] += float(self.Ms[gone].sum())
        self.ledger_["vented_J"] += float((self.Ms[gone] * (_h(m["lv"], self.Ts[gone]) + m["L_fus"] + m["L_vap"])).sum())
        self.static[gone] = False
        self.base[gone] = 0
        for a in (self.Ms, self.melt, self.boil):
            a[gone] = 0.0

    def _spawn(self, cells, T, mass, vel):
        """Cells (full-grid indices) become particles: their mass (a whole number of particles: a cell of the liquid's
        material holds exactly that), temperature (fully liquid at it) and velocity, spread evenly through each cell."""
        f = self.flip
        pm = f.particle_mass
        per = round(f.ppc ** (1 / 3))
        sub = (np.stack(np.meshgrid(*[np.arange(per)] * 3, indexing="ij"), -1).reshape(-1, 3) + 0.5) / per
        pos, vs, Ts = [], [], []
        for c, t, mkg, v in zip(cells, T, mass, vel):
            n = int(round(float(mkg) / pm))
            assert abs(n * pm - float(mkg)) < 1e-9 * max(1.0, float(mkg)), (n, pm, mkg)
            if n == 0:
                continue
            pick = sub[np.arange(n) % len(sub)]
            jitter = (self.rng.random((n, 3)) - 0.5) / per
            pos.append((c - self.lo + pick + jitter) * f.h)
            vs.append(np.tile(v, (n, 1)))
            Ts.append(np.full(n, t))
        if not pos:
            return
        f.pos = np.concatenate([f.pos, *pos])
        f.vel = np.concatenate([f.vel, *vs])
        self.pT = np.concatenate([self.pT, *Ts])
        k = sum(len(p) for p in pos)
        self.pL = np.concatenate([self.pL, np.ones(k)])
        self.pB = np.concatenate([self.pB, np.zeros(k)])

    def _drop(self, mask):
        keep = ~mask
        f = self.flip
        f.pos, f.vel = f.pos[keep], f.vel[keep]
        self.pT, self.pL, self.pB = self.pT[keep], self.pL[keep], self.pB[keep]

    def _freeze_cells(self):
        """A cell whose particles have all frozen becomes a wall of the liquid's material with exactly their mass
        and the temperature of their mean enthalpy (not their mean temperature: cp changes with T); their momentum is
        stopped (counted as dissipated)."""
        if not len(self.pT):
            return
        pc = self._particle_cells()
        flat = np.ravel_multi_index(pc.T, self.label.shape)
        size = self.label.size
        n = np.bincount(flat, minlength=size)
        liquid_left = np.bincount(flat, self.pL > 0, minlength=size)
        cells = np.nonzero((n > 0) & (liquid_left == 0))[0]
        if not len(cells):
            return
        cells = cells[~self.static.ravel()[cells] & (self._bodies_mask().ravel()[cells] == 0)]
        if not len(cells):
            return
        pm = self.flip.particle_mass
        lv = self.mat[self.li]["lv"]
        sH = np.bincount(flat, _h(lv, self.pT), minlength=size)
        go = np.isin(flat, cells)
        self.ledger_["dissipated_J"] += 0.5 * pm * float((self.flip.vel[go] ** 2).sum())
        idx = np.unravel_index(cells, self.label.shape)
        self.static[idx] = True
        self.label[idx] = self.li
        self.base[idx] = self.li
        self.Ms[idx] = n[cells] * pm
        self.Ts[idx] = _T(lv, sH[cells] / n[cells])
        self.melt[idx] = 0.0
        self.boil[idx] = 0.0
        self.ledger_["frozen_cells"] += len(cells)
        self._drop(go)

    # ── what is drawn, and the temperatures that go with it ───────────────────────────────────────────────────────
    @property
    def T(self):
        """Each cell's temperature (what a theater's glow reads)."""
        return self.temperature_grid()

    def temperature_grid(self):
        """Each cell's temperature (walls, body cells, a liquid cell's particle mean); ambient elsewhere."""
        return self._assemble()[1]

    def _draw(self):
        if len(self.flip.pos):
            super()._draw()
        else:
            lab = self.base.copy()
            for b in self.bodies:
                lab[self._body_mask(b)] = b["index"]
            self.label = lab
        self.label[self.static] = self.base[self.static]
        self.drawn = self.label.copy()
        # a wall that froze from a thin splash holds less than a cell's mass: per material, exactly as many blocks are
        # drawn as the partly full walls' mass (the fullest first, ties dithered), as voxel_resample's keep="mass" does,
        # so a 3 mm sheet shows as an even checkerboard with its mass, not as a 1 cm slab three times too heavy
        for k in range(1, len(self.materials)):
            full = self.mat[k]["rho"] * self.V
            part = self.static & (self.label == k) & (self.Ms < full * (1 - 1e-9))
            if not part.any():
                continue
            cells = np.argwhere(part)
            n_draw = int(round(float(self.Ms[part].sum()) / full))
            dither = ((cells * np.array([1, 2, 3])).sum(1) % 4) / 4.0 + 1e-3 * self.rng.random(len(cells))
            order = np.lexsort((dither, -self.Ms[part]))
            hide = cells[order[n_draw:]]
            self.drawn[tuple(hide.T)] = 0

    def readout(self):
        L = self.ledger
        p = L["momentum_Ns"]
        n = len(self.flip.pos)
        hot = float(self.temperature_grid()[self.label > 0].max()) if (self.label > 0).any() else self.ambient_T
        return (f"t {self.t:.2f} s | {self.mass_kg():.4g} kg | {self.liquid_name} liquid {n * self.flip.particle_mass:.3g} kg "
                f"| melted {L['melted_cells']} froze {L['frozen_cells']} cells | hottest {hot:.0f} K | thermal "
                f"{self.thermal_J():.6g} J, radiated {L['radiated_J']:.4g} J | p ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) N s" +
                "".join(f" | {b['material']} z {b['pos'][2] * self.pitch:.3f} m v {b['vel'][2]:+.2f} m/s" for b in self.bodies))
