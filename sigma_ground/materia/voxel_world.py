"""
VoxelWorld: a material voxel build that RUNS. Each tick: heat → phase → movement.

A Deckard ``VoxelField`` (from a MineBench-style build, ``deckard.voxel_exec``)
says what every cell is made of. This keeps, per cell, what it currently IS
(temperature, melted fraction, boiled fraction) and steps the world forward,
composing laws sigma-ground already has, never new numbers:

  heat      ``dynamics.fields.heat.diffuse_fvm`` on ``thermal_field.material_grids``
            (k, ρ·cp per material from ``field.interface.thermal``), bounded by the
            theater's atmosphere (``boundary_from_env``: air → ambient bath, vacuum →
            Stefan-Boltzmann loss, ledgered).
  phase     enthalpy method on ``phase_transition.PHASE_DATA``: a cell past its
            melting point first absorbs the latent heat of fusion (held at T_melt,
            ``melt`` rising 0 → 1) and only then is liquid; freezing gives it back.
            A liquid past its boiling point absorbs the latent heat of vaporisation
            and, fully boiled, VENTS: it leaves the world, its mass and energy
            written to the ledger.
  movement  under the theater's gravity (−z): every face-connected solid piece is
            one rigid body that sinks one cell through any fluid less dense than it
            (air counts as massless) and rises one cell through a denser liquid
            (it floats at the surface, never into air); then liquids fall, sink
            through lighter liquids, and spread sideways when pushed from above.

Honest limits (each one reported, none papered over):
  - Latent heat is used only where PHASE_DATA's "J/mol" has an unambiguous mole:
    a pure element (mole = atom). For compounds its basis varies (water_ice is per
    H₂O, ceramic_alumina per effective atom), so they keep their phase and the
    gap is reported. Wood, glass and granite therefore never "melt" (wood chars).
  - A material with no thermal data (liquid mercury, liquid water) is inert to
    heat: it neither conducts nor changes temperature.
  - k and ρ·cp are frozen at their T_ref values (``material_grids``' flagged
    approximation); a melted cell keeps its solid's thermal properties.
  - A cell's MASS is its build density × block volume, forever: melting changes
    which liquid it sinks through (liquid density = ρ_solid / (1 + ΔV_fus), from
    PHASE_DATA) but not its block's size.
  - Movement is quasi-static settling, one cell per tick, not dynamics: no
    velocities, no toppling, no strength (a solid piece resting on anything
    stays). Real fluid dynamics is ``dynamics.continuum`` (SPH).
  - The air bath's exchange with the world is the RESIDUAL of the heat step
    (``to_surroundings_J``), not an independent measurement; the radiative ledger
    in vacuum is exact.
"""
from __future__ import annotations

SOLID, LIQUID = 0, 1


def _molar_mass_kg(name):
    """kg/mol when ``name`` is a pure element (its composition's first token is an
    element symbol), else None: PHASE_DATA's mole basis is only unambiguous there."""
    import periodictable
    from ..field.interface.surface import MATERIALS

    comp = str(MATERIALS.get(name, {}).get("composition", "")).split(" ")[0].split("(")[0]
    el = getattr(periodictable, comp, None) if comp.isalpha() and len(comp) <= 2 else None
    if el is None or not hasattr(el, "mass"):
        return None
    return float(el.mass) / 1000.0


def material_card(name, density_kg_m3):
    """Everything the tick needs about one material, and what is missing (``gaps``)."""
    from ..field.interface import thermal
    from ..field.interface.phase_transition import PHASE_DATA
    from ..field.interface.resolve import material_profile

    card = {"name": name, "rho": float(density_kg_m3), "rho_liquid": None,
            "k": 0.0, "rho_cp": 0.0, "T_melt": None, "T_boil": None,
            "L_fus": None, "L_vap": None,                       # J/m³ of the solid block
            "starts_liquid": material_profile(name).get("material_class") == "liquid",
            "gaps": []}
    try:
        card["k"] = float(thermal.thermal_conductivity(name))
        card["rho_cp"] = float(thermal.heat_capacity_volumetric(name))
    except KeyError:
        card["gaps"].append("no thermal data: inert to heat")
    data = PHASE_DATA.get(name)
    if data is None:
        if card["starts_liquid"]:
            card["rho_liquid"] = card["rho"]
        card["gaps"].append("no phase data: keeps its phase")
        return card
    card["rho_liquid"] = card["rho"] / (1.0 + float(data.get("delta_V_fus", 0.0)))
    M = _molar_mass_kg(name)
    if M is None:
        card["gaps"].append("latent heat has no unambiguous mole basis (not a pure element): keeps its phase")
    elif card["rho_cp"] <= 0:
        card["gaps"].append("no heat capacity: keeps its phase")
    else:
        card["T_melt"], card["T_boil"] = float(data["T_melt_K"]), float(data["T_boil_K"])
        card["L_fus"] = float(data["L_fus_J_mol"]) / M * card["rho"]
        card["L_vap"] = float(data["L_vap_J_mol"]) / M * card["rho"]
    return card


class VoxelWorld:
    """A ``VoxelField`` with per-cell state, stepped by ``step(dt)``.

    ``temperature_k``: a scalar or a grid shaped like the field (default: the
    atmosphere's temperature). ``env``: "STP" | "IRT" | "ISM" or a theater's
    ``physics_env``. ``gravity``: settle along −z.
    """

    def __init__(self, field, *, temperature_k=None, env="STP", gravity=True, seed=1):
        import numpy as np
        from ..field.interface.atmosphere import atmosphere_preset

        self.env = atmosphere_preset(env) if isinstance(env, str) else dict(env)
        self.ambient_T = float(self.env.get("temperature_k", 293.15))
        self.pitch = float(field.voxel_size)
        self.cell_m3 = self.pitch ** 3
        self.center = field.center
        self.materials = list(field.materials)
        self.label = np.array(field.label_grid, dtype=np.int32)
        dens = field.density_by_label or {}
        self.cards = [None] + [material_card(m, dens[m]) for m in self.materials[1:]]
        self.T = np.full(self.label.shape, self.ambient_T) if temperature_k is None else \
            np.broadcast_to(np.asarray(temperature_k, dtype=float), self.label.shape).copy()
        self.melt = np.zeros(self.label.shape)
        self.boil = np.zeros(self.label.shape)
        for k, c in enumerate(self.cards):
            if c is None:
                continue
            cells = self.label == k
            if c["starts_liquid"]:
                self.melt[cells] = 1.0
            if c["T_melt"] is not None:
                self.melt[cells & (self.T >= c["T_melt"])] = 1.0
        # the ground: the theater's ground plane sits under the build's lowest layer
        # (theaters.plain), so nothing settles below it
        self.floor_z = int(np.argwhere(self.label > 0)[:, 2].min())
        self.gravity = bool(gravity)
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.ledger = {"radiated_J": 0.0, "to_surroundings_J": 0.0, "vented_kg": 0.0, "vented_J": 0.0}

    # ── per-cell property grids from the cards ──
    def _grid(self, key):
        import numpy as np
        lut = np.array([0.0] + [float(c[key] or 0.0) for c in self.cards[1:]])
        return lut[self.label]

    def gaps(self):
        """{material: [what sigma-ground cannot supply for it]}"""
        return {c["name"]: list(c["gaps"]) for c in self.cards[1:] if c["gaps"]}

    def mass_kg(self):
        return float(self._grid("rho").sum()) * self.cell_m3

    def energy_J(self):
        """Sensible heat of every conducting cell + latent heat held by melt/boil."""
        rho_cp = self._grid("rho_cp")
        return float(((rho_cp * self.T) + self.melt * self._grid("L_fus") + self.boil * self._grid("L_vap")).sum()
                     * self.cell_m3)

    def phase(self):
        import numpy as np
        return np.where(self.label == 0, -1, np.where(self.melt >= 1.0, LIQUID, SOLID))

    # ── the tick ──
    def step(self, dt):
        events = {"t": self.t + dt}
        events.update(self._heat(dt))
        events.update(self._phase())
        if self.gravity:
            events.update(self._settle())
        self.t += dt
        return events

    def run(self, n, dt):
        return [self.step(dt) for _ in range(n)]

    def _heat(self, dt):
        from ..dynamics.fields.heat import diffuse_fvm
        from .thermal_field import boundary_from_env

        k, rho_cp = self._grid("k"), self._grid("rho_cp")
        if not (rho_cp > 0).any():
            return {"heat_substeps": 0}
        bc = boundary_from_env(self.env, self.label > 0, emissivity_grid=None)
        if "radiative" in bc:
            from ..field.interface.thermal import emissivity
            import numpy as np
            lut = np.array([0.0] + [float(emissivity(c["name"])) if c["rho_cp"] > 0 else 0.0
                                    for c in self.cards[1:]])
            bc["radiative"]["emissivity"] = lut[self.label]
        E0 = float((rho_cp * self.T).sum()) * self.cell_m3
        self.T, info = diffuse_fvm(self.T, k, rho_cp, self.pitch, total_time=dt, **bc)
        E1 = float((rho_cp * self.T).sum()) * self.cell_m3
        self.ledger["radiated_J"] += info["radiated_J"]
        self.ledger["to_surroundings_J"] += E0 - E1 - info["radiated_J"]
        return {"heat_substeps": info["substeps"]}

    def _phase(self):
        """Enthalpy method: past a transition, energy goes into the latent store first."""
        import numpy as np
        V = self.cell_m3
        melted = froze = vented = 0
        for k, c in enumerate(self.cards):
            if c is None or c["T_melt"] is None:
                continue
            cells, cap, Tm, Tb = self.label == k, c["rho_cp"], c["T_melt"], c["T_boil"]
            # melting / freezing through T_melt
            for heating in (True, False):
                m = cells & ((self.T > Tm) & (self.melt < 1) & (self.boil == 0) if heating
                             else (self.T < Tm) & (self.melt > 0))
                if not m.any():
                    continue
                E = np.abs(self.T[m] - Tm) * cap                      # J/m³ past the line
                room = (1 - self.melt[m]) * c["L_fus"] if heating else self.melt[m] * c["L_fus"]
                done = E >= room
                sign = 1.0 if heating else -1.0
                self.melt[m] = np.where(done, 1.0 if heating else 0.0, self.melt[m] + sign * E / c["L_fus"])
                self.T[m] = np.where(done, Tm + sign * (E - room) / cap, Tm)
                if heating:
                    melted += int(done.sum())
                else:
                    froze += int(done.sum())
            # boiling through T_boil (liquid only); condensing back while not yet vented
            for heating in (True, False):
                m = cells & (self.melt >= 1) & ((self.T > Tb) if heating else (self.T < Tb) & (self.boil > 0))
                if not m.any():
                    continue
                E = np.abs(self.T[m] - Tb) * cap
                room = (1 - self.boil[m]) * c["L_vap"] if heating else self.boil[m] * c["L_vap"]
                done = E >= room
                sign = 1.0 if heating else -1.0
                self.boil[m] = np.where(done, 1.0 if heating else 0.0, self.boil[m] + sign * E / c["L_vap"])
                self.T[m] = np.where(done, Tb + sign * (E - room) / cap, Tb)
            gone = cells & (self.boil >= 1)
            if gone.any():
                n = int(gone.sum())
                self.ledger["vented_kg"] += n * c["rho"] * V
                self.ledger["vented_J"] += float((cap * self.T[gone] + c["L_fus"] + c["L_vap"]).sum()) * V
                self.label[gone] = 0
                self.T[gone], self.melt[gone], self.boil[gone] = self.ambient_T, 0.0, 0.0
                vented += n
        return {"melted": melted, "froze": froze, "vented": vented}

    # ── movement ──
    def _swap(self, a, b):
        for arr in (self.label, self.T, self.melt, self.boil):
            arr[a], arr[b] = arr[b], arr[a]

    def _density_at(self, idx):
        """Density a cell offers to something moving through it: air 0, liquid its liquid density."""
        k = self.label[idx]
        if k == 0:
            return 0.0
        c = self.cards[k]
        return c["rho_liquid"] if self.melt[idx] >= 1 else None   # None: a solid, not a fluid

    def _settle(self):
        import numpy as np
        import scipy.ndimage as ndi

        nz = self.label.shape[2]
        # 1) rigid solid pieces, lowest first
        solid = (self.label > 0) & (self.melt < 1)
        comp, n = ndi.label(solid)
        rho = self._grid("rho")
        moved_pieces = 0
        # each piece's cells, taken now: a move swaps only that piece's cells with fluid
        # cells, so the other pieces' lists stay true; pieces that end up touching merge
        # next tick
        pieces = [np.argwhere(comp == i + 1) for i in range(n)]
        for cells in sorted(pieces, key=lambda c: c[:, 2].min()):
            density = float(rho[tuple(cells.T)].mean())
            runs = {}
            for x, y, z in cells:                       # vertical runs per column
                runs.setdefault((x, y), []).append(z)
            spans = []
            for (x, y), zs in runs.items():
                zs = sorted(zs)
                start = zs[0]
                for a, b in zip(zs, zs[1:] + [None]):
                    if b != a + 1:
                        spans.append((x, y, start, a))
                        start = b
            for direction in (-1, +1):
                ok = True
                for x, y, z0, z1 in spans:
                    z = z0 - 1 if direction < 0 else z1 + 1
                    if not self.floor_z <= z < nz:
                        ok = False
                        break
                    d = self._density_at((x, y, z))
                    if d is None or (direction < 0 and not d < density) or \
                            (direction > 0 and (self.label[x, y, z] == 0 or not d > density)):
                        ok = False
                        break
                if ok:
                    for x, y, z0, z1 in spans:          # the run shifts; the fluid cell trades ends
                        rng_ = range(z0, z1 + 1) if direction < 0 else range(z1, z0 - 1, -1)
                        for z in rng_:
                            self._swap((x, y, z), (x, y, z + direction))
                    moved_pieces += 1
                    break
        # 2) liquids, bottom up
        liquid_moves = 0
        liq = np.argwhere((self.label > 0) & (self.melt >= 1))
        liq = liq[np.lexsort((self.rng.random(len(liq)), liq[:, 2]))]
        nx, ny = self.label.shape[:2]
        for x, y, z in liq:
            if not (self.label[x, y, z] > 0 and self.melt[x, y, z] >= 1) or z == self.floor_z:
                continue
            here = self.cards[self.label[x, y, z]]["rho_liquid"]
            below = self._density_at((x, y, z - 1))
            if below is not None and below < here:
                self._swap((x, y, z), (x, y, z - 1))
                liquid_moves += 1
                continue
            pushed = z + 1 < self.label.shape[2] and self.label[x, y, z + 1] > 0 and self.melt[x, y, z + 1] >= 1
            sides = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            self.rng.shuffle(sides)
            for sx, sy in sides:
                if not (0 <= sx < nx and 0 <= sy < ny) or self.label[sx, sy, z] != 0:
                    continue
                if pushed or self.label[sx, sy, z - 1] == 0:
                    self._swap((x, y, z), (sx, sy, z))
                    liquid_moves += 1
                    break
        return {"solid_pieces_moved": moved_pieces, "liquid_moves": liquid_moves}

    def to_field(self):
        """The world now, as a ``VoxelField`` (renderable), carrying ``temperature_k``
        and ``melt_fraction`` as per-cell scalar fields."""
        from ..deckard.voxelize import _finalize
        lo = tuple(c - n * self.pitch / 2 for c, n in zip(self.center, self.label.shape))
        rho = {c["name"]: c["rho"] for c in self.cards[1:]}
        f = _finalize(self.label.copy(), self.materials, self.pitch, lo, rho.__getitem__, 1.0,
                      extra_note=f"VoxelWorld at t = {self.t:.6g} s")
        f.scalar_fields = {"temperature_k": self.T.copy(), "melt_fraction": self.melt.copy()}
        return f


__all__ = ["VoxelWorld", "material_card", "SOLID", "LIQUID"]
