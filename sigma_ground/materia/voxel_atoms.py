"""
AtomicZoom: a photon meets the surface of the osmium grain, seen at the scale of its atoms. Numbers from ``field.interface.osmium_atom``.

Two panels, one clock, half an angstrom a block, 0.2 femtoseconds a tick (a visible wave flips every 0.8 fs):
  LEFT   the surface: five close-packed layers of osmium atoms (hcp, the library's lattice constant) and the electron sea that fills them
         (two conduction electrons an atom: ~2% of the blocks, a fizz of dots, never at rest). Above it the photon's field: at this scale a
         500 nm wave is a uniform field, 10,000 blocks to a wavelength, that flips every half period. The sea follows it: it sloshes, and
         what a sloshing sea of charge does is radiate: the photon comes back up, reflected, within a skin of ~14 nm of the surface.
  RIGHT  one atom, in a slice, on a LOGARITHMIC radial scale (each ring is farther by the factor the library's Slater radius says; the
         nucleus, 7 fm, is a hundred thousand times smaller than the outer ring and would be off the picture). 76 electrons as dots that
         jump about on their rings (a cloud, never an orbit). Only the outermost, the 6s that belong to the sea, follow the field: the inner
         groups are held by tens to thousands of eV and the photon has 2.5.

An ultraviolet photon (50 nm, above the plasma frequency) is not turned back: the sea cannot follow it fast enough, and it goes in.

Exaggerated, and said so: the sea's and the outer electrons' swing (a photon's push is far below an atom's diameter). The clock is the real
femtoseconds, so the slow-down factor is on the screen (about 10^15).
"""
from __future__ import annotations

import math

import numpy as np

from ..field.interface import levitated_grain as lg
from ..field.interface import osmium_atom as oa

PITCH = 0.5e-10
W, D, H = 100, 10, 36
JC = 5
LEFT = (0, 50)                               # the lattice panel's columns
ATOM_I, ATOM_K = 76, 17                      # the atom's centre
TICKS_PER_PERIOD = 8
DESCENT, SLOSH, RETURN, AFTER = 60, 60, 60, 60
T_SLOSH, T_RETURN, T_AFTER = DESCENT, DESCENT + SLOSH, DESCENT + SLOSH + RETURN
TICKS = T_AFTER + AFTER
LAYERS = 5
TOP_K = 26                                   # the surface layer's height
SEA_SWING_BLOCKS, ATOM_SWING_BLOCKS = 2.0, 3.0   # exaggerations
WAVELENGTHS_M = (500e-9, 50e-9)              # green, which the sea turns back, and hard ultraviolet, which it does not
TICK_VIEW_S = 1.0 / 6.0

MATERIALS = ["air", "osmium", "sea", "photon", "field", "ring", "ring bound", "electron", "glow", "grey"]
LOOKS = {"osmium": "netherite_block", "sea": "diamond_block", "photon": "glowstone", "field": "lapis_block", "ring": "gold_block",
         "ring bound": "polished_deepslate", "electron": "quartz_block", "glow": "sea_lantern", "grey": "iron_block"}
M = {name: i for i, name in enumerate(MATERIALS)}


class AtomicZoom:
    def __init__(self, seed=None):
        self.pitch = PITCH
        self.materials = list(MATERIALS)
        self.colour = 0
        self.t = 0.0
        self._seed = seed
        self.lat = oa.lattice()
        self.groups = oa.shell_groups()
        self.sea_props = oa.sea()
        self.reset()

    def reset(self, seed=None):
        self.rng = np.random.default_rng(self._seed if seed is None and self._seed is not None else seed)
        self.light = oa.light(WAVELENGTHS_M[self.colour])
        self.ticks = 0
        self.t = 0.0
        self.label = np.zeros((W, D, H), np.int32)
        self._atoms = self._lattice_sites()
        self._radii = self._ring_radii()
        self._draw()

    def press(self, what):
        if what != "colour":
            return ""
        self.colour = (self.colour + 1) % len(WAVELENGTHS_M)
        self.reset()
        lam = WAVELENGTHS_M[self.colour]
        if self.light["reflected"]:
            return (f"{lam * 1e9:.0f} nm light: below the sea's plasma frequency ({self.light['omega_over_plasma']:.2f} of it), so the sea follows the "
                    f"field and turns the photon back.")
        return f"{lam * 1e9:.0f} nm light: above the plasma frequency ({self.light['omega_over_plasma']:.1f}x), the sea cannot follow it: the photon goes in."

    def step(self, dt):
        self.ticks = min(self.ticks + 1, TICKS)
        self.t += dt
        self._draw()

    # ── time and the field ─────────────────────────────────────────────────────────────────────────────────────────
    def phase(self):
        n = self.ticks
        return ("descent" if n < T_SLOSH else "slosh" if n < T_RETURN else "return" if n < T_AFTER else "after")

    def real_time_s(self):
        return self.ticks * self.light["period_s"] / TICKS_PER_PERIOD

    def slow_down(self):
        return TICK_VIEW_S / (self.light["period_s"] / TICKS_PER_PERIOD)

    def envelope(self):
        """How strongly the sea is driven: a smooth pulse while the photon is at the surface."""
        n = self.ticks
        if not (T_SLOSH <= n < T_RETURN):
            return 0.0
        return math.sin(math.pi * (n - T_SLOSH) / SLOSH) ** 2

    def swing(self):
        """The exaggerated displacement factor, -1..1: the envelope times the field's cosine."""
        return self.envelope() * math.sin(2 * math.pi * self.ticks / TICKS_PER_PERIOD)

    # ── the structures ─────────────────────────────────────────────────────────────────────────────────────────────
    def _lattice_sites(self):
        a = self.lat["a_m"] / PITCH
        c = self.lat["layer_spacing_m"] / PITCH
        sites = []
        for layer in range(LAYERS):
            k = TOP_K - layer * c
            off = (a / 2.0) if layer % 2 else 0.0
            i = 2.0 + off
            while i < LEFT[1] - 1:
                sites.append((int(round(i)), int(round(k))))
                i += a
        return sites

    def _ring_radii(self):
        """Blocks from the centre for each group: a logarithmic scale, 4.7 blocks a decade, the smallest ring at 3 blocks."""
        logs = [math.log10(g["radius_m"] * 1e10) for g in self.groups]
        return [3.0 + 4.7 * (lg_ - logs[0]) for lg_ in logs]

    # ── drawing ────────────────────────────────────────────────────────────────────────────────────────────────────
    def _put(self, i, k, name, j=JC):
        i, k = int(round(i)), int(round(k))
        if 0 <= i < W and 0 <= k < H and 0 <= j < D:
            self.label[i, j, k] = M[name]

    def _draw(self):
        lab = self.label
        lab[:] = 0
        self._draw_lattice()
        self._draw_atom()
        self._draw_field()
        self._draw_photon()
        self._draw_gauge()

    def _draw_lattice(self):
        lab = self.label
        occupied = np.zeros((W, H), bool)
        for i, k in self._atoms:
            lab[i - 1:i + 2, JC - 1:JC + 2, k - 1:k + 2] = M["osmium"]
            occupied[max(0, i - 1):i + 2, max(0, k - 1):k + 2] = True
        # the sea: two electrons an atom, a fizz of dots, shifted together by the field (exaggerated)
        p = self.sea_props["n_e_m3"] * PITCH ** 3                                  # electrons per block
        shift = int(round(self.swing() * SEA_SWING_BLOCKS)) if self.light["reflected"] else 0
        k_lo, k_hi = TOP_K - (LAYERS - 1) * int(self.lat["layer_spacing_m"] / PITCH) - 2, TOP_K + 2
        rnd = self.rng.random((LEFT[1], 5, k_hi - k_lo + 1))
        for i, j, k in zip(*np.nonzero(rnd < p)):
            ii, kk = int(i) + shift, int(k) + k_lo
            if 0 <= ii < LEFT[1] and 0 <= ii < W and not occupied[ii, kk]:
                lab[ii, JC - 2 + int(j), kk] = M["sea"]

    def _draw_atom(self):
        """One atom in a slice: a ring for each group, its electrons as dots that jump about on it; the outer groups follow the field."""
        shift = self.swing() * ATOM_SWING_BLOCKS if self.light["reflected"] else 0.0
        for gi, (g, r) in enumerate(zip(self.groups, self._radii)):
            outer = gi >= len(self.groups) - 1                                    # the 6s: the sea's electrons
            dx = shift if outer else 0.0
            name = "ring" if outer else "ring bound"
            for n in range(int(2 * math.pi * r * 1.2)):
                ang = 2 * math.pi * n / int(2 * math.pi * r * 1.2)
                if gi in (len(self.groups) - 2, len(self.groups) - 1) or n % 2 == 0:       # the valence rings solid, the bound ones dotted
                    self._put(ATOM_I + dx + r * math.cos(ang), ATOM_K + r * math.sin(ang), name)
            for _ in range(g["electrons"]):                                      # a cloud: the dots jump every tick
                a = float(self.rng.uniform(0, 2 * math.pi))
                self._put(ATOM_I + dx + r * math.cos(a), ATOM_K + r * math.sin(a), "electron", JC + 1)
        self._put(ATOM_I, ATOM_K, "glow")                                           # the nucleus: far smaller than a block

    def _draw_field(self):
        """The photon's field above both panels: one colour while it points up, another while it points down, flipping every half period."""
        up = self.swing() >= 0 if self.envelope() > 0.02 else None
        for i in range(W):
            if up is not None:
                self._put(i, H - 2, "photon" if up else "field")
                self._put(i, H - 3, "photon" if up else "field")
            else:
                self._put(i, H - 2, "grey")

    def _draw_gauge(self):
        """The sea's displacement against the lattice, as a marker on a scale (exaggerated like the swing itself): the collective slosh the
        fizz of dots hides."""
        for i in range(5, 46):
            self._put(i, 3, "grey")
        self._put(25, 4, "grey")
        mark = 25 + (self.swing() * 18 if self.light["reflected"] else 0)
        for dk in (0, 1):
            self._put(mark, 3 + dk, "sea")

    def _draw_photon(self):
        n = self.ticks
        reflected = self.light["reflected"]
        # the packet: down toward the surface, then (reflected) back up, or (ultraviolet) on into the lattice
        if n < T_SLOSH:
            k = (H - 5) - (H - 5 - (TOP_K + 3)) * n / DESCENT
        elif n < T_RETURN:
            k = TOP_K + 3 - (0 if reflected else (n - T_SLOSH) / SLOSH * 14)
        elif n < T_AFTER:
            k = (TOP_K + 3) + (H - 5 - (TOP_K + 3)) * (n - T_RETURN) / RETURN if reflected else TOP_K + 3 - 14
        else:
            k = None
        if k is not None and (reflected or n < T_AFTER):
            for i in (LEFT[1] // 2 - 1, LEFT[1] // 2, ATOM_I - 1, ATOM_I):
                self._put(i, k, "photon", JC - 1)
                self._put(i, k + 1, "photon", JC - 1)

    # ── what is said ───────────────────────────────────────────────────────────────────────────────────────────────
    def captions(self):
        lat, s = self.lat, self.sea_props
        g = self.groups
        n_atoms = oa.atoms_in_grain(lg.GRAIN_RADIUS_M)
        return [
            (25, 0, 31.0, f"THE SURFACE: osmium atoms (hcp, {lat['a_m'] * 1e10:.2f} A apart) and the electron sea, 2 per atom"),
            (25, 0, 30.0, "half an angstrom a block. The sea is never at rest."),
            (25, 0, 1.5, "sea displacement (exaggerated): the whole sea sloshes together"),
            (25, 0, 6.5, f"light turns back within {s['skin_depth_m'] * 1e9:.0f} nm of the surface: the sea's plasma energy is {s['plasma_energy_ev']:.0f} eV"),
            (ATOM_I, 0, 31.0, "ONE ATOM: 76 electrons (a slice, log scale)"),
            (ATOM_I, 0, 30.0, f"rings: {g[0]['radius_m'] * 1e10:.3f} A to {g[-1]['radius_m'] * 1e10:.1f} A. Dots are clouds, not orbits."),
            (ATOM_I, 0, 4.5, f"only the outer 6s follow the field. The nucleus ({oa.nucleus_radius_m() * 1e15:.0f} fm) is far smaller than a block"),
            (75, 0, 1.5, f"the whole grain's {n_atoms:.1e} atoms share the photon's momentum: {oa.recoil_per_atom(WAVELENGTHS_M[self.colour], lg.GRAIN_RADIUS_M):.1e} kg m/s each"),
        ]

    def readout(self):
        ph = self.phase()
        lam = WAVELENGTHS_M[self.colour]
        what = {"descent": "the photon comes down to the surface", "slosh": "the sea sloshes with the field (swing exaggerated)",
                "return": "the photon goes back up: reflected" if self.light["reflected"] else "the photon goes in: the sea cannot follow",
                "after": "the sea is at rest again; the lattice took the momentum"}[ph]
        return (f"Atoms | {lam * 1e9:.0f} nm | real time {self.real_time_s() * 1e15:.1f} fs (slowed {self.slow_down():.1e}x) | {what} | "
                f"light omega = {self.light['omega_over_plasma']:.2f} of the plasma frequency")

    def gaps(self):
        return oa.ATOM_GAPS
