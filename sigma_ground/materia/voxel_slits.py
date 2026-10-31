"""
SlitScreen: the double-slit experiment, one photon at a time, on a screen of voxels.

A laser shines on two slits; the screen behind them is a wall of blocks, each ``pitch`` m across (1 mm by default: the
screen is drawn at true scale, so a 5.2 mm fringe spacing is 5.2 blocks). Photons arrive ONE AT A TIME: each lands at a
random place drawn from the probability density of ``field.interface.quantum.double_slit_intensity`` (exact path lengths
from each slit, the single-slit envelope, and Englert-Greenberger-Yasin which-path distinguishability D). Early on the
screen is scattered dots; the fringes appear only as the dots pile up (Taylor 1909's faint-light experiment).

Three things to change at the screen, each starting a fresh run:
  - observe: someone learns which slit each photon went through (D = 1): the fringes vanish, two slits' worth of
    envelope remains;
  - one slit: close the second slit (``quantum.single_slit_intensity``): the single-slit envelope alone;
  - colour: the laser's wavelength (650, 532, 405 nm): the fringes are spaced λL/d.

The cells' looks are the laser's colour (``atomic_spectra.wavelength_to_rgb``) at five brightness levels by how many
photons each cell has caught relative to the busiest cell.

Exhibit choices (not physics, stated): slit width 0.04 mm, separation 0.25 mm, screen 2 m away (a classic teaching
setup), and a laser beam expanded to 12 mm (1/e² diameter) so the stripes are tall enough to see; along the slits the
pattern is just the beam's Gaussian profile.
"""
from __future__ import annotations

import math

import numpy as np

WAVELENGTHS_NM = (650, 532, 405)
LEVELS = 5


class SlitScreen:
    def __init__(self, *, width_blocks=128, height_blocks=32, pitch=1e-3, slit_width=0.04e-3, separation=0.25e-3,
                 distance=2.0, beam_diameter=12e-3, wavelength_nm=650, seed=1):
        self.pitch = float(pitch)
        self.W, self.H = int(width_blocks), int(height_blocks)
        self.a, self.d, self.L = float(slit_width), float(separation), float(distance)
        self.beam_sigma = float(beam_diameter) / 4                      # 1/e² diameter = 4σ
        self.materials = ["air", "screen"] + [f"light {nm} {lv}" for nm in WAVELENGTHS_NM for lv in range(1, LEVELS + 1)]
        self.rng = np.random.default_rng(seed)
        self.wavelength_nm = int(wavelength_nm)
        self.observed = False
        self.one_slit = False
        self.t = 0.0
        self.ticks = 0
        self.reset()

    # ── the experiment ─────────────────────────────────────────────────────────────────────────────────────────────
    def density(self, x):
        """Probability density (unnormalised) of a photon landing at screen position ``x`` (m, 0 = on axis)."""
        from ..field.interface.quantum import double_slit_intensity, single_slit_intensity
        lam = self.wavelength_nm * 1e-9
        if self.one_slit:
            return single_slit_intensity(x, self.L, lam, self.a, offset=-self.d / 2)
        return double_slit_intensity(x, self.d, self.L, lam, D=1.0 if self.observed else 0.0, a=self.a)

    def fringe_spacing_m(self):
        """λL/d (the same formula as Mentat's ``double_slit_fringe_spacing``)."""
        return self.wavelength_nm * 1e-9 * self.L / self.d

    def envelope_zero_m(self):
        """Where the single-slit envelope first goes dark: λL/a (``quantum`` small-angle form)."""
        return self.wavelength_nm * 1e-9 * self.L / self.a

    def reset(self):
        """A fresh screen: no photons yet; the arrival distribution for the current setup, sampled 16 times per block."""
        n = self.W * 16
        xs = ((np.arange(n) + 0.5) / 16 - self.W / 2) * self.pitch
        p = np.array([self.density(x) for x in xs])
        self._cdf = np.cumsum(p) / p.sum()
        self._xs = xs
        self.counts = np.zeros((self.W, self.H), np.int64)
        self.photons = 0
        self.ticks = 0
        self._draw()

    def press(self, what):
        """``observe`` | ``one_slit`` | ``colour``: change the setup and start again. Returns what changed, in words."""
        if what == "observe":
            self.observed = not self.observed
            msg = "Which-path now OBSERVED: no fringes." if self.observed else "Which-path unknown again: fringes."
        elif what == "one_slit":
            self.one_slit = not self.one_slit
            msg = "One slit open." if self.one_slit else "Both slits open."
        else:
            i = (WAVELENGTHS_NM.index(self.wavelength_nm) + 1) % len(WAVELENGTHS_NM)
            self.wavelength_nm = WAVELENGTHS_NM[i]
            msg = f"Laser now {self.wavelength_nm} nm: fringes {self.fringe_spacing_m() * 1e3:.2f} mm apart."
        self.reset()
        return msg

    def step(self, dt):
        """Let a batch of photons through: one at first, growing by a quarter each tick, up to 4000 a tick."""
        n = min(4000, int(1.25 ** self.ticks))
        u = self.rng.random(n)
        x = self._xs[np.minimum(np.searchsorted(self._cdf, u), len(self._xs) - 1)]
        x = x + (self.rng.random(n) - 0.5) * self.pitch / 16
        y = self.rng.normal(0.0, self.beam_sigma, n)
        i = np.floor(x / self.pitch + self.W / 2).astype(int)
        k = np.floor(y / self.pitch + self.H / 2).astype(int)
        ok = (i >= 0) & (i < self.W) & (k >= 0) & (k < self.H)
        np.add.at(self.counts, (i[ok], k[ok]), 1)
        self.photons += n
        self.ticks += 1
        self.t += dt
        self._draw()

    def _draw(self):
        lab = np.full((self.W, 1, self.H), 1, np.int32)                # the bare screen
        top = self.counts.max()
        if top:
            level = np.ceil(LEVELS * self.counts / top).astype(int)
            base = 2 + WAVELENGTHS_NM.index(self.wavelength_nm) * LEVELS
            lab[:, 0, :] = np.where(level > 0, base + level - 1, 1)
        self.label = lab

    # ── looks and books ────────────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def light_rgb(wavelength_nm, level):
        """The laser's colour at a brightness level (1..LEVELS)."""
        from ..field.interface.atomic_spectra import wavelength_to_rgb
        return tuple(c * level / LEVELS for c in wavelength_to_rgb(wavelength_nm))

    def readout(self):
        setup = "one slit" if self.one_slit else ("two slits, which-path OBSERVED" if self.observed else "two slits")
        return (f"{self.wavelength_nm} nm | {setup} | {self.photons:,} photons | fringes lambda L/d "
                f"{self.fringe_spacing_m() * 1e3:.2f} mm | envelope dark at {self.envelope_zero_m() * 1e3:.1f} mm")

    def gaps(self):
        return ["the laser's colour is Bruton's spectral approximation (atomic_spectra.wavelength_to_rgb)",
                "the beam's 12 mm height and the slit dimensions are the exhibit's choice"]
