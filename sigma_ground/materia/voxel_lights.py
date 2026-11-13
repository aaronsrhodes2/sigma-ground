"""
LightsUp: a grain of osmium resting, then floating, then the lights come on, in real slow motion. The numbers come from
``field.interface.levitated_grain`` (the contact, the field, a photon's effect) and ``field.interface.schrodinger_cat`` (when the
light records it).

The stage is at 5 micrometres a block: the grain (20 um radius) is a voxelized sphere 4 blocks in radius, the floor a slab with a magnet
under it, a lamp on the left, an eye on the right. Nothing here is what a physical observer would see: the diorama is the STATE, drawn from
outside physics, and the grain is drawn as a density of blocks equal to its weight in each place it could be.

  REST   the grain sits on the floor. Its weight (nanonewtons) goes through a contact patch tens of nanometres across (a thousandth of a
         block) at tens of megapascals. The floor records where it is: it is in one place.
  LIFT   the superconducting grain rises on the magnet's field, which pushes up on it (osmium expels the field below its critical field):
         no contact, so nothing records where it is. It is now in two places at once: two ghost grains 100 um apart, each half drawn,
         joined by a bridge: cyan, the coherence in principle (the sealed box), gold, in practice.
  WAIT   the lamp comes on, a cold lamp, in slow motion (the clock shows the real milliseconds, slowed by thousands). Light streams out
         in all directions; almost none of it meets the grain (a grain is a ten-billionth of the sky from the lamp). Until one photon does.
         The time to the first is drawn from the rate the library computes (Poisson).
  FLIGHT the photon that will land, in slow motion a few billion times slower than real (it crosses a metre in 3.3 ns): it too is in two
         places, one component toward each ghost grain.
  AFTER  it scatters: in each world the photon now carries where the grain is, and goes on to the eye. The gold bridge is cut (the
         practice coherence is gone: that is the change); the cyan one stays (the box is sealed). The grain has not changed at all
         (a photon's kick is 1e-18 m/s, its heating 1e-13 K): what changed is what is recorded. The observer is in one of the two worlds.

Exhibit choices (stated, not physics): the distances in the diorama are compressed (the lamp is a metre away, drawn 60 blocks = 300 um
away: the clock counts the real metre's 3.3 ns); the number of photons drawn (a few per tick, none of which hits); the phase lengths.
"""
from __future__ import annotations

import math

import numpy as np

from ..field.interface import levitated_grain as lg
from ..field.interface import schrodinger_cat as sc

PITCH = 5.0e-6
W, D, H = 100, 10, 36
JC = 5                                       # the plane the diorama is drawn in (the grain's centre)
FLOOR_TOP = 3                                # the floor's top block row; the magnet is under it
R_BLOCKS = 4                                 # the grain's radius, in blocks (20 um at 5 um a block)
CK_REST = FLOOR_TOP + 1 + R_BLOCKS           # the grain's centre height at rest
CK_FLOAT = 20
CI_REST, CI_LEFT, CI_RIGHT = 68, 58, 78      # 20 blocks = 100 um between the two places
LAMP_I, LAMP_K = 8, 20                       # the lamp's centre
EYE_I = 95
REST_TICKS, LIFT_TICKS, WAIT_TICKS, FLIGHT_TICKS, AFTER_TICKS = 60, 40, 130, 80, 50
T_LIFT = REST_TICKS
T_LAMP = REST_TICKS + LIFT_TICKS
T_FLIGHT = T_LAMP + WAIT_TICKS
T_AFTER = T_FLIGHT + FLIGHT_TICKS
TICKS = T_AFTER + AFTER_TICKS
LAMPS = ((1.0e-9, 500e-9), (1.0e-6, 500e-9), (1.0e-3, 500e-9))      # a nanowatt, a microwatt, a milliwatt of green light
LAMP_DISTANCE_M = 1.0                        # the real distance, for the clock (the diorama compresses it)
TICK_S = 1.0 / 6.0                           # a tick, in the viewer's seconds

MATERIALS = ["air", "osmium", "floor", "magnet", "lamp", "frame", "eye", "stress", "photon", "bridge", "bridge practice", "field", "you",
             "bar", "bar fill", "bar practice"]
LOOKS = {"osmium": "netherite_block", "floor": "smooth_stone", "magnet": "iron_block", "lamp": "sea_lantern", "frame": "polished_deepslate",
         "eye": "quartz_block", "stress": "redstone_block", "photon": "glowstone", "bridge": "diamond_block", "bridge practice": "gold_block",
         "field": "lapis_block", "you": "amethyst_block", "bar": "iron_block", "bar fill": "diamond_block", "bar practice": "gold_block"}
M = {name: i for i, name in enumerate(MATERIALS)}


def _sphere(ci, ck, r=R_BLOCKS):
    """The cells of a voxelized sphere of radius ``r`` blocks centred on (ci, JC, ck)."""
    out = []
    for i in range(ci - r, ci + r + 1):
        for j in range(JC - r, JC + r + 1):
            for k in range(ck - r, ck + r + 1):
                if (i - ci) ** 2 + (j - JC) ** 2 + (k - ck) ** 2 <= (r + 0.3) ** 2:
                    out.append((i, j, k))
    return out


def _density(i, j, k):
    return (((i * 73856093) ^ (j * 19349663) ^ (k * 83492791)) % 1000) / 1000.0


class LightsUp:
    def __init__(self, seed=None):
        self.pitch = PITCH
        self.materials = list(MATERIALS)
        self.lamp_i = 1
        self.t = 0.0
        self._seed = seed
        self.hertz = lg.hertz_contact()
        self.photon = lg.photon_effects(LAMPS[1][1])
        self.reset()

    # ── the run ────────────────────────────────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None):
        self.rng = np.random.default_rng(self._seed if seed is None and self._seed is not None else seed)
        power, lam = LAMPS[self.lamp_i]
        d = sc.lump_decoherence_times(0.0, radius_m=lg.GRAIN_RADIUS_M, separation_m=lg.GRAIN_SEPARATION_M, lamp_w=power, lamp_wavelength_m=lam)
        self.rate_hit = d["rate_lamp_per_s"]                                    # photons per second that land on the grain and bounce
        self.share = sc.resolving_fraction_wavelength(lg.GRAIN_SEPARATION_M, lam)    # the share of "which place" each carries
        self.t_hit = float(self.rng.exponential(1.0 / self.rate_hit))          # the first one: Poisson
        self.world = "left" if self.rng.random() < 0.5 else "right"             # which world the observer is in (Born: even odds)
        self.ticks = 0
        self.t = 0.0
        self.photons = []                                                       # (i, k, di, dk): representative photons, none of which hits
        self._state()
        self.label = np.zeros((W, D, H), np.int32)
        self._draw()

    def press(self, what):
        if what != "lamp":
            return ""
        self.lamp_i = (self.lamp_i + 1) % len(LAMPS)
        power, lam = LAMPS[self.lamp_i]
        self.reset()
        return (f"A {power:.0e} W lamp of cold green light: on average a photon lands on the grain every "
                f"{1e3 / self.rate_hit:.3g} ms, so the wait is about {1e3 / self.rate_hit:.3g} ms.")

    def step(self, dt):
        self.ticks = min(self.ticks + 1, TICKS)
        self.t += dt
        self._state()
        self._advance_photons()
        self._draw()

    # ── the state at this tick ─────────────────────────────────────────────────────────────────────────────────────
    def phase(self):
        n = self.ticks
        return ("rest" if n < T_LIFT else "lift" if n < T_LAMP else "wait" if n < T_FLIGHT else "flight" if n < T_AFTER else "after")

    def real_time_s(self):
        """The real seconds since the lamp came on (the first photon lands at ``t_hit``), or None before. In flight: the real flight time
        elapsed, and after the scatter, the same on to the eye."""
        n = self.ticks
        if n < T_LAMP:
            return None
        if n < T_FLIGHT:
            return self.t_hit * (n - T_LAMP) / WAIT_TICKS
        return lg.flight_time_s(LAMP_DISTANCE_M) * min(1.0, (n - T_FLIGHT) / FLIGHT_TICKS)

    def slow_down(self):
        """How many times slower than real the clock is running now."""
        n = self.ticks
        if n < T_LAMP:
            return 1.0
        per_tick = (self.t_hit / WAIT_TICKS) if n < T_FLIGHT else (lg.flight_time_s(LAMP_DISTANCE_M) / FLIGHT_TICKS)
        return TICK_S / per_tick

    def _state(self):
        n = self.ticks
        lift = min(1.0, max(0.0, (n - T_LIFT) / LIFT_TICKS))
        self.height = CK_REST + (CK_FLOAT - CK_REST) * lift               # the grain's centre height
        self.split = n >= T_LAMP - 10                                      # in two places once it floats (the ghost grains appear)
        self.coh_in_principle = 1.0 if self.split else 0.0
        self.coh_in_practice = 1.0 if (self.split and n < T_AFTER) else 0.0
        self.contact = n < T_LIFT + LIFT_TICKS * 0.15                        # touching the floor until it has risen

    def _advance_photons(self):
        n = self.ticks
        self.photons = [(i + di, k + dk, di, dk) for i, k, di, dk in self.photons if 0 <= i + di < W and 0 <= k + dk < H]
        if T_LAMP <= n < T_FLIGHT:                                           # representative photons: a few a tick, in every direction, none landing
            for _ in range(3):
                ang = math.radians(float(self.rng.uniform(-50, 50)))
                di, dk = 3.0 * math.cos(ang), 3.0 * math.sin(ang)
                end_k = LAMP_K + (CI_LEFT - LAMP_I) * math.tan(ang)
                end_k2 = LAMP_K + (CI_RIGHT - LAMP_I) * math.tan(ang)
                if abs(end_k - CK_FLOAT) > R_BLOCKS + 3 and abs(end_k2 - CK_FLOAT) > R_BLOCKS + 3:
                    self.photons.append((LAMP_I + 3.0, LAMP_K, di, dk))
        elif n >= T_FLIGHT:
            self.photons = []

    # ── drawing ────────────────────────────────────────────────────────────────────────────────────────────────────
    def _put(self, i, k, name, j=JC):
        i, k = int(round(i)), int(round(k))
        if 0 <= i < W and 0 <= k < H:
            self.label[i, j, k] = M[name]

    def _draw(self):
        lab = self.label
        lab[:] = 0
        lab[:, :, 0:2] = M["magnet"]
        lab[:, :, 2:FLOOR_TOP + 1] = M["floor"]
        n = self.ticks
        # the lamp (a lantern in a frame) and the eye
        lab[LAMP_I - 3:LAMP_I + 2, JC - 2:JC, LAMP_K - 2:LAMP_K + 3] = M["frame"]               # the lamp's housing, behind
        lab[LAMP_I - 3:LAMP_I + 2, JC:JC + 3, LAMP_K - 2:LAMP_K + 3] = M["lamp"] if n >= T_LAMP else M["frame"]
        lab[EYE_I - 2:EYE_I + 2, JC - 2:JC + 3, LAMP_K - 2:LAMP_K + 3] = M["eye"]
        if not self.split:
            self._grain(CI_REST, round(self.height))
            if self.contact:                                                    # the contact voxel under the grain: the floor's stress
                lab[CI_REST, JC, FLOOR_TOP] = M["stress"]
        else:
            for ci, weight in ((CI_LEFT, 0.5), (CI_RIGHT, 0.5)):
                self._grain(ci, CK_FLOAT, weight)
                for k in range(FLOOR_TOP + 1, CK_FLOAT - R_BLOCKS, 2):          # the field pressing up on it
                    lab[ci, JC, k] = M["field"]
            self._bridge()
        if n < T_LAMP and self.height > CK_REST + 0.5:                          # the field lifting it
            for k in range(FLOOR_TOP + 1, int(self.height) - R_BLOCKS, 2):
                lab[CI_REST, JC, k] = M["field"]
        self._draw_photons()
        self._draw_meters()
        if n >= T_AFTER:                                                         # you are in one of the two worlds
            ci = CI_LEFT if self.world == "left" else CI_RIGHT
            lab[ci - 1:ci + 2, JC, CK_FLOAT + R_BLOCKS + 2] = M["you"]

    def _grain(self, ci, ck, weight=1.0):
        for i, j, k in _sphere(ci, ck):
            if _density(i, j, k) < weight and 0 <= k < H:
                self.label[i, j, k] = M["osmium"]

    def _bridge(self):
        """The two places are joined while they can interfere: cyan, in principle (the box is sealed); gold, in practice, until the photon records it."""
        for i in range(CI_LEFT + R_BLOCKS + 1, CI_RIGHT - R_BLOCKS):
            self.label[i, JC, CK_FLOAT - 1] = M["bridge"]
            if self.coh_in_practice:
                self.label[i, JC, CK_FLOAT + 1] = M["bridge practice"]

    def _draw_photons(self):
        for i, k, di, dk in self.photons:
            self._put(i, k, "photon", JC - 1)
        n = self.ticks
        if T_FLIGHT <= n < T_AFTER + AFTER_TICKS:
            f = min(1.0, (n - T_FLIGHT) / FLIGHT_TICKS)
            for target in (CI_LEFT, CI_RIGHT):                                   # the photon is in both worlds: toward each place
                hit_i, hit_k = target - R_BLOCKS - 1, CK_FLOAT
                if f < 1.0 or n < T_AFTER:
                    i = LAMP_I + 3 + (hit_i - LAMP_I - 3) * f
                    k = LAMP_K + (hit_k - LAMP_K) * f
                    self._put(i, k, "photon", JC - 1)
                    self._put(i + 1, k, "photon", JC - 1)
                else:                                                            # on to the eye, in each world
                    g = (n - T_AFTER) / AFTER_TICKS
                    i = hit_i + (EYE_I - 2 - hit_i) * g
                    self._put(i, hit_k, "photon", JC - 1)
                    self._put(i + 1, hit_k, "photon", JC - 1)

    def _draw_meters(self):
        for row, (name, value) in ((H - 2, ("bar fill", self.coh_in_principle)), (H - 4, ("bar practice", self.coh_in_practice))):
            for n in range(60):
                self.label[20 + n, 0, row] = M[name] if n < round(60 * value) else M["bar"]

    # ── what is said ───────────────────────────────────────────────────────────────────────────────────────────────
    def captions(self):
        h = self.hertz
        return [
            (LAMP_I, 0, LAMP_K + 4, "COLD LAMP"),
            (EYE_I, 0, LAMP_K + 4, "THE EYE"),
            (CI_REST - 4, 0, 11.5, f"1. REST: weight {h['weight_n']:.1e} N through a patch {2e9 * h['patch_radius_m']:.0f} nm wide, {h['peak_pressure_pa'] / 1e6:.0f} MPa"),
            (CI_REST - 4, 0, 10.5, "   the floor records where it is: one place"),
            (CI_REST + 4, 0, 27.5, f"2. FLOAT: field pressure carries it (limit radius {lg.levitation_limit_radius_m() * 1e6:.0f} um)"),
            (CI_REST + 4, 0, 26.5, "   no contact: two places at once, block density = weight"),
            (CI_REST + 4, 0, 33.0, "cyan: coherence in principle. gold: in practice (cut by the first photon)"),
            (50, 0, 4.0, "MAGNET (osmium is a superconductor below 0.66 K)"),
        ]

    def readout(self):
        ph = self.phase()
        t = self.real_time_s()
        if ph == "rest":
            clock = f"rest: its weight {self.hertz['weight_n']:.1e} N, contact patch {2e9 * self.hertz['patch_radius_m']:.0f} nm wide"
        elif ph == "lift":
            clock = "the field lifts it: no contact, nothing records where it is"
        elif ph == "wait":
            clock = (f"lamp on, real time {t * 1e3:.2f} ms (slowed {self.slow_down():,.0f}x): light everywhere, none on the grain yet; "
                     f"first hit expected at {self.t_hit * 1e3:.2f} ms")
        elif ph == "flight":
            clock = f"the photon that lands: {t * 1e9:.2f} ns of its {lg.flight_time_s(LAMP_DISTANCE_M) * 1e9:.2f} ns flight (slowed {self.slow_down():,.0f}x)"
        else:
            clock = (f"recorded: the photon carries where the grain is. Grain unchanged (kick {self.photon['recoil_m_s']:.0e} m/s, "
                     f"warmed {self.photon['temperature_rise_k']:.0e} K). You are in the {self.world} world")
        return f"Lights up | {clock} | coherence in principle {self.coh_in_principle:.0f}, in practice {self.coh_in_practice:.0f}"

    def gaps(self):
        return lg.GRAIN_GAPS + sc.CAT_GAPS
