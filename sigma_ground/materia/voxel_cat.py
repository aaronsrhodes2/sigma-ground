"""
CatBox: Schrödinger's cat on a stage of voxels. Every number it draws comes from ``field.interface.schrodinger_cat``.

Left, the BOX: opaque and sealed (its walls are the claim that nothing leaves it). The cat, the sample, the counter and the vial are
inside, drawn as the branch that the observer inside the box experiences (one outcome, drawn with the Born rule's probability);
from outside you see walls. Step inside to be that observer. Open the door and the outside learns the same outcome.

Right, the LEDGER: the quantum state, which no one inside the box can see and no one outside can see either, drawn as bars:
  - the hour: the sample decays with the isotope's measured half-life; the odds that the atom has decayed by now;
  - five pairs of bars (atom, counter, vial, cat, observer): the weight of "alive-branch" (green) and "dead-branch" (red). They are
    the same five bars: everything in the box is correlated, which is the whole puzzle;
  - two meters, the coherence of the chain between its two branches IN PRACTICE (the air and light inside the box traced out)
    and IN PRINCIPLE (an outside experimenter who could reverse every molecule inside);
  - the zoom: the same coherence against log10 of the seconds since the branching, 1e-33 s to 10 s. The air inside the box records the
    cat in ~1e-27 s; a person notices in ~0.1 s;
  - the outside test: many runs of "is the box a superposition or a mixture?", asked in principle and in practice.

The run has three phases: THE HOUR (the weights grow as the odds of a decay do), THE ZOOM (the logarithmic clock sweeps the first
instants and the coherence goes), THE TEST (the outside test's tallies build up).

Switches (each starts a fresh run, a new draw of the Born rule): ``leak`` (a one square millimetre pinhole: the outside is told),
``air`` (pump the air out: only the cat's own light records it). ``open`` opens the door now: the outside is entangled, and sees the
same outcome as the observer inside.

Exhibit choices (stated, not physics): a block is 25 cm; the phase lengths; the five bar pairs; the scene's colours are plain blocks.
"""
from __future__ import annotations

import math

import numpy as np

from ..field.interface import schrodinger_cat as sc

PITCH = 0.25
W, D, H = 86, 14, 40
JF = D - 1                                   # the plane of the front: the board and the box's door wall
HOUR_TICKS, ZOOM_TICKS, TEST_TICKS = 110, 120, 70
TICKS = HOUR_TICKS + ZOOM_TICKS + TEST_TICKS
LOG_FROM, LOG_TO = -33, 1                    # decades of seconds in the zoom: 1e-33 s .. 10 s
REACTION_S = 0.1                             # a person's reaction time (EMPIRICAL-INPUT, the mark on the chart)
TRIALS_PER_TICK = 400

BOX = (1, 13)                                # outer i of the box
BOX_J = (2, JF)                              # outer j: back wall .. door wall
BOX_H = 9                                    # floor k=0, roof k=8
BOARD_I0 = 17
OBSERVER_CELL = (10, 10, 1)                  # where the player stands inside the box

MATERIALS = ["air", "wall", "floor", "lamp", "atom", "atom decayed", "counter", "counter clicked", "vial", "vial broken",
             "cat alive", "cat dead", "panel", "green", "red", "white", "gold", "cyan", "violet", "blue", "glow", "grey"]
LOOKS = {"wall": "polished_deepslate", "floor": "smooth_stone", "lamp": "sea_lantern", "atom": "quartz_block",
         "atom decayed": "glowstone", "counter": "iron_block", "counter clicked": "copper_block", "vial": "slime_block",
         "vial broken": "mud", "cat alive": "honeycomb_block", "cat dead": "stone", "panel": "polished_blackstone",
         "green": "emerald_block", "red": "redstone_block", "white": "quartz_block", "gold": "gold_block", "cyan": "diamond_block",
         "violet": "amethyst_block", "blue": "lapis_block", "glow": "glowstone", "grey": "iron_block"}
M = {name: i for i, name in enumerate(MATERIALS)}


class CatBox:
    OBSERVER_CELL = OBSERVER_CELL

    def __init__(self, seed=None):
        self.pitch = PITCH
        self.materials = list(MATERIALS)
        self.sealed_leak = False                # a pinhole in the box
        self.vacuum = False                     # the air pumped out
        self.door_open = False
        self.t = 0.0
        self._seed = seed
        self.reset()

    # ── the run ────────────────────────────────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None):
        """A fresh run: a new draw of the Born rule (the observer's outcome and when the atom decayed)."""
        self.rng = np.random.default_rng(self._seed if seed is None and self._seed is not None else seed)
        self.sample = sc.sample()
        self.times = sc.decoherence_times(self.vacuum)
        self.tau_in = self.times["total_s"]
        self.tau_leak = 1.0 / sc.leak_rate() if self.sealed_leak else math.inf
        p_end = sc.decay_probability(sc.WINDOW_S, self.sample)
        self.outcome = sc.observer_sees(p_end, self.rng)          # what the observer inside experiences
        lam_n = self.sample["atoms"] * self.sample["decay_constant_per_s"]
        u = float(self.rng.random())
        self.t_decay = math.inf if self.outcome == "alive" else -math.log(1.0 - u * (1.0 - math.exp(-lam_n * sc.WINDOW_S))) / lam_n
        self.ticks = 0
        self.door_open = False
        self.opened_at_tick = None
        self.yes_in_principle = self.yes_in_practice = self.trials = 0
        self.label = np.zeros((W, D, H), np.int32)
        self._state()
        self._draw()

    def press(self, what):
        """``leak`` | ``air`` | ``open``: the first two change the setup and start again; ``open`` opens the door now."""
        if what == "leak":
            self.sealed_leak = not self.sealed_leak
            msg = ("A pinhole, 1 mm2: a photon that bounced off the cat can leave, and the outside is told." if self.sealed_leak
                   else "Sealed again: nothing leaves the box.")
        elif what == "air":
            self.vacuum = not self.vacuum
            msg = "The air pumped out: only the cat's own light records it." if self.vacuum else "Air back in the box."
        elif what == "open":
            if not self.door_open:
                self.door_open = True
                self.opened_at_tick = self.ticks
                self._state()
                self._draw()
            return "The door is open: the outside looked, and sees what the observer inside saw: the cat is " + self.outcome.upper() + "."
        else:
            return ""
        self.reset()
        return msg

    def step(self, dt):
        self.ticks = min(self.ticks + 1, TICKS)
        self.t += dt
        if self.phase() == "test":
            n = TRIALS_PER_TICK
            self.yes_in_principle += int(self.rng.binomial(n, self.p_test_principle))
            self.yes_in_practice += int(self.rng.binomial(n, self.p_test_practice))
            self.trials += n
        self._state()
        self._draw()

    # ── the state at this tick ─────────────────────────────────────────────────────────────────────────────────────
    def phase(self):
        return "hour" if self.ticks < HOUR_TICKS else ("zoom" if self.ticks < HOUR_TICKS + ZOOM_TICKS else "test")

    def hour_s(self):
        return sc.WINDOW_S * min(1.0, self.ticks / HOUR_TICKS)

    def log_t(self):
        """log10 of the seconds since the branching, as the zoom has swept it."""
        if self.phase() == "hour":
            return float(LOG_FROM)
        z = min(1.0, (self.ticks - HOUR_TICKS) / ZOOM_TICKS)
        return LOG_FROM + z * (LOG_TO - LOG_FROM)

    def _since_branching(self):
        """Seconds the environment has had to record the branching: the whole hour's worth once the zoom is done."""
        if self.phase() == "hour":
            return self.hour_s()
        return 10.0 ** self.log_t() if self.phase() == "zoom" else 10.0 ** LOG_TO

    def _gammas(self, t):
        g_in = sc.gamma_at(t, self.tau_in)
        g_out = 0.0 if self.door_open else sc.gamma_at(t, self.tau_leak)
        return g_in, g_out

    def _state(self):
        self.p = sc.decay_probability(self.hour_s(), self.sample)
        # in the hour the environment has had all of it to record the branching (any superposition is long recorded)
        t = max(self.hour_s(), 1.0) if self.phase() == "hour" else self._since_branching()
        self.g_in, self.g_out = self._gammas(t)
        psi = sc.box_state(self.p, self.g_in, self.g_out, observer_looked=True)
        self.coh_practice = sc.branch_coherence(psi)
        self.coh_principle = sc.branch_coherence(psi, tuple(range(sc._ENV_OUT)))       # the air kept: everything inside
        p_end = sc.decay_probability(sc.WINDOW_S, self.sample)
        g_in, g_out = self._gammas(10.0 ** LOG_TO)
        self.p_test_principle = sc.outside_test(p_end, g_in, g_out)
        self.p_test_practice = sc.outside_test(p_end, g_in, g_out, practical=True)

    # ── drawing ────────────────────────────────────────────────────────────────────────────────────────────────────
    def _put(self, i, k, name, j=JF):
        if 0 <= i < W and 0 <= k < H:
            self.label[i, j, k] = M[name]

    def _hbar(self, i0, k, length, filled, name, back="grey"):
        for n in range(length):
            self._put(i0 + n, k, name if n < filled else back)

    def _vbar(self, i, k0, height, filled, name, width=1):
        for w in range(width):
            for n in range(height):
                if n < filled:
                    self._put(i + w, k0 + n, name)

    def _draw(self):
        lab = self.label
        lab[:] = 0
        self._draw_box()
        self._draw_board()

    def _draw_box(self):
        lab = self.label
        i0, i1 = BOX
        j0, j1 = BOX_J
        lab[i0:i1 + 1, j0:j1 + 1, 0] = M["floor"]
        lab[i0:i1 + 1, j0:j1 + 1, BOX_H - 1] = M["wall"]
        for k in range(1, BOX_H - 1):
            lab[i0:i1 + 1, j0, k] = M["wall"]                       # back wall
            lab[i0, j0:j1 + 1, k] = M["wall"]                       # left wall
            lab[i1, j0:j1 + 1, k] = M["wall"]                       # right wall
            if not self.door_open:
                lab[i0 + 1:i1, j1, k] = M["wall"]                   # the door wall (the front)
        lab[7, 7, BOX_H - 2] = M["lamp"]                            # a light in the ceiling
        dead = self.outcome == "dead" and self.hour_s() >= self.t_decay
        lab[4, 4, 1] = M["floor"]                                   # the sample on its pedestal
        lab[4, 4, 2] = M["atom decayed"] if dead else M["atom"]
        for ci in (7, 8):                                           # the counter
            lab[ci, 4, 1] = lab[ci, 4, 2] = M["counter clicked"] if dead else M["counter"]
        lab[11, 4, 1] = M["floor"]                                  # the vial on its pedestal
        lab[11, 4, 2] = M["vial broken"] if dead else M["vial"]
        cat = "cat dead" if dead else "cat alive"
        if not dead:                                                # sitting up: body, head, ears, tail
            lab[6:9, 6:10, 1:3] = M[cat]
            lab[7, 10, 3] = M[cat]
            lab[6, 10, 4] = lab[8, 10, 4] = M[cat]
            lab[7, 5, 2:4] = M[cat]
        else:                                                       # on its side: flat, the legs out
            lab[6:9, 6:10, 1] = M[cat]
            lab[7, 10, 1] = M[cat]
            for ii in (5, 9):
                lab[ii, 7:9, 1] = M[cat]
            lab[7, 5, 1] = M[cat]

    def _draw_board(self):
        lab = self.label
        b0 = BOARD_I0
        lab[b0:W, JF, 0:H] = M["panel"]
        # the hour: a strip along the top
        frac = self.hour_s() / sc.WINDOW_S if self.phase() == "hour" else 1.0
        self._hbar(b0 + 2, H - 3, 64, round(64 * frac), "glow", "grey")
        self._hbar(b0 + 2, H - 2, 64, round(64 * frac), "glow", "grey")
        # five pairs: the weight of each branch (they are the same bar, five times)
        top = 10
        for n in range(5):
            x = b0 + 4 + 13 * n
            self._vbar(x, 23, top, round(top * (1.0 - self.p)), "green", 2)
            self._vbar(x + 3, 23, top, round(top * self.p), "red", 2)
            self._hbar(x, 22, 5, 5, "white", "white")
        # two meters: the chain's coherence between its branches
        self._hbar(b0 + 4, 19, 60, round(60 * self.coh_practice), "gold", "grey")
        self._hbar(b0 + 4, 20, 60, round(60 * self.coh_practice), "gold", "grey")
        self._hbar(b0 + 4, 16, 60, round(60 * self.coh_principle), "cyan", "grey")
        self._hbar(b0 + 4, 17, 60, round(60 * self.coh_principle), "cyan", "grey")
        # the zoom chart: coherence (rows 3..14) against log10 t, one column per decade (35)
        cols = LOG_TO - LOG_FROM + 1
        c0 = b0 + 2
        for c in range(cols):
            lg = LOG_FROM + c
            t = 10.0 ** lg
            g_in, g_out = self._gammas(t)
            self._put(c0 + c, 2, "white")
            for k_val, name in ((g_in * g_out, "gold"), (g_out, "cyan")):
                row = 3 + round(11 * k_val)
                self._put(c0 + c, min(row, 14), name)
        # marks: where the air inside records the cat, and a person's reaction
        for lg, name in ((math.log10(self.tau_in), "violet"), (math.log10(REACTION_S), "red")):
            c = round(lg) - LOG_FROM
            if 0 <= c < cols:
                for k in range(3, 15):
                    if lab[c0 + c, JF, k] == M["panel"]:
                        self._put(c0 + c, k, name)
        if math.isfinite(self.tau_leak):
            c = round(math.log10(self.tau_leak)) - LOG_FROM
            for k in range(3, 15):
                if 0 <= c < cols and lab[c0 + c, JF, k] == M["panel"]:
                    self._put(c0 + c, k, "blue")
        if self.phase() != "hour":                                  # the marker: now
            c = min(cols - 1, max(0, round(self.log_t()) - LOG_FROM)) if self.phase() == "zoom" else cols - 1
            for k in range(1, 16):
                self._put(c0 + c, k, "glow")
        # the outside test: yes/no in principle, yes/no in practice
        n = max(1, self.trials)
        bx = b0 + 42
        for idx, (yes, name_yes) in enumerate(((self.yes_in_principle, "cyan"), (self.yes_in_practice, "gold"))):
            x = bx + 12 * idx
            self._vbar(x, 3, 12, round(12 * yes / n) if self.trials else 0, name_yes, 4)
            self._vbar(x + 5, 3, 12, round(12 * (self.trials - yes) / n) if self.trials else 0, "red", 4)

    # ── what is said ───────────────────────────────────────────────────────────────────────────────────────────────
    def captions(self):
        """Text for the stage: ``(i, j, k, text)`` in grid cells (j the front plane)."""
        b0 = BOARD_I0
        return [
            (7, JF, 11, "THE BOX: sealed. Opaque to the outside."),
            (b0 + 34, JF, 37, "THE HOUR (the sample's decay clock)"),
            (b0 + 34, JF, 35, "weights: green = nothing happened, red = it did"),
            (b0 + 34, JF, 22.4, "atom        counter        vial          cat         observer"),
            (b0 + 34, JF, 21, "chain coherence  IN PRACTICE (gold)  /  IN PRINCIPLE (cyan)"),
            (b0 + 19, JF, 1, "log10 seconds since the branching: -33 ... +1"),
            (b0 + 19, JF, 15.6, "violet: the air records the cat. red: you notice (0.1 s)."),
            (b0 + 54, JF, 15.6, "OUTSIDE TEST: yes (cyan/gold) | no (red)"),
            (b0 + 48, JF, 1, "in principle        in practice"),
        ]

    def readout(self):
        ph = {"hour": f"the hour {self.hour_s() / 60:.0f} min: P(decay) {self.p:.2f}",
              "zoom": f"zoom: {10.0 ** self.log_t():.0e} s after the branching",
              "test": f"outside test {self.trials:,} runs: yes {self.yes_in_principle / max(1, self.trials):.0%} in principle, "
                      f"{self.yes_in_practice / max(1, self.trials):.0%} in practice"}[self.phase()]
        box = ("open" if self.door_open else "sealed") + (", pinhole" if self.sealed_leak else "") + (", vacuum" if self.vacuum else ", air")
        seen = self.outcome.upper() if (self.outcome == "alive" or self.hour_s() >= self.t_decay) else "alive so far"
        return (f"Schrodinger's cat | {ph} | chain coherence in practice {self.coh_practice:.2f}, in principle {self.coh_principle:.2f} | "
                f"box {box} | observer inside sees: {seen}")

    def gaps(self):
        return sc.CAT_GAPS
