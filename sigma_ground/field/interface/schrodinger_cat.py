"""
Schrödinger's cat, computed: a sealed box, a decaying sample, a counter, a vial, a cat, an observer in the box, and the air.

What the physics says (and what this module computes, so that a renderer only draws it):

  Sample      N atoms of a measured isotope with decay constant λ (``radioactive_decay``), N chosen so that over the window
              (Schrödinger's hour) at least one atom decays with probability 1/2: P(t) = 1 − exp(−N λ t).
  Chain       a unitary von Neumann chain: atom → counter → vial → cat → observer (controlled gates on two-level systems).
              Nothing in it collapses anything: the box holds  α|U,idle,intact,alive,saw-alive⟩ + β|D,click,broken,dead,saw-dead⟩.
  Environment the cat is not alone. Each air molecule that hits it, each thermal photon it emits, carries away which state
              the cat is in. The environment states for "alive" and "dead" are orthogonal after one such event, so the overlap
              ⟨e_alive|e_dead⟩ = Γ = Π cos θ_k → exp(−t/τ). That is decoherence (Zurek 1981, Joos & Zeh 1985): the cat's own
              coherence between alive and dead is gone in τ, with or without anyone looking, and a box that shuts out the
              OUTSIDE does not shut out the AIR INSIDE it.
  Sealed      with nothing leaving the box, the whole content (cat, air, observer, all of it) stays one pure state: in principle
              an outside experimenter who could reverse every molecule would still find the superposition (Wigner's friend).
              In practice nobody can: the coherence of the macroscopic variables is gone.
  Leaky       one photon through a pinhole that scattered off the cat tells the outside; then even in principle the outside
              sees a mixture.
  Looking     the observer inside is part of the chain: she sees one definite outcome, drawn with the Born rule's probability;
              the other branch contains another her.

Everything is an exact numerical calculation on a small register (7 two-level systems = 128 amplitudes): a state vector for the
chain, the environment as two more qubits whose "alive" and "dead" states overlap by Γ, partial traces for what each party can
measure. Closed forms (the chain's branch coherence Γ_in Γ_out, the outside test P(+) = ½ + √(p(1−p)) Γ_out) are the tests' cross-check.

Origin tags:
  - Decay constant: MEASURED half-life of the isotope (radioactive_decay.ISOTOPES; beta decays use it directly)
  - Born rule, unitary chain, partial trace, thermal de Broglie wavelength, kinetic-theory flux, photon number density
    n = 2 ζ(3)/π² (kT/ħc)³: FIRST_PRINCIPLES
  - Air at the box's pressure and temperature: the ideal gas, the library's mean molecular mass (atmosphere)
  - The cat's surface area, skin temperature, and the separation of "alive" and "dead": EMPIRICAL-INPUT (typical house cat)
  - The pinhole's area: EXHIBIT choice
  - Each collision or emitted photon fully records the cat's state: valid when the probe's de Broglie wavelength is far below the
    separation (checked: ``regime``); the saturated regime of scattering decoherence (Schlosshauer 2007, section 3.3)

Not in it (CAT_GAPS): the real cat is not a two-level system and "alive" is not one state; the box is not a closed quantum
system to a gravitational field; internal air is taken at rest and uniform; photons absorbed by the cat are ignored (only
emitted ones); the sample's decay is the exponential law (the Weisskopf-Wigner amplitude), not a derived nuclear process.
"""

import math

import numpy as np

from ..constants import AMU_KG, C, HBAR, H_PLANCK, K_B, N_AVOGADRO
from . import atmosphere as _atm
from . import radioactive_decay as _decay

# ── Inputs (each tagged) ─────────────────────────────────────────────────
ISOTOPE = "Co60"            # EXHIBIT choice: a gamma emitter a Geiger counter really counts; its half-life is measured
WINDOW_S = 3600.0           # Schrödinger (1935): "in the course of an hour"
P_AT_WINDOW = 0.5           # Schrödinger: "perhaps one of the atoms decays": the sample is sized for even odds
CAT_AREA_M2 = 0.2           # EMPIRICAL-INPUT: a house cat's body surface (Meeh's formula, ~4 kg)
CAT_SKIN_K = 305.0          # EMPIRICAL-INPUT: a cat's skin temperature
CAT_SEPARATION_M = 0.30     # EMPIRICAL-INPUT: how far "dead" (lying) differs from "alive" (sitting), about a body length
BOX_T_K = 293.15            # EXHIBIT choice: room temperature
BOX_P_PA = 101325.0         # EXHIBIT choice: one atmosphere inside the box
PINHOLE_M2 = 1.0e-6         # EXHIBIT choice: a one square millimetre leak
BOX_RADIUS_M = 1.5          # EXHIBIT choice: the box's inner size, for the share of records that leave through a hole
ZETA_3 = 1.2020569031595942  # MEASURED (mathematical constant)

CAT_GAPS = (
    "The cat is two states (alive, dead) and the environment two orthogonal-in-the-limit states; a real cat has no such clean pair.",
    "Only the cat's own emitted photons and the air it meets are counted as the environment that records its state; absorbed photons, sound, smell and gravity add more, so the real decoherence is faster still.",
    "The sample's decay is the exponential law (a Weisskopf-Wigner amplitude), not a derived nuclear process; its half-life is the measured one.",
    "A sealed box is perfectly isolated by definition here: no real wall stops gravity or heat.",
    "The observer is a two-level pointer that records the cat's state; no brain, no consciousness, and none is needed.",
    "The lump's decoherence by thermal photons uses the scaling (separation/wavelength)^2 once the wavelength exceeds the separation, an order-of-magnitude estimate, not a derived coefficient; its emissivity is a typical polished metal's, not osmium's.",
    "No gravitational or collapse-model decoherence (Diosi-Penrose and others are hypotheses, and experiments so far bound them far below a lump this size); standard quantum mechanics has gravity decohere nothing here.",
)

QUBITS = ("atom", "counter", "vial", "cat", "observer", "air_in", "leak")      # indices 0..6
_N = len(QUBITS)
_ATOM, _COUNTER, _VIAL, _CAT, _OBS, _ENV_IN, _ENV_OUT = range(_N)


# ── The sample ───────────────────────────────────────────────────────────

def sample(isotope: str = ISOTOPE, window_s: float = WINDOW_S, p_window: float = P_AT_WINDOW) -> dict:
    """The sample Schrödinger imagined: how many atoms of ``isotope`` give ``p_window`` odds of a decay in ``window_s``."""
    lam = _decay.decay_constant(isotope)
    n = -math.log(1.0 - p_window) / (lam * window_s)
    iso = _decay.ISOTOPES[isotope]
    return {"isotope": isotope, "name": iso["name"], "half_life_s": iso["half_life_s"], "decay_constant_per_s": lam,
            "atoms": n, "mass_kg": n * iso["A"] * AMU_KG, "activity_Bq": n * lam,
            "window_s": window_s, "p_window": p_window}


def decay_probability(t_s: float, s: dict = None) -> float:
    """The odds that at least one atom has decayed by ``t_s``."""
    s = s or sample()
    return 1.0 - math.exp(-s["atoms"] * s["decay_constant_per_s"] * max(0.0, t_s))


# ── The environment that records the cat ─────────────────────────────────

def thermal_de_broglie_m(mass_kg: float, temperature_k: float) -> float:
    """The thermal de Broglie wavelength h / sqrt(2 pi m k T)."""
    return H_PLANCK / math.sqrt(2.0 * math.pi * mass_kg * K_B * temperature_k)


def air_collision_rate(area_m2: float = CAT_AREA_M2, temperature_k: float = BOX_T_K, pressure_pa: float = BOX_P_PA) -> float:
    """Air molecules striking the cat per second: n v̄ A / 4 (kinetic theory), n = P / kT, v̄ = sqrt(8 k T / pi m)."""
    m = _atm.mean_molecular_mass_kg()
    n = pressure_pa / (K_B * temperature_k)
    v = math.sqrt(8.0 * K_B * temperature_k / (math.pi * m))
    return n * v * area_m2 / 4.0


def photon_emission_rate(area_m2: float = CAT_AREA_M2, temperature_k: float = CAT_SKIN_K) -> float:
    """Thermal photons the cat's skin radiates per second: (c/4) n_gamma A, n_gamma = (2 zeta(3) / pi^2) (kT / hbar c)^3 (a black body)."""
    n_gamma = 2.0 * ZETA_3 / math.pi ** 2 * (K_B * temperature_k / (HBAR * C)) ** 3
    return C / 4.0 * n_gamma * area_m2


def leak_rate(hole_m2: float = PINHOLE_M2, temperature_k: float = BOX_T_K) -> float:
    """Thermal photons per second that leave the box through a hole, each having scattered off the cat on its way: (c/4) n_gamma A_hole."""
    n_gamma = 2.0 * ZETA_3 / math.pi ** 2 * (K_B * temperature_k / (HBAR * C)) ** 3
    return C / 4.0 * n_gamma * hole_m2


def regime(separation_m: float = CAT_SEPARATION_M) -> dict:
    """Is each probe's wavelength far below the separation, so that one event fully records the state (the saturated regime)?"""
    air_nm = thermal_de_broglie_m(_atm.mean_molecular_mass_kg(), BOX_T_K) * 1e9
    ir_um = 2.0 * math.pi * HBAR * C / (2.82 * K_B * CAT_SKIN_K) * 1e6       # Wien's displacement in photon-energy form
    return {"air_de_broglie_nm": air_nm, "photon_peak_um": ir_um, "separation_m": separation_m,
            "saturated": air_nm * 1e-9 < separation_m / 100.0 and ir_um * 1e-6 < separation_m / 100.0}


def decoherence_times(vacuum: bool = False) -> dict:
    """Seconds for the cat's superposition to be recorded by its own surroundings inside the box: by the air (absent in a vacuum), by its
    own thermal photons, and all together (the rates add)."""
    r_air = 0.0 if vacuum else air_collision_rate()
    r_ph = photon_emission_rate()
    return {"air_s": math.inf if r_air == 0 else 1.0 / r_air, "photons_s": 1.0 / r_ph, "total_s": 1.0 / (r_air + r_ph),
            "rate_air_per_s": r_air, "rate_photons_per_s": r_ph}


# ── A lump of inert matter instead of a cat ──────────────────────────────
OSMIUM_DENSITY_KG_M3 = 22590.0   # MEASURED: CRC Handbook (osmium, the densest natural element)
LUMP_RADIUS_M = 0.01             # EXHIBIT choice: a 1 cm sphere (95 g of osmium)
LUMP_SEPARATION_M = 0.01         # EXHIBIT choice: the trigger kicks it 1 cm sideways: "here" and "there" differ by that
LUMP_EMISSIVITY = 0.05           # EMPIRICAL-INPUT: a polished metal in the far infrared (the library holds no osmium optics)
OBSERVER_AREA_M2 = 1.8           # EMPIRICAL-INPUT: an adult's skin
OBSERVER_SKIN_K = 305.0          # EMPIRICAL-INPUT: skin temperature
OBSERVER_DISTANCE_M = 1.0        # EXHIBIT choice: how far the observer stands from the lump


def lump_mass_kg(radius_m: float = LUMP_RADIUS_M) -> float:
    return OSMIUM_DENSITY_KG_M3 * 4.0 / 3.0 * math.pi * radius_m ** 3


def photon_flux_per_area(temperature_k: float) -> float:
    """Thermal photons per second through a square metre in a radiation field at ``temperature_k``: (c/4) n_gamma, n_gamma = (2 zeta(3)/pi^2)(kT/hbar c)^3."""
    if temperature_k <= 0.0:
        return 0.0
    return C / 4.0 * 2.0 * ZETA_3 / math.pi ** 2 * (K_B * temperature_k / (HBAR * C)) ** 3


def thermal_wavelength_m(temperature_k: float) -> float:
    """The wavelength at the peak of the photon spectrum, 2 pi hbar c / (2.82 k T) (infinite at 0 K)."""
    return math.inf if temperature_k <= 0.0 else 2.0 * math.pi * HBAR * C / (2.82 * K_B * temperature_k)


def resolving_fraction(separation_m: float, temperature_k: float) -> float:
    """How much of "which side?" one thermal photon carries: 1 when its wavelength is below the separation, (separation/wavelength)^2 above
    (the long-wavelength scaling of Joos & Zeh 1985: an order-of-magnitude estimate, not a coefficient)."""
    lam = thermal_wavelength_m(temperature_k)
    return 0.0 if math.isinf(lam) else min(1.0, (separation_m / lam) ** 2)


def lump_decoherence_times(temperature_k: float = 0.0, box_k: float = None, vacuum: bool = True, observer_inside: bool = False,
                           radius_m: float = LUMP_RADIUS_M, separation_m: float = LUMP_SEPARATION_M) -> dict:
    """Seconds for the surroundings to record which side a lump of inert matter (osmium) is on, by channel and all together.

    own        its thermal photons: emissivity x area x photon flux x the share of "which side" each carries
    box        the box's thermal radiation scattering off it (the box at ``box_k``, by default the lump's own temperature)
    observer   an observer at the skin temperature standing in the box: her photons scatter off the lump
    gas        residual gas (none when ``vacuum``); at 0 K the walls would freeze it out anyway
    """
    box_k = temperature_k if box_k is None else box_k
    area = 4.0 * math.pi * radius_m ** 2
    own = LUMP_EMISSIVITY * area * photon_flux_per_area(temperature_k) * resolving_fraction(separation_m, temperature_k)
    box = (1.0 - LUMP_EMISSIVITY) * area * photon_flux_per_area(box_k) * resolving_fraction(separation_m, box_k)
    obs = 0.0
    if observer_inside:
        emitted = OBSERVER_AREA_M2 * photon_flux_per_area(OBSERVER_SKIN_K)
        obs = (1.0 - LUMP_EMISSIVITY) * emitted * radius_m ** 2 / (4.0 * OBSERVER_DISTANCE_M ** 2) * resolving_fraction(separation_m, OBSERVER_SKIN_K)
    gas = 0.0 if vacuum else air_collision_rate(area, BOX_T_K, BOX_P_PA)
    total = own + box + obs + gas
    inv = lambda r: math.inf if r <= 0.0 else 1.0 / r
    return {"own_s": inv(own), "box_s": inv(box), "observer_s": inv(obs), "gas_s": inv(gas), "total_s": inv(total),
            "rate_own_per_s": own, "rate_box_per_s": box, "rate_observer_per_s": obs, "rate_gas_per_s": gas,
            "mass_kg": lump_mass_kg(radius_m), "temperature_k": temperature_k}


# ── The quantum state ────────────────────────────────────────────────────

def _apply1(psi, gate, q):
    psi = np.moveaxis(psi.reshape((2,) * _N), q, 0)
    psi = np.tensordot(gate, psi, axes=([1], [0]))
    return np.moveaxis(psi, 0, q).reshape(-1)


def _controlled(psi, gate, control, target):
    """Apply ``gate`` to ``target`` on the part of the state where ``control`` is 1."""
    t = psi.reshape((2,) * _N).copy()
    idx = [slice(None)] * _N
    idx[control] = 1
    sub = np.moveaxis(t[tuple(idx)], target if target < control else target - 1, 0)
    sub = np.tensordot(gate, sub, axes=([1], [0]))
    t[tuple(idx)] = np.moveaxis(sub, 0, target if target < control else target - 1)
    return t.reshape(-1)


_X = np.array([[0.0, 1.0], [1.0, 0.0]])


def _ry(theta):
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]])


def box_state(p: float, gamma_in: float = 1.0, gamma_out: float = 1.0, observer_looked: bool = True, atom: float = None):
    """The state vector of the whole thing, 7 qubits, for decay odds ``p``.

    The atom is √(1−p)|undecayed⟩ + √p|decayed⟩; the counter, the vial and the cat each follow the one before (a controlled flip:
    nothing collapses); the observer looks at the cat if ``observer_looked``. The two environments (the air and photons inside, and what
    leaks out) each end up in a state that depends on the cat, the "alive" and "dead" states overlapping by ``gamma_in`` and
    ``gamma_out``. ``atom`` forces the atom to 0 or 1 (a branch).
    """
    psi = np.zeros(2 ** _N)
    psi[0] = 1.0
    if atom is None:
        psi = _apply1(psi, _ry(2.0 * math.asin(math.sqrt(max(0.0, min(1.0, p))))), _ATOM)
    elif atom:
        psi = _apply1(psi, _X, _ATOM)
    for a, b in ((_ATOM, _COUNTER), (_COUNTER, _VIAL), (_VIAL, _CAT)):
        psi = _controlled(psi, _X, a, b)
    if observer_looked:
        psi = _controlled(psi, _X, _CAT, _OBS)
    # an environment whose state is |0> when the cat is alive and cos(a)|0> + sin(a)|1> when dead: overlap cos(a) = gamma
    for env, g in ((_ENV_IN, gamma_in), (_ENV_OUT, gamma_out)):
        psi = _controlled(psi, _ry(2.0 * math.acos(max(0.0, min(1.0, g)))), _CAT, env)
    return psi


def reduced(psi, keep):
    """The density matrix of the qubits ``keep`` (a tuple of indices, kept in order), the others traced out."""
    t = psi.reshape((2,) * _N)
    keep = list(keep)
    rest = [q for q in range(_N) if q not in keep]
    m = np.transpose(t, keep + rest).reshape(2 ** len(keep), 2 ** len(rest))
    return m @ m.conj().T


def branch_coherence(psi, keep=tuple(range(_ENV_IN))) -> float:
    """How much interference between the two branches ("the atom decayed, the counter clicked, ... the cat is dead" and "nothing
    happened, ... the cat is alive") the qubits ``keep`` still hold: |ρ(alive branch, dead branch)| / sqrt(ρ_aa ρ_dd), 1 for a pure
    superposition, 0 for a mixture. The default is the macroscopic chain (atom, counter, vial, cat, observer), the air and the leak
    traced out: the cat itself is entangled with the counter and the rest, so its own state is a mixture from the first moment, and
    what the environment can destroy is the coherence of the chain as a whole."""
    keep = tuple(keep)
    rho = reduced(psi, keep)
    n = len(keep)
    d = rho[0, 0].real * rho[2 ** n - 1, 2 ** n - 1].real if n else 0.0
    return 0.0 if d <= 0.0 else float(abs(rho[0, 2 ** n - 1]) / math.sqrt(d))


def outside_test(p: float, gamma_in: float, gamma_out: float, observer_looked: bool = True, practical: bool = False) -> float:
    """The best an experimenter OUTSIDE the box can do, asking: "is the box's content the superposition (|alive branch⟩+|dead branch⟩)/√2
    rather than a mixture?" The odds of the answer yes. A pure even superposition gives 1; an even mixture, 1/2.

    ``practical`` False: she can reverse everything inside, the inside air and photons too (in principle). True: she can only measure the
    macroscopic variables (atom, counter, vial, cat, observer): the air and photons inside are traced out, as they are in any real test.
    """
    psi = box_state(p, gamma_in, gamma_out, observer_looked)
    keep = tuple(range(_ENV_IN)) if practical else tuple(range(_ENV_OUT))
    rho = reduced(psi, keep)
    plus = (_branch_vec(0, gamma_in, observer_looked, practical) + _branch_vec(1, gamma_in, observer_looked, practical)) / math.sqrt(2.0)
    return float((plus @ rho @ plus).real)


def _branch_vec(atom: int, gamma_in: float, observer_looked: bool, practical: bool):
    """The pure state of the interior if the atom is definitely ``atom``: the macroscopic variables all agree, and (unless ``practical``)
    the air's record is |0> for "nothing happened" and has overlap ``gamma_in`` with it for the other branch."""
    obs = 1 if observer_looked else 0
    base = 0
    for b in [atom] * 4 + [atom * obs]:
        base = base * 2 + b
    if practical:
        v = np.zeros(2 ** _ENV_IN)
        v[base] = 1.0
        return v
    v = np.zeros(2 ** _ENV_OUT)
    g = gamma_in if atom else 1.0
    v[base * 2] = g
    v[base * 2 + 1] = math.sqrt(max(0.0, 1.0 - g * g))
    return v


def chain_coherence(p: float, gamma_in: float, gamma_out: float, observer_looked: bool = True, in_principle: bool = False) -> float:
    """The interference between the two branches that the box's contents still hold: |<B0|rho|B1>| / sqrt(<B0|rho|B0><B1|rho|B1>), where
    B0 and B1 are "nothing happened" and "the atom decayed, the counter clicked, ..., the cat is dead". 1 is a pure superposition, 0 a mixture.
    In practice (default) only the macroscopic variables are kept: the air and light inside, and what leaks out, are traced away. In principle
    the air inside is kept too (an experimenter who can reverse every molecule); only what leaks out of the box is lost."""
    psi = box_state(p, gamma_in, gamma_out, observer_looked)
    practical = not in_principle
    rho = reduced(psi, tuple(range(_ENV_IN)) if practical else tuple(range(_ENV_OUT)))
    b0, b1 = _branch_vec(0, gamma_in, observer_looked, practical), _branch_vec(1, gamma_in, observer_looked, practical)
    d = float((b0 @ rho @ b0).real * (b1 @ rho @ b1).real)
    return 0.0 if d <= 1e-30 else float(abs(b0 @ rho @ b1) / math.sqrt(d))


def observer_sees(p: float, rng) -> str:
    """What the observer inside experiences: one outcome, drawn with the Born rule's probability (the other branch holds another observer)."""
    return "dead" if rng.random() < p else "alive"


def gamma_at(t_s: float, tau_s: float) -> float:
    """The overlap of the environment's two records after ``t_s`` of recording at one event every ``tau_s`` seconds: exp(−t/τ)."""
    return 0.0 if tau_s <= 0 else math.exp(-max(0.0, t_s) / tau_s)


def timeline(vacuum: bool = False, hole: bool = False) -> dict:
    """The numbers a renderer draws: the sample, the decoherence times, the leak, the regime, and the gaps."""
    s = sample()
    d = decoherence_times(vacuum)
    return {"sample": s, "decoherence": d, "regime": regime(), "leak_tau_s": 1.0 / leak_rate() if hole else math.inf,
            "leak_rate_per_s": leak_rate(), "gaps": list(CAT_GAPS)}
