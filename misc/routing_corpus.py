"""Routing corpus + validation harness (task 24).

The "questions and answers" that both DRIVE and VERIFY verb→tool matching:
  - positive: every verb's example questions must route to THAT verb,
  - negative: non-physics / metaphor questions must DECLINE (no verb).

Run deterministically (no qwen) so it is fast, free, and reproducible — the
deterministic layer is what we tune to reliability; qwen is the residual.
"""
import sys

sys.path.insert(0, r"D:\Aaron\development\sigma-ground")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sigma_ground.materia.manifest import VERB_MANIFEST
from sigma_ground.materia.translator import translate

# Adversarial negatives — NOT physics sims; the router must DECLINE these
# (a stray "fast"/"hot"/"fall"/"crash" is a fact or metaphor, not a simulation).
NEGATIVES = [
    "how fast is my internet connection",
    "what is the speed of my car on the highway",
    "the stock market crashed today",
    "how hot is it outside right now",
    "i think i might throw up",
    "the economy is in free fall",
    "just drop the mic",
    "what is the capital of France",
    "how do i bake a chocolate cake",
    "interest rates are rising this quarter",
    "my approval rating fell ten points",
    "the waterfall in the valley is beautiful",
    "prices are falling at the grocery store",
    "how fast can you type a sentence",
    "the speed of my download is slow",
    "she has a magnetic personality",
    "that was a heated argument",
    "the quantum leap in our sales figures",
]


# Held-out generalization set — phrasings DIFFERENT from the manifest examples,
# NOT tuned against. Measures whether the triggers generalize or just memorise.
HELDOUT = [
    ("what is the entropy of a black hole", "black_hole"),
    ("Debye screening length in a tokamak plasma", "plasma_physics"),
    ("electric field a centimetre from a point charge", "charged_particle"),
    ("how long until half the radon-222 decays", "nuclear_decay"),
    ("inductance per metre of a twisted pair cable", "transmission_line"),
    ("de Broglie wavelength of a slow neutron", "matter_wave"),
    ("rest energy of a neutron in MeV", "nucleon_masses"),
    ("scanning tunnelling microscope current", "quantum_tunneling"),
    ("London penetration depth of lead", "superconductor"),
    ("Seebeck coefficient of bismuth telluride", "thermoelectric"),
    ("coercivity of a permanent magnet", "magnetic_hysteresis"),
    ("how fast does carbon steel rust", "corrosion_attack"),
    ("pH of a 0.01 molar HCl solution", "chemistry_lab"),
    ("maximum height of a cannonball fired at 60 degrees", "projectile_motion"),
    ("two cars crash head-on, how much energy is lost", "collision_dynamics"),
    ("how much does a clock slow at half the speed of light", "relativistic_motion"),
    ("rms speed of helium atoms at room temperature", "thermal_statistics"),
    ("how fast does sound travel through aluminum", "acoustics"),
    ("is the flow turbulent in this pipe", "fluid_flow"),
    ("wavelength of the hydrogen-alpha spectral line", "hydrogen_spectrum"),
    ("how does the sun produce energy by fusion", "stellar_fusion"),
    ("intrinsic carrier concentration in germanium", "semiconductor_device"),
    ("how does annealing change the grain size of a metal", "metallurgy"),
    ("stiffness of a fibreglass composite", "composite_material"),
    ("Hawking lifetime of a primordial black hole", "black_hole"),
    ("Grover's quantum search", "quantum_algorithms_demo"),
    ("entangle two qubits into a bell state", "quantum_gates"),
    ("energy levels of a quantum dot nanocrystal", "quantum_dot"),
    ("air pressure on the summit of Everest", "atmospheric_profile"),
    ("nitrogen triple bond energy", "molecular_bond"),
]


def route_verbs(q):
    spec = translate(q, use_qwen=False)
    return [s.verb for s in spec.steps]


def main():
    total = hits = 0
    misroutes = []
    for verb, m in VERB_MANIFEST.items():
        for ex in m.get("examples", []):
            total += 1
            routed = route_verbs(ex)
            if verb in routed:
                hits += 1
            else:
                misroutes.append((verb, ex, routed))

    neg_ok = 0
    neg_fail = []
    for q in NEGATIVES:
        routed = route_verbs(q)
        if not routed:
            neg_ok += 1
        else:
            neg_fail.append((q, routed))

    ho_hits = 0
    ho_fail = []
    for q, exp in HELDOUT:
        routed = route_verbs(q)
        if exp in routed:
            ho_hits += 1
        else:
            ho_fail.append((exp, q, routed))

    print(f"POSITIVE routing: {hits}/{total} = {hits/total*100:.1f}%")
    print(f"HELD-OUT (unseen phrasings): {ho_hits}/{len(HELDOUT)} = "
          f"{ho_hits/len(HELDOUT)*100:.1f}%")
    print(f"NEGATIVE decline: {neg_ok}/{len(NEGATIVES)} = "
          f"{neg_ok/len(NEGATIVES)*100:.1f}%")
    if ho_fail:
        print(f"\n--- HELD-OUT MISSES ({len(ho_fail)}) [expected <- question : GOT] ---")
        for v, q, got in ho_fail:
            print(f"  {v:24} <- \"{q[:48]}\"  GOT {got or 'DECLINE'}")
    if misroutes:
        print(f"\n--- MISROUTES ({len(misroutes)})  [expected <- question : GOT] ---")
        for v, ex, got in misroutes:
            print(f"  {v:24} <- \"{ex[:48]}\"  GOT {got or 'DECLINE'}")
    if neg_fail:
        print(f"\n--- NEGATIVE FAILURES ({len(neg_fail)})  [should DECLINE] ---")
        for q, got in neg_fail:
            print(f"  \"{q[:42]}\"  GOT {got}")


if __name__ == "__main__":
    main()
