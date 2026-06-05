"""Routing-reliability guard (task 24).

A representative slice of the question->verb corpus plus adversarial negatives,
asserted on the DETERMINISTIC router (no qwen) — fast, free, reproducible. Keeps
verb->tool matching from silently regressing as verbs/triggers change. The full
corpus + held-out generalization set live in misc/routing_corpus.py.
"""
from sigma_ground.materia.translator import translate

# (expected_verb, question) — one canonical question per domain verb.
POSITIVE = [
    ("black_hole", "Hawking temperature of a 10 solar-mass black hole"),
    ("plasma_physics", "Debye length of a plasma at 10 eV"),
    ("plasma_physics", "gyroradius of a proton in a magnetic field"),
    ("charged_particle", "coulomb force between two charges 1 cm apart"),
    ("transmission_line", "characteristic impedance of a coaxial cable"),
    ("optical_fiber", "numerical aperture of an optical fiber"),
    ("semiconductor_device", "band gap of silicon"),
    ("nuclear_decay", "half-life of uranium-238"),
    ("nucleon_masses", "what is the mass of a proton"),
    ("stellar_fusion", "energy rate of the proton-proton chain"),
    ("hydrogen_spectrum", "Balmer series of hydrogen"),
    ("atomic_coupling", "term symbols of a carbon atom"),
    ("matter_wave", "de Broglie wavelength of a 100 eV electron"),
    ("quantum_dot", "energy levels of a quantum dot"),
    ("quantum_gates", "build a Bell state with a CNOT gate"),
    ("quantum_algorithms_demo", "run Grover's search algorithm"),
    ("quantum_tunneling", "tunneling probability through a barrier"),
    ("condensed_matter", "Fermi energy of a metal"),
    ("superconductor", "BCS gap of niobium"),
    ("thermoelectric", "thermoelectric figure of merit"),
    ("magnetic_hysteresis", "hysteresis loop of iron"),
    ("magnetic_material", "is iron ferromagnetic"),
    ("chemistry_lab", "pH of 0.1 M acetic acid"),
    ("solution_chemistry", "boiling point elevation of salt water"),
    ("molecular_bond", "C-O bond energy"),
    ("corrosion_attack", "corrosion rate of iron"),
    ("combustion_flow", "adiabatic flame temperature of burning carbon"),
    ("projectile_motion", "range of a projectile launched at 45 degrees"),
    ("collision_dynamics", "elastic collision of two balls"),
    ("rolling_object", "how fast does a sphere roll down a ramp"),
    ("relativistic_motion", "time dilation at 0.9c"),
    ("thermal_statistics", "average speed of nitrogen molecules at 300 K"),
    ("acoustics", "speed of sound in steel"),
    ("elastic_solid", "Lamé parameters of aluminum"),
    ("metallurgy", "Hall-Petch yield strength vs grain size"),
    ("tribology", "wear rate of steel sliding on iron"),
    ("fluid_flow", "reynolds number of water flowing in a pipe"),
    ("diffusion_transport", "diffusion coefficient of a particle in water"),
    ("water_state", "density of water at 20 C"),
    ("atmospheric_profile", "air density at 5000 m altitude"),
    # a faller going supersonic is a descent, not a fired projectile
    ("high_altitude_descent", "a skydiver free-falls from 30 km, do they go "
                              "supersonic"),
]

# Non-physics / metaphor — the router MUST decline (stray fast/hot/fall/crash).
NEGATIVE = [
    "how fast is my internet connection",
    "the stock market crashed today",
    "the economy is in free fall",
    "i think i might throw up",
    "what is the capital of France",
    "interest rates are rising this quarter",
    "she has a magnetic personality",
    "the quantum leap in our sales figures",
]


def _route(q):
    return [s.verb for s in translate(q, use_qwen=False).steps]


def test_domain_verbs_route_correctly():
    misses = [(v, q, _route(q)) for v, q in POSITIVE if v not in _route(q)]
    assert not misses, f"routing misses: {misses}"


def test_non_physics_questions_decline():
    leaks = [(q, _route(q)) for q in NEGATIVE if _route(q)]
    assert not leaks, f"non-physics leaked to a verb: {leaks}"
