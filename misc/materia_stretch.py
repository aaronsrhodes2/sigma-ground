"""Materia stretch test — can we construct ANY physics scenario?

Throw scenarios from a dozen physics families at the translator and see what it
can actually build vs. what it honestly declines. This maps coverage, not
robustness — the question is breadth, not phrasing.
"""
import sys
sys.path.insert(0, r"D:\Aaron\development\sigma-ground")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sigma_ground.materia import translate

FAMILIES = {
    "Ballistic / drag (BUILT)": [
        "how fast does a 5 cm steel ball hit the ground from 10 km?",
        "does an iron cannonball heat up falling from the stratosphere?",
        "throw a copper ball up at 400 m/s — how fast and how hot when it lands?",
        "a bullet at Mach 3 — how does its drag change past the sound barrier?",
        "does a skydiver from 35 km go supersonic then slow down?",
    ],
    "Statics / structures": [
        "how much does a 10 ft steel cantilever beam sag under a 500 kg tip load?",
        "how far does a copper cable sag strung between two towers 100 m apart?",
        "what load buckles a 3 m vertical steel column?",
        "which leg of an asymmetric tripod holding a 2 ton safe is overloaded?",
    ],
    "Rigid-body / rotation": [
        "a double pendulum released from horizontal — how fast are both arms spinning at the bottom?",
        "hit a pool ball above center — how far does it slide before it rolls?",
        "top speed of a rocket sled at burnout with quadratic drag?",
        "how long does a spinning gyroscope precess before toppling?",
    ],
    "Orbital / celestial": [
        "what's the orbital velocity of a satellite at 400 km altitude?",
        "if you cut the tether between two co-orbiting satellites, what orbits result?",
        "could the Moon have its own moon, and would it be stable?",
    ],
    "Fluids": [
        "peak pressure spike when a valve slams shut on a 3 m/s water pipe?",
        "how long does a cylindrical tank take to drain through a hole in the bottom?",
        "viscous torque of shear-thinning fluid in a Couette viscometer?",
    ],
    "Thermal (transient/PDE)": [
        "how many seconds until the outside of a hot iron pipe reaches 150 C?",
        "how long for an 800 K copper sphere to radiatively cool to 400 K in vacuum?",
    ],
    "Collisions / momentum": [
        "two equal steel balls collide head-on elastically — what are their final speeds?",
        "drop a steel ball and a lead ball — which one hits harder?",
    ],
    "EM / optics / quantum": [
        "what's the capacitance of two 10 cm plates 1 mm apart?",
        "how far does a 5 mW laser spread over 1 km?",
        "interference pattern from a double slit with 500 nm light?",
    ],
    "Extreme / exotic": [
        "kinetic energy of a 1 g projectile at 0.9c hitting a wall?",
        "surface gravity of a bullet made of neutron-star material?",
        "Hawking temperature of a 1 solar-mass black hole?",
    ],
}


def main():
    covered = gaps = 0
    for fam, qs in FAMILIES.items():
        n_route = 0
        print(f"\n■ {fam}")
        for q in qs:
            steps = [s.verb for s in translate(q, use_qwen=False).steps]
            if steps:
                n_route += 1
                print(f"    ✓ {','.join(steps):42s} {q[:48]}")
            else:
                print(f"    · [decline]{'':33s} {q[:48]}")
        if n_route:
            covered += 1
        else:
            gaps += 1
    print("\n" + "=" * 72)
    print(f"FAMILIES with at least one buildable scenario: {covered}")
    print(f"FAMILIES entirely declined (coverage gaps):    {gaps}")


if __name__ == "__main__":
    main()
