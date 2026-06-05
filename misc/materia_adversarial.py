"""Materia adversarial battery — try to BREAK the translator with out-of-test
questions: slang, non-physics trigger-word bait, malformed input, edge values,
and chains. The dangerous failure is a MISROUTE (a non-physics question with a
stray 'speed'/'hot' getting a confident falling-object answer).

expected: 'route'   → should route to a verb (any non-decline)
          'decline' → should decline (clarify), NOT fake an answer
          'chain'   → should be a multi-step plan
"""
import sys
import traceback

sys.path.insert(0, r"D:\Aaron\development\sigma-ground")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sigma_ground.materia import translate

CASES = [
    # ── varied physics phrasings (should ROUTE) ──────────────────────────
    ("route", "chuck a lead cannonball off a 2 km cliff, how quick is it going when it splats?"),
    ("route", "velocity at impact for a 10 cm tungsten sphere released from 5000 metres"),
    ("route", "terminal velocity of a steel marble"),
    ("route", "a meteorite plummets from 80 km — impact speed?"),
    ("route", "does an iron cannonball get hot falling from the sky?"),
    ("route", "5cm copper ball, 10km up, how hard does it hit"),
    ("route", "what speed does a dropped bowling ball reach at the ground"),
    # ── chains (should be MULTI-STEP) ────────────────────────────────────
    ("chain", "if I yeet a copper ball straight up at 250 m/s, how fast and how hot is it when it comes back?"),
    ("chain", "throw an iron sphere upward at 300 m/s and tell me its landing speed"),
    # ── NON-PHYSICS trigger-word BAIT (should DECLINE) ───────────────────
    ("decline", "how fast is my internet connection?"),
    ("decline", "how hot is it outside in Phoenix today?"),
    ("decline", "any speed dating advice?"),
    ("decline", "how hard should I push myself at the gym?"),
    ("decline", "I'm burning out at work, any tips?"),
    ("decline", "what's a good recipe to cook for dinner?"),
    ("decline", "how fast can I learn Spanish?"),
    ("decline", "my phone is running hot, is that normal?"),
    ("decline", "what's the top speed of a Ferrari?"),
    ("decline", "this song is a banger, it's fire"),
    ("decline", "she gave me the cold shoulder"),
    ("decline", "how hard is the bar exam?"),
    ("decline", "melt in your mouth chocolate, where to buy?"),
    ("decline", "what's the impact of inflation on velocity of money?"),
    # ── out-of-model physics families (should DECLINE) ───────────────────
    ("decline", "how fast does light travel in a vacuum?"),
    ("decline", "speed of sound at sea level?"),
    ("route",   "how fast does the Earth orbit the Sun?"),   # orbital — now BUILT
    ("decline", "what's the muzzle velocity of an AK-47?"),
    # ── substring / metaphor traps (should DECLINE) ─────────────────────
    ("route",   "how fast does the Earth orbit the Sun?"),        # orbital — now BUILT
    ("decline", "is diversity training making us faster?"),       # diver→diversity
    ("decline", "what's the objective top speed of progress?"),   # object→objective
    ("decline", "how hard did the stock market fall today?"),     # metaphor 'fall'
    ("decline", "what's the velocity of money in the economy?"),
    ("decline", "how hot is this new mixtape?"),
    ("decline", "I might throw up after that ride"),              # 'throw up' = vomit
    ("decline", "how fast does a waterfall flow?"),               # 'fall' in waterfall
    ("decline", "drop the mic, how hard was that?"),              # 'drop' idiom
    ("decline", "prices are falling, how fast?"),                 # metaphor fall
    ("decline", "the temperature is falling fast tonight"),       # weather
    ("decline", "how fast can a person learn to code?"),
    ("decline", "how hard did he fall for her?"),                 # metaphor
    # ── slang + object (should ROUTE) ───────────────────────────────────
    ("route", "lob a brick off the roof — how hard does it land?"),
    ("route", "a watermelon dropped off a building, splat speed?"),
    ("route", "yeet a steak off a cliff, does it cook on the way down?"),
    # ── malformed / empty / nonsense (should DECLINE, no crash) ──────────
    ("decline", ""),
    ("decline", "   "),
    ("decline", "asdkfjalskdfjqwe"),
    ("decline", "????"),
    ("decline", "the the the and and"),
    ("decline", "purple monkey dishwasher"),
]


def main():
    bad = []
    for expected, q in CASES:
        try:
            spec = translate(q, use_qwen=False)
            steps = [s.verb for s in spec.steps]
            got = ("decline" if not steps else
                   "chain" if len(steps) > 1 else "route")
        except Exception as e:
            steps, got = [f"CRASH: {e}"], "crash"
            traceback.print_exc()
        ok = (got == expected
              or (expected == "route" and got in ("route", "chain"))
              or (expected == "chain" and got == "chain"))
        mark = "  " if ok else "✗ "
        if not ok:
            bad.append((expected, got, q, steps))
        verbs = ",".join(steps) if steps else "—"
        print(f"{mark}[{expected:7s}→{got:7s}] {verbs:45s} {q[:46]}")

    # Edge-value CRASH tests — actually RUN the verb on degenerate inputs.
    from sigma_ground.materia import answer
    print("\n--- edge-value crash tests (run the verb) ---")
    EDGE = [
        "how fast does a 0 cm ball hit the ground from 0 km?",
        "drop a steel ball from 1000000000 km, impact speed?",
        "does a 0 mm iron ball heat up falling from 10 km?",
        "how hard does a 500 cm boulder hit from 100 km?",
    ]
    for q in EDGE:
        try:
            out = answer(q, use_qwen=False)
            head = (out.splitlines()[0] if out else "(empty)")[:58]
            print(f"  ok    {q[:44]} -> {head}")
        except Exception as e:
            print(f"  CRASH {q[:44]} -> {type(e).__name__}: {e}")
            bad.append(("edge", "crash", q, [f"{type(e).__name__}: {e}"]))

    print("\n" + "=" * 70)
    if not bad:
        print("NO MISTAKES — battery clean.")
    else:
        print(f"{len(bad)} MISTAKES:")
        for expected, got, q, steps in bad:
            kind = ("MISROUTE (confidently wrong)" if expected == "decline"
                    and got in ("route", "chain") else
                    "BAD-DECLINE (should answer)" if expected in ("route", "chain")
                    and got == "decline" else f"{expected}→{got}")
            print(f"  • {kind}: {','.join(steps)}  <- {q[:55]}")


if __name__ == "__main__":
    main()
