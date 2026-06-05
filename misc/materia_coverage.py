"""Materia coverage check — do we actually pass the sample questions?

Runs the accumulated sample questions (the 'regular person' phrasings of the
ten stress scenarios, plus canonical cases) through the deterministic
translator and scores each:

  ANSWERED  — routed to a built verb and ran           (good, if expected)
  DECLINED  — asked for clarification / not-yet-modeled (good, if expected)
  MISROUTE  — grabbed a verb for a scenario we cannot   (BAD: confidently wrong)
  MISS      — declined something we should answer        (gap)

The point: success is NOT "answer everything." It is "answer what we built,
and HONESTLY DECLINE the rest." A MISROUTE is the only real failure.
"""
import sys
sys.path.insert(0, r"D:\Aaron\development\sigma-ground")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sigma_ground.materia import translate

# (question, expected)  expected ∈ {"answer", "decline"}
# Concise paraphrases of the ten 'regular person' scenarios + canon.
CASES = [
    # statics — not built yet → should DECLINE
    ("A long steel pole sticks out of a wall and tapers skinny at the tip; hang a "
     "big weight on the end — how much does it sag before it snaps?", "decline"),
    ("A heavy safe hangs from a lopsided tripod with hollow legs at crooked angles "
     "— which leg is overloaded and will it buckle?", "decline"),
    ("Tie a heavy copper wire between two towers and pull it tight — how far does "
     "the middle sag down?", "decline"),
    # rigid-body — not built → DECLINE (watch 'top speed', 'how fast')
    ("Hit a pool ball near the top so it skids before it rolls — how far does it "
     "slide first?", "decline"),
    ("A rocket car burns fuel fast and gets lighter while the wind pushes back "
     "harder — what's its top speed when it runs out of gas?", "decline"),
    ("A swinging stick with another stick dangling off it, dropped from sideways "
     "— how fast are both pieces spinning as the top swings straight down?", "decline"),
    # fluid/thermal PDE — not built → DECLINE (watch 'how hard', 'hot', 'burn')
    ("Spin a metal cylinder inside a bigger one with gooey slime in the gap that "
     "thins as it spins — how hard does it drag on the outer wall?", "decline"),
    ("A cold iron pipe gets blasted inside with boiling liquid — how many seconds "
     "until the outside is hot enough to burn your hand?", "decline"),
    # advanced drag — BUILT → should ANSWER
    ("A bullet shot so fast it breaks the sound barrier; as it drops below the "
     "speed of sound the drag eases — how does the model handle that?", "answer"),
    ("A skydiver jumps from so high there's almost no air and goes supersonic — "
     "does the model show them slowing in the thick air below?", "answer"),
    # canon
    ("How fast does a 5 cm copper ball hit the ground dropped from 10 km?", "answer"),
    ("What's the capital of France?", "decline"),
]


def main():
    answered = declined = misroute = miss = 0
    rows = []
    for q, expected in CASES:
        spec = translate(q, use_qwen=False)
        got = "answer" if spec.is_runnable() else "decline"
        verb = spec.steps[0].verb if spec.steps else "—"
        if expected == "answer" and got == "answer":
            verdict, mark = "ANSWERED", "✓"; answered += 1
        elif expected == "decline" and got == "decline":
            verdict, mark = "declined", "✓"; declined += 1
        elif expected == "decline" and got == "answer":
            verdict, mark = f"MISROUTE→{verb}", "✗"; misroute += 1
        else:
            verdict, mark = "MISS", "✗"; miss += 1
        rows.append((mark, verdict, q[:62]))

    for mark, verdict, q in rows:
        print(f"  {mark} {verdict:18s} {q}")
    n = len(CASES)
    print(f"\n  {answered} answered · {declined} honestly declined · "
          f"{misroute} MISROUTE · {miss} miss   (of {n})")
    print("  → MISROUTE is the only real failure (confidently wrong). "
          "Target: 0.")


if __name__ == "__main__":
    main()
