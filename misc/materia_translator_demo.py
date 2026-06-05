"""Materia translator demo — plain English → simulation → answer.

The whole thesis on one screen: arbitrary What-If phrasings get translated to a
Simulation Spec (one or two verbs), the engine runs it, and the worked,
self-validated answer comes back. The deterministic path needs no LLM; the
qwen path (if ollama is up) handles phrasings the keywords miss.
"""
import sys
sys.path.insert(0, r"D:\Aaron\development\sigma-ground")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sigma_ground.materia import answer, translate
from sigma_ground.materia.translator import _ollama_chat

print("=" * 70)
print("MATERIA TRANSLATOR — natural language → simulation → answer")
print("=" * 70)

# Deterministic routing — different questions, different verbs, no LLM.
DETERMINISTIC = [
    "How fast does a 5 cm copper ball hit the ground if you drop it from 10 km?",
    "Does an iron cannonball heat up much falling from the stratosphere?",
    "Drop a 2 cm lead sphere from 30 km — how fast does it land and how hot does it get?",
    "What's the capital of France?",          # → clarification, not a guess
]

for q in DETERMINISTIC:
    print("\n" + "▌" + "─" * 68)
    print(f"▌ Q: {q}")
    print("▌" + "─" * 68)
    print(answer(q, use_qwen=False))

# Family D — advanced drag: high-altitude descent + supersonic projectile.
FAMILY_D = [
    "If a skydiver free-falls from 35 km, do they go supersonic and then slow down?",
    "How does a bullet's drag change as it breaks the sound barrier at Mach 2.5?",
]
for q in FAMILY_D:
    print("\n" + "▌" + "─" * 68)
    print(f"▌ Q: {q}")
    print("▌" + "─" * 68)
    print(answer(q, use_qwen=False))

# qwen residual — a phrasing the keywords miss ("toasty", "marble").
print("\n\n" + "=" * 70)
print("qwen RESIDUAL  (handles phrasings the keywords miss)")
print("=" * 70)
weird = "If I chucked a titanium marble off a 20-kilometer cliff, would it get toasty?"
print(f"\nQ: {weird}")
if _ollama_chat("ping") is None:
    print("  (ollama not reachable at localhost:11434 — deterministic path only.\n"
          "   With qwen up, this routes to drag_heating_drop(titanium, …).)")
    # Show the deterministic verdict for transparency:
    print("  deterministic →", "clarify" if not translate(weird, use_qwen=False).is_runnable()
          else "matched")
else:
    print(answer(weird, use_qwen=True))
