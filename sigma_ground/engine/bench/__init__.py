"""engine/bench — the timing + determinism harness (Phase E0).

Prerequisite for every native-porting decision in the game-engine plan:
before this existed, no ms/step number existed anywhere in the repo for
any integrator. ``harness.py`` measures wall time per step at realistic
entity counts and writes committed JSON baselines; ``determinism.py``
proves each backend reproduces bit-identical state from a fixed seed —
the gate a native port must also pass against its Python twin.
"""
