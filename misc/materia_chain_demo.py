"""Materia chain demo — decomposition with data flow.

A complex question broken into sub-solves: throw a steel ball up, then ask how
fast and how hot it is when it lands. The apex altitude computed by step 0
flows into the descent and heating steps as their drop altitude — proving the
'sub-solve → feed forward → combine' mechanism.
"""
import sys
sys.path.insert(0, r"D:\Aaron\development\sigma-ground")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sigma_ground.materia import (SimulationSpec, SpecStep, run_spec,
                                  synthesize_chain)

chain = SimulationSpec(
    "Throw a steel ball straight up at 400 m/s — how fast and how hot is it "
    "when it lands?",
    steps=[
        SpecStep("vertical_launch",
                 {"material_key": "steel_mild", "radius_m": 0.05,
                  "launch_speed_m_s": 400.0}),
        SpecStep("terminal_velocity_drop",
                 {"material_key": "steel_mild", "radius_m": 0.05},
                 bindings={"drop_altitude_m": (0, "apex_altitude_m")}),
        SpecStep("drag_heating_drop",
                 {"material_key": "steel_mild", "radius_m": 0.05},
                 bindings={"drop_altitude_m": (0, "apex_altitude_m")}),
    ])

print("=" * 70)
print("MATERIA CHAIN — sub-solve → feed forward → combine")
print("=" * 70)
results = run_spec(chain)

apex = results[0].outputs["apex_altitude_m"]
print(f"\nDATA FLOW: step 0 computed apex = {apex:.0f} m")
print(f"           → step 1 received drop_altitude_m = "
      f"{results[1].inputs['drop_altitude_m']:.0f} m")
print(f"           → step 2 received drop_altitude_m = "
      f"{results[2].inputs['drop_altitude_m']:.0f} m")

print("\n" + synthesize_chain(chain, results))

print("\n" + "-" * 70)
print("Per-step detail:")
for r in results:
    print("\n" + r.render())
