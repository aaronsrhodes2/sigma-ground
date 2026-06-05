"""Materia smoke demo — run the first scenario and the engine self-check.

Pins sys.path to the canonical sigma-ground (the cwd worktree carries a stale
sigma_ground that would shadow it).
"""
import sys
sys.path.insert(0, r"D:\Aaron\development\sigma-ground")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sigma_ground.materia import (terminal_velocity_drop, drag_heating_drop,
                                  simulate_fall, analytic_terminal_velocity)

print("=" * 64)
print("MATERIA — combined physics of matter and energy")
print("=" * 64)

# 1) The headline scenario: copper sphere from 10 km (the Terminal-Velocity Pivot)
res = terminal_velocity_drop("copper", radius_m=0.05, drop_altitude_m=10_000.0)
print(res.render())

# 2) Engine self-check: in a UNIFORM sea-level atmosphere the simulated
#    asymptote must equal the closed-form terminal velocity to integration tol.
print("\n" + "-" * 64)
print("ENGINE SELF-CHECK (uniform density → exact closed form)")
u = simulate_fall("copper", 0.05, 8000.0, uniform=True)
v_t = analytic_terminal_velocity(u["mass_kg"], u["area_m2"],
                                 u["sea_level_density_kg_m3"], cd=0.44)
err = abs(u["impact_speed_m_s"] - v_t) / v_t
print(f"  simulated asymptote : {u['impact_speed_m_s']:.3f} m/s")
print(f"  closed-form v_t      : {v_t:.3f} m/s")
print(f"  relative error       : {err*100:.2f}%   "
      f"({'PASS' if err < 0.02 else 'FAIL'} @ 2%)")
print(f"  steps integrated     : {u['n_steps']}")

# 3) Material sweep — same sphere, different matter (shows the cascade)
print("\n" + "-" * 64)
print("MATTER SWEEP (5 cm sphere from 10 km, realistic atmosphere)")
for mk in ("lead", "steel_mild", "copper", "iron"):
    s = simulate_fall(mk, 0.05, 10_000.0)
    print(f"  {s['material_name']:<8} ρ={s['density_kg_m3']:>5.0f}  "
          f"m={s['mass_kg']:5.2f} kg  →  impact {s['impact_speed_m_s']:6.1f} m/s  "
          f"(peak {s['max_speed_m_s']:6.1f} m/s aloft)")

# 4) Movement → thermal: drag heating (the v2 coupling — movement as CAUSE)
print("\n" + "-" * 64)
print("MOVEMENT → THERMAL  (drag heating, v2)")
print(drag_heating_drop("iron", radius_m=0.05, drop_altitude_m=10_000.0).render())
