"""Deckard smoke demo — a name becomes validated matter."""
import sys
sys.path.insert(0, r"D:\Aaron\development\sigma-ground")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sigma_ground.deckard import identify, research

print("=" * 64)
print("DECKARD — the shape researcher")
print("=" * 64)

# 1) Identify a known item → researched dimensions → compiled, validated matter
print("\n" + research("coffee cup").note())
cup = identify("coffee cup", resolution=72)
print(cup.render())

# 2) material_at probe — the layered SDF resolves the right material everywhere
print("\n  material_at probe (x,y,z in m):")
probes = [
    ("outer skin", (0.0398, 0.0, 0.05)),
    ("wall",       (0.037,  0.0, 0.05)),
    ("water",      (0.0,    0.0, 0.03)),
    ("headspace",  (0.0,    0.0, 0.09)),
    ("base",       (0.0,    0.0, 0.003)),
]
for label, (x, y, z) in probes:
    print(f"    {label:10s} ({x:.4f},{y:.3f},{z:.3f}) → {cup.material_at(x, y, z)}")

# 3) Unidentified item → flagged best-guess, never a confident fake
print("\n" + "-" * 64)
spec = research("flux capacitor")
print(spec.note())
u = identify("flux capacitor", resolution=40)
print(f"  identified={u.identified}; defaulted mass ≈ {u.mass_kg*1000:.0f} g (flagged)")
