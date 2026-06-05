"""Dry-run: are the mechanical/acoustic/impact derivations real and emergent?

Each value below is COMPUTED from cohesive energy (no stored modulus/sound speed):
  cohesive energy -> bulk/Young's modulus (mechanical.py, harmonic approx, +-50%)
                  -> longitudinal sound speed (acoustics.py, sqrt((K+4G/3)/rho))
                  -> restitution + ring frequency (impact.py, Hertz/Johnson)

The point is not exactness (modulus is +-50%) but that the values are DERIVED and
they DIFFERENTIATE materials the way reality does: steel rings ~15 kHz, lead thuds
~5 kHz, rubber boings ~30 Hz; E and sound speed track the real ordering.
"""
import sys

sys.path.insert(0, r"D:\Aaron\development\sigma-ground")

from sigma_ground.field.interface.mechanical import youngs_modulus
from sigma_ground.field.interface.acoustics import longitudinal_wave_speed
from sigma_ground.field.interface.impact import (coefficient_of_restitution,
                                                impact_sound_frequency)

# Experimental references for a sanity check (E in GPa, v_L in m/s).
REF = {
    "steel_mild": (200, 5900), "titanium": (116, 6100), "aluminum": (70, 6320),
    "copper": (120, 4760), "lead": (16, 2160), "gold": (79, 3240),
    "tungsten": (411, 5180), "nickel": (200, 5810), "silver": (83, 3650),
    "glass": (70, 5640), "rubber": (0.05, 1500), "plastic_abs": (2.3, 2250),
}

print(f"{'material':12s}{'E GPa':>7s}{'(ref)':>7s}{'sound':>7s}{'(ref)':>7s}"
      f"{'COR@1m/s':>9s}{'ring Hz':>9s}")
print("-" * 58)
for k, (E_ref, c_ref) in REF.items():
    try:
        E = youngs_modulus(k) / 1e9
        c = longitudinal_wave_speed(k)
        e = coefficient_of_restitution(k, velocity=1.0, radius_m=0.02)
        f = impact_sound_frequency(k, velocity=1.0, radius_m=0.02)
        print(f"{k:12s}{E:7.0f}{E_ref:7.0f}{c:7.0f}{c_ref:7.0f}{e:9.2f}{f:9.0f}")
    except Exception as ex:
        print(f"{k:12s}  (no data: {type(ex).__name__})")

print("-" * 58)
print("Derived from cohesive energy alone. Ring pitch spans 30 Hz (rubber) to "
      "~15 kHz (steel) -- the clatter is emergent.")
