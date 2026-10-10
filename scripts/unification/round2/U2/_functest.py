"""Functional smoke of the U2 machinery at N=16, T=0.05 t_ff (NOT a measurement of the N=64 experiment)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import u2_common as C
import run_sph as R
import run_nbody as NB
from sigma_ground.dynamics.continuum import _R_GAS
print("R_GAS engine", _R_GAS, "Mentat", C.leaves()["R_gas"]["value"])
S = C.setup(16)
for a in (0.0, 1.0):
    R.echo(S, "isothermal", a, 0.05, 600)
for a in (0.0, 1.0):
    R.echo(S, "ideal", a, 0.05, 600)
NB.echo(S, 0.05, 600)
