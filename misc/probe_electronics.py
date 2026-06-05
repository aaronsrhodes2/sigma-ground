"""Verify electronics function signatures + values before exposing."""
import inspect
import sys

sys.path.insert(0, r"D:\Aaron\development\sigma-ground")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from sigma_ground.field.interface import electronics as E

SEMI, METAL = "silicon", "copper"
_E = inspect.Parameter.empty


def guess(p, fname):
    n = p.name.lower()
    ann = p.annotation
    semi_ctx = any(k in fname for k in ("band", "carrier", "diode", "junction",
                                        "depletion", "intrinsic", "dos", "fermi"))
    if "sigma" in n:
        return 0.0
    if "semicond" in n or n in ("semi", "semiconductor_key"):
        return SEMI
    if (("metal" in n or "material" in n or n.endswith("_key") or n == "key")
            and "doping" not in n):
        return SEMI if semi_ctx else METAL
    if n in ("t", "t_k") or "temp" in n:
        return 300.0
    if "doping" in n or n.startswith(("n_a", "n_d", "n_i")) or "concentration" in n:
        return 1e22
    if "voltage" in n or n == "v" or n.endswith("_v"):
        return 0.5
    if "area" in n:
        return 1e-4
    if any(k in n for k in ("gap", "distance", "separation", "thickness")) or n == "d":
        return 1e-3
    if "permittivity" in n or "epsilon" in n or "dielectric" in n or "_r" == n[-2:]:
        return 11.7
    if "current" in n or n.endswith("_a"):
        return 1e-12
    if n.endswith("_m"):
        return 1e-3
    if ann is int:
        return 1
    if ann is float:
        return 1.0
    return None


FUNCS = ["resistivity", "carrier_mobility", "band_gap",
         "intrinsic_carrier_concentration", "hall_coefficient", "mean_free_path",
         "parallel_plate_capacitance", "built_in_voltage", "diode_current",
         "depletion_width", "junction_capacitance", "carrier_concentration",
         "effective_dos_conduction", "free_electron_density"]

for fn in FUNCS:
    f = getattr(E, fn, None)
    if not f:
        print(f"{fn}: MISSING")
        continue
    sig = inspect.signature(f)
    args, ok = [], True
    for p in sig.parameters.values():
        if p.default is not _E or p.kind not in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY):
            continue
        g = guess(p, fn)
        if g is None:
            ok = False
            break
        args.append(g)
    try:
        r = f(*args) if ok else "(could not build args)"
        rs = f"{r:.4g}" if isinstance(r, (int, float)) else str(r)[:60]
        print(f"{fn}{sig}\n    args={args} -> {rs}")
    except Exception as e:
        print(f"{fn}{sig}\n    args={args} -> ERR {type(e).__name__}: {str(e)[:50]}")
