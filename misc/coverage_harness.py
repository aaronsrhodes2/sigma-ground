"""Coverage harness â€” call every calculation function in the science tree.

Auto-infers valid arguments per function (from parameter names, annotations, and
each module's own data-table keys), calls it with stdout suppressed, and reports:
  OK   â€” returned a value (exercised âœ“)
  ERR  â€” raised (potential bug, or needs better args)
  SKIP â€” couldn't construct required args (needs a hand-written question)

This is the engine for the coverage campaign. Run it, drive OK upward, and the
ERR/SKIP lists are the worklist. Writes misc/coverage_report.txt.
"""
import contextlib
import importlib
import inspect
import io
import pkgutil
import sys

sys.path.insert(0, r"D:\Aaron\development\sigma-ground-mentat")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

with contextlib.redirect_stdout(io.StringIO()):
    import sigma_ground

TARGET = ("sigma_ground.field", "sigma_ground.inventory")
EXCLUDE = ("proof", "demo", "render", "sandbox", "scorecard", "__main__",
           "patent", "ephemeris", "jpl", "horizons", "adapters", "fixtures",
           ".tests", "test_", "benchmark", "__pycache__")

GLOBAL_KEYS = ["copper", "iron", "water", "aluminum", "silicon", "gold",
               "acetic_acid", "silver_chloride", "hydrogen", "zinc", "C", "H", "O"]

_EMPTY = inspect.Parameter.empty
_POS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
_KEY = ("__KEY__",)


_OBJ = {"body", "formula", "formulas", "composition", "parcel", "structure",
        "scene", "molecule", "atoms", "config", "graph", "tree", "nodes",
        "layers", "obj", "shape", "df", "data"}
_KEYTOK = {"material", "metal", "salt", "acid", "compound", "solvent",
           "cathode", "anode", "element", "ion", "species", "reaction",
           "liquid", "key", "isotope", "nuclide"}
_UNIT_SUFFIX = ("_s", "_m", "_kg", "_fm", "_hz", "_pa", "_v", "_a", "_mol",
                "_nm", "_pm", "_k", "_j", "_w", "_deg", "_rad", "_mev",
                "_kev", "_ev", "_gev", "_km", "_au", "_yr")


def infer(p):
    n = p.name.lower()
    ann = p.annotation
    toks = set(n.replace("_", " ").split())
    if "sigma" in n:
        return 0.0
    if toks & _OBJ:                       # structured arg â†’ skip (no spurious ERR)
        return None
    if "atom" in toks:
        return "C"
    # fractions / ratios first, so 'ratio_base_acid' isn't read as an acid key
    if any(k in n for k in ("ratio", "fraction", "efficiency", "coefficient",
                            "factor", "emissivity", "albedo", "quotient")):
        return 0.5
    if "temperature" in n or n in ("t", "t_k", "t0", "temp"):
        return 300.0
    if "wavelength" in n:
        return 5e-7
    if "radius" in n or n in ("r", "r0", "dr", "d", "r_s", "rho"):
        return 1.0
    if any(k in n for k in ("molality", "molarity", "concentration")):
        return 1.0
    if "pressure" in n:
        return 101325.0
    if "frequency" in n:
        return 1e9
    if "voltage" in n or n == "v":
        return 1.0
    if "current" in n:
        return 1.0
    if "time" in n or n in ("t_s", "tau"):
        return 1.0
    if "angle" in n:
        return 30.0
    if "mass" in n or n in ("m", "m_kg"):
        return 1.0
    if "energy" in n or n in ("q", "q_mev"):
        return 1.0
    if any(k in n for k in ("velocity", "speed")):
        return 100.0
    if n in ("z", "atomic_number"):
        return 6
    if n == "a":                          # mass number (also fine as a generic dim)
        return 12
    if n in ("n", "n0") or "number" in n:
        return 1000.0
    if (n.startswith("n_") or n in ("count", "electrons", "domains",
                                    "lone_pairs", "bond_order", "order",
                                    "replicas", "z_cation", "z_anion",
                                    "z_daughter", "z_alpha", "a_daughter")):
        return 1
    if n.endswith(_UNIT_SUFFIX):
        return 1.0
    if n in ("b", "c", "l", "x", "y", "h", "g", "p", "e", "f", "w", "u"):
        return 1.0
    if toks & _KEYTOK or n.endswith("_key"):
        return _KEY
    if ann is int:
        return 1
    if ann is float:
        return 1.0
    if ann is str:
        return _KEY
    return None


def module_keys(m):
    ks = []
    for nm in dir(m):
        if nm.startswith("_"):
            continue
        obj = getattr(m, nm, None)
        if isinstance(obj, dict):
            ks.extend(list(obj.keys())[:6])
    return [k for k in ks if isinstance(k, str)]


def try_call(f, required, mkeys):
    inferred = [infer(p) for p in required]
    if any(x is None for x in inferred):
        return "SKIP", "args"
    key_pos = [i for i, x in enumerate(inferred) if x == _KEY]
    candidates = (mkeys + GLOBAL_KEYS) if key_pos else [None]
    last = None
    for key in candidates[:18]:
        args = [key if inferred[i] == _KEY else inferred[i]
                for i in range(len(inferred))]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                f(*args)
            return "OK", None
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:40]}"
    return "ERR", last


def main():
    results = {}   # area -> {OK,ERR,SKIP}
    errs, skips = [], []
    total = 0
    for info in pkgutil.walk_packages(sigma_ground.__path__, "sigma_ground."):
        name = info.name
        if not name.startswith(TARGET) or any(s in name for s in EXCLUDE):
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                m = importlib.import_module(name)
        except Exception:
            continue
        mkeys = module_keys(m)
        area = ".".join(name.split(".")[1:3])
        bucket = results.setdefault(area, {"OK": 0, "ERR": 0, "SKIP": 0})
        for fn in dir(m):
            f = getattr(m, fn, None)
            if (fn.startswith("_") or not inspect.isfunction(f)
                    or getattr(f, "__module__", "") != name):
                continue
            try:
                sig = inspect.signature(f)
            except (TypeError, ValueError):
                continue
            required = [p for p in sig.parameters.values()
                        if p.default is _EMPTY and p.kind in _POS]
            total += 1
            verdict, detail = try_call(f, required, mkeys)
            bucket[verdict] += 1
            if verdict == "ERR":
                errs.append(f"{name}.{fn}  ({detail})")
            elif verdict == "SKIP":
                skips.append(f"{name}.{fn}{sig}")

    ok = sum(b["OK"] for b in results.values())
    er = sum(b["ERR"] for b in results.values())
    sk = sum(b["SKIP"] for b in results.values())
    print("â•â• coverage harness (field + inventory) â•â•")
    for area in sorted(results, key=lambda a: -sum(results[a].values())):
        b = results[area]
        print(f"  {b['OK']:4d} ok  {b['ERR']:4d} err  {b['SKIP']:4d} skip   {area}")
    print(f"\n  TOTAL {total} funcs â†’ {ok} OK ({100*ok/max(total,1):.0f}%) Â· "
          f"{er} ERR Â· {sk} SKIP(need args)")

    out = r"D:\Aaron\development\sigma-ground-mentat\misc\coverage_report.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"COVERAGE: {ok}/{total} OK, {er} ERR, {sk} SKIP\n\n=== ERR ===\n")
        f.write("\n".join(sorted(errs)))
        f.write("\n\n=== SKIP (need hand-specified args) ===\n")
        f.write("\n".join(sorted(skips)))
    print(f"  worklist â†’ {out}")


if __name__ == "__main__":
    main()

