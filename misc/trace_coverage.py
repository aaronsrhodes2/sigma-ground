"""Transitive coverage â€” which library methods actually EXECUTE when Mentat
answers science questions.

A method counts as covered if it runs as a byproduct of a question, even if we
never call it directly (per the coverage definition). We install a function-call
tracer, run the question corpus (every MCP tool + the procedures, each a science
question), plus a direct pass over field/inventory, and take the UNION of every
sigma_ground function that executed.

Writes misc/uncovered.txt â€” the functions no question reached yet.
"""
import contextlib
import importlib
import inspect
import io
import os
import pkgutil
import sys

sys.path.insert(0, r"D:\Aaron\development\sigma-ground-mentat")
sys.path.insert(0, r"D:\Aaron\development\sigma-ground-mentat\misc")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

with contextlib.redirect_stdout(io.StringIO()):
    import sigma_ground
from coverage_harness import infer, module_keys, try_call, TARGET, EXCLUDE  # reuse arg engine

_EMPTY = inspect.Parameter.empty
_POS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)


def dotted(path):
    parts = path.replace("\\", "/").split("/")
    if "sigma_ground" not in parts:
        return None
    i = parts.index("sigma_ground")
    mod = ".".join(parts[i:]).rsplit(".py", 1)[0]
    # Normalize a package __init__ to the package name so functions defined in
    # __init__.py match the registry (which keys them by the package name).
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    return mod


covered = set()


def tracer(frame, event, arg):
    if event == "call":
        mod = dotted(frame.f_code.co_filename)
        if mod and mod.startswith("sigma_ground"):
            covered.add((mod, frame.f_code.co_name))
    return None


def collect(prefixes):
    """All public module-level functions under the given package prefixes."""
    out = []
    for info in pkgutil.walk_packages(sigma_ground.__path__, "sigma_ground."):
        name = info.name
        if not name.startswith(prefixes) or any(s in name for s in EXCLUDE):
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                m = importlib.import_module(name)
        except Exception:
            continue
        for fn in dir(m):
            f = getattr(m, fn, None)
            if (not fn.startswith("_") and inspect.isfunction(f)
                    and getattr(f, "__module__", "") == name):
                out.append((name, fn, f, m))
    return out


def exercise(items):
    for _, _, f, m in items:
        try:
            req = [p for p in inspect.signature(f).parameters.values()
                   if p.default is _EMPTY and p.kind in _POS]
        except (TypeError, ValueError):
            continue
        try:
            try_call(f, req, module_keys(m))
        except Exception:
            pass


def main():
    tree = collect(TARGET)                         # field + inventory = denominator
    target = {(mod, fn) for mod, fn, _, _ in tree}
    tools = collect(("sigma_ground.mcp.tools",))   # the question/answer surface

    sys.settrace(tracer)
    try:
        exercise(tools)                            # questions â†’ transitive library reach
        exercise(tree)                             # direct pass (also counts)
        from sigma_ground.mcp import procedures as P
        for fn, args in [("black_hole_profile", (1.989e30,)),
                         ("photon_spectrum", (5e-7,)),
                         ("relativistic_particle", (1e6, "electron")),
                         ("projectile_trajectory", (100.0, 45.0)),
                         ("stellar_blackbody", (5778.0,))]:
            try:
                getattr(P, fn)(*args)
            except Exception:
                pass
    finally:
        sys.settrace(None)

    hit = target & covered
    print(f"â•â• transitive coverage (field + inventory) â•â•")
    print(f"  covered {len(hit)}/{len(target)} = {100*len(hit)/max(len(target),1):.0f}%")
    print(f"  (total distinct sigma_ground funcs that executed: {len(covered)})")

    uncovered = sorted(target - covered)
    by_area = {}
    for mod, fn in uncovered:
        a = ".".join(mod.split(".")[1:3])
        by_area[a] = by_area.get(a, 0) + 1
    print("  top uncovered areas:")
    for a in sorted(by_area, key=lambda x: -by_area[x])[:12]:
        print(f"     {by_area[a]:4d}  {a}")
    out = r"D:\Aaron\development\sigma-ground-mentat\misc\uncovered.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(f"{m}.{fn}" for m, fn in uncovered))
    print(f"  uncovered list â†’ {out}")


if __name__ == "__main__":
    main()

