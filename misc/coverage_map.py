"""Enumerate sigma-ground's calculation tree — every public function across the
science modules. This is the denominator for the coverage goal: a question that
calls every method.

Writes the full method list to misc/sigma_ground_method_tree.txt and prints a
per-area count.
"""
import importlib
import inspect
import pkgutil
import sys

sys.path.insert(0, r"D:\Aaron\development\sigma-ground")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import sigma_ground

SKIP = (".tests", ".test_", "test_", "benchmark", "__pycache__", ".games")


def walk():
    methods = {}      # module name -> [func names]
    classes = {}      # module name -> [(class, [public methods])]
    failed = {}
    for info in pkgutil.walk_packages(sigma_ground.__path__, "sigma_ground."):
        name = info.name
        if any(s in name for s in SKIP):
            continue
        try:
            m = importlib.import_module(name)
        except Exception as e:
            failed[name] = type(e).__name__
            continue
        funcs = [n for n in dir(m)
                 if not n.startswith("_")
                 and inspect.isfunction(getattr(m, n, None))
                 and getattr(getattr(m, n), "__module__", "") == name]
        if funcs:
            methods[name] = sorted(funcs)
        cls = []
        for cn in dir(m):
            obj = getattr(m, cn, None)
            if (not cn.startswith("_") and inspect.isclass(obj)
                    and getattr(obj, "__module__", "") == name):
                cmeths = [mn for mn in dir(obj)
                          if not mn.startswith("_") and callable(getattr(obj, mn, None))]
                if cmeths:
                    cls.append((cn, cmeths))
        if cls:
            classes[name] = cls
    return methods, classes, failed


def area_of(modname):
    parts = modname.split(".")
    return ".".join(parts[1:3]) if len(parts) > 2 else (parts[1] if len(parts) > 1 else "root")


def main():
    methods, classes, failed = walk()
    total_funcs = sum(len(v) for v in methods.values())
    total_methods = sum(len(ms) for cs in classes.values() for _, ms in cs)

    # per-area function tally
    areas = {}
    for mod, fns in methods.items():
        areas.setdefault(area_of(mod), 0)
        areas[area_of(mod)] += len(fns)

    print("══ sigma-ground calculation tree ══")
    for area in sorted(areas, key=lambda a: -areas[a]):
        print(f"  {areas[area]:4d}  {area}")
    print(f"\n  module-level functions : {total_funcs}  (in {len(methods)} modules)")
    print(f"  public class methods   : {total_methods}  (in {len(classes)} modules)")
    if failed:
        print(f"  modules that failed to import: {len(failed)}")
        for n, e in list(failed.items())[:8]:
            print(f"     - {n}: {e}")

    out = r"D:\Aaron\development\sigma-ground\misc\sigma_ground_method_tree.txt"
    with open(out, "w", encoding="utf-8") as f:
        for mod in sorted(methods):
            f.write(f"# {mod}\n")
            for fn in methods[mod]:
                f.write(f"{mod}.{fn}\n")
    print(f"\n  full function list → {out}")


if __name__ == "__main__":
    main()
