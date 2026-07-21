"""Smoke-test baseline for the sigma_ground.mcp.tools._safe() investigation.

Walks every public top-level function across the 19 mcp/tools/*.py files that
use the local ``_safe(fn, *a, **k)`` swallow-everything helper, calls each one
via ``misc/coverage_harness.py``'s existing infer/module_keys/try_call engine,
and records the OK/ERR/SKIP verdict for every function.

This is NOT a new inference engine -- it reuses coverage_harness.py verbatim
(infer/module_keys/try_call are imported, not reimplemented). It exists because
coverage_harness.py's own main() only walks sigma_ground.field/.inventory and
discards the verdict; this script walks sigma_ground.mcp.tools.* instead and
keeps every verdict, so we have a per-function BEFORE snapshot of what
currently works vs errors vs can't-be-called, ahead of reworking the _safe()
error-swallowing pattern (see sigma_ground/mcp/tools/mathx.py's _ok()/_err()
idiom for the target shape).

Import-order note: coverage_harness.py itself does
``sys.path.insert(0, r"D:\\Aaron\\development\\sigma-ground-mentat")`` at
module scope before its own ``import sigma_ground`` -- sigma-ground-mentat
carries its OWN (older, smaller) sigma_ground package, so a naive import of
coverage_harness would silently shadow this repo's sigma_ground with that
stale copy. We dodge it by importing sigma_ground correctly (from this repo)
*before* importing coverage_harness, so the cached sys.modules entry wins and
coverage_harness's redundant `import sigma_ground` becomes a no-op. Nothing in
coverage_harness.py itself is modified.

Usage:
    python misc/mcp_tools_safe_baseline.py
    python misc/mcp_tools_safe_baseline.py > misc/mcp_tools_safe_baseline_before.txt
"""
from __future__ import annotations

import contextlib
import importlib
import inspect
import io
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_MISC = os.path.join(_ROOT, "misc")
if _MISC not in sys.path:
    sys.path.insert(0, _MISC)

# Import the REAL sigma_ground (this repo) first and force it into sys.modules
# before coverage_harness.py gets a chance to point sys.path at the stale
# sigma-ground-mentat copy. See module docstring above.
with contextlib.redirect_stdout(io.StringIO()):
    import sigma_ground  # noqa: E402

_expected = os.path.join(_ROOT, "sigma_ground", "__init__.py")
if os.path.normcase(os.path.abspath(sigma_ground.__file__)) != os.path.normcase(_expected):
    raise RuntimeError(
        "sigma_ground resolved from the wrong location -- expected "
        f"{_expected!r}, got {sigma_ground.__file__!r}. Refusing to run "
        "the baseline against a possibly-stale physics library."
    )

from coverage_harness import infer, module_keys, try_call  # noqa: E402,F401

_EMPTY = inspect.Parameter.empty
_POS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)

# The 19 files under sigma_ground/mcp/tools/ that define a local _safe() helper.
TOOL_FILES = [
    "materials_micro",
    "transport",
    "tribology",
    "thermal_systems",
    "rotation",
    "quantum_solids",
    "relativity_spectra",
    "playground",
    "qcomputing",
    "plasma_em",
    "photonics",
    "misc_physics",
    "mechanical_response",
    "materials_strength",
    "inventory_tools",
    "fluid_surface",
    "electroceramics",
    "devices",
    "chemistry_extended",
]


def collect_functions(mod):
    """Public, top-level, module-owned functions of `mod` (skip re-exports)."""
    out = []
    for fn in sorted(dir(mod)):
        if fn.startswith("_"):
            continue
        f = getattr(mod, fn, None)
        if not inspect.isfunction(f) or getattr(f, "__module__", "") != mod.__name__:
            continue
        try:
            sig = inspect.signature(f)
        except (TypeError, ValueError):
            continue
        out.append((fn, f, sig))
    return out


def run():
    results = []  # list of (file, function, verdict, detail)
    import_errors = []

    for file_stem in TOOL_FILES:
        mod_name = f"sigma_ground.mcp.tools.{file_stem}"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                mod = importlib.import_module(mod_name)
        except Exception as e:
            import_errors.append((file_stem, f"{type(e).__name__}: {e}"))
            continue

        mkeys = module_keys(mod)
        for fn_name, f, sig in collect_functions(mod):
            required = [p for p in sig.parameters.values()
                        if p.default is _EMPTY and p.kind in _POS]
            verdict, detail = try_call(f, required, mkeys)
            results.append((file_stem, fn_name, verdict, detail))

    return results, import_errors


def summarize(results, import_errors):
    lines = []
    w = lines.append

    w("=" * 78)
    w("mcp/tools _safe() baseline -- smoke test of every public function")
    w("across the 19 sigma_ground/mcp/tools/*.py files that use _safe()")
    w("=" * 78)
    w(f"sigma_ground resolved from: {sigma_ground.__file__}")
    w("")

    if import_errors:
        w("!! MODULE IMPORT ERRORS (function walk skipped for these files) !!")
        for file_stem, msg in import_errors:
            w(f"  {file_stem}: {msg}")
        w("")

    total = len(results)
    oks = [r for r in results if r[2] == "OK"]
    errs = [r for r in results if r[2] == "ERR"]
    skips = [r for r in results if r[2] == "SKIP"]

    w(f"TOTAL functions tried: {total}")
    w(f"  OK   : {len(oks)}")
    w(f"  ERR  : {len(errs)}")
    w(f"  SKIP : {len(skips)}")
    w("")

    # Per-file breakdown
    w("-" * 78)
    w("Per-file breakdown")
    w("-" * 78)
    per_file = {}
    for file_stem, fn_name, verdict, detail in results:
        b = per_file.setdefault(file_stem, {"OK": 0, "ERR": 0, "SKIP": 0})
        b[verdict] += 1
    for file_stem in TOOL_FILES:
        b = per_file.get(file_stem)
        if b is None:
            w(f"  {file_stem:24s}  (import failed, see above)")
            continue
        w(f"  {file_stem:24s}  OK={b['OK']:3d}  ERR={b['ERR']:3d}  SKIP={b['SKIP']:3d}")
    w("")

    w("-" * 78)
    w(f"ERR entries ({len(errs)}) -- file / function / exception")
    w("-" * 78)
    if not errs:
        w("  (none)")
    for file_stem, fn_name, verdict, detail in errs:
        w(f"  {file_stem}.{fn_name}  ->  {detail}")
    w("")

    w("-" * 78)
    w(f"SKIP entries ({len(skips)}) -- file / function (couldn't infer required args)")
    w("-" * 78)
    if not skips:
        w("  (none)")
    for file_stem, fn_name, verdict, detail in skips:
        w(f"  {file_stem}.{fn_name}")
    w("")

    return "\n".join(lines)


if __name__ == "__main__":
    results, import_errors = run()
    report = summarize(results, import_errors)
    print(report)
