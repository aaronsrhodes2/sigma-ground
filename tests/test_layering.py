"""Architectural layering guard for the Mentat monorepo (zero-dependency).

Enforces the role-tier import contract — a module in a given role may only
import roles at its own tier or below:

    kernel(0) <- {field, inventory, dynamics}(1) <- deckard(2)
              <- {materia, radiance, labs}(3) <- mcp(4)

Notes
-----
* ``field.constants`` is the authoritative physical-constants foundation and
  is importable from anywhere (treated as tier 0).
* Relative imports (``from ..shapes import X``) are resolved to absolute so the
  check sees the same edges Python does.
* The top-level package facade ``sigma_ground/__init__.py`` is exempt — it is
  the public aggregator and re-exports across tiers by design.

Pure stdlib (``ast`` + ``pathlib``) so the zero-dependency core stays that way.
Run as part of every checkpoint; a layering regression fails loudly here.
"""
from __future__ import annotations

import ast
import pathlib

PKG = "sigma_ground"
ROOT = pathlib.Path(__file__).resolve().parent.parent / PKG
REPO = ROOT.parent

# role (first path component under sigma_ground/) -> tier
TIER = {
    "kernel": 0,
    "field": 1, "inventory": 1, "dynamics": 1,
    "deckard": 2,
    "materia": 3, "radiance": 3, "labs": 3,
    "mcp": 4,
}

# Pre-existing backward edges to RETIRE (grandfathered, not new debt). Each
# entry is (importer_module, imported_prefix). Remove the entry once the edge
# is gone — the guard then enforces the contract with no exceptions.
ALLOW = {
    # materia.scenarios.orbital_velocity() wraps the body-aware orbital math
    # that currently lives in mcp.tools.orbital (tier 4). The fix is to extract
    # that math to a shared lower tier; deferred because orbital.py is part of
    # in-flight stashed WIP and refactoring it now would collide on stash pop.
    ("sigma_ground.materia.scenarios", "sigma_ground.mcp.tools.orbital"),
    # The web cockpit (radiance/web/serve.py) wires the Mentat front door into a
    # /chat endpoint — a tier-3 static server reaching the tier-4 dispatcher. It's
    # a composition seam (an entry point), not library coupling. RETIRE by moving
    # the dynamic server to a tier-4 entry (e.g. sigma_ground/mcp/cockpit.py) that
    # imports both serve (static) and front_door; serve.py then stays pure static.
    ("sigma_ground.radiance.web.serve", "sigma_ground.mcp.front_door"),
}


def _dotted(path: pathlib.Path) -> str:
    """Fully-qualified module name for a .py file under sigma_ground/."""
    parts = list(path.relative_to(REPO).parts)
    parts[-1] = parts[-1][:-3]  # strip .py
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _containing_package(dotted: str, is_init: bool) -> list[str]:
    parts = dotted.split(".")
    return parts if is_init else parts[:-1]


def _role_tier(role: str) -> int:
    return TIER.get(role, 0)  # loose top-level modules -> kernel tier


def _imported_tier(module: str) -> int | None:
    """Tier of an imported module, or None if it is not intra-package."""
    parts = module.split(".")
    if parts[0] != PKG or len(parts) < 2:
        return None
    if parts[1] == "field" and len(parts) >= 3 and parts[2] == "constants":
        return 0  # authoritative constants = universal foundation
    return _role_tier(parts[1])


def _resolve(node: ast.ImportFrom, pkg_parts: list[str]) -> str | None:
    if node.level == 0:
        return node.module
    base = pkg_parts[: len(pkg_parts) - (node.level - 1)]
    if node.module:
        return ".".join(base + node.module.split("."))
    return ".".join(base) or None


def _imports(tree: ast.AST, pkg_parts: list[str]):
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = _resolve(node, pkg_parts)
            if mod:
                yield mod
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name


def test_role_layering():
    violations: list[str] = []
    for py in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in py.parts or py.name.startswith("test_"):
            continue
        dotted = _dotted(py)
        if dotted == PKG:
            continue  # the public facade is exempt by design
        own_parts = dotted.split(".")
        own_tier = _role_tier(own_parts[1] if len(own_parts) >= 2 else "kernel")
        pkg_parts = _containing_package(dotted, py.name == "__init__.py")
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError as exc:  # pragma: no cover
            violations.append(f"{dotted}: syntax error: {exc}")
            continue
        for mod in _imports(tree, pkg_parts):
            dep_tier = _imported_tier(mod)
            if dep_tier is not None and dep_tier > own_tier:
                if any(dotted == imp and mod.startswith(pre) for imp, pre in ALLOW):
                    continue  # grandfathered edge — see ALLOW
                violations.append(
                    f"{dotted} (tier {own_tier}) imports {mod} (tier {dep_tier})"
                )
    assert not violations, "Layering violations:\n  " + "\n  ".join(violations)
