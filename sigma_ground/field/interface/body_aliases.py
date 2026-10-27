"""Solar-system body aliases: formal / nickname / mythological name -> canonical key.

Lives in the field tier so physics modules (body_data, where_when) can resolve
"Luna" or "the Red Planet" without importing the MCP layer;
`sigma_ground.mcp.tools.aliases` re-exports BODY_ALIASES and `_norm` from here.
Keys are normalized at import, so spell them naturally.
"""

from __future__ import annotations


def _norm(s: str) -> str:
    """Lowercase, trim, collapse separators. Returns space-separated form.

    All alias-dict keys go through this on lookup, AND the alias dicts are
    normalized at module import (see `mcp.tools.aliases._normalize_dict()`),
    so keys can be spelled naturally ('k_B', 'h-bar', 'α CMa') without
    worrying about matching the lookup form.
    """
    return (
        s.strip()
         .lower()
         .replace("'", "")
         .replace("’", "")
         .replace("-", " ")
         .replace("_", " ")
    )


# Canonical keys are those of mcp.tools.astronomy._SOLAR_SYSTEM_BODIES.
_BODY_ALIASES: dict[str, str] = {
    # Sun
    "sol": "sun",
    "helios": "sun",
    "the sun": "sun",
    "our star": "sun",
    # Mercury (no common alternate name)
    # Venus (Morning Star, Evening Star, Phosphorus, Hesperus)
    "morning star": "venus",
    "evening star": "venus",
    "phosphorus": "venus",
    "hesperus": "venus",
    # Earth
    "terra": "earth",
    "gaia": "earth",
    "the earth": "earth",
    "our planet": "earth",
    "planet earth": "earth",
    "world": "earth",
    # Moon (Luna)
    "luna": "moon",
    "the moon": "moon",
    "earths moon": "moon",
    # Mars (Red Planet, Ares)
    "ares": "mars",
    "red planet": "mars",
    "the red planet": "mars",
    # Jupiter (Zeus, Jove)
    "zeus": "jupiter",
    "jove": "jupiter",
    # Saturn (Cronus, Kronos)
    "cronus": "saturn",
    "kronos": "saturn",
    # Uranus
    "ouranos": "uranus",
    # Neptune (Poseidon)
    "poseidon": "neptune",
    # Pluto (Hades; dwarf planet)
    "hades": "pluto",
    "dwarf planet pluto": "pluto",
}

BODY_ALIASES: dict[str, str] = {_norm(k): v for k, v in _BODY_ALIASES.items()}
