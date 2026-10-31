"""Heights of well-known landmarks and mountains, each with its source, for "how many X tall is Y" questions.

Everest in bananas (adv_units_001, 2026-10-30) needs Everest's height; no tool here had one. Heights are the surveyed or official
figures; a mountain's is its summit above mean sea level, a building's its architectural or tip height as stated.
"""

from __future__ import annotations

from sigma_ground.mcp.provenance import ToolResult

LANDMARKS: dict[str, tuple[float, str, str]] = {          # key -> (height_m, note, source)
    "mount everest": (8848.86, "summit above mean sea level", "Nepal-China joint survey, December 2020"),
    "k2": (8611.0, "summit above mean sea level", "Survey of Pakistan / commonly cited"),
    "kangchenjunga": (8586.0, "summit above mean sea level", "commonly cited survey figure"),
    "mont blanc": (4808.7, "summit above mean sea level", "IGN 2017 survey (ice cap varies by about a metre)"),
    "kilimanjaro": (5895.0, "Uhuru Peak above mean sea level", "Tanzania National Parks / commonly cited"),
    "denali": (6190.0, "summit above mean sea level", "USGS 2015 survey"),
    "mauna kea": (4207.3, "summit above mean sea level (about 10,200 m from its base on the sea floor)", "USGS"),
    "burj khalifa": (828.0, "architectural height to the tip", "CTBUH"),
    "eiffel tower": (330.0, "to the tip of the antenna", "SETE (tour-eiffel.fr), 2022"),
    "empire state building": (443.2, "to the tip of the antenna (381 m to the roof)", "CTBUH"),
    "statue of liberty": (93.0, "ground to torch", "US National Park Service"),
    "big ben": (96.0, "Elizabeth Tower to the tip", "UK Parliament"),
    "great pyramid of giza": (138.5, "current height (146.6 m when built)", "commonly cited survey figures"),
    "mariana trench": (10935.0, "depth of the Challenger Deep below sea level", "Gardner et al. 2014 / 2021 pressure-sensor survey"),
}
_ALIASES = {"everest": "mount everest", "mt everest": "mount everest", "mt. everest": "mount everest", "sagarmatha": "mount everest",
            "chomolungma": "mount everest", "mount kilimanjaro": "kilimanjaro", "mt kilimanjaro": "kilimanjaro", "mount denali": "denali",
            "mount mckinley": "denali", "mckinley": "denali", "mount mauna kea": "mauna kea", "the eiffel tower": "eiffel tower",
            "the burj khalifa": "burj khalifa", "the empire state building": "empire state building", "the statue of liberty": "statue of liberty",
            "elizabeth tower": "big ben", "great pyramid": "great pyramid of giza", "pyramid of giza": "great pyramid of giza",
            "challenger deep": "mariana trench", "the mariana trench": "mariana trench"}


def _key(name: str) -> str | None:
    k = " ".join(name.strip().lower().replace("_", " ").split())
    k = _ALIASES.get(k, k)
    return k if k in LANDMARKS else None


def landmark_height(name: str) -> ToolResult:
    """Height of a named landmark or mountain in metres (Mount Everest 8848.86 m, Burj Khalifa 828 m, Eiffel Tower 330 m, ...), with its source."""
    k = _key(name)
    if k is None:
        return ToolResult(value=None, source="sigma-ground (landmark heights)", inputs={"name": name},
                          notes=f"'{name}' is not in the landmark table. Known: {', '.join(sorted(LANDMARKS))}.")
    h, note, src = LANDMARKS[k]
    return ToolResult(value=h, units="m", source=src, provenance_tag="VERIFIED", inputs={"name": name, "landmark": k},
                      notes=f"{k.title()}: {note}.")
