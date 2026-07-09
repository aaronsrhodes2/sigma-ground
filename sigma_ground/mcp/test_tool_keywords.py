"""Guard: every tool_keywords entry must map to a real registered tool.

Stale keyword rows (for renamed/removed tools) are inert dead data that confuse
maintenance and imply routing support that doesn't exist. 8 such ghosts were
removed 2026-06-05 (elastic/inelastic_collision_*, ohms_law_resistance,
temperature_*_to_*, thermal_conductivity, speed_of_em_in_medium,
lyman_alpha_wavelength); this test keeps them gone.
"""
from sigma_ground.mcp import tool_keywords as tk
from sigma_ground.mcp import manifest as M


def test_no_ghost_keyword_entries():
    registered = {t["name"] for t in M._PRIMARY_TOOLS}
    ghosts = sorted(set(tk.TOOL_KEYWORDS) - registered)
    assert not ghosts, (
        f"tool_keywords.py has entries for {len(ghosts)} non-existent tools "
        f"(rename/remove drift): {ghosts}"
    )
