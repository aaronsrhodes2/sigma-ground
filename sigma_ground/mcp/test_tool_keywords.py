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


# ── coverage ratchets (Wolfram-parity natural-language-parity track) ────────
# These lock in progress, not a target: the floor rises with each PR that
# adds coverage, but a regression (a new tool landing with zero keywords, or
# an old one losing its colloquial entries) fails immediately instead of
# silently degrading discoverability -- the gap this whole track exists to
# close. Bump the floor UP when coverage improves; never bump it down to
# make a test pass.

def test_keyword_coverage_floor():
    registered = {t["name"] for t in M._PRIMARY_TOOLS}
    covered = sum(1 for name in registered if tk.TOOL_KEYWORDS.get(name))
    pct = 100.0 * covered / len(registered)
    floor = 99.0   # 2026-07-14: swept to 226/226 (100%) -- small margin for drift
    assert pct >= floor, (
        f"formal keyword coverage dropped to {pct:.1f}% ({covered}/{len(registered)}), "
        f"below the {floor}% floor -- a tool lost its TOOL_KEYWORDS entry"
    )


def test_colloquial_coverage_floor():
    registered = {t["name"] for t in M._PRIMARY_TOOLS}
    covered = sum(1 for name in registered if tk._COLLOQUIAL.get(name))
    pct = 100.0 * covered / len(registered)
    floor = 99.0   # 2026-07-15: Wave 5 swept to 228/228 (100%) -- small margin for drift
    assert pct >= floor, (
        f"colloquial (layman-phrasing) coverage dropped to {pct:.1f}% "
        f"({covered}/{len(registered)}), below the {floor}% floor"
    )
