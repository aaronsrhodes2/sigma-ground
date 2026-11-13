"""The sky table the SkippyCraft mod ships (mod/gen_sky.py) is sigma-ground's, current: regenerate it if this fails."""
import json
from pathlib import Path

from sigma_ground.field.interface import sky

RES = Path(__file__).resolve().parents[1] / "stage" / "skippycraft" / "mod" / "src" / "main" / "resources" / "skippycraft_sky.json"


def test_shipped_sky_table_matches_the_library():
    shipped = json.loads(RES.read_text(encoding="utf-8"))
    rec = sky.sky_record()
    assert shipped["sky"] == rec["sky"]["rgb_display"]
    assert shipped["ground"] == rec["ground"]["rgb_display"]
    assert shipped["sun_elevation_deg"] == rec["sun_elevation_deg"]
    assert shipped["view_elevation_deg"] == rec["view_elevation_deg"]
    assert shipped["relative_azimuth_deg"] == rec["relative_azimuth_deg"]
    for key in ("material", "sun_elevation_deg", "view_elevation_deg", "relative_azimuth_deg", "wind_m_s", "rgb_display", "alpha"):
        assert shipped["water"][key] == rec["water"][key], key
