"""Write src/main/resources/skippycraft_sky.json: sigma-ground's clear-sky table (field.interface.sky), which the mod only
interpolates to colour what a clear surface reflects."""
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[2]))                     # the sigma-ground checkout
from sigma_ground.field.interface import sky  # noqa: E402

rec = sky.sky_record()
out = here / "src" / "main" / "resources" / "skippycraft_sky.json"
# the mod reads the display (sRGB-encoded) tables; the linear ones stay in the library
wire = {
    "schema": rec["schema"],
    "sun_elevation_deg": rec["sun_elevation_deg"],
    "view_elevation_deg": rec["view_elevation_deg"],
    "relative_azimuth_deg": rec["relative_azimuth_deg"],
    "sky": rec["sky"]["rgb_display"],
    "ground": rec["ground"]["rgb_display"],
    "white_card": rec["white_card"],
    "gaps": rec["gaps"],
}
out.write_text(json.dumps(wire, separators=(",", ":")), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
