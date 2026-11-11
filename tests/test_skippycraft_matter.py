"""SkippyCraft's door onto the matter record (stage/skippycraft/matter.py): it builds nothing itself."""
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage" / "skippycraft"))

import matter  # noqa: E402
from sigma_ground.field.interface import matter_record  # noqa: E402


def test_the_stage_door_is_sigma_grounds_record():
    assert matter.record is matter_record.matter_record and matter.to_wire is matter_record.to_wire and matter.SCHEMA == matter_record.SCHEMA
    r = matter.record("gold", None, 0.0125)
    assert r["material"] == "gold" and r["tables"] is None


@pytest.mark.skipif(shutil.which("docker") is None, reason="needs Docker to build the mod")
def test_the_mod_sources_exist_and_name_the_frames_the_service_sends():
    root = Path(__file__).resolve().parents[1] / "stage" / "skippycraft" / "mod" / "src"
    bridge = (root / "main/java/local/skippy/craft/SkippyBridge.java").read_text(encoding="utf-8")
    for magic in ("0x534B4346", "0x534B434D", "0x534B4350"):                # SKCF, SKCM, SKCP
        assert magic in bridge
