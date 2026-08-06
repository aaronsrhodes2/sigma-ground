"""The gallery index covers every scene on disk.

This is the regression guard for a real, shipped defect: four build scripts
each wrote their own index file, the viewer only ever fetched one of them,
and scenes saved by other paths (front_door's _save_bundle, the trajectory
demos, the theater probes) were listed nowhere at all — 89 scene files on
disk, 42 reachable in the UI. The fix (build_scene_index.py) derives the
index from the disk instead of from what a build script remembered to
append; this test is what keeps it that way.
"""
import glob
import json
import os

from sigma_ground.radiance.web import build_scene_index as bsi


def _slugs_on_disk():
    return {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(bsi.DATA, "*.json"))
            if not bsi._is_evidence_scene(os.path.basename(p))     # R4.4 ladder
            } - set(n[:-5] for n in bsi.LEGACY_INDEXES + bsi.NON_SCENES)


def test_every_scene_on_disk_is_indexed():
    """No orphans: the shipped index lists every scene file. If this fails,
    run `python -m sigma_ground.radiance.web.build_scene_index`."""
    with open(bsi.INDEX, encoding="utf-8") as fh:
        indexed = {e["slug"] for e in json.load(fh)}
    missing = _slugs_on_disk() - indexed
    assert not missing, (f"{len(missing)} scene(s) on disk but absent from "
                         f"scenes.json (unreachable in the gallery): "
                         f"{sorted(missing)[:10]}")


def test_index_has_no_dangling_entries():
    """The inverse: nothing is listed that would 404 when clicked."""
    with open(bsi.INDEX, encoding="utf-8") as fh:
        indexed = {e["slug"] for e in json.load(fh)}
    dangling = indexed - _slugs_on_disk()
    assert not dangling, f"index lists missing file(s): {sorted(dangling)}"


def test_rebuild_is_idempotent_and_matches_what_shipped():
    """Rebuilding from disk reproduces the committed index exactly — proving
    the file is derived, not hand-edited (a hand-edit would silently drift)."""
    with open(bsi.INDEX, encoding="utf-8") as fh:
        shipped = json.load(fh)
    assert bsi.build() == shipped, (
        "scenes.json is out of date with the scene files on disk — "
        "run `python -m sigma_ground.radiance.web.build_scene_index`")


def test_every_entry_carries_what_the_gallery_renders():
    """viewer.js's buildLibrary() reads these fields off every card."""
    with open(bsi.INDEX, encoding="utf-8") as fh:
        entries = json.load(fh)
    assert entries, "index is empty"
    for e in entries:
        assert e.get("slug") and e.get("title"), e
        assert e.get("group"), e
        assert e.get("kind") in ("static", "trajectory"), e
        for flag in ("has_bounds", "has_opacity", "has_energy"):
            assert isinstance(e.get(flag), bool), (flag, e["slug"])


def test_trajectory_entries_report_a_frame_count():
    with open(bsi.INDEX, encoding="utf-8") as fh:
        entries = json.load(fh)
    for e in entries:
        if e["kind"] == "trajectory":
            assert e.get("frames", 0) > 0, e["slug"]
