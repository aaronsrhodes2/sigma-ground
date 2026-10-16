"""Tests for tools/data_bundle.py: a bundle is exactly what its manifest says, a damaged one is refused, and extraction never silently replaces a different file."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
_spec = importlib.util.spec_from_file_location("data_bundle", REPO / "tools" / "data_bundle.py")
B = importlib.util.module_from_spec(_spec)
sys.modules["data_bundle"] = B
_spec.loader.exec_module(B)


def _make_root(tmp_path):
    root = tmp_path / "local-cache"
    (root / "ds").mkdir(parents=True)
    (root / "seeds").mkdir()
    (root / "ds" / "train.jsonl").write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n', encoding="utf-8")
    (root / "ds" / "rewrites.jsonl").write_text('{"r": 1}\n', encoding="utf-8")
    (root / "seeds" / "seeds.jsonl").write_text('{"s": 1}\n{"s": 2}\n', encoding="utf-8")
    (root / "suite.json").write_text('{"items": []}', encoding="utf-8")
    profile = {"description": "tiny", "train": ["ds/train.jsonl", "suite.json"], "rebuild": ["ds/rewrites.jsonl", "seeds/seeds.jsonl"], "reference": [],
               "rebuild_out": "ds", "rebuild_expect_files": ["train.jsonl", "rewrites.jsonl"], "rebuild_args": ["--out", "{out}"],
               "rebuild_seed_file": ("ds/rewrites.jsonl", "rewrites.jsonl")}
    return root, profile


def test_create_records_hashes_counts_and_the_rebuild_expectation(tmp_path):
    root, profile = _make_root(tmp_path)
    out = tmp_path / "b.zip"
    m = B.create_bundle(root, "tiny", profile, out)
    by = {f["path"]: f for f in m["files"]}
    assert by["ds/train.jsonl"]["lines"] == 3 and by["seeds/seeds.jsonl"]["lines"] == 2 and "lines" not in by["suite.json"]
    assert by["ds/train.jsonl"]["sha256"] == B.sha256_file(root / "ds" / "train.jsonl")
    assert m["rebuild"]["expect"]["train.jsonl"] == by["ds/train.jsonl"]["sha256"]
    assert m["totals"]["train"]["files"] == 2 and m["totals"]["rebuild"]["files"] == 2
    assert B.verify_zip(out) == []


def test_a_changed_byte_is_caught_and_nothing_is_extracted(tmp_path):
    root, profile = _make_root(tmp_path)
    good, bad = tmp_path / "good.zip", tmp_path / "bad.zip"
    B.create_bundle(root, "tiny", profile, good)
    with zipfile.ZipFile(good) as zin, zipfile.ZipFile(bad, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "ds/train.jsonl":
                data = data.replace(b'"a": 2', b'"a": 9')               # same length, different content
            zout.writestr(info.filename, data)
    problems = B.verify_zip(bad)
    assert len(problems) == 1 and "ds/train.jsonl" in problems[0]
    dest = tmp_path / "elsewhere"
    with pytest.raises(SystemExit):
        B.extract_zip(bad, dest)
    assert not dest.exists()


def test_extra_and_missing_members_are_reported(tmp_path):
    root, profile = _make_root(tmp_path)
    src, odd = tmp_path / "src.zip", tmp_path / "odd.zip"
    B.create_bundle(root, "tiny", profile, src)
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(odd, "w") as zout:
        for info in zin.infolist():
            if info.filename != "suite.json":                               # drop one
                zout.writestr(info.filename, zin.read(info.filename))
        zout.writestr("unlisted.txt", b"x")                                  # add one
    text = "\n".join(B.verify_zip(odd))
    assert "unlisted.txt" in text and "suite.json" in text


def test_extract_skips_identical_refuses_different_and_force_replaces(tmp_path):
    root, profile = _make_root(tmp_path)
    out = tmp_path / "b.zip"
    B.create_bundle(root, "tiny", profile, out)
    dest = tmp_path / "new-machine"
    first = B.extract_zip(out, dest)
    assert len(first["wrote"]) == 4 and not first["already_identical"]
    assert (dest / "ds" / "train.jsonl").read_text(encoding="utf-8") == (root / "ds" / "train.jsonl").read_text(encoding="utf-8")
    again = B.extract_zip(out, dest)
    assert not again["wrote"] and len(again["already_identical"]) == 4
    (dest / "ds" / "train.jsonl").write_text("local edits\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        B.extract_zip(out, dest)
    assert (dest / "ds" / "train.jsonl").read_text(encoding="utf-8") == "local edits\n"       # untouched
    B.extract_zip(out, dest, force=True)
    assert (dest / "ds" / "train.jsonl").read_text(encoding="utf-8").startswith('{"a": 1}')


def test_tiers_can_be_extracted_separately(tmp_path):
    root, profile = _make_root(tmp_path)
    out = tmp_path / "b.zip"
    B.create_bundle(root, "tiny", profile, out)
    dest = tmp_path / "train-only"
    B.extract_zip(out, dest, tiers=("train",))
    assert (dest / "ds" / "train.jsonl").exists() and not (dest / "seeds").exists()


def test_the_manifest_carries_a_line_ending_independent_hash_for_json_files(tmp_path):
    root, profile = _make_root(tmp_path)
    m = B.create_bundle(root, "tiny", profile, tmp_path / "b.zip")
    by = {f["path"]: f for f in m["files"]}
    assert by["ds/train.jsonl"]["sha256_lf"] == B.sha256_lf(root / "ds" / "train.jsonl") and "sha256_lf" in by["suite.json"]
    assert m["rebuild"]["expect_lf"]["train.jsonl"] == by["ds/train.jsonl"]["sha256_lf"]


def test_the_same_text_with_other_line_endings_is_reported_as_that_and_a_real_difference_still_fails(tmp_path):
    """Windows writes "\\r\\n" where Linux writes "\\n" (the Linux rehearsal of the M5 run found it): 39,512 identical lines, two different hashes."""
    lf, crlf, other = tmp_path / "lf.jsonl", tmp_path / "crlf.jsonl", tmp_path / "other.jsonl"
    lf.write_bytes(b'{"a": 1}\n{"a": 2}\n')
    crlf.write_bytes(b'{"a": 1}\r\n{"a": 2}\r\n')
    other.write_bytes(b'{"a": 1}\n{"a": 3}\n')
    want, want_lf = B.sha256_file(lf), B.sha256_lf(lf)
    assert B.compare_output(lf, want, want_lf) == {"identical": True, "line_endings_differ": False, "expected": want[:16], "got": want[:16]}
    r = B.compare_output(crlf, want, want_lf)
    assert r["identical"] and r["line_endings_differ"]
    assert not B.compare_output(other, want, want_lf)["identical"]                       # different content: still a failure
    assert not B.compare_output(crlf, want, None)["identical"]                           # an older manifest without the hash stays strict
    assert not B.compare_output(tmp_path / "missing.jsonl", want, want_lf)["identical"]


def test_a_missing_input_stops_the_build_instead_of_making_a_short_bundle(tmp_path):
    root, profile = _make_root(tmp_path)
    (root / "seeds" / "seeds.jsonl").unlink()
    with pytest.raises(SystemExit):
        B.create_bundle(root, "tiny", profile, tmp_path / "b.zip")
    assert not (tmp_path / "b.zip").exists()


@pytest.mark.parametrize("name", sorted(B.PROFILES))
def test_the_real_profiles_are_self_consistent(name):
    p = B.PROFILES[name]
    listed = [f for t in ("train", "rebuild", "reference") for f in p[t]]
    assert len(listed) == len(set(listed)), "a file is listed twice"
    for f in p["rebuild_expect_files"]:
        assert f"{p['rebuild_out']}/{f}" in listed, f"{f} is checked after a rebuild but not carried in the bundle"
    assert p["rebuild_seed_file"][0] in listed
    seeds = [a for a in p["rebuild_args"] if a.startswith("{root}/router_") and a != "{root}/router_dataset"]
    assert all(f"{Path(s).name}/seeds.jsonl" in listed for s in seeds), "a seed folder used by the rebuild is not carried"
    assert B.DEAD_TEACHER in p["rebuild_args"], "a bundle rebuild must never reach a teacher"
