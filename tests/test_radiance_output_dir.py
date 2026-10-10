"""Radiance's run-time output goes to radiance_data_dir(), never the source tree
during tests (repo-root conftest.py)."""
import os

from sigma_ground.radiance import output_dir


def test_tests_write_to_a_temp_folder_not_the_repo():
    d = output_dir.radiance_data_dir()
    assert os.path.realpath(d) != os.path.realpath(output_dir.DEFAULT)
    assert os.environ[output_dir.ENV_VAR] == d


def test_default_is_the_folder_the_viewer_serves(monkeypatch):
    monkeypatch.delenv(output_dir.ENV_VAR, raising=False)
    assert output_dir.radiance_data_dir() == output_dir.DEFAULT
    assert output_dir.DEFAULT.replace("\\", "/").endswith("sigma_ground/radiance/web/data")


def test_front_door_bundles_land_in_the_output_dir(tmp_path, monkeypatch):
    from sigma_ground.mcp import front_door
    monkeypatch.setenv(output_dir.ENV_VAR, str(tmp_path))
    path = front_door._save_bundle("probe_bundle", {"frames": []})
    assert os.path.dirname(path) == str(tmp_path) and os.path.exists(path)
