"""Repo-wide pytest setup: tests don't write into the source tree.

Radiance's run-time output (viewer bundles from the front door, bell-strike
audio) goes to sigma_ground.radiance.output_dir.radiance_data_dir(). For a test
session that is a temp folder, set here through SIGMA_GROUND_RADIANCE_DATA_DIR
(subprocesses such as a spawned MCP server inherit it). Before this, seven test
files rewrote tracked demo bundles in radiance/web/data/, and test_push_route's
rewrite then failed test_scene_index whenever the two ran in that order.

The sentinel below keeps it that way: if anything in a test session writes into
radiance/web/data/ again, the session fails and names the files.
"""
import os
import shutil
import tempfile
from pathlib import Path

import pytest

_REPO_DATA = Path(__file__).parent / "sigma_ground" / "radiance" / "web" / "data"
_ENV = "SIGMA_GROUND_RADIANCE_DATA_DIR"


def _snapshot():
    if not _REPO_DATA.is_dir():
        return {}
    return {p: p.stat().st_mtime_ns for p in _REPO_DATA.rglob("*") if p.is_file()}


def pytest_configure(config):
    if not os.environ.get(_ENV):
        config._sg_radiance_tmp = tempfile.mkdtemp(prefix="sg_radiance_data_")
        os.environ[_ENV] = config._sg_radiance_tmp
    config._sg_radiance_snapshot = _snapshot()


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    before = getattr(config, "_sg_radiance_snapshot", None)
    if before is not None:
        after = _snapshot()
        touched = sorted(str(p.relative_to(_REPO_DATA)) for p in after
                         if before.get(p) != after[p])
        if touched:
            reporter = config.pluginmanager.get_plugin("terminalreporter")
            if reporter is not None:
                reporter.ensure_newline()
                reporter.write_line(
                    f"FAILED: tests wrote into the source tree ({_REPO_DATA}): "
                    f"{touched}. Write through radiance_data_dir() instead.", red=True)
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
    tmp = getattr(config, "_sg_radiance_tmp", None)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
        os.environ.pop(_ENV, None)
