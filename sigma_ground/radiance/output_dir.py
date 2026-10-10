"""Where Radiance writes its run-time output: viewer bundles and audio.

Default: radiance/web/data/, the folder the viewer serves. The environment
variable SIGMA_GROUND_RADIANCE_DATA_DIR overrides it -- the repo-root
conftest.py points it at a temp folder for every test session, so tests stop
rewriting the tracked demo bundles (seven test files did, and one rewrite then
failed test_scene_index depending on test order).
"""
import os

ENV_VAR = "SIGMA_GROUND_RADIANCE_DATA_DIR"
DEFAULT = os.path.abspath(os.path.join(os.path.dirname(__file__), "web", "data"))


def radiance_data_dir() -> str:
    """The output folder, created if missing."""
    path = os.environ.get(ENV_VAR) or DEFAULT
    os.makedirs(path, exist_ok=True)
    return path
