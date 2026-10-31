"""The double slit, photon by photon (materia.voxel_slits): the pattern sigma-ground predicts builds up on the screen."""
import numpy as np
import pytest

from sigma_ground.materia.voxel_slits import SlitScreen


def _profile(s, ticks=60):
    for _ in range(ticks):
        s.step(0.1)
    return s.counts.sum(1)


def _bright_columns(prof):
    return [i for i in range(1, len(prof) - 1)
            if prof[i] > prof[i - 1] and prof[i] >= prof[i + 1] and prof[i] > 0.3 * prof.max()]


def test_fringes_are_spaced_lambda_L_over_d():
    s = SlitScreen()
    peaks = _bright_columns(_profile(s))
    assert len(peaks) >= 5
    assert np.diff(peaks).mean() == pytest.approx(s.fringe_spacing_m() / s.pitch, rel=0.05)


def test_shorter_wavelength_packs_the_fringes_closer():
    red, blue = SlitScreen(), SlitScreen()
    blue.press("colour"), blue.press("colour")                         # 650 → 532 → 405 nm
    assert blue.wavelength_nm == 405
    r, b = np.diff(_bright_columns(_profile(red))).mean(), np.diff(_bright_columns(_profile(blue))).mean()
    assert b / r == pytest.approx(405 / 650, rel=0.1)


def test_observing_which_path_washes_the_fringes_out():
    def contrast(prof):
        c = prof[len(prof) // 2 - 8: len(prof) // 2 + 8].astype(float)
        return (c.max() - c.min()) / (c.max() + c.min())
    seen = SlitScreen()
    seen.press("observe")
    assert contrast(_profile(SlitScreen())) > 0.8
    assert contrast(_profile(seen)) < 0.15


def test_photons_arrive_one_at_a_time_at_first():
    s = SlitScreen()
    s.step(0.1)
    assert s.photons == 1 and s.counts.sum() <= 1
