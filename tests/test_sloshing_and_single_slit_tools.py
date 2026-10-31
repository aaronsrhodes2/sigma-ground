"""sloshing_frequency and single_slit_intensity, checked against things that do not use the library's own routines.

* Sloshing: the full finite-depth dispersion relation written out here with scipy's g and handbook water values; the deep-water limit sqrt(g k), the shallow-water limit k sqrt(g h) and the
  1/L scaling that follow from it.
* Single slit: the sinc^2 envelope from numpy's own sinc, the exact zeros at sin(theta) = n lambda / a, and the well-known 4.7 % first secondary maximum (a constant of the sinc^2 curve)."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import constants as sc

from sigma_ground.mcp.tools import fluid_surface as F
from sigma_ground.mcp.tools import optics as O

SIGMA_20C, RHO_20C = 0.0728, 998.2                       # handbook values for water at 20 C (N/m, kg/m^3)


def _omega(L, h, n, sigma=SIGMA_20C, rho=RHO_20C):
    k = n * math.pi / L
    return math.sqrt((sc.g * k + sigma * k ** 3 / rho) * math.tanh(k * h))


class TestSloshing:
    @pytest.mark.parametrize("L,h,n", [(0.3, 0.1, 1), (0.3, 0.1, 2), (1.0, 0.25, 1), (0.05, 0.02, 1), (4.0, 0.5, 3)])
    def test_the_dispersion_relation_written_out_here(self, L, h, n):
        r = F.sloshing_frequency(L, h, n).to_dict()
        assert r["value"] == pytest.approx(_omega(L, h, n) / (2 * math.pi), rel=0.01)                  # handbook water against the seeds: within 1 %
        assert r["units"] == "Hz"

    def test_deep_water_the_depth_does_not_matter_and_the_frequency_is_sqrt_g_k(self):
        L = 20.0                                                                                     # long tank: capillarity negligible
        deep = F.sloshing_frequency(L, 50.0).to_dict()["value"]
        deeper = F.sloshing_frequency(L, 500.0).to_dict()["value"]
        assert deep == pytest.approx(deeper, rel=1e-5)                                              # tanh(kh) = 1 to 3e-7 at kh = 7.9
        assert deep * 2 * math.pi == pytest.approx(math.sqrt(sc.g * math.pi / L), rel=1e-3)

    def test_shallow_water_the_wave_runs_at_sqrt_g_h(self):
        L, h = 10.0, 0.01
        f = F.sloshing_frequency(L, h).to_dict()["value"]
        assert f * 2 * math.pi == pytest.approx((math.pi / L) * math.sqrt(sc.g * h), rel=2e-3)

    def test_a_longer_tank_sloshes_more_slowly_and_a_higher_mode_faster(self):
        assert F.sloshing_frequency(0.6, 0.1).to_dict()["value"] < F.sloshing_frequency(0.3, 0.1).to_dict()["value"]
        assert F.sloshing_frequency(0.3, 0.1, 2).to_dict()["value"] > F.sloshing_frequency(0.3, 0.1, 1).to_dict()["value"]

    def test_the_sentence_carries_the_regime_and_the_other_forms(self):
        n = F.sloshing_frequency(0.3, 0.1).to_dict()["notes"]
        assert "rad/s" in n and "period" in n and "intermediate depth" in n and "surface tension" in n
        assert "deep water" in F.sloshing_frequency(20.0, 50.0).to_dict()["notes"]

    @pytest.mark.parametrize("args", [(0, 0.1), (0.3, 0), (-1, 0.1), (0.3, 0.1, 0), (0.3, 0.1, 1.5), (0.3, 0.1, 1, 500.0)])
    def test_what_it_cannot_do_is_declined_with_a_reason(self, args):
        r = F.sloshing_frequency(*args).to_dict()
        assert r["value"] is None and r["notes"]


class TestSingleSlit:
    L, LAM, A = 1.0, 550e-9, 5e-5

    def _envelope(self, y):
        r = math.hypot(self.L, y)
        return float(np.sinc(self.A * (y / r) / self.LAM) ** 2) * (self.L / r)                       # numpy's sinc(x) = sin(pi x)/(pi x)

    @pytest.mark.parametrize("y", [0.0, 0.001, 0.005, 0.0123, -0.005, 0.03])
    def test_it_equals_the_sinc_squared_envelope_from_numpys_own_sinc(self, y):
        got = O.single_slit_intensity(self.LAM, self.A, self.L, y).to_dict()["value"]
        assert got == pytest.approx(self._envelope(y), rel=1e-9, abs=1e-15)

    def test_the_centre_is_one_and_the_pattern_is_symmetric(self):
        assert O.single_slit_intensity(self.LAM, self.A, self.L).to_dict()["value"] == pytest.approx(1.0, rel=1e-12)
        assert O.single_slit_intensity(self.LAM, self.A, self.L, 0.007).to_dict()["value"] == pytest.approx(O.single_slit_intensity(self.LAM, self.A, self.L, -0.007).to_dict()["value"], rel=1e-12)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_the_zeros_are_at_sin_theta_equal_n_lambda_over_a(self, n):
        sin_t = n * self.LAM / self.A
        y = self.L * sin_t / math.sqrt(1 - sin_t ** 2)                                               # y/r = sin theta
        assert O.single_slit_intensity(self.LAM, self.A, self.L, y).to_dict()["value"] < 1e-12

    def test_the_first_secondary_maximum_is_about_4_7_percent_of_the_centre(self):
        sin_t = 1.4303 * self.LAM / self.A                                                           # x = 1.4303 pi, where d(sinc^2)/dx = 0
        y = self.L * sin_t / math.sqrt(1 - sin_t ** 2)
        assert O.single_slit_intensity(self.LAM, self.A, self.L, y).to_dict()["value"] == pytest.approx(0.04719, abs=2e-4)

    def test_it_says_when_the_screen_is_too_close_for_the_far_field(self):
        far = O.single_slit_intensity(self.LAM, self.A, 1.0, 0.005).to_dict()["notes"]
        near = O.single_slit_intensity(self.LAM, 5e-3, 0.1, 0.005).to_dict()["notes"]                 # a 5 mm slit 10 cm away: Fresnel number ~ 450
        assert "the Fraunhofer form holds" in far and "NOT far field" in near

    @pytest.mark.parametrize("args", [(0, 5e-5, 1.0), (550e-9, 0, 1.0), (550e-9, 5e-5, -1.0)])
    def test_nonpositive_dimensions_are_declined(self, args):
        r = O.single_slit_intensity(*args).to_dict()
        assert r["value"] is None and r["notes"]
