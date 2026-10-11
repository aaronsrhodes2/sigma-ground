"""Regression tests for rolling_analysis.py (the shoot-out's analysis pass).

Ground truth: the DE440 fixture (each body's parent and state), and
synthetic result files whose statistics are known in closed form.
"""

from __future__ import annotations

import contextlib
import io
import json
import math

import numpy as np
import pandas as pd
import pytest

from sigma_ground.field.interface import rolling_analysis as ra

AU = ra.AU_KM
_DE440 = json.loads(ra._DE440_FILE.read_text())


def _sv(p, v):
    return {"x_km": p[0], "y_km": p[1], "z_km": p[2],
            "vx_km_s": v[0], "vy_km_s": v[1], "vz_km_s": v[2]}


def _snap(jd, bodies):
    return {"epoch": {"jd_tdb": jd},
            "bodies": [{"name": n, "state_vector": _sv(p, v)} for n, (p, v) in bodies.items()]}


# A Sun off the SSB and an Earth on a circular orbit, in a synthetic fixture.
_SUN_P = np.array([1.0e6, -4.0e5, 1.0e5]); _SUN_V = np.array([0.009, -0.012, -0.005])
_E_P = _SUN_P + [AU, 0.0, 0.0];            _E_V = _SUN_V + [0.0, 29.78, 0.0]
_ERR = np.array([1.0, 2.0, 3.0]) * 1e-3     # AU, along (R, T, N) for this Earth


def _row(body, pred_km, jd, err_au=float(np.linalg.norm(_ERR))):
    return {"predictor": "x", "body": body, "window_start_jd": jd, "sample_jd": jd,
            "error_au": err_au, "predicted_km": pred_km}


def _write_results(path, errors, predictors, bodies, starts, taus=(0, 1, 2, 3)):
    """Results file with error(p, b, w, tau) = errors(p, b, w, tau) (None = missing)."""
    samples = []
    for p in predictors:
        for b in bodies:
            for w, s0 in enumerate(starts):
                for tau in taus:
                    samples.append({"predictor": p, "body": b, "window_start_jd": s0,
                                    "sample_jd": s0 + 365.25 * tau,
                                    "error_au": errors(p, b, w, tau), "predicted_km": None})
    meta = {"n_windows": len(starts), "window_start_keys": [f"w{i}" for i in range(len(starts))],
            "prediction_horizon_yr": float(max(taus)), "sample_interval_yr": 1.0,
            "predictors": list(predictors)}
    path.write_text(json.dumps({"metadata": meta, "samples": samples}))


class TestRTN:
    def test_frame_is_built_about_the_fixture_parent(self):
        """Tethys's frame must be built about Saturn (its fixture anchor), not
        the Sun: a 1000 km error along Tethys's Saturn-relative radius must
        read as pure R."""
        key = "j2015"
        jd = _DE440["snapshots"][key]["epoch"]["jd_tdb"]
        st = {b["name"]: b["state_vector"] for b in _DE440["snapshots"][key]["bodies"]}
        pos = lambda n: np.array([st[n]["x_km"], st[n]["y_km"], st[n]["z_km"]])
        r_hat = (pos("Tethys") - pos("Saturn")) / np.linalg.norm(pos("Tethys") - pos("Saturn"))
        pred = pos("Tethys") - pos("Sun") + 1000.0 * r_hat
        out = ra.add_rtn_components(pd.DataFrame([_row("Tethys", list(pred), jd, 1000.0 / AU)]), _DE440)
        dR, dT, dN = out[["dR_au", "dT_au", "dN_au"]].to_numpy()[0] * AU
        assert dR == pytest.approx(1000.0, abs=1e-3)
        assert abs(dT) < 1e-3 and abs(dN) < 1e-3

    def test_parents_follow_the_fixture_anchor(self):
        parents = ra._parents_from_fixture(_DE440)
        for b in _DE440["snapshots"]["j2015"]["bodies"]:
            if b["name"] != "Sun":
                assert parents[b["name"]] == b["gravitational_anchor"], b["name"]

    def test_truth_lookup_keys_are_exact(self):
        """Snapshots at JD 2451545.5 and 2451546.5 both round() to 2451546;
        the error at the first must still be decomposed with its own truth."""
        base = {"Sun": (_SUN_P, _SUN_V), "Earth": (_E_P, _E_V)}
        moved = {"Sun": (_SUN_P, _SUN_V), "Earth": (_E_P + [0.0, 2.573e6, 0.0], _E_V)}
        de = {"snapshots": {"a": _snap(2451545.5, base), "b": _snap(2451546.5, moved)}}
        assert len(ra._build_truth_state_lookup(de)) == 4
        pred = (_E_P - _SUN_P) + _ERR * AU
        out = ra.add_rtn_components(pd.DataFrame([_row("Earth", list(pred), 2451545.5)]), de)
        np.testing.assert_allclose(out[["dR_au", "dT_au", "dN_au"]].to_numpy()[0], _ERR,
                                   rtol=1e-9)

    @pytest.mark.parametrize("pred", [float("nan"), None, [None, None, None],
                                      [float("nan")] * 3])
    def test_missing_prediction_gives_a_nan_row(self, pred):
        de = {"snapshots": {"a": _snap(2451545.0, {"Sun": (_SUN_P, _SUN_V),
                                                     "Earth": (_E_P, _E_V)})}}
        df = pd.DataFrame([_row("Earth", [0.0, 0.0, 0.0], 2451545.0)])
        df["predicted_km"] = [pred]
        out = ra.add_rtn_components(df, de)
        assert out[["dR_au", "dT_au", "dN_au"]].isna().all(axis=None)


class TestSummaries:
    def test_growth_std_is_undefined_for_one_window(self, tmp_path):
        f = tmp_path / "one.json"
        _write_results(f, lambda p, b, w, t: 1e-6 * t, ["p"], ["Earth"], [2451545.0])
        _, df = ra.load_results(f)
        s = ra.per_predictor_body_summary(df)
        assert math.isnan(s["growth_std_au_yr"].iloc[0])
        assert s["growth_rate_au_yr"].iloc[0] == pytest.approx(1e-6)

    def test_sub_delta_uses_adjacent_windows_only(self, tmp_path):
        """error = d * w * tau; window 1 missing. Only the (2, 3) pair is
        adjacent: RMS over tau of d * tau. Pairing 0 with 2 doubles it."""
        d = 1e-7
        taus = (0, 1, 2, 3)
        f = tmp_path / "gap.json"
        _write_results(f, lambda p, b, w, t: None if w == 1 else d * w * t,
                       ["p"], ["Mars"], [2451545.0, 2451910.25, 2452275.5, 2452640.75], taus)
        _, df = ra.load_results(f)
        s = ra.per_predictor_body_summary(df)
        want = d * math.sqrt(np.mean(np.array(taus) ** 2))
        assert s["sub_delta_rms_au"].iloc[0] == pytest.approx(want, rel=1e-12)


class TestFingerprintDiff:
    def test_reports_removed_predictors_and_sorts_by_absolute_change(self, tmp_path):
        starts = [2451545.0, 2451910.25, 2452275.5]
        bodies = ["Earth", "Jupiter", "Mars"]
        level = {"Earth": 1e-4, "Jupiter": 1e-8, "Mars": 1e-4}
        scale = {("standard", "Earth"): 0.5,       # big absolute gain, ratio 0.5
                 ("standard", "Jupiter"): 0.1}     # tiny absolute gain, ratio 0.1
        base, new = tmp_path / "base.json", tmp_path / "new.json"
        _write_results(base, lambda p, b, w, t: level[b] * t,
                       ["pure_newton", "standard"], bodies, starts)
        _write_results(new, lambda p, b, w, t: level[b] * t * scale.get((p, b), 1.0),
                       ["standard"], bodies, starts)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ra.fingerprint_diff(base, new)
        text = buf.getvalue()
        assert "pure_newton" in text.split("REMOVED")[1]
        imp = text.split("IMPROVEMENTS")[1].split("REGRESSIONS")[0]
        assert imp.index("Earth") < imp.index("Jupiter")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ra.fingerprint_diff(base, new, focus_predictors=["pure_newton"])
        assert "pure_newton" in buf.getvalue()
