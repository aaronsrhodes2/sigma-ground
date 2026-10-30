"""The layer-ablation predictors differ from `restated_zon` in ONE layer each, and adding them changes nothing that was stored (misc/CANDIDATE_LEDGER.md C23)."""
from __future__ import annotations

import dataclasses
import unittest

from sigma_ground.field.interface import rolling_shootout as rs

_FIELDS = ("toggles", "perturber_sets", "lunar_c22", "zonal_overrides", "dt_days", "integrator", "use_kepler")


class TestLayerAblations(unittest.TestCase):
    def setUp(self):
        self.base = [p for p in rs.PREDICTORS if p.name == "restated_zon"][0]
        self.arms = {k: [p for p in rs.PREDICTORS if p.name == f"restated_zon_no_{k}"][0] for k in rs.LAYER_ABLATIONS}

    def test_every_layer_has_an_arm(self):
        self.assertEqual(set(self.arms), {"eih", "zonals", "tides", "srp", "asteroids", "ring", "kbo", "c22"})

    def test_each_arm_changes_exactly_the_layer_it_names(self):
        t = self.base.toggles
        for k, p in self.arms.items():
            changed = {f for f in _FIELDS if getattr(p, f) != getattr(self.base, f)}
            self.assertEqual(changed, {"lunar_c22"} if k == "c22" else ({"perturber_sets"} if k in ("asteroids", "ring", "kbo") else {"toggles"}), k)
        def off(arm, *names):
            d = dataclasses.asdict(self.arms[arm].toggles)
            return {n for n, v in d.items() if v != dataclasses.asdict(t)[n]}
        self.assertEqual(off("eih", "eih_cross"), {"eih_cross"})
        self.assertEqual(off("zonals"), {"j2_zonal", "j3_zonal", "j4_zonal"})
        self.assertEqual(off("tides"), {"tidal_force"})
        self.assertEqual(off("srp"), {"srp"})

    def test_the_perturber_arms_remove_one_set_and_keep_the_others(self):
        self.assertEqual(set(self.base.perturber_sets), {"asteroids", "ring", "kbo"})
        self.assertEqual(set(self.arms["asteroids"].perturber_sets), {"ring", "kbo"})
        self.assertEqual(set(self.arms["ring"].perturber_sets), {"asteroids", "kbo"})
        self.assertEqual(set(self.arms["kbo"].perturber_sets), {"asteroids", "ring"})
        self.assertEqual(self.arms["c22"].lunar_c22, 0.0)
        self.assertNotEqual(self.base.lunar_c22, 0.0)

    def test_the_stored_labels_and_the_forecast_fingerprint_are_untouched(self):
        names = [p.name for p in rs.PREDICTORS]
        self.assertEqual(len(names), len(set(names)))
        for old in ("restated_zon", "restated_hdr", "restated_ast_ring_kbo_c22", "de440_stack"):
            self.assertIn(old, names)
        from sigma_ground.field.interface import reduced_model as rm
        import json
        from pathlib import Path
        manifest = json.loads((Path(rs.__file__).parent / "fixtures" / "forecast_manifest.json").read_text(encoding="utf-8"))
        recorded = manifest["tables"][-1].get("model_fingerprint")
        if recorded:
            self.assertEqual(rm.model_fingerprint("restated_zon"), recorded)             # adding predictors must not make the sealed forecast read as stale


if __name__ == "__main__":
    unittest.main()
