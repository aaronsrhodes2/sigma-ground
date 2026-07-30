"""fetch_materials_project.py -- mocked, no-network tests for CI.

Not part of pytest's configured testpaths (tools/ has no test coverage
today, same precedent as fetch_tpsx.py / dataset_minder.py) -- run
explicitly: `pytest tools/test_fetch_materials_project.py -v`.

The real live-API path is exercised separately (not here, to keep CI
network-free): `python tools/fetch_materials_project.py` was run for real
against the live Materials Project API during implementation and its
output committed at sigma_ground/inventory/data/materials_project_cache.json
-- see test_mp_cache_loaded_from_real_fetched_file_has_expected_shape in
sigma_ground/field/interface/test_resolve.py for a check against that real
output.
"""
import json

import fetch_materials_project as fmp


class _FakeNotFound(Exception):
    pass


def test_no_op_when_not_configured(monkeypatch, capsys):
    monkeypatch.setattr(fmp, "is_configured", lambda: False)
    fmp.main()
    out = capsys.readouterr().out
    assert "not configured" in out.lower()


def test_writes_expected_shape_with_mocked_client(tmp_path, monkeypatch):
    monkeypatch.setattr(fmp, "is_configured", lambda: True)

    def _fake_elastic_properties(formula):
        if formula == "Zn":
            raise fmp.MaterialsProjectNotFound("no elasticity data for Zn (mocked)")
        return {
            "name": formula, "material_id": f"mp-fake-{formula}",
            "formula_pretty": formula, "bulk_modulus_vrh": 100.0,
            "shear_modulus_vrh": 40.0, "young_modulus": None,
            "poisson_ratio": 0.3, "universal_anisotropy": 1.0,
            "source": f"Materials Project (DFT-computed elastic tensor, mp-fake-{formula})",
            "provenance_tag": "DFT-COMPUTED", "notes": "mocked",
        }

    monkeypatch.setattr(fmp, "elastic_properties", _fake_elastic_properties)
    out_path = tmp_path / "materials_project_cache.json"
    monkeypatch.setattr(fmp, "_OUT", str(out_path))

    fmp.main()

    assert out_path.is_file()
    doc = json.loads(out_path.read_text(encoding="utf-8"))
    assert "_meta" in doc
    assert doc["_meta"]["generated_by"] == "tools/fetch_materials_project.py"
    # every curated formula except zinc (mocked NotFound) should be present
    expected_present = set(fmp.CURATED_FORMULAS) - {"zinc"}
    assert expected_present <= set(doc.keys())
    assert "zinc" not in doc
    for canon, formula in fmp.CURATED_FORMULAS.items():
        if canon == "zinc":
            continue
        assert doc[canon]["formula_queried"] == formula
        assert doc[canon]["bulk_modulus_vrh"] == 100.0


def test_query_failure_is_reported_not_raised(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fmp, "is_configured", lambda: True)

    def _always_fail(formula):
        raise fmp.MaterialsProjectError(f"simulated network failure for {formula}")

    monkeypatch.setattr(fmp, "elastic_properties", _always_fail)
    out_path = tmp_path / "materials_project_cache.json"
    monkeypatch.setattr(fmp, "_OUT", str(out_path))

    fmp.main()                                       # must not raise

    doc = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(doc.keys()) == {"_meta"}               # nothing else resolved
    out = capsys.readouterr().out
    assert "FAILED" in out
