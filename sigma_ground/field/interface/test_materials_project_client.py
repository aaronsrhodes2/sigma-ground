"""
Tests for materials_project_client.py -- offline/mocked only.

No MATERIALS_PROJECT_API_KEY is configured in this dev tree as of writing
(see .env.reference.md), and per PLATINUM_RULES.md sec8 / this task's own
scope, sigma-ground never registers an account or generates a key on the
Captain's behalf. So this suite verifies:

  1. The "not configured" error path fires for real (no key present --
     this is an actual, not simulated, environment condition).
  2. The "mp-api not installed" error path fires for real (mp-api is not
     installed in this environment either).
  3. Response-parsing logic against hand-constructed payloads shaped like
     the real API's documented response format (both dict-shaped, as a
     stand-in for a JSON payload, and attribute-shaped, as a stand-in for
     the pydantic documents mp-api actually returns) -- using mocks so no
     network call happens.
  4. `_load_dev_root_env` finds and loads a temp .env file correctly.

None of these tests make a live network call.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

from sigma_ground.field.interface import materials_project_client as mpc


class _AttrDoc:
    """Minimal stand-in for an mp-api pydantic document: attribute access,
    not dict access. Mirrors the real ElasticityDoc/DielectricDoc shape
    closely enough to exercise the parsing helpers."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestNotConfigured(unittest.TestCase):
    """MATERIALS_PROJECT_API_KEY is genuinely absent from this environment
    (confirmed: not in D:\\Aaron\\development\\.env, not in os.environ) --
    this exercises the real refusal path, not a simulated one."""

    def setUp(self):
        self._saved = os.environ.pop(mpc.MATERIALS_PROJECT_API_KEY_ENV, None)

    def tearDown(self):
        if self._saved is not None:
            os.environ[mpc.MATERIALS_PROJECT_API_KEY_ENV] = self._saved

    def test_is_configured_false(self):
        self.assertFalse(mpc.is_configured())

    def test_elastic_properties_raises_not_configured(self):
        with self.assertRaises(mpc.MaterialsProjectNotConfigured) as ctx:
            mpc.elastic_properties("Zn")
        msg = str(ctx.exception)
        self.assertIn("MATERIALS_PROJECT_API_KEY", msg)
        self.assertIn("next-gen.materialsproject.org", msg)
        self.assertIn(".env", msg)

    def test_dielectric_properties_raises_not_configured(self):
        with self.assertRaises(mpc.MaterialsProjectNotConfigured):
            mpc.dielectric_properties("Si")

    def test_empty_query_raises_value_error_before_key_check(self):
        with self.assertRaises(ValueError):
            mpc.elastic_properties("   ")


class TestMpApiNotInstalled(unittest.TestCase):
    """mp-api is genuinely not installed in this environment -- confirmed
    via `python -c "import mp_api"` failing with ModuleNotFoundError."""

    def setUp(self):
        os.environ[mpc.MATERIALS_PROJECT_API_KEY_ENV] = "fake-key-for-test"

    def tearDown(self):
        os.environ.pop(mpc.MATERIALS_PROJECT_API_KEY_ENV, None)

    def test_import_mprester_raises_clear_error(self):
        with self.assertRaises(mpc.MaterialsProjectError) as ctx:
            mpc._import_mprester()
        msg = str(ctx.exception)
        self.assertIn("mp-api", msg)
        self.assertIn("pip install", msg)

    def test_elastic_properties_raises_clear_error_without_mp_api(self):
        with self.assertRaises(mpc.MaterialsProjectError) as ctx:
            mpc.elastic_properties("Zn")
        self.assertIn("mp-api", str(ctx.exception))


class TestElasticityParsing(unittest.TestCase):
    """Parse a hand-constructed payload shaped like the documented
    ElasticityDoc response, via a mocked MPRester -- no network call."""

    def _fake_mprester_module(self, docs):
        """Build a fake `mp_api.client` module exposing a MPRester whose
        `.materials.elasticity.search(...)` returns `docs`."""
        fake_mpr_instance = mock.MagicMock()
        fake_mpr_instance.__enter__ = mock.Mock(return_value=fake_mpr_instance)
        fake_mpr_instance.__exit__ = mock.Mock(return_value=False)
        fake_mpr_instance.materials.elasticity.search = mock.Mock(return_value=docs)
        fake_mpr_instance.materials.dielectric.search = mock.Mock(return_value=docs)

        fake_mprester_cls = mock.Mock(return_value=fake_mpr_instance)

        fake_client_module = types.ModuleType("mp_api.client")
        fake_client_module.MPRester = fake_mprester_cls

        fake_mp_api_module = types.ModuleType("mp_api")
        fake_mp_api_module.client = fake_client_module
        return fake_mp_api_module, fake_client_module, fake_mprester_cls

    def setUp(self):
        os.environ[mpc.MATERIALS_PROJECT_API_KEY_ENV] = "fake-key-for-test"

    def tearDown(self):
        os.environ.pop(mpc.MATERIALS_PROJECT_API_KEY_ENV, None)
        sys.modules.pop("mp_api", None)
        sys.modules.pop("mp_api.client", None)

    def test_parses_attribute_style_doc(self):
        doc = _AttrDoc(
            material_id="mp-79",
            formula_pretty="Zn",
            bulk_modulus=_AttrDoc(voigt=70.2, reuss=65.1, vrh=67.6),
            shear_modulus=_AttrDoc(voigt=45.0, reuss=38.2, vrh=41.6),
            young_modulus=108.3,
            homogeneous_poisson=0.249,
            universal_anisotropy=0.9,
        )
        fake_mp_api, fake_client, _ = self._fake_mprester_module([doc])
        with mock.patch.dict(sys.modules, {"mp_api": fake_mp_api, "mp_api.client": fake_client}):
            result = mpc.elastic_properties("Zn")

        self.assertEqual(result["material_id"], "mp-79")
        self.assertEqual(result["formula_pretty"], "Zn")
        self.assertEqual(result["bulk_modulus_vrh"], 67.6)
        self.assertEqual(result["shear_modulus_vrh"], 41.6)
        self.assertEqual(result["young_modulus"], 108.3)
        self.assertEqual(result["poisson_ratio"], 0.249)
        self.assertEqual(result["universal_anisotropy"], 0.9)
        self.assertEqual(result["provenance_tag"], "DFT-COMPUTED")
        self.assertIn("mp-79", result["source"])
        self.assertIn("fatigue", result["notes"].lower())

    def test_parses_dict_style_doc(self):
        # Stand-in for a raw JSON payload shaped like the documented
        # ElasticityDoc response.
        doc = {
            "material_id": "mp-30",
            "formula_pretty": "Cu",
            "bulk_modulus": {"voigt": 142.0, "reuss": 138.0, "vrh": 140.0},
            "shear_modulus": {"voigt": 50.0, "reuss": 46.0, "vrh": 48.0},
            "young_modulus": 130.0,
            "homogeneous_poisson": 0.34,
            "universal_anisotropy": 1.2,
        }
        fake_mp_api, fake_client, _ = self._fake_mprester_module([doc])
        with mock.patch.dict(sys.modules, {"mp_api": fake_mp_api, "mp_api.client": fake_client}):
            result = mpc.elastic_properties("mp-30")

        self.assertEqual(result["material_id"], "mp-30")
        self.assertEqual(result["bulk_modulus_vrh"], 140.0)
        self.assertEqual(result["shear_modulus_vrh"], 48.0)
        self.assertEqual(result["poisson_ratio"], 0.34)

    def test_not_found_raises_with_honest_message(self):
        fake_mp_api, fake_client, _ = self._fake_mprester_module([])
        with mock.patch.dict(sys.modules, {"mp_api": fake_mp_api, "mp_api.client": fake_client}):
            with self.assertRaises(mpc.MaterialsProjectNotFound) as ctx:
                mpc.elastic_properties("unobtainium")
        msg = str(ctx.exception)
        self.assertIn("unobtainium", msg)
        self.assertIn("formula", msg)

    def test_mp_id_routes_to_material_ids_search(self):
        doc = _AttrDoc(material_id="mp-79", formula_pretty="Zn")
        fake_mp_api, fake_client, mprester_cls = self._fake_mprester_module([doc])
        with mock.patch.dict(sys.modules, {"mp_api": fake_mp_api, "mp_api.client": fake_client}):
            mpc.elastic_properties("mp-79")
        instance = mprester_cls.return_value
        instance.materials.elasticity.search.assert_called_once_with(material_ids=["mp-79"])

    def test_formula_routes_to_formula_search(self):
        doc = _AttrDoc(material_id="mp-79", formula_pretty="Zn")
        fake_mp_api, fake_client, mprester_cls = self._fake_mprester_module([doc])
        with mock.patch.dict(sys.modules, {"mp_api": fake_mp_api, "mp_api.client": fake_client}):
            mpc.elastic_properties("Zn")
        instance = mprester_cls.return_value
        instance.materials.elasticity.search.assert_called_once_with(formula="Zn")

    def test_api_exception_wrapped_in_materials_project_error(self):
        fake_mpr_instance = mock.MagicMock()
        fake_mpr_instance.__enter__ = mock.Mock(return_value=fake_mpr_instance)
        fake_mpr_instance.__exit__ = mock.Mock(return_value=False)
        fake_mpr_instance.materials.elasticity.search = mock.Mock(
            side_effect=RuntimeError("simulated network failure")
        )
        fake_mprester_cls = mock.Mock(return_value=fake_mpr_instance)
        fake_client_module = types.ModuleType("mp_api.client")
        fake_client_module.MPRester = fake_mprester_cls
        fake_mp_api_module = types.ModuleType("mp_api")
        fake_mp_api_module.client = fake_client_module

        with mock.patch.dict(sys.modules, {"mp_api": fake_mp_api_module, "mp_api.client": fake_client_module}):
            with self.assertRaises(mpc.MaterialsProjectError) as ctx:
                mpc.elastic_properties("Zn")
        self.assertIn("simulated network failure", str(ctx.exception))


class TestDielectricParsing(unittest.TestCase):
    """Parse a hand-constructed payload shaped like the documented
    DielectricDoc response, via a mocked MPRester -- no network call."""

    def _fake_mprester_module(self, docs):
        fake_mpr_instance = mock.MagicMock()
        fake_mpr_instance.__enter__ = mock.Mock(return_value=fake_mpr_instance)
        fake_mpr_instance.__exit__ = mock.Mock(return_value=False)
        fake_mpr_instance.materials.dielectric.search = mock.Mock(return_value=docs)
        fake_mprester_cls = mock.Mock(return_value=fake_mpr_instance)
        fake_client_module = types.ModuleType("mp_api.client")
        fake_client_module.MPRester = fake_mprester_cls
        fake_mp_api_module = types.ModuleType("mp_api")
        fake_mp_api_module.client = fake_client_module
        return fake_mp_api_module, fake_client_module, fake_mprester_cls

    def setUp(self):
        os.environ[mpc.MATERIALS_PROJECT_API_KEY_ENV] = "fake-key-for-test"

    def tearDown(self):
        os.environ.pop(mpc.MATERIALS_PROJECT_API_KEY_ENV, None)
        sys.modules.pop("mp_api", None)
        sys.modules.pop("mp_api.client", None)

    def test_parses_dielectric_doc(self):
        doc = _AttrDoc(
            material_id="mp-149",
            formula_pretty="Si",
            e_total=13.0,
            e_ionic=1.3,
            e_electronic=11.7,
            n=3.42,
        )
        fake_mp_api, fake_client, _ = self._fake_mprester_module([doc])
        with mock.patch.dict(sys.modules, {"mp_api": fake_mp_api, "mp_api.client": fake_client}):
            result = mpc.dielectric_properties("Si")

        self.assertEqual(result["material_id"], "mp-149")
        self.assertEqual(result["eps_electronic"], 11.7)
        self.assertEqual(result["eps_ionic"], 1.3)
        self.assertEqual(result["eps_total"], 13.0)
        self.assertEqual(result["refractive_index_n"], 3.42)
        self.assertIn("eps_inf", result["notes"])

    def test_not_found_mentions_metal_caveat(self):
        fake_mp_api, fake_client, _ = self._fake_mprester_module([])
        with mock.patch.dict(sys.modules, {"mp_api": fake_mp_api, "mp_api.client": fake_client}):
            with self.assertRaises(mpc.MaterialsProjectNotFound) as ctx:
                mpc.dielectric_properties("Fe")
        self.assertIn("metals", str(ctx.exception).lower())


class TestDevRootEnvLoading(unittest.TestCase):
    """_load_dev_root_env finds and loads a .env file without clobbering
    existing environment variables."""

    def test_loads_temp_env_file_without_dotenv_dependency(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "SOME_TEST_VAR_XYZ=hello\n# a comment\nANOTHER=world\n",
                encoding="utf-8",
            )
            self.assertTrue(env_path.is_file())

            os.environ.pop("SOME_TEST_VAR_XYZ", None)
            os.environ.pop("ANOTHER", None)
            try:
                mpc._load_env_file(env_path)
                self.assertEqual(os.environ.get("SOME_TEST_VAR_XYZ"), "hello")
                self.assertEqual(os.environ.get("ANOTHER"), "world")
            finally:
                os.environ.pop("SOME_TEST_VAR_XYZ", None)
                os.environ.pop("ANOTHER", None)

    def test_existing_env_var_not_overridden(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("SOME_TEST_VAR_XYZ=fromfile\n", encoding="utf-8")

            os.environ["SOME_TEST_VAR_XYZ"] = "fromshell"
            try:
                # Manual-parser branch only overrides when python-dotenv
                # is absent; force that branch to test the "don't clobber"
                # contract explicitly.
                with mock.patch.dict(sys.modules, {"dotenv": None}):
                    mpc._load_env_file(env_path)
                self.assertEqual(os.environ.get("SOME_TEST_VAR_XYZ"), "fromshell")
            finally:
                os.environ.pop("SOME_TEST_VAR_XYZ", None)


if __name__ == "__main__":
    unittest.main()
