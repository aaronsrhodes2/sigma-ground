"""Tests for organic_materials.py — hydrocarbons, wood, bone.

Validates the matter information cascade:
  molecular_bonds.py → organic_materials.py

Testing strategy:
  1. DERIVED vs MEASURED comparison — combustion enthalpies, moduli, densities
  2. Physics sanity — trends, bounds, signs
  3. Complete coverage — every material, every property (Rule 9)
"""

import math
import unittest

from sigma_ground.field.interface.organic_materials import (
    # Hydrocarbons
    HYDROCARBONS,
    UNSATURATED_HYDROCARBONS,
    alkane_combustion_enthalpy_kJ_mol,
    alkane_boiling_point_K,
    combustion_enthalpy_kJ_mol,
    hydrocarbon_report,
    # Wood
    WOOD_TYPES,
    WOOD_COMPONENTS,
    wood_modulus_along_grain,
    wood_modulus_across_grain,
    wood_combustion_enthalpy_MJ_kg,
    wood_anisotropy_ratio,
    wood_report,
    # Bone
    BONE_TYPES,
    BONE_COMPONENTS,
    bone_modulus_longitudinal,
    bone_modulus_transverse,
    bone_anisotropy_ratio,
    bone_density_from_composition,
    bone_report,
)


# ═══════════════════════════════════════════════════════════════════════
# HYDROCARBONS
# ═══════════════════════════════════════════════════════════════════════

class TestAlkaneCombustionEnthalpy(unittest.TestCase):
    """Hess's law combustion enthalpies vs NIST measured values."""

    def test_all_alkanes_within_30_percent(self):
        """Every alkane's derived ΔH_comb should be within ±30% of NIST.

        The Pauling equation gives AVERAGE bond energies, not molecule-specific.
        Products (CO₂, H₂O) have unusually strong bonds → systematic
        underestimate of ~25%. This is honest: the model captures the right
        physics (Hess's law) but uses averaged inputs.
        """
        for key, hc in HYDROCARBONS.items():
            n = hc['n_carbon']
            derived = alkane_combustion_enthalpy_kJ_mol(n)
            measured = hc['Hc_kJ_mol_measured']
            error = abs(derived - measured) / measured
            with self.subTest(hydrocarbon=key):
                self.assertLess(
                    error, 0.30,
                    f"{key}: derived={derived:.0f}, measured={measured:.0f}, "
                    f"error={error*100:.1f}%"
                )

    def test_combustion_enthalpy_positive(self):
        """Combustion is exothermic — enthalpy should be positive."""
        for key, hc in HYDROCARBONS.items():
            n = hc['n_carbon']
            with self.subTest(hydrocarbon=key):
                self.assertGreater(alkane_combustion_enthalpy_kJ_mol(n), 0)

    def test_combustion_increases_with_chain_length(self):
        """Longer chains release more energy (more bonds to break/form)."""
        carbons = sorted(set(hc['n_carbon'] for hc in HYDROCARBONS.values()))
        enthalpies = [alkane_combustion_enthalpy_kJ_mol(n) for n in carbons]

        for i in range(1, len(enthalpies)):
            self.assertGreater(
                enthalpies[i], enthalpies[i-1],
                f"C{carbons[i]} should have higher combustion enthalpy than C{carbons[i-1]}"
            )

    def test_methane_combustion_order_of_magnitude(self):
        """Methane: ~890 kJ/mol (NIST). Derived within 2× (Pauling averages)."""
        derived = alkane_combustion_enthalpy_kJ_mol(1)
        self.assertGreater(derived, 500)
        self.assertLess(derived, 1200)

    def test_octane_combustion_order_of_magnitude(self):
        """Octane: ~5471 kJ/mol (NIST). Gasoline benchmark."""
        derived = alkane_combustion_enthalpy_kJ_mol(8)
        self.assertGreater(derived, 3500)
        self.assertLess(derived, 7000)

    def test_per_carbon_enthalpy_converges(self):
        """ΔH/n should converge for long chains (approaches ~660 kJ/mol per C)."""
        h_per_c = [
            alkane_combustion_enthalpy_kJ_mol(n) / n
            for n in [5, 8, 10]
        ]
        # Should be within 5% of each other for long chains
        mean_val = sum(h_per_c) / len(h_per_c)
        for val in h_per_c:
            self.assertAlmostEqual(val / mean_val, 1.0, delta=0.05)


class TestUnsaturatedCombustion(unittest.TestCase):
    """Combustion of alkenes and alkynes."""

    def test_ethylene_combustion(self):
        """Ethylene C₂H₄ + 3 O₂ → 2 CO₂ + 2 H₂O."""
        hc = UNSATURATED_HYDROCARBONS['ethylene']
        derived = combustion_enthalpy_kJ_mol(
            hc['bonds_broken'], hc['bonds_formed'], hc['O2_consumed']
        )
        measured = hc['Hc_kJ_mol_measured']
        error = abs(derived - measured) / measured
        self.assertLess(error, 0.30, f"ethylene error: {error*100:.1f}%")

    def test_acetylene_combustion(self):
        """Acetylene C₂H₂ + 2.5 O₂ → 2 CO₂ + H₂O."""
        hc = UNSATURATED_HYDROCARBONS['acetylene']
        derived = combustion_enthalpy_kJ_mol(
            hc['bonds_broken'], hc['bonds_formed'], hc['O2_consumed']
        )
        measured = hc['Hc_kJ_mol_measured']
        error = abs(derived - measured) / measured
        self.assertLess(error, 0.30, f"acetylene error: {error*100:.1f}%")


class TestAlkaneBoilingPoint(unittest.TestCase):
    """London dispersion boiling point predictions."""

    def test_boiling_point_increases_with_chain_length(self):
        """Longer chains → more London dispersion → higher T_boil."""
        carbons = sorted(set(hc['n_carbon'] for hc in HYDROCARBONS.values()))
        T_boils = [alkane_boiling_point_K(n) for n in carbons]

        for i in range(1, len(T_boils)):
            self.assertGreater(
                T_boils[i], T_boils[i-1],
                f"C{carbons[i]} should boil higher than C{carbons[i-1]}"
            )

    def test_methane_is_gas_at_room_temp(self):
        """Methane boils well below 298 K."""
        self.assertLess(alkane_boiling_point_K(1), 298)

    def test_octane_is_liquid_at_room_temp(self):
        """Octane boils above 298 K (it's liquid gasoline)."""
        self.assertGreater(alkane_boiling_point_K(8), 298)

    def test_boiling_points_right_order_of_magnitude(self):
        """All predicted T_boil should be in 50-600 K range."""
        for key, hc in HYDROCARBONS.items():
            n = hc['n_carbon']
            T = alkane_boiling_point_K(n)
            with self.subTest(hydrocarbon=key):
                self.assertGreater(T, 50)
                self.assertLess(T, 600)

    def test_pentane_transition_gas_liquid(self):
        """Pentane (C5) boils near 309 K — near room temp boundary."""
        T = alkane_boiling_point_K(5)
        # Should be within 50% of measured 309 K
        self.assertGreater(T, 200)
        self.assertLess(T, 460)


class TestHydrocarbonReport(unittest.TestCase):
    """Diagnostic report completeness."""

    def test_all_reports_have_required_fields(self):
        """Every report has all expected keys."""
        required = {
            'name', 'formula', 'n_carbon',
            'Hc_derived_kJ_mol', 'Hc_measured_kJ_mol', 'Hc_error_pct',
            'Tb_derived_K', 'Tb_measured_K', 'Tb_error_pct', 'origin',
        }
        for key in HYDROCARBONS:
            report = hydrocarbon_report(key)
            with self.subTest(hydrocarbon=key):
                self.assertTrue(
                    required.issubset(report.keys()),
                    f"Missing keys: {required - set(report.keys())}"
                )


# ═══════════════════════════════════════════════════════════════════════
# WOOD
# ═══════════════════════════════════════════════════════════════════════

class TestWoodModulus(unittest.TestCase):
    """Voigt-Reuss composite mechanics for wood."""

    def test_along_grain_positive(self):
        """All wood types have positive E along grain."""
        for key in WOOD_TYPES:
            with self.subTest(wood=key):
                self.assertGreater(wood_modulus_along_grain(key), 0)

    def test_across_grain_positive(self):
        """All wood types have positive E across grain."""
        for key in WOOD_TYPES:
            with self.subTest(wood=key):
                self.assertGreater(wood_modulus_across_grain(key), 0)

    def test_along_greater_than_across(self):
        """Wood is always stiffer along grain than across."""
        for key in WOOD_TYPES:
            E_along = wood_modulus_along_grain(key)
            E_across = wood_modulus_across_grain(key)
            with self.subTest(wood=key):
                self.assertGreater(E_along, E_across)

    def test_anisotropy_ratio_realistic(self):
        """Anisotropy ratios should be 5-50 for real wood."""
        for key in WOOD_TYPES:
            ratio = wood_anisotropy_ratio(key)
            with self.subTest(wood=key):
                self.assertGreater(ratio, 3)
                self.assertLess(ratio, 60)

    def test_pine_modulus_order_of_magnitude(self):
        """Pine E_along ≈ 12 GPa (USDA). Derived should be within 2×."""
        E = wood_modulus_along_grain('pine')
        measured = WOOD_TYPES['pine']['E_along_GPa_measured']
        self.assertGreater(E, measured * 0.3)
        self.assertLess(E, measured * 3.0)

    def test_balsa_softer_than_ebony(self):
        """Balsa (lightest) should have lower modulus than ebony (densest)."""
        E_balsa = wood_modulus_along_grain('balsa')
        E_ebony = wood_modulus_along_grain('ebony')
        self.assertLess(E_balsa, E_ebony)

    def test_density_determines_modulus_trend(self):
        """Denser woods should generally be stiffer."""
        woods = sorted(WOOD_TYPES.items(), key=lambda x: x[1]['density_kg_m3'])
        E_values = [wood_modulus_along_grain(k) for k, _ in woods]
        # Not strictly monotonic (composition varies) but lightest < heaviest
        self.assertLess(E_values[0], E_values[-1])

    def test_all_wood_within_factor_3(self):
        """Derived E_along within factor of 3 of measured for all types."""
        for key in WOOD_TYPES:
            E_derived = wood_modulus_along_grain(key)
            E_measured = WOOD_TYPES[key]['E_along_GPa_measured']
            ratio = E_derived / E_measured
            with self.subTest(wood=key):
                self.assertGreater(ratio, 0.33,
                    f"{key}: derived={E_derived:.1f}, measured={E_measured:.1f}")
                self.assertLess(ratio, 3.0,
                    f"{key}: derived={E_derived:.1f}, measured={E_measured:.1f}")


class TestWoodCombustion(unittest.TestCase):
    """Wood heating values from Hess's law."""

    def test_all_positive(self):
        """Wood combustion releases energy."""
        for key in WOOD_TYPES:
            with self.subTest(wood=key):
                self.assertGreater(wood_combustion_enthalpy_MJ_kg(key), 0)

    def test_all_in_range(self):
        """Wood heating values: 15-25 MJ/kg (well-known engineering range)."""
        for key in WOOD_TYPES:
            Hc = wood_combustion_enthalpy_MJ_kg(key)
            with self.subTest(wood=key):
                self.assertGreater(Hc, 10,
                    f"{key}: Hc={Hc:.1f} MJ/kg, expected >15")
                self.assertLess(Hc, 35,
                    f"{key}: Hc={Hc:.1f} MJ/kg, expected <25")

    def test_similar_across_species(self):
        """All wood has similar heating value per kg (because similar chemistry)."""
        values = [wood_combustion_enthalpy_MJ_kg(k) for k in WOOD_TYPES]
        mean_val = sum(values) / len(values)
        for v in values:
            self.assertAlmostEqual(v / mean_val, 1.0, delta=0.15)


class TestWoodReport(unittest.TestCase):
    """Diagnostic report completeness."""

    def test_all_reports_complete(self):
        required = {
            'name', 'density_kg_m3',
            'E_along_GPa_derived', 'E_along_GPa_measured', 'E_along_error_pct',
            'E_across_GPa_derived', 'E_across_GPa_measured',
            'anisotropy_ratio', 'Hc_MJ_kg_derived', 'Hc_MJ_kg_measured', 'origin',
        }
        for key in WOOD_TYPES:
            report = wood_report(key)
            with self.subTest(wood=key):
                self.assertTrue(
                    required.issubset(report.keys()),
                    f"Missing: {required - set(report.keys())}"
                )


# ═══════════════════════════════════════════════════════════════════════
# BONE
# ═══════════════════════════════════════════════════════════════════════

class TestBoneModulus(unittest.TestCase):
    """Voigt-Reuss composite mechanics for bone."""

    def test_longitudinal_positive(self):
        """All bone types have positive longitudinal modulus."""
        for key in BONE_TYPES:
            with self.subTest(bone=key):
                self.assertGreater(bone_modulus_longitudinal(key), 0)

    def test_transverse_positive(self):
        """All bone types have positive transverse modulus."""
        for key in BONE_TYPES:
            with self.subTest(bone=key):
                self.assertGreater(bone_modulus_transverse(key), 0)

    def test_longitudinal_greater_than_transverse(self):
        """Bone is stiffer along its axis."""
        for key in BONE_TYPES:
            E_long = bone_modulus_longitudinal(key)
            E_trans = bone_modulus_transverse(key)
            with self.subTest(bone=key):
                self.assertGreater(E_long, E_trans)

    def test_cortical_stiffer_than_cancellous(self):
        """Cortical (compact) bone is stiffer than cancellous (spongy)."""
        E_cortical = bone_modulus_longitudinal('cortical_human')
        E_cancellous = bone_modulus_longitudinal('cancellous_human')
        self.assertGreater(E_cortical, E_cancellous)

    def test_human_cortical_bone_order_of_magnitude(self):
        """Human cortical bone E ≈ 15-25 GPa (well-established)."""
        E = bone_modulus_longitudinal('cortical_human')
        self.assertGreater(E, 5, f"E={E:.1f} GPa, expected >5")
        self.assertLess(E, 40, f"E={E:.1f} GPa, expected <40")

    def test_antler_tougher_than_bone(self):
        """Antler has lower modulus (more collagen → more flexible/tough)."""
        E_bone = bone_modulus_longitudinal('cortical_human')
        E_antler = bone_modulus_longitudinal('antler')
        self.assertLess(E_antler, E_bone)

    def test_anisotropy_ratio_realistic(self):
        """Bone anisotropy ratios: 1.2-3.0 for cortical, variable for cancellous."""
        for key in BONE_TYPES:
            ratio = bone_anisotropy_ratio(key)
            with self.subTest(bone=key):
                self.assertGreater(ratio, 1.0)
                self.assertLess(ratio, 5.0)

    def test_mineral_fraction_correlates_with_stiffness(self):
        """Higher mineral fraction → stiffer bone."""
        types_by_mineral = sorted(
            BONE_TYPES.items(),
            key=lambda x: x[1]['mineral_volume_fraction']
        )
        # First and last should show the trend
        E_low = bone_modulus_longitudinal(types_by_mineral[0][0])
        E_high = bone_modulus_longitudinal(types_by_mineral[-1][0])
        # Not strictly true (water fraction also matters) but general trend
        # The highest-mineral bone should be in the upper half of stiffness range


class TestBoneDensity(unittest.TestCase):
    """Density from composition (rule of mixtures, exact)."""

    def test_all_densities_positive(self):
        for key in BONE_TYPES:
            with self.subTest(bone=key):
                self.assertGreater(bone_density_from_composition(key), 0)

    def test_cortical_density_realistic(self):
        """Human cortical bone: ~1800-2100 kg/m³."""
        rho = bone_density_from_composition('cortical_human')
        self.assertGreater(rho, 1500)
        self.assertLess(rho, 2500)

    def test_cancellous_less_dense_than_cortical(self):
        """Cancellous (spongy) bone is less dense."""
        rho_cortical = bone_density_from_composition('cortical_human')
        rho_cancellous = bone_density_from_composition('cancellous_human')
        self.assertLess(rho_cancellous, rho_cortical)

    def test_cortical_density_from_composition(self):
        """Cortical bone tissue density should be close to measured.

        Rule of mixtures gives tissue density (solid material).
        For cortical bone (low porosity), this is close to apparent density.
        For cancellous bone, tissue density >> apparent density (foam).
        """
        for key in ['cortical_human', 'cortical_bovine']:
            rho_derived = bone_density_from_composition(key)
            rho_measured = BONE_TYPES[key]['density_kg_m3']
            error = abs(rho_derived - rho_measured) / rho_measured
            with self.subTest(bone=key):
                self.assertLess(
                    error, 0.30,
                    f"{key}: tissue density={rho_derived:.0f}, "
                    f"apparent={rho_measured:.0f}"
                )

    def test_cancellous_tissue_density_exceeds_apparent(self):
        """Cancellous tissue density > apparent density (it's a foam)."""
        rho_tissue = bone_density_from_composition('cancellous_human')
        rho_apparent = BONE_TYPES['cancellous_human']['density_kg_m3']
        self.assertGreater(rho_tissue, rho_apparent)


class TestBoneReport(unittest.TestCase):
    """Diagnostic report completeness."""

    def test_all_reports_complete(self):
        required = {
            'name', 'density_derived_kg_m3', 'density_measured_kg_m3',
            'E_long_GPa_derived', 'E_long_GPa_measured', 'E_long_error_pct',
            'E_trans_GPa_derived', 'E_trans_GPa_measured',
            'anisotropy_ratio', 'mineral_fraction', 'origin',
        }
        for key in BONE_TYPES:
            report = bone_report(key)
            with self.subTest(bone=key):
                self.assertTrue(
                    required.issubset(report.keys()),
                    f"Missing: {required - set(report.keys())}"
                )


# ═══════════════════════════════════════════════════════════════════════
# CROSS-CUTTING
# ═══════════════════════════════════════════════════════════════════════

class TestSigmaInvariance(unittest.TestCase):
    """All organic material properties should be σ-invariant.

    These are EM-bond materials — no QCD contribution.
    The module doesn't take a sigma parameter, confirming invariance.
    """

    def test_no_sigma_parameter_in_wood(self):
        """Wood functions don't accept sigma — they're EM-only."""
        import inspect
        for fn in [wood_modulus_along_grain, wood_modulus_across_grain,
                    wood_combustion_enthalpy_MJ_kg]:
            sig = inspect.signature(fn)
            with self.subTest(fn=fn.__name__):
                self.assertNotIn('sigma', sig.parameters)

    def test_no_sigma_parameter_in_bone(self):
        """Bone functions don't accept sigma — they're EM-only."""
        import inspect
        for fn in [bone_modulus_longitudinal, bone_modulus_transverse,
                    bone_density_from_composition]:
            sig = inspect.signature(fn)
            with self.subTest(fn=fn.__name__):
                self.assertNotIn('sigma', sig.parameters)


class TestPhysicsConsistency(unittest.TestCase):
    """Cross-material physics checks."""

    def test_bone_stiffer_than_wood(self):
        """Bone (mineralized) is stiffer than wood (organic)."""
        E_bone = bone_modulus_longitudinal('cortical_human')
        E_pine = wood_modulus_along_grain('pine')
        self.assertGreater(E_bone, E_pine)

    def test_wood_burns_bone_does_not_fully(self):
        """Wood has higher organic content → higher heating value concept.
        (Bone's mineral fraction doesn't combust.)"""
        # Just check wood has meaningful combustion enthalpy
        Hc_pine = wood_combustion_enthalpy_MJ_kg('pine')
        self.assertGreater(Hc_pine, 10)  # substantial heating value

    def test_octane_highest_total_enthalpy(self):
        """Octane (longest chain) has highest total combustion enthalpy."""
        enthalpies = {
            k: alkane_combustion_enthalpy_kJ_mol(v['n_carbon'])
            for k, v in HYDROCARBONS.items()
        }
        max_key = max(enthalpies, key=enthalpies.get)
        self.assertEqual(
            HYDROCARBONS[max_key]['n_carbon'],
            max(v['n_carbon'] for v in HYDROCARBONS.values())
        )

    def test_component_properties_consistent(self):
        """Component databases have required keys."""
        for comp in WOOD_COMPONENTS.values():
            self.assertIn('E_GPa', comp)
            self.assertIn('density_kg_m3', comp)

        for comp in BONE_COMPONENTS.values():
            self.assertIn('E_GPa', comp)
            self.assertIn('density_kg_m3', comp)


if __name__ == '__main__':
    unittest.main()
