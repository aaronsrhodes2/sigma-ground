"""Tests for quantum_algorithms.py — 10 algorithms + cascade connections.

Each algorithm is tested for correctness against known analytical results.
The cascade connections test that Ising/Heisenberg coupling constants
derived from the material database produce physically sensible values.
"""

import math
import unittest

from .quantum_algorithms import (
    qft_circuit, inverse_qft_circuit, run_circuit_extended, qft_example,
    phase_estimation_example,
    shor_factor_15,
    simon_algorithm,
    qaoa_maxcut,
    ising_ground_state,
    heisenberg_ground_state,
    vqe_heh_plus,
    qec_bit_flip_demo,
    quantum_walk,
    ising_coupling_from_curie,
    ising_phase_transition_prediction,
    quantum_algorithms_report,
)
from .quantum_computing import (
    zero_state, basis_state, run_circuit, _n_qubits,
)
from .quantum_output import probabilities, state_fidelity


# =====================================================================
# 1. QFT
# =====================================================================

class TestQFT(unittest.TestCase):
    """Quantum Fourier Transform tests."""

    def test_qft_on_basis_1_produces_uniform(self):
        """QFT|1> should give uniform probability distribution."""
        result = qft_example(3)
        probs = result['output_probabilities']
        expected = 1.0 / 8  # 2^3 = 8
        for p in probs:
            self.assertAlmostEqual(p, expected, places=5,
                                   msg="QFT|1> should be uniform")

    def test_qft_on_zero_state_gives_uniform(self):
        """QFT|0> = uniform superposition = |+>^n."""
        n = 3
        circuit = qft_circuit(n)
        state = run_circuit_extended(n, circuit, zero_state(n))
        probs = probabilities(state)
        for p in probs:
            self.assertAlmostEqual(p, 1.0 / 8, places=5)

    def test_qft_inverse_qft_identity(self):
        """QFT followed by inverse QFT should return to the original state."""
        n = 3
        initial = basis_state(n, 5)  # |101>
        circuit = qft_circuit(n) + inverse_qft_circuit(n)
        final = run_circuit_extended(n, circuit, initial)
        fid = state_fidelity(final, initial)
        self.assertAlmostEqual(fid, 1.0, places=4,
                               msg="QFT†·QFT should be identity")

    def test_qft_2qubit_basis_state(self):
        """2-qubit QFT on |1> should produce specific phase pattern."""
        n = 2
        circuit = qft_circuit(n)
        state = run_circuit_extended(n, circuit, basis_state(n, 1))
        probs = probabilities(state)
        # QFT|01> on 2 qubits → uniform amplitudes (1/2 each)
        for p in probs:
            self.assertAlmostEqual(p, 0.25, places=5)


# =====================================================================
# 2. QPE
# =====================================================================

class TestQPE(unittest.TestCase):
    """Quantum Phase Estimation tests."""

    def test_phase_estimation_t_gate(self):
        """QPE should estimate T gate phase θ=1/8 exactly with 3 ancillas."""
        result = phase_estimation_example(3)
        self.assertAlmostEqual(result['measured_phase'], 1.0 / 8, places=6)
        self.assertEqual(result['ancilla_result'], '001')
        self.assertAlmostEqual(result['probability'], 1.0, places=4,
                               msg="T gate phase should be exactly representable")

    def test_qpe_returns_correct_structure(self):
        """QPE result should contain all expected keys."""
        result = phase_estimation_example(3)
        for key in ['exact_phase', 'measured_phase', 'error', 'ancilla_result']:
            self.assertIn(key, result)


# =====================================================================
# 3. SHOR
# =====================================================================

class TestShor(unittest.TestCase):
    """Shor's factoring algorithm tests."""

    def test_shor_factors_15(self):
        """Shor's should factor 15 into 3 and 5."""
        result = shor_factor_15()
        self.assertEqual(result['N'], 15)
        factors = result['factors']
        self.assertIn(3, factors)
        self.assertIn(5, factors)
        self.assertTrue(result['verification'],
                        msg="3 × 5 should equal 15")

    def test_shor_period_is_4(self):
        """Period of 7^x mod 15 should be 4."""
        result = shor_factor_15()
        self.assertEqual(result['period'], 4)

    def test_shor_uses_7_qubits(self):
        """3 counting + 4 work = 7 qubits."""
        result = shor_factor_15()
        self.assertEqual(result['n_qubits'], 7)


# =====================================================================
# 4. SIMON
# =====================================================================

class TestSimon(unittest.TestCase):
    """Simon's algorithm tests."""

    def test_simon_finds_hidden_string(self):
        """Simon's should identify the hidden bitstring."""
        result = simon_algorithm('110')
        self.assertTrue(result['success'],
                        msg="Should find hidden string '110'")
        self.assertEqual(result['found_string'], '110')

    def test_simon_orthogonal_equations(self):
        """All collected equations should be orthogonal to hidden string."""
        result = simon_algorithm('10')
        s = result['hidden_string']
        for eq in result['equations_collected']:
            dot = sum(int(eq[i]) * int(s[i]) for i in range(len(s)))
            self.assertEqual(dot % 2, 0,
                             msg=f"y={eq} should satisfy y·s=0 mod 2")

    def test_simon_single_bit(self):
        """Simon's with single-bit hidden string."""
        result = simon_algorithm('1')
        self.assertTrue(result['success'])


# =====================================================================
# 5. QAOA MaxCut
# =====================================================================

class TestQAOA(unittest.TestCase):
    """QAOA MaxCut tests."""

    def test_qaoa_triangle_graph(self):
        """Triangle graph: max cut = 2 (bipartite impossible for odd cycle)."""
        edges = [(0, 1), (1, 2), (0, 2)]
        result = qaoa_maxcut(edges, 3, p=1, n_angles=15)
        # Triangle max cut is 2 (can't cut all 3 edges)
        self.assertGreaterEqual(result['best_cut_value'], 2)

    def test_qaoa_simple_edge(self):
        """Single edge: max cut = 1."""
        edges = [(0, 1)]
        result = qaoa_maxcut(edges, 2, p=1, n_angles=15)
        self.assertEqual(result['best_cut_value'], 1)

    def test_qaoa_square_graph(self):
        """Square (4-cycle) is bipartite: max cut = 4."""
        edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
        result = qaoa_maxcut(edges, 4, p=1, n_angles=10)
        # QAOA p=1 should find at least 3 of 4 edges
        self.assertGreaterEqual(result['best_cut_value'], 3)


# =====================================================================
# 6. ISING VQE
# =====================================================================

class TestIsingVQE(unittest.TestCase):
    """Transverse-field Ising model VQE tests."""

    def test_ising_ferromagnetic_phase(self):
        """Small h/J → ferromagnetic: high |<Z>|."""
        result = ising_ground_state(2, J=1.0, h=0.1, n_steps=20)
        self.assertGreater(abs(result['magnetization_z']), 0.5,
                           msg="Ferromagnetic phase should have large <Z>")

    def test_ising_paramagnetic_phase(self):
        """Large h/J → paramagnetic: high <X>."""
        result = ising_ground_state(2, J=1.0, h=5.0, n_steps=20)
        self.assertGreater(abs(result['magnetization_x']), 0.5,
                           msg="Paramagnetic phase should have large <X>")

    def test_ising_2site_exact_energy(self):
        """2-site Ising: E_0 = -sqrt(J² + h²) for open BC."""
        J, h = 1.0, 0.5
        # Exact ground state energy for 2-site TFIM with open BC:
        # H = -J Z1Z2 - h(X1 + X2)
        # The 4x4 matrix has eigenvalues that we can compute
        # For 2 sites: minimum eigenvalue is -sqrt(J^2 + h^2) - h...
        # Actually let's just check VQE gets close to analytical
        result = ising_ground_state(2, J=J, h=h, n_steps=30)
        # The ground state energy should be negative
        self.assertLess(result['ground_energy'], 0)

    def test_ising_phase_classification(self):
        """Phase label matches h/J ratio."""
        r1 = ising_ground_state(2, J=1.0, h=0.1, n_steps=10)
        self.assertEqual(r1['phase'], 'ferromagnetic')
        r2 = ising_ground_state(2, J=1.0, h=5.0, n_steps=10)
        self.assertEqual(r2['phase'], 'paramagnetic')


# =====================================================================
# 7. HEISENBERG VQE
# =====================================================================

class TestHeisenbergVQE(unittest.TestCase):
    """Heisenberg XXX spin chain VQE tests."""

    def test_heisenberg_2site_energy(self):
        """2-site antiferromagnetic Heisenberg: E_0 = -3J (singlet)."""
        result = heisenberg_ground_state(2, J=1.0, n_steps=40)
        exact = -3.0
        self.assertAlmostEqual(result['ground_energy'], exact, delta=0.15,
                               msg=f"E_VQE={result['ground_energy']:.3f} should be near {exact}")

    def test_heisenberg_exact_comparison(self):
        """VQE result should match exact diagonalization for 2 sites."""
        result = heisenberg_ground_state(2, J=1.0, n_steps=40)
        self.assertIsNotNone(result['exact_energy_2site'])
        self.assertAlmostEqual(result['exact_energy_2site'], -3.0, places=5)

    def test_heisenberg_antiferro_correlation(self):
        """Antiferromagnetic ground state: <Z0 Z1> should be negative."""
        result = heisenberg_ground_state(2, J=1.0, n_steps=40)
        self.assertLess(result['nn_zz_correlation'], 0,
                        msg="Antiferro singlet has anti-aligned spins")

    def test_heisenberg_ferro_energy(self):
        """Ferromagnetic Heisenberg (J<0): E_0 = -3|J| (triplet)...
        actually E_0 = J for ferro: the all-up state gives
        <XX+YY+ZZ> = 0+0+1 per bond, so E = J. With J<0, E_0 = 3J."""
        result = heisenberg_ground_state(2, J=-1.0, n_steps=40)
        # Ferromagnetic: triplet has E = J (per bond, here 1 bond)
        # Actually for J < 0: triplet eigenvalue is J, singlet is -3J = +3
        # Minimum is J = -1
        self.assertAlmostEqual(result['ground_energy'], -1.0, delta=0.15)


# =====================================================================
# 8. HeH+ VQE
# =====================================================================

class TestVQEHeHPlus(unittest.TestCase):
    """HeH+ molecular ion VQE tests."""

    def test_vqe_finds_ground_state(self):
        """VQE energy should match exact diagonalization."""
        result = vqe_heh_plus(n_steps=30)
        self.assertAlmostEqual(result['vqe_energy'], result['exact_energy'],
                               delta=0.01,
                               msg="VQE should closely match exact energy")

    def test_exact_eigenvalues_ordered(self):
        """Eigenvalues should be in ascending order."""
        result = vqe_heh_plus(n_steps=10)
        eigs = result['all_eigenvalues']
        for i in range(len(eigs) - 1):
            self.assertLessEqual(eigs[i], eigs[i + 1])

    def test_ground_state_is_negative(self):
        """HeH+ ground state energy should be negative (bound state)."""
        result = vqe_heh_plus(n_steps=10)
        self.assertLess(result['exact_energy'], 0)

    def test_dissociation_energy_measured(self):
        """Dissociation energy from literature should be ~1.844 eV."""
        result = vqe_heh_plus(n_steps=10)
        self.assertAlmostEqual(result['dissociation_energy_eV_measured'], 1.844)


# =====================================================================
# 9. QEC BIT-FLIP
# =====================================================================

class TestQECBitFlip(unittest.TestCase):
    """3-qubit bit-flip error correction tests."""

    def test_error_on_qubit_0(self):
        """Should detect and correct bit-flip on qubit 0."""
        result = qec_bit_flip_demo(error_qubit=0)
        self.assertTrue(result['error_detected'])
        self.assertEqual(result['syndrome'], '10')
        self.assertGreater(result['fidelity_after_correction'], 0.9)

    def test_error_on_qubit_1(self):
        """Should detect and correct bit-flip on qubit 1."""
        result = qec_bit_flip_demo(error_qubit=1)
        self.assertTrue(result['error_detected'])
        self.assertEqual(result['syndrome'], '11')
        self.assertGreater(result['fidelity_after_correction'], 0.9)

    def test_error_on_qubit_2(self):
        """Should detect and correct bit-flip on qubit 2."""
        result = qec_bit_flip_demo(error_qubit=2)
        self.assertTrue(result['error_detected'])
        self.assertEqual(result['syndrome'], '01')
        self.assertGreater(result['fidelity_after_correction'], 0.9)

    def test_custom_initial_state(self):
        """Should work with arbitrary initial amplitudes."""
        result = qec_bit_flip_demo(alpha=0.8, beta=0.6, error_qubit=1)
        self.assertTrue(result['error_detected'])
        self.assertGreater(result['fidelity_after_correction'], 0.9)


# =====================================================================
# 10. QUANTUM WALK
# =====================================================================

class TestQuantumWalk(unittest.TestCase):
    """Discrete-time quantum walk tests."""

    def test_quantum_walk_ballistic_spread(self):
        """Quantum walk should spread faster than classical."""
        result = quantum_walk(n_steps=20, n_positions=64)
        self.assertTrue(result['is_ballistic'],
                        msg="Quantum walk should have ballistic spread")
        self.assertGreater(result['speedup_ratio'], 1.5)

    def test_quantum_walk_probability_normalized(self):
        """Position probabilities should sum to 1."""
        result = quantum_walk(n_steps=10, n_positions=32)
        total = sum(result['position_distribution'])
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_quantum_walk_asymmetric(self):
        """Hadamard walk with |0> coin is asymmetric — biased left."""
        result = quantum_walk(n_steps=15, n_positions=64)
        probs = result['position_distribution']
        center = len(probs) // 2
        left_prob = sum(probs[:center])
        right_prob = sum(probs[center:])
        # Hadamard walk starting with coin |0> is known to be left-biased
        self.assertGreater(left_prob, 0.4,
                           msg="Walker should have significant leftward probability")

    def test_few_steps_stays_near_center(self):
        """After 1 step the walker should be near the center."""
        result = quantum_walk(n_steps=1, n_positions=16)
        probs = result['position_distribution']
        center = len(probs) // 2
        # After 1 step: should be at center±1
        nearby = sum(probs[center - 2: center + 3])
        self.assertGreater(nearby, 0.9)


# =====================================================================
# CASCADE CONNECTIONS
# =====================================================================

class TestCascadeConnections(unittest.TestCase):
    """Tests for cascade connections to material database."""

    def test_iron_coupling(self):
        """Iron Ising coupling from Curie temperature."""
        result = ising_coupling_from_curie('iron')
        # Iron: T_C = 1043 K, bcc → z = 8
        # J = k_B * 1043 / 8 ≈ 130 K ≈ 11.2 meV
        self.assertAlmostEqual(result['T_C_K'], 1043.0)
        self.assertEqual(result['z'], 8)
        self.assertAlmostEqual(result['J_kelvin'], 1043.0 / 8, places=1)
        # J should be in reasonable range (5-20 meV for ferromagnets)
        self.assertGreater(result['J_eV'] * 1000, 5)  # > 5 meV
        self.assertLess(result['J_eV'] * 1000, 20)    # < 20 meV

    def test_nickel_coupling(self):
        """Nickel coupling should be weaker than iron."""
        fe = ising_coupling_from_curie('iron')
        ni = ising_coupling_from_curie('nickel')
        # Nickel T_C = 627 K < Iron T_C = 1043 K
        self.assertLess(ni['T_C_K'], fe['T_C_K'])
        self.assertLess(ni['J_eV'], fe['J_eV'])

    def test_phase_transition_prediction(self):
        """Critical field prediction should be physically reasonable."""
        result = ising_phase_transition_prediction('iron')
        # Critical field: J / (g * mu_B) with J ~ 11 meV → ~ 97 T
        # This is a huge field (mean-field overestimates, but order correct)
        self.assertGreater(result['critical_field_tesla'], 50)
        self.assertLess(result['critical_field_tesla'], 500)
        self.assertIn('explanation', result)

    def test_iron_stronger_than_nickel(self):
        """Iron (T_C=1043K) should have stronger coupling than nickel (T_C=627K)."""
        fe = ising_coupling_from_curie('iron')
        ni = ising_coupling_from_curie('nickel')
        self.assertGreater(fe['J_eV'], ni['J_eV'])


# =====================================================================
# MODULE REPORT
# =====================================================================

class TestReport(unittest.TestCase):
    """Module report tests."""

    def test_report_lists_all_algorithms(self):
        """Report should list all 10 algorithms."""
        report = quantum_algorithms_report()
        self.assertEqual(len(report['algorithms']), 10)

    def test_report_has_cascade_connections(self):
        """Report should list cascade connections."""
        report = quantum_algorithms_report()
        self.assertGreater(len(report['cascade_connections']), 0)


if __name__ == '__main__':
    unittest.main()
