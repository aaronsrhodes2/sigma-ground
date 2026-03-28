"""
Tests for quantum_output.py — measurement, sampling, extraction, algorithms.

Strategy:
  - Test measurement: |0> always gives 0, |1> always gives 1
  - Test probabilities: equal superposition gives 50/50
  - Test sampling: histogram converges to probabilities (statistical)
  - Test expectation: <0|Z|0> = +1, <1|Z|1> = -1, <+|X|+> = +1
  - Test fidelity: identical states → 1.0, orthogonal → 0.0
  - Test entanglement entropy: product state → 0, Bell state → ln(2)
  - Test Bloch sphere: |0> → (0, 0), |1> → (π, 0)
  - Test Deutsch-Jozsa: correctly identifies constant and balanced
  - Test Grover: finds marked item with high probability
  - Test Bell state: histogram shows only |00> and |11>
  - Test Bernstein-Vazirani: recovers hidden string
  - Test teleportation: qubit state transferred
  - Test Rule 9: reports
"""

import math
import random
import unittest

from sigma_ground.field.interface.quantum_computing import (
    zero_state,
    basis_state,
    product_state,
    gate_h,
    gate_x,
    gate_cnot,
    run_circuit,
)

from sigma_ground.field.interface.quantum_output import (
    probabilities,
    probability,
    measure,
    measure_all,
    sample,
    sample_marginal,
    expectation_pauli,
    expectation_observable,
    state_fidelity,
    entanglement_entropy,
    state_to_bloch,
    schmidt_coefficients,
    extract_max_probability,
    extract_phase,
    bell_state_example,
    deutsch_jozsa,
    grover_search,
    teleportation_example,
    bernstein_vazirani,
    quantum_output_report,
    full_report,
)


class TestProbabilities(unittest.TestCase):
    """Probability computation (no collapse)."""

    def test_zero_state_certain(self):
        """P(qubit=0) = 1 for |00...0>."""
        s = zero_state(2)
        self.assertAlmostEqual(probability(s, 0, 0), 1.0, places=14)
        self.assertAlmostEqual(probability(s, 0, 1), 0.0, places=14)

    def test_superposition_half(self):
        """Equal superposition gives P=0.5 for each outcome."""
        s = gate_h(zero_state(1), 0)
        self.assertAlmostEqual(probability(s, 0, 0), 0.5, places=14)
        self.assertAlmostEqual(probability(s, 0, 1), 0.5, places=14)

    def test_probabilities_sum_to_one(self):
        """probabilities() entries sum to 1."""
        s = gate_h(zero_state(2), 0)
        s = gate_cnot(s, 0, 1)  # Bell state
        probs = probabilities(s)
        self.assertAlmostEqual(sum(probs), 1.0, places=14)

    def test_bell_state_probabilities(self):
        """Bell state: P(|00>) = P(|11>) = 0.5, others = 0."""
        s = gate_h(zero_state(2), 0)
        s = gate_cnot(s, 0, 1)
        probs = probabilities(s)
        self.assertAlmostEqual(probs[0], 0.5, places=10)
        self.assertAlmostEqual(probs[1], 0.0, places=10)
        self.assertAlmostEqual(probs[2], 0.0, places=10)
        self.assertAlmostEqual(probs[3], 0.5, places=10)


class TestMeasurement(unittest.TestCase):
    """Projective measurement with collapse."""

    def test_measure_zero_state(self):
        """|0> always measures as 0."""
        s = zero_state(1)
        for _ in range(10):
            _, outcome = measure(s, 0)
            self.assertEqual(outcome, 0)

    def test_measure_one_state(self):
        """|1> always measures as 1."""
        s = basis_state(1, 1)
        for _ in range(10):
            _, outcome = measure(s, 0)
            self.assertEqual(outcome, 1)

    def test_measure_collapses(self):
        """After measurement, repeated measurement gives same result."""
        random.seed(42)
        s = gate_h(zero_state(1), 0)  # |+>
        s, outcome = measure(s, 0)
        # After collapse, same measurement gives same outcome
        for _ in range(10):
            _, repeat = measure(s, 0)
            self.assertEqual(repeat, outcome)

    def test_measure_all_returns_bitstring(self):
        """measure_all returns a valid bitstring."""
        s = zero_state(3)
        _, bs = measure_all(s)
        self.assertEqual(bs, '000')
        self.assertEqual(len(bs), 3)


class TestSampling(unittest.TestCase):
    """Multi-shot sampling."""

    def test_sample_zero_state(self):
        """|00> always gives '00'."""
        s = zero_state(2)
        hist = sample(s, 100)
        self.assertEqual(hist, {'00': 100})

    def test_sample_bell_state_only_00_11(self):
        """Bell state only produces '00' and '11'."""
        random.seed(42)
        s = gate_h(zero_state(2), 0)
        s = gate_cnot(s, 0, 1)
        hist = sample(s, 1000)
        for key in hist:
            self.assertIn(key, ('00', '11'))

    def test_sample_converges(self):
        """Equal superposition converges to ~50/50."""
        random.seed(42)
        s = gate_h(zero_state(1), 0)
        hist = sample(s, 10000)
        p0 = hist.get('0', 0) / 10000
        self.assertAlmostEqual(p0, 0.5, delta=0.03)

    def test_sample_marginal_subset(self):
        """Marginal sampling returns correct length bitstrings."""
        s = zero_state(3)
        hist = sample_marginal(s, [0, 2], 100)
        for key in hist:
            self.assertEqual(len(key), 2)

    def test_sample_total_shots(self):
        """Total counts equal n_shots."""
        s = gate_h(zero_state(1), 0)
        hist = sample(s, 500)
        self.assertEqual(sum(hist.values()), 500)


class TestExpectationValues(unittest.TestCase):
    """Expectation values for Pauli strings and observables."""

    def test_z_expectation_zero_state(self):
        """<0|Z|0> = +1."""
        s = zero_state(1)
        self.assertAlmostEqual(expectation_pauli(s, 'Z'), 1.0, places=14)

    def test_z_expectation_one_state(self):
        """<1|Z|1> = -1."""
        s = basis_state(1, 1)
        self.assertAlmostEqual(expectation_pauli(s, 'Z'), -1.0, places=14)

    def test_x_expectation_plus_state(self):
        """<+|X|+> = +1."""
        s = gate_h(zero_state(1), 0)
        self.assertAlmostEqual(expectation_pauli(s, 'X'), 1.0, places=14)

    def test_x_expectation_zero_state(self):
        """<0|X|0> = 0."""
        s = zero_state(1)
        self.assertAlmostEqual(expectation_pauli(s, 'X'), 0.0, places=14)

    def test_identity_expectation(self):
        """<psi|I|psi> = 1 for any normalized state."""
        s = gate_h(zero_state(1), 0)
        self.assertAlmostEqual(expectation_pauli(s, 'I'), 1.0, places=14)

    def test_two_qubit_pauli(self):
        """<00|ZZ|00> = +1 (both qubits in |0>)."""
        s = zero_state(2)
        self.assertAlmostEqual(expectation_pauli(s, 'ZZ'), 1.0, places=14)

    def test_bell_state_zz(self):
        """<Bell|ZZ|Bell> = +1 (perfectly correlated)."""
        s = run_circuit(2, [('h', 0), ('cnot', 0, 1)])
        self.assertAlmostEqual(expectation_pauli(s, 'ZZ'), 1.0, places=10)

    def test_bell_state_xx(self):
        """<Bell|XX|Bell> = +1 (Bell state is eigenstate of XX)."""
        s = run_circuit(2, [('h', 0), ('cnot', 0, 1)])
        self.assertAlmostEqual(expectation_pauli(s, 'XX'), 1.0, places=10)

    def test_diagonal_observable(self):
        """Diagonal observable with eigenvalues [1, -1] matches Z."""
        s = gate_h(zero_state(1), 0)
        val = expectation_observable(s, [1, -1])
        self.assertAlmostEqual(val, 0.0, places=14)

    def test_wrong_pauli_length_raises(self):
        """Pauli string length mismatch raises ValueError."""
        s = zero_state(2)
        with self.assertRaises(ValueError):
            expectation_pauli(s, 'Z')  # 1 char for 2 qubits


class TestStateAnalysis(unittest.TestCase):
    """Fidelity, entropy, Bloch sphere, Schmidt decomposition."""

    def test_fidelity_identical(self):
        """Fidelity of identical states is 1."""
        s = gate_h(zero_state(1), 0)
        self.assertAlmostEqual(state_fidelity(s, s), 1.0, places=14)

    def test_fidelity_orthogonal(self):
        """Fidelity of orthogonal states is 0."""
        s0 = zero_state(1)
        s1 = basis_state(1, 1)
        self.assertAlmostEqual(state_fidelity(s0, s1), 0.0, places=14)

    def test_fidelity_symmetric(self):
        """Fidelity is symmetric: F(a,b) = F(b,a)."""
        s0 = gate_h(zero_state(1), 0)
        s1 = zero_state(1)
        self.assertAlmostEqual(
            state_fidelity(s0, s1), state_fidelity(s1, s0), places=14
        )

    def test_entropy_product_state(self):
        """Product state has zero entanglement entropy."""
        s = zero_state(2)  # |00> is a product state
        ent = entanglement_entropy(s, 0)
        self.assertAlmostEqual(ent, 0.0, delta=1e-10)

    def test_entropy_bell_state(self):
        """Bell state has maximum entanglement entropy ln(2)."""
        s = run_circuit(2, [('h', 0), ('cnot', 0, 1)])
        ent = entanglement_entropy(s, 0)
        self.assertAlmostEqual(ent, math.log(2), delta=0.01)

    def test_bloch_zero(self):
        """|0> maps to north pole (θ=0)."""
        theta, phi = state_to_bloch([complex(1), complex(0)])
        self.assertAlmostEqual(theta, 0.0, places=10)

    def test_bloch_one(self):
        """|1> maps to south pole (θ=π)."""
        theta, phi = state_to_bloch([complex(0), complex(1)])
        self.assertAlmostEqual(theta, math.pi, places=10)

    def test_bloch_plus(self):
        """|+> maps to equator (θ=π/2)."""
        r2 = 1.0 / math.sqrt(2)
        theta, phi = state_to_bloch([complex(r2), complex(r2)])
        self.assertAlmostEqual(theta, math.pi / 2, places=10)

    def test_schmidt_product_state(self):
        """Product state has one Schmidt coefficient = 1."""
        s = zero_state(2)
        coeffs = schmidt_coefficients(s, [0])
        self.assertAlmostEqual(coeffs[0], 1.0, delta=0.01)
        if len(coeffs) > 1:
            self.assertAlmostEqual(coeffs[1], 0.0, delta=0.01)

    def test_schmidt_bell_state(self):
        """Bell state has two equal Schmidt coefficients."""
        s = run_circuit(2, [('h', 0), ('cnot', 0, 1)])
        coeffs = schmidt_coefficients(s, [0])
        r2 = 1.0 / math.sqrt(2)
        self.assertAlmostEqual(coeffs[0], r2, delta=0.01)
        self.assertAlmostEqual(coeffs[1], r2, delta=0.01)


class TestClassicalExtraction(unittest.TestCase):
    """Classical extraction chain."""

    def test_extract_max_zero_state(self):
        """Most probable outcome of |00> is '00' with P=1."""
        s = zero_state(2)
        bs, p = extract_max_probability(s)
        self.assertEqual(bs, '00')
        self.assertAlmostEqual(p, 1.0, places=14)

    def test_extract_max_basis_state(self):
        """Most probable outcome of |11> is '11'."""
        s = basis_state(2, 3)
        bs, p = extract_max_probability(s)
        self.assertEqual(bs, '11')

    def test_extract_phase_real(self):
        """Phase of real positive amplitude is 0."""
        s = zero_state(1)
        self.assertAlmostEqual(extract_phase(s), 0.0, places=10)


class TestExampleAlgorithms(unittest.TestCase):
    """Example quantum algorithms."""

    def test_bell_state_example(self):
        """Bell state example produces entangled histogram."""
        random.seed(42)
        result = bell_state_example()
        self.assertIn('circuit', result)
        self.assertIn('histogram', result)
        # Only |00> and |11> should appear
        for key in result['histogram']:
            self.assertIn(key, ('00', '11'))

    def test_deutsch_jozsa_constant(self):
        """DJ correctly identifies constant oracle."""
        random.seed(42)
        result = deutsch_jozsa('constant', 3)
        self.assertEqual(result['answer'], 'constant')

    def test_deutsch_jozsa_balanced(self):
        """DJ correctly identifies balanced oracle."""
        random.seed(42)
        result = deutsch_jozsa('balanced', 3)
        self.assertEqual(result['answer'], 'balanced')

    def test_deutsch_jozsa_various_sizes(self):
        """DJ works for different numbers of qubits."""
        random.seed(42)
        for n in (2, 3, 4, 5):
            with self.subTest(n=n):
                result = deutsch_jozsa('balanced', n)
                self.assertEqual(result['answer'], 'balanced')

    def test_grover_2_qubits(self):
        """Grover finds marked item with 2 qubits."""
        random.seed(42)
        result = grover_search(2, 2)
        # With 2 qubits, Grover should find the item with very high prob
        self.assertEqual(result['answer'], 2)

    def test_grover_3_qubits(self):
        """Grover finds marked item with 3 qubits."""
        random.seed(42)
        result = grover_search(3, 5)
        self.assertEqual(result['answer'], 5)

    def test_teleportation(self):
        """Teleportation transfers qubit state."""
        random.seed(42)
        result = teleportation_example()
        self.assertIn('teleported_qubit_probs', result)
        # For default |+> state, P(0) ≈ P(1) ≈ 0.5
        p0 = result['teleported_qubit_probs']['0']
        p1 = result['teleported_qubit_probs']['1']
        self.assertAlmostEqual(p0 + p1, 1.0, delta=0.01)

    def test_teleportation_preserves_basis(self):
        """Teleportation of |0> gives P(0)=1."""
        random.seed(42)
        result = teleportation_example(alpha=1.0, beta=0.0)
        self.assertAlmostEqual(
            result['teleported_qubit_probs']['0'], 1.0, delta=0.01
        )

    def test_bernstein_vazirani(self):
        """BV recovers hidden bitstring."""
        random.seed(42)
        result = bernstein_vazirani('101')
        self.assertEqual(result['answer'], '101')

    def test_bernstein_vazirani_all_ones(self):
        """BV recovers all-ones string."""
        random.seed(42)
        result = bernstein_vazirani('1111')
        self.assertEqual(result['answer'], '1111')

    def test_bernstein_vazirani_single_bit(self):
        """BV works for single-bit string."""
        random.seed(42)
        result = bernstein_vazirani('1')
        self.assertEqual(result['answer'], '1')


class TestReports(unittest.TestCase):
    """Rule 9: reports."""

    def test_report_has_fields(self):
        """quantum_output_report has expected fields."""
        random.seed(42)
        r = quantum_output_report()
        self.assertIn('measurement_types', r)
        self.assertIn('example_algorithms', r)
        self.assertIn('bell_state_histogram', r)

    def test_full_report(self):
        """full_report returns dict with algorithm results."""
        random.seed(42)
        r = full_report()
        self.assertIsInstance(r, dict)
        self.assertIn('deutsch_jozsa_answer', r)
        self.assertIn('bernstein_vazirani_answer', r)


if __name__ == '__main__':
    unittest.main()
