"""
Tests for quantum_computing.py — state vectors, gates, qubit parameters, circuits.

Strategy:
  - Test state creation: zero_state normalization, basis_state orthogonality
  - Test product_state: tensor product matches manual computation
  - Test gate unitarity: every gate preserves normalization
  - Test gate identities: HXH=Z, HZH=X, X²=I, H²=I, S²=Z, T²=S
  - Test CNOT truth table: |00>→|00>, |10>→|11>, |11>→|10>
  - Test Bell state: H+CNOT gives (|00>+|11>)/√2
  - Test qubit parameters: transmon ~5 GHz for Al, spin scales with B
  - Test circuit runner: simple circuits match manual gate application
  - Test Toffoli: only flips target when both controls are |1>
  - Test Fredkin: controlled swap
  - Test σ-field wiring
  - Test Rule 9: reports

Reference values (MEASURED):
  Transmon (Al): ~4-8 GHz (typical)
  Spin qubit at 1T: ~28 GHz (Larmor)
  NV center: 2.87 GHz (zero-field splitting)
  GaAs effective mass: 0.067 m_e
"""

import math
import unittest

from sigma_ground.field.interface.quantum_computing import (
    zero_state,
    basis_state,
    product_state,
    state_norm,
    normalize,
    _n_qubits,
    apply_single_gate,
    apply_controlled_gate,
    gate_x,
    gate_y,
    gate_z,
    gate_h,
    gate_s,
    gate_t,
    gate_rx,
    gate_ry,
    gate_rz,
    gate_phase,
    gate_cnot,
    gate_cz,
    gate_swap,
    gate_iswap,
    gate_toffoli,
    gate_fredkin,
    run_circuit,
    supported_gates,
    transmon_frequency_GHz,
    spin_qubit_frequency_GHz,
    qd_qubit_frequency_GHz,
    nv_qubit_frequency_GHz,
    qubit_summary,
    sigma_adjusted_frequency,
    quantum_computing_report,
    full_report,
)


def _approx_eq(a, b, tol=1e-10):
    """Check two complex numbers are approximately equal."""
    return abs(a - b) < tol


def _states_close(s1, s2, tol=1e-10):
    """Check two state vectors are approximately equal (up to global phase)."""
    # Find first nonzero element to determine phase
    phase = None
    for a, b in zip(s1, s2):
        if abs(a) > tol and abs(b) > tol:
            phase = a / b
            break
    if phase is None:
        # Both might be zero or one is zero
        return all(abs(a) < tol and abs(b) < tol for a, b in zip(s1, s2))
    # Check all elements match up to this phase
    return all(abs(a - phase * b) < tol for a, b in zip(s1, s2))


class TestStateCreation(unittest.TestCase):
    """State vector creation and basic operations."""

    def test_zero_state_normalization(self):
        """zero_state is normalized."""
        for n in (1, 2, 3, 5):
            with self.subTest(n=n):
                s = zero_state(n)
                self.assertAlmostEqual(state_norm(s), 1.0, places=14)

    def test_zero_state_first_amplitude(self):
        """|00...0> has amplitude 1 at index 0."""
        s = zero_state(3)
        self.assertEqual(s[0], complex(1))
        for i in range(1, 8):
            self.assertEqual(s[i], complex(0))

    def test_basis_state_orthogonality(self):
        """Different basis states are orthogonal."""
        s0 = basis_state(2, 0)
        s1 = basis_state(2, 1)
        s2 = basis_state(2, 2)
        inner_01 = sum(a.conjugate() * b for a, b in zip(s0, s1))
        inner_02 = sum(a.conjugate() * b for a, b in zip(s0, s2))
        self.assertAlmostEqual(abs(inner_01), 0.0, places=14)
        self.assertAlmostEqual(abs(inner_02), 0.0, places=14)

    def test_basis_state_self_overlap(self):
        """<i|i> = 1."""
        for i in range(4):
            s = basis_state(2, i)
            inner = sum(a.conjugate() * a for a in s)
            self.assertAlmostEqual(inner.real, 1.0, places=14)

    def test_product_state_simple(self):
        """|+> ⊗ |0> = (|00> + |10>)/√2."""
        r2 = 1.0 / math.sqrt(2)
        s = product_state([[r2, r2], [1, 0]])
        self.assertAlmostEqual(abs(s[0]), r2, places=10)
        self.assertAlmostEqual(abs(s[1]), 0.0, places=10)
        self.assertAlmostEqual(abs(s[2]), r2, places=10)
        self.assertAlmostEqual(abs(s[3]), 0.0, places=10)

    def test_product_state_normalized(self):
        """Product of normalized states is normalized."""
        r2 = 1.0 / math.sqrt(2)
        s = product_state([[r2, r2], [r2, -r2], [1, 0]])
        self.assertAlmostEqual(state_norm(s), 1.0, places=14)

    def test_normalize_preserves_direction(self):
        """normalize scales without changing direction."""
        s = [complex(3), complex(4)]
        ns = normalize(s)
        self.assertAlmostEqual(state_norm(ns), 1.0, places=14)
        self.assertAlmostEqual(ns[0].real / ns[1].real, 3.0 / 4.0, places=10)

    def test_too_many_qubits_raises(self):
        """n_qubits > 25 raises ValueError."""
        with self.assertRaises(ValueError):
            zero_state(26)

    def test_basis_state_out_of_range_raises(self):
        """Out-of-range index raises ValueError."""
        with self.assertRaises(ValueError):
            basis_state(2, 4)


class TestSingleQubitGates(unittest.TestCase):
    """Single-qubit gate correctness and identities."""

    def test_x_gate_flips(self):
        """X|0> = |1>, X|1> = |0>."""
        s = gate_x(zero_state(1), 0)
        self.assertAlmostEqual(abs(s[1]), 1.0, places=14)
        s2 = gate_x(s, 0)
        self.assertAlmostEqual(abs(s2[0]), 1.0, places=14)

    def test_y_gate(self):
        """Y|0> = i|1>."""
        s = gate_y(zero_state(1), 0)
        self.assertAlmostEqual(s[1], 1j, places=14)

    def test_z_gate(self):
        """Z|0> = |0>, Z|1> = -|1>."""
        s0 = gate_z(zero_state(1), 0)
        self.assertAlmostEqual(s0[0], 1.0, places=14)
        s1 = gate_z(basis_state(1, 1), 0)
        self.assertAlmostEqual(s1[1], -1.0, places=14)

    def test_h_creates_superposition(self):
        """H|0> = (|0> + |1>)/√2."""
        s = gate_h(zero_state(1), 0)
        r2 = 1.0 / math.sqrt(2)
        self.assertAlmostEqual(s[0].real, r2, places=10)
        self.assertAlmostEqual(s[1].real, r2, places=10)

    def test_x_squared_is_identity(self):
        """X² = I."""
        s = zero_state(1)
        s = gate_x(gate_x(s, 0), 0)
        self.assertAlmostEqual(abs(s[0]), 1.0, places=14)

    def test_h_squared_is_identity(self):
        """H² = I."""
        s = zero_state(1)
        s = gate_h(gate_h(s, 0), 0)
        self.assertAlmostEqual(abs(s[0]), 1.0, places=14)

    def test_s_squared_is_z(self):
        """S² = Z."""
        s1 = basis_state(1, 1)
        ss = gate_s(gate_s(s1, 0), 0)
        sz = gate_z(s1, 0)
        for a, b in zip(ss, sz):
            self.assertAlmostEqual(a, b, places=14)

    def test_t_squared_is_s(self):
        """T² = S."""
        s1 = basis_state(1, 1)
        tt = gate_t(gate_t(s1, 0), 0)
        ss = gate_s(s1, 0)
        for a, b in zip(tt, ss):
            self.assertAlmostEqual(a, b, places=12)

    def test_hxh_equals_z(self):
        """HXH = Z."""
        s = basis_state(1, 1)
        hxh = gate_h(gate_x(gate_h(s, 0), 0), 0)
        z = gate_z(s, 0)
        for a, b in zip(hxh, z):
            self.assertAlmostEqual(a, b, places=12)

    def test_hzh_equals_x(self):
        """HZH = X."""
        s = basis_state(1, 1)
        hzh = gate_h(gate_z(gate_h(s, 0), 0), 0)
        x = gate_x(s, 0)
        for a, b in zip(hzh, x):
            self.assertAlmostEqual(a, b, places=12)

    def test_all_gates_preserve_norm(self):
        """Every single-qubit gate preserves normalization."""
        r2 = 1.0 / math.sqrt(2)
        s = product_state([[r2, r2]])  # |+>
        gates = [gate_x, gate_y, gate_z, gate_h, gate_s, gate_t]
        for g in gates:
            with self.subTest(gate=g.__name__):
                result = g(s, 0)
                self.assertAlmostEqual(state_norm(result), 1.0, places=13)

    def test_rotation_gates_preserve_norm(self):
        """Rotation gates preserve normalization."""
        s = product_state([[1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]])
        for theta in [0.0, 0.5, 1.0, math.pi, 2 * math.pi]:
            for g in [gate_rx, gate_ry, gate_rz]:
                with self.subTest(gate=g.__name__, theta=theta):
                    result = g(s, 0, theta)
                    self.assertAlmostEqual(state_norm(result), 1.0, places=13)

    def test_rx_pi_equals_x(self):
        """Rx(π) = -iX (same action up to global phase)."""
        s = zero_state(1)
        rx_pi = gate_rx(s, 0, math.pi)
        x_s = gate_x(s, 0)
        # |Rx(π)|0>| should match |X|0>| in magnitude
        self.assertAlmostEqual(abs(rx_pi[1]), abs(x_s[1]), places=12)

    def test_phase_gate(self):
        """Phase(π) on |1> gives -|1>."""
        s = basis_state(1, 1)
        result = gate_phase(s, 0, math.pi)
        self.assertAlmostEqual(result[1].real, -1.0, places=12)


class TestTwoQubitGates(unittest.TestCase):
    """Two-qubit gate correctness."""

    def test_cnot_truth_table(self):
        """CNOT truth table: |00>→|00>, |01>→|01>, |10>→|11>, |11>→|10>."""
        cases = [
            (0, [1, 0, 0, 0]),  # |00> → |00>
            (1, [0, 1, 0, 0]),  # |01> → |01>
            (2, [0, 0, 0, 1]),  # |10> → |11>
            (3, [0, 0, 1, 0]),  # |11> → |10>
        ]
        for idx, expected in cases:
            with self.subTest(input=format(idx, '02b')):
                s = basis_state(2, idx)
                result = gate_cnot(s, 0, 1)
                for i, e in enumerate(expected):
                    self.assertAlmostEqual(abs(result[i]), e, places=14)

    def test_bell_state(self):
        """H|0> then CNOT → (|00> + |11>)/√2."""
        s = gate_h(zero_state(2), 0)
        s = gate_cnot(s, 0, 1)
        r2 = 1.0 / math.sqrt(2)
        self.assertAlmostEqual(abs(s[0]), r2, places=10)
        self.assertAlmostEqual(abs(s[1]), 0.0, places=10)
        self.assertAlmostEqual(abs(s[2]), 0.0, places=10)
        self.assertAlmostEqual(abs(s[3]), r2, places=10)

    def test_cz_symmetric(self):
        """CZ is symmetric in control/target."""
        s = product_state([[1.0 / math.sqrt(2), 1.0 / math.sqrt(2)],
                          [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]])
        s1 = gate_cz(s, 0, 1)
        s2 = gate_cz(s, 1, 0)
        for a, b in zip(s1, s2):
            self.assertAlmostEqual(a, b, places=14)

    def test_swap_swaps(self):
        """SWAP|01> = |10>."""
        s = basis_state(2, 1)  # |01>
        result = gate_swap(s, 0, 1)
        self.assertAlmostEqual(abs(result[2]), 1.0, places=13)  # |10>

    def test_swap_preserves_norm(self):
        """SWAP preserves normalization."""
        r2 = 1.0 / math.sqrt(2)
        s = product_state([[r2, r2], [1, 0]])
        result = gate_swap(s, 0, 1)
        self.assertAlmostEqual(state_norm(result), 1.0, places=13)

    def test_iswap_basis_states(self):
        """iSWAP|01> = i|10>, iSWAP|10> = i|01>."""
        s01 = basis_state(2, 1)  # |01>
        r01 = gate_iswap(s01, 0, 1)
        self.assertAlmostEqual(abs(r01[2]), 1.0, places=13)  # goes to |10>

        s10 = basis_state(2, 2)  # |10>
        r10 = gate_iswap(s10, 0, 1)
        self.assertAlmostEqual(abs(r10[1]), 1.0, places=13)  # goes to |01>

    def test_cnot_preserves_norm(self):
        """CNOT preserves normalization on superposition."""
        s = gate_h(zero_state(2), 0)
        result = gate_cnot(s, 0, 1)
        self.assertAlmostEqual(state_norm(result), 1.0, places=13)


class TestThreeQubitGates(unittest.TestCase):
    """Three-qubit gate correctness."""

    def test_toffoli_only_flips_when_both_controls_set(self):
        """Toffoli flips target only when both controls are |1>."""
        # |110> → |111>
        s = basis_state(3, 6)  # 110
        result = gate_toffoli(s, 0, 1, 2)
        self.assertAlmostEqual(abs(result[7]), 1.0, places=14)  # 111

        # |100> → |100> (second control not set)
        s = basis_state(3, 4)  # 100
        result = gate_toffoli(s, 0, 1, 2)
        self.assertAlmostEqual(abs(result[4]), 1.0, places=14)  # unchanged

        # |010> → |010> (first control not set)
        s = basis_state(3, 2)  # 010
        result = gate_toffoli(s, 0, 1, 2)
        self.assertAlmostEqual(abs(result[2]), 1.0, places=14)  # unchanged

    def test_toffoli_preserves_norm(self):
        """Toffoli preserves normalization."""
        s = zero_state(3)
        s = gate_h(s, 0)
        s = gate_h(s, 1)
        result = gate_toffoli(s, 0, 1, 2)
        self.assertAlmostEqual(state_norm(result), 1.0, places=13)

    def test_fredkin_swaps_when_control_set(self):
        """Fredkin swaps q1,q2 when control is |1>."""
        # |110> → |101> (control=1, swap |1,0> → |0,1>)
        s = basis_state(3, 6)  # 110
        result = gate_fredkin(s, 0, 1, 2)
        self.assertAlmostEqual(abs(result[5]), 1.0, places=14)  # 101

    def test_fredkin_no_swap_when_control_clear(self):
        """Fredkin does nothing when control is |0>."""
        s = basis_state(3, 2)  # 010
        result = gate_fredkin(s, 0, 1, 2)
        self.assertAlmostEqual(abs(result[2]), 1.0, places=14)  # unchanged


class TestCircuitRunner(unittest.TestCase):
    """Circuit execution."""

    def test_bell_circuit(self):
        """H-CNOT circuit produces Bell state."""
        state = run_circuit(2, [('h', 0), ('cnot', 0, 1)])
        r2 = 1.0 / math.sqrt(2)
        self.assertAlmostEqual(abs(state[0]), r2, places=10)
        self.assertAlmostEqual(abs(state[3]), r2, places=10)

    def test_empty_circuit(self):
        """Empty circuit returns |00...0>."""
        state = run_circuit(3, [])
        self.assertAlmostEqual(abs(state[0]), 1.0, places=14)

    def test_circuit_matches_manual(self):
        """Circuit runner matches manual gate application."""
        # X on qubit 0, then H on qubit 1
        manual = gate_x(zero_state(2), 0)
        manual = gate_h(manual, 1)

        circuit_result = run_circuit(2, [('x', 0), ('h', 1)])

        for a, b in zip(manual, circuit_result):
            self.assertAlmostEqual(a, b, places=14)

    def test_unknown_gate_raises(self):
        """Unknown gate name raises ValueError."""
        with self.assertRaises(ValueError):
            run_circuit(1, [('banana', 0)])

    def test_supported_gates_nonempty(self):
        """supported_gates returns a non-empty list."""
        gates = supported_gates()
        self.assertGreater(len(gates), 10)
        self.assertIn('h', gates)
        self.assertIn('cnot', gates)

    def test_circuit_with_rotation(self):
        """Circuit with rotation gate works."""
        state = run_circuit(1, [('rx', 0, math.pi)])
        # Rx(π)|0> = -i|1>
        self.assertAlmostEqual(abs(state[1]), 1.0, places=12)

    def test_circuit_with_initial_state(self):
        """Circuit runner accepts initial_state."""
        init = basis_state(2, 3)  # |11>
        state = run_circuit(2, [('cnot', 0, 1)], initial_state=init)
        # CNOT|11> = |10>
        self.assertAlmostEqual(abs(state[2]), 1.0, places=14)


class TestQubitParameters(unittest.TestCase):
    """Qubit parameter derivation from cascade physics."""

    def test_transmon_frequency_range(self):
        """Transmon frequency for Al is in 4-8 GHz range (MEASURED typical)."""
        freq = transmon_frequency_GHz()
        self.assertGreater(freq, 3.0)
        self.assertLess(freq, 10.0)

    def test_spin_qubit_frequency_1T(self):
        """Spin qubit at 1T: ~28 GHz (Larmor frequency)."""
        freq = spin_qubit_frequency_GHz(1.0)
        self.assertAlmostEqual(freq, 28.0, delta=1.0)

    def test_spin_qubit_scales_with_B(self):
        """Spin qubit frequency scales linearly with B."""
        f1 = spin_qubit_frequency_GHz(1.0)
        f2 = spin_qubit_frequency_GHz(2.0)
        self.assertAlmostEqual(f2 / f1, 2.0, delta=0.001)

    def test_nv_center_frequency(self):
        """NV center frequency is 2.87 GHz (MEASURED)."""
        self.assertAlmostEqual(nv_qubit_frequency_GHz(), 2.87, places=2)

    def test_qd_qubit_positive(self):
        """QD qubit frequency is positive and finite."""
        freq = qd_qubit_frequency_GHz(5e-9, 'GaAs')
        self.assertGreater(freq, 0)
        self.assertTrue(math.isfinite(freq))

    def test_qd_qubit_scales_with_size(self):
        """Smaller dot → higher frequency (stronger confinement)."""
        f_small = qd_qubit_frequency_GHz(3e-9)
        f_large = qd_qubit_frequency_GHz(10e-9)
        self.assertGreater(f_small, f_large)

    def test_qubit_summary_transmon(self):
        """qubit_summary returns dict with expected keys."""
        s = qubit_summary('transmon')
        self.assertIn('frequency_GHz', s)
        self.assertIn('T1_estimate_us', s)
        self.assertIn('qubit_type', s)
        self.assertEqual(s['qubit_type'], 'transmon')

    def test_qubit_summary_spin(self):
        """qubit_summary works for spin qubits."""
        s = qubit_summary('spin', B_tesla=0.5)
        self.assertGreater(s['frequency_GHz'], 0)

    def test_qubit_summary_unknown_raises(self):
        """Unknown qubit type raises ValueError."""
        with self.assertRaises(ValueError):
            qubit_summary('flux_capacitor')


class TestSigmaWiring(unittest.TestCase):
    """σ-field wiring."""

    def test_sigma_zero_no_change(self):
        """σ=0 leaves frequency unchanged."""
        self.assertAlmostEqual(
            sigma_adjusted_frequency(5.0, 0.0), 5.0, places=10
        )

    def test_sigma_positive_reduces_frequency(self):
        """σ>0 reduces frequency (compression shifts energy down)."""
        self.assertLess(
            sigma_adjusted_frequency(5.0, 1.0), 5.0
        )


class TestReports(unittest.TestCase):
    """Rule 9: reports."""

    def test_report_has_required_fields(self):
        """quantum_computing_report has all expected fields."""
        r = quantum_computing_report()
        required = ['supported_gates', 'n_gates', 'qubit_types',
                     'transmon_frequency_GHz', 'max_qubits_recommended']
        for field in required:
            with self.subTest(field=field):
                self.assertIn(field, r)

    def test_full_report(self):
        """full_report returns dict with bell_state."""
        r = full_report()
        self.assertIsInstance(r, dict)
        self.assertIn('bell_state', r)
        self.assertIn('gate_aliases', r)


if __name__ == '__main__':
    unittest.main()
