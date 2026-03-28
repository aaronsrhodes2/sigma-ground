"""
Quantum Computing Simulator — qubit parameters from cascade physics.

Derives qubit operating frequencies and parameters from the existing sigma-ground
physics cascade (BCS superconductivity, Zeeman splitting, quantum confinement).
Provides a decoherence-free state-vector simulator: our advantage is that the
simulation is exact (no noise channels), while physical parameters are grounded
in measured constants and first-principles derivations.

State representation:
    State vectors are plain Python lists of complex numbers, length 2^n for
    n qubits.  |00...0> is state[0] = 1, all others 0.  Qubit ordering is
    MSB-first: qubit 0 is the most significant bit.

Gate engine:
    Bit-manipulation indexing over the state vector.  Single-qubit, controlled,
    and doubly-controlled gates are supported.  All standard gates (Pauli, H, S,
    T, rotations, CNOT, CZ, SWAP, iSWAP, Toffoli, Fredkin) are included.

Qubit parameter derivation:
    - Transmon: BCS gap -> Josephson energy -> transmon frequency
    - Spin qubit: Zeeman splitting -> Larmor frequency
    - Quantum dot: confinement energy -> level spacing -> frequency
    - NV center: zero-field splitting (MEASURED)

Pure Python, zero external dependencies.
"""

import cmath
import math

from ..constants import (
    HBAR, E_CHARGE, K_B, C, M_ELECTRON_KG, MU_BOHR,
    EV_TO_J, H_PLANCK, ALPHA, EPS_0,
)

# ---------------------------------------------------------------------------
#  State Representation
# ---------------------------------------------------------------------------


def zero_state(n_qubits):
    """Return |00...0> as a list of 2^n complex numbers.

    The all-zero computational basis state.  Entry 0 has amplitude 1,
    all others have amplitude 0.

    Args:
        n_qubits: number of qubits (1..25).

    Returns:
        List of complex numbers of length 2^n_qubits.

    Raises:
        ValueError: if n_qubits > 25 (would require >512 MB).
    """
    if n_qubits > 25:
        raise ValueError("n_qubits > 25 requires >512MB; use n_qubits <= 25")
    dim = 1 << n_qubits
    state = [complex(0)] * dim
    state[0] = complex(1)
    return state


def basis_state(n_qubits, index):
    """Return |index> as a list of 2^n complex numbers.

    A single computational basis state with amplitude 1 at position *index*.

    Args:
        n_qubits: number of qubits.
        index: integer label of the basis state (0 .. 2^n - 1).

    Returns:
        List of complex numbers of length 2^n_qubits.

    Raises:
        ValueError: if index is out of range.
    """
    dim = 1 << n_qubits
    if index < 0 or index >= dim:
        raise ValueError(f"index {index} out of range for {n_qubits} qubits")
    state = [complex(0)] * dim
    state[index] = complex(1)
    return state


def product_state(single_qubit_states):
    """Tensor product of single-qubit states.

    Given a list of single-qubit states, each [alpha, beta], compute the
    full multi-qubit state via the Kronecker product.

    Args:
        single_qubit_states: list of 2-element lists [alpha, beta].

    Returns:
        State vector as list of complex numbers.
    """
    state = [complex(1)]
    for sq in single_qubit_states:
        new_state = []
        for amp in state:
            for s in sq:
                new_state.append(amp * s)
        state = new_state
    return state


def state_norm(state):
    """Return the squared norm sum |alpha_i|^2.

    For a valid quantum state this should equal 1.0.

    Args:
        state: list of complex amplitudes.

    Returns:
        Float, sum of squared magnitudes.
    """
    return sum(abs(a) ** 2 for a in state)


def normalize(state):
    """Return a normalized copy of the state vector.

    Args:
        state: list of complex amplitudes.

    Returns:
        New list with the same direction but unit norm.

    Raises:
        ValueError: if the state is the zero vector.
    """
    n = state_norm(state) ** 0.5
    if n == 0:
        raise ValueError("Cannot normalize zero vector")
    return [a / n for a in state]


# ---------------------------------------------------------------------------
#  Gate Engine — bit-manipulation indexing, qubit 0 = MSB
# ---------------------------------------------------------------------------


def _n_qubits(state):
    """Infer number of qubits from state vector length.

    Args:
        state: list of complex amplitudes (length must be a power of 2).

    Returns:
        Integer number of qubits.

    Raises:
        ValueError: if length is not a power of 2.
    """
    dim = len(state)
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError(f"State length {dim} is not a power of 2")
    return n


def apply_single_gate(state, qubit, matrix):
    """Apply a 2x2 unitary matrix to the specified qubit.

    Uses bit-manipulation to iterate over amplitude pairs that differ
    only in the target qubit.

    Args:
        state: list of complex amplitudes.
        qubit: target qubit index (0 = MSB).
        matrix: 2x2 list-of-lists [[a, b], [c, d]].

    Returns:
        New state vector after gate application.

    Raises:
        ValueError: if qubit index is out of range.
    """
    n = _n_qubits(state)
    if qubit < 0 or qubit >= n:
        raise ValueError(f"qubit {qubit} out of range for {n}-qubit state")
    new_state = list(state)
    bit = n - 1 - qubit  # bit position (MSB convention)
    for i in range(len(state)):
        if i & (1 << bit):  # bit is set -> |1> component; skip (process from |0> side)
            continue
        j = i | (1 << bit)  # j has bit set (|1>), i has bit cleared (|0>)
        a, b = state[i], state[j]
        new_state[i] = matrix[0][0] * a + matrix[0][1] * b
        new_state[j] = matrix[1][0] * a + matrix[1][1] * b
    return new_state


def apply_controlled_gate(state, control, target, matrix):
    """Apply a 2x2 matrix to the target qubit, conditioned on control = |1>.

    Only amplitude pairs where the control qubit is in |1> are modified.

    Args:
        state: list of complex amplitudes.
        control: control qubit index.
        target: target qubit index.
        matrix: 2x2 list-of-lists.

    Returns:
        New state vector.
    """
    n = _n_qubits(state)
    new_state = list(state)
    ctrl_bit = n - 1 - control
    tgt_bit = n - 1 - target
    for i in range(len(state)):
        if not (i & (1 << ctrl_bit)):  # control not set -> skip
            continue
        if i & (1 << tgt_bit):  # only process from |0> side of target
            continue
        j = i | (1 << tgt_bit)
        a, b = state[i], state[j]
        new_state[i] = matrix[0][0] * a + matrix[0][1] * b
        new_state[j] = matrix[1][0] * a + matrix[1][1] * b
    return new_state


def apply_doubly_controlled_gate(state, ctrl1, ctrl2, target, matrix):
    """Apply a 2x2 matrix to the target when both control qubits are |1>.

    Args:
        state: list of complex amplitudes.
        ctrl1: first control qubit index.
        ctrl2: second control qubit index.
        target: target qubit index.
        matrix: 2x2 list-of-lists.

    Returns:
        New state vector.
    """
    n = _n_qubits(state)
    new_state = list(state)
    c1_bit = n - 1 - ctrl1
    c2_bit = n - 1 - ctrl2
    tgt_bit = n - 1 - target
    for i in range(len(state)):
        if not (i & (1 << c1_bit) and i & (1 << c2_bit)):
            continue
        if i & (1 << tgt_bit):
            continue
        j = i | (1 << tgt_bit)
        a, b = state[i], state[j]
        new_state[i] = matrix[0][0] * a + matrix[0][1] * b
        new_state[j] = matrix[1][0] * a + matrix[1][1] * b
    return new_state


# ---------------------------------------------------------------------------
#  Gate Definitions
# ---------------------------------------------------------------------------

_SQRT2_INV = 1.0 / math.sqrt(2.0)

# Pauli matrices
_X = [[0, 1], [1, 0]]
_Y = [[0, -1j], [1j, 0]]
_Z = [[1, 0], [0, -1]]

# Hadamard
_H = [[_SQRT2_INV, _SQRT2_INV], [_SQRT2_INV, -_SQRT2_INV]]

# Phase gates
_S = [[1, 0], [0, 1j]]
_T = [[1, 0], [0, cmath.exp(1j * math.pi / 4)]]


def gate_x(state, qubit):
    """Pauli-X (bit flip) gate.

    |0> -> |1>,  |1> -> |0>.

    Args:
        state: state vector.
        qubit: target qubit index.

    Returns:
        New state vector.
    """
    return apply_single_gate(state, qubit, _X)


def gate_y(state, qubit):
    """Pauli-Y gate.

    |0> -> i|1>,  |1> -> -i|0>.

    Args:
        state: state vector.
        qubit: target qubit index.

    Returns:
        New state vector.
    """
    return apply_single_gate(state, qubit, _Y)


def gate_z(state, qubit):
    """Pauli-Z (phase flip) gate.

    |0> -> |0>,  |1> -> -|1>.

    Args:
        state: state vector.
        qubit: target qubit index.

    Returns:
        New state vector.
    """
    return apply_single_gate(state, qubit, _Z)


def gate_h(state, qubit):
    """Hadamard gate.

    |0> -> (|0> + |1>)/sqrt(2),  |1> -> (|0> - |1>)/sqrt(2).

    Args:
        state: state vector.
        qubit: target qubit index.

    Returns:
        New state vector.
    """
    return apply_single_gate(state, qubit, _H)


def gate_s(state, qubit):
    """S (phase) gate.

    |0> -> |0>,  |1> -> i|1>.  Equivalent to Z^(1/2).

    Args:
        state: state vector.
        qubit: target qubit index.

    Returns:
        New state vector.
    """
    return apply_single_gate(state, qubit, _S)


def gate_t(state, qubit):
    """T gate.

    |0> -> |0>,  |1> -> e^(i*pi/4)|1>.  Equivalent to Z^(1/4).

    Args:
        state: state vector.
        qubit: target qubit index.

    Returns:
        New state vector.
    """
    return apply_single_gate(state, qubit, _T)


def gate_rx(state, qubit, theta):
    """Rotation about X axis by angle theta.

    R_x(theta) = cos(theta/2) I - i sin(theta/2) X.

    Args:
        state: state vector.
        qubit: target qubit index.
        theta: rotation angle in radians.

    Returns:
        New state vector.
    """
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    m = [[c, -1j * s], [-1j * s, c]]
    return apply_single_gate(state, qubit, m)


def gate_ry(state, qubit, theta):
    """Rotation about Y axis by angle theta.

    R_y(theta) = cos(theta/2) I - i sin(theta/2) Y.

    Args:
        state: state vector.
        qubit: target qubit index.
        theta: rotation angle in radians.

    Returns:
        New state vector.
    """
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    m = [[c, -s], [s, c]]
    return apply_single_gate(state, qubit, m)


def gate_rz(state, qubit, theta):
    """Rotation about Z axis by angle theta.

    R_z(theta) = diag(e^(-i*theta/2), e^(i*theta/2)).

    Args:
        state: state vector.
        qubit: target qubit index.
        theta: rotation angle in radians.

    Returns:
        New state vector.
    """
    m = [[cmath.exp(-1j * theta / 2), 0], [0, cmath.exp(1j * theta / 2)]]
    return apply_single_gate(state, qubit, m)


def gate_phase(state, qubit, phi):
    """General phase gate P(phi).

    |0> -> |0>,  |1> -> e^(i*phi)|1>.

    Args:
        state: state vector.
        qubit: target qubit index.
        phi: phase angle in radians.

    Returns:
        New state vector.
    """
    m = [[1, 0], [0, cmath.exp(1j * phi)]]
    return apply_single_gate(state, qubit, m)


# ---- Two-qubit gates ----


def gate_cnot(state, control, target):
    """Controlled-NOT (CNOT / CX) gate.

    Flips target qubit when control qubit is |1>.

    Args:
        state: state vector.
        control: control qubit index.
        target: target qubit index.

    Returns:
        New state vector.
    """
    return apply_controlled_gate(state, control, target, _X)


def gate_cz(state, control, target):
    """Controlled-Z (CZ) gate.

    Applies Z to target when control is |1>.  Symmetric in control/target.

    Args:
        state: state vector.
        control: control qubit index.
        target: target qubit index.

    Returns:
        New state vector.
    """
    return apply_controlled_gate(state, control, target, _Z)


def gate_swap(state, q1, q2):
    """SWAP gate — exchanges the states of two qubits.

    Implemented as three CNOTs: CNOT(q1,q2) CNOT(q2,q1) CNOT(q1,q2).

    Args:
        state: state vector.
        q1: first qubit index.
        q2: second qubit index.

    Returns:
        New state vector.
    """
    state = gate_cnot(state, q1, q2)
    state = gate_cnot(state, q2, q1)
    state = gate_cnot(state, q1, q2)
    return state


def gate_iswap(state, q1, q2):
    """iSWAP gate — SWAP with i phase on swapped components.

    |00> -> |00>,  |01> -> i|10>,  |10> -> i|01>,  |11> -> |11>.
    Native gate for many superconducting qubit architectures.

    Args:
        state: state vector.
        q1: first qubit index.
        q2: second qubit index.

    Returns:
        New state vector.
    """
    n = _n_qubits(state)
    new_state = list(state)
    b1 = n - 1 - q1
    b2 = n - 1 - q2
    for i in range(len(state)):
        v1 = (i >> b1) & 1
        v2 = (i >> b2) & 1
        if v1 == 0 and v2 == 1:
            j = (i | (1 << b1)) & ~(1 << b2)  # swap: 01 -> 10
            new_state[i] = 1j * state[j]
            new_state[j] = 1j * state[i]
        # |00> -> |00> and |11> -> |11>: no change
        # v1==1, v2==0 already handled by v1==0, v2==1 case
    return new_state


# ---- Three-qubit gates ----


def gate_toffoli(state, ctrl1, ctrl2, target):
    """Toffoli (CCX) gate — controlled-controlled-NOT.

    Flips target when both ctrl1 and ctrl2 are |1>.

    Args:
        state: state vector.
        ctrl1: first control qubit index.
        ctrl2: second control qubit index.
        target: target qubit index.

    Returns:
        New state vector.
    """
    return apply_doubly_controlled_gate(state, ctrl1, ctrl2, target, _X)


def gate_fredkin(state, ctrl, q1, q2):
    """Fredkin (CSWAP) gate — controlled-SWAP.

    Swaps q1 and q2 when ctrl is |1>.

    Args:
        state: state vector.
        ctrl: control qubit index.
        q1: first swap qubit index.
        q2: second swap qubit index.

    Returns:
        New state vector.
    """
    n = _n_qubits(state)
    new_state = list(state)
    cb = n - 1 - ctrl
    b1 = n - 1 - q1
    b2 = n - 1 - q2
    for i in range(len(state)):
        if not (i & (1 << cb)):
            continue
        v1 = (i >> b1) & 1
        v2 = (i >> b2) & 1
        if v1 != v2:
            j = i ^ (1 << b1) ^ (1 << b2)
            if i < j:  # process each pair once
                new_state[i], new_state[j] = state[j], state[i]
    return new_state


# ---------------------------------------------------------------------------
#  Circuit Runner
# ---------------------------------------------------------------------------

_GATE_DISPATCH = {
    'x': gate_x, 'y': gate_y, 'z': gate_z,
    'h': gate_h, 's': gate_s, 't': gate_t,
    'rx': gate_rx, 'ry': gate_ry, 'rz': gate_rz,
    'phase': gate_phase,
    'cnot': gate_cnot, 'cx': gate_cnot,
    'cz': gate_cz, 'swap': gate_swap, 'iswap': gate_iswap,
    'toffoli': gate_toffoli, 'ccx': gate_toffoli,
    'fredkin': gate_fredkin, 'cswap': gate_fredkin,
}


def run_circuit(n_qubits, circuit, initial_state=None):
    """Execute a quantum circuit on a state-vector simulator.

    The circuit is a list of instruction tuples.  Each tuple starts with
    the gate name (string) followed by qubit indices and optional parameters.

    Examples::

        [('h', 0), ('cnot', 0, 1)]             # Bell state
        [('rx', 0, 0.5), ('rz', 1, 1.2)]       # rotations
        [('toffoli', 0, 1, 2)]                  # three-qubit gate

    Args:
        n_qubits: number of qubits in the circuit.
        circuit: list of tuples (gate_name, qubit, ..., [angle]).
        initial_state: optional starting state (default: |00...0>).

    Returns:
        Final state vector as list of complex numbers.

    Raises:
        ValueError: if an unknown gate name is encountered.
    """
    state = initial_state if initial_state is not None else zero_state(n_qubits)
    for instruction in circuit:
        gate_name = instruction[0]
        args = instruction[1:]
        if gate_name not in _GATE_DISPATCH:
            raise ValueError(f"Unknown gate: {gate_name}")
        state = _GATE_DISPATCH[gate_name](state, *args)
    return state


def supported_gates():
    """Return sorted list of supported gate names (including aliases).

    Returns:
        List of strings.
    """
    return sorted(set(_GATE_DISPATCH.keys()))


# ---------------------------------------------------------------------------
#  Qubit Parameter Derivation — from cascade physics
# ---------------------------------------------------------------------------


def transmon_frequency_GHz(material_key='aluminum', sigma=0.0):
    """Transmon qubit frequency derived from the BCS gap.

    Derivation chain:
        T_c  (from SUPERCONDUCTORS table, MEASURED)
        -> Delta(0) = 1.764 k_B T_c  (BCS, FIRST_PRINCIPLES)
        -> E_J = Delta(0)/2  (single-junction Ambegaokar-Baratoff)
        -> omega_01 = sqrt(8 E_J E_C) - E_C  (transmon regime)
        -> frequency in GHz

    E_C ~ 250 MHz is a MEASURED typical charging energy for transmon qubits.

    Args:
        material_key: key into SUPERCONDUCTORS dict (default: 'aluminum').
        sigma: sigma-field value for frequency adjustment (default: 0.0).

    Returns:
        Transmon qubit frequency in GHz.

    Raises:
        KeyError: if material_key is not in SUPERCONDUCTORS.
    """
    from .superconductivity import bcs_gap_zero, SUPERCONDUCTORS

    if material_key not in SUPERCONDUCTORS:
        raise KeyError(f"Unknown superconductor: {material_key}")

    T_c = SUPERCONDUCTORS[material_key]['T_c_K']
    gap_J = bcs_gap_zero(T_c)   # Delta(0) in Joules
    gap_eV = gap_J / EV_TO_J    # convert to eV

    # Josephson energy: E_J = Delta(0)/2 for a single tunnel junction
    # (Ambegaokar-Baratoff relation: I_c = pi*Delta/(2eR_n),
    #  E_J = hbar*I_c/(2e) = pi*Delta/(4e^2 R_n))
    # For typical transmon: E_J/E_C ~ 50, E_C ~ 0.25 GHz ~ 1.03e-6 eV
    E_C_eV = 1.03e-6   # ~250 MHz charging energy (MEASURED typical for transmon)
    E_J_eV = gap_eV / 2  # single junction approximation

    # Transmon frequency: omega_01 ~ sqrt(8 E_J E_C) - E_C
    omega_eV = math.sqrt(8 * E_J_eV * E_C_eV) - E_C_eV

    # Convert eV to GHz: E = hf -> f = E/h
    freq_Hz = omega_eV * EV_TO_J / H_PLANCK
    freq_GHz = freq_Hz / 1e9

    if sigma != 0.0:
        freq_GHz = sigma_adjusted_frequency(freq_GHz, sigma)

    return freq_GHz


def spin_qubit_frequency_GHz(B_tesla, g_factor=2.0023):
    """Spin qubit (Larmor) frequency from Zeeman splitting.

    Derivation chain:
        omega_L = g mu_B B / hbar  (FIRST_PRINCIPLES, Zeeman)
        f = g mu_B B / h

    Args:
        B_tesla: external magnetic field strength in Tesla.
        g_factor: electron g-factor (default: free electron, 2.0023).

    Returns:
        Spin qubit frequency in GHz.
    """
    freq_Hz = g_factor * MU_BOHR * B_tesla / H_PLANCK
    return freq_Hz / 1e9


def qd_qubit_frequency_GHz(radius_m, material='GaAs'):
    """Quantum dot qubit frequency from confinement level spacing.

    Derivation chain:
        E_n = n^2 pi^2 hbar^2 / (2 m* L^2)  (particle-in-box, FIRST_PRINCIPLES)
        -> delta_E = E_2 - E_1
        -> f = delta_E / h

    Approximates the dot as a 1D box with L = 2R (diameter).

    Args:
        radius_m: quantum dot radius in metres.
        material: semiconductor material key. One of 'GaAs', 'InAs', 'Si', 'Ge'.
                  Default: 'GaAs'.

    Returns:
        Quantum dot qubit frequency in GHz.
    """
    from .quantum_wells import box_energy_1d_eV

    # Effective masses (MEASURED, from Vurgaftman et al. / Kittel)
    _QD_MASSES = {
        'GaAs': 0.067 * M_ELECTRON_KG,
        'InAs': 0.023 * M_ELECTRON_KG,
        'Si':   0.26 * M_ELECTRON_KG,    # transverse effective mass
        'Ge':   0.082 * M_ELECTRON_KG,   # light hole
    }
    mass = _QD_MASSES.get(material, 0.067 * M_ELECTRON_KG)

    # Approximate as 1D box with L = 2R (diameter)
    L = 2 * radius_m
    E1 = box_energy_1d_eV(1, L, mass)
    E2 = box_energy_1d_eV(2, L, mass)
    delta_E_eV = E2 - E1

    freq_Hz = delta_E_eV * EV_TO_J / H_PLANCK
    return freq_Hz / 1e9


# NV center zero-field splitting — MEASURED (Doherty et al. 2013)
_NV_ZFS_GHZ = 2.87  # GHz


def nv_qubit_frequency_GHz():
    """NV center qubit frequency from zero-field splitting.

    The nitrogen-vacancy center ground state (^3 A_2) splits into m_s = 0
    and m_s = +/-1 sublevels separated by D = 2.87 GHz (MEASURED,
    Doherty et al. 2013).

    Returns:
        NV center qubit frequency in GHz (2.87).
    """
    return _NV_ZFS_GHZ


def qubit_summary(qubit_type, **kwargs):
    """Summary of qubit parameters for a given type.

    Computes the operating frequency from cascade physics and augments
    with typical MEASURED coherence times and gate fidelities.

    Args:
        qubit_type: one of 'transmon', 'spin', 'quantum_dot', 'nv_center'.
        **kwargs: passed to the relevant frequency function.  E.g.
                  material_key='aluminum' for transmon, B_tesla=1.0 for spin,
                  radius_m=5e-9 for quantum_dot.

    Returns:
        Dict with keys: qubit_type, frequency_GHz, T1_estimate_us,
        T2_estimate_us, gate_fidelity_estimate, note.

    Raises:
        ValueError: if qubit_type is unknown.
    """
    # MEASURED typical coherence times and fidelities
    _QUBIT_PROPS = {
        'transmon':    {'T1_us': 100.0,  'T2_us': 50.0,   'fidelity': 0.999},
        'spin':        {'T1_us': 1000.0, 'T2_us': 200.0,  'fidelity': 0.998},
        'quantum_dot': {'T1_us': 10.0,   'T2_us': 1.0,    'fidelity': 0.99},
        'nv_center':   {'T1_us': 5000.0, 'T2_us': 2000.0, 'fidelity': 0.995},
    }

    _FREQ_FN = {
        'transmon':    transmon_frequency_GHz,
        'spin':        spin_qubit_frequency_GHz,
        'quantum_dot': qd_qubit_frequency_GHz,
        'nv_center':   nv_qubit_frequency_GHz,
    }

    if qubit_type not in _FREQ_FN:
        raise ValueError(
            f"Unknown qubit type: {qubit_type}. "
            f"Use: {list(_FREQ_FN.keys())}"
        )

    freq = _FREQ_FN[qubit_type](**kwargs)
    props = _QUBIT_PROPS[qubit_type]

    return {
        'qubit_type': qubit_type,
        'frequency_GHz': freq,
        'T1_estimate_us': props['T1_us'],
        'T2_estimate_us': props['T2_us'],
        'gate_fidelity_estimate': props['fidelity'],
        'note': 'T1/T2/fidelity are typical MEASURED values; '
                'simulation is decoherence-free',
    }


# ---------------------------------------------------------------------------
#  sigma-Field Wiring
# ---------------------------------------------------------------------------


def sigma_adjusted_frequency(base_freq_GHz, sigma):
    """Adjust qubit frequency for the sigma-field.

    Energy scales shift as e^(-sigma) for QCD-origin energies.
    EM-origin energies (Zeeman, Coulomb) are sigma-invariant.
    BCS gap (phonon-mediated) inherits material property shifts.

    Args:
        base_freq_GHz: qubit frequency at sigma = 0 (GHz).
        sigma: sigma-field value.

    Returns:
        Adjusted frequency in GHz.
    """
    return base_freq_GHz * math.exp(-sigma)


# ---------------------------------------------------------------------------
#  Reports (Rule 9 — complete coverage)
# ---------------------------------------------------------------------------


def quantum_computing_report():
    """Standard report for the quantum computing module.

    Returns a dict with supported gates, qubit types, representative
    frequencies, and simulator capabilities.

    Returns:
        Dict of module summary information.
    """
    return {
        'supported_gates': supported_gates(),
        'n_gates': len(set(_GATE_DISPATCH.values())),
        'qubit_types': ['transmon', 'spin', 'quantum_dot', 'nv_center'],
        'transmon_frequency_GHz': transmon_frequency_GHz(),
        'spin_qubit_1T_GHz': spin_qubit_frequency_GHz(1.0),
        'nv_center_frequency_GHz': nv_qubit_frequency_GHz(),
        'max_qubits_recommended': 25,
        'state_vector_type': 'list[complex]',
        'decoherence': 'none (perfect simulation)',
    }


def full_report():
    """Extended report with all qubit types and gate info.

    Includes a quantum-dot frequency, a Bell-state generation example,
    and gate alias information.

    Returns:
        Dict of extended module summary information.
    """
    r = quantum_computing_report()
    r['qd_qubit_5nm_GHz'] = qd_qubit_frequency_GHz(5e-9)
    r['bell_state'] = run_circuit(2, [('h', 0), ('cnot', 0, 1)])
    r['gate_aliases'] = {'cx': 'cnot', 'ccx': 'toffoli', 'cswap': 'fredkin'}
    return r
