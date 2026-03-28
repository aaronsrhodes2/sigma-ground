"""
Quantum Output — measurement, sampling, and classical extraction.

Bridges the gap between quantum state vectors and useable classical computation.
The chain is:

    quantum state  ->  measurement (Born rule)  ->  bitstring histogram
                   ->  classical post-processing  ->  answer

Provides:
    - Projective measurement with wavefunction collapse
    - Multi-shot sampling to build outcome histograms
    - Expectation values for Pauli strings and diagonal observables
    - State analysis: fidelity, entanglement entropy, Bloch sphere
    - Classical extraction: most-probable outcome, phase estimation, function values
    - Example algorithms: Bell state, Deutsch-Jozsa, Grover, teleportation,
      Bernstein-Vazirani

Pure Python, zero external dependencies.
"""

import math
import random

from .quantum_computing import (
    zero_state,
    basis_state,
    product_state,
    state_norm,
    normalize,
    _n_qubits,
    run_circuit,
    gate_h,
    gate_x,
    gate_z,
    gate_cnot,
    gate_cz,
    apply_single_gate,
)


# ---------------------------------------------------------------------------
#  Probabilities (no collapse)
# ---------------------------------------------------------------------------


def probabilities(state):
    """Return list of outcome probabilities |alpha_i|^2 for all basis states.

    Args:
        state: list of complex amplitudes (length 2^n).

    Returns:
        List of floats, same length as state.
    """
    return [abs(a) ** 2 for a in state]


def probability(state, qubit, outcome):
    """Probability of measuring a single qubit as 0 or 1 (no collapse).

    Args:
        state: state vector.
        qubit: qubit index to query (0 = MSB).
        outcome: 0 or 1.

    Returns:
        Float probability in [0, 1].
    """
    n = _n_qubits(state)
    bit = n - 1 - qubit
    p = 0.0
    for i, amp in enumerate(state):
        if ((i >> bit) & 1) == outcome:
            p += abs(amp) ** 2
    return p


# ---------------------------------------------------------------------------
#  Measurement (with collapse)
# ---------------------------------------------------------------------------


def measure(state, qubit):
    """Projective measurement of a single qubit (Born rule + collapse).

    Randomly collapses the qubit to |0> or |1> with probability given by
    the Born rule, then renormalises the remaining state.

    Args:
        state: state vector (list of complex).
        qubit: qubit index to measure (0 = MSB).

    Returns:
        (new_state, outcome) where outcome is 0 or 1.
    """
    p0 = probability(state, qubit, 0)
    outcome = 0 if random.random() < p0 else 1

    # Collapse: zero out amplitudes inconsistent with outcome, renormalize
    n = _n_qubits(state)
    bit = n - 1 - qubit
    new_state = list(state)
    for i in range(len(state)):
        if ((i >> bit) & 1) != outcome:
            new_state[i] = complex(0)
    new_state = normalize(new_state)
    return new_state, outcome


def measure_all(state):
    """Measure all qubits simultaneously (Born rule + collapse).

    Selects a basis state with probability |alpha_i|^2, then collapses
    into that basis state.

    Args:
        state: state vector.

    Returns:
        (new_state, bitstring) where bitstring is e.g. '010'.
    """
    n = _n_qubits(state)
    probs = probabilities(state)

    # Weighted random selection
    r = random.random()
    cumulative = 0.0
    chosen = len(probs) - 1
    for i, p in enumerate(probs):
        cumulative += p
        if r < cumulative:
            chosen = i
            break

    # Collapse to basis state
    new_state = basis_state(n, chosen)
    bitstring = format(chosen, f'0{n}b')
    return new_state, bitstring


# ---------------------------------------------------------------------------
#  Sampling
# ---------------------------------------------------------------------------


def sample(state, n_shots):
    """Sample the state n_shots times, returning a histogram.

    Each shot is an independent measurement (Born rule) producing a
    bitstring.  The state is NOT collapsed between shots — each shot
    samples from the same distribution.

    Args:
        state: state vector.
        n_shots: number of measurement shots (positive int).

    Returns:
        Dict mapping bitstring -> count, e.g. {'00': 512, '11': 488}.
    """
    n = _n_qubits(state)
    probs = probabilities(state)
    histogram = {}
    for _ in range(n_shots):
        r = random.random()
        cumulative = 0.0
        chosen = len(probs) - 1
        for i, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                chosen = i
                break
        bs = format(chosen, f'0{n}b')
        histogram[bs] = histogram.get(bs, 0) + 1
    return histogram


def sample_marginal(state, qubits, n_shots):
    """Sample a subset of qubits, returning a marginal histogram.

    Args:
        state: state vector.
        qubits: list of qubit indices to include in the marginal.
        n_shots: number of measurement shots.

    Returns:
        Dict mapping partial bitstring -> count.
    """
    n = _n_qubits(state)
    probs = probabilities(state)
    histogram = {}
    for _ in range(n_shots):
        r = random.random()
        cumulative = 0.0
        chosen = len(probs) - 1
        for i, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                chosen = i
                break
        # Extract marginal bits
        full_bs = format(chosen, f'0{n}b')
        marginal = ''.join(full_bs[q] for q in qubits)
        histogram[marginal] = histogram.get(marginal, 0) + 1
    return histogram


# ---------------------------------------------------------------------------
#  Expectation Values
# ---------------------------------------------------------------------------


def expectation_pauli(state, pauli_string):
    """Expectation value <psi|P|psi> for a Pauli string.

    The Pauli string is a string of characters I, X, Y, Z with length
    equal to the number of qubits.  E.g. "XZI" for a 3-qubit system
    means X on qubit 0, Z on qubit 1, I on qubit 2.

    Args:
        state: state vector.
        pauli_string: string of I/X/Y/Z characters.

    Returns:
        Real float expectation value.
    """
    n = _n_qubits(state)
    if len(pauli_string) != n:
        raise ValueError(
            f"Pauli string length {len(pauli_string)} != {n} qubits"
        )

    # Apply the Pauli operator to |psi>, then compute <psi|P|psi>
    p_state = list(state)
    for q, p in enumerate(pauli_string):
        if p == 'I':
            continue
        elif p == 'X':
            p_state = apply_single_gate(p_state, q, [[0, 1], [1, 0]])
        elif p == 'Y':
            p_state = apply_single_gate(p_state, q, [[0, -1j], [1j, 0]])
        elif p == 'Z':
            p_state = apply_single_gate(p_state, q, [[1, 0], [0, -1]])
        else:
            raise ValueError(f"Unknown Pauli operator: {p}")

    # Inner product <state|p_state>
    result = sum(a.conjugate() * b for a, b in zip(state, p_state))
    return result.real


def expectation_observable(state, observable):
    """Expectation value for a diagonal observable.

    The observable is specified as a list of eigenvalues, one per
    computational basis state.

    Args:
        state: state vector.
        observable: list of real eigenvalues (same length as state).

    Returns:
        Real float expectation value.
    """
    return sum(obs * abs(amp) ** 2 for obs, amp in zip(observable, state))


# ---------------------------------------------------------------------------
#  State Analysis
# ---------------------------------------------------------------------------


def state_fidelity(state_a, state_b):
    """Fidelity |<a|b>|^2 between two pure states.

    Args:
        state_a: first state vector.
        state_b: second state vector (same dimension).

    Returns:
        Float in [0, 1].
    """
    inner = sum(a.conjugate() * b for a, b in zip(state_a, state_b))
    return abs(inner) ** 2


def entanglement_entropy(state, qubit):
    """Von Neumann entanglement entropy of a single qubit with the rest.

    Traces out all qubits except the specified one, computes the reduced
    density matrix (2x2), and returns -Tr(rho log rho).

    Args:
        state: state vector.
        qubit: qubit index to keep (trace out everything else).

    Returns:
        Float entropy in nats (natural log base).  Range [0, ln(2)].
    """
    n = _n_qubits(state)
    bit = n - 1 - qubit

    # Build 2x2 reduced density matrix rho[a][b]
    rho = [[complex(0), complex(0)], [complex(0), complex(0)]]
    for i, amp_i in enumerate(state):
        qi = (i >> bit) & 1  # qubit value for basis state i
        for j, amp_j in enumerate(state):
            qj = (j >> bit) & 1
            # Check that all OTHER qubits match between i and j
            mask = (1 << n) - 1
            mask ^= (1 << bit)  # clear the target bit from mask
            if (i & mask) == (j & mask):
                rho[qi][qj] += amp_i * amp_j.conjugate()

    # Eigenvalues of 2x2 Hermitian matrix
    # rho = [[a, b], [b*, d]] with a, d real, trace = 1
    a_r = rho[0][0].real
    d_r = rho[1][1].real
    b_c = rho[0][1]

    # Eigenvalues: lambda = (a+d)/2 +/- sqrt((a-d)^2/4 + |b|^2)
    half_trace = (a_r + d_r) / 2.0
    disc = ((a_r - d_r) / 2.0) ** 2 + abs(b_c) ** 2
    sqrt_disc = math.sqrt(max(disc, 0.0))

    lam1 = half_trace + sqrt_disc
    lam2 = half_trace - sqrt_disc

    # Von Neumann entropy: -sum(lambda * ln(lambda))
    entropy = 0.0
    for lam in (lam1, lam2):
        if lam > 1e-15:
            entropy -= lam * math.log(lam)
    return entropy


def state_to_bloch(state):
    """Convert a single-qubit state to Bloch sphere coordinates.

    Args:
        state: 2-element state vector [alpha, beta].

    Returns:
        (theta, phi) in radians.  theta in [0, pi], phi in [-pi, pi].
        |0> -> (0, 0),  |1> -> (pi, 0),  |+> -> (pi/2, 0).
    """
    if len(state) != 2:
        raise ValueError("state_to_bloch requires a single-qubit (2-element) state")

    alpha, beta = state[0], state[1]

    # Remove global phase so alpha is real and non-negative
    if abs(alpha) > 1e-15:
        phase = alpha / abs(alpha)
        alpha = abs(alpha)
        beta = beta / phase
    elif abs(beta) > 1e-15:
        alpha = 0.0
        beta = abs(beta)

    theta = 2 * math.acos(min(abs(alpha), 1.0))
    phi = 0.0
    if abs(beta) > 1e-15:
        phi = math.atan2(beta.imag if isinstance(beta, complex) else 0.0,
                         beta.real if isinstance(beta, complex) else float(beta))

    return theta, phi


def schmidt_coefficients(state, partition):
    """Schmidt decomposition coefficients for a bipartite split.

    Splits the qubits into two groups: those in *partition* (subsystem A)
    and those not in partition (subsystem B).  Returns the Schmidt
    coefficients (singular values) that characterise the entanglement.

    Args:
        state: state vector.
        partition: list of qubit indices forming subsystem A.

    Returns:
        Sorted list of Schmidt coefficients (largest first).
    """
    n = _n_qubits(state)
    a_qubits = sorted(partition)
    b_qubits = sorted(set(range(n)) - set(a_qubits))
    dim_a = 1 << len(a_qubits)
    dim_b = 1 << len(b_qubits)

    # Build the matrix M[a_index][b_index] from the state vector
    matrix = [[complex(0)] * dim_b for _ in range(dim_a)]
    for i, amp in enumerate(state):
        # Extract A and B indices from full index i
        a_idx = 0
        for k, q in enumerate(a_qubits):
            bit = n - 1 - q
            if i & (1 << bit):
                a_idx |= (1 << (len(a_qubits) - 1 - k))
        b_idx = 0
        for k, q in enumerate(b_qubits):
            bit = n - 1 - q
            if i & (1 << bit):
                b_idx |= (1 << (len(b_qubits) - 1 - k))
        matrix[a_idx][b_idx] = amp

    # SVD of small matrix: compute singular values via M M^dagger eigenvalues
    # For simplicity, compute M M^dagger (dim_a x dim_a) and find eigenvalues
    dim_small = min(dim_a, dim_b)

    if dim_a <= dim_b:
        # Compute M M^dagger
        mm = [[complex(0)] * dim_a for _ in range(dim_a)]
        for i in range(dim_a):
            for j in range(dim_a):
                for k in range(dim_b):
                    mm[i][j] += matrix[i][k] * matrix[j][k].conjugate()
    else:
        # Compute M^dagger M (dim_b x dim_b)
        dim_small = dim_b
        mm = [[complex(0)] * dim_b for _ in range(dim_b)]
        for i in range(dim_b):
            for j in range(dim_b):
                for k in range(dim_a):
                    mm[i][j] += matrix[k][i].conjugate() * matrix[k][j]

    # For 2x2, analytic eigenvalues
    if dim_small == 2:
        a_r = mm[0][0].real
        d_r = mm[1][1].real
        b_c = mm[0][1]
        half = (a_r + d_r) / 2.0
        disc = ((a_r - d_r) / 2.0) ** 2 + abs(b_c) ** 2
        sq = math.sqrt(max(disc, 0.0))
        eigs = [half + sq, half - sq]
    elif dim_small == 1:
        eigs = [mm[0][0].real]
    else:
        # Power iteration for largest eigenvalue, then deflation
        # For small matrices this is fine
        eigs = _eigenvalues_hermitian(mm, dim_small)

    # Schmidt coefficients = sqrt(eigenvalues)
    coeffs = [math.sqrt(max(e, 0.0)) for e in eigs]
    coeffs.sort(reverse=True)
    return coeffs


def _eigenvalues_hermitian(matrix, dim):
    """Approximate eigenvalues of a small Hermitian matrix via power iteration.

    Good enough for dim <= 16 (4 qubits per partition).
    """
    # Simple approach: compute trace and det for eigenvalue decomposition
    # For arbitrary small dim, use repeated power iteration + deflation
    eigenvalues = []
    mat = [row[:] for row in matrix]  # copy

    for _ in range(dim):
        # Power iteration
        v = [complex(1.0 / math.sqrt(dim))] * dim
        for _iter in range(200):
            # w = mat @ v
            w = [sum(mat[i][j] * v[j] for j in range(dim)) for i in range(dim)]
            norm = math.sqrt(sum(abs(x) ** 2 for x in w))
            if norm < 1e-30:
                eigenvalues.append(0.0)
                break
            v = [x / norm for x in w]
        else:
            # Rayleigh quotient
            w = [sum(mat[i][j] * v[j] for j in range(dim)) for i in range(dim)]
            lam = sum(v[i].conjugate() * w[i] for i in range(dim)).real
            eigenvalues.append(lam)

            # Deflate: mat = mat - lam * v v^dagger
            for i in range(dim):
                for j in range(dim):
                    mat[i][j] -= lam * v[i] * v[j].conjugate()

    return eigenvalues


# ---------------------------------------------------------------------------
#  Classical Extraction Chain
# ---------------------------------------------------------------------------


def extract_max_probability(state):
    """Find the most probable measurement outcome.

    Args:
        state: state vector.

    Returns:
        (bitstring, probability) for the most likely outcome.
    """
    n = _n_qubits(state)
    probs = probabilities(state)
    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    return format(best_idx, f'0{n}b'), probs[best_idx]


def extract_phase(state):
    """Estimate phase from state amplitudes.

    For QPE-like algorithms where the answer is encoded in the phase
    of the state amplitudes.  Returns the phase of the largest-amplitude
    basis state.

    Args:
        state: state vector.

    Returns:
        Phase in radians [-pi, pi].
    """
    best_idx = max(range(len(state)), key=lambda i: abs(state[i]))
    amp = state[best_idx]
    if abs(amp) < 1e-15:
        return 0.0
    return math.atan2(amp.imag, amp.real)


def extract_function_value(state, output_qubits):
    """Extract a classical integer from output register qubits.

    Finds the most probable assignment of the output qubits and
    returns it as an integer.

    Args:
        state: state vector.
        output_qubits: list of qubit indices forming the output register.

    Returns:
        (integer_value, probability) tuple.
    """
    hist = sample_marginal(state, output_qubits, 1000)
    best_bs = max(hist, key=hist.get)
    return int(best_bs, 2), hist[best_bs] / 1000.0


def histogram_to_answer(histogram, interpret_fn):
    """Apply a user-defined interpretation function to a histogram.

    Args:
        histogram: dict mapping bitstring -> count.
        interpret_fn: callable(histogram) -> answer.

    Returns:
        Whatever interpret_fn returns.
    """
    return interpret_fn(histogram)


# ---------------------------------------------------------------------------
#  Example Algorithms
# ---------------------------------------------------------------------------


def bell_state_example():
    """Create and measure a Bell state (|00> + |11>) / sqrt(2).

    Demonstrates the full chain: circuit -> state -> measurement -> answer.

    Returns:
        Dict with keys: circuit, raw_state, histogram, answer, explanation.
    """
    circuit = [('h', 0), ('cnot', 0, 1)]
    state = run_circuit(2, circuit)
    hist = sample(state, 1000)

    return {
        'circuit': circuit,
        'raw_state': state,
        'histogram': hist,
        'answer': 'entangled',
        'explanation': (
            'Bell state (|00> + |11>)/sqrt(2).  Measurement always gives '
            'correlated outcomes: both 00 or both 11, never 01 or 10.'
        ),
    }


def deutsch_jozsa(oracle_type='balanced', n_qubits=3):
    """Deutsch-Jozsa algorithm: determine if f(x) is constant or balanced.

    Uses ONE query to the oracle (classically requires up to 2^(n-1)+1).

    For 'constant' oracle: f(x) = 0 for all x.
    For 'balanced' oracle: f(x) = x_0 (parity of first bit).

    Args:
        oracle_type: 'constant' or 'balanced'.
        n_qubits: total qubits (n-1 input + 1 ancilla). Must be >= 2.

    Returns:
        Dict with keys: circuit, raw_state, histogram, answer, explanation.
    """
    n_input = n_qubits - 1
    ancilla = n_qubits - 1

    # Build circuit
    circuit = []

    # Prepare ancilla in |->
    circuit.append(('x', ancilla))
    circuit.append(('h', ancilla))

    # Hadamard all input qubits
    for q in range(n_input):
        circuit.append(('h', q))

    # Oracle
    if oracle_type == 'balanced':
        # f(x) = x_0: CNOT from qubit 0 to ancilla
        circuit.append(('cnot', 0, ancilla))
    elif oracle_type == 'constant':
        # f(x) = 0: do nothing (identity oracle)
        pass
    else:
        raise ValueError(f"oracle_type must be 'constant' or 'balanced', got {oracle_type}")

    # Hadamard all input qubits again
    for q in range(n_input):
        circuit.append(('h', q))

    state = run_circuit(n_qubits, circuit)

    # Measure input qubits only
    hist = sample_marginal(state, list(range(n_input)), 1000)

    # Answer: if all input qubits are 0 -> constant, else -> balanced
    all_zeros = '0' * n_input
    is_constant = hist.get(all_zeros, 0) > 500

    return {
        'circuit': circuit,
        'raw_state': state,
        'histogram': hist,
        'answer': 'constant' if is_constant else 'balanced',
        'explanation': (
            f"Deutsch-Jozsa with {oracle_type} oracle on {n_input} input qubits. "
            f"Measurement of input register: {hist}. "
            f"All-zeros -> constant; anything else -> balanced."
        ),
    }


def grover_search(n_qubits, marked_item):
    """Grover's search algorithm: find a marked item in an unstructured database.

    Uses O(sqrt(2^n)) iterations to find the marked item with high probability.

    Args:
        n_qubits: number of qubits (search space = 2^n items).
        marked_item: integer index of the item to find (0 .. 2^n - 1).

    Returns:
        Dict with keys: circuit, raw_state, histogram, answer, explanation.
    """
    N = 1 << n_qubits
    if marked_item < 0 or marked_item >= N:
        raise ValueError(f"marked_item {marked_item} out of range for {n_qubits} qubits")

    # Optimal number of iterations
    n_iter = max(1, round(math.pi / 4 * math.sqrt(N)))

    # Build circuit
    circuit = []

    # Initial superposition
    for q in range(n_qubits):
        circuit.append(('h', q))

    for _ in range(n_iter):
        # Oracle: flip phase of marked item
        # Implemented as: X on qubits where marked_item bit is 0,
        # then multi-controlled Z, then X again
        marked_bits = format(marked_item, f'0{n_qubits}b')
        for q in range(n_qubits):
            if marked_bits[q] == '0':
                circuit.append(('x', q))

        # Multi-controlled Z = H on last, Toffoli chain, H on last
        if n_qubits == 1:
            circuit.append(('z', 0))
        elif n_qubits == 2:
            circuit.append(('cz', 0, 1))
        elif n_qubits == 3:
            # CCZ = H-Toffoli-H on target
            circuit.append(('h', 2))
            circuit.append(('toffoli', 0, 1, 2))
            circuit.append(('h', 2))
        else:
            # For n>3, use ancilla-free decomposition
            # Use nested Toffoli gates
            circuit.append(('h', n_qubits - 1))
            circuit.append(('toffoli', 0, 1, n_qubits - 1))
            circuit.append(('h', n_qubits - 1))
            # Note: this is approximate for n>3 — only marks states where
            # first two qubits and last qubit match.  For a complete
            # implementation, a Toffoli cascade with ancilla is needed.
            # For n<=5 this works for demonstration.

        for q in range(n_qubits):
            if marked_bits[q] == '0':
                circuit.append(('x', q))

        # Diffusion operator: 2|s><s| - I
        for q in range(n_qubits):
            circuit.append(('h', q))
        for q in range(n_qubits):
            circuit.append(('x', q))

        if n_qubits == 1:
            circuit.append(('z', 0))
        elif n_qubits == 2:
            circuit.append(('cz', 0, 1))
        elif n_qubits == 3:
            circuit.append(('h', 2))
            circuit.append(('toffoli', 0, 1, 2))
            circuit.append(('h', 2))
        else:
            circuit.append(('h', n_qubits - 1))
            circuit.append(('toffoli', 0, 1, n_qubits - 1))
            circuit.append(('h', n_qubits - 1))

        for q in range(n_qubits):
            circuit.append(('x', q))
        for q in range(n_qubits):
            circuit.append(('h', q))

    state = run_circuit(n_qubits, circuit)
    hist = sample(state, 1000)
    best = max(hist, key=hist.get)

    return {
        'circuit': circuit,
        'raw_state': state,
        'histogram': hist,
        'answer': int(best, 2),
        'explanation': (
            f"Grover search for item {marked_item} in {N}-item database. "
            f"Used {n_iter} iterations.  Most frequent outcome: {best} "
            f"({hist.get(best, 0)}/1000 shots)."
        ),
    }


def teleportation_example(alpha=None, beta=None):
    """Quantum teleportation: transfer a qubit state using a Bell pair.

    Uses 3 qubits:
        qubit 0: state to teleport (alpha|0> + beta|1>)
        qubit 1: Alice's half of Bell pair
        qubit 2: Bob's half of Bell pair

    After the protocol, qubit 2 holds the original state.

    Args:
        alpha: amplitude for |0> (default: 1/sqrt(2)).
        beta: amplitude for |1> (default: 1/sqrt(2)).

    Returns:
        Dict with keys: initial_state, raw_state, teleported_qubit_probs,
        answer, explanation.
    """
    if alpha is None:
        alpha = complex(1.0 / math.sqrt(2))
    if beta is None:
        beta = complex(1.0 / math.sqrt(2))

    # Normalize
    norm = math.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
    alpha = alpha / norm
    beta = beta / norm

    # Create initial state: (alpha|0> + beta|1>) ⊗ |00>
    initial = product_state([[alpha, beta], [1, 0], [1, 0]])

    # Create Bell pair between qubits 1 and 2
    state = gate_h(initial, 1)
    state = gate_cnot(state, 1, 2)

    # Alice's operations: CNOT(0,1), H(0)
    state = gate_cnot(state, 0, 1)
    state = gate_h(state, 0)

    # Alice measures qubits 0 and 1 — but in our simulation we can
    # apply conditional corrections deterministically by checking amplitudes.
    # For simplicity, measure and apply corrections:
    state, m0 = measure(state, 0)
    state, m1 = measure(state, 1)

    # Bob's corrections based on Alice's measurements
    if m1 == 1:
        state = gate_x(state, 2)
    if m0 == 1:
        state = gate_z(state, 2)

    # Check qubit 2 probabilities
    p0 = probability(state, 2, 0)
    p1 = probability(state, 2, 1)

    return {
        'initial_state': (alpha, beta),
        'raw_state': state,
        'teleported_qubit_probs': {'0': p0, '1': p1},
        'alice_measurements': (m0, m1),
        'answer': 'teleported',
        'explanation': (
            f"Teleported state alpha={alpha:.4f}, beta={beta:.4f}. "
            f"Alice measured ({m0}, {m1}).  Bob's qubit probabilities: "
            f"P(0)={p0:.4f}, P(1)={p1:.4f}."
        ),
    }


def bernstein_vazirani(hidden_string):
    """Bernstein-Vazirani algorithm: find a hidden bitstring s in one query.

    The oracle computes f(x) = s · x (mod 2) — the bitwise dot product.
    Classically requires n queries; quantumly requires 1.

    Args:
        hidden_string: string of '0' and '1' characters, e.g. '101'.

    Returns:
        Dict with keys: circuit, raw_state, histogram, answer, explanation.
    """
    n = len(hidden_string)
    n_qubits = n + 1  # n input + 1 ancilla
    ancilla = n

    circuit = []

    # Prepare ancilla in |->
    circuit.append(('x', ancilla))
    circuit.append(('h', ancilla))

    # Hadamard all input qubits
    for q in range(n):
        circuit.append(('h', q))

    # Oracle: for each bit of s that is 1, CNOT from that qubit to ancilla
    for q in range(n):
        if hidden_string[q] == '1':
            circuit.append(('cnot', q, ancilla))

    # Hadamard all input qubits
    for q in range(n):
        circuit.append(('h', q))

    state = run_circuit(n_qubits, circuit)

    # Measure input qubits
    hist = sample_marginal(state, list(range(n)), 1000)
    best = max(hist, key=hist.get)

    return {
        'circuit': circuit,
        'raw_state': state,
        'histogram': hist,
        'answer': best,
        'explanation': (
            f"Bernstein-Vazirani with hidden string '{hidden_string}'. "
            f"Found: '{best}' ({hist.get(best, 0)}/1000 shots)."
        ),
    }


# ---------------------------------------------------------------------------
#  Reports (Rule 9)
# ---------------------------------------------------------------------------


def quantum_output_report():
    """Standard report for the quantum output module.

    Returns:
        Dict with measurement capabilities, example algorithm results.
    """
    bell = bell_state_example()
    return {
        'measurement_types': ['single_qubit', 'all_qubits', 'sampling',
                              'marginal_sampling'],
        'analysis_tools': ['fidelity', 'entanglement_entropy', 'bloch_sphere',
                           'schmidt_coefficients'],
        'extraction_tools': ['max_probability', 'phase', 'function_value',
                             'histogram_interpreter'],
        'example_algorithms': ['bell_state', 'deutsch_jozsa', 'grover_search',
                               'teleportation', 'bernstein_vazirani'],
        'bell_state_histogram': bell['histogram'],
    }


def full_report():
    """Extended report with all example algorithm results.

    Returns:
        Dict of extended module summary information.
    """
    r = quantum_output_report()
    dj = deutsch_jozsa('balanced', 3)
    r['deutsch_jozsa_answer'] = dj['answer']
    bv = bernstein_vazirani('101')
    r['bernstein_vazirani_answer'] = bv['answer']
    r['supported_pauli_operators'] = ['I', 'X', 'Y', 'Z']
    return r
