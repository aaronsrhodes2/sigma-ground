"""Quantum-solids analysis tools (standard physics).

Composite tools cascading through field.interface.{superconductivity, tunneling,
quantum_wells, band_structure, quantum_matter}.
"""
from __future__ import annotations

from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_SRC = "sigma_ground.field.interface (superconductivity, tunneling, quantum_wells, band_structure, quantum_matter)"

# μ₀ (H/m) — converts an H-field in A/m to a B-field in tesla.
_MU_0 = 1.25663706212e-6


def _safe_call(fn, *a, **k):
    """Call a leaf function, returning (value, error_note) instead of a bare
    value-or-None. On success: (value, None). On failure: (None, note) where
    `note` is the exception's type and message (e.g. "KeyError:
    'unobtainium'") -- deliberately NOT fn-name-specific, so that two
    different leaf calls failing on the SAME underlying bad input (e.g. the
    same bad material_key) produce byte-identical notes and can be
    deduplicated by _notes_from_errors() below.

    Like _safe() above, this never raises -- one bad sub-call only ever
    poisons its own (key, value) pair in the caller's results dict, so the
    caller always goes on to make the rest of the cascade's calls. The
    difference from _safe() is purely that the reason for a failure is
    handed back instead of swallowed, so it can be threaded into
    ToolResult(notes=...) rather than surfacing a bare `None` with nothing
    explaining why."""
    try:
        return fn(*a, **k), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _notes_from_errors(errors: dict[str, str]) -> str:
    """Fold a {result_key: error_note} map (only the FAILED keys) into one
    ToolResult.notes string.

    Groups result keys whose error_note is byte-identical -- the signature
    of a shared root cause, e.g. N calls in one cascade that all take the
    same bad material_key -- so the explanation is stated once per DISTINCT
    cause (naming every result key it took down), not once per affected
    key. Keys with distinct causes each keep their own clause. Returns ""
    when `errors` is empty (the all-success case), matching ToolResult's
    default notes="".
    """
    if not errors:
        return ""
    groups: dict[str, list[str]] = {}
    for key, note in errors.items():
        groups.setdefault(note, []).append(key)
    return " | ".join(f"{', '.join(keys)}: {note}" for note, keys in groups.items())


def _resolve_sc_key(material: str):
    """Map a user material string to a SUPERCONDUCTORS database key.

    Case/space/hyphen-insensitive; matches either the dict key or the
    human-readable name. Returns None if no match.
    """
    from sigma_ground.field.interface import superconductivity as SC
    if material in SC.SUPERCONDUCTORS:
        return material

    def _norm(s):
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    target = _norm(material)
    for key, data in SC.SUPERCONDUCTORS.items():
        if target == _norm(key) or target == _norm(data.get("name", "")):
            return key
    return None


def superconducting_gap_analysis(critical_temp_k: float = 9.2) -> dict[str, Any]:
    """BCS spectroscopic gap frequency f = 2*Delta/h of a superconductor from
    its critical temperature (Delta = 1.764 k_B Tc). Defaults to niobium
    (Tc=9.2 K -> ~677 GHz). e.g. superconducting_gap_analysis(9.2).

    (For the Ginzburg-Landau parameter kappa and the Hc1/Hc2 critical fields,
    see superconductor_critical_field_analysis.)"""
    from sigma_ground.field.interface import superconductivity as SC
    f_gap, err = _safe_call(SC.gap_frequency, critical_temp_k)
    results = {"gap_frequency_Hz": f_gap}
    errors = {"gap_frequency_Hz": err} if err else {}
    return ToolResult(value=results, units="Hz",
                      source="sigma_ground.field.interface.superconductivity",
                      provenance_tag="DERIVED",
                      formula="f = 2*Delta/h, Delta = 1.764 k_B Tc (BCS weak coupling)",
                      notes=_notes_from_errors(errors),
                      inputs={"critical_temp_k": critical_temp_k}).to_dict()


def superconductor_critical_field_analysis(material: str = "niobium") -> dict[str, Any]:
    """Critical magnetic fields of a named superconductor: the Ginzburg-Landau
    parameter kappa, the thermodynamic critical field Hc, and (for Type-II)
    the lower/upper critical fields Hc1 and Hc2. The material is looked up in
    the library's superconductor database (e.g. niobium, NbTi, Nb3Sn, lead,
    aluminum, YBCO). e.g. superconductor_critical_field_analysis('NbTi')."""
    from sigma_ground.field.interface import superconductivity as SC

    key = _resolve_sc_key(material)
    if key is None:
        return ToolResult(
            value={"error": f"unknown superconductor '{material}'"},
            units="", source="sigma_ground.field.interface.superconductivity",
            provenance_tag="DERIVED",
            notes="Material not found in the superconductor database.",
            inputs={"material": material}).to_dict()

    data = SC.SUPERCONDUCTORS[key]
    n_e, v_F, T_c = data["n_e_m3"], data["v_F_m_s"], data["T_c_K"]

    if not data.get("is_superconductor", True) or T_c <= 0:
        return ToolResult(
            value={"name": data["name"], "is_superconductor": False,
                   "suppression": data.get("suppression")},
            units="", source="sigma_ground.field.interface.superconductivity",
            provenance_tag="DERIVED",
            notes=(f"{data['name']} does not superconduct at ambient conditions "
                   "— no critical fields."),
            inputs={"material": material}).to_dict()

    sc_type = data.get("type")
    is_type_II = sc_type == "II"
    # gl_parameter_effective returns the MEASURED (dirty-limit) kappa where the
    # DB has it — essential for alloys/compounds (the clean-limit free-electron
    # gl_parameter badly underestimates kappa, e.g. Nb 0.11 vs measured 1.05).
    kappa, err_kappa = _safe_call(SC.gl_parameter_effective, key)
    H_c, err_Hc = _safe_call(SC.thermodynamic_critical_field, n_e, T_c, 0.0)
    H_c1, err_Hc1 = (_safe_call(SC.lower_critical_field, n_e, v_F, T_c, 0.0, key)
                     if is_type_II else (None, None))
    H_c2, err_Hc2 = (_safe_call(SC.upper_critical_field, n_e, v_F, T_c, 0.0, key)
                     if is_type_II else (None, None))

    results = {
        "name": data["name"],
        "type": sc_type,
        "T_c_K": T_c,
        "gl_parameter_kappa": kappa,
        "kappa_source": data.get("kappa_source"),
        "Hc_thermodynamic_A_per_m": H_c,
        "Hc_thermodynamic_T": (H_c * _MU_0) if H_c is not None else None,
        "Hc1_lower_A_per_m": H_c1,
        "Hc2_upper_A_per_m": H_c2,
        "Hc2_upper_T": (H_c2 * _MU_0) if H_c2 is not None else None,
    }
    errors = {k: v for k, v in {
        "gl_parameter_kappa": err_kappa,
        "Hc_thermodynamic_A_per_m": err_Hc,
        "Hc1_lower_A_per_m": err_Hc1,
        "Hc2_upper_A_per_m": err_Hc2,
    }.items() if v}
    note = ("Type-II: Hc1 < Hc < Hc2 (Abrikosov vortex/mixed state between them)."
            if is_type_II else
            "Type-I: a single critical field Hc; no mixed state, so Hc1/Hc2 "
            "are not defined.")
    note += (" kappa is the measured (dirty-limit) value where available. "
             "Hc uses a free-electron DOS N(0): ~20% accurate for simple metals, "
             "a lower bound for high-DOS d-band/A15 materials.")
    return ToolResult(
        value=results,
        units="A/m for H-fields (T = mu0*H given alongside)",
        source="sigma_ground.field.interface.superconductivity",
        provenance_tag="DERIVED",
        formula=("Hc = Delta*sqrt(N(0)/mu0), Delta=1.764 k_B Tc, N(0)=3 n_e/(4 E_F); "
                 "Hc1 = Hc*(ln k + 0.5)/(sqrt2*k); Hc2 = sqrt2*k*Hc (Abrikosov)"),
        notes=" | ".join(filter(None, [note, _notes_from_errors(errors)])),
        inputs={"material": material}).to_dict()


def quantum_tunneling_analysis(barrier_height_eV: float = 1.0,
                               particle_energy_eV: float = 0.5,
                               barrier_width_nm: float = 1.0) -> dict[str, Any]:
    """WKB transmission probability of a particle through a rectangular
    potential barrier. e.g. quantum_tunneling_analysis(1.0, 0.5, 1.0) -> a
    0.5 eV electron through a 1 eV, 1 nm barrier."""
    from sigma_ground.field.interface import tunneling as T
    height = barrier_height_eV

    def _barrier(x):
        return height

    width_m = barrier_width_nm * 1.0e-9
    prob, err = _safe_call(T.wkb_transmission, _barrier, particle_energy_eV, 0.0, width_m)
    results = {"transmission_probability": prob}
    errors = {"transmission_probability": err} if err else {}
    return ToolResult(value=results, units="dimensionless",
                      source="sigma_ground.field.interface.tunneling",
                      provenance_tag="DERIVED",
                      formula="T = exp(-2 int kappa dx), kappa=sqrt(2m(V-E))/hbar (WKB)",
                      notes=_notes_from_errors(errors),
                      inputs={"barrier_height_eV": barrier_height_eV,
                              "particle_energy_eV": particle_energy_eV,
                              "barrier_width_nm": barrier_width_nm}).to_dict()


def quantum_box_energy_analysis(n1: int = 1, n2: int = 1, n3: int = 1,
                                box_size_nm: float = 1.0) -> dict[str, Any]:
    """Energy of a quantum state (n1,n2,n3) for a particle in a 3D cubic box
    (infinite well). e.g. quantum_box_energy_analysis(1, 1, 1, 1.0) -> ground
    state of an electron in a 1 nm box."""
    from sigma_ground.field.interface import quantum_wells as QW
    L = box_size_nm * 1.0e-9
    E, err = _safe_call(QW.box_energy_3d_eV, n1, n2, n3, L)
    results = {"energy_eV": E}
    errors = {"energy_eV": err} if err else {}
    return ToolResult(value=results, units="eV",
                      source="sigma_ground.field.interface.quantum_wells",
                      provenance_tag="DERIVED",
                      formula="E = (hbar^2 pi^2 / 2m)(n1^2/L1^2 + n2^2/L2^2 + n3^2/L3^2)",
                      notes=_notes_from_errors(errors),
                      inputs={"n1": n1, "n2": n2, "n3": n3,
                              "box_size_nm": box_size_nm}).to_dict()


def band_dos_shape_analysis(structure: str = "bcc",
                            d_electron_count: int = 5) -> dict[str, Any]:
    """Tight-binding density-of-states shape factor at the Fermi level for a
    transition metal (van Hove peak > 1, pseudogap < 1). structure in
    bcc/fcc/hcp, d-count 1-9. e.g. band_dos_shape_analysis('bcc', 5)."""
    from sigma_ground.field.interface import band_structure as BS
    g, err = _safe_call(BS.dos_shape_factor, structure, d_electron_count)
    results = {"dos_shape_factor": g}
    errors = {"dos_shape_factor": err} if err else {}
    return ToolResult(value=results, units="dimensionless",
                      source="sigma_ground.field.interface.band_structure",
                      provenance_tag="DERIVED",
                      formula="g_dos = TB DOS(E_F) / rectangular DOS (avg-normalized)",
                      notes=_notes_from_errors(errors),
                      inputs={"structure": structure,
                              "d_electron_count": d_electron_count}).to_dict()


def magnetic_exchange_analysis(atomic_number: int = 24, oxidation_state: int = 3,
                               coord_key: str = "oxide_oct") -> dict[str, Any]:
    """Two-site Heisenberg model for a magnetic ion: exchange coupling J from
    crystal-field / Goodenough-Kanamori, VQE vs exact ground-state energy, and
    the resulting spin state (singlet/triplet). Defaults: Cr3+ octahedral oxide.
    e.g. magnetic_exchange_analysis(24, 3, 'oxide_oct')."""
    from sigma_ground.field.interface import quantum_matter as QM
    r, err = _safe_call(QM.two_site_spin_hamiltonian_from_crystal_field,
                        atomic_number, oxidation_state, coord_key)
    if isinstance(r, dict):
        results = {
            "J_exchange_eV": r.get("J_exchange_eV"),
            "vqe_ground_energy_eV": r.get("E_vqe_eV"),
            "exact_ground_energy_eV": r.get("E_exact_eV"),
            "spin_state": r.get("spin_state"),
            "is_high_spin": r.get("is_high_spin"),
        }
    else:
        results = {"J_exchange_eV": None, "vqe_ground_energy_eV": None,
                   "exact_ground_energy_eV": None, "spin_state": None,
                   "is_high_spin": None}
    # One underlying call feeds all five keys, so a failure poisons all five --
    # attribute `err` to every key rather than just one, so _notes_from_errors
    # names the whole set in its single shared-cause clause.
    errors = {k: err for k in results} if err else {}
    return ToolResult(value=results, units="eV", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="H = J(X1X2+Y1Y2+Z1Z2); J from crystal field; VQE vs exact",
                      notes=_notes_from_errors(errors),
                      inputs={"atomic_number": atomic_number,
                              "oxidation_state": oxidation_state,
                              "coord_key": coord_key}).to_dict()
