"""Extended chemistry analysis tools (standard physics/chemistry).

Composite tools cascading through field.interface.{acid_base, solution,
electrochemistry, chemical_reactions, radioactive_decay}.
"""
from __future__ import annotations

from typing import Any

from sigma_ground.mcp.provenance import ToolResult

_SRC = "sigma_ground.field.interface (acid_base, solution, electrochemistry, chemical_reactions, radioactive_decay)"


def _safe_call(fn, *a, **k):
    """Call a leaf function, returning (value, error_note) instead of a bare
    value-or-None. On success: (value, None). On failure: (None, note) where
    `note` is the exception's type and message (e.g. "KeyError:
    'unobtainium'") -- deliberately NOT fn-name-specific, so that two
    different leaf calls failing on the SAME underlying bad input (e.g. the
    same bad isotope_key) produce byte-identical notes and can be
    deduplicated by _notes_from_errors() below.

    This never raises -- one bad sub-call only ever poisons its own (key,
    value) pair in the caller's results dict, so the caller always goes on
    to make the rest of the cascade's calls. The reason for a failure is
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
    same bad input -- so the explanation is stated once per DISTINCT
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


def titration_analysis(acid_concentration_M: float = 0.1,
                       acid_volume_mL: float = 25.0,
                       base_concentration_M: float = 0.1,
                       base_volume_mL: float = 10.0,
                       weak_acid_key: str = "acetic_acid") -> dict[str, Any]:
    """pH at a point in an acid-base titration, for both a strong acid (HCl-type)
    and a weak acid (Henderson-Hasselbalch / buffer region), titrated with a
    strong base. Weak-acid keys: acetic_acid, formic_acid, hydrofluoric_acid, etc.
    e.g. titration_analysis(0.1, 25, 0.1, 10, 'acetic_acid')."""
    from sigma_ground.field.interface import acid_base as AB
    _raw = {
        "strong_acid_pH": _safe_call(AB.titration_strong_acid_strong_base,
                                acid_concentration_M, acid_volume_mL,
                                base_concentration_M, base_volume_mL),
        "weak_acid_pH": _safe_call(AB.titration_weak_acid_strong_base, weak_acid_key,
                              acid_concentration_M, acid_volume_mL,
                              base_concentration_M, base_volume_mL),
    }
    results = {k: v for k, (v, e) in _raw.items()}
    errors = {k: e for k, (v, e) in _raw.items() if e}
    return ToolResult(value=results, units="pH", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="strong: pH=-log[H+] from net mol; weak: Henderson-Hasselbalch",
                      notes=_notes_from_errors(errors),
                      inputs={"acid_concentration_M": acid_concentration_M,
                              "acid_volume_mL": acid_volume_mL,
                              "base_concentration_M": base_concentration_M,
                              "base_volume_mL": base_volume_mL,
                              "weak_acid_key": weak_acid_key}).to_dict()


def acid_speciation_analysis(ph: float = 7.0,
                             pka_list: list[float] | None = None) -> dict[str, Any]:
    """Fractional abundance (alpha) of each protonation state of a polyprotic
    acid at a given pH. Default is phosphoric acid (pKa 2.15, 7.20, 12.35).
    e.g. acid_speciation_analysis(7.0, [2.15, 7.2, 12.35])."""
    from sigma_ground.field.interface import acid_base as AB
    if pka_list is None:
        pka_list = [2.15, 7.20, 12.35]
    alphas, err = _safe_call(AB.polyprotic_alpha, pka_list, ph)
    results = {"species_fractions": alphas}
    errors = {"species_fractions": err} if err else {}
    return ToolResult(value=results, units="dimensionless (fractions sum to 1)",
                      source="sigma_ground.field.interface.acid_base",
                      provenance_tag="DERIVED",
                      formula="alpha_i from stepwise Ka and [H+]",
                      notes=_notes_from_errors(errors),
                      inputs={"ph": ph, "pka_list": pka_list}).to_dict()


def solution_analysis(initial_concentration_M: float = 1.0,
                      initial_volume_mL: float = 10.0,
                      final_volume_mL: float = 100.0,
                      second_concentration_M: float = 0.5,
                      second_volume_mL: float = 50.0,
                      salt_key: str = "silver_chloride",
                      cation_concentration_M: float = 1.0e-3,
                      anion_concentration_M: float = 1.0e-3) -> dict[str, Any]:
    """Solution concentration & solubility: dilution (C1V1=C2V2), the
    concentration of two mixed solutions, and whether a sparingly-soluble salt
    precipitates (ion product vs Ksp). Salt keys: silver_chloride, barium_sulfate,
    lead_chloride, etc. e.g. solution_analysis(1.0, 10, 100)."""
    from sigma_ground.field.interface import solution as SOL
    _raw = {
        "diluted_concentration_M": _safe_call(SOL.dilution, initial_concentration_M,
                                         initial_volume_mL, final_volume_mL),
        "mixed_concentration_M": _safe_call(SOL.mixing_concentration, initial_concentration_M,
                                      initial_volume_mL, second_concentration_M,
                                      second_volume_mL),
        "will_precipitate": _safe_call(SOL.will_precipitate, salt_key,
                                 cation_concentration_M, anion_concentration_M),
    }
    results = {k: v for k, (v, e) in _raw.items()}
    errors = {k: e for k, (v, e) in _raw.items() if e}
    return ToolResult(value=results, units="mol/L, mol/L, bool", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="C1 V1 = C2 V2; c_mix=(C1V1+C2V2)/(V1+V2); Q vs Ksp",
                      notes=_notes_from_errors(errors),
                      inputs={"initial_concentration_M": initial_concentration_M,
                              "initial_volume_mL": initial_volume_mL,
                              "final_volume_mL": final_volume_mL,
                              "salt_key": salt_key}).to_dict()


def electrochemistry_analysis(current_density: float = 10.0,
                              exchange_current_density: float = 1.0e-3,
                              lambda_cation: float = 73.5,
                              lambda_anion: float = 76.3,
                              concentration_M: float = 0.1) -> dict[str, Any]:
    """Electrochemistry: Tafel activation overpotential for a given current
    density, limiting molar conductivity (Kohlrausch sum), and the solution
    conductivity at a concentration. Default ions ~ KCl.
    e.g. electrochemistry_analysis(10, 1e-3, 73.5, 76.3, 0.1)."""
    from sigma_ground.field.interface import electrochemistry as EC
    Lm, err_Lm = _safe_call(EC.molar_conductivity_dilute, lambda_cation, lambda_anion)
    kappa, err_kappa = (_safe_call(EC.solution_conductivity, concentration_M, Lm)
                        if Lm is not None else (None, err_Lm))
    _raw = {
        "tafel_overpotential_V": _safe_call(EC.tafel_overpotential, current_density, exchange_current_density),
        "molar_conductivity": (Lm, err_Lm),
        "solution_conductivity": (kappa, err_kappa),
    }
    results = {k: v for k, (v, e) in _raw.items()}
    errors = {k: e for k, (v, e) in _raw.items() if e}
    return ToolResult(value=results, units="V, S.cm^2/mol, mS/cm (dilute-limit)", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="eta=(RT/alpha n F)ln(j/j0); Lambda_m=lambda+ + lambda-; kappa=c Lambda_m",
                      notes=_notes_from_errors(errors),
                      inputs={"current_density": current_density,
                              "exchange_current_density": exchange_current_density,
                              "lambda_cation": lambda_cation, "lambda_anion": lambda_anion,
                              "concentration_M": concentration_M}).to_dict()


def reaction_kinetics_analysis(rate_constant: float = 0.01,
                               activation_energy_eV: float = 0.5,
                               prefactor: float = 1.0e13,
                               target_rate: float = 0.1,
                               m_a_amu: float = 16.0, m_b_amu: float = 32.0,
                               r_a_pm: float = 150.0, r_b_pm: float = 150.0) -> dict[str, Any]:
    """Chemical kinetics: collision-theory pre-exponential factor, the half-life
    of a first-order reaction (ln2/k), and the temperature needed to reach a
    target rate constant (Arrhenius inversion).
    e.g. reaction_kinetics_analysis(0.01, 0.5, 1e13)."""
    from sigma_ground.field.interface import chemical_reactions as CR
    _raw = {
        "collision_prefactor": _safe_call(CR.collision_prefactor, m_a_amu, m_b_amu, r_a_pm, r_b_pm),
        "half_life_s": _safe_call(CR.half_life, rate_constant),
        "temperature_for_target_rate_K": _safe_call(CR.temperature_for_rate, target_rate, prefactor, activation_energy_eV),
    }
    results = {k: v for k, (v, e) in _raw.items()}
    errors = {k: e for k, (v, e) in _raw.items() if e}
    return ToolResult(value=results, units="varies, s, K", source=_SRC,
                      provenance_tag="DERIVED",
                      formula="A from collision theory; t_half=ln2/k; T from k=A exp(-Ea/kT)",
                      notes=_notes_from_errors(errors),
                      inputs={"rate_constant": rate_constant,
                              "activation_energy_eV": activation_energy_eV,
                              "prefactor": prefactor, "target_rate": target_rate}).to_dict()


def radioactivity_analysis(isotope_key: str = "C14",
                           n_atoms: float = 6.022e23) -> dict[str, Any]:
    """Radioactive activity (becquerel) of a sample, A = lambda N. Isotope keys:
    U238, Ra226, Po210, C14, Co60, K40, free_neutron.
    e.g. radioactivity_analysis('C14', 6.022e23) -> activity of 1 mole of C-14."""
    from sigma_ground.field.interface import radioactive_decay as RD
    A, err = _safe_call(RD.activity_becquerel, isotope_key, n_atoms)
    results = {"activity_Bq": A, "activity_Ci": (A / 3.7e10 if A is not None else None)}
    errors = {"activity_Bq": err, "activity_Ci": err} if err else {}
    return ToolResult(value=results, units="Bq, Ci",
                      source="sigma_ground.field.interface.radioactive_decay",
                      provenance_tag="DERIVED",
                      formula="A = lambda N = (ln2 / t_half) N",
                      notes=_notes_from_errors(errors),
                      inputs={"isotope_key": isotope_key, "n_atoms": n_atoms}).to_dict()
