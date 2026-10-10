"""U2 pre-registration — written and frozen BEFORE any N=64 echo is measured.
Run once; writes prereg.json with a UTC timestamp. Do not edit after the runs start."""
import json, datetime, os
HERE = os.path.dirname(os.path.abspath(__file__))
prereg = {
 "frozen_utc": datetime.datetime.utcnow().isoformat() + "Z",
 "original_design_criterion_verbatim": (
  "SEPARATE if eps_rec(N-body), eps_rec(SPH alpha=0) <= 1e-8 R and shrink under dt-halving while "
  "eps_rec(SPH alpha=1) >= 1e-3 R and does not, with min S_dot_visc >= 0 and ledger closure <= 1e-3; "
  "SAME if all three within 10x; growth above 1e-8 that halves with dt is Lyapunov, not irreversibility. "
  "S8 sentinel: sum m u_int(T) = Q_cum(T) to 1e-3 and K + W_soft + U_int conserved to 1e-4."),
 "evaluation_point": "primary verdict at T = 1 t_ff (the X2 trimmed design); T = 0.3 t_ff reported alongside",
 "design_corrections_identified_before_running": [
  "C1 'shrink under dt-halving' is wrong for the Hamiltonian engines: velocity-Verlet with position-only forces is exactly "
  "time-symmetric, so the echo error is roundoff x chaotic amplification, not truncation. Halving dt doubles the roundoff "
  "injections and leaves the amplification unchanged; expected ratio eps(dt/2)/eps(dt) ~ 1-2, not < 1. The original "
  "criterion therefore cannot return SEPARATE whatever the physics. Corrected discriminator below (perturbation-size control).",
  "C2 'ledger closure' against the softened mechanical energy K + U_soft: with eos='isothermal' the PdV work also goes to "
  "the bath, so the closure only holds if the mechanical energy includes the isothermal free energy F = c_s^2 sum m ln rho. "
  "Literal closure (K + U_soft only) reported too; predicted to fail for alpha=1 (density field does not return).",
  "C3 S8 sentinel 'sum m u_int(T) = Q_cum(T)': for eos='isothermal' u_int == 0 by design (work goes to scene.energy_bath_J); "
  "for eos='ideal' Delta U_int = Q_visc + W_PdV, not Q_visc alone. Literal sentinel predicted to FAIL for both EOS for "
  "this reason, not for an engine defect. Corrected sentinel: |Delta(K + U_soft + U_int + bath)| / |U_soft(0)| <= 1e-4, and "
  "the engine's own viscous share (pair-decomposed) equals the independent ledger Q_led to 1e-3 at dt = t_ff/1200.",
  "C4 min S_dot_visc >= 0 holds by construction (Pi_ij > 0 only for approaching pairs; cubic-spline W' <= 0), as does the "
  "Fourier face-by-face k dT^2/(Ta Tb) >= 0: engine validation, not evidence.",
  "C5 Reversibility of N-body and inviscid SPH follows from the symplectic integrator and position-only forces; it holds "
  "for any force law and any constants. SEPARATE is engine validation of where the closure terms were put, not evidence "
  "about the unification thesis."],
 "corrected_criterion": {
  "SEPARATE": [
   "H1: eps_rec(N-body) and eps_rec(SPH alpha=0, isothermal) <= 1e-8 R at both dt, OR above 1e-8 R but linear in an injected "
   "displacement delta (log-slope in [0.7, 1.3] over delta points with 3 eps_rec(0) <= eps_rec(delta) <= 1e-2 R) = Lyapunov",
   "H2: Hamiltonian eps_rec is dt-independent at roundoff: eps(dt/2)/eps(dt) in [0.1, 10]",
   "D1: eps_rec(SPH alpha=1) >= 1e-3 R at both dt AND dt-converged |eps(dt/2)/eps(dt) - 1| <= 0.1 AND "
   "delta-independent |eps(delta=1e-9 R)/eps(0) - 1| <= 0.1",
   "D2: min S_dot_led >= 0 every step (by construction) and engine per-step viscous booking q_visc >= 0 every step",
   "D3: corrected ledger closure |Q_led - (-Delta(K + U_soft + F))| / Q_led <= 1e-3 at dt = t_ff/1200"],
  "SAME": "all three eps_rec within 10x of each other AND the Hamiltonian echo errors are delta-independent (genuine loss)",
  "otherwise": "MIXED (report which clause failed)"},
 "predictions": {
  "nbody_eps_rec_T0.3": "1e-16 .. 1e-14 R",
  "nbody_eps_rec_T1": "<= 1e-8 R (moderate confidence: collapse to a rotating disk near t_ff amplifies roundoff)",
  "sph_a0_eps_rec_T0.3": "1e-16 .. 1e-14 R",
  "sph_a0_eps_rec_T1": "<= 1e-12 R (pressure-supported cloud, 2K_th/|W| = 1.25)",
  "sph_a1_eps_rec_T0.3": "~1e-2 R (smoke at N=32: 1.5e-2)",
  "sph_a1_eps_rec_T1": "1e-2 .. 1e-1 R, dt-ratio within 0.9-1.1",
  "dt_halving_hamiltonian_ratio": "1 .. 2 (fails the original 'shrink' clause)",
  "delta_control": "Hamiltonian slope ~ 1; viscous flat",
  "S_dot_min": ">= 0 (by construction)",
  "corrected_closure": "1e-4 .. 1e-2, scaling ~ dt (engine viscous force uses pre-kick velocities)",
  "ideal_gas_alpha0_echo": "NOT roundoff: the cached end-of-step force uses the pre-update temperature, which breaks "
                            "discrete time-symmetry; expect eps_rec >> 1e-14 R and ~ proportional to dt (ratio ~0.5)",
  "ideal_sentinel_energy": "|Delta(K + U_soft + U_int)| / |U_soft(0)| <= 1e-4 (O(dt^2) from gravity only)",
  "fourier": "S_dot >= 0 every slice (by construction), S monotone, energy conserved to roundoff; engine refuses a reversed run",
  "verdict_corrected": "SEPARATE", "verdict_original": "not SEPARATE (fails the dt-halving 'shrink' clause) -> MIXED"},
}
with open(os.path.join(HERE, "prereg.json"), "w", encoding="utf-8") as f:
    json.dump(prereg, f, indent=1)
print("frozen", prereg["frozen_utc"])
