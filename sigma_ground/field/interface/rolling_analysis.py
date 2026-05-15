"""Sub-delta analysis for the rolling-window n-body shootout.

Loads fixtures/rolling_shootout_results.json (produced by rolling_shootout.py)
and computes the diagnostic statistics that turn raw errors into a fingerprint
of each predictor's failure mode.

EXPERIMENTAL DESIGN RECAP
-------------------------
For each (predictor, body, window_idx) we have an error curve e(tau) where tau
is time-since-window-start. The "sub-delta" is the difference between this
curve across adjacent windows:

    sub_delta(predictor, body, w -> w+1, tau) = e_{w+1}(tau) - e_w(tau)

If the model is correct AND the chaos horizon is far away, sub_delta ≈ 0 for
all tau (the model's residuals don't drift as we slide forward).

A non-zero sub_delta means SOMETHING about the model's relationship to
ground truth is changing across windows. Possible sources:
  - Missing physics that varies with the start state (e.g. close-approach
    geometry changes between windows)
  - Initial-condition precision floor (DE440 snapshots have finite precision)
  - Numerical instability of the integrator at the chosen dt

By comparing sub-deltas across predictors that differ in ONE physics (e.g.
over_physics vs over_physics_no_j2), we can attribute drift to that physics.

OUTPUTS
-------
1. Summary table  -- mean error & growth rate per (predictor, body)
2. Sub-delta table -- window-to-window variance per (predictor, body)
3. Ablation table -- over_physics minus over_physics_no_j2 = the J2 effect
                  -- standard minus pure_newton = the GR effect
4. Heatmap PNGs   -- error-magnitude and sub-delta-variance grids
5. Pattern report -- ranked list of biggest anomalies and best explanations

USAGE
-----
    python -m sigma_ground.field.interface.rolling_analysis
    python -m sigma_ground.field.interface.rolling_analysis --plots
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


_FIXTURES   = Path(__file__).parent / "fixtures"
_RESULTS_FILE = _FIXTURES / "rolling_shootout_results.json"
_DE440_FILE   = _FIXTURES / "de440_state_vectors.json"
_PLOTS_DIR  = _FIXTURES / "rolling_shootout_plots"

AU_KM = 1.495978707e8

# Parent body for each tracked body. For moons this is the planet they orbit;
# for planets and dwarf-planets it's the Sun. Used to build the RTN frame
# (Radial / along-Track / Normal) for position-error decomposition.
_PARENT_OF: dict[str, str] = {
    "Mercury":   "Sun",
    "Venus":     "Sun",
    "Earth":     "Sun",
    "Moon":      "Earth",
    "Mars":      "Sun",
    "Phobos":    "Mars",
    "Deimos":    "Mars",
    "Jupiter":   "Sun",
    "Io":        "Jupiter",
    "Europa":    "Jupiter",
    "Ganymede":  "Jupiter",
    "Callisto":  "Jupiter",
    "Saturn":    "Sun",
    "Enceladus": "Saturn",
    "Titan":     "Saturn",
    "Uranus":    "Sun",
    "Miranda":   "Uranus",
    "Ariel":     "Uranus",
    "Umbriel":   "Uranus",
    "Titania":   "Uranus",
    "Oberon":    "Uranus",
    "Neptune":   "Sun",
    "Triton":    "Neptune",
    "Pluto":     "Sun",     # heliocentric (could also use Pluto-Charon barycenter)
    "Charon":    "Pluto",
}


# Orbital element references (semi-major axis in AU, eccentricity, inclination
# in degrees from ecliptic). Used to correlate sub-deltas with physical parameters.
# Source: JPL Planetary Fact Sheets.
_ORBITAL: dict[str, tuple[float, float, float]] = {
    # body          a(AU)     e        i(deg)
    "Mercury":   (0.3871,  0.2056,   7.005),
    "Venus":     (0.7233,  0.0068,   3.395),
    "Earth":     (1.0000,  0.0167,   0.000),
    "Moon":      (1.0000,  0.0549,   5.145),  # ecliptic; Moon orbits Earth at 5° to ecliptic
    "Mars":      (1.5237,  0.0934,   1.850),
    "Phobos":    (1.5237,  0.0151,  26.040),  # parent Mars's heliocentric a; phobos i to Mars eq.
    "Deimos":    (1.5237,  0.0002,  27.580),
    "Jupiter":   (5.2026,  0.0484,   1.304),
    "Io":        (5.2026,  0.0041,   2.213),
    "Europa":    (5.2026,  0.0094,   1.791),
    "Ganymede":  (5.2026,  0.0011,   2.214),
    "Callisto":  (5.2026,  0.0074,   2.017),
    "Saturn":    (9.5549,  0.0539,   2.485),
    "Enceladus": (9.5549,  0.0047,  28.060),  # Enceladus i to Saturn eq + Saturn tilt
    "Titan":     (9.5549,  0.0288,  27.720),
    "Uranus":    (19.218,  0.0473,   0.770),
    "Miranda":   (19.218,  0.0013,  97.770),  # Uranus's pole is tilted 97.77° !
    "Ariel":     (19.218,  0.0012,  97.770),
    "Umbriel":   (19.218,  0.0039,  97.770),
    "Titania":   (19.218,  0.0011,  97.770),
    "Oberon":    (19.218,  0.0014,  97.770),
    "Neptune":   (30.110,  0.0086,   1.770),
    "Triton":    (30.110,  0.0000, 130.000),  # retrograde
    "Pluto":     (39.482,  0.2488,  17.160),
    "Charon":    (39.482,  0.0022,  17.160),
}


# -- RTN decomposition ----------------------------------------------------

def _build_truth_state_lookup(de440: dict) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    """Index DE440 snapshots: (round(jd), body_name) -> (pos_km_ssb, vel_km_s_ssb)."""
    out: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for _, snap in de440["snapshots"].items():
        jd = round(snap["epoch"]["jd_tdb"])
        for b in snap["bodies"]:
            sv = b["state_vector"]
            pos = np.array([sv["x_km"], sv["y_km"], sv["z_km"]])
            vel = np.array([sv["vx_km_s"], sv["vy_km_s"], sv["vz_km_s"]])
            out[(jd, b["name"])] = (pos, vel)
    return out


def _rtn_decompose(
    err_helio_km: np.ndarray,
    body_pos_ssb: np.ndarray,
    body_vel_ssb: np.ndarray,
    primary_pos_ssb: np.ndarray,
    primary_vel_ssb: np.ndarray,
) -> tuple[float, float, float]:
    """Decompose position error VECTOR (predicted - truth) into RTN.

    The RTN frame is built from the body's truth state RELATIVE to its primary:
      r_hat = (r_body - r_primary) / |...|     radial outward from primary
      n_hat = (r_rel x v_rel) / |...|          orbital-plane normal
      t_hat = n_hat x r_hat                    along-track (motion direction)

    err_helio_km is a difference of two positions, so the primary's position
    cancels under subtraction -- we can express it in any inertial frame and
    still recover the same RTN components.

    Returns (dR, dT, dN) in AU.
    """
    r_rel = body_pos_ssb - primary_pos_ssb
    v_rel = body_vel_ssb - primary_vel_ssb

    r_norm = float(np.linalg.norm(r_rel))
    if r_norm == 0.0:
        return (float("nan"), float("nan"), float("nan"))
    r_hat = r_rel / r_norm

    h_vec = np.cross(r_rel, v_rel)
    h_norm = float(np.linalg.norm(h_vec))
    if h_norm == 0.0:
        return (float("nan"), float("nan"), float("nan"))
    n_hat = h_vec / h_norm
    t_hat = np.cross(n_hat, r_hat)

    dR = float(np.dot(err_helio_km, r_hat)) / AU_KM
    dT = float(np.dot(err_helio_km, t_hat)) / AU_KM
    dN = float(np.dot(err_helio_km, n_hat)) / AU_KM
    return dR, dT, dN


def add_rtn_components(df: pd.DataFrame, de440: dict | None = None) -> pd.DataFrame:
    """Add dR_au, dT_au, dN_au columns to a results DataFrame.

    Requires df to have 'predicted_km' column. Loads DE440 fixture for the
    truth state vectors (positions and velocities).
    """
    if de440 is None:
        de440 = json.loads(_DE440_FILE.read_text())
    lookup = _build_truth_state_lookup(de440)

    dRs: list[float] = []
    dTs: list[float] = []
    dNs: list[float] = []
    for _, row in df.iterrows():
        err_au_v = row.get("error_au")
        pred_km  = row.get("predicted_km")
        if (err_au_v is None
                or (isinstance(err_au_v, float) and np.isnan(err_au_v))
                or pred_km is None or pred_km[0] is None):
            dRs.append(np.nan); dTs.append(np.nan); dNs.append(np.nan)
            continue

        sample_key = round(row["sample_jd"])
        body = row["body"]
        primary = _PARENT_OF.get(body, "Sun")

        body_state    = lookup.get((sample_key, body))
        primary_state = lookup.get((sample_key, primary))
        sun_state     = lookup.get((sample_key, "Sun"))
        if body_state is None or primary_state is None or sun_state is None:
            dRs.append(np.nan); dTs.append(np.nan); dNs.append(np.nan)
            continue

        body_pos_ssb,    body_vel_ssb    = body_state
        primary_pos_ssb, primary_vel_ssb = primary_state
        sun_pos_ssb,     _               = sun_state

        truth_helio_km = body_pos_ssb - sun_pos_ssb
        pred_helio_km  = np.array(pred_km)
        err_vec_km     = pred_helio_km - truth_helio_km

        dR, dT, dN = _rtn_decompose(
            err_vec_km, body_pos_ssb, body_vel_ssb,
            primary_pos_ssb, primary_vel_ssb,
        )
        dRs.append(dR); dTs.append(dT); dNs.append(dN)

    df = df.copy()
    df["dR_au"] = dRs
    df["dT_au"] = dTs
    df["dN_au"] = dNs
    return df


def _load_predicted_km_into_df(payload_path: Path) -> pd.DataFrame:
    """Like load_results() but ALSO keeps the predicted_km vectors (for RTN)."""
    payload = json.loads(payload_path.read_text())
    rows = []
    for s in payload["samples"]:
        rows.append({
            "predictor":       s["predictor"],
            "body":            s["body"],
            "window_start_jd": s["window_start_jd"],
            "sample_jd":       s["sample_jd"],
            "error_au":        s["error_au"],
            "predicted_km":    s.get("predicted_km"),
        })
    df = pd.DataFrame(rows)
    df["t_yr"] = (df["sample_jd"] - df["window_start_jd"]) / 365.25
    df["t_yr_bin"] = df["t_yr"].round().astype(int)
    starts = sorted(df["window_start_jd"].unique())
    start_to_idx = {s: i for i, s in enumerate(starts)}
    df["window_idx"] = df["window_start_jd"].map(start_to_idx)
    return df


def rtn_summary(df_with_rtn: pd.DataFrame) -> pd.DataFrame:
    """Per (predictor, body): RMS of each RTN component + dominant-axis label.

    The dominant-axis label says which way the error MOSTLY points:
      'T' = phase / timing error (along-track dominant)
      'R' = energy / shape error (radial dominant)
      'N' = orbital-plane error (cross-track dominant)
    """
    rows = []
    for (pred, body), grp in df_with_rtn.dropna(subset=["dR_au"]).groupby(
            ["predictor", "body"]):
        rms_R = float(np.sqrt(np.mean(grp["dR_au"] ** 2)))
        rms_T = float(np.sqrt(np.mean(grp["dT_au"] ** 2)))
        rms_N = float(np.sqrt(np.mean(grp["dN_au"] ** 2)))
        total = rms_R ** 2 + rms_T ** 2 + rms_N ** 2
        if total == 0:
            rows.append({"predictor": pred, "body": body,
                         "rms_R_au": 0.0, "rms_T_au": 0.0, "rms_N_au": 0.0,
                         "frac_R": np.nan, "frac_T": np.nan, "frac_N": np.nan,
                         "dominant": "-"})
            continue
        fR = rms_R ** 2 / total
        fT = rms_T ** 2 / total
        fN = rms_N ** 2 / total
        dominant = max(("R", fR), ("T", fT), ("N", fN), key=lambda x: x[1])[0]
        rows.append({
            "predictor": pred, "body": body,
            "rms_R_au": rms_R, "rms_T_au": rms_T, "rms_N_au": rms_N,
            "frac_R": fR, "frac_T": fT, "frac_N": fN,
            "dominant": dominant,
        })
    return pd.DataFrame(rows)


# -- Loading --------------------------------------------------------------

def load_results(path: Path = _RESULTS_FILE) -> tuple[dict, pd.DataFrame]:
    """Load the rolling-shootout JSON into a DataFrame.

    Returns
    -------
    metadata : dict (n_windows, prediction_horizon_yr, ...)
    df       : DataFrame with columns
                 predictor, body, window_start_jd, sample_jd, error_au,
                 t_yr (time since window start, in years),
                 window_idx (0-indexed for sorting)
    """
    payload = json.loads(path.read_text())
    meta = payload["metadata"]
    rows = []
    for s in payload["samples"]:
        rows.append({
            "predictor":       s["predictor"],
            "body":            s["body"],
            "window_start_jd": s["window_start_jd"],
            "sample_jd":       s["sample_jd"],
            "error_au":        s["error_au"],
        })
    df = pd.DataFrame(rows)
    df["t_yr"] = (df["sample_jd"] - df["window_start_jd"]) / 365.25
    # Normalize t_yr to integer-year bins (samples ~365.2d apart)
    df["t_yr_bin"] = df["t_yr"].round().astype(int)
    # Window index = rank of unique window_start_jd
    starts = sorted(df["window_start_jd"].unique())
    start_to_idx = {s: i for i, s in enumerate(starts)}
    df["window_idx"] = df["window_start_jd"].map(start_to_idx)
    return meta, df


# ── Aggregations ─────────────────────────────────────────────────────────

def per_predictor_body_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (predictor, body): mean & max error, error growth rate, sub-delta stats.

    Columns
    -------
    n_samples           : how many (window, t_yr) cells contributed
    mean_err_au         : mean error across all cells
    max_err_au          : worst error
    growth_rate_au_yr   : average dE/dt across windows (least-squares slope)
    growth_std_au_yr    : window-to-window std of slopes (the SUB-DELTA)
    sub_delta_rms_au    : RMS of (e_{w+1}(tau) - e_w(tau)) across all (w, tau)
    """
    rows = []
    for (pred, body), grp in df.groupby(["predictor", "body"]):
        valid = grp.dropna(subset=["error_au"])
        if len(valid) < 2:
            continue
        # Per-window slope from t_yr -> error
        slopes = []
        for w_idx, w_grp in valid.groupby("window_idx"):
            t = w_grp["t_yr"].to_numpy()
            e = w_grp["error_au"].to_numpy()
            if len(t) >= 2:
                slope, _ = np.polyfit(t, e, 1)
                slopes.append(slope)
        slopes_arr = np.array(slopes) if slopes else np.array([np.nan])

        # Sub-delta: compare error curves between adjacent windows at matching t_yr_bin
        sub_delta_rms = _sub_delta_rms(valid)

        rows.append({
            "predictor":          pred,
            "body":               body,
            "n_samples":          int(len(valid)),
            "mean_err_au":        float(valid["error_au"].mean()),
            "max_err_au":         float(valid["error_au"].max()),
            "growth_rate_au_yr":  float(np.nanmean(slopes_arr)),
            "growth_std_au_yr":   float(np.nanstd(slopes_arr)),
            "sub_delta_rms_au":   float(sub_delta_rms),
        })
    return pd.DataFrame(rows)


def _sub_delta_rms(grp: pd.DataFrame) -> float:
    """RMS of e_{w+1}(tau) - e_w(tau) over all matched (w, tau) pairs within one (predictor, body)."""
    # Pivot to (window_idx, t_yr_bin) -> error_au
    piv = grp.pivot_table(index="window_idx", columns="t_yr_bin",
                          values="error_au", aggfunc="mean")
    if piv.shape[0] < 2:
        return float("nan")
    diffs = piv.diff(axis=0).iloc[1:].to_numpy()
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(diffs ** 2)))


def ablation_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Isolate individual physics contributions.

    j2_effect_au_yr  = growth_rate(over_physics)  - growth_rate(over_physics_no_j2)
    srp_effect_au_yr = growth_rate(over_physics_no_j2) - growth_rate(standard)
    gr_effect_au_yr  = growth_rate(standard)      - growth_rate(pure_newton)

    Positive values mean the more-physics predictor has LARGER error growth
    (i.e. enabling that physics made things WORSE for that body -- likely a
    bug, numerical issue, or genuine model mismatch with truth).

    Negative values mean enabling that physics CLOSED the gap.
    """
    wide = summary.pivot(index="body", columns="predictor", values="growth_rate_au_yr")
    out = pd.DataFrame(index=wide.index)
    if "over_physics" in wide and "over_physics_no_j2" in wide:
        out["j2_effect_au_yr"] = wide["over_physics"] - wide["over_physics_no_j2"]
    if "over_physics_no_j2" in wide and "standard" in wide:
        out["srp_effect_au_yr"] = wide["over_physics_no_j2"] - wide["standard"]
    if "standard" in wide and "pure_newton" in wide:
        out["gr_effect_au_yr"] = wide["standard"] - wide["pure_newton"]
    return out


def cross_predictor_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Per body: correlation matrix of error curves across predictors.

    For each body, build a long-form table (window_idx, t_yr_bin, predictor -> error_au),
    pivot to get a column per predictor, and compute the predictor-predictor
    correlation matrix. High correlation = predictors agree on this body's
    error pattern (likely shared cause: data limitation, body chaos, etc.).
    Low correlation = predictors disagree (physics-level difference).
    """
    rows = []
    for body, grp in df.dropna(subset=["error_au"]).groupby("body"):
        piv = grp.pivot_table(
            index=["window_idx", "t_yr_bin"], columns="predictor",
            values="error_au", aggfunc="mean",
        ).dropna()
        if piv.shape[0] < 5:  # need enough cells for a meaningful correlation
            continue
        corr = piv.corr()
        # Flatten to long form for plotting / display
        for p1 in corr.index:
            for p2 in corr.columns:
                if p1 >= p2:
                    continue
                rows.append({
                    "body": body, "predictor_a": p1, "predictor_b": p2,
                    "corr": float(corr.loc[p1, p2]),
                })
    return pd.DataFrame(rows)


def orbital_correlation(summary: pd.DataFrame) -> pd.DataFrame:
    """Correlate each predictor's growth rate / sub-delta with orbital elements.

    If, for predictor X, growth_rate correlates strongly with inclination across
    bodies, that's a signature of a missing inclination-coupled physics term
    (typically J2 or non-equatorial perturbations).
    """
    rows = []
    for pred, grp in summary.groupby("predictor"):
        grp = grp.copy()
        grp["a_au"] = grp["body"].map(lambda b: _ORBITAL.get(b, (np.nan,)*3)[0])
        grp["e"]    = grp["body"].map(lambda b: _ORBITAL.get(b, (np.nan,)*3)[1])
        grp["i_deg"]= grp["body"].map(lambda b: _ORBITAL.get(b, (np.nan,)*3)[2])
        valid = grp.dropna(subset=["a_au", "growth_rate_au_yr"])
        if len(valid) < 4:
            continue
        for metric in ("growth_rate_au_yr", "growth_std_au_yr", "sub_delta_rms_au"):
            for elem in ("a_au", "e", "i_deg"):
                c = valid[metric].corr(valid[elem])
                rows.append({"predictor": pred, "metric": metric,
                             "orbital_elem": elem, "corr": float(c)})
    return pd.DataFrame(rows)


# ── Reporting ────────────────────────────────────────────────────────────

def print_report(meta: dict, df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("ROLLING-WINDOW SHOOTOUT -- SUB-DELTA ANALYSIS REPORT")
    print("=" * 78)
    print(f"\nWindows:    {meta['n_windows']}   "
          f"({meta['window_start_keys'][0]} -> {meta['window_start_keys'][-1]})")
    print(f"Horizon:    {meta['prediction_horizon_yr']:.0f} yr")
    print(f"Predictors: {', '.join(meta['predictors'])}")
    print(f"Samples:    {len(df):,}")

    summary = per_predictor_body_summary(df)

    # ── 1. Top-line: which predictor "wins" each body? ────────────────────
    print("\n" + "-" * 78)
    print("WINNERS (lowest mean error across all windows x times)")
    print("-" * 78)
    pivot = summary.pivot(index="body", columns="predictor", values="mean_err_au")
    if not pivot.empty:
        wins = pivot.idxmin(axis=1).value_counts()
        for pred, n in wins.items():
            print(f"  {pred:<22s} wins for {n} bodies")

    # ── 2. Ablation: contribution of each physics layer ────────────────────
    ab = ablation_table(summary)
    if not ab.empty:
        print("\n" + "-" * 78)
        print("ABLATION -- growth-rate difference, AU/yr")
        print("(positive = enabling this physics made the error grow FASTER)")
        print("(negative = enabling this physics CLOSED the gap)")
        print("-" * 78)
        cols = [c for c in ("gr_effect_au_yr", "srp_effect_au_yr", "j2_effect_au_yr") if c in ab.columns]
        display = ab[cols].copy()
        # Order bodies by magnitude of J2 effect (if present), else GR
        sort_col = "j2_effect_au_yr" if "j2_effect_au_yr" in display.columns else cols[0]
        display = display.reindex(display[sort_col].abs().sort_values(ascending=False).index)
        print(display.head(15).to_string(float_format=lambda x: f"{x:+.3e}"))

    # ── 3. Cross-predictor correlation ─────────────────────────────────────
    corr_df = cross_predictor_correlation(df)
    if not corr_df.empty:
        print("\n" + "-" * 78)
        print("CROSS-PREDICTOR CORRELATION OF ERROR CURVES")
        print("(high = predictors agree -> shared cause; low = physics-level disagreement)")
        print("-" * 78)
        # Average correlation across predictor pairs, by body
        body_corr = corr_df.groupby("body")["corr"].mean().sort_values()
        print("Lowest-correlation bodies (predictors most divergent):")
        for body, c in body_corr.head(8).items():
            print(f"  {body:<12s}  mean inter-predictor corr = {c:+.3f}")
        print("\nHighest-correlation bodies (predictors most aligned):")
        for body, c in body_corr.tail(5).items():
            print(f"  {body:<12s}  mean inter-predictor corr = {c:+.3f}")

    # ── 4. Orbital-element correlations ───────────────────────────────────
    oc = orbital_correlation(summary)
    if not oc.empty:
        print("\n" + "-" * 78)
        print("ORBITAL-ELEMENT CORRELATIONS (Pearson r)")
        print("(strong correlation hints at the physics-dimension of the residual)")
        print("-" * 78)
        # Show the growth_rate_au_yr correlations for each predictor
        gr = oc[oc["metric"] == "growth_rate_au_yr"]
        if not gr.empty:
            wide = gr.pivot(index="predictor", columns="orbital_elem", values="corr")
            print("\ngrowth_rate_au_yr correlated with:")
            print(wide.to_string(float_format=lambda x: f"{x:+.3f}"))
        sd = oc[oc["metric"] == "sub_delta_rms_au"]
        if not sd.empty:
            wide_sd = sd.pivot(index="predictor", columns="orbital_elem", values="corr")
            print("\nsub_delta_rms_au correlated with:")
            print(wide_sd.to_string(float_format=lambda x: f"{x:+.3f}"))

    # ── 5. Outliers: (predictor, body) cells with anomalous sub-delta ─────
    print("\n" + "-" * 78)
    print("TOP-10 ANOMALIES -- largest sub_delta_rms relative to mean error")
    print("(big sub-delta means error PATTERN is unstable across windows)")
    print("-" * 78)
    summary = summary.copy()
    summary["sub_delta_ratio"] = summary["sub_delta_rms_au"] / summary["mean_err_au"]
    summary = summary.replace([np.inf, -np.inf], np.nan).dropna(subset=["sub_delta_ratio"])
    top = summary.nlargest(10, "sub_delta_ratio")
    print(top[["predictor", "body", "mean_err_au", "sub_delta_rms_au", "sub_delta_ratio"]]
          .to_string(index=False,
                     float_format=lambda x: f"{x:.4e}"))


# ── Plots ────────────────────────────────────────────────────────────────

def save_plots(meta: dict, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Heatmap of mean error per (predictor, body)
    pivot_mean = summary.pivot(index="body", columns="predictor", values="mean_err_au")
    pivot_mean_log = np.log10(pivot_mean.clip(lower=1e-10))
    fig, ax = plt.subplots(figsize=(8, max(6, 0.3 * len(pivot_mean))))
    im = ax.imshow(pivot_mean_log.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot_mean.columns)))
    ax.set_xticklabels(pivot_mean.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot_mean.index)))
    ax.set_yticklabels(pivot_mean.index, fontsize=8)
    plt.colorbar(im, ax=ax, label="log10(mean error / AU)")
    ax.set_title(f"Mean prediction error\n"
                 f"({meta['n_windows']} windows, {meta['prediction_horizon_yr']}y horizon)")
    plt.tight_layout()
    plt.savefig(_PLOTS_DIR / "heatmap_mean_error.png", dpi=120)
    plt.close(fig)

    # 2. Heatmap of sub-delta RMS (per predictor x body)
    pivot_sd = summary.pivot(index="body", columns="predictor", values="sub_delta_rms_au")
    pivot_sd_log = np.log10(pivot_sd.clip(lower=1e-10))
    fig, ax = plt.subplots(figsize=(8, max(6, 0.3 * len(pivot_sd))))
    im = ax.imshow(pivot_sd_log.to_numpy(), aspect="auto", cmap="magma")
    ax.set_xticks(range(len(pivot_sd.columns)))
    ax.set_xticklabels(pivot_sd.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot_sd.index)))
    ax.set_yticklabels(pivot_sd.index, fontsize=8)
    plt.colorbar(im, ax=ax, label="log10(sub-delta RMS / AU)")
    ax.set_title(f"Window-to-window error-curve sub-delta RMS\n"
                 f"(high = error pattern is unstable across windows)")
    plt.tight_layout()
    plt.savefig(_PLOTS_DIR / "heatmap_subdelta.png", dpi=120)
    plt.close(fig)

    # 3. Per-body error curves (one panel per body, all windows x predictors)
    bodies = sorted(df["body"].unique())
    predictors = sorted(df["predictor"].unique())
    n_cols = 4
    n_rows = (len(bodies) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.0 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    colors = plt.cm.tab10.colors[: len(predictors)]
    for i, body in enumerate(bodies):
        ax = axes[i]
        body_df = df[df["body"] == body].dropna(subset=["error_au"])
        for j, pred in enumerate(predictors):
            sub = body_df[body_df["predictor"] == pred]
            for w_idx, w_grp in sub.groupby("window_idx"):
                w_grp = w_grp.sort_values("t_yr")
                ax.plot(w_grp["t_yr"], w_grp["error_au"],
                        alpha=0.35, color=colors[j], lw=0.7)
            # Overlay the median curve in solid
            piv = sub.pivot_table(index="t_yr_bin", columns="window_idx",
                                  values="error_au", aggfunc="mean")
            ax.plot(piv.index, piv.median(axis=1),
                    color=colors[j], lw=1.8,
                    label=pred if i == 0 else None)
        ax.set_yscale("log")
        ax.set_title(body, fontsize=9)
        ax.set_xlabel("t since window start (yr)", fontsize=7)
        ax.set_ylabel("error (AU)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    for k in range(len(bodies), len(axes)):
        axes[k].axis("off")
    fig.legend(loc="lower right", fontsize=8)
    fig.suptitle("Error curves: faint=each window, solid=median",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(_PLOTS_DIR / "per_body_curves.png", dpi=120)
    plt.close(fig)

    print(f"\nPlots saved to {_PLOTS_DIR}/")


# -- Fingerprint diff -----------------------------------------------------

def fingerprint_diff(baseline_path: Path, new_path: Path,
                     focus_predictors: list[str] | None = None) -> None:
    """Print a side-by-side diff of two result JSONs.

    For each (predictor, body) cell, compares mean_err, growth_rate, and
    sub_delta_rms between baseline and new. Sorts by absolute improvement.
    """
    _, df_b = load_results(baseline_path)
    _, df_n = load_results(new_path)

    sum_b = per_predictor_body_summary(df_b)
    sum_n = per_predictor_body_summary(df_n)

    merged = sum_b.merge(
        sum_n, on=["predictor", "body"], how="outer",
        suffixes=("_base", "_new"), indicator=True,
    )

    if focus_predictors:
        merged = merged[merged["predictor"].isin(focus_predictors)]

    in_both = merged[merged["_merge"] == "both"].copy()
    new_only = merged[merged["_merge"] == "right_only"].copy()

    if not in_both.empty:
        in_both["mean_err_ratio"] = in_both["mean_err_au_new"] / in_both["mean_err_au_base"]
        in_both["subdelta_ratio"] = in_both["sub_delta_rms_au_new"] / in_both["sub_delta_rms_au_base"]

        print("\n" + "=" * 78)
        print("FINGERPRINT DIFF -- baseline vs new")
        print("=" * 78)
        print(f"baseline: {baseline_path.name}")
        print(f"new:      {new_path.name}")
        print()

        print("-" * 78)
        print("TOP-15 IMPROVEMENTS (largest mean-error reduction)")
        print("-" * 78)
        improvements = in_both[in_both["mean_err_ratio"] < 0.999].sort_values(
            "mean_err_ratio")
        cols = ["predictor", "body", "mean_err_au_base", "mean_err_au_new", "mean_err_ratio"]
        if not improvements.empty:
            print(improvements[cols].head(15).to_string(
                index=False, float_format=lambda x: f"{x:.4e}"))
        else:
            print("  (none)")

        print()
        print("-" * 78)
        print("TOP-10 REGRESSIONS (mean error got WORSE)")
        print("-" * 78)
        regressions = in_both[in_both["mean_err_ratio"] > 1.001].sort_values(
            "mean_err_ratio", ascending=False)
        if not regressions.empty:
            print(regressions[cols].head(10).to_string(
                index=False, float_format=lambda x: f"{x:.4e}"))
        else:
            print("  (none -- no cell got worse)")

        print()
        print("-" * 78)
        print("SUB-DELTA SHIFT -- pattern-stability change")
        print("(ratio < 1 means error pattern is now more stable across windows)")
        print("-" * 78)
        sd_top = in_both.dropna(subset=["subdelta_ratio"]).sort_values(
            "subdelta_ratio").head(15)
        sd_cols = ["predictor", "body", "sub_delta_rms_au_base",
                   "sub_delta_rms_au_new", "subdelta_ratio"]
        if not sd_top.empty:
            print(sd_top[sd_cols].to_string(
                index=False, float_format=lambda x: f"{x:.4e}"))

    if not new_only.empty:
        print()
        print("-" * 78)
        print("NEW PREDICTORS (not in baseline)")
        print("-" * 78)
        for pred, grp in new_only.groupby("predictor"):
            print(f"\n  {pred} -- top-10 worst bodies:")
            cols2 = ["body", "mean_err_au_new", "sub_delta_rms_au_new"]
            top = grp.nlargest(10, "mean_err_au_new")[cols2]
            print(top.to_string(index=False, float_format=lambda x: f"{x:.4e}"))


# -- Entry point ----------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze rolling-window shootout results")
    parser.add_argument("--results", type=Path, default=_RESULTS_FILE,
                        help="path to results JSON")
    parser.add_argument("--plots", action="store_true",
                        help="also generate matplotlib heatmaps + per-body curves")
    parser.add_argument("--diff", type=Path, default=None,
                        help="path to baseline JSON to diff against; when given, "
                             "--results is the NEW file and we print the diff")
    parser.add_argument("--focus", nargs="+", default=None,
                        help="restrict diff to these predictors")
    parser.add_argument("--rtn", action="store_true",
                        help="also print RTN (Radial/Track/Normal) decomposition")
    args = parser.parse_args()

    if args.diff is not None:
        if not args.diff.exists():
            raise SystemExit(f"baseline file not found: {args.diff}")
        if not args.results.exists():
            raise SystemExit(f"new results file not found: {args.results}")
        fingerprint_diff(args.diff, args.results, focus_predictors=args.focus)
    else:
        if not args.results.exists():
            raise SystemExit(f"results file not found: {args.results}\n"
                             f"Run rolling_shootout.py first to generate it.")
        meta, df = load_results(args.results)
        print_report(meta, df)

        if args.rtn:
            print("\n" + "=" * 78)
            print("RTN DECOMPOSITION -- where in the orbit does the error live?")
            print("=" * 78)
            df_rtn_src = _load_predicted_km_into_df(args.results)
            df_rtn = add_rtn_components(df_rtn_src)
            rtn = rtn_summary(df_rtn)

            print("\nDominant-axis tally per predictor:")
            tally = rtn.groupby(["predictor", "dominant"]).size().unstack(
                fill_value=0)
            print(tally.to_string())

            print("\nTop-15 cells by total RTN RMS (largest errors first):")
            rtn["rms_total_au"] = np.sqrt(
                rtn["rms_R_au"]**2 + rtn["rms_T_au"]**2 + rtn["rms_N_au"]**2)
            cols = ["predictor", "body", "rms_R_au", "rms_T_au", "rms_N_au",
                    "frac_R", "frac_T", "frac_N", "dominant"]
            print(rtn.nlargest(15, "rms_total_au")[cols].to_string(
                index=False,
                float_format=lambda x: f"{x:.3e}" if abs(x) < 0.01 or abs(x) > 100 else f"{x:.3f}"))

            print("\nFor key bodies -- RTN breakdown across predictors:")
            for body in ("Mercury", "Io", "Europa", "Miranda", "Ariel",
                         "Ganymede", "Earth", "Jupiter"):
                rows = rtn[rtn["body"] == body]
                if rows.empty:
                    continue
                print(f"\n  {body}:")
                print(rows[cols[:-1] + ["dominant"]].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.3e}" if abs(x) < 0.01 else f"{x:.3f}"))

        if args.plots:
            summary = per_predictor_body_summary(df)
            save_plots(meta, df, summary)
