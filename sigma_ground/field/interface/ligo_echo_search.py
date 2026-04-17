"""
Phase I — LIGO matched-filter search for ξ-shell echoes in GW ringdown.

Pulls public GWOSC strain data for named events, fits the dominant QNM
to the ringdown, subtracts it, whitens the residual against a PSD
estimated from pre-merger quiet data, and matched-filters a
QNM-shaped template at the sigma-ground-predicted echo delays
Δt_n = 2·r_s·n·σ_conv/c.  Background SNR distribution is estimated
from random off-source delays that are NOT multiples of Δt_1.

──────────────────────────────────────────────────────────────────────────────
USE
──────────────────────────────────────────────────────────────────────────────

    from sigma_ground.field.interface.ligo_echo_search import search_event

    result = search_event('GW150914', n_echoes=5, n_background=2000)
    print(result['combined_snr'], result['p_value'])

──────────────────────────────────────────────────────────────────────────────
DEPENDENCIES
──────────────────────────────────────────────────────────────────────────────

Pure-pip: numpy, scipy, requests, h5py.  No lalsuite / pycbc / CMake.

──────────────────────────────────────────────────────────────────────────────
CAVEATS
──────────────────────────────────────────────────────────────────────────────

- Single-detector per-event analysis (no inter-detector coherence).
- QNM fit is a damped-sinusoid proxy, not a full Teukolsky mode fit.
- Whitening uses Welch PSD from 8 seconds of pre-merger quiet data.
- Background uses random delays in [10·Δt_1, 500 ms] excluding predicted
  multiples of Δt_1 ± 1 ms.

These are adequate for a first-pass Phase I bound; a sharper re-analysis
would use pycbc matched-filter infrastructure and QNM templates from
SXS waveforms.
"""

import os
import math
import time

import numpy as np
import h5py
import requests
from scipy import signal, optimize

from sigma_ground.field.constants import SIGMA_CONV, XI


G_SI = 6.67430e-11
C_SI = 299792458.0
M_SUN_KG = 1.98892e30


# Per-event metadata.
# M_rem and a_spin: best-fit remnant parameters from GWTC LVC papers.
# f_qnm and tau_qnm: Berti-Cardoso-Starinets fitting formulas for Kerr
#   2,2,0 fundamental mode (approximate — used as initial guess for fit).
# file_start_gps: GPS of 32-s-window strain segment start.
# strain_urls: direct GWOSC event-API HDF5 URLs (verified 2026-04).
EVENTS = {
    'GW150914': {
        'gps_event': 1126259462.4,
        'M_rem_msun': 62.2,
        'a_spin': 0.67,
        'f_qnm_hz': 251.0,
        'tau_qnm_s': 4.0e-3,
        'file_start_gps': 1126259447,
        'duration_s': 32,
        'strain_urls': {
            'H1_4k': 'https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/H-H1_GWOSC_4KHZ_R1-1126259447-32.hdf5',
            'L1_4k': 'https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/L-L1_GWOSC_4KHZ_R1-1126259447-32.hdf5',
            'H1_16k': 'https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/H-H1_GWOSC_16KHZ_R1-1126259447-32.hdf5',
            'L1_16k': 'https://gwosc.org/eventapi/json/GWTC-1-confident/GW150914/v3/L-L1_GWOSC_16KHZ_R1-1126259447-32.hdf5',
        },
    },
    'GW151226': {
        'gps_event': 1135136350.6,
        'M_rem_msun': 20.8,
        'a_spin': 0.74,
        'f_qnm_hz': 737.0,
        'tau_qnm_s': 1.2e-3,
        'file_start_gps': 1135136335,
        'duration_s': 32,
        'strain_urls': {
            'H1_4k': 'https://gwosc.org/eventapi/json/GWTC-1-confident/GW151226/v2/H-H1_GWOSC_4KHZ_R1-1135136335-32.hdf5',
            'L1_4k': 'https://gwosc.org/eventapi/json/GWTC-1-confident/GW151226/v2/L-L1_GWOSC_4KHZ_R1-1135136335-32.hdf5',
            'H1_16k': 'https://gwosc.org/eventapi/json/GWTC-1-confident/GW151226/v2/H-H1_GWOSC_16KHZ_R1-1135136335-32.hdf5',
            'L1_16k': 'https://gwosc.org/eventapi/json/GWTC-1-confident/GW151226/v2/L-L1_GWOSC_16KHZ_R1-1135136335-32.hdf5',
        },
    },
}


def schwarzschild_radius_m(M_msun):
    """Schwarzschild radius of a mass in solar units, in metres."""
    M_kg = M_msun * M_SUN_KG
    return 2.0 * G_SI * M_kg / (C_SI ** 2)


def echo_delay_n(M_msun, n):
    """Predicted n-th ξ-shell echo delay in seconds.

    Δt_n = 2 · r_s · n · σ_conv / c
    """
    r_s = schwarzschild_radius_m(M_msun)
    return 2.0 * r_s * n * SIGMA_CONV / C_SI


def fetch_strain(event_name, detector, fs_khz=4,
                 cache_dir='local-cache/gwosc', timeout=120):
    """Download strain HDF5 for (event, detector), return local path."""
    os.makedirs(cache_dir, exist_ok=True)
    key = f'{detector}_{fs_khz}k'
    url = EVENTS[event_name]['strain_urls'][key]
    fname = os.path.basename(url)
    path = os.path.join(cache_dir, fname)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    r = requests.get(url, timeout=timeout, stream=True)
    r.raise_for_status()
    with open(path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    if os.path.getsize(path) < 1000:
        raise RuntimeError(f'Downloaded file too small: {path}')
    return path


def read_strain(path):
    """Read HDF5 strain file.  Returns (strain, t_start_gps, fs)."""
    with h5py.File(path, 'r') as f:
        strain = np.array(f['strain/Strain'][:])
        dt = f['strain/Strain'].attrs['Xspacing']
        t0 = f['meta/GPSstart'][()]
    fs = 1.0 / float(dt)
    return strain.astype(np.float64), float(t0), fs


def estimate_psd(strain, fs, segment_start_idx, segment_length_samples,
                 nperseg_seconds=4.0):
    """Welch PSD from a pre-merger quiet segment."""
    seg = strain[segment_start_idx: segment_start_idx + segment_length_samples]
    nperseg = int(min(nperseg_seconds * fs, len(seg) // 2))
    freqs, psd = signal.welch(seg, fs=fs, nperseg=nperseg,
                              detrend='constant', window='hann')
    return freqs, psd


def whiten(data, fs, psd_freqs, psd_vals):
    """Frequency-domain whitening: FFT(data) / sqrt(PSD · fs / 2)."""
    n = len(data)
    window = signal.windows.tukey(n, alpha=0.2)
    x = data * window
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    X = np.fft.rfft(x)
    psd_interp = np.interp(freqs, psd_freqs, psd_vals,
                           left=psd_vals[1], right=psd_vals[-1])
    floor = np.nanmax(psd_interp) * 1e-10
    psd_interp = np.maximum(psd_interp, floor)
    norm = np.sqrt(psd_interp * fs / 2.0)
    Xw = X / norm
    return np.fft.irfft(Xw, n=n).astype(np.float64)


def bandpass(data, fs, lo=35.0, hi=None):
    """Butterworth bandpass to tame low-freq seismic and Nyquist edge."""
    if hi is None:
        hi = fs / 2.0 * 0.45
    sos = signal.butter(4, [lo, hi], btype='bandpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, data)


def qnm_damped_sinusoid(t, A, f, tau, phi):
    """Single-mode ringdown template."""
    return A * np.exp(-t / tau) * np.cos(2.0 * np.pi * f * t + phi)


def fit_qnm(data, fs, start_idx, duration_ms=20.0,
            f_guess=250.0, tau_guess=4e-3):
    """Fit damped-sinusoid to data[start_idx : start_idx + duration].

    Two-parameter linear least squares with f and tau fixed from the
    caller-supplied literature values.  Decomposes

        y(t) = A · exp(-t/τ) · cos(2πft + φ)
             = exp(-t/τ) · [A·cos(φ)·cos(2πft) − A·sin(φ)·sin(2πft)]
             = c · g(t) + s · h(t)

    where g(t) = exp(-t/τ)·cos(2πft),  h(t) = −exp(-t/τ)·sin(2πft).
    Solve (c, s) by lstsq, then A = √(c² + s²), φ = atan2(s, c).

    Deterministic, closed-form, no scale sensitivity.  Returns
    (popt, model, start_idx, end_idx); popt = [A, f, τ, φ].
    """
    n_samples = int(duration_ms * 1e-3 * fs)
    end_idx = start_idx + n_samples
    if end_idx > len(data):
        return None, None, start_idx, end_idx
    t = np.arange(n_samples) / fs
    y = np.asarray(data[start_idx:end_idx], dtype=np.float64)
    if len(y) < n_samples:
        return None, None, start_idx, end_idx
    envelope = np.exp(-t / tau_guess)
    g = envelope * np.cos(2.0 * np.pi * f_guess * t)
    h = -envelope * np.sin(2.0 * np.pi * f_guess * t)
    M = np.column_stack([g, h])
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    c, s = float(coef[0]), float(coef[1])
    A = math.sqrt(c * c + s * s)
    phi = math.atan2(s, c)
    popt = np.array([A, float(f_guess), float(tau_guess), phi])
    model = qnm_damped_sinusoid(t, A, f_guess, tau_guess, phi)
    return popt, model, start_idx, end_idx


_KERR_MODE_CACHE = {}


def _kerr_overtone_spectrum(M_msun, a_spin, n_overtones=3, l=2, m=2, s=-2):
    """Kerr QNM frequencies and damping times for n=0..n_overtones-1.

    Uses the `qnm` library.  Returns list of (f_Hz, tau_s) per overtone.
    """
    import qnm as qnm_lib  # Pure-python after initial cache warm.
    M_kg = M_msun * M_SUN_KG
    M_seconds = G_SI * M_kg / (C_SI ** 3)
    out = []
    for n in range(n_overtones):
        key = (s, l, m, n)
        mode = _KERR_MODE_CACHE.get(key)
        if mode is None:
            mode = qnm_lib.modes_cache(s=s, l=l, m=m, n=n)
            _KERR_MODE_CACHE[key] = mode
        omega, _A, _C = mode(a=a_spin)
        # qnm's ω is dimensionless (M·ω); damped modes have ω.imag < 0.
        f_hz = float(omega.real) / (2.0 * np.pi * M_seconds)
        tau_s = -M_seconds / float(omega.imag)
        out.append((f_hz, tau_s))
    return out


def fit_kerr_overtones(data, fs, start_idx, M_msun, a_spin,
                       duration_ms=20.0, n_overtones=3):
    """Linear LS fit of Kerr overtone damped sinusoids to data.

    For n=0..n_overtones-1 the Kerr spectrum fixes (f_n, τ_n) from
    `qnm`.  Each mode contributes a two-column basis
      g_n(t) = exp(-t/τ_n)·cos(2πf_n·t)
      h_n(t) = −exp(-t/τ_n)·sin(2πf_n·t)
    and the full basis M = [g_0, h_0, g_1, h_1, ...].  Solve (c_n, s_n)
    by lstsq; recover A_n = √(c_n² + s_n²), φ_n = atan2(s_n, c_n).

    Returns (modes, combined_model_at_fit_window, start_idx, end_idx)
    where modes is a list of per-overtone dicts.
    """
    n_samples = int(duration_ms * 1e-3 * fs)
    end_idx = start_idx + n_samples
    if end_idx > len(data):
        return None, None, start_idx, end_idx
    spec = _kerr_overtone_spectrum(M_msun, a_spin, n_overtones=n_overtones)
    t = np.arange(n_samples) / fs
    y = np.asarray(data[start_idx:end_idx], dtype=np.float64)
    cols = []
    for f, tau in spec:
        env = np.exp(-t / tau)
        cols.append(env * np.cos(2.0 * np.pi * f * t))
        cols.append(-env * np.sin(2.0 * np.pi * f * t))
    M = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    modes = []
    combined = np.zeros(n_samples, dtype=np.float64)
    for i, (f, tau) in enumerate(spec):
        c, s = float(coef[2 * i]), float(coef[2 * i + 1])
        A = math.sqrt(c * c + s * s)
        phi = math.atan2(s, c)
        env = np.exp(-t / tau)
        combined += A * env * np.cos(2.0 * np.pi * f * t + phi)
        modes.append({'n': i, 'f_hz': float(f), 'tau_s': float(tau),
                      'A': A, 'phi': phi, 'c_coef': c, 's_coef': s})
    return modes, combined, start_idx, end_idx


def evaluate_overtone_sum(modes, fs, n_samples):
    """Evaluate Σ_n A_n·exp(-t/τ_n)·cos(2πf_n·t + φ_n) over n_samples."""
    t = np.arange(n_samples) / fs
    out = np.zeros(n_samples, dtype=np.float64)
    for mode in modes:
        out += (mode['A'] * np.exp(-t / mode['tau_s']) *
                np.cos(2.0 * np.pi * mode['f_hz'] * t + mode['phi']))
    return out


# NR-informed amplitude ratios (Giesler-Isi-Scheel-Teukolsky 2019,
# "Black hole ringdown: the importance of overtones"; Table I, values
# for binary BH mergers at a* ≈ 0.7, q ≈ 1).  A_1/A_0 and A_2/A_0 are
# relative magnitudes of the first two overtones at peak strain.
# These are NOT exact — real binary-BH ringdowns scatter ~20 % around
# these values as a function of mass ratio and spin alignment.
GIESLER_NR_AMPLITUDE_RATIOS = (1.0, 0.80, 0.40)


def fit_kerr_overtones_constrained(data, fs, start_idx, M_msun, a_spin,
                                     duration_ms=10.0, n_overtones=3,
                                     nr_ratios=None):
    """Sequential Kerr overtone fit with NR-fixed amplitude ratios.

    Phase I.2 unconstrained 6-param fit overfit noise catastrophically
    (A_n/A_0 came back 3–25× instead of NR-expected 0.4–0.9) because
    f_0 ≈ f_1 ≈ f_2 within 20 Hz at a* = 0.67 makes the system ill-
    conditioned on a 20 ms fit window.  This constrained variant fixes
    |A_n| = α_n · |A_0| using NR-informed ratios (Giesler-Isi-Scheel-
    Teukolsky 2019) and only fits the phase φ_n, reducing free params
    from 2·n_overtones to (n_overtones + 1).

    Method:
      1. Fit n=0 unconstrained via the standard 2-parameter linear LS
         (inherits fit_qnm's closed-form solve), yielding (A_0, φ_0).
      2. Subtract the n=0 model from the fit-window data.
      3. For n = 1, 2, …: compute the best phase φ_n that matches the
         residual at (f_n, τ_n), via φ_n = atan2(<r, h_n>, <r, g_n>).
         Set A_n = α_n · A_0 (fixed by NR prior).  Subtract, iterate.

    The per-overtone phase match is the standard matched-filter phase
    solution (maximising the inner product of residual with the
    phase-varying template at fixed frequency and envelope).

    Returns (modes, combined_model, start_idx, end_idx) matching the
    fit_kerr_overtones contract.  Each mode dict also carries
    `amplitude_source`: `'unconstrained_ls'` (n=0) or
    `'nr_ratio_x_A_0'` (n≥1).
    """
    if nr_ratios is None:
        nr_ratios = GIESLER_NR_AMPLITUDE_RATIOS
    if len(nr_ratios) < n_overtones:
        raise ValueError(
            f'need {n_overtones} NR ratios, got {len(nr_ratios)}')
    spec = _kerr_overtone_spectrum(M_msun, a_spin, n_overtones=n_overtones)
    n_samples = int(duration_ms * 1e-3 * fs)
    end_idx = start_idx + n_samples
    if end_idx > len(data):
        return None, None, start_idx, end_idx
    t = np.arange(n_samples) / fs
    y = np.asarray(data[start_idx:end_idx], dtype=np.float64).copy()

    modes = []
    # Step 1: unconstrained n=0 fit.
    f0, tau0 = spec[0]
    env0 = np.exp(-t / tau0)
    g0 = env0 * np.cos(2.0 * np.pi * f0 * t)
    h0 = -env0 * np.sin(2.0 * np.pi * f0 * t)
    M0 = np.column_stack([g0, h0])
    coef0, *_ = np.linalg.lstsq(M0, y, rcond=None)
    c0, s0 = float(coef0[0]), float(coef0[1])
    A0 = math.sqrt(c0 * c0 + s0 * s0)
    phi0 = math.atan2(s0, c0)
    modes.append({'n': 0, 'f_hz': float(f0), 'tau_s': float(tau0),
                  'A': A0, 'phi': phi0, 'c_coef': c0, 's_coef': s0,
                  'amplitude_source': 'unconstrained_ls'})
    # Subtract n=0.
    y = y - (A0 * env0 * np.cos(2.0 * np.pi * f0 * t + phi0))

    # Steps 2+: constrained fit for higher overtones.
    for i in range(1, n_overtones):
        fi, taui = spec[i]
        Ai = nr_ratios[i] * A0
        envi = np.exp(-t / taui)
        gi = envi * np.cos(2.0 * np.pi * fi * t)
        hi = -envi * np.sin(2.0 * np.pi * fi * t)
        # Best phase: atan2(<y, h_i>, <y, g_i>) — the phase that maximises
        # <y, env·cos(2πf·t + φ)> = <y, cos(φ)·g_i + sin(φ)·h_i>.  When
        # <g_i, g_i> ≈ <h_i, h_i> and <g_i, h_i> ≈ 0 (the oscillating-
        # envelope case), this is also the unconstrained-amplitude
        # least-squares phase; we use it as the constrained-amplitude
        # phase.
        inner_g = float(np.dot(y, gi))
        inner_h = float(np.dot(y, hi))
        phi_i = math.atan2(inner_h, inner_g)
        modes.append({'n': i, 'f_hz': float(fi), 'tau_s': float(taui),
                      'A': Ai, 'phi': phi_i,
                      'c_coef': Ai * math.cos(phi_i),
                      's_coef': -Ai * math.sin(phi_i),
                      'amplitude_source': f'nr_ratio_{nr_ratios[i]:.2f}_x_A_0'})
        y = y - (Ai * envi * np.cos(2.0 * np.pi * fi * t + phi_i))

    combined = evaluate_overtone_sum(modes, fs, n_samples)
    return modes, combined, start_idx, end_idx


def make_echo_template(fs, f_qnm, tau_qnm, length_seconds=0.020):
    """Template shape to matched-filter echoes against.

    A decaying sinusoid at the QNM frequency, normalised to unit L2 norm.
    """
    n = int(length_seconds * fs)
    t = np.arange(n) / fs
    tmpl = np.exp(-t / tau_qnm) * np.cos(2.0 * np.pi * f_qnm * t)
    tmpl = tmpl - np.mean(tmpl)
    nrm = np.sqrt(np.sum(tmpl ** 2))
    if nrm > 0:
        tmpl = tmpl / nrm
    return tmpl


def correlate_at_delay(whitened_residual, template, fs, t_ref_idx, delay_s):
    """Correlation of template against residual starting at (t_ref + delay).

    Returns the matched-filter inner product (SNR in whitened units).
    """
    start = t_ref_idx + int(round(delay_s * fs))
    n = len(template)
    end = start + n
    if start < 0 or end > len(whitened_residual):
        return float('nan')
    seg = whitened_residual[start:end]
    return float(np.dot(seg, template))


def background_distribution(whitened_residual, template, fs, t_ref_idx,
                            delta_t1_s, n_samples=2000, seed=1):
    """Sample matched-filter SNR at random delays that avoid predicted Δt_n.

    Delays are drawn from [10·Δt_1, 500 ms] excluding ±1 ms neighbourhoods
    around each multiple of Δt_1 up to the 50th.
    """
    rng = np.random.default_rng(seed)
    lo = 10.0 * delta_t1_s
    hi = 0.5  # 500 ms after ringdown start
    if hi <= lo:
        hi = lo + 0.1
    forbidden = [n * delta_t1_s for n in range(1, 51)]
    forbidden_halfwidth = 1.5e-3
    samples = []
    attempts = 0
    max_attempts = n_samples * 50
    while len(samples) < n_samples and attempts < max_attempts:
        d = rng.uniform(lo, hi)
        if any(abs(d - f) < forbidden_halfwidth for f in forbidden):
            attempts += 1
            continue
        snr = correlate_at_delay(whitened_residual, template, fs, t_ref_idx, d)
        if not math.isnan(snr):
            samples.append(snr)
        attempts += 1
    return np.array(samples)


def find_merger_sample(event_name, t_start_gps, fs):
    """Sample index of merger time within the 32-s segment."""
    t_event = EVENTS[event_name]['gps_event']
    return int(round((t_event - t_start_gps) * fs))


def search_event(event_name, detector='H1', fs_khz=4, n_echoes=5,
                 n_background=2000, ringdown_start_ms=3.0,
                 qnm_fit_duration_ms=20.0, template_length_ms=20.0,
                 subtraction_mode='n0',
                 use_kerr_overtones=False, n_overtones=3,
                 verbose=True):
    # subtraction_mode ∈ {'none', 'n0', 'kerr_constrained', 'kerr_unconstrained'}
    #   - 'none': skip ringdown subtraction, matched-filter raw whitened
    #     strain.  Phase I.3 found this is cleaner than 'n0' for these
    #     two events because the n=0 fit amplifies f_QNM-band noise
    #     onto the residual (A_fit 5-24× when the local SNR is low).
    #   - 'n0': single-mode damped-sinusoid fit + subtract (Phase I).
    #   - 'kerr_constrained': NR-fixed A_n/A_0, phase-only multi-overtone
    #     subtraction (Phase I.3a — still worse than 'n0' on real data).
    #   - 'kerr_unconstrained': 2·n_overtones-parameter linear LS
    #     (Phase I.2 — catastrophically overfits noise, DO NOT USE).
    # The back-compat `use_kerr_overtones=True` maps to 'kerr_unconstrained'.
    # NOTE on use_kerr_overtones: Phase I.2 experimented with multi-mode
    # (n=0..2) Kerr subtraction using the `qnm` library.  The unconstrained
    # 6-parameter linear LS overfits noise badly — on every (event, det,
    # fit-start, duration) combo tested, the 3-mode residual std was
    # 1.3–65× worse than n=0-only, with A_1/A_0 and A_2/A_0 ratios 3–25×
    # instead of the NR-expected 0.4–0.9.  The near-frequency degeneracy
    # (f_1 − f_0 ≈ 6 Hz at a*=0.67) makes the system ill-conditioned on
    # real GWOSC strain without an NR-constrained prior.  The code path is
    # preserved for future constrained-fit experiments but the default is
    # n=0-only, matching Phase I.
    # See misc/bh_phase_i_2_overtone_coherence_results.md for the full
    # write-up of this cherished failure.
    """Full pipeline for one event on one detector.

    Returns a dict with per-echo SNRs, background percentiles, p-values.
    """
    meta = EVENTS[event_name]
    if verbose:
        print(f'[{event_name} / {detector} / {fs_khz}kHz] fetching strain...')
    path = fetch_strain(event_name, detector, fs_khz=fs_khz)
    strain, t0_gps, fs = read_strain(path)
    if verbose:
        print(f'  loaded {len(strain)} samples at {fs} Hz, t0 = {t0_gps}')

    merger_idx = find_merger_sample(event_name, t0_gps, fs)
    ringdown_start_idx = merger_idx + int(ringdown_start_ms * 1e-3 * fs)

    # PSD from 8 seconds ending ~2 s before merger.
    psd_seg_len = int(8.0 * fs)
    psd_start = merger_idx - int(2.0 * fs) - psd_seg_len
    if psd_start < 0:
        psd_start = 0
    psd_freqs, psd_vals = estimate_psd(strain, fs, psd_start, psd_seg_len)

    # Bandpass before fitting and whitening — kills low-f drift.
    bp = bandpass(strain, fs, lo=35.0, hi=min(fs / 2 * 0.45, 1500.0))

    # Dispatch subtraction mode.  Default f_qnm/tau_qnm are the literature
    # Berti-Cardoso-Starinets values; 'n0' and Kerr variants overwrite them
    # with the fitted fundamental.  residual starts as bandpassed strain and
    # gets its ringdown window subtracted (or not, for 'none').
    kerr_modes = None
    popt = None
    qnm_start = ringdown_start_idx
    mode = subtraction_mode
    # Back-compat: legacy use_kerr_overtones=True routes to unconstrained.
    if use_kerr_overtones and mode == 'n0':
        mode = 'kerr_unconstrained'
    if mode not in ('none', 'n0', 'kerr_constrained', 'kerr_unconstrained'):
        raise ValueError(f'unknown subtraction_mode: {subtraction_mode!r}')
    f_qnm = meta['f_qnm_hz']
    tau_qnm = meta['tau_qnm_s']
    residual = bp.copy()

    if mode == 'none':
        if verbose:
            print(f'  subtraction_mode=none: raw whitened strain; '
                  f'template f={f_qnm:.1f} Hz τ={tau_qnm*1000:.2f} ms')

    elif mode == 'n0':
        popt, _qnm_model, qnm_start, _qnm_end = fit_qnm(
            bp, fs, ringdown_start_idx,
            duration_ms=qnm_fit_duration_ms,
            f_guess=meta['f_qnm_hz'],
            tau_guess=meta['tau_qnm_s'],
        )
        if popt is None:
            if verbose:
                print('  QNM fit FAILED — using literature values as template params.')
        else:
            A_fit, f_fit, tau_fit, phi_fit = popt
            if verbose:
                print(f'  QNM fit (n0): A={A_fit:.3e}, f={f_fit:.1f} Hz, '
                      f'tau={tau_fit*1000:.2f} ms, phi={phi_fit:.2f}')
            f_qnm = float(f_fit)
            tau_qnm = float(tau_fit)
            full_len_samples = int(min(10.0 * tau_qnm * fs,
                                       len(bp) - qnm_start))
            if full_len_samples > 0:
                t_full = np.arange(full_len_samples) / fs
                qnm_full = qnm_damped_sinusoid(t_full, A_fit, f_qnm,
                                                tau_qnm, phi_fit)
                residual[qnm_start:qnm_start + full_len_samples] -= qnm_full

    else:  # kerr_unconstrained | kerr_constrained
        fit_fn = (fit_kerr_overtones if mode == 'kerr_unconstrained'
                  else fit_kerr_overtones_constrained)
        try:
            kerr_modes, _fit_model, qnm_start, _qnm_end = fit_fn(
                bp, fs, ringdown_start_idx,
                M_msun=meta['M_rem_msun'], a_spin=meta['a_spin'],
                duration_ms=qnm_fit_duration_ms,
                n_overtones=n_overtones,
            )
        except ImportError as exc:
            if verbose:
                print(f'  qnm library unavailable ({exc}); falling back to '
                      f'no-subtraction + literature template params.')
            kerr_modes = None
        if kerr_modes is not None:
            if verbose:
                for m_i in kerr_modes:
                    print(f'  Kerr n={m_i["n"]}: f={m_i["f_hz"]:.1f} Hz, '
                          f'τ={m_i["tau_s"]*1000:.2f} ms, '
                          f'A={m_i["A"]:.3e}, φ={m_i["phi"]:+.2f}')
            f_qnm = float(kerr_modes[0]['f_hz'])
            tau_qnm = float(kerr_modes[0]['tau_s'])
            full_len_samples = int(min(10.0 * kerr_modes[0]['tau_s'] * fs,
                                        len(bp) - qnm_start))
            if full_len_samples > 0:
                overtone_full = evaluate_overtone_sum(kerr_modes, fs,
                                                       full_len_samples)
                residual[qnm_start:qnm_start + full_len_samples] -= overtone_full

    # Whiten the full residual.
    whitened = whiten(residual, fs, psd_freqs, psd_vals)
    # Diagnostic: std of whitened data in a clean pre-merger window.
    clean_start = max(0, merger_idx - int(2.0 * fs) - int(4.0 * fs))
    clean_end = max(clean_start + 1, merger_idx - int(2.0 * fs))
    whitened_std_clean = float(np.std(whitened[clean_start:clean_end]))
    if verbose:
        print(f'  whitened pre-merger std: {whitened_std_clean:.3f} '
              f'(ideal ~1.0)')

    # Build echo template matched to the fit QNM.
    template = make_echo_template(fs, f_qnm, tau_qnm,
                                  length_seconds=template_length_ms * 1e-3)

    # Compute SNR at each predicted delay.
    delta_t1 = echo_delay_n(meta['M_rem_msun'], 1)
    echo_snrs = []
    for n in range(1, n_echoes + 1):
        dt = echo_delay_n(meta['M_rem_msun'], n)
        snr = correlate_at_delay(whitened, template, fs,
                                 ringdown_start_idx, dt)
        echo_snrs.append({'n': n, 'delay_ms': dt * 1000.0, 'snr': snr})

    # Background.
    if verbose:
        print(f'  bootstrapping {n_background} off-source background SNRs...')
    bg = background_distribution(whitened, template, fs, ringdown_start_idx,
                                 delta_t1, n_samples=n_background)
    bg_std_raw = float(np.std(bg))

    # Calibrate to unit-variance Gaussian background.  P-values are invariant
    # under uniform scaling (echo SNRs and bg scale together), but this
    # restores the |SNR| ~ 2.3 intuition expected from a whitened detector.
    if bg_std_raw > 0:
        bg = bg / bg_std_raw
        for e in echo_snrs:
            e['snr'] = e['snr'] / bg_std_raw
    bg_mean = float(np.mean(bg))
    bg_std = float(np.std(bg))
    if verbose:
        for e in echo_snrs:
            print(f'  echo n={e["n"]}: Δt = {e["delay_ms"]:.3f} ms, '
                  f'SNR = {e["snr"]:+.3f} (calibrated)')
        print(f'  background: mean={bg_mean:+.3f}, std={bg_std:.3f}, '
              f'p99={float(np.percentile(np.abs(bg), 99)):.3f}')

    # Combined SNR across echoes (quadrature).
    snr_values = np.array([e['snr'] for e in echo_snrs])
    combined = float(np.sqrt(np.sum(snr_values ** 2)))
    # P-value: fraction of background-combined samples at least as extreme.
    # Approximate by sampling n_echoes random delays at a time.
    rng = np.random.default_rng(7)
    n_bg_trials = max(1000, n_background // n_echoes)
    bg_combined = np.zeros(n_bg_trials)
    for i in range(n_bg_trials):
        idx = rng.choice(len(bg), size=n_echoes, replace=False)
        bg_combined[i] = np.sqrt(np.sum(bg[idx] ** 2))
    p_value = float(np.mean(bg_combined >= combined))

    # Per-echo p-values: |SNR| tail against background |SNR| distribution.
    bg_abs = np.abs(bg)
    for e in echo_snrs:
        e['p_value'] = float(np.mean(bg_abs >= abs(e['snr'])))
        e['z_score'] = (abs(e['snr']) - bg_mean) / bg_std if bg_std > 0 else 0.0

    qnm_fit_params = {'mode': mode}
    if kerr_modes is not None:
        qnm_fit_params['n_overtones'] = len(kerr_modes)
        qnm_fit_params['modes'] = kerr_modes
    elif popt is not None:
        qnm_fit_params['A'] = float(popt[0])
        qnm_fit_params['f_hz'] = f_qnm
        qnm_fit_params['tau_s'] = tau_qnm
        qnm_fit_params['phi'] = float(popt[3])
    return {
        'event': event_name,
        'detector': detector,
        'fs_hz': fs,
        't0_gps': t0_gps,
        'merger_sample_idx': merger_idx,
        'ringdown_start_idx': ringdown_start_idx,
        'qnm_fit_params': qnm_fit_params,
        'predicted_delta_t1_ms': delta_t1 * 1000.0,
        'n_echoes': n_echoes,
        'echo_snrs': echo_snrs,
        'combined_snr': combined,
        'p_value_combined': p_value,
        'n_background': len(bg),
        'background_mean': bg_mean,
        'background_std': bg_std,
        'background_p99': float(np.percentile(np.abs(bg), 99)),
        'background_p999': float(np.percentile(np.abs(bg), 99.9)),
        'background_samples_calibrated': bg,
        'whitened_pre_merger_std': whitened_std_clean,
        'bg_std_raw_before_calibration': bg_std_raw,
    }


def cross_detector_coherence(snr_h1, snr_l1, bg_h1, bg_l1,
                              n_bg_trials=5000, seed=3):
    """Inter-detector coherence statistic at predicted Δt_n.

    The per-echo calibrated SNRs are paired across detectors:
      S = Σ_n  SNR_H1[n] · SNR_L1[n]
    A real echo signal is correlated at the same delay in both
    detectors (antenna-pattern sign flip allowed but rare), so the
    cross-product sum is biased positive.  Pure detector noise yields
    a zero-mean distribution; the null is obtained by pairing random
    off-source SNR samples from the two detectors.

    Returns dict with the coherence statistic, null mean/std,
    one-sided p-value, per-echo products, and sign-concordance
    fraction.
    """
    h = np.asarray(snr_h1, dtype=np.float64)
    l = np.asarray(snr_l1, dtype=np.float64)
    if len(h) != len(l) or len(h) == 0:
        return None
    n = len(h)
    products = h * l
    stat = float(np.sum(products))

    # Magnitude ratio |L1|/|H1| per echo — real astrophysical signal has
    # this O(1) (antenna-pattern factor), detector-specific noise can
    # have it arbitrary.  A >5× ratio flags probable contamination.
    magnitude_ratio = [float(abs(l[i]) / abs(h[i])) if abs(h[i]) > 0 else float('nan')
                       for i in range(n)]

    # Sign-only coherence: independent of magnitude-artefact contamination.
    sign_matches = int(np.sum(np.sign(h) * np.sign(l) > 0))
    sign_concordance = float(sign_matches / n)
    # Binomial p(≥k matches in n trials | p=0.5).
    from math import comb
    sign_p_value = float(sum(comb(n, k) for k in range(sign_matches, n + 1))
                         / (2 ** n))

    rng = np.random.default_rng(seed)
    bg_h = np.asarray(bg_h1, dtype=np.float64)
    bg_l = np.asarray(bg_l1, dtype=np.float64)
    null = np.zeros(n_bg_trials, dtype=np.float64)
    for i in range(n_bg_trials):
        hi_idx = rng.choice(len(bg_h), size=n, replace=False)
        li_idx = rng.choice(len(bg_l), size=n, replace=False)
        null[i] = float(np.sum(bg_h[hi_idx] * bg_l[li_idx]))
    p_value = float(np.mean(null >= stat))
    null_mean = float(np.mean(null))
    null_std = float(np.std(null))
    z_score = ((stat - null_mean) / null_std) if null_std > 0 else 0.0
    return {
        'coherence_statistic': stat,
        'per_echo_products': [float(p) for p in products],
        'magnitude_ratio_l_over_h': magnitude_ratio,
        'sign_matches': sign_matches,
        'sign_concordance': sign_concordance,
        'sign_p_value_binomial': sign_p_value,
        'null_mean': null_mean,
        'null_std': null_std,
        'z_score': float(z_score),
        'p_value': p_value,
        'n_bg_trials': n_bg_trials,
    }


def search_event_both_detectors(event_name, fs_khz=4, **kwargs):
    """Run search on H1 and L1, return per-detector dict + combined."""
    results = {}
    for det in ('H1', 'L1'):
        try:
            results[det] = search_event(event_name, detector=det,
                                        fs_khz=fs_khz, **kwargs)
        except Exception as exc:
            results[det] = {'error': str(exc)}
    snrs = [r.get('combined_snr', 0.0) for r in results.values()
            if isinstance(r, dict) and 'combined_snr' in r]
    p_values = [r.get('p_value_combined', 1.0) for r in results.values()
                if isinstance(r, dict) and 'p_value_combined' in r]
    coherent = {}
    if snrs:
        coherent.update({
            'combined_snr_quadrature': float(np.sqrt(np.sum(np.array(snrs) ** 2))),
            'p_value_min': float(min(p_values)),
            'p_value_product': float(np.prod(p_values)),
        })
    h1 = results.get('H1')
    l1 = results.get('L1')
    if (isinstance(h1, dict) and isinstance(l1, dict)
            and 'echo_snrs' in h1 and 'echo_snrs' in l1
            and 'background_samples_calibrated' in h1
            and 'background_samples_calibrated' in l1):
        snr_h1 = [e['snr'] for e in h1['echo_snrs']]
        snr_l1 = [e['snr'] for e in l1['echo_snrs']]
        coh = cross_detector_coherence(
            snr_h1, snr_l1,
            h1['background_samples_calibrated'],
            l1['background_samples_calibrated'],
        )
        if coh is not None:
            coherent['cross_detector'] = coh
    if coherent:
        results['_coherent'] = coherent
    return results


if __name__ == '__main__':
    # GW151226: 16 kHz because Δt_1 ≈ 0.76 ms → only 3 samples at 4 kHz.
    # GW150914: 4 kHz is sufficient (Δt_1 ≈ 2.26 ms = 9 samples).
    plan = [
        ('GW151226', 16),
        ('GW150914', 4),
    ]
    # Phase I.3: compare n0-subtract (the Phase I/I.2 default) vs no-subtract.
    # Phase I.2 overtone/coherence analysis showed that the n=0 fit catches
    # broadband f_QNM noise with A_fit 5–24× the physical amplitude, and
    # subtracting this injects noise into the residual where the matched
    # filter then picks it up as a spurious echo.  Running both modes in a
    # single pass makes the subtraction-induced pathology visible.
    modes = ('n0', 'none')
    all_results = {}
    for ev, fs_khz in plan:
        all_results[ev] = {}
        for sub_mode in modes:
            print(f'\n{"="*70}\n{ev}  ({fs_khz} kHz)  '
                  f'subtraction_mode={sub_mode}\n{"="*70}')
            res = search_event_both_detectors(ev, fs_khz=fs_khz,
                                              n_echoes=5, n_background=2000,
                                              subtraction_mode=sub_mode,
                                              use_kerr_overtones=False)
            all_results[ev][sub_mode] = res
            for det, det_res in res.items():
                if det.startswith('_'):
                    continue
                if 'error' in det_res:
                    print(f'  {det}: ERROR {det_res["error"]}')
                    continue
                print(f'\n  --- {det} ---')
                print(f'    Δt_1 predicted: {det_res["predicted_delta_t1_ms"]:.3f} ms')
                print(f'    whitened std:   {det_res["whitened_pre_merger_std"]:.3f} '
                      f'(ideal 1.0)')
                for e in det_res['echo_snrs']:
                    print(f'    n={e["n"]}: Δt={e["delay_ms"]:.3f} ms, '
                          f'SNR={e["snr"]:+.3f}σ, p={e["p_value"]:.4f}')
                print(f'    combined SNR:   {det_res["combined_snr"]:+.3f}')
                print(f'    p-value:        {det_res["p_value_combined"]:.4f}')
                print(f'    bg |SNR| p99:   {det_res["background_p99"]:.3f}')
            if '_coherent' in res:
                c = res['_coherent']
                print(f'\n  === H1+L1 combined ===')
                print(f'    quadrature SNR: {c["combined_snr_quadrature"]:.3f}')
                print(f'    min p-value:    {c["p_value_min"]:.4f}')
                if 'cross_detector' in c:
                    cc = c['cross_detector']
                    print(f'    --- cross-detector coherence ---')
                    print(f'    per-echo products: '
                          + ', '.join(f'{p:+.3f}' for p in cc['per_echo_products']))
                    print(f'    |L1|/|H1| per n:   '
                          + ', '.join(f'{r:.2f}' for r in cc['magnitude_ratio_l_over_h']))
                    print(f'    sign matches:      {cc["sign_matches"]}/5  '
                          f'(p={cc["sign_p_value_binomial"]:.3f} binomial)')
                    print(f'    Σ SNR_H1·SNR_L1:   {cc["coherence_statistic"]:+.3f}  '
                          f'(weighted by |SNR|; artefact-sensitive)')
                    print(f'    null μ, σ:         {cc["null_mean"]:+.3f}, '
                          f'{cc["null_std"]:.3f}')
                    print(f'    z-score:           {cc["z_score"]:+.3f}')
                    print(f'    p-value (1-sided): {cc["p_value"]:.4f}')

    # Side-by-side comparison: what the subtraction itself created.
    print(f'\n{"="*70}\nPhase I.3 summary — subtraction mode × event × detector'
          f'\n{"="*70}')
    print(f'{"event":<10} {"det":<4} {"n":>3}  '
          f'{"|SNR|_n0":>9} {"|SNR|_none":>10} {"Δ":>7}')
    for ev, fs_khz in plan:
        for det in ('H1', 'L1'):
            n0_res = all_results[ev]['n0'].get(det, {})
            none_res = all_results[ev]['none'].get(det, {})
            n0_echoes = n0_res.get('echo_snrs', [])
            none_echoes = none_res.get('echo_snrs', [])
            for e_n0, e_none in zip(n0_echoes, none_echoes):
                delta = abs(e_n0['snr']) - abs(e_none['snr'])
                print(f'{ev:<10} {det:<4} {e_n0["n"]:>3}  '
                      f'{abs(e_n0["snr"]):>9.3f} {abs(e_none["snr"]):>10.3f} '
                      f'{delta:>+7.3f}')
