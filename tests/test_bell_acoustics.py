"""Bell acoustics gates: WAV synthesis correctness (closed-form sample
count/envelope) and earth-atmosphere propagation (closed-form arrival time
+ inverse-square amplitude)."""
import math
import os
import struct
import wave

import pytest

from sigma_ground.radiance.bell_acoustics import (synthesize_bell_tone,
                                                   propagate_in_air)


def test_wav_has_the_expected_sample_count_and_duration(tmp_path):
    path = str(tmp_path / "bell.wav")
    freq, tau, sr = 440.0, 0.5, 22050
    out_path, dur = synthesize_bell_tone(path, freq, 0.8, tau, sample_rate=sr)
    assert out_path == path
    expected_n = round(8.0 * tau * sr)
    assert dur == pytest.approx(expected_n / sr, rel=1e-9)
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == sr
        assert w.getnframes() == expected_n


def test_envelope_decays_exponentially_at_the_stated_tau(tmp_path):
    path = str(tmp_path / "bell.wav")
    freq, tau, sr = 300.0, 0.3, 22050
    synthesize_bell_tone(path, freq, 1.0, tau, duration_s=2.0, sample_rate=sr)
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
    samples = struct.unpack(f"<{n}h", raw)
    # peak amplitude within the first tau vs. within the last window before
    # 2*tau..3*tau: ratio should track exp(-Δt/tau)
    def peak_over(t0, t1):
        i0, i1 = int(t0 * sr), int(t1 * sr)
        return max(abs(s) for s in samples[i0:i1]) or 1e-9

    early = peak_over(0.0, tau * 0.5)
    later = peak_over(2.0 * tau, 2.5 * tau)
    ratio = later / early
    expected = math.exp(-2.0 * tau / tau)          # centers ~2*tau apart
    assert ratio == pytest.approx(expected, rel=0.35)   # loose: peak-sampling noise


def test_rejects_nonpositive_frequency_or_tau(tmp_path):
    path = str(tmp_path / "bell.wav")
    with pytest.raises(ValueError):
        synthesize_bell_tone(path, 0.0, 1.0, 0.5)
    with pytest.raises(ValueError):
        synthesize_bell_tone(path, 400.0, 1.0, 0.0)


def test_propagation_arrival_time_matches_closed_form():
    r = propagate_in_air(500.0, distance_m=343.0, T=288.15)
    from sigma_ground.field.interface.atmosphere import speed_of_sound
    v = speed_of_sound(288.15)
    assert r["speed_of_sound_m_s"] == pytest.approx(v)
    assert r["arrival_time_s"] == pytest.approx(343.0 / v, rel=1e-9)


def test_propagation_amplitude_follows_inverse_square():
    r1 = propagate_in_air(500.0, distance_m=10.0)
    r2 = propagate_in_air(500.0, distance_m=20.0)
    assert r1["attenuated_amplitude"] == pytest.approx(
        4.0 * r2["attenuated_amplitude"], rel=1e-9)


def test_propagation_rejects_nonpositive_distance():
    with pytest.raises(ValueError):
        propagate_in_air(500.0, distance_m=0.0)
