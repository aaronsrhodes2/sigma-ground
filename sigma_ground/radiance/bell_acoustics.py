"""Bell-strike acoustics -- tone synthesis + earth-atmosphere propagation.

Pure stdlib (the ``wave`` module), no numpy: a struck bell rings down as an
exponentially-decaying sinusoid at its ring frequency
(``field.interface.acoustics.ring_frequency`` -- the SAME first-principles
formula the ``acoustics()`` Materia verb already cites for bells:
f_ring = v_L / (pi*d), the circumferential-wave condition for a cylindrical
shell). Everything past that one real, cited number is a SIMPLIFIED_MODEL,
stated plainly rather than hidden: real bells ring with multiple inharmonic
partials (hum/prime/tierce/quint/nominal, each its own decay rate); this
module synthesizes exactly ONE mode with a caller-supplied (Choice-cited,
not invented here) decay time constant.

Propagation through earth's atmosphere reuses
``field.interface.atmosphere.speed_of_sound`` (Newton-Laplace,
v = sqrt(gamma*R*T/M)) for arrival time, and models only inverse-square
(spherical-spreading) amplitude loss -- frequency-dependent atmospheric
absorption is NOT modeled, flagged rather than silently dropped.
"""
from __future__ import annotations

import math
import struct
import wave

_SAMPLE_RATE_HZ = 22050
_FULL_SCALE_16BIT = 32000.0        # a hair under int16 max, headroom for rounding


def synthesize_bell_tone(path: str, frequency_hz: float, amplitude: float,
                         decay_tau_s: float, duration_s: float | None = None,
                         sample_rate: int = _SAMPLE_RATE_HZ) -> tuple[str, float]:
    """Write a mono 16-bit PCM WAV of a single-mode decaying-sinusoid bell
    strike: x(t) = amplitude * exp(-t/tau) * sin(2*pi*f*t).

    ``amplitude`` in [0, 1] (fraction of full 16-bit scale). Default
    ``duration_s`` = 8*tau (e^-8 ≈ 0.03% of initial amplitude -- effectively
    silent, a defensible "ring has died out" cutoff). Returns
    (path, duration_s_actually_written).
    """
    if frequency_hz <= 0:
        raise ValueError(f"frequency_hz must be > 0, got {frequency_hz}")
    if decay_tau_s <= 0:
        raise ValueError(f"decay_tau_s must be > 0, got {decay_tau_s}")
    dur = duration_s if duration_s is not None else 8.0 * decay_tau_s
    n = max(1, int(round(dur * sample_rate)))
    two_pi_f = 2.0 * math.pi * frequency_hz
    amp = max(0.0, min(1.0, amplitude)) * _FULL_SCALE_16BIT
    samples = bytearray()
    for i in range(n):
        t = i / sample_rate
        v = amp * math.exp(-t / decay_tau_s) * math.sin(two_pi_f * t)
        samples += struct.pack("<h", int(round(v)))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(samples))
    return path, n / sample_rate


def propagate_in_air(frequency_hz: float, distance_m: float,
                     source_level: float = 1.0, T: float = 288.15) -> dict:
    """Real, cited physics for the first-order questions "when does it
    arrive" and "how loud": arrival_time_s = distance / speed_of_sound(T)
    (Newton-Laplace). Amplitude falls off by inverse-square (spherical
    spreading) from a 1 m reference distance -- frequency-dependent
    atmospheric absorption is explicitly NOT modeled (SIMPLIFIED_MODEL,
    stated in the returned dict, not silently dropped)."""
    from ..field.interface.atmosphere import speed_of_sound
    if distance_m <= 0:
        raise ValueError(f"distance_m must be > 0, got {distance_m}")
    v = speed_of_sound(T)
    t_arrival = distance_m / v
    ref = 1.0
    amp = source_level * (ref / max(distance_m, ref)) ** 2
    return {
        "speed_of_sound_m_s": v,
        "arrival_time_s": t_arrival,
        "attenuated_amplitude": amp,
        "model": "inverse-square (spherical spreading) from a 1 m "
                "reference only; frequency-dependent atmospheric "
                "absorption NOT modeled (SIMPLIFIED_MODEL)",
    }


__all__ = ["synthesize_bell_tone", "propagate_in_air"]
