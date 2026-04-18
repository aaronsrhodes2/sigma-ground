"""
Procedural sound synthesis — no audio files required.
Uses pygame.sndarray to generate sounds from numpy arrays.

Weapon fire sounds (atmosphere only — vacuum is silent):
  kinetic:     crack/boom (white noise burst + low thump)
  laser:       high-pitched whine + sustained hum
  plasma:      low roar/hiss (noise filtered to low freq)
  explosive:   deep boom + rumble decay
  particle:    sci-fi zap (frequency sweep)
  gravitational: deep subsonic throb (edge of hearing)

Impact sounds:
  metal hit:   sharp clang
  ceramic:     sharp crack/shatter
  vaporized:   sizzle + pop
  breach:      heavy boom
"""

import math
import array
import pygame

_SAMPLE_RATE = 44100
_initialized = False


def _init():
    global _initialized
    if not _initialized:
        pygame.mixer.init(frequency=_SAMPLE_RATE, size=-16, channels=1, buffer=512)
        _initialized = True


def _make_sound(samples) -> pygame.mixer.Sound:
    """Convert a list of floats [-1,1] to a pygame Sound."""
    n = len(samples)
    raw = array.array('h', [int(max(-32767, min(32767, s * 32767))) for s in samples])
    sound = pygame.mixer.Sound(buffer=raw)
    return sound


def _noise(n, amplitude=1.0):
    import random
    return [random.uniform(-amplitude, amplitude) for _ in range(n)]


def _sine(freq, n, amplitude=1.0, phase=0.0):
    return [amplitude * math.sin(2 * math.pi * freq * i / _SAMPLE_RATE + phase) for i in range(n)]


def _envelope(samples, attack=0.01, decay=0.1, sustain=0.7, release=0.2):
    n = len(samples)
    a = int(attack * n)
    d = int(decay * n)
    s = int(sustain * n)
    r = n - a - d - s
    out = []
    for i, v in enumerate(samples):
        if i < a:
            gain = i / max(a, 1)
        elif i < a + d:
            gain = 1.0 - 0.3 * (i - a) / max(d, 1)
        elif i < a + d + s:
            gain = 0.7
        else:
            gain = 0.7 * (1.0 - (i - a - d - s) / max(r, 1))
        out.append(v * gain)
    return out


def _lowpass(samples, cutoff_hz=400):
    rc = 1.0 / (2 * math.pi * cutoff_hz)
    dt = 1.0 / _SAMPLE_RATE
    alpha = dt / (rc + dt)
    out = [samples[0]]
    for s in samples[1:]:
        out.append(out[-1] + alpha * (s - out[-1]))
    return out


def _mix(a, b, ratio=0.5):
    n = min(len(a), len(b))
    return [a[i] * ratio + b[i] * (1 - ratio) for i in range(n)]


# ── Pre-built sounds ────────────────────────────────────────────

_cache = {}


def _build_kinetic_fire():
    dur = int(0.25 * _SAMPLE_RATE)
    crack = _noise(dur, 0.8)
    boom = _sine(80, dur, 0.6)
    boom2 = _sine(40, dur, 0.4)
    mixed = _mix(crack, [boom[i] + boom2[i] for i in range(dur)], 0.4)
    return _make_sound(_envelope(mixed, 0.001, 0.05, 0.1, 0.84))


def _build_laser_fire():
    dur = int(0.4 * _SAMPLE_RATE)
    whine = _sine(2400, dur, 0.5)
    sweep = [0.3 * math.sin(2 * math.pi * (1200 + 1200 * i / dur) * i / _SAMPLE_RATE)
             for i in range(dur)]
    mixed = [whine[i] + sweep[i] for i in range(dur)]
    return _make_sound(_envelope(mixed, 0.02, 0.1, 0.6, 0.28))


def _build_plasma_fire():
    dur = int(0.5 * _SAMPLE_RATE)
    roar = _noise(dur, 0.7)
    filtered = _lowpass(roar, 300)
    bass = _sine(60, dur, 0.4)
    mixed = [filtered[i] + bass[i] for i in range(dur)]
    return _make_sound(_envelope(mixed, 0.05, 0.2, 0.5, 0.25))


def _build_explosive_fire():
    dur = int(0.8 * _SAMPLE_RATE)
    bang = _noise(dur, 0.9)
    rumble = _sine(35, dur, 0.7)
    rumble2 = _sine(60, dur, 0.4)
    mixed = _mix(bang, [rumble[i] + rumble2[i] for i in range(dur)], 0.3)
    return _make_sound(_envelope(mixed, 0.001, 0.05, 0.2, 0.749))


def _build_particle_fire():
    dur = int(0.3 * _SAMPLE_RATE)
    zap = [0.7 * math.sin(2 * math.pi * (800 + 3000 * (1 - i / dur)) * i / _SAMPLE_RATE)
           for i in range(dur)]
    hiss = _noise(dur, 0.2)
    mixed = [zap[i] + hiss[i] for i in range(dur)]
    return _make_sound(_envelope(mixed, 0.001, 0.05, 0.15, 0.8))


def _build_gravitational_fire():
    dur = int(0.6 * _SAMPLE_RATE)
    throb = _sine(28, dur, 0.6)
    throb2 = _sine(55, dur, 0.3)
    mixed = [throb[i] + throb2[i] for i in range(dur)]
    return _make_sound(_envelope(mixed, 0.1, 0.3, 0.4, 0.2))


def _build_impact_metal():
    dur = int(0.15 * _SAMPLE_RATE)
    clang = _sine(900, dur, 0.6)
    clang2 = _sine(1800, dur, 0.3)
    noise = _noise(dur, 0.2)
    mixed = [clang[i] + clang2[i] + noise[i] for i in range(dur)]
    return _make_sound(_envelope(mixed, 0.001, 0.02, 0.05, 0.929))


def _build_impact_breach():
    dur = int(0.5 * _SAMPLE_RATE)
    boom = _sine(50, dur, 0.8)
    crack = _noise(dur, 0.5)
    mixed = _mix(boom, crack, 0.6)
    return _make_sound(_envelope(mixed, 0.001, 0.1, 0.2, 0.699))


def _build_impact_sizzle():
    dur = int(0.3 * _SAMPLE_RATE)
    hiss = _noise(dur, 0.6)
    filtered = _lowpass(hiss, 2000)
    pop = _sine(400, dur, 0.3)
    mixed = [filtered[i] + pop[i] * (1.0 - i / dur) for i in range(dur)]
    return _make_sound(_envelope(mixed, 0.001, 0.05, 0.3, 0.649))


_BUILDERS = {
    'fire_kinetic':     _build_kinetic_fire,
    'fire_laser':       _build_laser_fire,
    'fire_plasma':      _build_plasma_fire,
    'fire_explosive':   _build_explosive_fire,
    'fire_particle':    _build_particle_fire,
    'fire_gravitational': _build_gravitational_fire,
    'impact_metal':     _build_impact_metal,
    'impact_breach':    _build_impact_breach,
    'impact_sizzle':    _build_impact_sizzle,
}


def get(name: str) -> pygame.mixer.Sound:
    _init()
    if name not in _cache:
        builder = _BUILDERS.get(name)
        if builder:
            _cache[name] = builder()
    return _cache.get(name)


def play_fire(weapon_type: str, environment: str):
    if environment == 'vacuum':
        return  # no sound in space
    s = get(f'fire_{weapon_type}')
    if s:
        s.play()


def play_impact(result, environment: str):
    if environment == 'vacuum':
        return
    if result.breached:
        s = get('impact_breach')
    elif result.layers[result.layers_penetrated].temperature_k > 2000:
        s = get('impact_sizzle')
    else:
        s = get('impact_metal')
    if s:
        s.play()
