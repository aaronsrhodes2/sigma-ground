"""Thermal jitter — the one deliberate effect on top of direct-physics-to-pixel.

The entangler samples matter on a regular Fibonacci lattice. A *regular* sampling
of a smooth surface beats against the pixel grid and against itself, leaving a
visible hex / moiré texture — the "same atoms always answer the ray" artefact the
Captain called out. Real matter does not sit still: every atom vibrates about its
lattice site (Debye–Waller), so which atom answers a given ray is never the same
twice. This module reintroduces that — a small, temperature-scaled, per-frame
random displacement that decorrelates the lattice so the render stops looking like
a stencil and starts looking like a living surface.

HONEST SCALE NOTE (read this before believing the picture):
  The literal atomic Debye–Waller amplitude is ~0.1 Å — utterly sub-pixel. If we
  displaced nodes by the true ⟨u²⟩^½ nothing visible would change. The entangler's
  nodes are also a *sampled subset*, not atoms. So the jitter here is scaled to a
  fraction of the inter-node spacing, MODULATED by the cited thermal-roughness ratio
  (√T, from ``texture.thermal_roughness``). Its visible job is therefore:
    (a) decorrelating the node sampling so it does not moiré, and
    (b) carrying the temperature dependence (hotter → rougher → more scatter),
  NOT a literal view of atomic vibration. ``_JITTER_BASE_FRAC`` is flagged as a
  display/anti-alias parameter, exactly like ``site_response._GLOW_P0`` — the
  *temperature scaling* is physics; the absolute amplitude is a render choice.

A strict ⟨u²⟩(T) helper from the Debye temperature is an easy follow-on; this slice
deliberately stays in "decorrelate the sampling" territory.
"""
from __future__ import annotations

import copy
import math

from .vec import Vec3
from ...field.interface import texture

# Fraction of one inter-node gap a node is displaced at the reference temperature.
# DISPLAY/ANTI-ALIAS parameter (not a physics claim): it sets how hard we scatter
# the lattice to kill moiré. The √T temperature *scaling* on top of it IS physics.
_JITTER_BASE_FRAC = 0.5

# Reference (ambient) temperature the roughness ratio is taken against.
_REFERENCE_T = 293.15


def thermal_jitter_sigma(material_key, temperature_K, spacing,
                         reference_T=_REFERENCE_T):
    """Gaussian jitter amplitude (σ, scene units) for a node.

    ``σ = _JITTER_BASE_FRAC × spacing × roughness_ratio(T)``, capped at ``spacing``
    so a node never wanders past its nearest neighbour.

    ``roughness_ratio`` is ``thermal_roughness(key, T) / thermal_roughness(key, T_ref)``
    — the *cited* √T Boltzmann step-excitation roughness (``field.interface.texture``)
    — so the amplitude grows with temperature exactly as the surface does. When the
    material is not in the texture database (or no key is given — the legacy path),
    we fall back to the Debye–Waller high-T limit ⟨u²⟩ ∝ T ⇒ amplitude ∝ √T, which is
    the same physics without the per-material step energy. Either way the ratio is 1.0
    at the reference temperature, so cold matter still gets the anti-moiré base jitter.

    Returns 0.0 for non-positive spacing.
    """
    if spacing <= 0.0:
        return 0.0

    ratio = None
    if material_key:
        try:
            r_T = texture.thermal_roughness(material_key, float(temperature_K))
            r_ref = texture.thermal_roughness(material_key, float(reference_T))
            if r_ref > 0.0:
                ratio = r_T / r_ref
        except Exception:
            ratio = None
    if ratio is None:
        # Debye–Waller high-T limit: ⟨u²⟩ ∝ T  →  amplitude ∝ √(T / T_ref).
        ratio = math.sqrt(max(0.0, float(temperature_K)) / float(reference_T))

    sigma = _JITTER_BASE_FRAC * spacing * ratio
    return min(sigma, spacing)


def _frame_seed(scene_seed, frame):
    """Deterministic 32-bit seed from (scene, frame).

    Pure integer arithmetic (Knuth/Fibonacci multipliers) — no ``hash()``, so it is
    stable across processes and independent of ``PYTHONHASHSEED``. Same (scene, frame)
    reproduces a frame exactly; incrementing ``frame`` gives a fully different scatter
    (a living surface).
    """
    s = (int(scene_seed) * 2654435761 + int(frame) * 40503 + 0x9E3779B1)
    return s & 0xFFFFFFFF


def _tangent_basis(normal):
    """Two orthonormal vectors spanning the tangent plane of ``normal``."""
    helper = Vec3(1.0, 0.0, 0.0) if abs(normal.x) < 0.9 else Vec3(0.0, 1.0, 0.0)
    t1 = normal.cross(helper).normalized()
    t2 = normal.cross(t1)
    return t1, t2


def apply_thermal_jitter(nodes, material_key, temperature_K, *,
                         spacing, frame=0, scene_seed=0,
                         reference_T=_REFERENCE_T):
    """Return a decorrelated copy of ``nodes`` with thermal jitter applied.

    Surface nodes (those carrying a ``normal``) are displaced within their tangent
    plane — they stay on the surface, only their *sampling position* scatters, which
    is what breaks the moiré. Volume nodes (no normal — interior matter has no
    preferred direction, per ``volume_nodes``) are displaced isotropically in 3D.

    The input list is left untouched (each node is shallow-copied, preserving its
    type and every other slot); successive frames therefore differ while a given
    ``(scene_seed, frame)`` reproduces exactly. Displacement magnitude is clamped to
    ``spacing``.

    Args:
        nodes:         list of SurfaceNode and/or VolumeNode.
        material_key:  field key for the cited roughness ratio (None → √T fallback).
        temperature_K: matter temperature driving the amplitude.
        spacing:       characteristic inter-node distance (scene units).
        frame:         frame index — changes the scatter, reproducibly.
        scene_seed:    per-scene salt so two scenes don't share a jitter pattern.

    Returns:
        A new list of jittered node copies.
    """
    sigma = thermal_jitter_sigma(material_key, temperature_K, spacing, reference_T)
    if sigma <= 0.0 or not nodes:
        # Nothing to do — still return copies so callers can treat the result uniformly.
        return [copy.copy(n) for n in nodes]

    import random
    rng = random.Random(_frame_seed(scene_seed, frame))
    cap = spacing
    out = []
    for node in nodes:
        normal = getattr(node, "normal", None)
        if normal is not None:
            # Tangent-plane scatter (stays on the surface).
            a = rng.gauss(0.0, sigma)
            b = rng.gauss(0.0, sigma)
            t1, t2 = _tangent_basis(normal)
            disp = t1 * a + t2 * b
        else:
            # Isotropic 3D scatter (interior matter — no preferred direction).
            dx = rng.gauss(0.0, sigma)
            dy = rng.gauss(0.0, sigma)
            dz = rng.gauss(0.0, sigma)
            disp = Vec3(dx, dy, dz)

        # Clamp the displacement so a node never crosses into its neighbour's cell.
        mag = disp.length()
        if mag > cap and mag > 0.0:
            disp = disp * (cap / mag)

        clone = copy.copy(node)
        clone.position = node.position + disp
        out.append(clone)

    return out


__all__ = ["apply_thermal_jitter", "thermal_jitter_sigma"]
