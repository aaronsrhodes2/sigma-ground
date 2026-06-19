"""
Demo — direct physics → pixel, with thermal jitter (the holodeck render keystone).

One scene, rendered straight out of the cited physics library — no stored RGB in the
temperature-aware path:

  * a COPPER SPHERE whose colour is emergent: true copper (measured n+k) when cold,
    ramping dull-red → orange → white as we raise its temperature through the Draper
    point — the Planck glow, never a chosen colour;
  * a GLASS SLAB (a flattened ellipsoid) that TRANSMITS — per-channel Beer–Lambert
    through its volume — and carries a Fresnel surface glint; glass is a dielectric,
    so site_response returns no metal colour and the slab keeps its own tint
    (confirming the legacy transmission path still rules dielectrics);
  * a far WALL (the background) the ray reports when it hits nothing else.

We render at T ∈ {293, 1000, 1800, 3000} K, jitter OFF and ON, and print a
quantitative summary so the proof is in the terminal as well as the PPMs:
  - the per-temperature site_response ramp (cold_rgb stable, glow climbing);
  - the mean colour of the rendered sphere at each T (the ramp, in actual pixels);
  - an off-vs-on jitter delta (the lattice decorrelation, at the image level).

Run:  python -m sigma_ground.radiance.entangler.demo_direct_physics
Output PPMs land in  <this dir>/holodeck_demo_out/  (→ PNG if ImageMagick present).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from sigma_ground.radiance.entangler.vec import Vec3
from sigma_ground.radiance.entangler.shapes import (
    EntanglerSphere, EntanglerEllipsoid, rotation_matrix,
)
from sigma_ground.radiance.entangler.projection import PushCamera
from sigma_ground.radiance.entangler.illumination import PushLight
from sigma_ground.radiance.entangler.engine import entangle, _write_ppm
from sigma_ground.radiance.entangler.site_response import site_response
from sigma_ground.radiance.materials.material import Material

TEMPS = (293.0, 1000.0, 1800.0, 3000.0)
WIDTH, HEIGHT = 240, 160
DENSITY = 120
BG = Vec3(0.05, 0.05, 0.06)          # the far wall — dark so the glow ramp pops


def _copper(T):
    """A copper sphere that answers from physics (material_key set)."""
    return Material("copper", Vec3(0.95, 0.64, 0.54),
                    material_key="copper", temperature_K=T,
                    density_kg_m3=8960, mean_Z=29, mean_A=63.5,
                    alpha_r=3.0e6, alpha_g=3.0e6, alpha_b=3.0e6)   # opaque metal


def _glass():
    """A glass slab — dielectric: no metal_rgb, transmits via Beer–Lambert."""
    return Material("glass", Vec3(0.85, 0.92, 0.95),
                    opacity=0.06, ior=1.5,            # ~4–6% Fresnel surface
                    alpha_r=0.08, alpha_g=0.05, alpha_b=0.12,   # faint blue-green tint
                    density_kg_m3=2500, mean_Z=10, mean_A=20)


def _scene(T):
    sphere = EntanglerSphere(Vec3(-1.4, 0.0, 0.0), 1.1, _copper(T))
    slab = EntanglerEllipsoid(
        center=Vec3(1.5, 0.0, 0.0),
        radii=Vec3(1.1, 1.3, 0.18),                   # flat → a slab
        rotation=rotation_matrix(ry=0.5),
        material=_glass(), fill_volume=True,
    )
    return [sphere, slab]


def _camera():
    return PushCamera(Vec3(0.0, 0.6, 6.0), Vec3(0.0, 0.0, 0.0),
                      WIDTH, HEIGHT, fov=50)


def _light():
    return PushLight(Vec3(4.0, 4.0, 6.0), intensity=1.1)


def _matter_stats(pixels, bg, thresh=0.04):
    """Mean colour + coverage of pixels that differ from the background.

    Returns (mean_rgb, coverage_fraction). The mean over 'matter' pixels shows the
    glow ramp in the actual render; coverage hints at lattice fill (jitter).
    """
    n_total = len(pixels) * len(pixels[0])
    sr = sg = sb = 0.0
    n_hit = 0
    for row in pixels:
        for p in row:
            if (abs(p.x - bg.x) + abs(p.y - bg.y) + abs(p.z - bg.z)) > thresh:
                sr += p.x; sg += p.y; sb += p.z
                n_hit += 1
    if n_hit == 0:
        return (0.0, 0.0, 0.0), 0.0
    return (sr / n_hit, sg / n_hit, sb / n_hit), n_hit / n_total


def _mean_abs_pixel_delta(a, b):
    """Mean |ΔRGB| between two equally-sized pixel grids (image-level decorrelation)."""
    h, w = len(a), len(a[0])
    acc = 0.0
    for y in range(h):
        for x in range(w):
            pa, pb = a[y][x], b[y][x]
            acc += abs(pa.x - pb.x) + abs(pa.y - pb.y) + abs(pa.z - pb.z)
    return acc / (h * w)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "holodeck_demo_out")
    os.makedirs(out_dir, exist_ok=True)
    cam, light = _camera(), _light()

    print("=" * 70)
    print("DIRECT PHYSICS -> PIXEL   (copper sphere + glass slab, far wall)")
    print("=" * 70)

    # 1) The physics the shade reads, per temperature — no stored RGB.
    print("\nsite_response('copper', T) — emergent appearance:")
    print(f"  {'T(K)':>6}  {'cold_rgb (n+k)':>22}  {'glowing':>7}  "
          f"{'glow_rgb':>20}  {'level':>5}")
    for T in TEMPS:
        r = site_response("copper", T)
        c = r["cold_rgb"]
        g = r["glow_rgb"]
        print(f"  {int(T):>6}  ({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})".ljust(34)
              + f"  {str(r['glowing']):>7}"
              + f"  ({g[0]:.2f},{g[1]:.2f},{g[2]:.2f})".ljust(22)
              + f"  {r['glow_level']:.2f}")

    # 2) Render the scene at each temperature, jitter off then on.
    print("\nRendered scene — mean colour of the copper sphere (the ramp in pixels):")
    print(f"  {'T(K)':>6}  {'jitter':>6}  {'mean sphere rgb':>22}  "
          f"{'coverage':>8}  file")
    pixel_cache = {}
    for T in TEMPS:
        objs = _scene(T)
        for tag, jitter in (("off", None), ("on", {"frame": 0})):
            pixels = entangle(objs, cam, light, density=DENSITY,
                              bg_color=BG, jitter=jitter)
            pixel_cache[(T, tag)] = pixels
            mean, cov = _matter_stats(pixels, BG)
            fname = f"copper_T{int(T)}_jitter_{tag}.ppm"
            _write_ppm(pixels, os.path.join(out_dir, fname))
            print(f"  {int(T):>6}  {tag:>6}  "
                  f"({mean[0]:.2f},{mean[1]:.2f},{mean[2]:.2f})".ljust(34)
                  + f"  {cov*100:6.1f}%  {fname}")

    # 3) Jitter decorrelation at the image level (off vs on, one temperature).
    d = _mean_abs_pixel_delta(pixel_cache[(1800.0, "off")],
                              pixel_cache[(1800.0, "on")])
    print(f"\nJitter image-level decorrelation @1800K (mean |dRGB| off vs on): {d:.4f}")
    print("  > 0 means the lattice was scattered — the hex/moiré stencil is broken.")
    print(f"\nWrote {len(pixel_cache)} PPMs to {out_dir}")
    print("Every rendered channel above traces to a field-library function:")
    print("  cold = optics.metal_rgb | glow = thermal_emission + thermal (eps*sigma*T^4)")
    print("  transmission = per-channel Beer-Lambert | jitter sigma = texture.thermal_roughness")


if __name__ == "__main__":
    main()
