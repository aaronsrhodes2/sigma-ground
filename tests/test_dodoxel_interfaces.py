"""12-neighbor part-interface scan gates -- hinge arc Phase B. The dodoxel
scan's headline property over the cubic version: contact normals are EXACT
discrete FCC directions (each rhombic face is perpendicular to its
neighbor direction, a Voronoi-cell fact gated in Phase 0), not PCA
estimates.
"""
import math

import pytest

from sigma_ground.deckard.dodoxelize import dodoxelize_parts

_SQRT2 = math.sqrt(2.0)
PITCH = 0.008
G = PITCH / _SQRT2
FACE_AREA = 3.0 * _SQRT2 * PITCH * PITCH / 12.0


def _density_of(name):
    return 5000.0


def test_single_contact_reports_exact_normal_area_and_centroid():
    """Two single-site parts touching along exactly one FCC offset: the
    scan must report that offset's unit direction as the (single) contact
    normal, the exact closed-form rhombic face area, and the midpoint of
    the two site centers as the centroid."""
    # site A at even-parity index (2,2,2) -> world (2g,2g,2g); site B at
    # (3,3,2) -> (3g,3g,2g): offset (1,1,0), an FCC neighbor
    pa = (2 * G, 2 * G, 2 * G)
    pb = (3 * G, 3 * G, 2 * G)

    def near(c):
        return lambda x, y, z: ((x - c[0]) ** 2 + (y - c[1]) ** 2 +
                                (z - c[2]) ** 2) <= (0.35 * G) ** 2

    field = dodoxelize_parts(
        [("a", "iron", near(pa)), ("b", "iron", near(pb))],
        PITCH, (0.0, 0.0, 0.0), (5 * G, 5 * G, 5 * G), _density_of)

    assert field.parts[0]["site_count"] == 1
    assert field.parts[1]["site_count"] == 1
    assert (0, 1) in field.part_interfaces
    iface = field.part_interfaces[(0, 1)]
    assert iface["n_contacts"] == 1
    assert iface["area_m2"] == pytest.approx(FACE_AREA, rel=1e-12)
    normals = list(iface["contact_normals"].items())
    assert len(normals) == 1
    (ndir, count), = normals
    assert count == 1
    assert ndir[0] == pytest.approx(1.0 / _SQRT2)     # from a toward b
    assert ndir[1] == pytest.approx(1.0 / _SQRT2)
    assert ndir[2] == pytest.approx(0.0)
    expected_centroid = tuple((a + b) / 2.0 for a, b in zip(pa, pb))
    for got, want in zip(iface["centroid_m"], expected_centroid):
        assert got == pytest.approx(want, abs=1e-12)


def test_planar_slab_interface_normals_average_to_the_slab_normal():
    """Two slabs split by a plane x = const: individual contacts run along
    the four diagonal FCC offsets with a +x component, but their
    area-weighted MEAN must recover the true macroscopic +x interface
    normal -- the discrete normals carry real geometric information."""
    split = 3.5 * G

    def left(x, y, z):
        return x < split

    def right(x, y, z):
        return x >= split

    span = 8 * G
    field = dodoxelize_parts(
        [("left", "iron", left), ("right", "iron", right)],
        PITCH, (0.0, 0.0, 0.0), (span, span, span), _density_of)

    iface = field.part_interfaces[(0, 1)]
    assert iface["n_contacts"] > 10
    # every individual normal is one of the four +x-component FCC dirs
    expected = {(round(1 / _SQRT2, 9), round(s / _SQRT2, 9), 0.0)
                for s in (1.0, -1.0)} | \
               {(round(1 / _SQRT2, 9), 0.0, round(s / _SQRT2, 9))
                for s in (1.0, -1.0)}
    for ndir in iface["contact_normals"]:
        assert ndir in expected, f"unexpected contact normal {ndir}"
    # area-weighted mean normal ~ +x
    mx = sum(n[0] * c for n, c in iface["contact_normals"].items())
    my = sum(n[1] * c for n, c in iface["contact_normals"].items())
    mz = sum(n[2] * c for n, c in iface["contact_normals"].items())
    mag = math.sqrt(mx * mx + my * my + mz * mz)
    assert mx / mag == pytest.approx(1.0, abs=1e-6)
    assert abs(my / mag) < 0.15 and abs(mz / mag) < 0.15
    # a broad planar patch is NOT elongated like a hinge line
    assert iface["elongation"] < 4.0


def test_line_interface_is_strongly_elongated_along_its_axis():
    """A single-file line of contacts (the hinge-line case Phase C keys
    on): elongation must be large and the principal axis must align with
    the line direction."""
    # part A: a line of sites along z at (i,j)=(2,2); part B: the parallel
    # line at (3,3) -- every A site touches its B neighbor via (1,1,0)
    def line_a(x, y, z):
        return (abs(x - 2 * G) < 0.35 * G and abs(y - 2 * G) < 0.35 * G)

    def line_b(x, y, z):
        return (abs(x - 3 * G) < 0.35 * G and abs(y - 3 * G) < 0.35 * G)

    span = 14 * G
    field = dodoxelize_parts(
        [("a", "iron", line_a), ("b", "iron", line_b)],
        PITCH, (0.0, 0.0, 0.0), (5 * G, 5 * G, span), _density_of)

    iface = field.part_interfaces[(0, 1)]
    assert iface["n_contacts"] >= 4
    assert iface["elongation"] > 10.0                  # a genuine line
    pz = abs(iface["principal_dir"][2])
    assert pz == pytest.approx(1.0, abs=1e-6)          # hinge line along z
