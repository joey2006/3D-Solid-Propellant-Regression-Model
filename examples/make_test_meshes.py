"""Generate test meshes for the geometry-import pipeline (issue #172).

Writes a few STL files into ``examples/output/meshes/`` for exercising the
import path and the viewer app:

* ``bates.stl``      -- a BATES grain (cylinder with a central bore). The one
  geometry with a closed-form burnback answer, so importing it and comparing
  against ``BATESGrain`` is the strongest correctness check available.
* ``bates_holed.stl`` -- the same grain with faces deleted. Exercises the
  non-watertight path that motivated choosing the generalized winding number
  over ray casting (#158).
* ``finocyl.stl``    -- a bore with radial fins. Not axially uniform, so it
  cannot be represented as an extruded 2D cross-section; this is the case that
  genuinely requires 3D geometry.
* ``bates_torn.stl`` -- one *contiguous* 90-degree wedge removed. Where
  ``bates_holed`` shows the winding number shrugging off scattered damage,
  this is the case that shows where it actually degrades.

Each fixture is meant to isolate one variable, and ``main`` asserts that they
still do -- a shape fixture that is also accidentally damaged tests two things
at once and can fail for either reason (#178).

Run with::

    python examples/make_test_meshes.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

# Standard validation dimensions, matching tests/test_bates_analytical.py.
INNER_RADIUS = 0.01
OUTER_RADIUS = 0.05
LENGTH = 0.12

OUT_DIR = Path(__file__).parent / "output" / "meshes"


def make_bates(
    inner: float = INNER_RADIUS,
    outer: float = OUTER_RADIUS,
    length: float = LENGTH,
    sections: int = 128,
) -> trimesh.Trimesh:
    """Cylinder with a concentric cylindrical bore, as a closed mesh.

    Built directly with ``trimesh.creation.annulus`` rather than by boolean
    subtraction: an annulus *is* a hollow cylinder, so this needs no boolean
    backend and produces an exactly closed surface with no risk of coplanar
    slivers where a subtracted bore meets the end faces.

    ``sections`` controls the faceting -- higher means the mesh approximates the
    true circle more closely. This is exactly the tessellation-density knob that
    limits how accurately an imported grain can match the analytical one.
    """
    return trimesh.creation.annulus(
        r_min=inner, r_max=outer, height=length, sections=sections
    )


def make_finocyl(
    inner: float = INNER_RADIUS,
    outer: float = OUTER_RADIUS,
    length: float = LENGTH,
    n_fins: int = 6,
    fin_length: float = 0.025,
    fin_width: float = 0.006,
    fin_fraction: float = 0.6,
) -> trimesh.Trimesh:
    """A bore with radial fins running along part of the grain's length.

    The fins stop partway down the grain, so the cross-section *changes* with
    axial position. That is what makes this shape impossible to express as an
    extruded 2D profile (#139) and therefore a genuine 3D-only case.
    """
    grain = make_bates(inner, outer, length)

    fin_axial_length = length * fin_fraction
    # Offset the fins toward one end so the geometry is asymmetric in z --
    # a uniform-along-z shape would not test anything 3D-specific.
    z_centre = -length / 2 + fin_axial_length / 2

    for i in range(n_fins):
        angle = 2 * np.pi * i / n_fins
        fin = trimesh.creation.box(
            extents=(fin_length, fin_width, fin_axial_length)
        )
        # Rotate about the origin first, then push the slot outward so it
        # starts at the bore and cuts into the web. Order matters: translating
        # before rotating would swing the slot away from the bore.
        fin.apply_transform(
            trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        )
        fin.apply_translation(
            [(inner + fin_length / 2) * np.cos(angle),
             (inner + fin_length / 2) * np.sin(angle),
             z_centre]
        )
        grain = grain.difference(fin)

    return grain


def make_holed(mesh: trimesh.Trimesh, fraction: float = 0.15,
               seed: int = 0) -> trimesh.Trimesh:
    """Delete a random fraction of faces, producing a non-watertight mesh.

    Simulates the real-world defects that break ray-casting-based inside/outside
    tests: a ray fired through a gap escapes and the crossing-parity count is
    wrong. The generalized winding number should still sign these correctly.
    """
    rng = np.random.default_rng(seed)
    n_faces = len(mesh.faces)
    keep = rng.permutation(n_faces)[: int(n_faces * (1.0 - fraction))]
    holed = mesh.copy()
    holed.update_faces(np.isin(np.arange(n_faces), keep))
    return holed


def make_torn(mesh: trimesh.Trimesh, degrees: float = 90.0) -> trimesh.Trimesh:
    """Remove one contiguous angular wedge, leaving a single large hole.

    This is the damage that actually tests the generalized winding number.
    ``make_holed`` deletes faces at *random*, which scatters isolated gaps that
    are almost all smaller than the grid spacing -- invisible to a field
    sampled on a grid, and comfortably inside the regime where the winding
    number is exact. It shows the method does not fall over; it does not find
    where it breaks.

    A wedge does. The winding number measures how much of the full surround the
    boundary covers, so removing a contiguous solid angle drives ``w`` down in
    proportion to what is missing, and the 0.5 threshold is crossed when roughly
    half the surface is gone. Measured on a 128-section BATES:

        wedge     faces removed    sign accuracy    mean w inside
          30 deg        86             99.5%            0.914
          60 deg       170             97.1%            0.832
          90 deg       256             93.0%            0.746
         180 deg       512             83.4%            0.495
         240 deg       682             77.3%            0.328

    So the honest statement of the method's limit is not "it tolerates holes"
    but "it tolerates missing surround, and degrades in proportion to it".
    """
    centres = mesh.triangles_center
    angle = np.degrees(np.arctan2(centres[:, 1], centres[:, 0]))
    torn = mesh.copy()
    torn.update_faces(np.abs(angle) > degrees / 2.0)
    return torn


def _report(name: str, mesh: trimesh.Trimesh) -> None:
    print(
        f"  {name:20s} {len(mesh.faces):>7,} tris  "
        f"watertight={str(mesh.is_watertight):<5}  "
        f"extents={np.round(mesh.extents, 4).tolist()}"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"writing test meshes to {OUT_DIR}")

    bates = make_bates()
    bates.export(OUT_DIR / "bates.stl")
    _report("bates.stl", bates)

    holed = make_holed(bates)
    holed.export(OUT_DIR / "bates_holed.stl")
    _report("bates_holed.stl", holed)

    finocyl = make_finocyl()
    finocyl.export(OUT_DIR / "finocyl.stl")
    _report("finocyl.stl", finocyl)

    torn = make_torn(bates)
    torn.export(OUT_DIR / "bates_torn.stl")
    _report("bates_torn.stl", torn)

    # The fixtures are only useful if each isolates the variable it is named
    # for (#178). finocyl in particular is meant to test axial non-uniformity,
    # and was once accidentally non-watertight as well -- which quietly made
    # every test using it exercise the damaged-mesh path too.
    for name, mesh, should_be_closed in (
        ("bates", bates, True),
        ("finocyl", finocyl, True),
        ("bates_holed", holed, False),
        ("bates_torn", torn, False),
    ):
        if mesh.is_watertight is not should_be_closed:
            raise AssertionError(
                f"{name}.stl is {'not ' if should_be_closed else ''}watertight "
                f"but was meant to be {'closed' if should_be_closed else 'open'}"
                " -- the fixture no longer isolates the variable it is named for."
            )
    print("\nfixtures verified: closed meshes closed, damaged meshes damaged")

    print("\nExpected for bates.stl:")
    print(f"  bore radius   {INNER_RADIUS} m")
    print(f"  outer radius  {OUTER_RADIUS} m")
    print(f"  length        {LENGTH} m")
    volume = np.pi * (OUTER_RADIUS**2 - INNER_RADIUS**2) * LENGTH
    print(f"  volume        {volume:.6e} m^3 (analytical)")
    print(f"  volume        {bates.volume:.6e} m^3 (mesh)")


if __name__ == "__main__":
    main()
