"""Recovered grain dimensions, against the BATES analytical oracle (#190).

``grain_measurements`` is the first thing a user sees after an import and the
main signal that the import worked, so a silently wrong web thickness would be
believed. It is also a *heuristic* rather than a formula -- faces are sorted
into "outer wall", "bore" and "end cap" by how their normals line up with the
radial direction, with :data:`RADIAL_THRESHOLD` as the cutoff -- and a heuristic
with no test is just an assumption.

An exact oracle is available, which is what makes this worth testing properly:
an annulus of known bore, outer radius and length *is* a BATES grain, so every
recovered number can be checked against closed form rather than against a
previous run.

Two properties get most of the attention here:

* **radii are exact at any tessellation**, because they are measured at
  vertices. A tessellated circle is an inscribed polygon whose vertices lie on
  the true circle while every face centroid sits inside it. Measuring at
  centroids under-reports every diameter by ``cos(pi/N)`` -- the bug fixed in
  ``a35748e``, and the reason for :class:`TestVertexRadiiRegression`.
* **volume converges second-order**, because it is a genuine polygon-area
  deficit that no amount of cleverness removes -- only refinement.
"""

from __future__ import annotations

import numpy as np
import pytest

from srm_burnback.geometry.measurements import RADIAL_THRESHOLD, grain_measurements

trimesh = pytest.importorskip("trimesh")

# The project's standard validation dimensions, matching
# tests/test_bates_analytical.py and examples/make_test_meshes.py. Metres.
BORE_RADIUS = 0.01
OUTER_RADIUS = 0.05
LENGTH = 0.12
DENSITY = 1750.0  # kg/m^3, a typical composite


def bates_mesh(sections: int = 128, bore: float = BORE_RADIUS,
               outer: float = OUTER_RADIUS, length: float = LENGTH):
    """A BATES grain as a closed mesh, built the way the test fixtures are.

    ``annulus`` rather than a boolean subtraction: an annulus *is* a hollow
    cylinder, so this needs no boolean backend and produces an exactly closed
    surface with no coplanar slivers where a subtracted bore meets an end face.
    """
    return trimesh.creation.annulus(
        r_min=bore, r_max=outer, height=length, sections=sections
    )


def analytical_volume(bore=BORE_RADIUS, outer=OUTER_RADIUS, length=LENGTH) -> float:
    return float(np.pi * (outer**2 - bore**2) * length)


class TestAgainstTheOracle:
    """Recovered dimensions versus the values the grain was built from."""

    @pytest.fixture(scope="class")
    def measured(self):
        return grain_measurements(bates_mesh(), density=DENSITY)

    def test_length_is_exact(self, measured):
        # Length comes from the bounding box and the end caps are planar, so
        # tessellation cannot affect it at all.
        assert measured["length"] == pytest.approx(LENGTH, rel=1e-12)

    def test_outer_radius(self, measured):
        assert measured["outer_radius"] == pytest.approx(OUTER_RADIUS, rel=1e-9)

    def test_bore_radius(self, measured):
        assert measured["bore_radius"] == pytest.approx(BORE_RADIUS, rel=1e-9)

    def test_diameters_are_twice_the_radii(self, measured):
        assert measured["outer_diameter"] == pytest.approx(2 * OUTER_RADIUS, rel=1e-9)
        assert measured["bore_diameter"] == pytest.approx(2 * BORE_RADIUS, rel=1e-9)

    def test_web_thickness(self, measured):
        # The web is what has to burn through, so it sets the burn time -- the
        # single most consequential number on the panel.
        assert measured["web_thickness"] == pytest.approx(
            OUTER_RADIUS - BORE_RADIUS, rel=1e-9
        )

    def test_length_to_diameter(self, measured):
        assert measured["length_to_diameter"] == pytest.approx(
            LENGTH / (2 * OUTER_RADIUS), rel=1e-9
        )

    def test_volume_within_tessellation_slack(self, measured):
        # 128 sections leaves ~0.04% of the true area outside the inscribed
        # polygon. See TestConvergence for where that number comes from.
        assert measured["volume"] == pytest.approx(analytical_volume(), rel=1e-3)

    def test_mass_is_volume_times_density(self, measured):
        assert measured["mass"] == pytest.approx(measured["volume"] * DENSITY, rel=1e-12)

    def test_port_fraction_matches_the_bore_it_was_cut_with(self, measured):
        # Fraction of the envelope hollowed out: (r_bore/r_outer)^2 for a BATES.
        expected = (BORE_RADIUS / OUTER_RADIUS) ** 2
        assert measured["port_fraction"] == pytest.approx(expected, rel=2e-2)


class TestVertexRadiiRegression:
    """Radii must come from vertices, not face centroids (regression: a35748e).

    This is the whole point of measuring at vertices: it holds at *coarse*
    tessellation, where the centroid bug was largest and most visible. A part
    modelled as 1.000 in must read 1.000 in, not 0.996 in.
    """

    @pytest.mark.parametrize("sections", [8, 12, 16, 32, 64, 128, 256])
    def test_radii_are_exact_at_every_tessellation(self, sections):
        measured = grain_measurements(bates_mesh(sections=sections))
        assert measured["outer_radius"] == pytest.approx(OUTER_RADIUS, rel=1e-9)
        assert measured["bore_radius"] == pytest.approx(BORE_RADIUS, rel=1e-9)

    def test_coarse_mesh_would_fail_under_centroid_measurement(self):
        # Pin the size of the bug this guards against, so the test above cannot
        # quietly become vacuous: at 8 sections a centroid-derived outer radius
        # is short by ~7.6%, far outside the 1e-9 tolerance asserted above.
        sections = 8
        centroid_radius = OUTER_RADIUS * np.cos(np.pi / sections)
        assert abs(centroid_radius - OUTER_RADIUS) / OUTER_RADIUS > 0.07
        assert grain_measurements(bates_mesh(sections=sections))[
            "outer_radius"
        ] == pytest.approx(OUTER_RADIUS, rel=1e-9)


class TestConvergence:
    """Volume error must be a function of facet size, not an arbitrary number.

    An inscribed N-gon of circumradius r has area ``0.5 N r^2 sin(2 pi / N)``,
    short of the circle by ``O(1/N^2)``. So the tolerance on volume is not a
    matter of taste: it is set by tessellation density, and it must fall by 4x
    each time the section count doubles.
    """

    SECTIONS = [16, 32, 64, 128, 256]

    @pytest.fixture(scope="class")
    def errors(self):
        exact = analytical_volume()
        return [
            abs(grain_measurements(bates_mesh(sections=n))["volume"] - exact) / exact
            for n in self.SECTIONS
        ]

    def test_error_shrinks_monotonically_under_refinement(self, errors):
        assert all(b < a for a, b in zip(errors, errors[1:])), errors

    def test_convergence_is_second_order(self, errors):
        # Slope of log(error) against log(sections). Second order in facet size
        # means -2 here; anything shallower would mean the volume is limited by
        # something other than tessellation, which would be a real defect.
        slope = np.polyfit(np.log(self.SECTIONS), np.log(errors), 1)[0]
        assert slope == pytest.approx(-2.0, abs=0.15), f"order {slope:.2f}"

    def test_tolerance_matches_the_predicted_polygon_deficit(self, errors):
        # The measured error should track the closed-form area deficit of the
        # inscribed polygon, tying the tolerance to geometry rather than to a
        # previously observed number.
        for n, error in zip(self.SECTIONS, errors):
            predicted = 1.0 - (n / (2 * np.pi)) * np.sin(2 * np.pi / n)
            assert error == pytest.approx(predicted, rel=0.05)


class TestDegenerateGeometry:
    """Absent quantities must come back ``None``, never fabricated."""

    def test_open_mesh_has_no_volume_or_mass(self):
        mesh = bates_mesh(sections=64)
        rng = np.random.default_rng(0)
        n_faces = len(mesh.faces)
        keep = rng.permutation(n_faces)[: int(n_faces * 0.85)]
        mesh.update_faces(np.isin(np.arange(n_faces), keep))
        assert not mesh.is_watertight

        measured = grain_measurements(mesh, density=DENSITY)
        # Volume of an open surface is meaningless, and trimesh will happily
        # return a number for one anyway -- so this must be refused here.
        assert measured["volume"] is None
        assert measured["mass"] is None
        assert measured["port_fraction"] is None

    def test_open_mesh_still_reports_radii(self):
        # Radii come from surviving faces, not from closure, so a damaged mesh
        # should still be measurable -- that is the whole reason for holding
        # each quantity independently rather than bailing out wholesale.
        mesh = bates_mesh(sections=64)
        rng = np.random.default_rng(0)
        n_faces = len(mesh.faces)
        keep = rng.permutation(n_faces)[: int(n_faces * 0.85)]
        mesh.update_faces(np.isin(np.arange(n_faces), keep))

        measured = grain_measurements(mesh)
        assert measured["outer_radius"] == pytest.approx(OUTER_RADIUS, rel=1e-9)
        assert measured["bore_radius"] == pytest.approx(BORE_RADIUS, rel=1e-9)

    def test_solid_cylinder_has_no_bore_and_does_not_raise(self):
        solid = trimesh.creation.cylinder(radius=OUTER_RADIUS, height=LENGTH,
                                          sections=64)
        measured = grain_measurements(solid, density=DENSITY)
        assert measured["bore_radius"] is None
        assert measured["bore_diameter"] is None
        assert measured["web_thickness"] is None
        assert measured["outer_radius"] == pytest.approx(OUTER_RADIUS, rel=1e-9)
        # Near zero rather than exactly zero: the residue is the inscribed
        # polygon's area deficit, not a bore.
        assert measured["port_fraction"] == pytest.approx(0.0, abs=5e-3)

    def test_no_density_means_no_mass_but_everything_else_survives(self):
        measured = grain_measurements(bates_mesh(sections=32))
        assert measured["mass"] is None
        assert measured["volume"] is not None

    def test_measurements_are_plain_python_scalars(self):
        # The panel renders these and save/load (#155) will serialize them;
        # numpy scalars leak dtypes into both.
        for key, value in grain_measurements(bates_mesh(sections=16),
                                             density=DENSITY).items():
            assert value is None or isinstance(value, float), f"{key}: {type(value)}"


class TestRadialThreshold:
    """How far the face-classification heuristic can be pushed (#190, step 5).

    Classification depends on the angle between a face normal and the radial
    direction, so the quantity that matters is not "is this shape complicated"
    but "how steeply is this wall inclined". A wall rising at ``alpha`` from the
    grain axis has radial alignment ``cos(alpha)``, so the cutoff is reached at
    ``acos(RADIAL_THRESHOLD)`` -- about 69.5 degrees at the current 0.35.

    That is far past anything real: bore tapers in motor design are single-digit
    degrees. The tests below establish where the boundary actually is, and --
    more importantly -- that crossing it makes the bore *unreported* rather than
    misreported.
    """

    OUTER = 0.05
    GRAIN_LENGTH = 0.02  # short, so a steep taper still fits inside the web
    BORE = 0.008

    def tapered_bore(self, alpha_deg: float):
        """A cylinder with a conical bore inclined ``alpha_deg`` from the axis."""
        body = trimesh.creation.cylinder(
            radius=self.OUTER, height=self.GRAIN_LENGTH, sections=128
        )
        slope = np.tan(np.radians(alpha_deg)) * self.GRAIN_LENGTH
        profile = np.array(
            [
                [0.0, -self.GRAIN_LENGTH],
                [self.BORE - slope / 2, -self.GRAIN_LENGTH],
                [self.BORE + slope + 1e-3, self.GRAIN_LENGTH],
                [0.0, self.GRAIN_LENGTH],
            ]
        )
        cone = trimesh.creation.revolve(profile, sections=128)
        return body.difference(cone)

    def test_threshold_is_where_the_docstring_says_it_is(self):
        assert np.degrees(np.arccos(RADIAL_THRESHOLD)) == pytest.approx(69.5, abs=0.5)

    @pytest.mark.parametrize("alpha", [1, 5, 30, 60, 65])
    def test_realistic_tapers_are_classified_as_bore(self, alpha):
        measured = grain_measurements(self.tapered_bore(alpha))
        assert measured["bore_radius"] is not None
        assert 0.0 < measured["bore_radius"] < self.OUTER

    def test_beyond_the_cutoff_the_bore_is_dropped_not_mismeasured(self):
        # An 80-degree wall reads as an end cap, so no inward-facing surface is
        # found. Returning None is the correct failure: the panel shows a blank
        # rather than a plausible wrong number, which is the outcome that
        # matters for a value the user is about to trust.
        measured = grain_measurements(self.tapered_bore(80))
        assert measured["bore_radius"] is None
        assert measured["web_thickness"] is None
        # The outer wall is unaffected -- one mis-classified surface must not
        # take the rest of the measurement down with it.
        assert measured["outer_radius"] == pytest.approx(self.OUTER, rel=1e-6)

    def test_axially_varying_grain_is_measured_from_its_extremes(self):
        """A finocyl: fins cut into the web for part of the length only.

        The cross-section changes with z, so there is no single "the bore".
        Extremes rather than means is what makes the answer well defined --
        the bore is the closest material to the axis anywhere, and the outer
        wall the furthest, with the slot walls in between pulling a mean around.
        """
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
        from make_test_meshes import make_finocyl

        measured = grain_measurements(make_finocyl())
        assert measured["bore_radius"] == pytest.approx(BORE_RADIUS, rel=1e-4)
        assert measured["outer_radius"] == pytest.approx(OUTER_RADIUS, rel=1e-4)
        # Slots remove propellant, so more of the envelope is open than the
        # plain BATES (r_bore/r_outer)^2 = 4%.
        assert measured["port_fraction"] > (BORE_RADIUS / OUTER_RADIUS) ** 2


class TestUnitsInvariant:
    """Measurements are metres, because the import brought the mesh to metres.

    Guards the class of defect in #170: a millimetre-drawn grain that was never
    converted reports a volume 10^9 too large, and nothing about the number
    itself says so.
    """

    def test_millimetre_stl_measures_in_metres_once_scaled(self, tmp_path):
        from srm_burnback.geometry.import_mesh import load_mesh

        drawn_in_mm = trimesh.creation.annulus(
            r_min=10.0, r_max=50.0, height=120.0, sections=128
        )
        path = tmp_path / "grain_mm.stl"
        drawn_in_mm.export(path)

        measured = grain_measurements(
            load_mesh(path, assume_units="mm"), density=DENSITY
        )
        # Looser than the in-memory tests above: STL stores single-precision
        # floats, so a round trip through the file costs ~7 significant digits
        # no matter how exact the measurement is. Still ~5 orders of magnitude
        # tighter than the 1000x error this guards against.
        assert measured["outer_radius"] == pytest.approx(OUTER_RADIUS, rel=1e-6)
        assert measured["length"] == pytest.approx(LENGTH, rel=1e-6)
        assert measured["volume"] == pytest.approx(analytical_volume(), rel=1e-3)

    def test_a_real_grain_masses_in_kilograms_not_tonnes(self):
        # The end-to-end smell test: this grain is a few kilograms. A unit slip
        # anywhere upstream lands this in the thousands or the millionths.
        mass = grain_measurements(bates_mesh(), density=DENSITY)["mass"]
        assert 0.5 < mass < 50.0
