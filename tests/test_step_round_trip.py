"""A real STEP export, all the way to a burnback, against closed form (#171).

Every other test in the suite builds its geometry in memory, which means the
whole CAD path — a file written by Inventor, its declared units, the
OpenCASCADE tessellation, the vertex merge — is exercised nowhere. This one
starts from `BATES.stp` as exported and ends at a burnout time, so a regression
anywhere along that chain fails here.

The part is 1.000 in outside diameter, 0.200 in bore, 2.000 in long, drawn in
inches. Those are the numbers every assertion below is written against, so a
unit-handling regression shows up as a wrong dimension rather than as a
mysterious tolerance failure.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from srm_burnback.geometry.import_mesh import (
    load_mesh,
    mesh_stats,
    orient_grain_axis_to_z,
)
from srm_burnback.geometry.measurements import grain_measurements

trimesh = pytest.importorskip("trimesh")
# The CAD path is an optional extra; without it this whole file is meaningless
# rather than failing.
cascadio = pytest.importorskip("cascadio")

STEP_FILE = Path(__file__).resolve().parents[1] / "BATES.stp"

INCH = 0.0254
OUTER_DIAMETER = 1.000 * INCH
BORE_DIAMETER = 0.200 * INCH
LENGTH = 2.000 * INCH
WEB = (OUTER_DIAMETER - BORE_DIAMETER) / 2.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def imported():
    """The STEP file, imported exactly as the application imports it."""
    if not STEP_FILE.is_file():
        pytest.skip(f"{STEP_FILE.name} is not present")
    mesh = load_mesh(STEP_FILE)
    if not mesh.is_winding_consistent:
        mesh.fix_normals()
    mesh, _ = orient_grain_axis_to_z(mesh)
    return mesh


@pytest.fixture(scope="module")
def centred(imported):
    """The same part, moved onto the origin.

    Inventor exported this one with its base at z = 0 rather than its centre,
    so as imported it spans z = 0 .. 0.0508. Every other geometry in the
    package is origin-centred -- ``BATESGrain`` measures radius from the origin
    and ``|z|`` against ``length / 2``, and ``build_coords`` lays the domain
    symmetrically about it -- so comparing the raw import against the oracle
    compares two parts an inch apart.

    ``MeshGrain`` does this centring itself, which is why the burnback test
    below needs no equivalent. This fixture is what that would look like done
    by hand.
    """
    mesh = imported.copy()
    mesh.apply_translation(-(imported.bounds[0] + imported.bounds[1]) / 2.0)
    return mesh


class TestTheFixtureItself:
    def test_it_is_committed_and_small_enough_for_version_control(self):
        assert STEP_FILE.is_file()
        # A few kilobytes of B-rep, against megabytes for a fine STL of the
        # same part -- which is the argument for keeping CAD in the repo.
        assert STEP_FILE.stat().st_size < 100_000

    def test_it_declares_inches(self):
        from srm_burnback.geometry.units import step_length_unit

        name, metres = step_length_unit(STEP_FILE)
        assert name == "inch"
        assert metres == pytest.approx(INCH)


class TestImportedDimensions:
    """The part as drawn, recovered from the tessellation."""

    def test_the_geometry_arrives_in_metres(self, imported):
        stats = mesh_stats(imported)
        assert stats["source_units"] == "inch"
        assert stats["units_origin"] == "declared"
        assert max(stats["extents"]) == pytest.approx(LENGTH, rel=1e-3)

    def test_the_tessellation_is_closed(self, imported):
        """glTF stores vertices per face, so this fails without the merge.

        Volume, mass and port fraction are all gated on watertightness, so an
        unmerged import silently loses three of the numbers a user most wants.
        """
        assert imported.is_watertight

    def test_recovered_dimensions_match_the_part_as_drawn(self, imported):
        measured = grain_measurements(imported, density=1750.0)
        assert measured["length"] == pytest.approx(LENGTH, rel=2e-3)
        assert measured["outer_diameter"] == pytest.approx(OUTER_DIAMETER, rel=2e-3)
        assert measured["bore_diameter"] == pytest.approx(BORE_DIAMETER, rel=2e-2)
        assert measured["web_thickness"] == pytest.approx(WEB, rel=2e-2)
        assert measured["volume"] is not None
        assert measured["mass"] is not None


class TestPhiAgainstClosedForm:
    """Imported φ versus ``BATESGrain``, the analytical SDF."""

    @pytest.fixture(scope="class")
    def field(self, centred):
        from srm_burnback.geometry.BATESGrain import BATESGrain
        from srm_burnback.geometry.CoordinateBuilder import build_coords
        from srm_burnback.geometry.winding_number import phi_from_labelled_mesh

        resolution = 64 if DEVICE == "cuda" else 32
        # The domain has to enclose the part: it is 2 in long, so a half-extent
        # smaller than 1 in would cut through it and the comparison would be
        # against an oracle evaluated outside its own geometry.
        half = LENGTH / 2 * 1.1
        coords, h = build_coords(resolution, half, length=2 * half, device=DEVICE)
        coords = tuple(c.double() for c in coords)

        # ``ends="burning"`` on purpose: that is the labelling BATESGrain
        # models -- max(bore, |z| - L/2), bore and both end faces burning, no
        # outer wall. Comparing the shipped inhibited-ends default against it
        # would be comparing two different motors. The inhibited configuration
        # is exercised by the burnback test below, where the oracle is the
        # closed-form web/rate rather than BATESGrain's field.
        phi, phi_outer, _ = phi_from_labelled_mesh(
            coords, centred, ends="burning", dtype=torch.float64
        )
        oracle = BATESGrain(
            BORE_DIAMETER / 2, OUTER_DIAMETER / 2, LENGTH
        ).signed_distance(coords)
        return coords, h, phi, phi_outer, oracle

    def test_signs_agree_inside_the_casing(self, field):
        coords, h, phi, _, oracle = field
        X, Y, Z = coords
        radius = torch.sqrt(X**2 + Y**2)
        inside = (radius < OUTER_DIAMETER / 2 - 2 * h) & (
            Z.abs() < LENGTH / 2 - 2 * h
        )
        assert ((phi > 0) == (oracle > 0))[inside].all()

    def test_distance_matches_closed_form_near_the_surface(self, field):
        coords, h, phi, _, oracle = field
        X, Y, Z = coords
        radius = torch.sqrt(X**2 + Y**2)
        band = (
            (oracle.abs() < 3 * h)
            & (radius < OUTER_DIAMETER / 2 - 2 * h)
            & (Z.abs() < LENGTH / 2 - 2 * h)
        )
        error = (phi - oracle).abs()[band].mean().item() / h
        # The measured cost of going through CAD rather than closed form:
        # **0.0029 h mean, 0.0054 h max** at 64 cells across, from 652
        # OpenCASCADE triangles. The residual is the tessellation -- the bore
        # is a polygon inscribed in the circle the oracle uses exactly.
        #
        # For scale, the rejected sign-field reconstruction path was 0.166 h
        # (see test_winding_number.py), so the whole CAD round trip costs
        # roughly fifty times less than that one shortcut would have.
        assert error < 0.02, f"mean error {error:.4f} h"


class TestBurnbackAgainstClosedForm:
    """The whole chain: STEP file to burnout time."""

    def test_burnout_matches_the_analytical_answer(self, imported):
        from srm_burnback.geometry.MeshGrain import MeshGrain
        from srm_burnback.physics.vieille import VieilleBurnRate
        from srm_burnback.simulation.config import SimulationConfig
        from srm_burnback.simulation.runner import BurnbackSimulation

        grain = MeshGrain(imported, ends="inhibited")
        burn_rate = VieilleBurnRate(a=0.005, n=0.35)
        rate = float(burn_rate.compute_burn_rate(5.0))
        analytical = WEB / rate

        simulation = BurnbackSimulation(
            SimulationConfig(
                resolution=40 if DEVICE == "cuda" else 24,
                domain_size=grain.default_domain_size(),
                length=grain.default_length(),
                device=DEVICE,
                max_time=10.0,
                pressure=5.0,
                verbose=False,
            ),
            grain,
            burn_rate,
        )
        simulation.run()

        # Burnout, not the cap.
        assert simulation.time < 10.0
        # Coarse grids over-predict -- the front has to cross a whole cell to
        # register -- so the tolerance is one-sided and generous. The point is
        # that a real CAD file produces the right quantity end to end, not that
        # a 40-cell grid is converged.
        assert analytical <= simulation.time <= analytical * 1.6
