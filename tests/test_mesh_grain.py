"""An imported mesh driving the solver, as a ``GrainGeometry`` (#192).

This is the join between two halves that were previously tested separately: the
import pipeline produced a validated φ, and the runner consumed a
``GrainGeometry``, and nothing connected them. What matters here is not that φ
is correct -- ``test_winding_number.py`` covers that -- but that a mesh behaves
like a grain: the runner accepts it unchanged, the casing clamp works on it,
and a burnback on an imported BATES lands where the parametric one does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from srm_burnback.geometry.BATESGrain import BATESGrain
from srm_burnback.geometry.MeshGrain import MeshGrain
from srm_burnback.physics.vieille import VieilleBurnRate
from srm_burnback.simulation.config import SimulationConfig
from srm_burnback.simulation.runner import BurnbackSimulation

trimesh = pytest.importorskip("trimesh")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from make_test_meshes import (  # noqa: E402
    INNER_RADIUS,
    LENGTH,
    OUTER_RADIUS,
    make_bates,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Cost is O(cells x triangles); these run the full solver, so keep both small.
RESOLUTION = 40 if DEVICE == "cuda" else 24
SECTIONS = 64
PRESSURE = 5.0


@pytest.fixture(scope="module")
def grain():
    return MeshGrain(make_bates(sections=SECTIONS), ends="inhibited")


@pytest.fixture(scope="module")
def burn_rate():
    return VieilleBurnRate(a=0.005, n=0.35)


def config(grain, resolution=RESOLUTION):
    """A run configuration sized to ``grain``.

    ``default_length`` is a MeshGrain addition -- the parametric grains size
    their axial extent from their own ``length`` -- so the parametric
    comparison below reuses the mesh grain's grid rather than asking for one.
    Comparing burnout times only means anything on an identical grid anyway.
    """
    return SimulationConfig(
        resolution=resolution,
        domain_size=grain.default_domain_size(),
        length=grain.default_length(),
        device=DEVICE,
        max_time=12.0,
        pressure=PRESSURE,
        verbose=False,
    )


class TestGeometryInterface:
    def test_it_is_a_grain_geometry(self, grain):
        from srm_burnback.geometry.GrainGeometry import GrainGeometry

        assert isinstance(grain, GrainGeometry)
        assert grain.ndim == 3

    def test_sign_convention_matches_the_other_geometries(self, grain):
        phi, coords, _ = grain.initialize_grid(
            RESOLUTION, length=grain.default_length(), device=DEVICE
        )
        X, Y, Z = coords
        radius = torch.sqrt(X**2 + Y**2)
        mid = Z.abs() < LENGTH / 2 - 0.005
        # Positive in the open bore, negative in the propellant. Backwards
        # would burn the grain inward from infinity and still look smooth.
        assert (phi[(radius < 0.008) & mid] > 0).all()
        assert (phi[(radius > 0.015) & (radius < 0.045) & mid] < 0).all()

    def test_casing_field_has_the_right_sign_convention(self, grain):
        _, coords, _ = grain.initialize_grid(
            RESOLUTION, length=grain.default_length(), device=DEVICE
        )
        outer = grain.outer_boundary_distance(coords)
        X, Y, Z = coords
        radius = torch.sqrt(X**2 + Y**2)
        # Negative inside the wall, positive beyond -- what the runner's
        # min(phi, -phi_outer) clamp expects.
        assert (outer[(radius < 0.04) & (Z.abs() < 0.05)] < 0).all()
        assert (outer[radius > 0.058] > 0).all()

    def test_the_mesh_is_centred_on_the_origin(self):
        """Every other geometry is origin-centred, so this one must be too.

        ``BATESGrain`` measures radius from the origin and ``|z|`` against
        ``length / 2``, and ``build_coords`` lays the domain symmetrically about
        it. A grain sitting somewhere else would need ``initialize_grid`` --
        the one genuinely grain-agnostic piece of setup -- to know it was
        dealing with a mesh.
        """
        mesh = make_bates(sections=SECTIONS)
        mesh.apply_translation([0.3, -0.2, 0.7])
        offset_grain = MeshGrain(mesh)

        assert offset_grain.offset == pytest.approx((0.3, -0.2, 0.7), abs=1e-6)
        centre = (offset_grain.mesh.bounds[0] + offset_grain.mesh.bounds[1]) / 2
        assert centre == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)

    def test_domain_defaults_enclose_the_grain_with_room_to_spare(self, grain):
        assert grain.default_domain_size() > OUTER_RADIUS
        assert grain.default_length() > LENGTH
        # A cube, so h is identical on every axis -- the Godunov scheme and the
        # CFL condition each take one scalar h.
        assert grain.default_length() == pytest.approx(
            2 * grain.default_domain_size()
        )

    def test_a_mesh_with_no_burning_surface_is_refused(self):
        """A solid cylinder has no bore, so there is nothing to ignite.

        Failing at construction beats producing a field with no zero contour
        and letting the runner report instant burnout.
        """
        solid = trimesh.creation.cylinder(radius=OUTER_RADIUS, height=LENGTH,
                                          sections=32)
        with pytest.raises(ValueError, match="no burning surface"):
            MeshGrain(solid)

    def test_two_dimensional_coords_are_refused(self, grain):
        coords = (torch.zeros(4, 4), torch.zeros(4, 4))
        with pytest.raises(ValueError, match="3D geometry"):
            grain.signed_distance(coords)


class TestFieldIsComputedOnce:
    """φ costs seconds; the interface asks for it through two methods."""

    def test_the_casing_call_reuses_the_field(self, grain):
        import time

        fresh = MeshGrain(make_bates(sections=SECTIONS), ends="inhibited")
        started = time.perf_counter()
        _, coords, _ = fresh.initialize_grid(
            RESOLUTION, length=fresh.default_length(), device=DEVICE
        )
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        first = time.perf_counter() - started

        started = time.perf_counter()
        fresh.outer_boundary_distance(coords)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        second = time.perf_counter() - started

        # A cache hit, not a second traversal. Measured ~1.3 s then ~0.05 ms.
        assert second < first / 20

    def test_a_different_grid_is_recomputed(self, grain):
        phi_a, coords_a, _ = grain.initialize_grid(
            RESOLUTION, length=grain.default_length(), device=DEVICE
        )
        phi_b, coords_b, _ = grain.initialize_grid(
            RESOLUTION + 8, length=grain.default_length(), device=DEVICE
        )
        # Caching on tensor identity must not hand back a stale field of the
        # wrong shape when the grid changes.
        assert phi_a.shape != phi_b.shape
        assert phi_b.shape == coords_b[0].shape


class TestBurnbackAgainstTheParametricGrain:
    """The round-trip this issue exists for: does an STL burn like a BATES?"""

    @staticmethod
    def _burnout(geometry, burn_rate, grid=None, resolution=RESOLUTION):
        simulation = BurnbackSimulation(
            config(grid or geometry, resolution), geometry, burn_rate
        )
        simulation.run()
        return simulation

    def test_the_runner_accepts_a_mesh_unchanged(self, grain, burn_rate):
        simulation = self._burnout(grain, burn_rate)
        assert simulation.step > 0
        assert simulation.time > 0.0
        # Burnout, not the max_time cap.
        assert simulation.time < config(grain).max_time

    def test_burnout_matches_the_parametric_grain(self, grain, burn_rate):
        mesh_time = self._burnout(grain, burn_rate).time
        parametric = BATESGrain(INNER_RADIUS, OUTER_RADIUS, LENGTH)
        parametric_time = self._burnout(parametric, burn_rate, grid=grain).time

        # Within tessellation tolerance. The residual is the imported bore
        # being a polygon inscribed in the circle the parametric grain uses
        # exactly; both converge on the analytical answer as resolution rises
        # (measured 9.5% apart at 32^3 down to 3.2% at 96^3). The tolerance is
        # loose because the grid here is deliberately coarse to keep the suite
        # quick -- this asserts they agree, not that either is converged.
        assert mesh_time == pytest.approx(parametric_time, rel=0.2)

    def test_burnout_is_the_right_order_against_closed_form(
        self, grain, burn_rate
    ):
        rate = float(burn_rate.compute_burn_rate(PRESSURE))
        analytical = (OUTER_RADIUS - INNER_RADIUS) / rate
        mesh_time = self._burnout(grain, burn_rate).time
        # Coarse grids over-predict; the point is the answer is the right
        # quantity, not that a 40^3 grid is converged.
        assert analytical < mesh_time < analytical * 1.5

    def test_the_casing_holds_the_outer_wall(self, grain, burn_rate):
        """Inhibited surfaces must not recede -- the #176 acceptance test.

        Without the clamp the front advances from every zero contour, and the
        grain is eaten from the outside in as well as the inside out.
        """
        simulation = self._burnout(grain, burn_rate)
        X, Y, _ = simulation.coords
        radius = torch.sqrt(X**2 + Y**2)

        beyond_wall = radius > OUTER_RADIUS + 2 * simulation.h
        live = (simulation.phi < 0) & (simulation.phi_outer < 0)
        assert not bool((live & beyond_wall).any())

        # And the casing mask itself still ends at the wall it was built from.
        inside = radius[simulation.phi_outer < 0]
        assert float(inside.max()) == pytest.approx(OUTER_RADIUS, abs=2 * simulation.h)


class TestEndFaceSetting:
    def test_burning_ends_are_refused_for_simulation(self):
        """Shelved under #194, and refused loudly rather than left to fail slowly.

        With burning ends the inhibited set is the outer wall alone -- an open
        cylinder with nothing capping it -- so "inside the casing" is undefined
        past its rim and the grain never burns out. It ran to the 12 s cap with
        1176 cells still counted as propellant, all of them outside the wall.

        Refusing at construction turns that into an immediate, explained error
        instead of a simulation that quietly never terminates.
        """
        with pytest.raises(ValueError, match="#194"):
            MeshGrain(make_bates(sections=SECTIONS), ends="burning")

    def test_the_labels_are_carried_on_the_geometry(self):
        grain = MeshGrain(make_bates(sections=SECTIONS), ends="inhibited")
        assert grain.labels["ends"] == "inhibited"
        assert grain.labels["n_burning"] > 0
        assert grain.labels["n_inhibited"] > grain.labels["n_burning"]
