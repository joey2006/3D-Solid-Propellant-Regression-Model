"""φ from a boundary: sign, distance, and the claims that justified it (#158).

The issue chose the generalized winding number over ray casting, and
closest-point-on-primitive over reconstructing distance from a ±1 sign field.
Both choices were argued from measurements, so the measurements are pinned
here -- if a future change quietly makes the import path grid-bound again,
these fail rather than the burn times drifting.

The sphere is the oracle throughout: its signed distance is ``|p| - r`` in
closed form, so every recovered value has an exact answer to check against.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from srm_burnback.geometry.CoordinateBuilder import build_coords
from srm_burnback.geometry.winding_number import (
    INSIDE_THRESHOLD,
    phi_from_boundary,
    phi_from_mesh,
    winding_number_and_distance,
)
from srm_burnback.sdf.initialization import signed_distance_from_sign

trimesh = pytest.importorskip("trimesh")

DTYPE = torch.float64

# The grid tests are O(cells x triangles) and run on whatever hardware is
# present. On the GPU that is sub-second; on a CPU-only machine the same
# work is ~90x slower, so the grids shrink rather than the assertions
# weakening -- the properties under test (ordering, ratios, |grad phi|) hold
# at any resolution.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID = 64 if DEVICE == "cuda" else 32
BATES_SECTIONS = (32, 64, 128) if DEVICE == "cuda" else (16, 24, 32)


def sphere(subdivisions: int = 4, radius: float = 1.0):
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)


def tensors(mesh):
    return (
        torch.tensor(mesh.vertices, dtype=DTYPE),
        torch.tensor(mesh.faces, dtype=torch.long),
    )


def damaged(mesh, fraction: float = 0.20, seed: int = 0):
    """Delete a fraction of faces at random, leaving an open surface."""
    rng = np.random.default_rng(seed)
    n_faces = len(mesh.faces)
    keep = rng.permutation(n_faces)[: int(n_faces * (1.0 - fraction))]
    holed = mesh.copy()
    holed.update_faces(np.isin(np.arange(n_faces), keep))
    return holed


class TestWindingNumber3D:
    def test_closed_surface_gives_one_inside_and_zero_outside(self):
        vertices, faces = tensors(sphere())
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.5]],
            dtype=DTYPE,
        )
        winding, _ = winding_number_and_distance(points, vertices, faces)
        assert winding[:2].numpy() == pytest.approx(1.0, abs=1e-9)
        assert winding[2:].numpy() == pytest.approx(0.0, abs=1e-9)

    def test_signs_are_exactly_right_on_a_clean_mesh(self):
        vertices, faces = tensors(sphere())
        points = torch.tensor(
            np.random.default_rng(1).uniform(-2, 2, (4000, 3)), dtype=DTYPE
        )
        winding, _ = winding_number_and_distance(points, vertices, faces)
        truth = (points.norm(dim=1) < 1.0).numpy()
        assert ((winding > INSIDE_THRESHOLD).numpy() == truth).all()

    def test_damaged_mesh_degrades_without_flipping(self):
        """The whole reason this is a winding number and not a ray cast.

        20% of faces deleted. ``w`` slides off 1.0 but stays clear of the 0.5
        threshold, so signs survive. Ray casting flips entire regions under the
        same damage, because a ray through a gap inverts the crossing parity of
        everything behind it.
        """
        vertices, faces = tensors(damaged(sphere()))
        points = torch.tensor(
            np.random.default_rng(1).uniform(-2, 2, (4000, 3)), dtype=DTYPE
        )
        winding, _ = winding_number_and_distance(points, vertices, faces)
        truth = points.norm(dim=1).numpy() < 1.0

        accuracy = ((winding > INSIDE_THRESHOLD).numpy() == truth).mean()
        assert accuracy > 0.999, accuracy

        # Degraded but still separated: the inside mean drops well below 1.0,
        # which is the graceful part, yet stays clear of the threshold.
        inside_mean = winding.numpy()[truth].mean()
        outside_mean = winding.numpy()[~truth].mean()
        assert 0.6 < inside_mean < 0.95
        assert outside_mean < 0.2
        assert inside_mean - outside_mean > 0.5


class TestDistance3D:
    def test_distance_matches_the_sphere_in_closed_form(self):
        vertices, faces = tensors(sphere(subdivisions=5))
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=DTYPE
        )
        _, distance = winding_number_and_distance(points, vertices, faces)
        expected = np.abs(points.norm(dim=1).numpy() - 1.0)
        # A tessellated sphere is inscribed, so recovered distances sit a hair
        # inside the true surface; the gap closes with subdivision.
        assert distance.numpy() == pytest.approx(expected, abs=2e-3)

    def test_distance_is_correct_for_points_beside_a_single_triangle(self):
        """Exercises the seven-region case analysis directly.

        A point above the face interior, beyond an edge, and beyond a vertex
        take three different branches, and a bug in any one of them would still
        leave the other two correct -- so they need separating.
        """
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        points = torch.tensor(
            [
                [0.25, 0.25, 0.5],   # over the face interior
                [-1.0, -1.0, 0.0],   # beyond vertex A, in-plane
                [2.0, 0.0, 0.0],     # beyond vertex B, along the edge line
                [0.5, 0.5, 0.0],     # exactly on edge BC
            ],
            dtype=DTYPE,
        )
        _, distance = winding_number_and_distance(points, vertices, faces)
        assert distance.numpy() == pytest.approx(
            [0.5, np.sqrt(2.0), 1.0, 0.0], abs=1e-12
        )


class TestBeatsTheSignFieldReconstruction:
    """The measurement that justified expanding this issue's scope.

    Handing a ±1 sign field to ``signed_distance_from_sign`` makes its crossing
    interpolation ``lo / (lo - hi)`` evaluate to exactly 0.5, so every surface
    crossing lands at a cell-edge midpoint and all sub-grid information is lost
    before the distance code runs. Taking distance from the primitives instead
    should be far better -- and, critically, better in a way that does not
    depend on ``h``.
    """

    @pytest.fixture(scope="class")
    def field(self):
        coords, h = build_coords(GRID, 1.5, length=3.0, device=DEVICE)
        coords = tuple(c.to(DTYPE) for c in coords)
        phi = phi_from_mesh(coords, sphere(subdivisions=4), dtype=DTYPE)
        exact = torch.sqrt(sum(c**2 for c in coords)) - 1.0
        band = exact.abs() < 3 * h
        return coords, h, phi, exact, band

    def test_primitive_distance_is_well_under_a_tenth_of_a_cell(self, field):
        _, h, phi, exact, band = field
        error = (phi - exact).abs()[band]
        assert error.mean().item() / h < 0.1
        assert error.max().item() / h < 0.1

    def test_it_beats_reconstruction_from_a_sign_field(self, field):
        coords, h, phi, exact, band = field
        sign_field = torch.where(phi < 0, -1.0, 1.0).to(phi.dtype)
        reconstructed = signed_distance_from_sign(sign_field, coords)

        primitive_error = (phi - exact).abs()[band].mean().item() / h
        reconstructed_error = (reconstructed - exact).abs()[band].mean().item() / h

        # The midpoint-placement penalty is a factor of several, not a few
        # percent. Measured ~0.015 h against ~0.17 h.
        assert reconstructed_error > 0.1
        assert primitive_error < reconstructed_error / 5.0

    def test_error_is_bounded_by_tessellation_not_by_grid_spacing(self, field):
        """The claim that makes the primitive path worth its cost.

        Holding ``h`` fixed and refining the mesh must keep improving the
        answer. If the error were grid-bound this would flatten out at some
        multiple of ``h`` and refining would buy nothing.
        """
        coords, h, _, exact, band = field
        errors = []
        for subdivisions in (2, 3, 4):
            phi = phi_from_mesh(coords, sphere(subdivisions), dtype=DTYPE)
            errors.append((phi - exact).abs()[band].mean().item() / h)

        assert all(b < a for a, b in zip(errors, errors[1:])), errors
        # Each subdivision quarters the facet size, and the error should track
        # it rather than stalling: 320 -> 5120 triangles measured 0.238 h ->
        # 0.015 h, better than an order of magnitude at unchanged h.
        assert errors[0] / errors[-1] > 10.0

    def test_gradient_magnitude_is_one(self, field):
        """``|∇φ| ≈ 1`` -- the defining property of a signed distance field.

        The Godunov scheme, the CFL timestep and reinitialization are all built
        assuming it. This is the #175 headline check, verified here on real
        mesh-derived φ rather than an analytically generated field.
        """
        _, h, phi, _, band = field
        gradients = torch.gradient(phi, spacing=h)
        magnitude = torch.sqrt(sum(g**2 for g in gradients))[band]
        assert magnitude.mean().item() == pytest.approx(1.0, abs=0.01)
        assert magnitude.std().item() < 0.05


class TestBatesRoundTrip:
    """An imported BATES mesh against ``BATESGrain``, the analytical oracle."""

    @staticmethod
    def _setup(sections: int):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
        from make_test_meshes import (
            INNER_RADIUS,
            LENGTH,
            OUTER_RADIUS,
            make_bates,
        )
        from srm_burnback.geometry.BATESGrain import BATESGrain

        coords, h = build_coords(GRID, 0.07, length=0.14, device=DEVICE)
        coords = tuple(c.to(DTYPE) for c in coords)
        phi = phi_from_mesh(coords, make_bates(sections=sections), dtype=DTYPE)
        exact = BATESGrain(INNER_RADIUS, OUTER_RADIUS, LENGTH).signed_distance(coords)

        # Compare only where the two definitions agree. BATESGrain models the
        # bore alone -- the casing is a separate static field -- so outside the
        # outer wall it still reports "propellant" while the closed mesh
        # correctly reports void. The corner where the bore meets an end face
        # is excluded too: the analytic max() of two surfaces rounds it off,
        # which is an artefact of the oracle, not of the import.
        X, Y, Z = coords
        r = torch.sqrt(X**2 + Y**2)
        interior = (r < OUTER_RADIUS - 2 * h) & (Z.abs() < LENGTH / 2 - 2 * h)
        return h, phi, exact, interior

    def test_signs_agree_with_the_oracle_everywhere_inside_the_casing(self):
        _, phi, exact, interior = self._setup(BATES_SECTIONS[-1])
        assert ((phi > 0) == (exact > 0))[interior].all()

    def test_distance_agrees_within_tessellation_tolerance(self):
        h, phi, exact, interior = self._setup(BATES_SECTIONS[-1])
        band = interior & (exact.abs() < 3 * h)
        assert (phi - exact).abs()[band].mean().item() / h < 0.01

    def test_finer_tessellation_tracks_the_oracle_more_closely(self):
        errors = []
        for sections in BATES_SECTIONS:
            h, phi, exact, interior = self._setup(sections)
            band = interior & (exact.abs() < 3 * h)
            errors.append((phi - exact).abs()[band].mean().item() / h)
        assert all(b < a for a, b in zip(errors, errors[1:])), errors


class TestTwoDimensions:
    """2D polylines take the same path with a signed-angle kernel."""

    @staticmethod
    def circle(n: int = 256):
        theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        vertices = torch.tensor(
            np.stack([np.cos(theta), np.sin(theta)], axis=1), dtype=DTYPE
        )
        segments = torch.stack(
            [torch.arange(n), (torch.arange(n) + 1) % n], dim=1
        )
        return vertices, segments

    def test_winding_is_one_inside_and_zero_outside(self):
        vertices, segments = self.circle()
        points = torch.tensor([[0.0, 0.0], [0.5, 0.0], [2.0, 0.0]], dtype=DTYPE)
        winding, _ = winding_number_and_distance(points, vertices, segments)
        assert winding.numpy() == pytest.approx([1.0, 1.0, 0.0], abs=1e-9)

    def test_distance_matches_the_circle(self):
        vertices, segments = self.circle(1024)
        points = torch.tensor([[0.0, 0.0], [0.5, 0.0], [2.0, 0.0]], dtype=DTYPE)
        _, distance = winding_number_and_distance(points, vertices, segments)
        assert distance.numpy() == pytest.approx([1.0, 0.5, 1.0], abs=1e-4)

    def test_phi_on_a_2d_grid_matches_the_analytic_circle(self):
        coords, h = build_coords(GRID, 1.5, device=DEVICE)
        coords = tuple(c.to(DTYPE) for c in coords[:2])
        vertices, segments = self.circle(512)
        phi = phi_from_boundary(coords, vertices, segments)

        exact = torch.sqrt(coords[0] ** 2 + coords[1] ** 2) - 1.0
        band = exact.abs() < 3 * h
        assert (phi - exact).abs()[band].mean().item() / h < 0.01


class TestConventionsAndGuards:
    def test_phi_is_negative_in_the_solid_and_positive_in_the_void(self):
        """Matches ``GrainGeometry``: the mesh bounds propellant, which is < 0.

        Getting this backwards would burn the propellant inward from infinity
        and still produce smooth, plausible-looking output.
        """
        coords, _ = build_coords(32, 1.5, length=3.0, device=DEVICE)
        coords = tuple(c.to(DTYPE) for c in coords)
        phi = phi_from_mesh(coords, sphere(subdivisions=3), dtype=DTYPE)
        radius = torch.sqrt(sum(c**2 for c in coords))
        assert (phi[radius < 0.8] < 0).all()
        assert (phi[radius > 1.2] > 0).all()

    def test_blocking_does_not_change_the_answer(self):
        """Chunking is a memory strategy and must be invisible in the result."""
        import srm_burnback.geometry.winding_number as module

        vertices, faces = tensors(sphere(subdivisions=3))
        points = torch.tensor(
            np.random.default_rng(3).uniform(-2, 2, (500, 3)), dtype=DTYPE
        )
        reference = winding_number_and_distance(points, vertices, faces)

        original = module.PAIRS_PER_BLOCK
        try:
            module.PAIRS_PER_BLOCK = 64  # force many small blocks
            chunked = winding_number_and_distance(points, vertices, faces)
        finally:
            module.PAIRS_PER_BLOCK = original

        assert torch.allclose(reference[0], chunked[0], atol=1e-12)
        assert torch.allclose(reference[1], chunked[1], atol=1e-12)

    def test_mismatched_dimensions_are_rejected(self):
        vertices, faces = tensors(sphere(subdivisions=1))
        with pytest.raises(ValueError, match="2D and 3D"):
            winding_number_and_distance(
                torch.zeros(4, 4, dtype=DTYPE), vertices, faces
            )
        with pytest.raises(ValueError, match="but points are"):
            winding_number_and_distance(
                torch.zeros(4, 2, dtype=DTYPE), vertices, faces
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cuda_matches_cpu(self):
        vertices, faces = tensors(sphere(subdivisions=3))
        points = torch.tensor(
            np.random.default_rng(4).uniform(-2, 2, (2000, 3)), dtype=DTYPE
        )
        on_cpu = winding_number_and_distance(points, vertices, faces)
        on_gpu = winding_number_and_distance(
            points.cuda(), vertices.cuda(), faces.cuda()
        )
        assert torch.allclose(on_cpu[0], on_gpu[0].cpu(), atol=1e-9)
        assert torch.allclose(on_cpu[1], on_gpu[1].cpu(), atol=1e-9)


class TestBurningSurfaceOnly:
    """φ must measure distance to the burning surface, not to the object (#176).

    The bug this guards against is silent and severe: measure distance to the
    whole closed mesh and φ turns around at the middle of the web, crosses zero
    again at the outer wall, and the solver consumes the grain from the outside
    in. Burn area roughly doubles. Nothing complains -- ``|∇φ| = 1`` still holds
    perfectly, because the object SDF is a perfectly good SDF of the object.

    ``BATESGrain`` is the oracle again, and it is the right one: it models the
    bore alone, with the casing as a separate static field.
    """

    @staticmethod
    def _bates(ends: str):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
        from make_test_meshes import (
            INNER_RADIUS,
            LENGTH,
            OUTER_RADIUS,
            make_bates,
        )
        from srm_burnback.geometry.BATESGrain import BATESGrain
        from srm_burnback.geometry.winding_number import phi_from_labelled_mesh

        coords, h = build_coords(GRID, 0.07, length=0.14, device=DEVICE)
        coords = tuple(c.to(DTYPE) for c in coords)
        phi, phi_outer, labels = phi_from_labelled_mesh(
            coords, make_bates(sections=BATES_SECTIONS[-1]), ends=ends, dtype=DTYPE
        )
        exact = BATESGrain(INNER_RADIUS, OUTER_RADIUS, LENGTH).signed_distance(coords)
        return coords, h, phi, phi_outer, exact, labels

    def test_phi_never_turns_back_toward_zero_past_the_wall(self):
        coords, h, phi, _, _, _ = self._bates("inhibited")
        X, Y, Z = coords
        radius = torch.sqrt(X**2 + Y**2)

        # Along a radial line at mid-length, outside the bore, phi must keep
        # decreasing. The old object-SDF reversed at the web midpoint.
        row, mid = X.shape[1] // 2, X.shape[2] // 2
        line = phi[X.shape[0] // 2 :, row, mid]
        radii = radius[X.shape[0] // 2 :, row, mid]
        outside_bore = radii > 0.012
        values = line[outside_bore]
        assert (values[1:] < values[:-1]).all(), values

    def test_no_second_zero_crossing_at_the_outer_wall(self):
        _, _, phi, _, _, _ = self._bates("inhibited")
        # Every point outside the bore is propellant-side. If the outer wall
        # were in the burning set, a whole shell of positive phi would appear
        # beyond it.
        assert phi.max() > 0, "the bore itself must still be positive"
        positive_fraction = float((phi > 0).float().mean())
        # Only the bore is open void: a 10 mm bore in a 140 mm cube is small.
        assert positive_fraction < 0.05, positive_fraction

    def test_burning_ends_reproduce_the_analytical_bates(self):
        """With burning ends this *is* ``BATESGrain`` -- the strongest check.

        ``BATESGrain.signed_distance`` is ``max(bore, |z| - L/2)``: bore and
        both ends burning, outer wall absent. Labelling the mesh the same way
        reproduces it to **0.002 h** inside the casing, where the plain object
        SDF was wrong by whole cells.

        Restricted to ``r < outer_radius`` because the two genuinely differ
        outside it, and the oracle is the one being loose there: ``BATESGrain``
        has no outer wall at all, so it reports propellant receding forever,
        while this field measures distance to the nearest real burning face.
        That region is outside the casing and gets clamped before it is ever
        used.
        """
        coords, h, phi, _, exact, labels = self._bates("burning")
        assert labels["ends"] == "burning"

        X, Y, _ = coords
        inside_casing = torch.sqrt(X**2 + Y**2) < 0.05
        error = (phi - exact).abs()[inside_casing]
        assert float(error.mean()) / h < 0.01

    def test_inhibited_ends_do_not_burn_but_the_bore_still_does(self):
        coords, h, phi, _, exact, labels = self._bates("inhibited")
        X, Y, Z = coords
        radius = torch.sqrt(X**2 + Y**2)

        # Past the end faces, in the annulus of propellant, phi stays negative:
        # an inhibited end is not a surface the flame can advance from.
        beyond_end = Z.abs() > 0.062
        annulus = beyond_end & (radius > 0.014) & (radius < 0.05)
        assert (phi[annulus] < 0).all()

        # Directly past the bore opening is a deliberate exception, and it is
        # correct: that is the port continuing out of the grain -- open volume
        # the flame genuinely occupies -- not propellant. It stays positive.
        port = beyond_end & (radius < 0.008)
        assert (phi[port] > 0).all()

        # The bore is unaffected by the end-face choice.
        bore = (radius < 0.008) & (Z.abs() < 0.05)
        assert (phi[bore] > 0).all()

    def test_end_choice_matters_far_from_the_ends_too(self):
        """The end-face setting is not a local touch-up.

        A burning end is the nearest burning surface for everything within its
        own distance of it, so labelling the ends changes φ across a large
        fraction of the web -- measured at ~39% of cells inside the casing.
        That is precisely why it is exposed as a setting rather than guessed:
        getting it wrong is not a rounding error on the end faces, it moves the
        burn front through most of the grain.

        The two agree only deep in the region where the bore is unambiguously
        the closest burning surface.
        """
        coords, _, burning, _, _, _ = self._bates("burning")
        _, _, inhibited, _, _, _ = self._bates("inhibited")

        X, Y, Z = coords
        radius = torch.sqrt(X**2 + Y**2)

        # Near the bore and far from either end, distance-to-bore wins outright
        # under both labellings, so the fields must agree exactly.
        bore_dominates = (radius < 0.02) & (Z.abs() < 0.03)
        assert torch.allclose(
            burning[bore_dominates], inhibited[bore_dominates], atol=1e-12
        )

        # Out in the web they differ substantially.
        inside_casing = radius < 0.05
        differing = ((burning - inhibited).abs() > 1e-9)[inside_casing]
        assert float(differing.float().mean()) > 0.2

    def test_casing_field_is_negative_inside_and_positive_outside(self):
        coords, _, _, phi_outer, _, _ = self._bates("inhibited")
        X, Y, Z = coords
        radius = torch.sqrt(X**2 + Y**2)
        # Ready for the runner's min(phi, -phi_outer) clamp, which needs the
        # same sign convention as GrainGeometry.outer_boundary_distance.
        assert (phi_outer[(radius < 0.04) & (Z.abs() < 0.05)] < 0).all()
        assert (phi_outer[radius > 0.058] > 0).all()

    def test_sign_still_comes_from_the_whole_closed_mesh(self):
        """Restricting *distance* must not restrict the sign computation.

        The winding number needs a closed surface; handed only the bore faces
        it would be integrating an open patch and the inside/outside answer
        would be meaningless. The bore being positive and the web negative is
        what demonstrates the sign came from the full mesh.
        """
        coords, _, phi, _, _, _ = self._bates("inhibited")
        X, Y, Z = coords
        radius = torch.sqrt(X**2 + Y**2)
        mid = Z.abs() < 0.04
        assert (phi[(radius < 0.008) & mid] > 0).all()
        assert (phi[(radius > 0.015) & (radius < 0.045) & mid] < 0).all()
