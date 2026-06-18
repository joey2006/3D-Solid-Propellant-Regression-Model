"""Unit tests for the brute-force signed-distance reconstruction (issue #3).

Run with:  pytest tests/test_sdf.py

The core guarantee: given only the *sign* of a field, ``signed_distance_from_sign``
recovers true Euclidean distances whose zero contour matches the input and whose
gradient magnitude is ~1 everywhere.
"""

import math

import pytest
import torch

from srm_burnback.sdf import signed_distance_from_sign
from srm_burnback.geometry.BATESGrain import BATESGrain


def circle_grid(resolution: int = 101, half: float = 2.0, radius: float = 1.0):
    """An exact circle SDF (positive inside the disc) on a square grid.

    Returns ``(phi_exact, coords, h)``. Sign convention matches the project:
    here "inside the disc" plays the role of the open void (positive).
    """
    axis = torch.linspace(-half, half, resolution)
    X, Y = torch.meshgrid(axis, axis, indexing="ij")
    r = torch.sqrt(X**2 + Y**2)
    phi = radius - r  # positive inside the circle, zero on it, negative outside
    h = (axis[1] - axis[0]).item()
    return phi, (X, Y), h


def grad_magnitude(phi: torch.Tensor, h: float) -> torch.Tensor:
    """Central-difference ``|grad phi|`` on the interior of the grid."""
    grads = torch.gradient(phi, spacing=h)
    return torch.sqrt(sum(g**2 for g in grads))


# --- recovery from corrupted distances (the issue's headline test) -----------


def test_recovers_distances_from_corrupted_field():
    # Start from an exact circle SDF, scramble the magnitudes but keep the sign,
    # then rebuild. Distances must come back within one grid spacing.
    phi_exact, coords, h = circle_grid()
    torch.manual_seed(0)
    corrupted = torch.sign(phi_exact) * (1.0 + torch.rand_like(phi_exact) * 5.0)

    phi_rebuilt = signed_distance_from_sign(corrupted, coords)

    err = (phi_rebuilt - phi_exact).abs().max().item()
    assert err < h


def test_sign_field_only_input_recovers_distance():
    # The geometry-import path: feed a raw +/-1 sign field (no magnitude info at
    # all) and still recover the true distance field.
    phi_exact, coords, h = circle_grid()
    sign_only = torch.sign(phi_exact)

    phi_rebuilt = signed_distance_from_sign(sign_only, coords)

    err = (phi_rebuilt - phi_exact).abs().max().item()
    assert err < h


# --- defining property: |grad phi| ~= 1 --------------------------------------


def test_gradient_magnitude_is_unity():
    phi_exact, coords, h = circle_grid()
    phi_rebuilt = signed_distance_from_sign(torch.sign(phi_exact), coords)
    g = grad_magnitude(phi_rebuilt, h)
    # Away from the very center (where the distance function has a kink) the
    # gradient magnitude should be close to 1.
    interior = g[2:-2, 2:-2]
    assert (interior - 1.0).abs().mean().item() < 0.05


# --- the zero contour does not move ------------------------------------------


def test_zero_contour_stays_put():
    # Sign pattern before and after must be identical: the surface stays where it
    # was, so no grid point flips side.
    phi_exact, coords, _ = circle_grid()
    phi_rebuilt = signed_distance_from_sign(torch.sign(phi_exact), coords)
    assert torch.equal(torch.sign(phi_rebuilt), torch.sign(phi_exact))


def test_no_nan_or_inf():
    phi_exact, coords, _ = circle_grid()
    phi_rebuilt = signed_distance_from_sign(torch.sign(phi_exact), coords)
    assert torch.isfinite(phi_rebuilt).all()


# --- sub-grid interface accuracy ---------------------------------------------


def test_interface_interpolation_is_subgrid():
    # Put the true circle radius deliberately between grid lines. The recovered
    # distance at the center must equal that radius to better than one cell --
    # only possible if the interface was interpolated, not snapped to a node.
    resolution, half, radius = 101, 2.0, 0.93
    phi_exact, coords, h = circle_grid(resolution, half, radius)
    phi_rebuilt = signed_distance_from_sign(torch.sign(phi_exact), coords)
    center = resolution // 2
    # Distance from the center to the circle is exactly the radius.
    assert phi_rebuilt[center, center].item() == pytest.approx(radius, abs=h)


# --- boundary cells are clean ------------------------------------------------


def test_boundary_cells_finite_and_signed():
    phi_exact, coords, _ = circle_grid()
    phi_rebuilt = signed_distance_from_sign(torch.sign(phi_exact), coords)
    edges = torch.cat(
        [phi_rebuilt[0], phi_rebuilt[-1], phi_rebuilt[:, 0], phi_rebuilt[:, -1]]
    )
    assert torch.isfinite(edges).all()
    # All edge points are outside the unit circle -> must stay negative.
    assert (edges < 0).all()


# --- 3D: same routine, one more axis -----------------------------------------


def test_3d_sphere_recovery():
    # Dimension-agnostic: a sphere sign field rebuilds to a sphere SDF.
    res, half, radius = 41, 2.0, 1.0
    axis = torch.linspace(-half, half, res)
    X, Y, Z = torch.meshgrid(axis, axis, axis, indexing="ij")
    r = torch.sqrt(X**2 + Y**2 + Z**2)
    phi_exact = radius - r
    h = (axis[1] - axis[0]).item()

    phi_rebuilt = signed_distance_from_sign(torch.sign(phi_exact), (X, Y, Z))
    err = (phi_rebuilt - phi_exact).abs().max().item()
    assert err < 2 * h  # coarse 3D grid -> allow a slightly looser bound


# --- integration with a real grain SDF ---------------------------------------


def test_reinitializes_bates_annulus():
    # Feed an actual BATES initial field through the engine; the bore radius (the
    # evolving surface) must be preserved within a cell.
    grain = BATESGrain(inner_radius=0.5, outer_radius=2.0)
    phi, coords, h = grain.initialize_grid(201)
    phi_rebuilt = signed_distance_from_sign(torch.sign(phi), coords)

    # Center sits in the bore: distance to the bore wall == inner_radius.
    center = 201 // 2
    assert phi_rebuilt[center, center].item() == pytest.approx(
        grain.inner_radius, abs=2 * h
    )
    assert torch.equal(torch.sign(phi_rebuilt), torch.sign(phi))
