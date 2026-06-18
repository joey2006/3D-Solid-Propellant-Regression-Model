"""Tests for the Godunov upwind scheme (issue #4).

Run with:  pytest tests/test_godunov.py

The headline guarantee: on an exact signed-distance field, where ``|grad phi|``
is 1 by definition, the Godunov gradient magnitude returns ~1 in the interior.
"""

import math

import pytest
import torch

from srm_burnback.levelset.godunov import (
    _finite_differences,
    godunov_gradient_magnitude,
)
from srm_burnback.geometry.BATESGrain import BATESGrain


def _circle_sdf(resolution: int = 201, half: float = 2.0, radius: float = 1.0):
    """Exact circle SDF phi = radius - r on a square grid. Returns (phi, h)."""
    axis = torch.linspace(-half, half, resolution)
    X, Y = torch.meshgrid(axis, axis, indexing="ij")
    r = torch.sqrt(X**2 + Y**2)
    phi = radius - r
    h = (axis[1] - axis[0]).item()
    return phi, X, Y, h


# --- #40: finite differences on simple polynomials ---------------------------


def test_finite_difference_linear_is_constant():
    # phi = 2*i along axis 0 -> forward and backward differences are both 2/h
    # in the interior (a linear field has constant slope).
    h = 0.5
    n = 10
    phi = (2.0 * torch.arange(n, dtype=torch.float32)).reshape(n, 1).expand(n, 4).clone()
    d_fwd, d_bwd = _finite_differences(phi, axis=0, h=h)
    expected = 2.0 / h
    assert torch.allclose(d_fwd[1:-1], torch.full_like(d_fwd[1:-1], expected))
    assert torch.allclose(d_bwd[1:-1], torch.full_like(d_bwd[1:-1], expected))


def test_finite_difference_quadratic_slope():
    # phi = i^2: forward diff at i is ( (i+1)^2 - i^2 )/h = (2i+1)/h.
    h = 1.0
    n = 8
    i = torch.arange(n, dtype=torch.float32)
    phi = (i**2).reshape(n, 1)
    d_fwd, d_bwd = _finite_differences(phi, axis=0, h=h)
    # Interior forward differences match the analytic (2i+1).
    interior = slice(0, n - 1)
    assert torch.allclose(d_fwd[interior, 0], (2 * i + 1)[interior])


def test_finite_difference_each_axis():
    # The building block must work for axis 0, 1, and 2 (3D).
    phi = torch.rand(5, 6, 7)
    for axis in range(3):
        d_fwd, d_bwd = _finite_differences(phi, axis=axis, h=0.1)
        assert d_fwd.shape == phi.shape
        assert d_bwd.shape == phi.shape


# --- #41 / #44: constant gradient phi = x ------------------------------------


def test_constant_gradient_is_one():
    # phi = x  =>  |grad phi| = 1 everywhere. Check the interior (one-sided
    # ghost cells at the very edge are expected and live outside the domain).
    n, half = 101, 2.0
    axis = torch.linspace(-half, half, n)
    X, _ = torch.meshgrid(axis, axis, indexing="ij")
    h = (axis[1] - axis[0]).item()
    grad = godunov_gradient_magnitude(X.clone(), h, F_sign=1.0)
    interior = grad[1:-1, 1:-1]
    assert (interior - 1.0).abs().max().item() < 1e-4


# --- #41 / #44: circular SDF |grad phi| = 1 ----------------------------------


def test_circle_sdf_gradient_is_one():
    phi, X, Y, h = _circle_sdf()
    grad = godunov_gradient_magnitude(phi, h, F_sign=1.0)
    # On the smooth annulus -- away from the center cone-tip singularity (the
    # medial axis, where a circle's distance field has a genuine kink) and away
    # from the zero contour (sub-grid effects) -- |grad phi| ~ 1. First-order
    # Godunov on a *curved* surface leaves an O(h * curvature) error, so the
    # bound here is 2%, not the 1e-4 of the planar phi = x test above.
    r = torch.sqrt(X**2 + Y**2)
    mask = (r > 0.5) & (r < 1.6) & ((r - 1.0).abs() > 6 * h)
    err = (grad[mask] - 1.0).abs()
    assert err.max().item() < 0.02


def test_no_nan_or_inf_anywhere():
    phi, _, _, h = _circle_sdf()
    grad = godunov_gradient_magnitude(phi, h, F_sign=1.0)
    assert torch.isfinite(grad).all()


# --- #44: symmetry around the bore -------------------------------------------


def test_symmetry_around_bore():
    # On a centered circular SDF the gradient magnitude must be identical at the
    # four points reflected across the axes (same radius -> same value).
    phi, _, _, h = _circle_sdf(resolution=201)
    grad = godunov_gradient_magnitude(phi, h, F_sign=1.0)
    c = 201 // 2
    d = 40  # a point well inside the interior
    vals = [grad[c + d, c], grad[c - d, c], grad[c, c + d], grad[c, c - d]]
    for v in vals[1:]:
        assert v.item() == pytest.approx(vals[0].item(), abs=1e-5)


# --- #43: 2D and 3D share one function ---------------------------------------


def test_3d_sphere_gradient_is_one():
    res, half, radius = 81, 2.0, 1.0
    axis = torch.linspace(-half, half, res)
    X, Y, Z = torch.meshgrid(axis, axis, axis, indexing="ij")
    r = torch.sqrt(X**2 + Y**2 + Z**2)
    phi = radius - r
    h = (axis[1] - axis[0]).item()
    grad = godunov_gradient_magnitude(phi, h, F_sign=1.0)
    # Smooth shell away from the center singularity and the surface band; a 3D
    # grid + spherical curvature (two principal curvatures) -> allow 3%.
    mask = (r > 0.7) & (r < half - 0.3) & ((r - radius).abs() > 8 * h)
    assert (grad[mask] - 1.0).abs().max().item() < 0.03


def test_2d_matches_3d_slice():
    # A 3D field that is constant along z must give the same gradient on each
    # z-slice as the equivalent 2D field (the z term contributes nothing).
    n, half = 81, 2.0
    axis = torch.linspace(-half, half, n)
    X2, Y2 = torch.meshgrid(axis, axis, indexing="ij")
    h = (axis[1] - axis[0]).item()
    phi2 = 1.0 - torch.sqrt(X2**2 + Y2**2)
    grad2 = godunov_gradient_magnitude(phi2, h, F_sign=1.0)

    phi3 = phi2.unsqueeze(0).expand(5, n, n).clone()  # extrude along z
    grad3 = godunov_gradient_magnitude(phi3, h, F_sign=1.0)
    # Interior z-slice (away from the z ghost cells) equals the 2D result.
    assert torch.allclose(grad3[2], grad2, atol=1e-5)


# --- F_sign direction selection ----------------------------------------------


def test_f_sign_tensor_is_accepted():
    # A per-point F_sign (the erosive-burning case) must run and stay finite.
    phi, X, _, h = _circle_sdf(resolution=81)
    f_sign = torch.where(X > 0, 1.0, -1.0)
    grad = godunov_gradient_magnitude(phi, h, F_sign=f_sign)
    assert grad.shape == phi.shape
    assert torch.isfinite(grad).all()


# --- integration with a real BATES grain SDF ---------------------------------


def test_bates_initial_field_gradient():
    # The initial BATES field is an exact distance field in the annulus, so the
    # interior (excluding the center and casing edges) has |grad phi| ~ 1.
    grain = BATESGrain(inner_radius=0.5, outer_radius=2.0)
    phi, coords, h = grain.initialize_grid(201)
    grad = godunov_gradient_magnitude(phi, h, F_sign=1.0)
    X, Y = coords
    r = torch.sqrt(X**2 + Y**2)
    # Smooth annulus between the bore surface and the casing, excluding both the
    # surface band and the near-center high-curvature region.
    mask = (
        (r > grain.inner_radius + 6 * h)
        & ((r - grain.inner_radius).abs() > 6 * h)
        & (r < grain.outer_radius - 6 * h)
    )
    assert (grad[mask] - 1.0).abs().max().item() < 0.02
