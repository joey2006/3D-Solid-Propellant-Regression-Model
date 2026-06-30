"""Tests for SDF reinitialization (issue #6).

Run with:  pytest tests/test_reinitialization.py

The reinitialization PDE propagates the distance correction outward from the
interface at ~0.5h per iteration (the pseudo-timestep), so it restores
``|grad phi| ~ 1`` quickly *near the surface* -- which is all the simulation
needs -- and the critical guarantee is that the zero contour does not move.
"""

import time

import pytest
import torch

from srm_burnback.levelset.reinitialize import reinitialize, _smoothed_sign
from srm_burnback.levelset.godunov import godunov_gradient_magnitude


def _circle(res=201, half=2.0, R=1.0, dtype=torch.float64):
    axis = torch.linspace(-half, half, res, dtype=dtype)
    X, Y = torch.meshgrid(axis, axis, indexing="ij")
    r = torch.sqrt(X**2 + Y**2)
    h = (axis[1] - axis[0]).item()
    return (R - r), axis, X, Y, r, h


def _radius_along_x(phi, axis):
    mid = phi.shape[0] // 2
    row = phi[mid, mid:]
    xs = axis[mid:]
    i = int(((row[:-1] > 0) & (row[1:] <= 0)).nonzero()[0])
    f = row[i] / (row[i] - row[i + 1])
    return (xs[i] + f * (xs[i + 1] - xs[i])).item()


# --- #50: smoothed sign ------------------------------------------------------


def test_smoothed_sign_in_range():
    phi, _, _, _, _, h = _circle()
    S = _smoothed_sign(phi, h)
    assert S.min().item() >= -1.0 and S.max().item() <= 1.0


def test_smoothed_sign_zero_at_interface():
    # S(0) = 0 exactly -- this is what pins the zero contour in place.
    S = _smoothed_sign(torch.tensor([0.0]), 0.01)
    assert S.item() == pytest.approx(0.0)


def test_smoothed_sign_saturates_far_away():
    # |phi| >> h  ->  S ~ +/-1.
    h = 0.01
    far = torch.tensor([1.0, -1.0])
    S = _smoothed_sign(far, h)
    assert S[0].item() == pytest.approx(1.0, abs=1e-3)
    assert S[1].item() == pytest.approx(-1.0, abs=1e-3)


# --- #51 / #53: gradient restoration and contour preservation ----------------


def test_restores_gradient_near_surface():
    # Deliberately distort a valid SDF (x2 -> |grad| = 2), reinitialize, and
    # check |grad phi| ~ 1 is restored near the surface (within ~5 cells).
    phi, _, _, _, r, h = _circle(R=1.0)
    distorted = phi * 2.0
    fixed = reinitialize(distorted, h, n_iterations=20)
    grad = godunov_gradient_magnitude(fixed, h, F_sign=torch.sign(phi))
    near = (r > 0.3) & (r < 1.7) & ((r - 1.0).abs() >= 2 * h) & ((r - 1.0).abs() < 5 * h)
    assert (grad[near] - 1.0).abs().max().item() < 0.10  # within 10%


def test_zero_contour_does_not_move():
    # The showstopper guarantee: reinitialization must not move the burn front.
    phi, axis, _, _, _, h = _circle(R=1.0)
    distorted = phi * 2.0
    r_before = _radius_along_x(distorted, axis)
    fixed = reinitialize(distorted, h, n_iterations=20)
    r_after = _radius_along_x(fixed, axis)
    assert abs(r_after - r_before) < 0.1 * h  # < 0.1 grid cell


def test_more_iterations_reduce_error():
    # Convergence: the correction front advances with iterations, so a band a
    # bit further out is closer to |grad| = 1 after more iterations.
    phi, _, _, _, r, h = _circle(R=1.0)
    distorted = phi * 2.0
    band = (r > 0.3) & (r < 1.7) & ((r - 1.0).abs() >= 5 * h) & ((r - 1.0).abs() < 10 * h)

    def err(n):
        g = godunov_gradient_magnitude(
            reinitialize(distorted, h, n_iterations=n), h, F_sign=torch.sign(phi)
        )
        return (g[band] - 1.0).abs().max().item()

    assert err(40) < err(5)


# --- #52: narrow band --------------------------------------------------------


def test_narrow_band_leaves_outside_unchanged():
    phi, _, _, _, _, h = _circle(R=1.0)
    distorted = phi * 2.0
    bw = 6.0
    fixed = reinitialize(distorted, h, n_iterations=5, band_width=bw)
    band = distorted.abs() < bw * h
    # Every cell outside the band is bit-for-bit untouched.
    assert torch.equal(fixed[~band], distorted[~band])


def test_narrow_band_matches_full_near_interface():
    # Inside the band, band-mode and full-grid reinitialization agree.
    phi, _, _, _, _, h = _circle(R=1.0)
    distorted = phi * 2.0
    bw = 6.0
    full = reinitialize(distorted, h, n_iterations=5)
    band_mode = reinitialize(distorted, h, n_iterations=5, band_width=bw)
    band = distorted.abs() < (bw - 1) * h  # interior of the band, away from its edge
    assert torch.allclose(full[band], band_mode[band], atol=1e-9)


def test_narrow_band_is_faster_on_large_grid():
    # A localized interface on a 400x400 grid: band mode crops the work to the
    # interface bounding box and should be clearly faster than full-grid.
    res, half, R = 400, 2.0, 0.3  # small bore -> work concentrated near center
    axis = torch.linspace(-half, half, res, dtype=torch.float32)
    X, Y = torch.meshgrid(axis, axis, indexing="ij")
    h = (axis[1] - axis[0]).item()
    distorted = (R - torch.sqrt(X**2 + Y**2)) * 2.0

    def timed(fn, repeats=3):
        fn()  # warm up
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        return (time.perf_counter() - t0) / repeats

    t_full = timed(lambda: reinitialize(distorted, h, n_iterations=10))
    t_band = timed(lambda: reinitialize(distorted, h, n_iterations=10, band_width=6))
    assert t_band < 0.9 * t_full


# --- dimension-agnostic: 3D --------------------------------------------------


def test_3d_reinitialization():
    res, half, R = 61, 2.0, 1.0
    axis = torch.linspace(-half, half, res, dtype=torch.float64)
    X, Y, Z = torch.meshgrid(axis, axis, axis, indexing="ij")
    r = torch.sqrt(X**2 + Y**2 + Z**2)
    h = (axis[1] - axis[0]).item()
    distorted = (R - r) * 2.0
    fixed = reinitialize(distorted, h, n_iterations=20)
    grad = godunov_gradient_magnitude(fixed, h, F_sign=torch.sign(R - r))
    near = (r > 0.4) & (r < 1.6) & ((r - 1.0).abs() >= 2 * h) & ((r - 1.0).abs() < 5 * h)
    assert (grad[near] - 1.0).abs().max().item() < 0.12  # coarse 3D grid
    assert torch.isfinite(fixed).all()
