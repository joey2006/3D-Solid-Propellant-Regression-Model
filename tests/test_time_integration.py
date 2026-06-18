"""Tests for TVD-RK3 time integration and the CFL condition (issue #5).

Run with:  pytest tests/test_time_integration.py

A note on the convergence test (#49)
-------------------------------------
The issue suggests measuring 3rd-order accuracy on a *uniform*-burn BATES grain
against the analytical radius. But uniform burn is a degenerate case for the
time integrator: with a constant speed on a signed-distance field, ``L = -F`` is
constant in time, so the exact update is a pure translation that TVD-RK3 (indeed
even forward Euler) reproduces to machine precision at *any* timestep -- there is
no temporal truncation error to measure (``test_uniform_burn_is_exact`` shows
this directly). To actually exercise the 3rd-order behavior we use a *non-uniform*
(erosive-like) speed field, which makes the motion non-trivial and the temporal
error non-zero, and verify the order by self-convergence against a fine-dt
reference. That is what ``test_rk3_third_order`` does, and it cleanly shows a
slope of ~3.
"""

import math

import numpy as np
import pytest
import torch

from srm_burnback.levelset.time_integration import (
    _rhs,
    compute_cfl_timestep,
    tvd_rk3_step,
)


def _circle(res=201, half=2.0, R0=1.0, dtype=torch.float64):
    """phi = R0 - r on a square grid (project convention: bore positive)."""
    axis = torch.linspace(-half, half, res, dtype=dtype)
    X, Y = torch.meshgrid(axis, axis, indexing="ij")
    r = torch.sqrt(X**2 + Y**2)
    h = (axis[1] - axis[0]).item()
    return (R0 - r), X, Y, h


# --- #45: _rhs --------------------------------------------------------------


def test_rhs_is_minus_F_near_surface():
    # On an exact distance field |grad phi| = 1, so L = -F * 1 = -F. Check near
    # the surface (away from the center singularity).
    phi, X, Y, h = _circle(R0=1.0)
    F = 0.005
    rhs = _rhs(phi, F, h)
    r = torch.sqrt(X**2 + Y**2)
    band = (r > 0.6) & (r < 1.5) & ((r - 1.0).abs() > 6 * h)
    assert torch.allclose(rhs[band], torch.full_like(rhs[band], -F), atol=2e-4)


def test_rhs_accepts_tensor_speed():
    # Erosive case: F varies in space. RHS must stay finite and shaped like phi.
    phi, X, _, h = _circle(res=81)
    F = 0.005 * (1.0 + 0.5 * (X / 2.0))
    rhs = _rhs(phi, F, h)
    assert rhs.shape == phi.shape
    assert torch.isfinite(rhs).all()


# --- #46: compute_cfl_timestep ----------------------------------------------


def test_cfl_exact_value():
    # From the issue: F=0.005, h=0.001, cfl=0.5 -> 0.5 * 0.001 / 0.005 = 0.1.
    assert compute_cfl_timestep(0.005, 0.001, 0.5) == pytest.approx(0.1)


def test_cfl_tensor_uses_global_max():
    # With a tensor speed the global max|F| sets the step.
    F = torch.tensor([0.001, 0.004, 0.002])
    assert compute_cfl_timestep(F, 0.001, 0.5) == pytest.approx(0.5 * 0.001 / 0.004)


def test_cfl_positive_and_finite():
    dt = compute_cfl_timestep(torch.tensor([-0.003, 0.002]), 0.01, 0.5)
    assert dt > 0 and math.isfinite(dt)


# --- #47 / #48: tvd_rk3_step basic behavior & auto-dt -----------------------


def test_step_preserves_shape_and_returns_dt():
    phi, _, _, h = _circle(res=101, dtype=torch.float32)
    phi_new, dt = tvd_rk3_step(phi, -0.005, h)
    assert phi_new.shape == phi.shape
    assert isinstance(dt, float) and dt > 0


def test_auto_dt_matches_cfl():
    phi, _, _, h = _circle(res=101)
    F = -0.005
    _, dt = tvd_rk3_step(phi, F, h, cfl_factor=0.5)
    assert dt == pytest.approx(compute_cfl_timestep(F, h, 0.5))


def test_explicit_dt_overrides_auto():
    phi, _, _, h = _circle(res=101)
    _, dt = tvd_rk3_step(phi, -0.005, h, dt=1e-4)
    assert dt == pytest.approx(1e-4)


def test_3d_step_runs():
    # Dimension-agnostic: a 3D field integrates with the same call.
    res, half = 41, 2.0
    axis = torch.linspace(-half, half, res, dtype=torch.float64)
    X, Y, Z = torch.meshgrid(axis, axis, axis, indexing="ij")
    h = (axis[1] - axis[0]).item()
    phi = 1.0 - torch.sqrt(X**2 + Y**2 + Z**2)
    phi_new, dt = tvd_rk3_step(phi, -0.005, h)
    assert phi_new.shape == phi.shape and torch.isfinite(phi_new).all()


# --- uniform burn is exact (the reason #49 needs a non-uniform problem) -------


def _bore_radius(phi, axis):
    """Radius of the zero crossing along +x from the center (linear interp)."""
    mid = phi.shape[0] // 2
    row = phi[mid, mid:]
    xs = axis[mid:]
    i = int(((row[:-1] > 0) & (row[1:] <= 0)).nonzero()[0])
    f = row[i] / (row[i] - row[i + 1])
    return (xs[i] + f * (xs[i + 1] - xs[i])).item()


def test_uniform_burn_is_exact():
    # Uniform outward burn: analytical bore radius R(t) = R0 + b t. Because L is
    # constant in time, the integrator is exact -- the recovered radius matches
    # to ~machine precision and is independent of dt.
    res, half, R0, b = 401, 2.0, 0.8, 0.5
    axis = torch.linspace(-half, half, res, dtype=torch.float64)
    X, Y = torch.meshgrid(axis, axis, indexing="ij")
    h = (axis[1] - axis[0]).item()
    phi0 = 0.8 - torch.sqrt(X**2 + Y**2)  # R0 = 0.8
    t_end = 0.4
    F = -b  # negative => outward expansion in the phi = R0 - r convention
    for n in (25, 200):
        phi = phi0.clone()
        dt = t_end / n
        for _ in range(n):
            phi, _ = tvd_rk3_step(phi, F, h, dt=dt)
        R = _bore_radius(phi, axis)
        assert R == pytest.approx(R0 + b * t_end, abs=1e-6)


# --- #49: 3rd-order temporal convergence on a non-uniform speed --------------


def _evolve(phi0, F, h, n_steps, t_end, stepper):
    phi = phi0.clone()
    dt = t_end / n_steps
    for _ in range(n_steps):
        phi = stepper(phi, F, h, dt)
    return phi


def _rk3(phi, F, h, dt):
    out, _ = tvd_rk3_step(phi, F, h, dt=dt)
    return out


def _euler(phi, F, h, dt):
    return phi + dt * _rhs(phi, F, h)


T_END = 0.3


@pytest.fixture(scope="module")
def convergence_problem():
    """Non-uniform (erosive-like) expansion problem + a fine-dt RK3 reference.

    Module-scoped so the expensive reference solve runs only once.
    """
    res, half, R0 = 161, 2.0, 0.8
    axis = torch.linspace(-half, half, res, dtype=torch.float64)
    X, Y = torch.meshgrid(axis, axis, indexing="ij")
    r = torch.sqrt(X**2 + Y**2)
    h = (axis[1] - axis[0]).item()
    # Erosive-like outward speed: faster toward +x. Constant in time (autonomous)
    # but non-uniform in space, so the motion is non-trivial.
    F = -0.5 * (1.0 + 0.6 * (X / half))
    phi0 = (R0 - r).clone()
    band = (phi0.abs() < 0.3)  # measure error near the interface
    ref = _evolve(phi0, F, h, 1500, T_END, _rk3)
    return phi0, F, h, band, ref


def test_rk3_third_order(convergence_problem):
    phi0, F, h, band, ref = convergence_problem

    dts, errs = [], []
    for n in (25, 50, 100, 200):
        err = (_evolve(phi0, F, h, n, T_END, _rk3) - ref)[band].abs().max().item()
        dts.append(T_END / n)
        errs.append(err)

    slope = np.polyfit(np.log(dts), np.log(errs), 1)[0]
    assert 2.5 <= slope <= 3.5, f"expected ~3rd order, got slope {slope:.2f}"


def test_rk3_more_accurate_than_euler(convergence_problem):
    # At the same dt on the non-uniform problem, RK3 (3rd order) must beat
    # forward Euler (1st order). (On uniform burn both are exact, so this needs
    # the non-uniform speed.)
    phi0, F, h, band, ref = convergence_problem
    n = 50
    err_rk3 = (_evolve(phi0, F, h, n, T_END, _rk3) - ref)[band].abs().max().item()
    err_euler = (_evolve(phi0, F, h, n, T_END, _euler) - ref)[band].abs().max().item()
    assert err_rk3 < err_euler
