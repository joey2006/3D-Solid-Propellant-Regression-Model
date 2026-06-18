"""TVD Runge-Kutta 3 time integration for the level-set equation.

The burning surface is advanced by the Hamilton-Jacobi equation

    d(phi)/dt + F * |grad phi| = 0   =>   d(phi)/dt = L(phi) = -F * |grad phi|,

where ``F`` is the local burn rate (front speed along the normal) and
``|grad phi|`` comes from the Godunov upwind scheme. A plain forward-Euler step
accumulates oscillating error over the thousands of steps a burn-back needs;
the 3-stage TVD Runge-Kutta scheme of Shu & Osher (1988) is third-order accurate
in time and total-variation-diminishing, so it keeps the interface sharp without
introducing new wiggles. Each step calls ``L`` three times (three Godunov
evaluations).

The CFL condition ``dt <= cfl_factor * h / max|F|`` keeps the front from skipping
across more than one cell per step, which would make the scheme blow up.

Dimension-agnostic
------------------
Every operation here is elementwise on ``phi`` and the Godunov call loops over
``range(phi.ndim)``, so the identical code integrates a 2D cross-section and a
full 3D grain.

Sign convention
---------------
``L = -F * |grad phi|`` is implemented exactly as the level-set equation is
written. With the project's field convention (phi > 0 in the void, < 0 in the
propellant) the sign of ``F`` selects which way the front travels; the physical
coupling of ``F``'s sign to the burn direction is finalized in the burn-rate
model (#7) and the simulation runner (#9). ``_rhs`` itself just honors whatever
sign of ``F`` it is given, and uses that sign to pick the Godunov upwind side.
"""

from __future__ import annotations

import torch

from .godunov import godunov_gradient_magnitude


def _rhs(phi: torch.Tensor, F: float | torch.Tensor, h: float) -> torch.Tensor:
    """Right-hand side of the level-set PDE: ``L(phi) = -F * |grad phi|``.

    Parameters
    ----------
    phi:
        Level-set field (2D or 3D).
    F:
        Burn rate. Scalar for uniform burn, or a tensor the shape of ``phi`` for
        a spatially varying (e.g. erosive) rate.
    h:
        Grid spacing.

    Returns
    -------
    torch.Tensor
        The time derivative ``d(phi)/dt`` at every grid point, same shape as
        ``phi``. The sign of ``F`` is passed to the Godunov scheme so the
        upwind direction matches the direction the front is moving.
    """
    F_t = torch.as_tensor(F, dtype=phi.dtype, device=phi.device)
    grad = godunov_gradient_magnitude(phi, h, F_sign=torch.sign(F_t))
    return -F_t * grad


def compute_cfl_timestep(
    F: float | torch.Tensor,
    h: float,
    cfl_factor: float = 0.5,
) -> float:
    """Largest stable timestep from the CFL condition.

    ``dt = cfl_factor * h / max(|F|)``. The front must not advance more than one
    cell per step, so ``cfl_factor`` is kept below 1 (0.5 by default).

    Parameters
    ----------
    F:
        Burn rate, scalar or tensor. The global maximum of ``|F|`` is used.
    h:
        Grid spacing.
    cfl_factor:
        Safety factor in ``(0, 1)``.

    Returns
    -------
    float
        A positive, finite timestep.
    """
    max_speed = torch.as_tensor(F).abs().max().item()
    if max_speed <= 0.0:
        raise ValueError("CFL timestep undefined for a zero burn-rate field.")
    return cfl_factor * h / max_speed


def tvd_rk3_step(
    phi: torch.Tensor,
    F: float | torch.Tensor,
    h: float,
    dt: float | None = None,
    cfl_factor: float = 0.5,
) -> tuple[torch.Tensor, float]:
    """Advance ``phi`` one step with TVD-RK3 (Shu & Osher 1988).

    The three stages, with ``L(phi) = -F * |grad phi|``:

        phi1 = phi  + dt * L(phi)
        phi2 = 3/4 phi + 1/4 phi1 + 1/4 dt * L(phi1)
        new  = 1/3 phi + 2/3 phi2 + 2/3 dt * L(phi2)

    Parameters
    ----------
    phi:
        Current level-set field.
    F:
        Burn rate (scalar or tensor).
    h:
        Grid spacing.
    dt:
        Timestep. When ``None`` it is auto-computed from the CFL condition via
        :func:`compute_cfl_timestep` -- the default the simulation runner uses.
        Pass an explicit ``dt`` (e.g. for a convergence study) to override.
    cfl_factor:
        Safety factor for the auto ``dt``; ignored when ``dt`` is given.

    Returns
    -------
    (phi_new, dt_used):
        The advanced field and the timestep actually taken, so the caller can
        accumulate simulated time.
    """
    if dt is None:
        dt = compute_cfl_timestep(F, h, cfl_factor)

    phi1 = phi + dt * _rhs(phi, F, h)
    phi2 = 0.75 * phi + 0.25 * phi1 + 0.25 * dt * _rhs(phi1, F, h)
    phi_new = (1.0 / 3.0) * phi + (2.0 / 3.0) * phi2 + (2.0 / 3.0) * dt * _rhs(phi2, F, h)
    return phi_new, dt
