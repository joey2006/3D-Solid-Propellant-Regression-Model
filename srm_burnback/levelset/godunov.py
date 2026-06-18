"""Godunov upwind scheme for the level-set gradient magnitude.

The burning surface is the zero level set of phi, advected by the
Hamilton-Jacobi equation

    d(phi)/dt + F * |grad phi| = 0,

where ``F`` is the local burn rate (speed of the front along its normal). Solving
it numerically needs ``|grad phi|`` at every grid point, and the *way* that
gradient is discretized is what keeps the scheme stable. A naive central
difference rings and blows up near the front; the Godunov scheme instead looks
**upwind** -- it picks, per axis, the one-sided difference that information is
actually flowing from, chosen by the sign of ``F``.

For motion in the outward normal direction (``F > 0``):

    |grad phi|^2 = sum_axis  max( max(D-, 0)^2 , min(D+, 0)^2 )

and for inward motion (``F < 0``) the selection flips:

    |grad phi|^2 = sum_axis  max( min(D-, 0)^2 , max(D+, 0)^2 )

where ``D-`` and ``D+`` are the backward and forward differences along each axis.
This is the spatial discretization from Osher & Fedkiw, *Level Set Methods and
Dynamic Implicit Surfaces*, Ch. 5-6.

Dimension-agnostic by design
----------------------------
Everything loops over ``range(phi.ndim)``, so the identical code handles a 2D
cross-section ``(Ny, Nx)`` and a full 3D grain ``(Nz, Ny, Nx)`` -- 3D just adds
another term under the square root. No axis is ever hardcoded.

Boundaries
----------
Finite differences need a neighbor on each side. At the domain edge there isn't
one, so phi is padded with one layer of **replicated** edge values (a Neumann,
zero-normal-derivative ghost cell). That keeps the differencing in-bounds and
NaN-free everywhere. The domain is sized so the burn front never reaches the
edge, so the one-sided behavior there never touches the physics.
"""

from __future__ import annotations

import torch


def _pad_replicate(phi: torch.Tensor, axis: int) -> torch.Tensor:
    """Pad ``phi`` by one cell on both ends of ``axis``, replicating the edge.

    This is the Neumann (zero normal derivative) ghost cell: the value just
    outside the domain equals the value on the boundary, so a difference taken
    across the edge is zero rather than out of bounds.
    """
    first = phi.narrow(axis, 0, 1)
    last = phi.narrow(axis, phi.shape[axis] - 1, 1)
    return torch.cat([first, phi, last], dim=axis)


def _finite_differences(
    phi: torch.Tensor, axis: int, h: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward and backward differences of ``phi`` along one axis.

    Parameters
    ----------
    phi:
        The field, any number of dimensions.
    axis:
        Spatial axis to difference along (``0``, ``1``, or ``2``).
    h:
        Grid spacing.

    Returns
    -------
    (D_forward, D_backward):
        Two tensors the same shape as ``phi``.
        ``D_forward[i]  = (phi[i+1] - phi[i]) / h`` (forward / "looking ahead"),
        ``D_backward[i] = (phi[i] - phi[i-1]) / h`` (backward / "looking behind").
        At the edges the replicated ghost cell makes the missing-side difference
        zero, so both tensors keep phi's shape with no out-of-bounds access.
    """
    n = phi.shape[axis]
    p = _pad_replicate(phi, axis)  # length n + 2 along `axis`

    # Index into the padded tensor with narrow so the same code works for any
    # axis and dimension. Original index i lives at padded index i + 1.
    p_lo = p.narrow(axis, 0, n)       # padded[0 : n]   -> phi[i-1] (ghost at i=0)
    p_mid = p.narrow(axis, 1, n)      # padded[1 : n+1] -> phi[i]
    p_hi = p.narrow(axis, 2, n)       # padded[2 : n+2] -> phi[i+1] (ghost at i=n-1)

    d_forward = (p_hi - p_mid) / h
    d_backward = (p_mid - p_lo) / h
    return d_forward, d_backward


def godunov_gradient_magnitude(
    phi: torch.Tensor,
    h: float,
    F_sign: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Upwind ``|grad phi|`` via the Godunov scheme.

    Parameters
    ----------
    phi:
        Level-set field, 2D ``(Ny, Nx)`` or 3D ``(Nz, Ny, Nx)``.
    h:
        Grid spacing (assumed equal on every axis -- square/cubic cells).
    F_sign:
        Sign of the burn rate ``F`` selecting the upwind direction. ``+1`` for
        outward motion (the normal case -- the front advances into the
        propellant), ``-1`` for inward motion. May be a scalar or a tensor the
        shape of ``phi`` for a spatially varying rate (e.g. erosive burning,
        where the upwind direction is taken per grid point).

    Returns
    -------
    torch.Tensor
        ``|grad phi|`` at every grid point, same shape as ``phi``. On an exact
        signed-distance field this is ~1 everywhere in the interior (the
        defining property of an SDF).
    """
    # Per-point selector so a scalar and a tensor F_sign share one code path.
    f_pos = torch.as_tensor(F_sign, dtype=phi.dtype, device=phi.device) > 0

    grad_sq = torch.zeros_like(phi)
    for axis in range(phi.ndim):
        d_forward, d_backward = _finite_differences(phi, axis, h)

        # Outward (F > 0): keep the backward difference only where it is
        # positive and the forward difference only where it is negative -- that
        # is the information coming from upwind.
        pos = torch.maximum(
            d_backward.clamp(min=0) ** 2, d_forward.clamp(max=0) ** 2
        )
        # Inward (F < 0): the mirror selection.
        neg = torch.maximum(
            d_backward.clamp(max=0) ** 2, d_forward.clamp(min=0) ** 2
        )

        grad_sq = grad_sq + torch.where(f_pos, pos, neg)

    return torch.sqrt(grad_sq)
