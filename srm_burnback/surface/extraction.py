"""Extract the burning surface (the phi = 0 contour) from a 2D level set.

On a discrete grid no node sits exactly on the surface -- the zero level set
runs *between* grid points. ``skimage.measure.find_contours`` locates it to
sub-grid accuracy by linear interpolation along cell edges, returning closed
paths in array-index space. This module converts those to physical (x, y)
coordinates so the metrics module can measure them.

A grain can have several disconnected burning surfaces (e.g. a star grain's
points separating from the main bore), so extraction always returns a *list* of
contour paths.

2D / 3D
-------
This is the 2D path (contours). The 3D analogue is marching cubes (Phase 5,
#14); both feed the same dimension-agnostic metrics interface so downstream code
does not branch on dimension.
"""

from __future__ import annotations

import torch
from skimage.measure import find_contours

from ..geometry.GrainGeometry import Coords


def extract_contour_2d(
    phi: torch.Tensor, grid_x: torch.Tensor, grid_y: torch.Tensor
) -> list[torch.Tensor]:
    """Find the phi = 0 contour(s) as physical (x, y) coordinates.

    Parameters
    ----------
    phi:
        2D level-set field (positive in the void, negative in the propellant).
    grid_x, grid_y:
        The coordinate tensors phi was evaluated on, as built by
        ``torch.meshgrid(..., indexing="ij")``: ``grid_x[i, j]`` varies along
        axis 0 (rows), ``grid_y[i, j]`` along axis 1 (columns). Assumed a uniform
        grid (constant spacing per axis).

    Returns
    -------
    list[torch.Tensor]
        One ``(N, 2)`` tensor of ``(x, y)`` coordinates per connected contour.
        Empty when phi has no zero crossing (e.g. before ignition or after
        burnout).
    """
    if phi.ndim != 2:
        raise ValueError(f"extract_contour_2d expects a 2D field, got {phi.ndim}D")

    # Origin and spacing of each axis, read straight off the coordinate grids.
    x0 = grid_x[0, 0]
    y0 = grid_y[0, 0]
    hx = grid_x[1, 0] - grid_x[0, 0]
    hy = grid_y[0, 1] - grid_y[0, 0]

    # find_contours wants a plain array; coords come back as fractional
    # (row, col) indices along the cell edges where phi crosses 0.
    paths = find_contours(phi.detach().cpu().numpy(), level=0.0)

    contours: list[torch.Tensor] = []
    for path in paths:
        rc = torch.as_tensor(path, dtype=phi.dtype, device=phi.device)
        x = x0 + rc[:, 0] * hx  # row index -> x
        y = y0 + rc[:, 1] * hy  # col index -> y
        contours.append(torch.stack([x, y], dim=1))
    return contours
