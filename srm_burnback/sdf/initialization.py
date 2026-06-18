"""Brute-force signed-distance reconstruction from a sign field.

This is the general-purpose numerical SDF engine. Given only the *sign* of each
grid point -- which side of the burning surface it lies on -- it recovers a true
Euclidean signed-distance field. It is used in two places:

* **Reinitialization** -- after the level set evolves for a while the values of
  phi drift away from a true distance (``|grad phi|`` is no longer 1). Feeding
  the drifted phi back through here recomputes clean distances while leaving the
  zero contour exactly where it was.
* **Geometry import** -- an uploaded or otherwise arbitrary shape never had an
  analytical SDF. All we can produce for it is an inside/outside *sign* field
  (e.g. from the generalized winding number, #158, or directly from a CT/binary
  mask where the mask *is* the sign field). This routine turns that sign field
  into a usable phi.

Algorithm (dimension-agnostic, 2D or 3D)
----------------------------------------
1. **Zero-crossing detection** -- scan every axis for adjacent grid points whose
   sign differs. The burning surface passes between them.
2. **Interface interpolation** -- linearly interpolate along the edge between the
   positive and negative neighbour to locate where phi = 0 to sub-grid accuracy.
3. **Distance computation** -- for every grid point, find the Euclidean distance
   to the nearest interface point (vectorized with broadcasting).
4. **Sign restoration** -- multiply by the original sign so the result is
   negative inside the propellant and positive in the void.

Complexity is ``O(N_grid * N_interface)``. That is fine for 2D grids up to a few
hundred per side; for full 3D grids Phase 4 replaces this with the DiFi method.

Sign convention matches :mod:`srm_burnback.geometry`: **positive** in the open
void, **negative** inside the solid propellant, zero on the surface.
"""

from __future__ import annotations

import torch

from ..geometry.GrainGeometry import Coords


def _interface_points(phi: torch.Tensor, coords: Coords) -> torch.Tensor:
    """Locate the zero crossings of ``phi`` to sub-grid accuracy.

    Walks every axis looking for adjacent grid points with opposite sign and
    linearly interpolates the crossing position along that edge. Points that sit
    exactly on the surface (``phi == 0``) are included directly.

    Returns
    -------
    torch.Tensor
        ``(M, D)`` tensor of interface-point coordinates, where ``D`` is the
        spatial dimension and ``M`` the number of crossings found.
    """
    ndim = phi.ndim
    # Coordinates stacked as (..., D) so we can index whole points at once.
    grid = torch.stack(coords, dim=-1)

    points: list[torch.Tensor] = []

    # Grid points lying exactly on the surface are interface points themselves.
    exact = phi == 0
    if exact.any():
        points.append(grid[exact])

    # For each axis, compare each cell to its neighbour one step further along
    # that axis and interpolate where the sign flips.
    for axis in range(ndim):
        lo = phi.narrow(axis, 0, phi.shape[axis] - 1)
        hi = phi.narrow(axis, 1, phi.shape[axis] - 1)

        # A genuine crossing needs strictly opposite signs. Exact-zero endpoints
        # are handled above, so require both sides non-zero here to avoid
        # double-counting and division by a zero gap.
        crossing = (lo * hi < 0)
        if not crossing.any():
            continue

        grid_lo = grid.narrow(axis, 0, phi.shape[axis] - 1)
        grid_hi = grid.narrow(axis, 1, phi.shape[axis] - 1)

        lo_c = lo[crossing]
        hi_c = hi[crossing]
        # Fraction along the edge where phi = 0: phi is linear, so the zero sits
        # at lo / (lo - hi) of the way from the low point to the high point.
        frac = (lo_c / (lo_c - hi_c)).unsqueeze(-1)
        pts = grid_lo[crossing] + frac * (grid_hi[crossing] - grid_lo[crossing])
        points.append(pts)

    if not points:
        # No interface anywhere (the whole grid is one sign). Caller handles this.
        return grid.new_zeros((0, ndim))

    return torch.cat(points, dim=0)


def signed_distance_from_sign(
    phi: torch.Tensor,
    coords: Coords,
    chunk: int = 4096,
) -> torch.Tensor:
    """Rebuild a true signed-distance field from the sign of ``phi``.

    Only the *sign* of ``phi`` is used to place the surface and to sign the
    output; the magnitudes are recomputed from scratch as Euclidean distances to
    the interpolated zero contour. This is what makes the routine serve both
    reinitialization (drifted phi in) and geometry import (a raw sign field in).

    Parameters
    ----------
    phi:
        Field whose zero contour and sign define the surface. Any tensor with a
        valid sign structure works -- a drifted level set, or a raw +/-1 sign
        field from geometry import.
    coords:
        The coordinate grid ``phi`` lives on, ``(X, Y)`` in 2D or ``(X, Y, Z)``
        in 3D, every tensor matching ``phi``'s shape.
    chunk:
        Number of grid points to process per batch when computing nearest-
        interface distances. Caps peak memory on large grids; purely a
        performance knob with no effect on the result.

    Returns
    -------
    torch.Tensor
        A signed-distance field the same shape as ``phi``: positive in the void,
        negative in the propellant, ``|grad phi| ~= 1`` everywhere.
    """
    interface = _interface_points(phi, coords)

    # Original sign drives the output sign. Points exactly on the surface get
    # sign 0, which correctly yields distance 0 there.
    sign = torch.sign(phi)

    if interface.shape[0] == 0:
        # No surface on the grid: distance is undefined, but the sign tells us
        # which side everything is on. Return a large signed constant rather
        # than NaN so downstream masks still behave.
        far = float(max(c.abs().max().item() for c in coords)) * 2.0
        return sign * far

    flat_grid = torch.stack(coords, dim=-1).reshape(-1, phi.ndim)

    # Nearest-interface distance for every grid point, in memory-bounded chunks.
    dist_chunks: list[torch.Tensor] = []
    for start in range(0, flat_grid.shape[0], chunk):
        block = flat_grid[start : start + chunk]               # (B, D)
        # (B, M) pairwise distances from this block to every interface point.
        d = torch.cdist(block, interface)
        dist_chunks.append(d.min(dim=1).values)

    dist = torch.cat(dist_chunks, dim=0).reshape(phi.shape)
    return sign * dist
