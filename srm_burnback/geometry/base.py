"""Abstract base class for solid-propellant grain geometries.

A grain geometry is defined entirely by two signed distance functions (SDFs)
evaluated on a coordinate grid:

* :meth:`GrainGeometry.signed_distance` -- the *burning surface* (the initial
  propellant/void interface). By convention it is **positive** in the open
  void (e.g. the bore), **negative** inside the solid propellant, and **zero**
  on the surface itself. This is the field the level-set method advects.
* :meth:`GrainGeometry.outer_boundary_distance` -- the *casing wall*. The burn
  front cannot advance past this surface; the simulation uses it to mask the
  domain. Same sign convention: negative inside the casing, positive outside.

Why two SDFs (evolving vs. static)
----------------------------------
The two fields play very different roles at run time, which is why they are kept
separate rather than merged:

* ``signed_distance`` produces the **evolving** field (phi). The burn front
  moves, so phi is re-integrated across the *entire* grid every timestep by the
  level-set machinery (Godunov upwind, TVD-RK3, reinitialization). This is where
  essentially all the per-step cost lives.
* ``outer_boundary_distance`` produces a **static** field. The casing never
  moves, so it is evaluated *once* at initialization and thereafter only read --
  consulted each step as a cheap mask ("has the front reached the wall here?")
  to clamp the burn. It performs no work per step and adds no integration cost.

So per timestep the simulator evolves a single field (phi) plus one constant
guardrail. Merging them into one SDF would force phi to carry a kink where the
moving front meets the fixed wall, corrupting the gradients exactly where the
upwind scheme and reinitialization are most sensitive -- keeping them separate
is both cheaper and numerically cleaner.

Dimension-agnostic by design
-----------------------------
Coordinates are passed as a tuple of equally-shaped tensors -- ``(X, Y)`` for a
2D cross-section or ``(X, Y, Z)`` for a full 3D grain. Implementations must work
for both: branch on ``len(coords)`` rather than assuming a fixed dimension. The
2D case is only used to validate the math; 3D is the real product.

Units
-----
All lengths (radii, lengths, coordinates) must share a single consistent unit.
The project default for now is **inches**; a metric/imperial toggle will be
layered on later, so geometries should never bake in a particular unit -- they
just operate on whatever consistent numbers they are given.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

# A coordinate grid: (X, Y) in 2D or (X, Y, Z) in 3D. Every tensor has the same
# shape, typically produced by ``torch.meshgrid(..., indexing="ij")``.
Coords = tuple[torch.Tensor, ...]


class GrainGeometry(ABC):
    """Interface that every grain shape must implement.

    Subclasses store their defining parameters (radii, length, ...) and provide
    the two SDFs below. They should remain stateless with respect to the
    simulation -- given the same ``coords`` they always return the same field.
    """

    @abstractmethod
    def signed_distance(self, coords: Coords) -> torch.Tensor:
        """Signed distance to the initial burning surface.

        Parameters
        ----------
        coords:
            Tuple of coordinate tensors, ``(X, Y)`` in 2D or ``(X, Y, Z)`` in
            3D, all of identical shape.

        Returns
        -------
        torch.Tensor
            A tensor matching the shape of each coordinate tensor: positive in
            the open void, negative inside the propellant, zero on the surface.
        """

    @abstractmethod
    def outer_boundary_distance(self, coords: Coords) -> torch.Tensor:
        """Signed distance to the casing wall (the outer burn limit).

        Same convention and shape as :meth:`signed_distance`: negative inside
        the casing, positive outside it. The simulation uses this to stop the
        burn front from crossing the casing.
        """

    @property
    def ndim(self) -> int | None:
        """Spatial dimension the geometry is configured for (2 or 3).

        Returns ``None`` when the geometry can be evaluated in either dimension
        and the dimension is determined purely by the ``coords`` passed in.
        Subclasses may override when they are tied to a specific dimension.
        """
        return None
