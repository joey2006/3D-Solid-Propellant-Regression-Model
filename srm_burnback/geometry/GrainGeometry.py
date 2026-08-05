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
All lengths (radii, lengths, coordinates) are in **metres**, the project's
canonical unit -- see :mod:`srm_burnback.geometry.units`, which is the single
place that is stated. Imports are converted to metres at load time.

Inches appear only in the *display* layer: the measurements panel defaults to
them because experimental motor work is dimensioned in inches. That is a
formatting choice applied on the way out, and no stored number is ever in
inches. Geometries themselves bake in no unit -- they operate on whatever
consistent numbers they are given -- but everything that hands them numbers
hands them metres.
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
        self: actual grain object with its own parameters 

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

    def default_domain_size(self) -> float:
        """Half-extent of the domain to use when the caller passes no ``domain_size``.

        Grain-agnostic code can't know how big a given grain is, so each grain
        supplies its own sensible default (a parametric grain from its casing
        radius, an uploaded mesh from its bounding box). The base class has no
        way to guess, so subclasses must override this if they want
        :meth:`initialize_grid` to work without an explicit ``domain_size``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define a default domain size; "
            "pass domain_size explicitly to initialize_grid()."
        )

    def initialize_grid(
        self,
        resolution: int,
        domain_size: float | None = None,
        length: float | None = None,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, Coords, float]:
        """Build the starting field for any grain.

        This is intentionally grain-agnostic: it lays down a uniform grid and
        evaluates *this* grain's :meth:`signed_distance` on it. Because
        ``signed_distance`` is polymorphic, the same method produces the initial
        phi for a BATES grain, a star grain, or an uploaded 3D geometry -- the
        grain type is never referenced here.

        Parameters
        ----------
        resolution:
            Number of grid points along each axis.
        domain_size:
            Half-extent of the domain in x and y (axes span
            ``[-domain_size, +domain_size]``). When ``None`` the grain's
            :meth:`default_domain_size` is used.
        length:
            Axial length for a 3D grid. ``None`` builds a 2D grid.
        device:
            Torch device for the field (``"cpu"`` or ``"cuda"``).

        Returns
        -------
        phi:
            The initial signed-distance field (positive in the void, negative in
            the propellant). This is what the level-set method advects.
        coords:
            The coordinate grid phi was evaluated on.
        height:
            Grid spacing (voxel size), shared across axes for square cells.
        """
        from .CoordinateBuilder import build_coords

        if domain_size is None:
            domain_size = self.default_domain_size()

        coords, height = build_coords(resolution, domain_size, length, device)
        phi = self.signed_distance(coords)
        return phi, coords, height

    @property
    def ndim(self) -> int | None:
        """Spatial dimension the geometry is configured for (2 or 3).

        Returns ``None`` when the geometry can be evaluated in either dimension
        and the dimension is determined purely by the ``coords`` passed in.
        Subclasses may override when they are tied to a specific dimension.
        """
        return None
