"""An imported mesh as a :class:`GrainGeometry`, so the solver can run it (#192).

The import pipeline already goes all the way to a validated field::

    file -> load_mesh -> orient_grain_axis_to_z -> phi_from_labelled_mesh -> phi

but nothing could consume it. ``BurnbackSimulation.initialize()`` does not take
a phi tensor; it takes a ``GrainGeometry`` and calls two methods on it. This
class is the adapter that closes that gap. It is not new physics -- every piece
it needs already exists -- it is the piece that lets an uploaded object be
simulated exactly like a ``BATESGrain``.

Why phi is computed once and cached
-----------------------------------
The interface asks for the two fields through separate calls:
``signed_distance`` for the burning surface, ``outer_boundary_distance`` for the
casing. For a parametric grain those are two cheap formulas. For a mesh they are
the *same* traversal -- the generalized winding number visits every (grid point,
triangle) pair, and both fields fall out of it together -- and that traversal is
``O(cells x triangles)``: seconds on a GPU, minutes on a CPU.

Evaluating it twice would double the cost of every run for no benefit, so the
first call for a given grid computes both and the second is a lookup. The cache
is keyed on the identity of the coordinate tensors rather than their contents,
which is exact (the runner passes the very same tensors to both calls) and costs
nothing to check.

Why the mesh is centred on the origin
-------------------------------------
Every other geometry in the package is origin-centred: ``BATESGrain`` measures
radius from the origin and compares ``|z|`` against ``length / 2``, and
``build_coords`` lays a domain symmetrically about it. A mesh whose bounding box
sits somewhere else would still produce a correct field, but only if the grid
were moved to meet it -- which would make ``initialize_grid``, the one piece of
genuinely grain-agnostic setup code, need to know it was dealing with a mesh.

So the mesh is translated to the origin at construction instead, and the offset
is kept. That is the opposite of the choice ``grid_for_mesh`` makes, which
deliberately leaves imported coordinates in their original frame so they can be
compared against the source file. Both are right for their purpose: that
function serves the *viewer*, where matching the CAD file matters; this class
serves the *solver*, where matching the geometry convention matters.
"""

from __future__ import annotations

import torch

from .GrainGeometry import Coords, GrainGeometry
from .surfaces import DEFAULT_ENDS, surface_labels
from .winding_number import phi_from_labelled_boundary


class MeshGrain(GrainGeometry):
    """A triangle mesh, presented to the simulation as a grain geometry.

    Parameters
    ----------
    mesh:
        A ``trimesh.Trimesh``, already oriented so its long axis is Z (see
        :func:`import_mesh.orient_grain_axis_to_z`) and in metres (see
        :mod:`units`).
    ends:
        Whether the flat end faces burn or are inhibited. See
        :mod:`srm_burnback.geometry.surfaces` -- this materially changes burn
        area, so it is a parameter rather than a constant.
    margin:
        Fractional padding beyond the grain used by
        :meth:`default_domain_size`, so the burn front never reaches the domain
        edge where the Neumann ghost cells live.
    """

    def __init__(self, mesh, ends: str = DEFAULT_ENDS, margin: float = 0.05):
        if margin < 0.0:
            raise ValueError("margin must be non-negative")
        if ends != "inhibited":
            # Shelved, not merely untested (#194). With burning ends the
            # inhibited set is the outer wall alone -- an open cylinder with
            # nothing capping it -- so "inside the casing" has no meaning past
            # its rim. `phi_outer` decides that by comparing distances, and at
            # the rim the two are exactly equal because both surfaces share the
            # same edge circle. Cells outside the wall get reported as inside
            # the casing and the grain never burns out.
            #
            # phi itself is fine either way: its sign comes from the winding
            # number over the whole closed mesh. Only the simulation path is
            # closed off here; `phi_from_labelled_mesh` still accepts both, and
            # the burning-ends labelling remains the oracle check against
            # BATESGrain.
            raise ValueError(
                f"ends={ends!r} is not supported for simulation (#194). The "
                "casing field is ill-defined when the end faces burn, because "
                "the inhibited surface no longer encloses the grain. Use "
                "ends='inhibited'."
            )

        self.ends = ends
        self.margin = float(margin)

        centre = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
        self.offset = tuple(float(c) for c in centre)
        self.mesh = mesh.copy()
        self.mesh.apply_translation(-centre)

        # Labelled once: the classification is a property of the geometry, not
        # of any particular grid, and it is what the 3D view shows the user
        # before a run.
        self.labels = surface_labels(self.mesh, ends=ends)
        if self.labels["n_burning"] == 0:
            raise ValueError(
                "no burning surface was found on this mesh -- every face was "
                "classified as inhibited, so there is nothing to ignite. This "
                "usually means the grain has no bore, or its faces are wound "
                "inside-out."
            )

        self._cache_key = None
        self._cache: tuple[torch.Tensor, torch.Tensor] | None = None

    # -- the GrainGeometry interface ---------------------------------------

    def signed_distance(self, coords: Coords) -> torch.Tensor:
        """φ for the burning surface: negative in propellant, positive in void."""
        return self._fields(coords)[0]

    def outer_boundary_distance(self, coords: Coords) -> torch.Tensor:
        """The casing: negative inside the wall, positive beyond it.

        Built from the *inhibited* faces, which is what generalises the
        ``min(phi, -phi_outer)`` clamp beyond the cylindrical casing the
        parametric grains assume.
        """
        return self._fields(coords)[1]

    def default_domain_size(self) -> float:
        """Half-extent of the domain: the grain's largest half-extent, padded."""
        return float(max(self.mesh.extents)) / 2.0 * (1.0 + self.margin)

    def default_length(self) -> float:
        """Axial span for a 3D grid, matching :meth:`default_domain_size`.

        A cube, so the grid spacing ``h`` is identical on every axis -- the
        Godunov scheme and the CFL condition both take a single scalar ``h``.
        """
        return 2.0 * self.default_domain_size()

    @property
    def ndim(self) -> int:
        return 3

    # -- internals ---------------------------------------------------------

    def _fields(self, coords: Coords) -> tuple[torch.Tensor, torch.Tensor]:
        """Both fields for ``coords``, computing them at most once per grid."""
        if len(coords) != 3:
            raise ValueError(
                f"MeshGrain is a 3D geometry; got {len(coords)}D coordinates. "
                "A 2D cross-section import is #139."
            )

        key = tuple(
            (c.data_ptr(), tuple(c.shape), c.dtype, str(c.device)) for c in coords
        )
        if self._cache is not None and self._cache_key == key:
            return self._cache

        device, dtype = coords[0].device, coords[0].dtype
        vertices = torch.as_tensor(self.mesh.vertices, dtype=dtype, device=device)
        faces = torch.as_tensor(self.mesh.faces, dtype=torch.long, device=device)
        burning = torch.as_tensor(
            self.labels["burning"], dtype=torch.bool, device=device
        )

        self._cache = phi_from_labelled_boundary(coords, vertices, faces, burning)
        self._cache_key = key
        return self._cache

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"MeshGrain({len(self.mesh.faces):,} faces, ends={self.ends!r}, "
            f"{self.labels['n_burning']:,} burning)"
        )
