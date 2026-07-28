"""Load a grain geometry from a triangle-mesh file (issue #138).

This owns exactly one step of the shared geometry pipeline::

    object file -> MESH -> winding-number sign field (#158)
                        -> numerical distance (#3) -> phi

Everything after the first arrow is generic, grain-agnostic machinery shared by
every geometry source. This module's only job is turning a file on disk into a
triangle mesh plus the numbers needed to place it on a simulation grid.

Triangles are an import-time detail
-----------------------------------
A mesh file stores a *surface* as a list of vertices and triangular faces. The
triangles exist only so the winding number has something to sum over; once phi
is built nothing downstream ever refers to them again. The simulation operates
purely on the volumetric field.

Units
-----
Mesh files carry no reliable unit declaration (STL in particular has none), so
the numbers are taken as-is. ``mesh_stats`` reports the bounding box precisely
so a unit mismatch is obvious on sight: a grain measuring 50 x 50 x 120 is
millimetres, while 0.05 x 0.05 x 0.12 is metres.

Native CAD files *do* declare units internally, but the conversion goes through
OpenCASCADE and lands in millimetres, whereas the simulation works in metres.
So a STEP grain typically arrives 1000x too large and needs scaling. The extent
warning is what surfaces this; automatic handling is #170.

CAD versus mesh input
---------------------
STEP files are boundary representations -- exact analytic surfaces, not
triangles. Loading one *tessellates* it, and the tessellation tolerance is
chosen by the converter rather than by us. That matters because tessellation
density is what limits how accurately an imported grain matches its analytical
counterpart (#158). Control over that tolerance is #161.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .CoordinateBuilder import build_coords
from .GrainGeometry import Coords

# Extensions trimesh handles that make sense as a grain source.
MESH_SUFFIXES = {".stl", ".obj", ".ply", ".off", ".glb", ".gltf", ".3mf"}

# Native CAD (#161). These are B-rep formats, not meshes: the file describes
# exact surfaces, and a tessellation is generated on load. Handled by trimesh
# via the `cascadio` backend (OpenCASCADE), which is an optional dependency.
CAD_SUFFIXES = {".step", ".stp"}

SUPPORTED_SUFFIXES = MESH_SUFFIXES | CAD_SUFFIXES


class MeshImportError(RuntimeError):
    """Raised when a file cannot be read as a usable triangle mesh."""


def load_mesh(path: str | Path):
    """Read a mesh file into a single ``trimesh.Trimesh``.

    Parameters
    ----------
    path:
        File to load. Suffix must be one of :data:`SUPPORTED_SUFFIXES`.

    Returns
    -------
    trimesh.Trimesh
        The loaded mesh. Multi-body scenes are concatenated into one mesh --
        selecting an individual body is a CAD-format concern (#169), and STL
        has no body structure to select from in the first place.

    Raises
    ------
    MeshImportError
        If the suffix is unsupported, the file cannot be parsed, or the result
        contains no faces.
    """
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise MeshImportError(
            "trimesh is required to import mesh files. Install it with "
            "`pip install trimesh`."
        ) from exc

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise MeshImportError(
            f"unsupported mesh format '{path.suffix}'. Supported: {supported}"
        )

    if suffix in CAD_SUFFIXES:
        try:
            import cascadio  # noqa: F401  (registers the STEP loader)
        except ImportError as exc:
            raise MeshImportError(
                f"reading '{path.suffix}' CAD files needs the `cascadio` "
                "backend. Install it with `pip install cascadio`."
            ) from exc

    try:
        loaded = trimesh.load_mesh(str(path))
    except Exception as exc:
        raise MeshImportError(f"could not parse '{path.name}': {exc}") from exc

    # A file holding several bodies loads as a Scene; flatten it.
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise MeshImportError(f"'{path.name}' contains no geometry")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))

    if getattr(loaded, "faces", None) is None or len(loaded.faces) == 0:
        raise MeshImportError(f"'{path.name}' contains no triangles")

    return loaded


def mesh_stats(mesh) -> dict:
    """Summarise a mesh, focusing on what predicts downstream trouble.

    Each field maps to a specific failure mode:

    * ``n_triangles`` -- the winding number costs ``O(cells x triangles)``, so
      this sets whether a given grid resolution is tractable.
    * ``watertight`` -- a non-watertight mesh is exactly the case the
      generalized winding number exists to handle (#158). Knowing which path is
      being exercised matters when interpreting the result.
    * ``extents`` / ``bounds`` -- a unit error is visible here and almost
      nowhere else. Metres and millimetres differ by 1000x, which would produce
      a plausible-looking simulation with a wildly wrong timescale.
    * ``n_degenerate`` -- zero-area triangles subtend an undefined solid angle
      and can poison the winding-number sum.
    * ``volume`` -- only meaningful when watertight; ``None`` otherwise.

    Returns
    -------
    dict
        Plain Python scalars, safe to render or serialize.
    """
    watertight = bool(mesh.is_watertight)

    # A face is degenerate when its area is ~0: two vertices coincide or all
    # three are colinear, leaving no well-defined normal.
    areas = mesh.area_faces
    n_degenerate = int((areas <= 0.0).sum())

    bounds = mesh.bounds  # (2, 3): [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    extents = mesh.extents

    return {
        "n_triangles": int(len(mesh.faces)),
        "n_vertices": int(len(mesh.vertices)),
        "watertight": watertight,
        "n_degenerate": n_degenerate,
        "surface_area": float(mesh.area),
        # Volume is only meaningful for a closed surface; trimesh will happily
        # return a nonsense number for an open one.
        "volume": float(mesh.volume) if watertight else None,
        "bounds_min": [float(v) for v in bounds[0]],
        "bounds_max": [float(v) for v in bounds[1]],
        "extents": [float(v) for v in extents],
        "max_extent": float(max(extents)),
    }


def mesh_tensors(
    mesh, device: str = "cpu", dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mesh as torch tensors: ``(vertices (V, 3), faces (F, 3))``.

    Faces are int64 vertex indices, suitable for ``vertices[faces]`` gathering.
    """
    vertices = torch.as_tensor(mesh.vertices, dtype=dtype, device=device)
    faces = torch.as_tensor(mesh.faces, dtype=torch.long, device=device)
    return vertices, faces


def grid_for_mesh(
    mesh,
    resolution: int,
    margin: float = 0.05,
    device: str = "cpu",
) -> tuple[Coords, float]:
    """Build a simulation grid enclosing the mesh.

    The domain is a cube centred on the mesh's bounding-box centre, sized to the
    mesh's largest extent plus ``margin`` on every side.

    A cube (rather than a box tracking each axis independently) keeps the grid
    spacing ``h`` identical on all axes, which the Godunov scheme and the CFL
    condition both assume -- they take a single scalar ``h``.

    The margin matters for correctness, not neatness: the level-set boundary
    handling uses replicated (Neumann) ghost cells, which are only harmless
    while the burn front stays away from the domain edge. Padding guarantees
    that at ``t = 0`` and leaves room for the front to expand.

    Parameters
    ----------
    mesh:
        Mesh to enclose.
    resolution:
        Grid points per axis.
    margin:
        Fractional padding beyond the largest extent (0.05 = 5%).
    device:
        Torch device for the coordinate tensors.

    Returns
    -------
    (coords, h):
        Coordinate grid ``(X, Y, Z)`` and the scalar grid spacing.

    Notes
    -----
    The mesh is *not* moved. Instead the grid is centred on the mesh, so
    imported coordinates stay in their original frame and any later comparison
    against the source file remains meaningful.
    """
    if resolution < 3:
        raise ValueError("resolution must be at least 3 grid points")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")

    bounds = mesh.bounds
    centre = [(float(lo) + float(hi)) / 2.0 for lo, hi in zip(bounds[0], bounds[1])]
    half_extent = float(max(mesh.extents)) / 2.0 * (1.0 + margin)

    # build_coords spans [-domain_size, +domain_size] in x/y and
    # [-length/2, +length/2] in z. Passing length = 2 * half_extent makes the
    # z span match x/y exactly, giving a true cube and one shared spacing h.
    coords, h = build_coords(
        resolution, half_extent, length=2.0 * half_extent, device=device
    )
    shifted = tuple(c + centre[i] for i, c in enumerate(coords))
    return shifted, h


def estimate_winding_cost(n_cells: int, n_triangles: int, device: str) -> float:
    """Rough seconds for a brute-force winding-number evaluation.

    Calibrated from measured throughput on this project's hardware (RTX 3080 /
    CPU fallback). Used to warn before launching an intractable computation
    rather than letting the UI hang.

    The estimate is deliberately crude -- it exists to distinguish "seconds"
    from "an hour", not to be accurate.
    """
    # Measured: ~3.6e8 point-triangle pairs/sec on CUDA, ~4.0e6 on CPU.
    throughput = 3.6e8 if device == "cuda" else 4.0e6
    return n_cells * n_triangles / throughput
