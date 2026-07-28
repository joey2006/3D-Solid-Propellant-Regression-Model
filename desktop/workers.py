"""Background workers for the desktop app (#157).

Every long-running call must leave the UI thread. Mesh import is already
seconds-to-minutes at high resolution -- the generalized winding number is
``O(cells x triangles)`` -- and the solver will be worse. A frozen window during
that time reads as a crashed application.

The pattern throughout: a plain ``QObject`` worker moved onto a ``QThread``,
communicating results back by signal. Workers never touch widgets.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from srm_burnback.geometry.import_mesh import (
    AXIS_NAMES,
    MeshImportError,
    load_mesh,
    mesh_stats,
    orient_grain_axis_to_z,
)


class MeshLoadWorker(QObject):
    """Load and characterise a mesh file off the UI thread."""

    #: Emitted with ``(mesh, stats, path)`` when the load succeeds.
    finished = Signal(object, object, object)
    #: Emitted with a human-readable message when the load fails.
    failed = Signal(str)
    #: Emitted with a short status string as the work progresses.
    progress = Signal(str)

    def __init__(self, path: Path):
        super().__init__()
        self._path = Path(path)

    def run(self) -> None:
        try:
            self.progress.emit(f"Reading {self._path.name}...")
            mesh = load_mesh(self._path)

            # CAD tessellators do not guarantee consistent face winding, and a
            # face whose normal points the wrong way shades as if lit from
            # behind -- almost black, which against a dark background reads as
            # a hole in the model. It also breaks the inside/outside surface
            # classification, since that is decided from normal direction.
            # Cheap to repair (~1 ms) and a no-op when already correct.
            if not mesh.is_winding_consistent:
                self.progress.emit("Repairing face orientation...")
                mesh.fix_normals()

            # CAD packages disagree on the up axis -- Inventor and SolidWorks
            # commonly export Y-up. Everything downstream treats Z as the grain
            # axis, erosive burning most of all, so square it away at import.
            mesh, came_from = orient_grain_axis_to_z(mesh)

            # Watertightness and degeneracy checks walk the whole face list, so
            # this is not free on a large mesh -- keep it off the UI thread too.
            self.progress.emit("Analysing geometry...")
            stats = mesh_stats(mesh)
            stats["reoriented_from"] = (
                None if came_from == 2 else AXIS_NAMES[came_from]
            )

            self.finished.emit(mesh, stats, self._path)
        except MeshImportError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # a malformed file can fail in many ways
            self.failed.emit(f"Unexpected error reading '{self._path.name}': {exc}")
