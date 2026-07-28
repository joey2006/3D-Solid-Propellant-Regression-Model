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

from srm_burnback.geometry.import_mesh import MeshImportError, load_mesh, mesh_stats


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

            # Watertightness and degeneracy checks walk the whole face list, so
            # this is not free on a large mesh -- keep it off the UI thread too.
            self.progress.emit("Analysing geometry...")
            stats = mesh_stats(mesh)

            self.finished.emit(mesh, stats, self._path)
        except MeshImportError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # a malformed file can fail in many ways
            self.failed.emit(f"Unexpected error reading '{self._path.name}': {exc}")
