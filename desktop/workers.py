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


class PhiWorker(QObject):
    """Build φ from a mesh off the UI thread (#158, #175).

    Genuinely slow work, not defensive threading: the winding number is
    ``O(cells x triangles)``, so 128³ against a 5k-face mesh is ~10^9
    point-primitive pairs. Seconds on a GPU, minutes on a CPU.
    """

    #: Emitted with ``(phi, phi_outer, coords, h, stats)`` on success.
    finished = Signal(object, object, object, float, object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, mesh, resolution: int, margin: float, device: str,
                 ends: str = "inhibited"):
        super().__init__()
        self._mesh = mesh
        self._resolution = int(resolution)
        self._margin = float(margin)
        self._device = device
        self._ends = ends

    def run(self) -> None:
        try:
            import time

            import torch

            from srm_burnback.geometry.import_mesh import grid_for_mesh
            from srm_burnback.geometry.winding_number import (
                phi_from_labelled_mesh,
            )

            self.progress.emit(
                f"Building {self._resolution}³ grid around the mesh..."
            )
            coords, h = grid_for_mesh(
                self._mesh, self._resolution, self._margin, device=self._device
            )

            self.progress.emit(
                f"Computing φ over {self._resolution ** 3:,} points "
                f"× {len(self._mesh.faces):,} triangles..."
            )
            started = time.time()
            # The labelled path, not the plain object SDF: distance is
            # measured to the burning surface only, so the outer wall does
            # not become a second zero contour the solver would advance.
            phi, phi_outer, labels = phi_from_labelled_mesh(
                coords, self._mesh, ends=self._ends
            )
            if self._device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - started

            # |grad phi| = 1 is the defining property of a signed distance
            # field, and every downstream scheme assumes it. Measured in a
            # narrow band around the surface, because that is the only place
            # the solver ever evaluates it -- far from the interface the field
            # is unused and its gradient does not matter.
            gradients = torch.gradient(phi, spacing=h)
            magnitude = torch.sqrt(sum(g**2 for g in gradients))
            band = phi.abs() < 3 * h
            in_band = magnitude[band]

            stats = {
                "seconds": elapsed,
                "grad_mean": float(in_band.mean()),
                "grad_std": float(in_band.std()),
                "grad_within_1pct": float(
                    ((in_band - 1.0).abs() < 0.01).float().mean()
                ),
                "band_cells": int(band.sum()),
                "solid_fraction": float((phi < 0).float().mean()),
                "phi_min": float(phi.min()),
                "phi_max": float(phi.max()),
                "n_burning": labels["n_burning"],
                "n_inhibited": labels["n_inhibited"],
                "ends": labels["ends"],
            }
            self.finished.emit(phi, phi_outer, coords, h, stats)
        except Exception as exc:
            self.failed.emit(f"Could not build φ: {exc}")


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
