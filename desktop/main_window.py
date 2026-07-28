"""Main window for the SRM Burnback desktop application (#157, #130, #131).

Assembles the docks, the central views, and the actions that connect them.

Layering rule
-------------
This package imports ``srm_burnback``; the package never imports Qt. All real
logic lives in the engine and stays testable without a UI. Everything here is
presentation and wiring.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThread
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QTabWidget,
)

from srm_burnback.geometry.measurements import grain_measurements
from srm_burnback.geometry.import_mesh import (
    CAD_SUFFIXES,
    MESH_SUFFIXES,
    NATIVE_CAD_FORMATS,
    SUPPORTED_SUFFIXES,
    estimate_winding_cost,
    grid_for_mesh,
)

from . import theme
from .panels import (
    GeometryPanel,
    MeasurementsPanel,
    PropellantPanel,
    SimulationPanel,
)
from .views import FieldView, MeshDataView, MeshView
from .workers import MeshLoadWorker

ORG = "SRM Burnback"
APP = "Burnback Studio"
MAX_RECENT = 8


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP)
        self.resize(1440, 900)

        self._mesh = None
        self._stats = None
        self._path: Path | None = None
        self._thread: QThread | None = None
        self._worker: MeshLoadWorker | None = None
        self._settings = QSettings(ORG, APP)

        self._build_views()
        self._build_docks()
        self._build_actions()
        self._build_status_bar()

        self._restore_layout()
        self._refresh_grid_metrics()

    # -- Construction ------------------------------------------------------

    def _build_views(self) -> None:
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.mesh_view = MeshView()
        self.data_view = MeshDataView()
        self.field_view = FieldView()

        self.tabs.addTab(self.mesh_view, "3D View")
        self.tabs.addTab(self.data_view, "Mesh Data")
        self.tabs.addTab(self.field_view, "Signed Distance Field")
        self.setCentralWidget(self.tabs)

    def _build_docks(self) -> None:
        self.geometry_panel = GeometryPanel()
        self.measurements_panel = MeasurementsPanel()
        self.propellant_panel = PropellantPanel()
        self.simulation_panel = SimulationPanel()

        self.geometry_panel.open_requested.connect(self.open_mesh)
        self.geometry_panel.recent_selected.connect(
            lambda p: self.load_path(Path(p))
        )
        self.geometry_panel.changed.connect(self._refresh_grid_metrics)
        # Mass is volume x density, so it has to follow the propellant panel.
        self.propellant_panel.changed.connect(self._refresh_measurements)

        self._docks: dict[str, QDockWidget] = {}
        self._add_dock("Geometry", self.geometry_panel, Qt.LeftDockWidgetArea)
        self._add_dock(
            "Measurements", self.measurements_panel, Qt.LeftDockWidgetArea
        )
        self._add_dock("Propellant", self.propellant_panel, Qt.RightDockWidgetArea)
        self._add_dock("Simulation", self.simulation_panel, Qt.RightDockWidgetArea)

        self.resizeDocks(
            list(self._docks.values()), [330] * len(self._docks), Qt.Horizontal
        )

    def _add_dock(self, title: str, widget, area) -> None:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setObjectName(f"dock_{title.lower()}")
        dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(area, dock)
        self._docks[title] = dock

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        self.open_action = QAction("&Open mesh...", self)
        self.open_action.setShortcut(QKeySequence.Open)
        self.open_action.triggered.connect(self.open_mesh)
        file_menu.addAction(self.open_action)

        self.recent_menu = file_menu.addMenu("Open &recent")
        self._rebuild_recent_menu()

        self.close_action = QAction("&Close mesh", self)
        self.close_action.setShortcut(QKeySequence.Close)
        self.close_action.triggered.connect(self.close_mesh)
        self.close_action.setEnabled(False)
        file_menu.addAction(self.close_action)

        file_menu.addSeparator()
        quit_action = QAction("E&xit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("&View")
        for title, dock in self._docks.items():
            view_menu.addAction(dock.toggleViewAction())
        view_menu.addSeparator()
        reset_action = QAction("Reset &layout", self)
        reset_action.triggered.connect(self.reset_layout)
        view_menu.addAction(reset_action)

        help_menu = self.menuBar().addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _build_status_bar(self) -> None:
        self.status_message = QLabel("Ready")
        self.status_message.setStyleSheet(f"color:{theme.TEXT_MUTED};")
        self.statusBar().addWidget(self.status_message, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.setFixedWidth(140)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)

        self.status_device = QLabel()
        self.status_device.setStyleSheet(f"color:{theme.TEXT_FAINT};")
        self.statusBar().addPermanentWidget(self.status_device)

    # -- Mesh loading ------------------------------------------------------

    def open_mesh(self) -> None:
        supported = " ".join(f"*{s}" for s in sorted(SUPPORTED_SUFFIXES))
        native = " ".join(f"*{s}" for s in sorted(NATIVE_CAD_FORMATS))
        meshes = " ".join(f"*{s}" for s in sorted(MESH_SUFFIXES))
        cad = " ".join(f"*{s}" for s in sorted(CAD_SUFFIXES))

        # Native CAD files are listed in the default filter even though they
        # cannot be opened. Filtering them out hides the file a user is
        # plainly looking at, with no hint as to why -- far more confusing
        # than selecting it and being told to export a neutral format.
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open grain geometry",
            str(self._last_directory()),
            ";;".join(
                [
                    f"Grain geometry ({supported} {native})",
                    f"Mesh files ({meshes})",
                    f"CAD files ({cad})",
                    "All files (*)",
                ]
            ),
        )
        if path:
            self.load_path(Path(path))

    def load_path(self, path: Path) -> None:
        """Load a mesh off the UI thread, keeping the window responsive."""
        if self._thread is not None:
            return  # a load is already running

        self._set_busy(True, f"Reading {path.name}...")

        self._thread = QThread(self)
        self._worker = MeshLoadWorker(path)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_loaded)
        self._worker.failed.connect(self._on_failed)
        # Both terminal signals must tear the thread down, or a failed load
        # leaves it running and blocks every subsequent open.
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self._thread.start()

    def _cleanup_thread(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self._set_busy(False)

    def _on_progress(self, message: str) -> None:
        self.status_message.setText(message)

    def _on_loaded(self, mesh, stats, path) -> None:
        self._mesh = mesh
        self._stats = stats
        self._path = Path(path)

        self.setWindowTitle(f"{self._path.name} — {APP}")
        self.geometry_panel.set_file(self._path.name)
        self.close_action.setEnabled(True)
        self._remember_recent(self._path)
        self.geometry_panel.select_recent(str(self._path))

        self.mesh_view.show_mesh(mesh)
        self._populate_data_view()
        self._refresh_measurements()
        self._refresh_grid_metrics()

        self.status_message.setText(
            f"Loaded {self._path.name} — {stats['n_triangles']:,} triangles"
        )

    def _on_failed(self, message: str) -> None:
        self.status_message.setText("Load failed")
        QMessageBox.warning(self, "Could not open mesh", message)

    def close_mesh(self) -> None:
        self._mesh = None
        self._stats = None
        self._path = None
        self.setWindowTitle(APP)
        self.geometry_panel.set_file(None)
        self.close_action.setEnabled(False)
        self.mesh_view.clear()
        self.data_view.clear()
        self.measurements_panel.clear()
        self._refresh_grid_metrics()
        self.status_message.setText("Ready")

    # -- Diagnostics -------------------------------------------------------

    def _populate_data_view(self) -> None:
        stats = self._stats
        view = self.data_view
        view.title.setText(self._path.name if self._path else "mesh")

        view.topology.set("triangles", f"{stats['n_triangles']:,}")
        view.topology.set("vertices", f"{stats['n_vertices']:,}")
        view.topology.set(
            "watertight",
            "Yes" if stats["watertight"] else "No",
            "ok" if stats["watertight"] else "warn",
        )
        view.topology.set(
            "degenerate",
            f"{stats['n_degenerate']:,}",
            "ok" if stats["n_degenerate"] == 0 else "warn",
        )

        ex = stats["extents"]
        for key, value in zip(("x", "y", "z"), ex):
            view.extents.set(key, _fmt(value))
        view.extents.set(
            "volume",
            _fmt(stats["volume"]) if stats["volume"] is not None else "n/a",
        )

        messages = []
        if stats.get("reoriented_from"):
            messages.append(
                f"This model's long axis was {stats['reoriented_from']}; it has "
                "been rotated so the grain axis lies along Z. Everything "
                "downstream assumes that, erosive burning especially, since "
                "axial mass flux accumulates along the bore."
            )
        if not stats["watertight"]:
            messages.append(
                "This mesh is not watertight — it has holes or boundary edges. "
                "That is supported: the generalized winding number degrades "
                "gracefully rather than mis-signing. Ray casting would produce "
                "wrong signs here."
            )
        if stats["n_degenerate"]:
            messages.append(
                f"{stats['n_degenerate']} degenerate (zero-area) faces "
                "contribute no solid angle and are skipped, but a large count "
                "suggests a damaged mesh."
            )
        if stats["max_extent"] > 10.0:
            messages.append(
                "Largest extent exceeds 10. If this grain is in millimetres, "
                "the simulation works in metres and the geometry is ~1000× too "
                "large. STL files carry no unit declaration, so this cannot be "
                "detected automatically."
            )

        if messages:
            view.banner.show_message("  ".join(messages), "warn")
        else:
            view.banner.show_message(
                "Mesh is closed, non-degenerate, and dimensioned plausibly "
                "for metres.",
                "ok",
            )

    def _refresh_measurements(self) -> None:
        """Recompute grain dimensions for the current mesh and density."""
        if self._mesh is None:
            self.measurements_panel.clear()
            return
        try:
            self.measurements_panel.set_measurements(
                grain_measurements(
                    self._mesh, density=self.propellant_panel.density.value()
                )
            )
        except Exception:
            # Dimensions are informational; a mesh too odd to measure must not
            # stop it being viewed.
            self.measurements_panel.clear()

    def _refresh_grid_metrics(self) -> None:
        """Update the grid tiles. Cheap enough to run on every slider tick."""
        view = self.data_view
        resolution = self.geometry_panel.resolution_points()
        device = self.geometry_panel.device_string()
        n_cells = resolution ** 3

        self.status_device.setText(f"{resolution}³ · {device.upper()}")
        view.grid.set("cells", f"{n_cells:,}")

        if self._mesh is None:
            view.grid.set("spacing", "--")
            view.grid.set("across", "--")
            view.grid.set("cost", "--")
            return

        _, h = grid_for_mesh(
            self._mesh, resolution, margin=self.geometry_panel.margin_value()
        )
        view.grid.set("spacing", _fmt(h))

        across = self._stats["extents"][0] / h if h > 0 else 0
        view.grid.set(
            "across", f"{across:.0f}", "ok" if across >= 20 else "warn"
        )

        cost = estimate_winding_cost(n_cells, self._stats["n_triangles"], device)
        if cost > 60:
            view.grid.set("cost", f"{cost / 60:.1f} min", "error")
        elif cost > 5:
            view.grid.set("cost", f"{cost:.0f} s", "warn")
        else:
            view.grid.set("cost", f"{cost:.1f} s", "ok")

    # -- Recent files ------------------------------------------------------

    def _recent_files(self) -> list[str]:
        return list(self._settings.value("recent_files", []) or [])

    def _remember_recent(self, path: Path) -> None:
        # Resolve before comparing: the same file reached via a relative path
        # and via the file dialog's absolute one are different strings, and
        # would otherwise both sit in the list.
        resolved = str(Path(path).resolve())
        recent = [
            p for p in self._recent_files() if str(Path(p).resolve()) != resolved
        ]
        recent.insert(0, resolved)
        self._settings.setValue("recent_files", recent[:MAX_RECENT])
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        self.recent_menu.clear()
        recent = [p for p in self._recent_files() if Path(p).exists()]
        # The Geometry dock shows the same list; both are driven from here so
        # they cannot drift apart.
        self.geometry_panel.set_recent_files(recent)
        if not recent:
            empty = self.recent_menu.addAction("No recent files")
            empty.setEnabled(False)
            return
        for entry in recent:
            action = self.recent_menu.addAction(Path(entry).name)
            action.setToolTip(entry)
            action.triggered.connect(
                lambda _=False, p=entry: self.load_path(Path(p))
            )

    def _last_directory(self) -> Path:
        recent = self._recent_files()
        if recent:
            parent = Path(recent[0]).parent
            if parent.exists():
                return parent
        samples = Path(__file__).resolve().parent.parent / "examples" / "output" / "meshes"
        return samples if samples.exists() else Path.home()

    # -- Layout / lifecycle ------------------------------------------------

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self.progress.setVisible(busy)
        self.open_action.setEnabled(not busy)
        self.geometry_panel.open_button.setEnabled(not busy)
        if message:
            self.status_message.setText(message)

    def _restore_layout(self) -> None:
        geometry = self._settings.value("window_geometry")
        state = self._settings.value("window_state")
        if geometry:
            self.restoreGeometry(geometry)
        if state:
            self.restoreState(state)

    def reset_layout(self) -> None:
        self._settings.remove("window_geometry")
        self._settings.remove("window_state")
        for area, title in (
            (Qt.LeftDockWidgetArea, "Geometry"),
            (Qt.RightDockWidgetArea, "Propellant"),
            (Qt.RightDockWidgetArea, "Simulation"),
        ):
            dock = self._docks[title]
            dock.setFloating(False)
            dock.show()
            self.addDockWidget(area, dock)
        self.resize(1440, 900)

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            f"About {APP}",
            f"<h3>{APP}</h3>"
            "<p>3D solid propellant grain burnback simulation using the "
            "level set method.</p>"
            "<p style='color:#8b949e'>Geometry pipeline: object → mesh → "
            "winding-number sign → closest-point distance → φ</p>",
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._settings.setValue("window_geometry", self.saveGeometry())
        self._settings.setValue("window_state", self.saveState())
        # VTK holds a native render window that must be torn down explicitly,
        # or the process can hang or crash on exit.
        self.mesh_view.close_plotter()
        super().closeEvent(event)


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}g}"
