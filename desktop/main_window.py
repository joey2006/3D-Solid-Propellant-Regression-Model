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
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QScrollArea,
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
    mesh_stats,
)

from . import theme
from .panels import (
    GeometryPanel,
    MeasurementsPanel,
    PropellantPanel,
    SimulationPanel,
)
from .views import FieldView, MeshDataView, MeshView
from .workers import MeshLoadWorker, PhiWorker

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

        # Dropping a file on the window is the shortest path from "I have a
        # grain" to "I can see it" (#140). The file dialog stays, since
        # drag-and-drop is undiscoverable on its own.
        self.setAcceptDrops(True)

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
        self.field_view.build_requested.connect(self.build_field)
        self.setCentralWidget(self.tabs)

    # -- Drag and drop (#140) ----------------------------------------------

    def _dropped_path(self, event) -> Path | None:
        """The first supported local file in a drag, or ``None``.

        Accepting only recognised suffixes means the cursor shows "no entry"
        over a PDF instead of accepting the drop and then failing with a
        dialog -- the rejection happens while the user still has the file in
        hand, which is where it is useful.
        """
        if not event.mimeData().hasUrls():
            return None
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            suffix = path.suffix.lower()
            if suffix in SUPPORTED_SUFFIXES or suffix in NATIVE_CAD_FORMATS:
                return path
        return None

    def dragEnterEvent(self, event):  # noqa: N802 - Qt naming
        if self._dropped_path(event) is not None:
            event.acceptProposedAction()

    def dragMoveEvent(self, event):  # noqa: N802 - Qt naming
        if self._dropped_path(event) is not None:
            event.acceptProposedAction()

    def dropEvent(self, event):  # noqa: N802 - Qt naming
        path = self._dropped_path(event)
        if path is None:
            return
        event.acceptProposedAction()
        # Native CAD formats are matched above so the drop is accepted and the
        # loader can explain which menu item exports a neutral format. Silently
        # refusing the drop would leave the user with no idea why.
        self.load_path(path)

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
        # Remember the unit choice; it is a preference, not per-file state.
        self.measurements_panel.units_changed.connect(self._on_units_changed)
        units = str(self._settings.value("units", "in"))
        self.measurements_panel.set_units(units, notify=False)
        self.mesh_view.set_units(units)
        self.field_view.set_units(units)

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

        # Panels go inside a scroll area rather than straight into the dock.
        # A dock gives its contents whatever height is left over once the other
        # docks in the same column have taken theirs, and a widget taller than
        # that is simply clipped -- the bottom controls vanish with no
        # indication they exist. Scrolling keeps everything reachable at any
        # window size.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Wide enough that the labelled rows never need horizontal scrolling.
        scroll.setMinimumWidth(300)

        dock.setWidget(scroll)
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

    #: A grain whose largest extent falls outside this range, in metres, is
    #: almost certainly a unit error rather than an unusual motor. A 5 mm floor
    #: sits below any real grain; a 10 m ceiling sits above anything this tool
    #: is for.
    PLAUSIBLE_EXTENT_M = (5e-3, 10.0)

    def _confirm_units(self, mesh, stats):
        """Ask what unit an undeclared file was drawn in, when it looks wrong.

        Only STL and friends reach here -- a CAD file declares its unit and is
        converted on import (#170). For those formats nothing in the file says
        whether ``50`` means metres or millimetres, so the only evidence is the
        size itself, and the only reliable judge is the person who drew it.

        The prompt is deliberately conditional on an implausible size rather
        than shown on every import. A grain that already measures 0.12 m needs
        no interrogation, and a dialog on every open trains the user to dismiss
        it -- which is exactly the reflex that lets a real scale error through.
        """
        if stats.get("units_origin") != "unknown":
            return mesh, stats

        low, high = self.PLAUSIBLE_EXTENT_M
        if low <= stats["max_extent"] <= high:
            return mesh, stats

        choices = ["millimetres", "centimetres", "inches", "metres (leave as-is)"]
        factors = {choices[0]: 1e-3, choices[1]: 1e-2, choices[2]: 2.54e-2}

        choice, accepted = QInputDialog.getItem(
            self,
            "What unit is this drawn in?",
            f"'{Path(stats.get('name', '')).name or 'This file'}' measures "
            f"{stats['max_extent']:,.4g} along its longest axis, which is not a "
            "plausible motor in metres.\n\n"
            "Mesh formats carry no unit declaration, so this cannot be read "
            "from the file. A grain imported at the wrong scale still "
            "simulates — it simply reports a burn time wrong by orders of "
            "magnitude.\n\n"
            "The model was drawn in:",
            choices,
            0,
            False,
        )
        if not accepted or choice not in factors:
            return mesh, stats

        mesh.apply_scale(factors[choice])
        if isinstance(getattr(mesh, "metadata", None), dict):
            mesh.metadata["source_units"] = choice
            mesh.metadata["units_origin"] = "assumed"

        # Extents, volume and area all just changed, so the summary has to be
        # rebuilt rather than patched. The orientation note is not derived from
        # geometry and would be lost by a plain recompute.
        reoriented = stats.get("reoriented_from")
        stats = mesh_stats(mesh)
        stats["reoriented_from"] = reoriented
        return mesh, stats

    def _on_loaded(self, mesh, stats, path) -> None:
        stats = dict(stats, name=str(path))
        mesh, stats = self._confirm_units(mesh, stats)

        self._mesh = mesh
        self._stats = stats
        self._path = Path(path)

        self.setWindowTitle(f"{self._path.name} — {APP}")
        self.geometry_panel.set_file(self._path.name)
        self.close_action.setEnabled(True)
        self._remember_recent(self._path)
        self.geometry_panel.select_recent(str(self._path))

        self.mesh_view.show_mesh(mesh)
        # A new mesh invalidates any phi built from the previous one.
        self.field_view.clear()
        self.field_view.set_mesh_available(True)
        self._populate_data_view()
        self._refresh_measurements()
        self._refresh_grid_metrics()

        self.status_message.setText(
            f"Loaded {self._path.name} — {stats['n_triangles']:,} triangles"
        )

    def _on_failed(self, message: str) -> None:
        self.status_message.setText("Load failed")
        QMessageBox.warning(self, "Could not open mesh", message)

    # -- Building φ (#158, #175) -------------------------------------------

    def build_field(self) -> None:
        """Run the mesh through the sign field and distance, off the UI thread."""
        if self._mesh is None or self._thread is not None:
            return

        resolution = self.geometry_panel.resolution_points()
        device = self.geometry_panel.device_string()
        estimate = estimate_winding_cost(
            resolution**3, len(self._mesh.faces), device
        )
        # The cost is O(cells x triangles) and cubic in resolution, so it is
        # entirely possible to ask for an hour by nudging a slider. Warning
        # beforehand is cheaper than a cancel button that has to interrupt a
        # GPU kernel.
        if estimate > 30.0:
            answer = QMessageBox.question(
                self,
                "This will take a while",
                f"Building φ at {resolution}³ against "
                f"{len(self._mesh.faces):,} triangles is estimated at "
                f"{estimate / 60:.1f} minutes on the {device.upper()}.\n\n"
                "Lower the resolution in the Geometry panel to speed it up. "
                "Continue anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        self._set_busy(True, "Building φ...")
        self.field_view.build_button.setEnabled(False)

        self._thread = QThread(self)
        self._worker = PhiWorker(
            self._mesh,
            resolution,
            self.geometry_panel.margin_value(),
            device,
            ends=self.geometry_panel.ends_value(),
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_field_built)
        self._worker.failed.connect(self._on_field_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self._thread.start()

    def _on_field_built(self, phi, phi_outer, coords, h, stats) -> None:
        self.field_view.set_field(phi, phi_outer, coords, h, stats)
        self.field_view.build_button.setEnabled(True)
        self.tabs.setCurrentWidget(self.field_view)
        self.status_message.setText(
            f"φ built in {stats['seconds']:.2f} s — "
            f"|∇φ| = {stats['grad_mean']:.4f}"
        )

    def _on_field_failed(self, message: str) -> None:
        self.field_view.build_button.setEnabled(self._mesh is not None)
        self.status_message.setText("φ build failed")
        QMessageBox.warning(self, "Could not build φ", message)

    def close_mesh(self) -> None:
        self._mesh = None
        self._stats = None
        self._path = None
        self.setWindowTitle(APP)
        self.geometry_panel.set_file(None)
        self.close_action.setEnabled(False)
        self.mesh_view.clear()
        self.data_view.clear()
        self.field_view.clear()
        self.field_view.set_mesh_available(False)
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
        # The unit is the one import property that cannot be checked by eye
        # once the geometry is on screen: a grain drawn in millimetres looks
        # exactly like one drawn in metres. Say which it was, and how the
        # answer was arrived at (#170).
        if stats.get("units_origin") == "declared":
            messages.append(
                f"Drawn in {stats['source_units']} — the CAD file declares its "
                "unit, and the geometry has been converted to metres."
            )
        elif stats.get("units_origin") == "assumed":
            messages.append(
                f"Scaled from {stats['source_units']} to metres as you "
                "specified. This file declares no unit of its own, so that "
                "choice is not verifiable from the file."
            )
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
        if not stats.get("winding_consistent", True):
            messages.append(
                "Some faces are wound inside-out. They have been repaired on "
                "import, since inconsistent winding both mis-shades the "
                "surface and breaks the inside/outside classification."
            )
        if stats["max_extent"] > 10.0 and stats.get("units_origin") != "declared":
            messages.append(
                "Largest extent exceeds 10 m, which is not a motor. This file "
                "declares no unit — if the grain was drawn in millimetres it is "
                "~1000× too large. Re-open it and state the unit, or import a "
                "STEP file, which declares its own."
            )

        if messages:
            view.banner.show_message("  ".join(messages), "warn")
        else:
            view.banner.show_message(
                "Mesh is closed, non-degenerate, and dimensioned plausibly "
                "for metres.",
                "ok",
            )

    def _on_units_changed(self, units: str) -> None:
        """One unit choice drives the whole window, not just one panel."""
        self._settings.setValue("units", units)
        self.mesh_view.set_units(units)
        self.field_view.set_units(units)

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
            self.field_view.set_estimate(None)
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

        # The same estimate, put where the decision is actually made.
        # Reading it off a tile on another tab requires already knowing
        # to look; on the button it is unavoidable.
        self.field_view.set_estimate(cost)

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
