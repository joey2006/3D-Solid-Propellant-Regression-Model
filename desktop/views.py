"""Central result views for the desktop app (#157, #133).

Three tabs, matching the three stages the geometry actually passes through:
the mesh as read, the numbers that describe it, and the φ field derived from
it. Keeping them separate means a failure is attributable -- a bad φ with a
correct-looking mesh points at the sign field, not the importer.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .widgets import Banner, MetricGrid, divider

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor

    HAVE_3D = True
except Exception:  # pragma: no cover - depends on local VTK/Qt install
    HAVE_3D = False


class MeshView(QWidget):
    """Interactive 3D view of the imported mesh.

    ``QtInteractor`` is a genuine ``QWidget`` rendering through VTK on the local
    GPU -- the concrete payoff of choosing a desktop framework (#157). There is
    no server, no browser, and no image round-trip.
    """

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.plotter = None
        self._actor = None

        if not HAVE_3D:
            layout.addWidget(self._unavailable())
            return

        # Toolbar strip above the render view.
        bar = QWidget()
        bar.setStyleSheet(f"background:{theme.BG_BASE};")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(10, 7, 10, 7)
        bar_layout.setSpacing(6)

        self._style = "surface"
        self.surface_button = QPushButton("Surface")
        self.wireframe_button = QPushButton("Wireframe")
        for button, style in (
            (self.surface_button, "surface"),
            (self.wireframe_button, "wireframe"),
        ):
            button.setCheckable(True)
            button.clicked.connect(lambda _=False, s=style: self.set_style(s))
            bar_layout.addWidget(button)
        self.surface_button.setChecked(True)
        self.wireframe_button.setToolTip(
            "Wireframe shows tessellation density -- compare it against the "
            "grid spacing to judge whether the mesh is resolved."
        )

        bar_layout.addStretch(1)

        reset = QPushButton("Reset view")
        reset.clicked.connect(self.reset_camera)
        bar_layout.addWidget(reset)

        layout.addWidget(bar)
        layout.addWidget(divider())

        self.plotter = QtInteractor(self)
        self.plotter.set_background(theme.VIEW_BG)
        layout.addWidget(self.plotter.interactor, 1)

        self._empty = QLabel("Open a mesh to view it here")
        self._empty.setAlignment(Qt.AlignCenter)
        self._empty.setStyleSheet(f"color:{theme.TEXT_FAINT}; font-size:13px;")
        layout.addWidget(self._empty)

    def _unavailable(self) -> QWidget:
        label = QLabel(
            "3D rendering is unavailable.\n\n"
            "pyvistaqt or VTK could not be loaded. The rest of the "
            "application is unaffected -- this panel is display only."
        )
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet(f"color:{theme.WARN}; font-size:13px; padding:40px;")
        return label

    def show_mesh(self, mesh) -> None:
        """Render a trimesh object, replacing whatever was displayed."""
        if self.plotter is None:
            return
        self._empty.hide()
        self._mesh = mesh
        self.plotter.clear()
        self._actor = self.plotter.add_mesh(
            pv.wrap(mesh),
            style="wireframe" if self._style == "wireframe" else "surface",
            color=theme.ACCENT,
            show_edges=(self._style == "surface"),
            edge_color="#7a4527",
            line_width=1,
            smooth_shading=False,  # faceting is information, not a defect
        )
        self.plotter.add_axes()
        self.plotter.view_isometric()
        self.plotter.reset_camera()

    def set_style(self, style: str) -> None:
        self._style = style
        self.surface_button.setChecked(style == "surface")
        self.wireframe_button.setChecked(style == "wireframe")
        if getattr(self, "_mesh", None) is not None:
            self.show_mesh(self._mesh)

    def reset_camera(self) -> None:
        if self.plotter is not None:
            self.plotter.view_isometric()
            self.plotter.reset_camera()

    def clear(self) -> None:
        if self.plotter is not None:
            self.plotter.clear()
            self._mesh = None
            self._empty.show()

    def close_plotter(self) -> None:
        """Release the VTK render window. Must run before the app exits."""
        if self.plotter is not None:
            self.plotter.close()


class MeshDataView(QWidget):
    """Numerical diagnostics for the imported mesh.

    Every tile here maps to a specific downstream failure mode rather than
    being trivia -- watertightness selects which sign-field path is exercised,
    extents are the only place a unit error is visible, and triangle count sets
    whether a given resolution is tractable at all.
    """

    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        scroll.setWidget(body)

        self.title = QLabel("No mesh loaded")
        self.title.setProperty("role", "heading")
        layout.addWidget(self.title)

        self.banner = Banner()
        layout.addWidget(self.banner)

        layout.addWidget(self._section("Topology"))
        self.topology = MetricGrid(4)
        self.topology.add("triangles", "Triangles", "Sets the cost of the sign field.")
        self.topology.add("vertices", "Vertices")
        self.topology.add(
            "watertight",
            "Watertight",
            "A non-watertight mesh is exactly the case the generalized "
            "winding number handles and ray casting does not.",
        )
        self.topology.add(
            "degenerate",
            "Degenerate faces",
            "Zero-area triangles subtend an undefined solid angle.",
        )
        layout.addWidget(self.topology)

        layout.addWidget(self._section("Extents  —  check against expected units"))
        self.extents = MetricGrid(4)
        for key, label in (("x", "X"), ("y", "Y"), ("z", "Z")):
            self.extents.add(key, label)
        self.extents.add("volume", "Volume", "Only defined for a closed surface.")
        layout.addWidget(self.extents)

        layout.addWidget(self._section("Grid to be built"))
        self.grid = MetricGrid(4)
        self.grid.add("cells", "Cells")
        self.grid.add("spacing", "Spacing h")
        self.grid.add(
            "across",
            "Cells across",
            "How many cells span the object. Too few and the geometry is "
            "under-resolved regardless of mesh quality.",
        )
        self.grid.add(
            "cost",
            "Est. sign field",
            "Rough time for a brute-force winding-number evaluation.",
        )
        layout.addWidget(self.grid)

        layout.addStretch(1)

    def _section(self, text: str) -> QLabel:
        label = QLabel(text.upper())
        label.setStyleSheet(
            f"color:{theme.TEXT_FAINT}; font-size:10px; font-weight:600;"
            "letter-spacing:1.2px; margin-top:6px;"
        )
        return label

    def clear(self) -> None:
        self.title.setText("No mesh loaded")
        self.banner.hide()
        for grid in (self.topology, self.extents, self.grid):
            grid.reset()


class FieldView(QWidget):
    """Placeholder for the φ field, pending #158."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(14)
        layout.addStretch(1)

        title = QLabel("Signed distance field")
        title.setProperty("role", "heading")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        body = QLabel(
            "Not implemented yet — this panel needs φ, which comes from "
            "issue #158.\n\n"
            "Once it lands, this shows a slice through the volumetric field "
            "with the zero contour drawn, plus the |∇φ| diagnostic that says "
            "whether the import produced a valid signed distance field.\n\n"
            "|∇φ| ≈ 1 is the headline check: it is the defining property of a "
            "signed distance field, and every downstream numerical method "
            "assumes it holds."
        )
        body.setAlignment(Qt.AlignCenter)
        body.setWordWrap(True)
        body.setMaximumWidth(560)
        body.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:13px;")

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(body)
        row.addStretch(1)
        layout.addLayout(row)

        pipeline = QLabel(
            "mesh  →  winding-number sign  →  closest-point distance  →  φ"
        )
        pipeline.setAlignment(Qt.AlignCenter)
        pipeline.setStyleSheet(
            f"color:{theme.ACCENT}; font-family:{theme.FONT_MONO}; font-size:12px;"
        )
        layout.addWidget(pipeline)
        layout.addStretch(2)
