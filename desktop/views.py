"""Central result views for the desktop app (#157, #133).

Three tabs, matching the three stages the geometry actually passes through:
the mesh as read, the numbers that describe it, and the φ field derived from
it. Keeping them separate means a failure is attributable -- a bad φ with a
correct-looking mesh points at the sign field, not the importer.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
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
        self.surface_button.setToolTip(
            "Smooth-shaded solid. Sharp creases are preserved; only genuinely "
            "curved regions are smoothed."
        )
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

        # --- Section (cutaway) strip --------------------------------------
        section_bar = QWidget()
        section_bar.setStyleSheet(f"background:{theme.BG_BASE};")
        section_layout = QHBoxLayout(section_bar)
        section_layout.setContentsMargins(10, 6, 10, 8)
        section_layout.setSpacing(8)

        self.section_button = QPushButton("Section")
        self.section_button.setCheckable(True)
        self.section_button.setToolTip(
            "Cut the grain with a plane and look inside. Drag the slider to "
            "move the cut along the chosen axis."
        )
        self.section_button.toggled.connect(self._on_section_toggled)
        section_layout.addWidget(self.section_button)

        self.section_axis = QComboBox()
        self.section_axis.addItems(["Z (length)", "X", "Y"])
        self.section_axis.setFixedWidth(104)
        self.section_axis.currentIndexChanged.connect(self._on_section_changed)
        section_layout.addWidget(self.section_axis)

        self.section_slider = QSlider(Qt.Horizontal)
        self.section_slider.setRange(0, 1000)
        self.section_slider.setValue(500)
        self.section_slider.valueChanged.connect(self._on_section_changed)
        section_layout.addWidget(self.section_slider, 1)

        self.section_readout = QLabel("--")
        self.section_readout.setFixedWidth(78)
        self.section_readout.setStyleSheet(
            f"color:{theme.TEXT_MUTED}; font-family:{theme.FONT_MONO};"
            "font-size:12px;"
        )
        section_layout.addWidget(self.section_readout)

        self.section_flip = QPushButton("Flip")
        self.section_flip.setToolTip("Keep the other half instead.")
        self.section_flip.clicked.connect(self._on_flip)
        section_layout.addWidget(self.section_flip)

        layout.addWidget(section_bar)
        layout.addWidget(divider())

        self._section_invert = True
        self._set_section_enabled(False)

        # Facets meeting at less than this angle are treated as one smooth
        # surface; anything sharper stays a crease. 30 degrees keeps a
        # 128-sided bore (2.8 deg per facet) round while leaving slot corners
        # and end faces crisp.
        pv.global_theme.sharp_edges_feature_angle = 30.0

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
        # Wrapping copies the geometry, so cache it: the section slider
        # re-renders on every tick and must not re-convert each time.
        self._pv_mesh = pv.wrap(mesh)

        self._set_section_enabled(True)
        self._update_section_readout()
        self._render(reset_camera=True)

    def _render(self, reset_camera: bool = False) -> None:
        """Draw the cached mesh at the current style and section position."""
        if self.plotter is None or getattr(self, "_pv_mesh", None) is None:
            return

        mesh, cap = self._sectioned_mesh()
        wireframe = self._style == "wireframe"

        self.plotter.clear()
        if mesh is not None and mesh.n_points:
            self._actor = self.plotter.add_mesh(
                mesh,
                style="wireframe" if wireframe else "surface",
                color=theme.ACCENT,
                line_width=1,
                # Surface mode reads as a solid object: normals are averaged
                # across neighbouring facets so a 128-sided bore looks round
                # rather than polygonal. `split_sharp_edges` keeps that from
                # rounding off genuine creases -- slot corners and end faces
                # stay crisp, because smoothing is only applied where the angle
                # between facets is below the theme's sharp-edge feature angle
                # (set in __init__). Tessellation stays inspectable via
                # Wireframe.
                smooth_shading=not wireframe,
                split_sharp_edges=not wireframe,
                specular=0.3,
                specular_power=15,
            )

        # The cut face gets a paler, matte treatment so it reads as exposed
        # material rather than as more outer surface -- the same convention a
        # CAD section view uses. Always flat: it is planar by construction.
        if cap is not None and not wireframe:
            self.plotter.add_mesh(
                cap,
                color=theme.CUT_FACE,
                smooth_shading=False,
                specular=0.0,
                show_edges=False,
            )

        self.plotter.add_axes()
        if reset_camera:
            self.plotter.view_isometric()
            self.plotter.reset_camera()
        self.plotter.render()

    # --- Sectioning -------------------------------------------------------

    def _section_axis_index(self) -> int:
        """0/1/2 for X/Y/Z. Z is listed first because it is the grain axis."""
        return {0: 2, 1: 0, 2: 1}[self.section_axis.currentIndex()]

    def _section_position(self) -> float:
        """Slider fraction mapped onto the mesh bounds along the chosen axis."""
        axis = self._section_axis_index()
        bounds = self._pv_mesh.bounds
        lo, hi = bounds[2 * axis], bounds[2 * axis + 1]
        return lo + (hi - lo) * (self.section_slider.value() / 1000.0)

    def _sectioned_mesh(self):
        """``(body, cap)`` for the current cut; ``cap`` may be ``None``.

        The cut face is built by slicing and triangulating the contour rather
        than with ``clip_closed_surface``, which refuses any non-manifold input
        -- and boolean-derived grains like the finocyl are routinely
        non-manifold. The contour route works on those too, and correctly
        leaves the bore open instead of capping the grain into a solid rod.
        """
        if not self.section_button.isChecked():
            return self._pv_mesh, None

        axis = self._section_axis_index()
        normal = [0.0, 0.0, 0.0]
        normal[axis] = 1.0
        origin = list(self._pv_mesh.center)
        origin[axis] = self._section_position()

        body = self._pv_mesh.clip(
            normal=normal, origin=origin, invert=self._section_invert
        )
        return body, self._cut_face(normal, origin)

    def _cut_face(self, normal, origin):
        """Triangulated cross-section at the cut plane, or ``None``.

        Returns ``None`` when the contour cannot be closed: a mesh with holes
        produces broken contours that triangulate to slivers or nothing, and an
        uncapped cut is the honest result there rather than a fabricated face.
        """
        try:
            contour = self._pv_mesh.slice(normal=normal, origin=origin)
            if contour.n_cells == 0:
                return None
            cap = contour.triangulate_contours()
            if cap.n_cells == 0:
                return None
            # Reject slivers by comparing against the contour's own footprint.
            b = contour.bounds
            spans = sorted(
                [b[1] - b[0], b[3] - b[2], b[5] - b[4]], reverse=True
            )[:2]
            if cap.area < 0.01 * spans[0] * spans[1]:
                return None
            return cap
        except Exception:
            return None

    def _set_section_enabled(self, enabled: bool) -> None:
        for widget in (
            self.section_button,
            self.section_axis,
            self.section_slider,
            self.section_flip,
        ):
            widget.setEnabled(enabled)
        if not enabled:
            self.section_button.setChecked(False)
            self.section_readout.setText("--")

    def _update_section_readout(self) -> None:
        if getattr(self, "_pv_mesh", None) is None:
            return
        letter = "ZXY"[self.section_axis.currentIndex()]
        self.section_readout.setText(f"{letter} {self._section_position():.4g}")

    def _on_section_toggled(self, checked: bool) -> None:
        for widget in (self.section_axis, self.section_slider, self.section_flip):
            widget.setEnabled(checked)
        self._render()

    def _on_section_changed(self) -> None:
        self._update_section_readout()
        if self.section_button.isChecked():
            self._render()

    def _on_flip(self) -> None:
        self._section_invert = not self._section_invert
        if self.section_button.isChecked():
            self._render()

    # --- View state -------------------------------------------------------

    def set_style(self, style: str) -> None:
        self._style = style
        self.surface_button.setChecked(style == "surface")
        self.wireframe_button.setChecked(style == "wireframe")
        if getattr(self, "_pv_mesh", None) is not None:
            self._render()

    def reset_camera(self) -> None:
        if self.plotter is not None:
            self.plotter.view_isometric()
            self.plotter.reset_camera()

    def clear(self) -> None:
        if self.plotter is not None:
            self.plotter.clear()
            self._mesh = None
            self._pv_mesh = None
            self._set_section_enabled(False)
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
