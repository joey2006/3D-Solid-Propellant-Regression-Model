"""Central result views for the desktop app (#157, #133).

Three tabs, matching the three stages the geometry actually passes through:
the mesh as read, the numbers that describe it, and the φ field derived from
it. Keeping them separate means a failure is attributable -- a bad φ with a
correct-looking mesh points at the sign field, not the importer.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
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

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import AsinhNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from . import theme
from .widgets import Banner, HelpButton, MetricGrid, divider, hint

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
        self._cap_actor = None
        self._body_data = None
        self._cap_data = None
        self._built_wireframe = None

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
        # Which face colouring is on show: geometric regions, or the
        # burning/inhibited labelling phi is built from (#176).
        self._colouring = "regions"
        self._ends = "inhibited"
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

        bar_layout.addWidget(divider())

        self._colour_buttons = {}
        for code, text, tip in (
            ("regions", "Regions",
             "Colour by geometry: outer wall, bore and slots, end faces."),
            ("burning", "Burning surfaces",
             "Colour by what actually burns. Orange faces are lit and are the "
             "ones φ measures distance from; grey faces are inhibited and never "
             "recede. Check this before building φ — it is the assumption the "
             "whole burn rests on."),
        ):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setToolTip(tip)
            button.clicked.connect(lambda _=False, c=code: self.set_colouring(c))
            bar_layout.addWidget(button)
            self._colour_buttons[code] = button
        self._colour_buttons["regions"].setChecked(True)

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

        self.section_button = QPushButton("Section: off")
        self.section_button.setCheckable(True)
        self.section_button.setMinimumWidth(96)
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
        self.section_slider.setValue(0)  # start intact, cut in from there
        self.section_slider.setToolTip("How much of the grain to cut away.")
        self.section_slider.valueChanged.connect(self._on_section_changed)
        section_layout.addWidget(self.section_slider, 1)

        self.section_readout = QLabel("--")
        self.section_readout.setFixedWidth(124)
        self.section_readout.setToolTip(
            "How deep the cut reaches, measured from the end it is "
            "cutting in from."
        )
        self.section_readout.setStyleSheet(
            f"color:{theme.TEXT_MUTED}; font-family:{theme.FONT_MONO};"
            "font-size:12px;"
        )
        section_layout.addWidget(self.section_readout)

        self.section_flip = QPushButton("Flip")
        self.section_flip.setToolTip("Cut in from the opposite end instead.")
        self.section_flip.clicked.connect(self._on_flip)
        section_layout.addWidget(self.section_flip)

        layout.addWidget(section_bar)
        layout.addWidget(divider())

        self._section_invert = True
        # Display units for the cut readout, kept in step with the
        # Measurements panel so the whole app never mixes units.
        self._units = "in"
        self._set_section_enabled(False)

        # Facets meeting at less than this angle are treated as one smooth
        # surface; anything sharper stays a crease. 30 degrees keeps a
        # 128-sided bore (2.8 deg per facet) round while leaving slot corners
        # and end faces crisp.
        pv.global_theme.sharp_edges_feature_angle = 30.0


        self.plotter = QtInteractor(self)
        self.plotter.set_background(theme.VIEW_BG)
        self._apply_depth_effects()

        # Dragging the slider emits far more value changes than the view can
        # draw. Without coalescing, every one queues a render and the picture
        # falls progressively further behind the handle. A zero-interval timer
        # collapses a burst into a single render once the event loop is idle,
        # so the view always draws the *latest* position rather than a backlog.
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(0)
        self._render_timer.timeout.connect(self._render)
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
        self._pv_mesh = self._classify_surface(pv.wrap(mesh), mesh)

        self._set_section_enabled(True)
        self._update_section_readout()
        self._render(reset_camera=True)

    @staticmethod
    def _classify_surface(mesh, source=None):
        """Assign every face a flat colour plus a faint roughness grain.

        Colour has to distinguish the bore from the outer wall, and neither
        back-face darkening nor ambient occlusion can do it. Back-face
        darkening fails because a bore's normals point inward toward the axis,
        so through a cutaway you see its *front* faces. Occlusion is
        screen-space and washes out at exactly the shallow viewing angles where
        the bore is hardest to find.

        Geometry answers it directly: on a grain, the outer wall's normal
        points radially outward from the motor axis and the bore's points
        radially inward. The dot product of the face normal with the outward
        radial direction is therefore +1 on the outer wall, -1 in the bore and
        ~0 on the end faces.

        This is done **per face, not per vertex**. Vertex scalars get
        interpolated across each triangle, which smears the three regions into
        a gradient -- and on an annulus, whose side walls carry vertices only at
        their end rings, that gradient covers the entire end cap. Face colours
        do not interpolate, so each region reads as one flat tone.

        A surface texture was tried here and removed: it added nothing at the
        scale the grain is viewed, and re-uploading it on every frame was a
        plausible source of flicker while sectioning.
        """
        try:
            import numpy as np

            mesh = mesh.compute_normals(
                point_normals=False, cell_normals=True, consistent_normals=True,
                auto_orient_normals=False, inplace=False,
            )
            normals = np.asarray(mesh.cell_data["Normals"])
            centres = np.asarray(mesh.cell_centers().points)

            origin = np.asarray(mesh.center)
            # Radial direction about the z axis, which is the grain axis.
            radial = centres[:, :2] - origin[:2]
            length = np.linalg.norm(radial, axis=1, keepdims=True)
            radial = np.divide(
                radial, length, out=np.zeros_like(radial), where=length > 1e-12
            )
            alignment = np.einsum("ij,ij->i", normals[:, :2], radial)

            def rgb(hex_colour):
                h = hex_colour.lstrip("#")
                return np.array(
                    [int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8
                )

            # Three flat tones, selected by a hard threshold -- no blending.
            colours = np.empty((len(alignment), 3), dtype=np.uint8)
            colours[:] = rgb(theme.CUT_FACE)                  # end faces
            colours[alignment > 0.35] = rgb(theme.SURFACE)    # outer wall
            colours[alignment < -0.35] = rgb(theme.INTERIOR)  # bore

            mesh.cell_data["grain_rgb"] = colours

            # The same faces, coloured by whether they burn (#176). This is the
            # labelling phi is actually built from, shown on the geometry
            # *before* spending seconds computing phi -- which is the point: a
            # mislabelled grain is far cheaper to notice here than after.
            #
            # The labels come from `surface_labels`, the same function the
            # field builder calls, rather than being re-derived from the
            # alignment above. That is deliberate: if this view recomputed the
            # classification it could drift from the one phi actually used, and
            # the whole value of showing it is that the picture cannot lie
            # about what will burn.
            #
            # Burning faces take the same accent as the burning-surface contour
            # on the phi tab, so the two views agree on what orange means.
            if source is not None:
                from srm_burnback.geometry.surfaces import surface_labels

                for ends in ("inhibited", "burning"):
                    labels = surface_labels(source, ends=ends)["burning"]
                    burn = np.empty((len(labels), 3), dtype=np.uint8)
                    burn[:] = rgb(theme.SURFACE)
                    burn[labels] = rgb(theme.ACCENT)
                    mesh.cell_data[f"burn_rgb_{ends}"] = burn
            # Split vertices along sharp creases so smooth shading rounds
            # only genuinely curved regions, and bake the normals in --
            # this is why the render loop can skip `split_sharp_edges`.
            return mesh.compute_normals(
                point_normals=True, cell_normals=False, split_vertices=True,
                feature_angle=30, consistent_normals=True,
                auto_orient_normals=False, inplace=False,
            )
        except Exception:
            return mesh  # colouring is cosmetic; never block the view

    def _render(self, reset_camera: bool = False) -> None:
        """Draw the cached mesh at the current style and section position.

        Actors are created once and then *updated in place*. Clearing the
        renderer and re-adding every frame tears the whole VTK pipeline down and
        rebuilds it 60 times a second, and leaves a window in which the scene is
        momentarily empty -- visible as the model flickering see-through while
        sectioning.

        Each actor owns a **persistent** ``vtkPolyData`` that is shallow-copied
        into, rather than having its mapper re-pointed at a new object every
        frame. Re-pointing a live mapper mid-scene is the fragile version of
        this idiom; copying into a stable object is the standard one.
        """
        if self.plotter is None or getattr(self, "_pv_mesh", None) is None:
            return

        mesh, cap = self._sectioned_mesh()
        wireframe = self._style == "wireframe"
        camera = None if reset_camera else self.plotter.camera_position
        show_cap = cap is not None and not wireframe and cap.n_points > 0

        # A style change alters actor properties rather than data, so it is the
        # one case that still warrants a rebuild. It happens on a click, not on
        # a drag, so the cost is irrelevant.
        if self._actor is None or self._built_wireframe != wireframe:
            self.plotter.clear()
            self._body_data = pv.PolyData()
            self._body_data.shallow_copy(mesh)
            self._actor = self.plotter.add_mesh(
                self._body_data,
                style="wireframe" if wireframe else "surface",
                scalars=None if wireframe else self._scalar_array(),
                rgb=True if not wireframe else None,
                show_scalar_bar=False,
                color=None if not wireframe else theme.SURFACE,
                line_width=1,
                reset_camera=False,
                smooth_shading=not wireframe,
                # Crease-split normals are baked in at load, so re-deriving
                # them every frame would repeat ~2.6 ms of work for no change.
                split_sharp_edges=False,
                specular=0.08,
                specular_power=8,
                # A surface seen from behind is otherwise shaded as if lit from
                # the far side -- nearly black, which against the dark
                # background looks like a hole rather than a surface. Giving
                # back faces an explicit tone means the model can never appear
                # see-through, whatever the source mesh's winding.
                backface_params=(
                    None if wireframe else {"color": theme.INTERIOR}
                ),
            )
            self.plotter.add_axes()
            self._cap_actor = None
            self._cap_data = None
            self._built_wireframe = wireframe
        else:
            self._body_data.shallow_copy(mesh)
            self._body_data.Modified()

        # The cap actor is created only once there is a real cut face to show.
        # Adding an actor backed by an empty mesh and toggling its visibility
        # works on paper but asks VTK to render a degenerate dataset, which is
        # not worth risking for the sake of symmetry.
        if show_cap:
            if self._cap_actor is None:
                self._cap_data = pv.PolyData()
                self._cap_data.shallow_copy(cap)
                self._cap_actor = self.plotter.add_mesh(
                    self._cap_data,
                    color=theme.CUT_FACE,
                    smooth_shading=False,
                    specular=0.0,
                    show_edges=False,
                    reset_camera=False,
                )
                # The cut face is coplanar with the plane the body was clipped
                # on, so the depth buffer cannot separate them and they
                # z-fight. Polygon offset biases the cap in depth only -- the
                # geometry does not move, so the cut stays where it was put.
                try:
                    mapper = self._cap_actor.GetMapper()
                    mapper.SetResolveCoincidentTopologyToPolygonOffset()
                    mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(
                        -4.0, -4.0
                    )
                except Exception:
                    pass
            else:
                self._cap_data.shallow_copy(cap)
                self._cap_data.Modified()
                self._cap_actor.SetVisibility(True)
        elif self._cap_actor is not None:
            self._cap_actor.SetVisibility(False)

        if reset_camera:
            self.plotter.view_isometric()
            self.plotter.reset_camera()
        elif camera is not None:
            self.plotter.camera_position = camera

        self.plotter.render()

    def _apply_depth_effects(self) -> None:
        """Configure anti-aliasing. Called **once**, at construction.

        This used to run on every render, where it measured 44 ms per frame --
        by far the largest cost in the loop and the reason dragging the section
        slider was choppy. It is a renderer state change, not per-frame work.
        """
        try:
            self.plotter.enable_anti_aliasing("fxaa")
        except Exception:
            pass  # purely cosmetic; never let it break the view

    # --- Sectioning -------------------------------------------------------

    def _section_axis_index(self) -> int:
        """0/1/2 for X/Y/Z. Z is listed first because it is the grain axis."""
        return {0: 2, 1: 0, 2: 1}[self.section_axis.currentIndex()]

    def _section_position(self) -> float:
        """Cut-plane coordinate for the current slider setting.

        The slider reads as *how much is cut away*, not as an absolute
        coordinate: 0 is the intact grain and 1000 is fully cut, in whichever
        direction Flip has selected. Mapping it this way means enabling Section
        never makes half the grain vanish at once, and flipping direction never
        lands on an empty view.
        """
        axis = self._section_axis_index()
        bounds = self._pv_mesh.bounds
        lo, hi = bounds[2 * axis], bounds[2 * axis + 1]
        amount = self.section_slider.value() / 1000.0
        fraction = 1.0 - amount if self._section_invert else amount
        # Keep the plane off the bounds themselves: a cut coincident with an
        # end face makes the cap and that face z-fight into speckled garbage.
        fraction = min(max(fraction, 0.002), 0.998)
        return lo + (hi - lo) * fraction

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

        # At the extremes the cut plane lands exactly on an end face, so the
        # cap and that face render coplanar and z-fight -- which shows up as
        # the surface tearing into discoloured patches. Below a hair of travel,
        # show the intact grain instead; _section_position keeps the plane
        # inset from the bounds for everything above it.
        if self.section_slider.value() <= 2:
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
            self.section_button.setText("Section: off")
            self.section_readout.setText("--")

    def _update_section_readout(self) -> None:
        if getattr(self, "_pv_mesh", None) is None:
            return
        if not self.section_button.isChecked():
            self.section_readout.setText("no cut")
            return

        # Depth of the cut, not the plane's coordinate. A raw coordinate is in
        # the mesh's own frame -- "z = 0.0507" tells the user nothing, because
        # where zero sits depends on how the part happened to be modelled.
        # How far in the cut has travelled is the thing actually being chosen.
        axis = self._section_axis_index()
        bounds = self._pv_mesh.bounds
        span = bounds[2 * axis + 1] - bounds[2 * axis]
        amount = self.section_slider.value() / 1000.0
        depth = amount * span
        if self._units == "in":
            shown = f"{depth / 0.0254:.3f} in"
        else:
            shown = f"{depth * 1000:.1f} mm"
        self.section_readout.setText(f"{shown}  ({amount:.0%})")

    def _on_section_toggled(self, checked: bool) -> None:
        for widget in (self.section_axis, self.section_slider, self.section_flip):
            widget.setEnabled(checked)
        # The button label states the mode outright. Relying on the pressed
        # look alone left it ambiguous whether sectioning was active.
        self.section_button.setText("Section: ON" if checked else "Section: off")
        self._update_section_readout()
        self._render()

    def _request_render(self) -> None:
        """Ask for a redraw, coalescing bursts into one frame."""
        if not self._render_timer.isActive():
            self._render_timer.start()

    def _on_section_changed(self) -> None:
        self._update_section_readout()
        if self.section_button.isChecked():
            self._request_render()

    def _on_flip(self) -> None:
        self._section_invert = not self._section_invert
        if self.section_button.isChecked():
            self._request_render()

    # --- View state -------------------------------------------------------

    def _scalar_array(self) -> str:
        """Which per-face colour array the renderer should show."""
        name = f"burn_rgb_{self._ends}"
        if self._colouring == "burning" and self._pv_mesh is not None:
            if name in self._pv_mesh.cell_data:
                return name
        return "grain_rgb"

    def set_colouring(self, mode: str) -> None:
        """Switch between geometric regions and the burning/inhibited labels."""
        if mode not in ("regions", "burning"):
            return
        self._colouring = mode
        for code, button in self._colour_buttons.items():
            button.setChecked(code == mode)
        # The arrays are all precomputed at load, so this is a scalar swap
        # rather than a re-classification -- forcing the rebuild path is the
        # simplest way to repoint the actor at a different array.
        self._built_wireframe = None
        self._request_render()

    def set_ends(self, ends: str) -> None:
        """Follow the End faces setting from the Geometry panel."""
        if ends not in ("inhibited", "burning") or ends == self._ends:
            return
        self._ends = ends
        if self._colouring == "burning":
            self._built_wireframe = None
            self._request_render()

    def set_style(self, style: str) -> None:
        self._style = style
        self.surface_button.setChecked(style == "surface")
        self.wireframe_button.setChecked(style == "wireframe")
        if getattr(self, "_pv_mesh", None) is not None:
            self._render()

    def set_units(self, units: str) -> None:
        """Switch the cut readout between ``"mm"`` and ``"in"``."""
        if units in ("mm", "in") and units != getattr(self, "_units", None):
            self._units = units
            if self.plotter is not None:
                self._update_section_readout()

    def reset_camera(self) -> None:
        """Return the view to exactly how a mesh looks when first opened.

        "Reset view" previously only re-framed the camera, which left a section
        cut and a wireframe toggle in place -- so it appeared to do nothing
        beyond standing the model upright. It now clears every view setting.
        """
        if self.plotter is None:
            return

        # Block signals so clearing the controls does not trigger a render per
        # widget; one render happens at the end.
        for widget in (self.section_button, self.section_axis, self.section_slider):
            widget.blockSignals(True)
        self.section_button.setChecked(False)
        self.section_axis.setCurrentIndex(0)
        self.section_slider.setValue(0)
        for widget in (self.section_button, self.section_axis, self.section_slider):
            widget.blockSignals(False)

        self._section_invert = True
        for widget in (self.section_axis, self.section_slider, self.section_flip):
            widget.setEnabled(False)

        self._style = "surface"
        self.surface_button.setChecked(True)
        self.wireframe_button.setChecked(False)

        self._update_section_readout()
        self._render(reset_camera=True)

    def clear(self) -> None:
        if self.plotter is not None:
            self.plotter.clear()
            self._mesh = None
            self._pv_mesh = None
            # Drop the actor handles too: plotter.clear() destroyed them, and a
            # stale reference would be updated instead of a fresh actor built.
            self._actor = None
            self._cap_actor = None
            self._body_data = None
            self._cap_data = None
            self._built_wireframe = None
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

        self._add_section(
            layout,
            "Topology",
            "What kind of surface arrived. 'Triangles' sets how long the "
            "sign field takes to build, since the cost is cells × "
            "triangles. 'Watertight' says whether the surface is fully "
            "closed: an open one still works — the generalized winding "
            "number degrades gracefully where ray casting would mis-sign "
            "whole regions — but volume and mass cannot be defined for it. "
            "'Degenerate' counts zero-area triangles, which have no "
            "well-defined normal.",
        )
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

        self._add_section(
            layout,
            "Extents  —  check against expected units",
            "How big the grain is along each axis, in metres. Worth a "
            "glance: this is the one place a unit mistake is visible. A "
            "grain reading 50 × 50 × 120 was drawn in millimetres and is "
            "1000× too large; 0.05 × 0.05 × 0.12 is correct. CAD files "
            "declare their unit and are converted automatically, but STL "
            "carries none.",
        )
        self.extents = MetricGrid(4)
        for key, label in (("x", "X"), ("y", "Y"), ("z", "Z")):
            self.extents.add(key, label)
        self.extents.add("volume", "Volume", "Only defined for a closed surface.")
        layout.addWidget(self.extents)

        self._add_section(
            layout,
            "Grid to be built",
            "The 3D box of sample points the simulation will run on. "
            "'Spacing h' is the distance between neighbouring points, so "
            "any feature thinner than it — a slot, a thin web, a tear in "
            "the mesh — is invisible to the solver. 'Cells across' is how "
            "many points span the grain, and 'Est. sign field' is roughly "
            "how long the import will take at this resolution.",
        )
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

    def _add_section(self, layout, text: str, help_text: str) -> None:
        """Add a section heading with a ``?`` explaining the tiles beneath it.

        The tiles carry tooltips already, but a tooltip only helps someone who
        has guessed which tile to hover. One toggle per section explains the
        whole group at once and folds away again -- these diagnostics get read
        at a glance on every import, so the prose must not sit permanently
        between the user and the numbers.
        """
        label = QLabel(text.upper())
        label.setStyleSheet(
            f"color:{theme.TEXT_FAINT}; font-size:10px; font-weight:600;"
            "letter-spacing:1.2px; margin-top:6px;"
        )

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        header_layout.addWidget(label)
        button = HelpButton()
        header_layout.addWidget(button)
        header_layout.addStretch(1)

        # The explanation goes into the parent layout rather than the header
        # row, so it reflows to the full panel width instead of being squeezed
        # in beside the heading.
        explanation = hint(help_text)
        explanation.hide()
        button.toggled.connect(explanation.setVisible)

        layout.addWidget(header)
        layout.addWidget(explanation)

    def clear(self) -> None:
        self.title.setText("No mesh loaded")
        self.banner.hide()
        for grid in (self.topology, self.extents, self.grid):
            grid.reset()


class FieldView(QWidget):
    """A slice through φ, with the |∇φ| diagnostic (#175).

    This is the panel that answers whether the geometry pipeline actually
    works. Everything before it is inspectable by eye -- a mesh either looks
    like a grain or it does not -- but φ is a volumetric field, and a subtly
    wrong one produces a simulation that runs happily to a wrong answer.

    Two things are shown, and they fail in different ways:

    * **A slice with the zero contour drawn.** Catches sign errors and
      misplaced surfaces: the contour should trace the bore and the outer wall,
      with propellant negative and void positive.
    * **|∇φ| in a band around the surface.** ``|∇φ| = 1`` is the *defining*
      property of a signed distance field, and the Godunov scheme, the CFL
      timestep and reinitialization all assume it. Phase 1 only ever verified
      it on analytically generated fields, where it holds by construction.
      Here it is measured on real imported geometry -- the case that was never
      tested.
    """

    #: Emitted when the user asks for φ to be built.
    build_requested = Signal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._phi = None
        self._phi_outer = None
        self._coords = None
        self._phi_span = 1.0
        self._phi_core = 1.0
        self._h = 0.0
        self._axis = 2
        self._mode = "phi"
        self._colorbar = None
        # Kept in step with the Measurements panel so the app never mixes units.
        self._units = "in"

        # --- toolbar ------------------------------------------------------
        bar = QWidget()
        bar.setStyleSheet(f"background:{theme.BG_BASE};")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(10, 7, 10, 7)
        bar_layout.setSpacing(6)

        self.build_button = QPushButton("Build φ")
        self.build_button.setProperty("accent", True)
        self.build_button.setEnabled(False)
        self.build_button.setToolTip(
            "Run the mesh through the winding-number sign field and "
            "closest-point distance, at the resolution set in the Geometry "
            "panel."
        )
        self.build_button.clicked.connect(self.build_requested)
        bar_layout.addWidget(self.build_button)

        show = QLabel("Show")
        show.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:12px;")
        bar_layout.addWidget(show)

        # Labelled by what they answer, not by the symbol they plot. "φ" and
        # "|∇φ|" side by side read as decoration unless you already know the
        # notation, and the second one is a quality check rather than a
        # different view of the same thing.
        self._mode_buttons = {}
        for code, text, tip in (
            ("phi", "φ  field",
             "The signed distance field itself, with the burning surface drawn "
             "as the zero contour. Blue is propellant, red is open void."),
            ("grad", "|∇φ|  check",
             "Whether φ is a valid distance field. |∇φ| must be 1 near the "
             "surface; anywhere it is not, φ is misreporting distance and the "
             "solver would inherit the error."),
        ):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setToolTip(tip)
            button.clicked.connect(lambda _=False, c=code: self.set_mode(c))
            bar_layout.addWidget(button)
            self._mode_buttons[code] = button
        self._mode_buttons["phi"].setChecked(True)

        bar_layout.addStretch(1)

        self.axis_combo = QComboBox()
        self.axis_combo.addItems(["Slice along Z", "Slice along X", "Slice along Y"])
        self.axis_combo.setFixedWidth(124)
        self.axis_combo.currentIndexChanged.connect(self._on_axis)
        bar_layout.addWidget(self.axis_combo)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setRange(0, 100)
        self.slice_slider.setValue(50)
        self.slice_slider.setFixedWidth(200)
        self.slice_slider.setToolTip("Where along the axis to cut the slice.")
        self.slice_slider.valueChanged.connect(self._redraw)
        bar_layout.addWidget(self.slice_slider)

        self.slice_readout = QLabel("--")
        self.slice_readout.setFixedWidth(92)
        self.slice_readout.setStyleSheet(
            f"color:{theme.TEXT_MUTED}; font-family:{theme.FONT_MONO};"
            "font-size:12px;"
        )
        bar_layout.addWidget(self.slice_readout)

        self.help_button = HelpButton()
        self.help_button.toggled.connect(self._set_help_visible)
        bar_layout.addWidget(self.help_button)

        layout.addWidget(bar)
        layout.addWidget(divider())

        # --- diagnostics --------------------------------------------------
        strip = QWidget()
        strip.setStyleSheet(f"background:{theme.BG_BASE};")
        strip_layout = QVBoxLayout(strip)
        strip_layout.setContentsMargins(14, 10, 14, 12)
        strip_layout.setSpacing(8)

        # Hidden behind the "?" like every other panel. This one carries more
        # weight than most: φ is the least self-explanatory thing in the app,
        # and unlike a bore diameter there is nothing familiar to fall back on.
        self._help_widgets = []
        for text in (
            "<b>φ (phi) is the signed distance field.</b> For every point in a "
            "3D grid around the grain it stores one number: how far that point "
            "is from the burning surface, signed by which side it is on — "
            "negative inside the propellant, positive in the open void, and "
            "exactly zero <i>on</i> the surface. So the burning surface is not "
            "stored as an object at all; it is wherever φ happens to be zero.",

            "<b>Why store it that way?</b> Because burning is then just "
            "arithmetic. To advance the surface you subtract the burn rate "
            "from φ everywhere and the zero crossing moves on its own — and it "
            "keeps working when the front splits in two, merges, or forms a "
            "sharp corner, all of which happen in a real grain and all of "
            "which would break a model that tracked the surface as a mesh of "
            "points.",

            "<b>The slice</b> is one flat cut through that 3D field, because a "
            "volume cannot be drawn on a screen. Pick which way to cut with "
            "the axis box and where to cut with the slider. The orange line is "
            "the zero contour — the burning surface itself, and the only line "
            "here with physical meaning. On a BATES you should see two: the "
            "bore and the outer wall.",

            "<b>The five diagnostic numbers.</b> <i>|∇φ| mean</i> is the "
            "headline — it should read 1.0000, and anything past about "
            "0.02 off means the field is not a clean distance function. "
            "<i>Spread</i> is the standard deviation of the same "
            "measurement: a low mean with a high spread means it is "
            "right on average while being wrong in specific places, "
            "which is worse than it sounds. <i>Within 1%</i> is the "
            "share of near-surface cells landing between 0.99 and "
            "1.01 — the strictest of the three, and the one that "
            "notices a small badly-behaved region the mean would "
            "average away. <i>Solid cells</i> is how much of the grid "
            "the sign field called propellant; sanity-check it against "
            "port fraction in Measurements. <i>Burning faces</i> is how "
            "many mesh triangles φ measures distance from.",

            "<b>Only the burning surfaces are measured from.</b> The grain "
            "is lit in the bore, so φ is the distance to the bore and "
            "slot walls. The outer wall is bonded to the casing and "
            "never sees flame, so it is excluded — include it and φ "
            "would turn around in the middle of the web and cross zero "
            "again at the wall, and the solver would eat the grain from "
            "the outside in. The end faces are a setting, since nothing "
            "in the geometry distinguishes a painted end from a bare "
            "one.",

            "<b>|∇φ| (the gradient magnitude)</b> measures how fast φ changes "
            "as you step across the grid. For a true distance field it must be "
            "exactly 1: move 1 mm and the distance to the surface changes by "
            "1 mm. Anywhere it is not 1, φ is lying about distance — and the "
            "solver, the timestep and the reinitialisation all assume it is "
            "telling the truth. That is why this one number is the headline "
            "check on an import.",
        ):
            label = hint(text)
            label.setTextFormat(Qt.RichText)
            label.hide()
            self._help_widgets.append(label)
            strip_layout.addWidget(label)

        self.banner = Banner()
        strip_layout.addWidget(self.banner)

        self.metrics = MetricGrid(6)
        self.metrics.add(
            "grad", "|∇φ| median",
            "The defining property of a signed distance field: should be 1. "
            "Median rather than mean, because the edges of the burning surface "
            "kink for geometric reasons and would drag a mean around.",
        )
        self.metrics.add(
            "spread", "|∇φ| spread",
            "Interquartile range in the band — how tightly the middle half of "
            "the field sits on 1.",
        )
        self.metrics.add(
            "within", "Within 1%",
            "Share of near-surface cells whose gradient is within 1% of 1.",
        )
        self.metrics.add(
            "solid", "Solid cells",
            "Share of the domain the sign field calls propellant. Compare it "
            "against the port fraction in Measurements.",
        )
        self.metrics.add(
            "burning", "Burning faces",
            "How many mesh faces φ measures distance from. The outer "
            "wall is always excluded; whether the end faces are included "
            "is the End faces setting in the Geometry panel.",
        )
        self.metrics.add("time", "Build time")
        strip_layout.addWidget(self.metrics)

        layout.addWidget(strip)
        layout.addWidget(divider())

        # --- canvas -------------------------------------------------------
        # Constrained layout rather than tight_layout: the gradient mode pairs a
        # colorbar with a second subplot, which tight_layout cannot solve and
        # warns about on every redraw.
        self._figure = Figure(
            figsize=(6, 5), facecolor=theme.VIEW_BG, layout="constrained"
        )
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._axes = self._figure.add_subplot(111)
        layout.addWidget(self._canvas, 1)

        self._empty = QLabel(
            "Open a mesh, then press Build φ.\n\n"
            "mesh  →  winding-number sign  →  closest-point distance  →  φ"
        )
        self._empty.setAlignment(Qt.AlignCenter)
        self._empty.setStyleSheet(
            f"color:{theme.TEXT_FAINT}; font-size:13px; padding:30px;"
        )
        layout.addWidget(self._empty, 1)
        self._canvas.hide()

        # Shown only in gradient mode. A correct signed distance field has a
        # dark seam running down the middle of the web, and it looks exactly
        # like a defect unless it is named -- so name it, rather than let the
        # user distrust a good import.
        self._caption = QLabel(
            "The dark seam midway through the web is the medial axis, where "
            "the nearest surface switches from the bore to the outer wall. "
            "|∇φ| is genuinely undefined on it, so this is what a correct "
            "field looks like — not a defect. The diagnostic above ignores it, "
            "measuring only the band around the surface, which is the only "
            "region the solver evaluates."
        )
        self._caption.setWordWrap(True)
        self._caption.setStyleSheet(
            f"color:{theme.TEXT_FAINT}; font-size:11px; padding:6px 16px 10px;"
        )
        self._caption.hide()
        layout.addWidget(self._caption)

    # -- state -------------------------------------------------------------

    def _set_help_visible(self, visible: bool) -> None:
        for label in self._help_widgets:
            label.setVisible(visible)

    def set_units(self, units: str) -> None:
        """Follow the app-wide unit choice (part of #154).

        Every length on this tab is converted here, at the display edge: the
        slice position, both axis scales, and the φ colour bar. The field
        itself stays in metres, so switching units can never change a result --
        only how it is written down.

        |∇φ| is deliberately *not* converted. It is a ratio of a length to a
        length, so it is dimensionless and equals 1 in any unit system; giving
        it a unit would be wrong, not just noisy.
        """
        if units not in ("mm", "in"):
            return
        self._units = units
        self._redraw()

    def _to_display(self, values):
        """Metres to the current display unit."""
        return values / 0.0254 if self._units == "in" else values * 1000.0

    def set_estimate(self, seconds: float | None) -> None:
        """Show how long a build will take, before it is started.

        The cost is ``O(cells x triangles)`` and cubic in resolution, so the
        difference between a two-second build and a twenty-minute one is two
        notches on a slider in another panel. Putting the number on the button
        makes that visible at the moment of deciding, rather than after
        committing to it.
        """
        if seconds is None:
            self.build_button.setText("Build φ")
            self.build_button.setToolTip(
                "Run the mesh through the winding-number sign field and "
                "closest-point distance. Open a mesh first."
            )
            return

        if seconds >= 60:
            pretty = f"~{seconds / 60:.0f} min"
        elif seconds >= 1:
            pretty = f"~{seconds:.0f} s"
        else:
            pretty = "<1 s"
        self.build_button.setText(f"Build φ   {pretty}")
        self.build_button.setToolTip(
            f"Estimated {pretty} at the resolution set in the Geometry panel. "
            "Cost grows with the cube of resolution and linearly with triangle "
            "count, so one notch up the slider is 8x the work."
        )

    def set_mesh_available(self, available: bool) -> None:
        self.build_button.setEnabled(available)

    def set_field(self, phi, phi_outer, coords, h: float, stats: dict) -> None:
        """Take a freshly built field and show it."""
        self._phi = phi.detach().to("cpu").numpy()
        self._phi_outer = (
            None if phi_outer is None else phi_outer.detach().to("cpu").numpy()
        )

        # Colour limits come from the whole volume, once -- never per slice.
        # Recomputing them per slice made the scale jump as the slider moved,
        # and near the ends the in-casing mask holds only a handful of cells,
        # so the limits were being set by almost nothing and the picture fell
        # apart. Fixed limits also make two slices actually comparable.
        import numpy as _np

        self._phi_span = float(_np.abs(self._phi).max()) or 1.0
        if self._phi_outer is not None and (self._phi_outer < 0).any():
            self._phi_core = (
                float(_np.abs(self._phi[self._phi_outer < 0]).max()) or self._phi_span
            )
        else:
            self._phi_core = self._phi_span
        self._coords = [c.detach().to("cpu").numpy() for c in coords]
        self._h = h

        median = stats.get("grad_median", stats["grad_mean"])
        spread = stats.get("grad_iqr", stats["grad_std"])
        self.metrics.set(
            "grad", f"{median:.4f}", "ok" if abs(median - 1.0) < 0.01 else "warn"
        )
        self.metrics.set(
            "spread", f"{spread:.4f}", "ok" if spread < 0.02 else "warn"
        )
        self.metrics.set(
            "within", f"{stats['grad_within_1pct']:.1%}",
            "ok" if stats["grad_within_1pct"] > 0.8 else "warn",
        )
        self.metrics.set("solid", f"{stats['solid_fraction']:.1%}")
        self.metrics.set(
            "burning",
            f"{stats.get('n_burning', 0):,}",
            "ok" if stats.get("n_burning") else "error",
        )
        self.metrics.set("time", f"{stats['seconds']:.2f} s")

        # The verdict, stated rather than left to be inferred from four
        # numbers. This panel exists to answer one question.
        if abs(median - 1.0) < 0.01 and spread < 0.02:
            self.banner.show_message(
                f"|∇φ| = {median:.4f} ± {spread:.4f} near the surface. This is "
                "a valid signed distance field — the import produced geometry "
                "the solver can integrate.",
                "ok",
            )
        else:
            self.banner.show_message(
                f"|∇φ| = {median:.4f} (spread {spread:.4f}) is off 1. The "
                "Godunov scheme, the CFL timestep and reinitialization all "
                "assume it is 1, so this field will not integrate cleanly.",
                "warn",
            )

        self._empty.hide()
        self._canvas.show()
        self._redraw()

    def clear(self) -> None:
        self._phi = None
        self._phi_outer = None
        self._coords = None
        self.banner.hide()
        self.metrics.reset()
        self._canvas.hide()
        self._caption.hide()
        self._empty.show()
        self.slice_readout.setText("--")

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        for code, button in self._mode_buttons.items():
            button.setChecked(code == mode)
        self._caption.setVisible(mode == "grad" and self._phi is not None)
        self._redraw()

    def _on_axis(self, index: int) -> None:
        # Combo order is Z, X, Y; tensor axes are X, Y, Z.
        self._axis = (2, 0, 1)[index]
        self._redraw()

    # -- drawing -----------------------------------------------------------

    def _histogram(self, axes, values, np) -> None:
        """Distribution of |∇φ| in the near-surface band, against the ideal 1."""
        axes.set_facecolor(theme.VIEW_BG)
        axes.hist(
            np.clip(values.ravel(), 0.7, 1.3), bins=60, range=(0.7, 1.3),
            color=theme.ACCENT, alpha=0.85,
        )
        # The target, drawn so "is the mass on the line" is a glance rather
        # than a reading of the axis.
        axes.axvline(1.0, color=theme.OK, linewidth=1.2, linestyle="--")
        outside = float((np.abs(values - 1.0) > 0.1).mean())
        axes.set_title(
            f"|∇φ| near the surface — {outside:.1%} outside ±10%",
            color=theme.TEXT_MUTED, fontsize=9,
        )
        axes.set_yticks([])
        axes.tick_params(colors=theme.TEXT_MUTED, labelsize=8)
        for spine in axes.spines.values():
            spine.set_color(theme.BORDER)

    def _redraw(self) -> None:
        if self._phi is None:
            return
        import numpy as np

        axis = self._axis
        count = self._phi.shape[axis]
        index = min(
            count - 1,
            max(0, round(self.slice_slider.value() / 100 * (count - 1))),
        )

        field = np.take(self._phi, index, axis=axis)
        position = float(np.take(self._coords[axis], index, axis=axis).flat[0])
        self.slice_readout.setText(
            f"{self._to_display(position):.3f} in"
            if self._units == "in"
            else f"{self._to_display(position):.1f} mm"
        )

        # The two in-plane axes, in tensor order, so the picture is not
        # transposed relative to the 3D view.
        plane = [i for i in range(3) if i != axis]
        horizontal = self._to_display(np.take(self._coords[plane[0]], index, axis=axis))
        vertical = self._to_display(np.take(self._coords[plane[1]], index, axis=axis))

        # Rebuilt from scratch each time rather than cleared in place: the two
        # modes need different figure layouts (the gradient view adds a
        # histogram panel), and a colorbar is itself an axes, so reusing the
        # figure means tracking and removing both. At slider speed the rebuild
        # is not measurable.
        self._figure.clear()
        gradient_mode = self._mode == "grad"

        if gradient_mode:
            grid = self._figure.add_gridspec(2, 1, height_ratios=[3, 1])
            self._axes = self._figure.add_subplot(grid[0])
            histogram = self._figure.add_subplot(grid[1])
        else:
            self._axes = self._figure.add_subplot(111)
            histogram = None
        self._axes.set_facecolor(theme.VIEW_BG)

        casing_slice = (
            None if self._phi_outer is None
            else np.take(self._phi_outer, index, axis=axis)
        )

        if gradient_mode:
            gradients = np.gradient(field, self._h)
            magnitude = np.sqrt(sum(g**2 for g in gradients))
            # Fixed 0.8-1.2 window: this is a deviation plot, so the question
            # is "how far from 1", and an autoscaled range would hide that by
            # making any field look uniform.
            mesh = self._axes.pcolormesh(
                horizontal, vertical, magnitude,
                cmap="magma", vmin=0.8, vmax=1.2, shading="auto",
            )
            label = "|∇φ|"
        else:
            shown = self._to_display(field)
            # Two competing needs. phi keeps decreasing outside the casing --
            # correctly, and by design -- so a 0.12 m box around a 0.05 m grain
            # spends most of its range on empty space, flattening the grain
            # into one blue. But hard-clipping to the grain put a false hard
            # edge exactly at the outer wall, as though the field stopped there.
            #
            # An asinh norm does both: linear across the grain, where
            # `linear_width` is the largest |phi| inside the casing, then
            # smoothly compressed beyond it. The full range stays visible and
            # the field visibly keeps evolving past the wall, with no break.
            mesh = self._axes.pcolormesh(
                horizontal, vertical, shown, cmap="RdBu_r", shading="auto",
                norm=AsinhNorm(
                    linear_width=self._to_display(self._phi_core),
                    vmin=-self._to_display(self._phi_span),
                    vmax=self._to_display(self._phi_span),
                ),
            )
            label = f"φ  ({self._units})"

        self._colorbar = self._figure.colorbar(mesh, ax=self._axes)
        self._colorbar.set_label(label, color=theme.TEXT_MUTED)
        self._colorbar.ax.tick_params(colors=theme.TEXT_MUTED, labelsize=8)

        if histogram is not None:
            # The map says *where* the field degrades; the histogram says *how
            # much of it* does. A thin spike on 1 is a clean distance field; a
            # broad shoulder means the whole field is off, which the map cannot
            # show because everything is uniformly slightly wrong.
            near_surface = field.__abs__() < 3 * self._h
            values = magnitude[near_surface] if near_surface.any() else magnitude
            self._histogram(histogram, values, np)

        # Two boundaries, drawn differently on purpose, because they are not
        # the same kind of thing.
        #
        # The zero contour of phi is the *burning surface* -- the line that
        # moves. Solid accent, the most prominent thing in the picture.
        handles, names = [], []
        if field.min() < 0.0 < field.max():
            self._axes.contour(
                horizontal, vertical, self._to_display(field), levels=[0.0],
                colors=[theme.ACCENT], linewidths=1.6,
            )
            handles.append(Line2D([], [], color=theme.ACCENT, linewidth=1.6))
            names.append("burning surface (φ = 0)")

        # The casing wall is *static*. It is deliberately not part of phi --
        # including it is what made phi turn around mid-web and cross zero a
        # second time at the wall. But it still has to be visible, or the grain
        # appears to have no outer edge at all. Dashed and muted: this line is
        # a boundary the burn stops at, not one it advances along.
        if casing_slice is not None:
            casing = casing_slice
            if casing.min() < 0.0 < casing.max():
                # Drawn identically to the bore: same colour, weight and
                # solid style. They are different kinds of boundary -- one
                # moves, one does not -- but as an outline of the grain they
                # read better matched than contrasted, and the caption below
                # carries the distinction instead.
                self._axes.contour(
                    horizontal, vertical, casing, levels=[0.0],
                    colors=[theme.ACCENT], linewidths=1.6,
                )
                names.append("casing wall (inhibited)")

        if handles:
            legend = self._axes.legend(
                handles[:1],
                ["grain boundary — inner line burns, outer is the casing"],
                loc="upper right", fontsize=8,
                facecolor=theme.BG_RAISED, edgecolor=theme.BORDER, framealpha=0.9,
            )
            for text in legend.get_texts():
                text.set_color(theme.TEXT_MUTED)

        self._axes.set_aspect("equal")
        names = "XYZ"
        self._axes.set_xlabel(
            f"{names[plane[0]]} ({self._units})", color=theme.TEXT_MUTED
        )
        self._axes.set_ylabel(
            f"{names[plane[1]]} ({self._units})", color=theme.TEXT_MUTED
        )
        self._axes.tick_params(colors=theme.TEXT_MUTED, labelsize=8)
        for spine in self._axes.spines.values():
            spine.set_color(theme.BORDER)

        self._canvas.draw_idle()
