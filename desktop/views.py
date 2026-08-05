"""Central result views for the desktop app (#157, #133).

Three tabs, matching the three stages the geometry actually passes through:
the mesh as read, the numbers that describe it, and the φ field derived from
it. Keeping them separate means a failure is attributable -- a bad φ with a
correct-looking mesh points at the sign field, not the importer.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
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
        self._pv_mesh = self._classify_surface(pv.wrap(mesh))

        self._set_section_enabled(True)
        self._update_section_readout()
        self._render(reset_camera=True)

    @staticmethod
    def _classify_surface(mesh):
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


        except Exception:
            return mesh



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
                scalars="grain_rgb" if not wireframe else None,
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
