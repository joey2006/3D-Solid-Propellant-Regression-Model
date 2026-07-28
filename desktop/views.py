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

        self.texture_button = QPushButton("Texture")
        self.texture_button.setCheckable(True)
        self.texture_button.setChecked(True)
        self.texture_button.setToolTip(
            "Soft surface mottling. Turn off for a perfectly smooth finish."
        )
        self.texture_button.toggled.connect(lambda _=False: self._render())
        bar_layout.addWidget(self.texture_button)

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
        self.section_readout.setFixedWidth(78)
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
        self._set_section_enabled(False)

        # Facets meeting at less than this angle are treated as one smooth
        # surface; anything sharper stays a crease. 30 degrees keeps a
        # 128-sided bore (2.8 deg per facet) round while leaving slot corners
        # and end faces crisp.
        pv.global_theme.sharp_edges_feature_angle = 30.0

        # Built once and reused: regenerating noise per render would make
        # the surface crawl as the section slider moves.
        self._texture = self._grain_texture()

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

        Surface roughness is *not* done here. Tinting each face individually
        produces stripes rather than grain, because a cylinder wall's triangles
        span its full height -- so one face is one tall thin sliver. Roughness
        comes from a texture instead; see :meth:`_apply_texture_coords`.
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
            mesh = MeshView._apply_texture_coords(mesh)
            return mesh
        except Exception:
            return mesh  # colouring is cosmetic; never block the view

    # Grain size of the surface texture, in metres of real surface per tile.
    # 4 mm across a 256 px tile is ~15 um per speck: fine sand, not gravel.
    TEXTURE_TILE = 0.004

    @staticmethod
    def _apply_texture_coords(mesh):
        """Give the mesh world-space UVs so a noise texture reads as fine sand.

        Texture coordinates are derived from position rather than from any
        parameterisation of the mesh, so the grain size is set in *metres of
        real surface* and stays constant regardless of how coarsely the object
        was tessellated. That is the whole point: per-face tinting scales with
        the triangles and streaks, a texture does not.

        Two projections, chosen per vertex from the normal:

        * end faces (normal along the axis) project onto the XY plane;
        * everything else -- outer wall, bore, slot walls -- wraps
          cylindrically, using arc length so the texture is not stretched near
          the axis.

        ``split_vertices`` runs first so vertices are duplicated along sharp
        creases. Without it a triangle could straddle the wall/cap boundary
        with its corners on different projections, which smears the texture
        across that face.
        """
        try:
            import numpy as np

            mesh = mesh.compute_normals(
                point_normals=True, cell_normals=False, split_vertices=True,
                feature_angle=30, consistent_normals=True,
                auto_orient_normals=False, inplace=False,
            )
            points = np.asarray(mesh.points)
            normals = np.asarray(mesh.point_data["Normals"])
            origin = np.asarray(mesh.center)

            x = points[:, 0] - origin[0]
            y = points[:, 1] - origin[1]
            radius = np.hypot(x, y)
            axial = np.abs(normals[:, 2]) > 0.7

            uv = np.empty((len(points), 2))
            uv[:, 0] = np.where(
                axial, x, np.arctan2(y, x) * np.maximum(radius, 1e-9)
            )
            uv[:, 1] = np.where(axial, y, points[:, 2])

            mesh.active_texture_coordinates = (
                uv / MeshView.TEXTURE_TILE
            ).astype(np.float32)
            return mesh
        except Exception:
            return mesh

    @staticmethod
    def _grain_texture():
        """A soft, seamless mottle modulated over the base colour.

        Per-pixel white noise is what makes a surface look like television
        static and is genuinely unpleasant to look at: its energy sits at the
        highest representable frequency, so once the texture is minified on
        screen it aliases and crawls under any camera movement.

        Two things fix that. The noise is **low-pass filtered in the Fourier
        domain**, which both removes the high-frequency energy and -- because
        the FFT treats the image as periodic -- leaves it seamlessly tileable,
        so no seam appears where the texture wraps. And the texture is
        **mip-mapped**, so minified areas sample a pre-filtered smaller level
        instead of undersampling the full-resolution one.

        The result reads as the soft mottling of a cast composite rather than
        as speckle. Contrast is deliberately very low.
        """
        import numpy as np
        import vtk

        size = 256
        rng = np.random.default_rng(12345)
        field = rng.normal(0.0, 1.0, (size, size))

        # Gaussian low-pass. `cutoff` is in cycles across the tile: larger is
        # finer grain. ~14 gives blobs around 18 px, which stays soft on screen.
        cutoff = 14.0
        freq = np.fft.fftfreq(size) * size
        kx, ky = np.meshgrid(freq, freq, indexing="ij")
        spectrum = np.fft.fft2(field) * np.exp(-((np.hypot(kx, ky) / cutoff) ** 2))
        smooth = np.real(np.fft.ifft2(spectrum))
        smooth /= max(float(np.abs(smooth).max()), 1e-12)

        # +/- 5 levels out of 255: visible as texture, invisible as noise.
        grey = np.clip(242.0 + smooth * 5.0, 0, 255).astype("uint8")

        texture = pv.Texture(np.repeat(grey[:, :, None], 3, axis=2))
        texture.repeat = True
        texture.interpolate = True
        texture.mipmap = True
        texture.SetBlendingMode(
            vtk.vtkTexture.VTK_TEXTURE_BLENDING_MODE_MODULATE
        )
        return texture

    def _active_texture(self):
        """The grain texture, or None when the user has switched it off."""
        return self._texture if self.texture_button.isChecked() else None

    def _render(self, reset_camera: bool = False) -> None:
        """Draw the cached mesh at the current style and section position."""
        if self.plotter is None or getattr(self, "_pv_mesh", None) is None:
            return

        mesh, cap = self._sectioned_mesh()
        wireframe = self._style == "wireframe"

        # Every render clears and re-adds the actors, and `add_mesh` resets the
        # camera whenever it is adding the first actor to a renderer -- which
        # after a clear() is always. Left alone, that re-frames the view on
        # every slider tick, so a shrinking cut appears to zoom and distort.
        # Capture the camera and put it back.
        camera = None if reset_camera else self.plotter.camera_position

        self.plotter.clear()
        if mesh is not None and mesh.n_points:
            has_region = (
                not wireframe and "grain_rgb" in getattr(mesh, "cell_data", {})
            )
            self._actor = self.plotter.add_mesh(
                mesh,
                style="wireframe" if wireframe else "surface",
                # Direct per-face RGB rather than a colour map: no scalar range
                # to interpolate, so the three regions stay flat and the
                # roughness jitter survives exactly as computed.
                scalars="grain_rgb" if has_region else None,
                rgb=has_region or None,
                # Fine noise multiplied over the flat region tones. World-space
                # UVs mean the grain stays the same physical size whatever the
                # tessellation, which is what per-face tinting could not do.
                texture=self._active_texture() if not wireframe else None,
                show_scalar_bar=False,
                color=None if has_region else theme.SURFACE,
                line_width=1,
                reset_camera=False,
                # Surface mode reads as a solid object: normals are averaged
                # across neighbouring facets so a 128-sided bore looks round
                # rather than polygonal. `split_sharp_edges` keeps that from
                # rounding off genuine creases -- slot corners and end faces
                # stay crisp, because smoothing is only applied where the angle
                # between facets is below the theme's sharp-edge feature angle
                # (set in __init__). Tessellation stays inspectable via
                # Wireframe.
                smooth_shading=not wireframe,
                # Normals (already split along sharp creases) are baked
                # into the mesh at load, so re-splitting every frame just
                # repeats work: ~2.6 ms per frame for an identical result.
                split_sharp_edges=False,
                specular=0.08,
                specular_power=8,
            )

        # The cut face gets a paler, matte treatment so it reads as exposed
        # material rather than as more outer surface -- the same convention a
        # CAD section view uses. Always flat: it is planar by construction.
        if cap is not None and not wireframe:
            cap_actor = self.plotter.add_mesh(
                cap,
                color=theme.CUT_FACE,
                smooth_shading=False,
                specular=0.0,
                show_edges=False,
                reset_camera=False,
                texture=self._active_texture(),
            )
            # The cut face lies exactly on the plane the body was clipped with,
            # so the two are coplanar and the depth buffer cannot separate them.
            # The result is z-fighting: the cap flickers away for a frame and
            # the grain appears see-through. Polygon offset biases the cap
            # fractionally toward the camera in depth only -- it does not move
            # the geometry, so the cut stays exactly where the user put it.
            try:
                mapper = cap_actor.GetMapper()
                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(
                    -2.0, -2.0
                )
            except Exception:
                pass  # depth tuning is cosmetic; never let it break the view

        self.plotter.add_axes()

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
            # The occlusion radius is a world-space distance, so it has to
            # track the model's scale: a value tuned for a 0.1 m grain does
            # nothing on a 100 mm one.
            extent = max(
                self._pv_mesh.bounds[1] - self._pv_mesh.bounds[0],
                self._pv_mesh.bounds[3] - self._pv_mesh.bounds[2],
                self._pv_mesh.bounds[5] - self._pv_mesh.bounds[4],
            )
            self.plotter.enable_anti_aliasing("fxaa")
        except Exception:
            pass  # depth cues are cosmetic; never let them break the view

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
        letter = "ZXY"[self.section_axis.currentIndex()]
        self.section_readout.setText(f"{letter} {self._section_position():.4g}")

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
        self.texture_button.blockSignals(True)
        self.texture_button.setChecked(True)
        self.texture_button.blockSignals(False)

        self._update_section_readout()
        self._render(reset_camera=True)

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
