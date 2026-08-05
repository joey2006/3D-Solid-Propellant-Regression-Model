"""Dockable parameter panels for the desktop app (#157, #131).

Panels are pure input: they own widgets, expose their values, and emit a signal
when something changes. They never compute anything and never reach into the
main window. That keeps the wiring one-directional -- panel emits, window
decides -- which is what makes undo (#156) tractable later.
"""

from __future__ import annotations

import torch
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .widgets import FieldRow, HelpGroup

# Powers-of-two-ish resolutions. The winding number is O(cells x triangles), so
# cost scales with the cube of this -- a free-form spin box invites a user to
# type 500 and wait an hour.
RESOLUTIONS = [32, 48, 64, 96, 128, 192, 256]


class GeometryPanel(QWidget):
    """Grain source and the grid it will be discretised onto."""

    changed = Signal()
    open_requested = Signal()
    #: Emitted with a path when the user picks a previously opened file.
    recent_selected = Signal(str)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # --- Source -------------------------------------------------------
        source_box = HelpGroup("Source")
        source_layout = QVBoxLayout(source_box)
        source_layout.setSpacing(8)

        self.grain_type = QComboBox()
        self.grain_type.addItems(
            ["Imported mesh", "BATES", "Star", "Finocyl", "Polygon"]
        )
        # Only the imported path is wired up; the parametric grains are a later
        # addition layered on top (see #135). Showing them greyed communicates
        # the roadmap without pretending they work.
        for i in range(1, self.grain_type.count()):
            self.grain_type.model().item(i).setEnabled(False)
        self.grain_type.setToolTip(
            "The product input is an uploaded object. Parametric grains are a "
            "later addition (#135)."
        )
        source_layout.addWidget(FieldRow("Grain type", self.grain_type))
        source_layout.addWidget(
            source_box.add_help(
                "Where the shape comes from. Only imported meshes are wired "
                "up today: you supply a mesh (STL/OBJ/PLY) or a CAD file "
                "(STEP/STP) and it becomes the grain. "
                "The parametric options are a later addition (#135)."
            )
        )

        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet(
            f"color:{theme.TEXT_FAINT}; font-size:12px; padding:2px 0;"
        )
        self.file_label.setWordWrap(True)
        source_layout.addWidget(self.file_label)

        # Previously opened files, so switching between them does not mean
        # navigating the file dialog again.
        self.recent_combo = QComboBox()
        self.recent_combo.setToolTip("Files you have opened before.")
        self.recent_combo.activated.connect(self._on_recent_picked)
        source_layout.addWidget(FieldRow("Recent", self.recent_combo))
        source_layout.addWidget(
            source_box.add_help(
                "Files you have opened before. Picking one re-imports it "
                "without going back through the file dialog."
            )
        )

        self.open_button = QPushButton("Open mesh...")
        self.open_button.setProperty("accent", True)
        self.open_button.clicked.connect(self.open_requested)
        source_layout.addWidget(self.open_button)

        layout.addWidget(source_box)

        # --- Grid ---------------------------------------------------------
        grid_box = HelpGroup("Simulation grid")
        grid_layout = QVBoxLayout(grid_box)
        grid_layout.setSpacing(8)

        self.resolution = QSlider(Qt.Horizontal)
        self.resolution.setRange(0, len(RESOLUTIONS) - 1)
        self.resolution.setValue(RESOLUTIONS.index(64))
        self.resolution.valueChanged.connect(self._on_resolution)
        self.resolution_value = QLabel()
        self.resolution_value.setStyleSheet(
            f"color:{theme.TEXT}; font-family:{theme.FONT_MONO}; font-size:12px;"
        )
        self.resolution_value.setMinimumWidth(78)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        row_layout.addWidget(self.resolution, 1)
        row_layout.addWidget(self.resolution_value)
        grid_layout.addWidget(
            FieldRow(
                "Resolution",
                row,
                "Grid points per axis. The grid is volumetric -- cost scales "
                "with the cube of this number.",
            )
        )
        grid_layout.addWidget(
            grid_box.add_help(
                "How finely the grain is chopped up before simulating. The "
                "grid is a 3D box of points: 64³ is 262k points, 128³ is 2.1M. "
                "Higher resolves thin webs and slot corners, but cost grows "
                "with the cube — doubling it is 8× the work."
            )
        )

        self.margin = QDoubleSpinBox()
        self.margin.setRange(0.0, 0.5)
        self.margin.setSingleStep(0.01)
        self.margin.setValue(0.05)
        self.margin.setDecimals(2)
        self.margin.valueChanged.connect(self.changed)
        grid_layout.addWidget(
            FieldRow(
                "Domain margin",
                self.margin,
                "Padding beyond the mesh so the burn front never reaches the "
                "domain edge, where the Neumann ghost cells live.",
            )
        )
        grid_layout.addWidget(
            grid_box.add_help(
                "Empty space left around the grain, as a fraction of its size "
                "(0.05 = 5% padding). The burn front must never touch the edge "
                "of the box, where the maths stops being valid. Larger is "
                "safer but wastes resolution on empty space."
            )
        )

        layout.addWidget(grid_box)

        # --- Compute ------------------------------------------------------
        device_box = HelpGroup("Compute")
        device_layout = QVBoxLayout(device_box)
        device_layout.setSpacing(8)

        self.device = QComboBox()
        cuda = torch.cuda.is_available()
        self.device.addItems(["CUDA (GPU)", "CPU"] if cuda else ["CPU"])
        self.device.currentIndexChanged.connect(self.changed)
        device_layout.addWidget(FieldRow("Device", self.device))
        device_layout.addWidget(
            device_box.add_help(
                "Where the heavy maths runs. Converting a mesh into a "
                "simulation field is embarrassingly parallel and roughly "
                "90x faster on the GPU than the CPU."
            )
        )

        detail = QLabel(
            torch.cuda.get_device_name(0)
            if cuda
            else "No CUDA device detected - running on CPU"
        )
        detail.setStyleSheet(
            f"color:{theme.OK if cuda else theme.WARN}; font-size:11px;"
        )
        detail.setWordWrap(True)
        device_layout.addWidget(detail)

        layout.addWidget(device_box)
        layout.addStretch(1)

        self._on_resolution()

    def set_recent_files(self, paths: list[str]) -> None:
        """Populate the recent-files dropdown. Paths are stored on each item."""
        from pathlib import Path

        self.recent_combo.blockSignals(True)
        self.recent_combo.clear()
        if not paths:
            self.recent_combo.addItem("(none yet)")
            self.recent_combo.setEnabled(False)
        else:
            self.recent_combo.setEnabled(True)
            for entry in paths:
                self.recent_combo.addItem(Path(entry).name, entry)
                self.recent_combo.setItemData(
                    self.recent_combo.count() - 1, entry, Qt.ToolTipRole
                )
        self.recent_combo.blockSignals(False)

    def select_recent(self, path: str) -> None:
        """Show ``path`` as the current entry without emitting a signal."""
        index = self.recent_combo.findData(path)
        if index >= 0:
            self.recent_combo.blockSignals(True)
            self.recent_combo.setCurrentIndex(index)
            self.recent_combo.blockSignals(False)

    def _on_recent_picked(self, index: int) -> None:
        path = self.recent_combo.itemData(index)
        if path:
            self.recent_selected.emit(str(path))

    def _on_resolution(self) -> None:
        n = self.resolution_points()
        self.resolution_value.setText(f"{n}³")
        self.changed.emit()

    # --- Values -----------------------------------------------------------

    def resolution_points(self) -> int:
        return RESOLUTIONS[self.resolution.value()]

    def margin_value(self) -> float:
        return float(self.margin.value())

    def device_string(self) -> str:
        return "cuda" if self.device.currentText().startswith("CUDA") else "cpu"

    def set_file(self, name: str | None) -> None:
        if name:
            self.file_label.setText(name)
            self.file_label.setStyleSheet(
                f"color:{theme.TEXT}; font-size:12px; padding:2px 0;"
            )
        else:
            self.file_label.setText("No file loaded")
            self.file_label.setStyleSheet(
                f"color:{theme.TEXT_FAINT}; font-size:12px; padding:2px 0;"
            )


class PropellantPanel(QWidget):
    """Vieille burn-rate law and erosive burning coefficients."""

    changed = Signal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # --- Vieille ------------------------------------------------------
        vieille_box = HelpGroup("Burn rate  r = a Pⁿ")
        vieille_layout = QVBoxLayout(vieille_box)
        vieille_layout.setSpacing(8)

        self.a = QDoubleSpinBox()
        self.a.setRange(0.0001, 1.0)
        self.a.setDecimals(5)
        self.a.setSingleStep(0.001)
        self.a.setValue(0.005)
        self.a.valueChanged.connect(self.changed)
        vieille_layout.addWidget(
            FieldRow(
                "Coefficient a",
                self.a,
                "Vieille's law is unit-agnostic; a ≈ 0.005 is the MPa "
                "convention, so pressure must be supplied in MPa.",
            )
        )

        self.n = QDoubleSpinBox()
        self.n.setRange(0.0, 1.0)
        self.n.setDecimals(3)
        self.n.setSingleStep(0.01)
        self.n.setValue(0.35)
        self.n.valueChanged.connect(self.changed)
        vieille_layout.addWidget(
            FieldRow("Exponent n", self.n, "Pressure exponent. n ≥ 1 is unstable.")
        )
        vieille_layout.addWidget(
            vieille_box.add_help(
                "How fast the propellant burns, as rate = a x pressure^n. "
                "'a' sets the base speed and 'n' how strongly pressure "
                "accelerates it. n at or above 1 makes the motor unstable. "
                "'a' follows the MPa convention, so pressure goes in as MPa."
            )
        )

        self.density = QDoubleSpinBox()
        self.density.setRange(100.0, 3000.0)
        self.density.setDecimals(1)
        self.density.setSingleStep(10.0)
        self.density.setValue(1750.0)
        self.density.setSuffix("  kg/m³")
        self.density.valueChanged.connect(self.changed)
        vieille_layout.addWidget(FieldRow("Density ρ", self.density))
        vieille_layout.addWidget(
            vieille_box.add_help(
                "Propellant density, used to turn the measured grain volume "
                "into a propellant mass."
            )
        )

        layout.addWidget(vieille_box)

        # --- Erosive ------------------------------------------------------
        erosive_box = HelpGroup("Erosive burning  (Lenoir–Robillard)")
        erosive_layout = QVBoxLayout(erosive_box)
        erosive_layout.setSpacing(8)

        note = QLabel(
            "Every real motor has erosive burning. Uniform rate is a "
            "validation idealisation only — never a motor prediction."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{theme.WARN}; font-size:11px;")
        erosive_layout.addWidget(note)

        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.0, 10.0)
        self.alpha.setDecimals(4)
        self.alpha.setValue(0.0)
        self.alpha.setEnabled(False)
        erosive_layout.addWidget(FieldRow("Coefficient α", self.alpha))

        self.beta = QDoubleSpinBox()
        self.beta.setRange(0.0, 1000.0)
        self.beta.setDecimals(2)
        self.beta.setValue(0.0)
        self.beta.setEnabled(False)
        erosive_layout.addWidget(FieldRow("Coefficient β", self.beta))
        erosive_layout.addWidget(
            erosive_box.add_help(
                "Combustion gas accelerates down the bore toward the nozzle, "
                "so it scrubs the surface harder at the aft end and the grain "
                "burns faster there. α scales that extra rate with the local "
                "mass flux; β damps it where the surface is already receding "
                "fast. This is why the front is never a plain cylinder."
            )
        )

        pending = QLabel("Pending the erosive burn-rate model (#13).")
        pending.setStyleSheet(f"color:{theme.TEXT_FAINT}; font-size:11px;")
        erosive_layout.addWidget(pending)

        layout.addWidget(erosive_box)
        layout.addStretch(1)


class SimulationPanel(QWidget):
    """Run controls and stopping criteria."""

    run_requested = Signal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        box = HelpGroup("Run")
        box_layout = QVBoxLayout(box)
        box_layout.setSpacing(8)

        self.max_time = QDoubleSpinBox()
        self.max_time.setRange(0.01, 1000.0)
        self.max_time.setDecimals(2)
        self.max_time.setValue(10.0)
        self.max_time.setSuffix("  s")
        box_layout.addWidget(FieldRow("Max time", self.max_time))
        box_layout.addWidget(
            box.add_help(
                "How long to simulate before stopping, if the grain has not "
                "burnt out first."
            )
        )

        self.cfl = QDoubleSpinBox()
        self.cfl.setRange(0.05, 0.9)
        self.cfl.setDecimals(2)
        self.cfl.setSingleStep(0.05)
        self.cfl.setValue(0.4)
        box_layout.addWidget(
            FieldRow(
                "CFL number",
                self.cfl,
                "Timestep safety factor. The Godunov scheme is only stable "
                "below 1.",
            )
        )

        self.reinit_every = QSpinBox()
        self.reinit_every.setRange(1, 100)
        self.reinit_every.setValue(5)
        box_layout.addWidget(
            FieldRow(
                "Reinit every",
                self.reinit_every,
                "Steps between reinitialisations, which repair |∇φ| "
                "drift caused by curvature and non-uniform burn rate.",
            )
        )
        box_layout.addWidget(
            box.add_help(
                "CFL is the timestep safety factor — smaller is slower but "
                "more stable, and above 1 the solver blows up. Reinit "
                "periodically cleans up numerical drift in the surface field."
            )
        )

        layout.addWidget(box)

        self.run_button = QPushButton("Run simulation")
        self.run_button.setProperty("accent", True)
        self.run_button.setEnabled(False)
        self.run_button.setToolTip(
            "Requires φ, which needs the winding-number sign field (#158)."
        )
        self.run_button.clicked.connect(self.run_requested)
        layout.addWidget(self.run_button)

        blocked = QLabel("Blocked on φ generation (#158).")
        blocked.setStyleSheet(f"color:{theme.TEXT_FAINT}; font-size:11px;")
        blocked.setAlignment(Qt.AlignCenter)
        layout.addWidget(blocked)

        layout.addStretch(1)


class MeasurementsPanel(QWidget):
    """Engineering dimensions of the loaded grain, in the user's units.

    The engine works in metres and kilograms throughout. Conversion happens
    here, at the very edge, so nothing upstream ever has to know which units
    are on display -- and a unit bug can only ever be a display bug.

    Imperial is a first-class option rather than a nicety: amateur and
    experimental motor work is largely dimensioned in inches, and reading a
    0.2 in bore as "5.07 mm" makes an exact design number look like a
    measurement error. (Part of #154.)
    """

    #: Emitted when the user switches units, so the choice can be persisted.
    units_changed = Signal(str)

    # Exact by definition: the international inch is 25.4 mm.
    _M_PER_INCH = 0.0254
    _LB_PER_KG = 2.2046226218

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._units = "in"
        self._data: dict | None = None

        box = HelpGroup("Grain dimensions")
        box_layout = QVBoxLayout(box)
        box_layout.setSpacing(8)
        layout.addWidget(box)

        unit_row = QWidget()
        unit_layout = QHBoxLayout(unit_row)
        unit_layout.setContentsMargins(0, 0, 0, 0)
        unit_layout.setSpacing(6)
        label = QLabel("Units")
        label.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:12px;")
        unit_layout.addWidget(label)
        unit_layout.addStretch(1)

        self._unit_buttons = {}
        for code, text in (("mm", "mm"), ("in", "inch")):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setFixedWidth(56)
            button.clicked.connect(lambda _=False, c=code: self.set_units(c))
            unit_layout.addWidget(button)
            self._unit_buttons[code] = button
        # Imperial by default: experimental motor work is dimensioned in
        # inches, and it is the units most parts arrive in.
        self._unit_buttons["in"].setChecked(True)
        box_layout.addWidget(unit_row)
        box_layout.addWidget(
            box.add_help(
                "Which units these are shown in. The engine works in metres "
                "throughout and converts only here, at the very edge, so a "
                "unit mistake can only ever be a display mistake."
            )
        )

        # Each row is a name/value pair with its explanation on the line
        # underneath. The explanation is hidden until the box's "?" is on:
        # "Web" and "Port fraction" mean nothing on first sight, but once you
        # know them the prose is pure clutter sitting between you and the
        # numbers you came to read.
        self._rows: dict[str, QLabel] = {}
        for key, text, tip in (
            ("length", "Length", "End to end along the grain axis."),
            ("outer_diameter", "Outer dia.",
             "Across the outer wall — the widest the grain gets, which is what "
             "has to fit inside the casing."),
            ("bore_diameter", "Bore dia.",
             "Across the narrowest inward-facing surface: the hole up the "
             "middle that the exhaust flows through and that the burn starts "
             "from."),
            ("web_thickness", "Web",
             "Propellant between the bore and the outer wall — the thickness "
             "that has to burn through, so it sets the burn time."),
            ("length_to_diameter", "L/D",
             "Length divided by outer diameter. A long, slender grain has more "
             "gas accelerating down its bore, so it burns harder at the aft "
             "end."),
            ("volume", "Volume",
             "How much propellant there is. Needs a closed surface, so it "
             "reads '--' for a mesh with holes in it."),
            ("mass", "Mass",
             "Volume times the density set in the Propellant panel. This is "
             "the propellant load — what determines total impulse."),
            ("port_fraction", "Port fraction",
             "Share of the grain's envelope that is empty space. Near zero "
             "means no bore was found, which usually points at a winding or "
             "orientation problem rather than a solid grain."),
        ):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            name = QLabel(text)
            name.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:12px;")
            value = QLabel("--")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value.setStyleSheet(
                f"color:{theme.TEXT}; font-family:{theme.FONT_MONO}; font-size:13px;"
            )
            row_layout.addWidget(name)
            row_layout.addStretch(1)
            row_layout.addWidget(value)
            row.setToolTip(tip)
            box_layout.addWidget(row)
            box_layout.addWidget(box.add_help(tip))
            self._rows[key] = value

        box_layout.addWidget(
            box.add_help(
                "All of these are measured from the imported geometry, not "
                "entered by hand — an uploaded object arrives as triangles "
                "with no parameters attached, so the bore and web are "
                "recovered from the surface normals."
            )
        )
        layout.addStretch(1)

    # -- Units -------------------------------------------------------------

    def units(self) -> str:
        return self._units

    def set_units(self, units: str, notify: bool = True) -> None:
        """Switch between ``"mm"`` and ``"in"`` and redraw the current values."""
        if units not in ("mm", "in"):
            return
        self._units = units
        for code, button in self._unit_buttons.items():
            button.setChecked(code == units)
        if self._data is not None:
            self.set_measurements(self._data)
        if notify:
            self.units_changed.emit(units)

    # -- Values ------------------------------------------------------------

    def clear(self) -> None:
        self._data = None
        for value in self._rows.values():
            value.setText("--")

    def set_measurements(self, data: dict) -> None:
        """Render a :func:`grain_measurements` result in the current units."""
        self._data = data
        imperial = self._units == "in"

        def length(x):
            if x is None:
                return "--"
            if imperial:
                # 3 dp is 0.001 in = 25 um, finer than any real
                # tolerance, and avoids showing faceting noise.
                return f"{x / self._M_PER_INCH:.3f} in"
            return f"{x * 1000:.2f} mm"

        for key in ("length", "outer_diameter", "bore_diameter", "web_thickness"):
            self._rows[key].setText(length(data[key]))

        ld = data["length_to_diameter"]
        self._rows["length_to_diameter"].setText("--" if ld is None else f"{ld:.2f}")

        volume = data["volume"]
        if volume is None:
            self._rows["volume"].setText("--")
        elif imperial:
            self._rows["volume"].setText(
                f"{volume / self._M_PER_INCH ** 3:.3f} in³"
            )
        else:
            self._rows["volume"].setText(f"{volume * 1e6:.1f} cm³")

        mass = data["mass"]
        if mass is None:
            self._rows["mass"].setText("--")
        elif imperial:
            pounds = mass * self._LB_PER_KG
            self._rows["mass"].setText(
                f"{pounds * 16:.1f} oz" if pounds < 1.0 else f"{pounds:.3f} lb"
            )
        elif mass < 1.0:
            self._rows["mass"].setText(f"{mass * 1000:.0f} g")
        else:
            self._rows["mass"].setText(f"{mass:.3f} kg")

        port = data["port_fraction"]
        self._rows["port_fraction"].setText("--" if port is None else f"{port:.1%}")
