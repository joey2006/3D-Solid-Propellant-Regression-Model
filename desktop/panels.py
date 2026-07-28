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
    QGroupBox,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .widgets import FieldRow, hint

# Powers-of-two-ish resolutions. The winding number is O(cells x triangles), so
# cost scales with the cube of this -- a free-form spin box invites a user to
# type 500 and wait an hour.
RESOLUTIONS = [32, 48, 64, 96, 128, 192, 256]


class GeometryPanel(QWidget):
    """Grain source and the grid it will be discretised onto."""

    changed = Signal()
    open_requested = Signal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # --- Source -------------------------------------------------------
        source_box = QGroupBox("Source")
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
            hint(
                "Where the shape comes from. Only imported meshes are wired "
                "up today: you supply an STL/OBJ and it becomes the grain. "
                "The parametric options are a later addition (#135)."
            )
        )

        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet(
            f"color:{theme.TEXT_FAINT}; font-size:12px; padding:2px 0;"
        )
        self.file_label.setWordWrap(True)
        source_layout.addWidget(self.file_label)

        self.open_button = QPushButton("Open mesh...")
        self.open_button.setProperty("accent", True)
        self.open_button.clicked.connect(self.open_requested)
        source_layout.addWidget(self.open_button)

        layout.addWidget(source_box)

        # --- Grid ---------------------------------------------------------
        grid_box = QGroupBox("Simulation grid")
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
        from PySide6.QtWidgets import QHBoxLayout

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
            hint(
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
            hint(
                "Empty space left around the grain, as a fraction of its size "
                "(0.05 = 5% padding). The burn front must never touch the edge "
                "of the box, where the maths stops being valid. Larger is "
                "safer but wastes resolution on empty space."
            )
        )

        layout.addWidget(grid_box)

        # --- Compute ------------------------------------------------------
        device_box = QGroupBox("Compute")
        device_layout = QVBoxLayout(device_box)
        device_layout.setSpacing(8)

        self.device = QComboBox()
        cuda = torch.cuda.is_available()
        self.device.addItems(["CUDA (GPU)", "CPU"] if cuda else ["CPU"])
        self.device.currentIndexChanged.connect(self.changed)
        device_layout.addWidget(FieldRow("Device", self.device))
        device_layout.addWidget(
            hint(
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
        vieille_box = QGroupBox("Burn rate  r = a Pⁿ")
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
            hint(
                "How fast the propellant burns, as rate = a x pressure^n. "
                "'a' sets the base speed and 'n' how strongly pressure "
                "accelerates it. n at or above 1 makes the motor unstable."
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

        layout.addWidget(vieille_box)

        # --- Erosive ------------------------------------------------------
        erosive_box = QGroupBox("Erosive burning  (Lenoir–Robillard)")
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

        box = QGroupBox("Run")
        box_layout = QVBoxLayout(box)
        box_layout.setSpacing(8)

        self.max_time = QDoubleSpinBox()
        self.max_time.setRange(0.01, 1000.0)
        self.max_time.setDecimals(2)
        self.max_time.setValue(10.0)
        self.max_time.setSuffix("  s")
        box_layout.addWidget(FieldRow("Max time", self.max_time))
        box_layout.addWidget(
            hint("How long to simulate before stopping, if the grain has not "
                 "burnt out first.")
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
            hint(
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
