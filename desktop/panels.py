"""Dockable parameter panels for the desktop app (#157, #131).

Panels are pure input: they own widgets, expose their values, and emit a signal
when something changes. They never compute anything and never reach into the
main window. That keeps the wiring one-directional -- panel emits, window
decides -- which is what makes undo (#156) tractable later.
"""

from __future__ import annotations

import torch
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import theme
from srm_burnback.propellants import (
    CUSTOM,
    LIBRARY,
    Propellant,
    by_name,
    matches,
)
from srm_burnback.units import (
    format_value,
    from_si,
    to_si,
    units_for,
)

from . import user_propellants
from .widgets import FieldRow, HelpGroup

# Powers-of-two-ish resolutions. The winding number is O(cells x triangles), so
# cost scales with the cube of this -- a free-form spin box invites a user to
# type 500 and wait an hour.
#: Density units are the one place a raw unit string reaches a widget.
_PRETTY_DENSITY = {"kg/m3": "kg/m³", "g/cm3": "g/cm³", "lb/in3": "lb/in³"}

RESOLUTIONS = [32, 48, 64, 96, 128, 192, 256]

#: Empty space kept around the grain, as a fraction of its largest extent.
#: Fixed rather than offered as a control -- see `GeometryPanel.margin_value`
#: for why there is no useful direction to move it in.
DOMAIN_MARGIN = 0.05

#: Seconds of simulated time after which a run gives up, in case the grain
#: never burns out. Not a setting: a motor that has not finished in 5 minutes
#: of burn time has something wrong with it -- no bore, or a surface that was
#: labelled inhibited when it should burn -- and the answer is to fix that, not
#: to wait longer. The Stop button covers wanting a run to end early.
RUNAWAY_CAP = 300.0


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

        # There is no "Grain type" chooser. It listed four parametric grains
        # that were all permanently greyed out, so its only working setting was
        # the one already implied by everything else in the panel: the grain is
        # an uploaded object. A control with one selectable option is not a
        # choice, and this one cost a row explaining why you could not use it.
        # When parametric grains actually arrive (#135) they need a chooser
        # again -- built then, against how they really work.
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

        # "Open mesh..." read as "open whatever is selected above", so picking
        # something from Recent and then pressing it was a surprise when a file
        # dialog appeared instead. "Upload" names the action rather than the
        # object, so it cannot be misread as acting on the Recent selection.
        self.open_button = QPushButton("Upload mesh...")
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

        layout.addWidget(grid_box)

        # --- Burning surfaces (#176) --------------------------------------
        surface_box = HelpGroup("Burning surfaces")
        surface_layout = QVBoxLayout(surface_box)
        surface_layout.setSpacing(8)

        self.ends = QComboBox()
        self.ends.addItems(["Inhibited", "Burning"])
        # Burning ends are shelved pending #194: the casing field is
        # ill-defined when the inhibited faces no longer enclose the grain, and
        # the burn never terminates. Shown rather than removed so the option is
        # visibly deferred instead of silently absent.
        self.ends.model().item(1).setEnabled(False)
        self.ends.setEnabled(False)
        self.ends.setToolTip(
            "Whether the flat end faces burn or are painted with inhibitor. "
            "Fixed at Inhibited for now — burning ends are deferred under "
            "issue #194."
        )
        self.ends.currentIndexChanged.connect(self.changed)
        surface_layout.addWidget(FieldRow("End faces", self.ends))
        surface_layout.addWidget(
            surface_box.add_help(
                "The grain is lit in the bore and burns outward, so φ measures "
                "distance from the bore and the slots — never from the outer "
                "wall, which is bonded to the casing and never sees flame. The "
                "end faces are the one part geometry cannot settle: an "
                "inhibitor-painted end looks identical to a bare one. Inhibited "
                "is the default. Treating burning ends as inhibited (or the "
                "reverse) changes burn area and therefore the whole thrust "
                "curve, so it is worth being sure."
            )
        )
        surface_layout.addWidget(
            surface_box.add_help(
                "Note this is about which surfaces are <i>lit</i>, not how fast "
                "they burn. The aft end of the bore burns faster than the fore "
                "end because combustion gas accelerates toward the nozzle and "
                "scrubs the surface — that is erosive burning, a rate effect "
                "on the bore, and it happens whether or not the ends are "
                "inhibited."
            )
        )

        layout.addWidget(surface_box)

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
        # A placeholder always sits at the top and starts selected. Without it
        # the box showed the most recent file as its current entry the moment
        # the app opened, which claimed a grain was loaded when none was. It
        # carries no path, and `_on_recent_picked` ignores entries without one,
        # so choosing it does nothing rather than trying to load "nothing".
        self.recent_combo.addItem("None selected", None)
        for entry in paths:
            self.recent_combo.addItem(Path(entry).name, entry)
            self.recent_combo.setItemData(
                self.recent_combo.count() - 1, entry, Qt.ToolTipRole
            )
        self.recent_combo.setCurrentIndex(0)
        self.recent_combo.setEnabled(bool(paths))
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
        """The domain padding, no longer a user setting.

        This was a spinbox next to Resolution, which framed it as a peer
        decision. It is not one. For an inhibited outer wall -- the only case
        the solver accepts today (#194) -- the front travels *inward* from the
        bore and the casing clamp pins the wall, so the padding is never
        consumed and turning it down changes nothing. Turning it up is worse
        than useless: resolution is fixed at N^3 either way, so padding buys
        vacuum. DOMAIN_MARGIN of 0.05 already costs about 14% of the cells
        (1.05^3), and the old 0.5 maximum would have spent over 70% of the
        grid on empty space and quietly wrecked the resolution on the grain.

        The engine keeps `margin` as a real parameter on `MeshGrain` and
        `grid_for_mesh`, because burning ends (#194) make the front move
        outward and the padding start to matter. What went away is presenting
        it as a knob to turn.
        """
        return DOMAIN_MARGIN

    def device_string(self) -> str:
        return "cuda" if self.device.currentText().startswith("CUDA") else "cpu"

    def ends_value(self) -> str:
        """How the grain's end faces should be treated: burning or inhibited."""
        return self.ends.currentText().lower()

    # -- Restoring a saved design (#155) -----------------------------------
    #
    # Each setter blocks signals while it writes, because restoring a design
    # sets several controls in a row and every one of them would otherwise
    # emit `changed`. The window issues a single refresh once the whole design
    # is in place.

    def set_resolution(self, points: int) -> None:
        """Select the nearest available resolution to ``points``."""
        if not RESOLUTIONS:
            return
        nearest = min(range(len(RESOLUTIONS)),
                      key=lambda i: abs(RESOLUTIONS[i] - int(points)))
        self.resolution.blockSignals(True)
        self.resolution.setValue(nearest)
        self.resolution.blockSignals(False)
        self.resolution_value.setText(f"{self.resolution_points()}\u00b3")

    def set_margin(self, margin: float) -> None:
        """Accept and ignore a saved margin.

        Kept as a no-op rather than deleted so `.srmd` files written while the
        spinbox existed still open. Discarding the stored value is the honest
        thing to do -- reinstating a margin the UI can no longer show would
        leave a design silently running on a setting with no way to see it,
        let alone change it back.
        """

    def set_device(self, device: str) -> None:
        """Select CUDA or CPU, falling back when the requested one is absent."""
        wanted = "CUDA" if str(device).lower().startswith("cuda") else "CPU"
        for index in range(self.device.count()):
            if self.device.itemText(index).startswith(wanted):
                self.device.blockSignals(True)
                self.device.setCurrentIndex(index)
                self.device.blockSignals(False)
                return
        # A design saved on a GPU machine, reopened on one without: keep
        # whatever is available rather than refusing to load the design.

    def set_ends(self, ends: str) -> None:
        text = "Burning" if str(ends).lower() == "burning" else "Inhibited"
        self.ends.blockSignals(True)
        self.ends.setCurrentText(text)
        self.ends.blockSignals(False)

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
        self._units = "imperial"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # --- Vieille ------------------------------------------------------
        vieille_box = HelpGroup("Burn rate  r = a Pⁿ")
        vieille_layout = QVBoxLayout(vieille_box)
        vieille_layout.setSpacing(8)

        # Named propellants (#152). Naming the choice makes it reviewable:
        # "KNSB" can be checked at a glance in a way that a = 0.00513 cannot.
        self._user_propellants = user_propellants.load()
        self.propellant = QComboBox()
        self._rebuild_propellant_list()
        self.propellant.setToolTip(
            "Published reference values — a starting point, not a "
            "characterisation of your batch. Replace them with your own static "
            "test data before flying anything."
        )
        self.propellant.activated.connect(self._on_propellant_picked)
        vieille_layout.addWidget(FieldRow("Propellant", self.propellant))
        vieille_layout.addWidget(
            vieille_box.add_help(
                "Picks a known propellant's burn-rate coefficients and density "
                "in one step. Editing any of the three numbers switches this "
                "to <i>Custom</i>, so the name can never claim something the "
                "values no longer match.<br><br>"
                "<b>These are published starting points, not your propellant.</b> "
                "Burn rate depends on oxidiser particle size, binder ratio, "
                "cure and mixing, so a static test is the only thing that says "
                "how your batch actually burns. The error matters: chamber "
                "pressure goes as a^(1/(1-n)), so 10% off in <i>a</i> is about "
                "15% off in pressure at n = 0.35."
            )
        )

        # Saving a measured propellant is the point of the warning above: once
        # a static test says what a batch actually does, those numbers need
        # somewhere to live other than three spin boxes that reset on the next
        # selection.
        propellant_buttons = QWidget()
        propellant_button_layout = QHBoxLayout(propellant_buttons)
        propellant_button_layout.setContentsMargins(0, 0, 0, 0)
        propellant_button_layout.setSpacing(6)

        self.save_propellant_button = QPushButton("Save propellant...")
        self.save_propellant_button.clicked.connect(self._save_propellant)
        propellant_button_layout.addWidget(self.save_propellant_button)

        self.delete_propellant_button = QPushButton("Delete")
        self.delete_propellant_button.setToolTip(
            "Remove the selected propellant. Only your own saved propellants "
            "can be deleted; the published reference values are fixed."
        )
        self.delete_propellant_button.clicked.connect(self._delete_propellant)
        propellant_button_layout.addWidget(self.delete_propellant_button)
        self._refresh_propellant_buttons()

        vieille_layout.addWidget(propellant_buttons)
        vieille_layout.addWidget(
            vieille_box.add_help(
                "<i>Save propellant</i> stores the three numbers below under a "
                "name of your choosing, so your own static-test data is one "
                "click away in every future design. Saved propellants are kept "
                "with your settings, not in the design file, and re-saving an "
                "existing name updates it. The published entries cannot be "
                "overwritten or deleted — a name has to keep saying where its "
                "numbers came from."
            )
        )

        self.a = QDoubleSpinBox()
        self.a.setRange(0.0001, 1.0)
        self.a.setDecimals(5)
        self.a.setSingleStep(0.001)
        self.a.setValue(0.005)
        self.a.valueChanged.connect(self._on_coefficient_changed)
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
        self.n.valueChanged.connect(self._on_coefficient_changed)
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

        # Shown in the current system but stored in SI: `density_value()`
        # always returns kg/m^3 whatever the box says, so nothing downstream
        # has to know which system is on display.
        self._density_si = 1750.0
        self.density = QDoubleSpinBox()
        self.density.setDecimals(1)
        self.density.valueChanged.connect(self._on_density_changed)
        vieille_layout.addWidget(FieldRow("Density ρ", self.density))
        self._apply_density_units()
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

    # -- Propellant library (#152) -----------------------------------------

    # -- The saved-propellant store (#152) ---------------------------------

    def _rebuild_propellant_list(self, select: str | None = None) -> None:
        """Refill the combo: published first, then the user's, then Custom.

        Published entries stay at the top because they are the ones a newcomer
        needs and the ones that never change. The selection is restored by name
        afterwards, so saving does not silently move the user somewhere else.
        """
        keep = select or self.propellant.currentText()
        self.propellant.blockSignals(True)
        self.propellant.clear()
        self.propellant.addItems([p.name for p in LIBRARY])
        if self._user_propellants:
            self.propellant.insertSeparator(self.propellant.count())
            self.propellant.addItems([p.name for p in self._user_propellants])
        self.propellant.addItem(CUSTOM)
        index = self.propellant.findText(keep)
        self.propellant.setCurrentIndex(index if index >= 0 else 0)
        self.propellant.blockSignals(False)
        self._refresh_propellant_buttons()

    def _refresh_propellant_buttons(self) -> None:
        """Delete only applies to the user's own entries."""
        # The combo is built before the buttons that sit under it, so the first
        # rebuild happens while they do not exist yet. It is refreshed again as
        # soon as they do.
        if not hasattr(self, "delete_propellant_button"):
            return
        name = self.propellant.currentText()
        own = any(p.name == name for p in self._user_propellants)
        self.delete_propellant_button.setEnabled(own)

    def _known_propellant(self, name: str):
        """Look a name up in the published library *and* the user's store."""
        for propellant in self._user_propellants:
            if propellant.name == name:
                return propellant
        return by_name(name)

    def _save_propellant(self) -> None:
        name, ok = QInputDialog.getText(
            self,
            "Save propellant",
            "Name this propellant — use something that identifies the batch, "
            "not just the type:",
            text="" if self.propellant.currentText() == CUSTOM
                 else self.propellant.currentText(),
        )
        name = name.strip()
        if not ok or not name:
            return
        if name == CUSTOM:
            QMessageBox.warning(
                self,
                "Reserved name",
                f"{CUSTOM!r} is what the app calls coefficients that match no "
                "saved propellant, so it cannot also be one. Pick another name.",
            )
            return
        if user_propellants.is_builtin(name):
            QMessageBox.warning(
                self,
                "That name is taken",
                f"{name!r} is a published reference propellant and cannot be "
                "overwritten. Its numbers are citable, and they would stop "
                "being so if your batch could be saved under the same name.\n\n"
                "Add something that identifies your batch instead.",
            )
            return

        existing = any(p.name == name for p in self._user_propellants)
        if existing:
            reply = QMessageBox.question(
                self,
                "Replace it?",
                f"You already have a propellant called {name!r}. Replace its "
                "numbers with the ones on screen?",
            )
            if reply != QMessageBox.Yes:
                return

        current = self.propellant_value()
        self._user_propellants = user_propellants.save(
            Propellant(
                name=name,
                a=current.a,
                n=current.n,
                density=current.density,
                source="Saved from this app.",
            )
        )
        self._rebuild_propellant_list(select=name)
        self.changed.emit()

    def _delete_propellant(self) -> None:
        name = self.propellant.currentText()
        if not any(p.name == name for p in self._user_propellants):
            return
        reply = QMessageBox.question(
            self,
            "Delete propellant",
            f"Delete {name!r}? The coefficients stay on screen, so nothing "
            "about the current design changes — only the saved entry goes.",
        )
        if reply != QMessageBox.Yes:
            return
        self._user_propellants = user_propellants.delete(name)
        # Land on Custom rather than whatever happens to be first: the numbers
        # on screen are still the deleted propellant's, and they now match no
        # saved entry, which is exactly what Custom means.
        self._rebuild_propellant_list(select=CUSTOM)

    def _on_propellant_picked(self, index: int) -> None:
        """Load a named propellant's numbers into the three inputs."""
        chosen = self._known_propellant(self.propellant.itemText(index))
        self._refresh_propellant_buttons()
        if chosen is None:      # "Custom" -- leave the values alone
            return
        for widget, value in (
            (self.a, chosen.a),
            (self.n, chosen.n),
        ):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self._density_si = chosen.density
        self._apply_density_units()
        self.changed.emit()

    def _sync_propellant_name(self) -> None:
        """Show ``Custom`` once the numbers stop matching the named propellant.

        A label that keeps saying "KNSB" after the coefficients have been
        edited is worse than no label: it is a claim about what is loaded, and
        it would be false.
        """
        current = self.propellant.currentText()
        chosen = self._known_propellant(current)
        if chosen is None:
            return
        if not matches(chosen, self.a.value(), self.n.value(), self._density_si):
            self.propellant.setCurrentText(CUSTOM)
            self._refresh_propellant_buttons()

    def set_propellant(
        self, name: str, a: float, n: float, density: float
    ) -> None:
        """Restore a saved propellant, values first and the name after.

        The name is set last and only if the numbers still match it, so a
        design whose coefficients were hand-edited reopens as *Custom* rather
        than claiming a library propellant it no longer is.
        """
        for widget, value in ((self.a, a), (self.n, n)):
            widget.blockSignals(True)
            widget.setValue(float(value))
            widget.blockSignals(False)
        self._density_si = float(density)
        self._apply_density_units()

        known = self._known_propellant(name)
        fits = known is not None and matches(known, float(a), float(n), float(density))
        self.propellant.blockSignals(True)
        self.propellant.setCurrentText(name if fits else CUSTOM)
        self.propellant.blockSignals(False)
        self._refresh_propellant_buttons()

    def propellant_value(self) -> Propellant:
        """The current coefficients, named if they match a library entry."""
        return Propellant(
            name=self.propellant.currentText(),
            a=float(self.a.value()),
            n=float(self.n.value()),
            density=self._density_si,
        )

    # -- Units (#154) ------------------------------------------------------

    def set_units(self, units: str) -> None:
        """Follow the app-wide unit system."""
        if units not in ("metric", "imperial") or units == self._units:
            return
        self._units = units
        self._apply_density_units()

    def _apply_density_units(self) -> None:
        """Re-label and rescale the density box for the current system.

        Signals are blocked while the displayed number changes, because this is
        the *same* density expressed differently -- emitting `changed` here
        would look to everything downstream like the user had edited it.
        """
        unit = units_for("density", self._units)
        shown = from_si(self._density_si, "density", unit)

        self.density.blockSignals(True)
        # lb/in^3 puts a propellant at ~0.063, so the metric step and decimals
        # would quantise it to nothing.
        if unit == "lb/in3":
            self.density.setDecimals(4)
            self.density.setSingleStep(0.001)
        else:
            self.density.setDecimals(1)
            self.density.setSingleStep(10.0)
        self.density.setRange(
            from_si(100.0, "density", unit), from_si(3000.0, "density", unit)
        )
        self.density.setValue(shown)
        self.density.setSuffix(f"  {_PRETTY_DENSITY.get(unit, unit)}")
        self.density.blockSignals(False)

    def _on_density_changed(self, shown: float) -> None:
        self._density_si = to_si(
            shown, "density", units_for("density", self._units)
        )
        self._sync_propellant_name()
        self.changed.emit()

    def _on_coefficient_changed(self, _value: float) -> None:
        self._sync_propellant_name()
        self.changed.emit()

    def density_value(self) -> float:
        """Propellant density in kg/m^3, whatever the display system is."""
        return self._density_si


class SimulationPanel(QWidget):
    """Run controls and stopping criteria."""

    run_requested = Signal()
    stop_requested = Signal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        box = HelpGroup("Run")
        box_layout = QVBoxLayout(box)
        box_layout.setSpacing(8)

        # Each control's explanation is added directly beneath that control.
        # One combined paragraph at the end of the box used to describe CFL, but
        # sat under the row after it -- so the help for one control appeared to
        # belong to another. Help text belongs to a control, not to a box.
        self.cfl = QDoubleSpinBox()
        self.cfl.setRange(0.05, 0.9)
        self.cfl.setDecimals(2)
        self.cfl.setSingleStep(0.05)
        self.cfl.setValue(0.4)
        box_layout.addWidget(FieldRow("Timestep safety", self.cfl))
        box_layout.addWidget(
            box.add_help(
                "How big a step the simulation takes each time it advances, as "
                "a fraction of the largest step that is still safe. The burn "
                "front must never jump more than one grid cell in a step, or "
                "the solver loses track of it and the run falls apart — 0.4 "
                "means \"use 40% of the biggest legal step\". Lower is slower "
                "but safer; the default is fine unless a run misbehaves. "
                "(Its formal name is the CFL number.)"
            )
        )

        self.reinit_every = QSpinBox()
        self.reinit_every.setRange(1, 100)
        self.reinit_every.setValue(5)
        # The unit lives in the box rather than the label, so the row reads as
        # a sentence -- "Tidy the field every [5 steps]" -- instead of leaving
        # "every 5" to mean 5 of something the reader has to guess.
        self.reinit_every.setSuffix("  steps")
        box_layout.addWidget(FieldRow("Tidy the field every", self.reinit_every))
        box_layout.addWidget(
            box.add_help(
                "As the surface moves, the stored distances slowly stop being "
                "true distances — they stretch where the surface curves and "
                "where it burns at different speeds. Left alone that error "
                "compounds and the front drifts out of position. Every few "
                "steps the field is rebuilt so the numbers mean what they "
                "claim again. More often is more accurate and slower."
            )
        )

        layout.addWidget(box)

        self.run_button = QPushButton("Run simulation")
        self.run_button.setProperty("accent", True)
        self.run_button.setEnabled(False)
        self.run_button.setToolTip(
            "Burn the imported grain and record the history. Open a mesh first."
        )
        self.run_button.clicked.connect(self.run_requested)
        layout.addWidget(self.run_button)

        # Separate from Run rather than one toggling button: a control that
        # changes what it does under the pointer is how you stop a run you
        # meant to start.
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip(
            "Finish early and keep the history so far. A stopped run is a "
            "short result, not a discarded one."
        )
        self.stop_button.clicked.connect(self.stop_requested)
        layout.addWidget(self.stop_button)

        self.summary = QLabel("No run yet.")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet(f"color:{theme.TEXT_FAINT}; font-size:11px;")
        self.summary.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.summary)

        layout.addStretch(1)

    # -- Run state (#132) --------------------------------------------------

    def set_running(self, running: bool, can_run: bool = True) -> None:
        """Swap between the idle and running control states."""
        self.run_button.setEnabled(can_run and not running)
        self.stop_button.setEnabled(running)
        self.run_button.setText("Running..." if running else "Run simulation")

    def set_summary(self, text: str, level: str = "muted") -> None:
        colour = {
            "muted": theme.TEXT_FAINT,
            "ok": theme.OK,
            "warn": theme.WARN,
        }[level]
        self.summary.setStyleSheet(f"color:{colour}; font-size:11px;")
        self.summary.setText(text)

    def config_values(self) -> dict:
        """The solver settings, as the runner's config expects them."""
        return {
            "max_time": RUNAWAY_CAP,
            "cfl_factor": float(self.cfl.value()),
            "reinit_interval": int(self.reinit_every.value()),
        }


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

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._units = "imperial"
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
        labels = (("metric", "Metric"), ("imperial", "Imperial"))
        for code, text in labels:
            button = QPushButton(text)
            button.setCheckable(True)
            button.clicked.connect(lambda _=False, c=code: self.set_units(c))
            unit_layout.addWidget(button)
            self._unit_buttons[code] = button

        # Size the pair to their own text rather than a fixed width. A
        # QPushButton centres its label and clips it at *both* ends when it
        # overflows -- it does not elide -- so a button one character too narrow
        # renders "Imperial" as "mperia", which reads as a typo rather than a
        # layout bug. Measure at the heavier weight because the :checked rule in
        # `theme` bumps the font to 600, so the selected button is the wider one
        # and sizing to the unchecked text would clip the moment it is picked.
        heavy = QFont(self.font())
        heavy.setWeight(QFont.Weight.DemiBold)
        metrics = QFontMetrics(heavy)
        # 14px stylesheet padding + 1px border a side, plus slack for the focus
        # ring, and both buttons share the widest so the toggle stays symmetric.
        width = max(metrics.horizontalAdvance(t) for _, t in labels) + 34
        for button in self._unit_buttons.values():
            button.setFixedWidth(width)
        # Imperial by default: experimental motor work is dimensioned in
        # inches, and it is the units most parts arrive in.
        self._unit_buttons["imperial"].setChecked(True)
        box_layout.addWidget(unit_row)
        box_layout.addWidget(
            box.add_help(
                "Which system every number in the app is shown in — lengths, "
                "pressure, thrust, mass, density and burn rate together, not "
                "just the dimensions here. The engine works in SI throughout "
                "and converts only at the display edge, so a unit mistake can "
                "only ever be a display mistake, and switching systems can "
                "never change a result."
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
        """Switch between ``"metric"`` and ``"imperial"`` and redraw."""
        if units not in ("metric", "imperial"):
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
        """Render a :func:`grain_measurements` result in the current system.

        Every conversion goes through :mod:`srm_burnback.units` rather than a
        local factor. Three separate copies of 0.0254 used to live in this
        file, the 3D view and the field view, which is three chances to get one
        wrong in a way that looks plausible.
        """
        self._data = data
        system = self._units

        for key in ("length", "outer_diameter", "bore_diameter", "web_thickness"):
            self._rows[key].setText(
                format_value(data[key], "length", system)
            )

        self._rows["volume"].setText(format_value(data["volume"], "volume", system))
        self._rows["mass"].setText(format_value(data["mass"], "mass", system))

        # Dimensionless, so no unit and no conversion.
        ratio = data["length_to_diameter"]
        self._rows["length_to_diameter"].setText(
            "--" if ratio is None else f"{ratio:.2f}"
        )
        port = data["port_fraction"]
        self._rows["port_fraction"].setText("--" if port is None else f"{port:.1%}")
