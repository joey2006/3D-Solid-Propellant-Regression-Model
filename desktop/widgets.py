"""Small reusable widgets shared across the desktop panels (#157).

Nothing here knows anything about propellant or geometry -- these are
presentation primitives only, so panels stay declarative.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import theme


class MetricTile(QFrame):
    """A labelled value card: small caption above, prominent value below.

    Used for mesh diagnostics, where the value needs to be readable at a glance
    and its status (ok / warning / error) needs to be obvious without reading.
    """

    def __init__(self, label: str, value: str = "--", tooltip: str = ""):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._status = "neutral"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 10)
        layout.setSpacing(3)

        self._label = QLabel(label.upper())
        self._label.setStyleSheet(
            f"color:{theme.TEXT_FAINT}; font-size:10px; font-weight:600;"
            "letter-spacing:1px;"
        )

        self._value = QLabel(value)
        self._value.setStyleSheet(
            f"color:{theme.TEXT}; font-size:19px; font-weight:600;"
        )

        layout.addWidget(self._label)
        layout.addWidget(self._value)

        if tooltip:
            self.setToolTip(tooltip)
        self._apply_style()

    def _apply_style(self) -> None:
        colour = {
            "neutral": theme.BORDER,
            "ok": theme.OK,
            "warn": theme.WARN,
            "error": theme.ERROR,
        }[self._status]
        self.setStyleSheet(
            f"MetricTile {{ background:{theme.BG_RAISED};"
            f"border:1px solid {theme.BORDER};"
            f"border-left:3px solid {colour}; border-radius:6px; }}"
        )

    def set_value(self, value: str, status: str = "neutral") -> None:
        """Update the displayed value and its status accent."""
        self._value.setText(value)
        self._status = status
        self._apply_style()
        text_colour = {
            "neutral": theme.TEXT,
            "ok": theme.TEXT,
            "warn": theme.WARN,
            "error": theme.ERROR,
        }[status]
        self._value.setStyleSheet(
            f"color:{text_colour}; font-size:19px; font-weight:600;"
        )


class MetricGrid(QWidget):
    """A responsive grid of :class:`MetricTile`, addressed by key."""

    def __init__(self, columns: int = 4):
        super().__init__()
        self._columns = columns
        self._tiles: dict[str, MetricTile] = {}
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(8)

    def add(self, key: str, label: str, tooltip: str = "") -> MetricTile:
        tile = MetricTile(label, tooltip=tooltip)
        index = len(self._tiles)
        self._grid.addWidget(tile, index // self._columns, index % self._columns)
        self._tiles[key] = tile
        return tile

    def set(self, key: str, value: str, status: str = "neutral") -> None:
        self._tiles[key].set_value(value, status)

    def reset(self, value: str = "--") -> None:
        for tile in self._tiles.values():
            tile.set_value(value, "neutral")


class Banner(QFrame):
    """An inline status message. Hidden until it has something to say."""

    def __init__(self):
        super().__init__()
        self._label = QLabel()
        self._label.setWordWrap(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.addWidget(self._label)
        self.hide()

    def show_message(self, text: str, level: str = "info") -> None:
        colour = {
            "info": theme.ACCENT,
            "ok": theme.OK,
            "warn": theme.WARN,
            "error": theme.ERROR,
        }[level]
        self.setStyleSheet(
            f"Banner {{ background:{theme.BG_RAISED};"
            f"border:1px solid {theme.BORDER};"
            f"border-left:3px solid {colour}; border-radius:6px; }}"
        )
        self._label.setStyleSheet(f"color:{theme.TEXT}; font-size:12px;")
        self._label.setText(text)
        self.show()


class FieldRow(QWidget):
    """A label paired with an input control, aligned consistently."""

    def __init__(self, label: str, widget: QWidget, tooltip: str = ""):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        text = QLabel(label)
        text.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:12px;")
        text.setMinimumWidth(96)
        text.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        layout.addWidget(text)
        layout.addWidget(widget, 1)

        if tooltip:
            self.setToolTip(tooltip)
        self.widget = widget


def divider() -> QFrame:
    """A thin horizontal rule."""
    line = QFrame()
    line.setFixedHeight(1)
    line.setStyleSheet(f"background:{theme.BORDER};")
    return line
