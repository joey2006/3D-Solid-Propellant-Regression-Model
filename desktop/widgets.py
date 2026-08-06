"""Small reusable widgets shared across the desktop panels (#157).

Nothing here knows anything about propellant or geometry -- these are
presentation primitives only, so panels stay declarative.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
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
            # Shown but not to be acted on -- a number whose interpretation is
            # still an open question reads as noise unless it is visibly
            # set aside.
            "muted": theme.BORDER,
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
            "muted": theme.TEXT_FAINT,
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


def hint(text: str) -> QLabel:
    """A small explanatory caption under a control.

    Tooltips are invisible until hovered, which is no help to someone meeting
    a control for the first time and wondering what it even is. These are
    revealed together by the ``?`` on the enclosing :class:`HelpGroup` rather
    than sitting on the panel permanently -- see that class for why.
    """
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(
        f"color:{theme.TEXT_FAINT}; font-size:11px; padding:0 0 4px 0;"
    )
    # A word-wrapped QLabel still reports its *unwrapped* text width as its
    # size hint, so revealing one inside a dock asked that dock to grow to the
    # width of the whole sentence -- stealing space from the central view, and
    # not returning it when the help was hidden again. Ignoring the horizontal
    # hint makes the label wrap into whatever width it is given instead.
    # `setHeightForWidth` has to be set with it, or the label is allotted a
    # single line's height and the wrapped remainder is clipped.
    policy = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
    policy.setHeightForWidth(True)
    label.setSizePolicy(policy)
    return label


class HelpButton(QToolButton):
    """A small circular ``?`` that toggles explanatory text on and off.

    Checkable rather than momentary, and it stays lit while help is showing, so
    the button doubles as the indicator for the state it controls -- there is
    nothing else on screen saying "help is on".
    """

    SIZE = 18

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setText("?")
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setFocusPolicy(Qt.NoFocus)
        self.setToolTip("Explain these controls")
        self.setStyleSheet(
            f"""
            QToolButton {{
                background: transparent;
                border: 1px solid {theme.BORDER_STRONG};
                border-radius: {self.SIZE // 2}px;
                color: {theme.TEXT_FAINT};
                font-size: 11px;
                font-weight: 700;
                padding: 0;
            }}
            QToolButton:hover {{
                border-color: {theme.ACCENT};
                color: {theme.ACCENT};
            }}
            QToolButton:checked {{
                background: {theme.ACCENT_SOFT};
                border-color: {theme.ACCENT};
                color: {theme.ACCENT};
            }}
            """
        )


class HelpGroup(QGroupBox):
    """A titled box whose explanatory text is hidden behind a ``?`` toggle.

    Every control in this app needs a sentence of explanation -- the domain is
    unfamiliar enough that "Domain margin" or "CFL number" means nothing on
    sight. Printing all of that permanently was honest but turned each panel
    into a wall of grey prose, and the prose crowded out the controls it was
    describing.

    So the text still exists, in full, but it is collapsed by default and
    revealed per-box. Once you know what a box does you never expand it again;
    the first time you meet it, one click explains every control at once. That
    is deliberately *not* a tooltip: a tooltip explains one control to someone
    who already suspected which control to hover.

    Help lines are children of the box's own layout, so expanding one grows
    that box and nothing else moves independently of it.
    """

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(title, parent)
        self._help_widgets: list[QWidget] = []
        self._button = HelpButton(self)
        self._button.toggled.connect(self._set_help_visible)
        self._button.raise_()

    def add_help(self, text: str) -> QLabel:
        """Add a hidden explanatory line, revealed by this box's ``?``.

        Returns the label so the caller can place it in the layout wherever it
        belongs -- usually directly under the control it describes.
        """
        label = hint(text)
        label.setVisible(self._button.isChecked())
        self._help_widgets.append(label)
        return label

    def register_help(self, widget: QWidget) -> QWidget:
        """Put an existing widget under this box's ``?`` toggle."""
        widget.setVisible(self._button.isChecked())
        self._help_widgets.append(widget)
        return widget

    def set_help_visible(self, visible: bool) -> None:
        self._button.setChecked(visible)

    def _set_help_visible(self, visible: bool) -> None:
        for widget in self._help_widgets:
            widget.setVisible(visible)

    def resizeEvent(self, event):  # noqa: N802 - Qt naming
        # QGroupBox has no corner-widget slot, so the button is positioned by
        # hand into the title row: 20 px of margin-top above the frame, with
        # the title sitting at the left. Keeping it inside that band means it
        # never collides with the controls below, whatever they are.
        #
        # y = 0 rides the top of that band. Qt clips children to the parent
        # rect, so this is as high as the button can sit without losing its
        # top edge -- the button is 18 px inside a 20 px band, and the visual
        # lift comes from the 2 px of clearance now sitting entirely below it.
        super().resizeEvent(event)
        size = self._button.width()
        self._button.move(max(0, self.width() - size - 8), 0)


def divider() -> QFrame:
    """A thin horizontal rule."""
    line = QFrame()
    line.setFixedHeight(1)
    line.setStyleSheet(f"background:{theme.BORDER};")
    return line
