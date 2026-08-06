"""Visual theme for the desktop application (#157).

One place for every colour and the application stylesheet, so panels never
hard-code their own palette. Qt's default widget styling looks like a 2005
configuration dialog; this is what makes the app feel like a tool someone chose
to use rather than one they were handed.

The palette is built around the propellant orange already used by the 3D
previews, on a neutral dark ground so the render view -- which is the thing
users actually look at -- carries the colour instead of competing with the
chrome.
"""

from __future__ import annotations

# --- Palette --------------------------------------------------------------

BG_DEEPEST = "#141618"   # window ground, behind everything
BG_BASE = "#1b1e21"      # panel bodies
BG_RAISED = "#23272b"    # inputs, cards, tab bar
BG_HOVER = "#2c3136"
BORDER = "#31363b"
BORDER_STRONG = "#3d4349"

TEXT = "#e4e7ea"
TEXT_MUTED = "#8b949e"
TEXT_FAINT = "#6a7278"

ACCENT = "#d2743c"        # propellant orange
ACCENT_HOVER = "#e2854c"
ACCENT_PRESSED = "#b45f2d"
ACCENT_SOFT = "#3a2a20"

OK = "#5aa469"
WARN = "#d9a441"
ERROR = "#cf5f5f"

# Render-view background, kept slightly lighter than the panels so the grain
# silhouette reads clearly against it.
VIEW_BG = "#202426"

# --- Grain render palette -------------------------------------------------
# Neutral greys rather than a saturated colour: the shape is what matters, and
# grey shows shading gradients far more legibly than a strong hue, which tends
# to flatten into a single silhouette. Value descends outward -> cut -> inside,
# so depth is readable from brightness alone.
SURFACE = "#b6bbc0"    # outer skin, the lightest thing on screen
CUT_FACE = "#8f959b"   # exposed material at a section plane
INTERIOR = "#5f666c"   # back-facing geometry seen through a cutaway

FONT = "Segoe UI, Inter, system-ui, sans-serif"
FONT_MONO = "Cascadia Mono, Consolas, monospace"


def stylesheet() -> str:
    """The application-wide Qt stylesheet."""
    return f"""
* {{
    font-family: {FONT};
    font-size: 13px;
    color: {TEXT};
}}

QMainWindow, QDialog {{
    background: {BG_DEEPEST};
}}

/* --- Docks ----------------------------------------------------------- */

QDockWidget {{
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
    font-size: 11px;
    font-weight: 600;
    color: {TEXT_MUTED};
}}
QDockWidget::title {{
    background: {BG_DEEPEST};
    padding: 9px 12px 7px 12px;
    border-bottom: 1px solid {BORDER};
    text-transform: uppercase;
    letter-spacing: 1px;
}}
QDockWidget > QWidget {{
    background: {BG_BASE};
    border: 1px solid {BORDER};
}}

/* --- Group boxes ------------------------------------------------------ */

QGroupBox {{
    background: transparent;
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 20px;
    padding: 12px 10px 10px 10px;
    font-size: 11px;
    font-weight: 600;
    color: {TEXT_MUTED};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}}

/* --- Inputs ----------------------------------------------------------- */

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {BG_RAISED};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 6px 9px;
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
}}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
    border-color: {BORDER_STRONG};
    background: {BG_HOVER};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled {{
    color: {TEXT_FAINT};
    background: {BG_BASE};
}}

QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_MUTED};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background: {BG_RAISED};
    border: 1px solid {BORDER_STRONG};
    border-radius: 5px;
    padding: 4px;
    selection-background-color: {ACCENT};
    outline: none;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    width: 16px;
    background: transparent;
    border: none;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: none;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-bottom: 4px solid {TEXT_MUTED};
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: none;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-top: 4px solid {TEXT_MUTED};
}}

/* --- Buttons ---------------------------------------------------------- */

QPushButton {{
    background: {BG_RAISED};
    border: 1px solid {BORDER_STRONG};
    border-radius: 5px;
    padding: 7px 14px;
    font-weight: 500;
}}
QPushButton:hover {{
    background: {BG_HOVER};
    border-color: {ACCENT};
}}
QPushButton:pressed {{
    background: {BG_BASE};
}}
QPushButton:disabled {{
    color: {TEXT_FAINT};
    border-color: {BORDER};
    background: {BG_BASE};
}}
/* Toggle buttons must read as on or off at a glance. Without an explicit
   :checked rule a checkable QPushButton looks identical in both states. */
QPushButton:checked {{
    background: {ACCENT_SOFT};
    border: 1px solid {ACCENT};
    color: {ACCENT_HOVER};
    font-weight: 600;
}}
QPushButton:checked:hover {{
    background: {ACCENT_PRESSED};
    color: #ffffff;
}}
QPushButton[accent="true"] {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
    color: #ffffff;
    font-weight: 600;
}}
QPushButton[accent="true"]:hover {{
    background: {ACCENT_HOVER};
    border-color: {ACCENT_HOVER};
}}
QPushButton[accent="true"]:pressed {{
    background: {ACCENT_PRESSED};
}}
QPushButton[accent="true"]:disabled {{
    background: {ACCENT_SOFT};
    border-color: {ACCENT_SOFT};
    color: {TEXT_FAINT};
}}

/* --- Sliders ---------------------------------------------------------- */

QSlider::groove:horizontal {{
    height: 4px;
    background: {BG_RAISED};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}
/* The handle is drawn centred on the 4px groove, so it overhangs by half its
   own height in each direction. At 14px that was 5px of overhang against a
   -6px margin, with nothing guaranteeing the widget was tall enough to draw
   it -- so the top of the circle was clipped off. A smaller handle and an
   explicit minimum height leave room for the whole circle. */
QSlider:horizontal {{
    min-height: 18px;
}}
QSlider::handle:horizontal {{
    background: {TEXT};
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 6px;
}}
QSlider::handle:horizontal:hover {{
    background: #ffffff;
}}

/* --- Tabs ------------------------------------------------------------- */

QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    background: {BG_BASE};
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_MUTED};
    padding: 8px 18px;
    margin-right: 2px;
    border: 1px solid transparent;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-size: 12px;
}}
QTabBar::tab:hover {{
    color: {TEXT};
}}
QTabBar::tab:selected {{
    background: {BG_BASE};
    color: {TEXT};
    border-color: {BORDER};
    border-bottom-color: {BG_BASE};
}}

/* --- Menus / status --------------------------------------------------- */

QMenuBar {{
    background: {BG_DEEPEST};
    border-bottom: 1px solid {BORDER};
    padding: 2px 4px;
}}
QMenuBar::item {{
    padding: 6px 11px;
    border-radius: 4px;
    background: transparent;
}}
QMenuBar::item:selected {{
    background: {BG_RAISED};
}}
QMenu {{
    background: {BG_RAISED};
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
    padding: 5px;
}}
QMenu::item {{
    padding: 7px 26px 7px 14px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background: {ACCENT};
    color: #ffffff;
}}
QMenu::item:disabled {{
    color: {TEXT_FAINT};
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 5px 8px;
}}

QStatusBar {{
    background: {BG_DEEPEST};
    border-top: 1px solid {BORDER};
    color: {TEXT_MUTED};
    font-size: 12px;
}}
QStatusBar::item {{ border: none; }}

/* --- Misc ------------------------------------------------------------- */

QLabel[role="heading"] {{
    font-size: 15px;
    font-weight: 600;
    color: {TEXT};
}}
QLabel[role="caption"] {{
    color: {TEXT_MUTED};
    font-size: 12px;
}}
QLabel[role="mono"] {{
    font-family: {FONT_MONO};
    color: {TEXT_MUTED};
}}

QProgressBar {{
    background: {BG_RAISED};
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 3px;
}}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_STRONG};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {TEXT_FAINT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QToolTip {{
    background: {BG_RAISED};
    border: 1px solid {BORDER_STRONG};
    border-radius: 4px;
    padding: 6px 9px;
    color: {TEXT};
}}
"""
