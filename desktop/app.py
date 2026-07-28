"""Entry point for the SRM Burnback desktop application (#157).

Run with::

    python -m desktop

The product is standalone software the user downloads and runs on their own
machine -- no server, no browser. See #157 for the decision and #177 for
packaging it into an installer.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from . import theme


def main() -> int:
    # High-DPI pixmaps keep icons and the render view crisp on scaled displays,
    # which is the common case on the laptops this will run on.
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Burnback Studio")
    app.setOrganizationName("SRM Burnback")
    app.setStyle("Fusion")  # consistent base across platforms
    app.setStyleSheet(theme.stylesheet())

    # Imported after the QApplication exists: constructing a QtInteractor
    # requires a live application instance.
    from .main_window import MainWindow

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
