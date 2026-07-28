"""Desktop application for SRM Burnback (#157).

A standalone PySide6 application -- the product UI. The Streamlit app under
``app/`` is a developer/test harness for the geometry pipeline (#172) and is
not the product; it will be retired once this covers the same ground.

Layering rule: this package imports ``srm_burnback``; the engine never imports
Qt, so it stays testable without a UI.
"""

__all__ = ["main"]


def main() -> int:
    """Launch the application. Imported lazily so ``import desktop`` is cheap."""
    from .app import main as _main

    return _main()
