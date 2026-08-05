"""PyInstaller hook for cascadio, the OpenCASCADE STEP reader (#177, #161).

cascadio is a compiled extension that trimesh imports *by name* only when a
STEP file is opened, so nothing in the import graph reveals it. Without this
hook a frozen build opens STL files happily and fails on STEP with a missing
module — the worst possible failure, since it looks like a format problem
rather than a packaging one.
"""

from PyInstaller.utils.hooks import collect_dynamic_libs

binaries = collect_dynamic_libs("cascadio")
hiddenimports = ["cascadio"]
