# PyInstaller spec for Burnback Studio (#177).
#
# Build with:   pyinstaller packaging/burnback_studio.spec --noconfirm
#
# Two things make freezing this application harder than a typical Qt app, and
# both are handled explicitly below rather than left to dependency analysis:
#
#   VTK   ships large native libraries loaded at runtime by name. PyInstaller's
#         static analysis does not see those imports, so the modules have to be
#         named. Symptom when wrong: the window opens and the 3D tab is blank
#         or the process dies on first render.
#
#   torch ships its own native runtime and, in a CUDA build, several gigabytes
#         of kernels. Collecting it wholesale is slow and enormous; collecting
#         too little produces an app that imports and then fails the moment a
#         tensor is allocated.
#
# The CPU/GPU decision is made by BURNBACK_CUDA (see packaging/README.md).
# There is no way to make one build that is both small and CUDA-capable.

import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

BUNDLE_CUDA = os.environ.get("BURNBACK_CUDA", "0") == "1"

datas, binaries, hiddenimports = [], [], []

# VTK and pyvista: take everything. Their runtime loading defeats analysis, and
# a partial collection fails at render time rather than at build time, which is
# far more expensive to diagnose.
for package in ("vtkmodules", "pyvista", "pyvistaqt"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

# trimesh loads format handlers lazily by name, so importers for STL/OBJ/PLY
# are invisible to analysis and must be named.
hiddenimports += collect_submodules("trimesh")
hiddenimports += ["cascadio", "scipy.spatial.transform._rotation_groups"]

# matplotlib's Qt backend is selected by string at runtime.
hiddenimports += ["matplotlib.backends.backend_qtagg"]

torch_datas, torch_binaries, torch_hidden = collect_all("torch")
datas += torch_datas
hiddenimports += torch_hidden
if BUNDLE_CUDA:
    binaries += torch_binaries
else:
    # Drop the CUDA kernels, which are the bulk of the download and useless
    # without an NVIDIA card. The app already falls back to CPU at runtime.
    binaries += [
        entry for entry in torch_binaries
        if "cuda" not in entry[0].lower() and "cudnn" not in entry[0].lower()
    ]

a = Analysis(
    ["../desktop/__main__.py"],
    pathex=[".."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=["packaging/hooks"],
    excludes=[
        # Test frameworks and notebook machinery pulled in transitively.
        "pytest", "IPython", "jupyter", "notebook", "tkinter",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BurnbackStudio",
    debug=False,
    strip=False,
    upx=False,          # UPX-packed binaries are flagged by antivirus far more
    console=False,      # a GUI app; a console window would be noise
)

COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="BurnbackStudio",
)
