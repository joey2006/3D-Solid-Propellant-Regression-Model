# Packaging Burnback Studio (#177)

Building a downloadable Windows application, so a user needs no Python, no
PyTorch and no CUDA.

## Status

**Step 1 of #177 passes; the full build is not yet verified.**

A minimal PySide6 window was frozen with PyInstaller 6.21.0 and the resulting
`QtSpike.exe` ran, loading its Qt platform plugin from the bundle. So the
toolchain works on this machine and the hardest-to-diagnose class of failure —
Qt plugins not being found in a frozen build — does not occur by default.

For scale: **118 MB for a bare Qt window**, before torch, VTK or trimesh. That
is the floor, and it is why the CPU/GPU split below matters.

Outstanding: the full application build, and the only test that means anything
— a machine with no Python installed. Do not treat this as a finished release
process.

### Build in a short path

The first spike failed like this:

```
FileNotFoundError: ...\dist\QtSpike\_internal\PySide6\plugins\
    platforminputcontexts\qtvirtualkeyboardplugin.dll
```

That is Windows' 260-character path limit, not a missing file. PyInstaller
nests deeply (`dist/<name>/_internal/<package>/...`), so a project checked out
somewhere long — a OneDrive folder, say — overflows it partway through
collecting Qt. The same build from `C:\tmp\` succeeded immediately.

**Build from a short path**, or enable long paths system-wide
(`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1`).
The failure names a plugin DLL and looks like a PySide6 problem, which is what
makes it expensive to recognise.

## One build or two

**Two.** The question in #177 is settled as follows, and the reasoning is worth
keeping because it is the main cost driver:

A CUDA-enabled PyTorch is roughly 2–3 GB unpacked, and it is dead weight for
anyone without an NVIDIA card. A CPU-only build is a few hundred megabytes and
runs everywhere. Most people evaluating the tool will import a grain, look at
it, and build φ once at a modest resolution — which a CPU handles in seconds to
a minute.

The GPU matters for the winding number specifically, where it is worth about
**90×** (the test suite: 61 s on CUDA against 53 minutes on CPU for the same
work). That is decisive for someone running 256³ imports repeatedly, and
irrelevant for someone looking at a grain.

So: a small **CPU build** as the default download, and a **CUDA build** for
people who will use it. The alternative — one build that fetches CUDA on first
run — is better for users and considerably more machinery, and is not worth
building before anyone is actually using the tool.

```
                    approx. size    who it is for
CPU build           ~400 MB         everyone; the default download
CUDA build          ~3 GB           NVIDIA owners doing repeated high-res work
```

## Building

```powershell
pip install pyinstaller
pip install -e ".[desktop]"

# CPU build (default)
pyinstaller packaging/burnback_studio.spec --noconfirm

# CUDA build
$env:BURNBACK_CUDA = "1"
pyinstaller packaging/burnback_studio.spec --noconfirm
```

Output lands in `dist/BurnbackStudio/`.

## What is likely to break, and how it shows

Recorded in advance because each has a characteristic symptom that is otherwise
slow to recognise:

| Symptom | Cause |
|---|---|
| `could not load the Qt platform plugin "windows"` | PySide6 plugins not collected. Check `dist/.../PySide6/plugins/platforms/`. |
| App opens, 3D tab blank or crashes on first render | VTK native libraries missing. The spec's `collect_all("vtkmodules")` is what prevents this. |
| `No module named 'trimesh.exchange.stl'` when opening a file | trimesh loads format handlers by name; they need `collect_submodules`. |
| Imports fine, dies allocating a tensor | torch native libraries partially collected. |
| Windows Defender quarantines the exe | Unsigned PyInstaller output. Code signing is the real fix; see below. |

## Code signing

Unsigned executables are routinely flagged, and a scary warning on first run
will cost more users than any feature will gain. A standard certificate is
roughly $200–400/year; an EV certificate is more and buys immediate SmartScreen
reputation. Worth deciding before the first public release, not after the first
complaint.

## Still to do

1. Run the full build and record the actual sizes.
2. Test on a clean Windows VM with no Python and no CUDA — the dev machine will
   always pass, for the wrong reasons.
3. Confirm CPU fallback on a machine with no NVIDIA GPU.
4. Wrap `dist/BurnbackStudio/` in an Inno Setup or NSIS installer: Start Menu
   entry, uninstaller, and a file association for `.srmd` design files (#155).
5. Decide on signing.
6. Consider building release artifacts in CI once the process is stable.
