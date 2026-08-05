"""Which faces of an imported grain burn, and which are inhibited (#176).

The level set advances **the entire zero contour**: every point where φ = 0 is
pushed outward at the local burn rate. But φ on its own only says
propellant-or-not. It carries nothing about which parts of the boundary are
actually on fire.

For a real motor that distinction decides the answer. A BATES grain bonded into
its casing has a boundary made of the bore, the outer cylindrical wall and two
end faces, and only the bore is burning -- no flame ever reaches the wall. Take
the distance to *all* of them and φ turns around at the middle of the web and
crosses zero again at the outer wall, so the solver consumes the grain from the
outside in as well as the inside out. Burn area roughly doubles and every
number downstream is wrong, while ``|∇φ| = 1`` still holds perfectly and the
picture still looks plausible.

Parametric grains dodge this because we wrote their equations: ``BATESGrain``
models the bore alone and the casing is a separate static field consumed by the
``min(phi, -phi_outer)`` clamp. An imported triangle soup has no such
knowledge, which is what this module supplies.

How faces are labelled
----------------------
By how each face normal lines up with the outward radial direction -- the same
classification :mod:`measurements` uses to recover bore and outer diameter, and
the same one the 3D view uses to colour the grain. Keeping it in one place is
deliberate: if the bore is mis-detected, it is mis-measured *and* mis-coloured
*and* mis-burned, so the error is visible rather than silent.

* **inward-facing** (normal points at the axis) -- bore and slot walls. These
  face the flame, so they burn.
* **outward-facing** -- the outer wall. Bonded to the casing, so inhibited.
* **axial** -- the end faces. Genuinely ambiguous, and the caller decides.

Why the ends are a choice, not a detection
------------------------------------------
Nothing in the geometry distinguishes an inhibited end face from a burning one;
it depends on whether the motor builder painted it. The default here is
**inhibited**, matching a case-bonded grain, but it materially changes burn area
and therefore the whole thrust curve, so it is exposed rather than buried.

This is the automatic first guess of #176, not the whole of it: a user override
and a supplied liner mesh are still open there. What it does provide is a
labelling that is *visible before a simulation runs on it*.
"""

from __future__ import annotations

import numpy as np

from .measurements import RADIAL_THRESHOLD

#: How the end faces of an imported grain are treated by default.
DEFAULT_ENDS = "inhibited"


def surface_labels(mesh, ends: str = DEFAULT_ENDS) -> dict:
    """Label every face of a Z-oriented grain as burning or inhibited.

    Parameters
    ----------
    mesh:
        A ``trimesh.Trimesh``, already oriented so its long axis is Z.
    ends:
        ``"inhibited"`` (default) or ``"burning"``, for the axial end faces.

    Returns
    -------
    dict
        ``burning`` and ``inhibited`` boolean arrays over faces, plus the
        ``inner`` / ``outer`` / ``axial`` classification they were built from
        and the counts, so a caller can report what it decided.
    """
    if ends not in ("inhibited", "burning"):
        raise ValueError(f"ends must be 'inhibited' or 'burning', got {ends!r}")

    faces = np.asarray(mesh.faces)
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.face_normals)
    centres = vertices[faces].mean(axis=1)

    bounds = np.asarray(mesh.bounds)
    axis_centre = (bounds[0] + bounds[1]) / 2.0

    radial = centres[:, :2] - axis_centre[:2]
    radius = np.linalg.norm(radial, axis=1)
    unit = np.divide(
        radial,
        radius[:, None],
        out=np.zeros_like(radial),
        where=radius[:, None] > 1e-12,
    )
    alignment = np.einsum("ij,ij->i", normals[:, :2], unit)

    outer = alignment > RADIAL_THRESHOLD
    inner = alignment < -RADIAL_THRESHOLD
    axial = ~outer & ~inner

    burning = inner.copy()
    if ends == "burning":
        burning |= axial

    return {
        "burning": burning,
        "inhibited": ~burning,
        "inner": inner,
        "outer": outer,
        "axial": axial,
        "n_burning": int(burning.sum()),
        "n_inhibited": int((~burning).sum()),
        "ends": ends,
    }
