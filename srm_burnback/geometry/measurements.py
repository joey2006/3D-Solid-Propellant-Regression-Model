"""Engineering dimensions of an imported grain (#157).

``mesh_stats`` in :mod:`import_mesh` reports what a *mesh* is -- triangle
counts, watertightness, a bounding box. That answers "did this file import
cleanly". It does not answer "what motor is this", which needs bore diameter,
web thickness and propellant volume.

Those are recovered from the geometry rather than asked for, because the whole
premise of the project is that the input is an arbitrary uploaded object with
no parameters attached (see the object-first principle). A BATES grain does not
arrive labelled "inner radius 10 mm"; it arrives as triangles.

How the radii are recovered
---------------------------
The grain axis is Z (imports are oriented that way -- see
:func:`import_mesh.orient_grain_axis_to_z`). Every face is then classified by
how its normal relates to the outward radial direction:

* outer wall -- normal points radially outward, alignment near +1;
* bore and slot walls -- normal points radially inward, alignment near -1;
* end faces -- normal is axial, alignment near 0.

This is the same classification the 3D view uses to colour the grain, kept here
so the number and the picture can never disagree.

Everything is in the mesh's own units, which for this project means metres.
"""

from __future__ import annotations

import numpy as np

#: Faces beyond this alignment count as radial rather than axial.
RADIAL_THRESHOLD = 0.35


def grain_measurements(mesh, density: float | None = None) -> dict:
    """Engineering dimensions of a grain mesh whose axis is Z.

    Parameters
    ----------
    mesh:
        A ``trimesh.Trimesh``, already oriented so its long axis is Z.
    density:
        Propellant density in kg/m^3. When given, propellant mass is included.

    Returns
    -------
    dict
        Plain Python scalars, any of which may be ``None`` when the geometry
        does not support them -- an unclosed mesh has no meaningful volume, and
        a grain with no inward-facing surface has no bore.
    """
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

    # Faces are classified by their normals, but radii are measured at their
    # *vertices*, not their centroids. A tessellated circle is an inscribed
    # polygon: its vertices lie exactly on the true circle while every face
    # centroid sits inside it, short by a factor of cos(pi/N). Measuring at
    # centroids therefore under-reports every diameter -- a 128-sided 50 mm
    # bore reads 49.9866 mm. Measuring at vertices returns 50.0000 mm, so a
    # part modelled as 1.000 in reads as 1.000 in rather than 0.9996 in.
    #
    # The extremes rather than the means: the outer wall is the furthest
    # material from the axis and the bore the closest. A mean would be dragged
    # around by slot walls sitting between the two.
    vertex_radius = np.linalg.norm(vertices[:, :2] - axis_centre[:2], axis=1)
    outer_radius = (
        float(vertex_radius[np.unique(faces[outer])].max()) if outer.any() else None
    )
    bore_radius = (
        float(vertex_radius[np.unique(faces[inner])].min()) if inner.any() else None
    )

    web = (
        outer_radius - bore_radius
        if outer_radius is not None and bore_radius is not None
        else None
    )

    length = float(bounds[1][2] - bounds[0][2])
    watertight = bool(mesh.is_watertight)
    volume = float(mesh.volume) if watertight else None

    # Volume the grain would occupy if solid, so the port fraction says how
    # much of the envelope has been hollowed out -- a quick sanity check that
    # the bore was actually recognised.
    envelope = (
        float(np.pi * outer_radius**2 * length)
        if outer_radius is not None
        else None
    )
    port_fraction = (
        1.0 - volume / envelope
        if volume is not None and envelope not in (None, 0.0)
        else None
    )

    return {
        "length": length,
        "outer_radius": outer_radius,
        "outer_diameter": None if outer_radius is None else 2 * outer_radius,
        "bore_radius": bore_radius,
        "bore_diameter": None if bore_radius is None else 2 * bore_radius,
        "web_thickness": web,
        "volume": volume,
        "surface_area": float(mesh.area),
        "port_fraction": port_fraction,
        "mass": None if (volume is None or density is None) else volume * density,
        "length_to_diameter": (
            None
            if outer_radius is None or outer_radius == 0
            else length / (2 * outer_radius)
        ),
    }
