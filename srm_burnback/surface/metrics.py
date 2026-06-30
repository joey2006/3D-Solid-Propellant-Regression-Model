"""Geometric metrics of the burning surface.

These scalars are what the rest of the simulator consumes:

* **burning perimeter** (2D) / area (3D) -- how much surface is burning, which
  sets the gas generation rate and hence chamber pressure and thrust;
* **port area** -- the open cross-section the exhaust flows through, needed for
  the axial mass flux ``G`` in the erosive model;
* **hydraulic diameter** -- ``D_h = 4 A_port / P_wetted``, the other erosive-model
  input, generalizing "diameter" to non-circular ports.

For a circular bore of radius ``r`` these reduce to the familiar ``2*pi*r``,
``pi*r^2`` and ``2*r``, which is exactly how they are validated against the BATES
analytical oracle.
"""

from __future__ import annotations

import torch


def compute_burning_perimeter_2d(contours: list[torch.Tensor]) -> float:
    """Total arc length of all burning contours.

    The arc length of a discrete path is the sum of distances between
    consecutive points; ``find_contours`` returns closed loops, so the closing
    segment is already included. Summed across every contour, so disconnected
    burning surfaces all contribute.

    Parameters
    ----------
    contours:
        List of ``(N, 2)`` coordinate tensors from
        :func:`srm_burnback.surface.extraction.extract_contour_2d`.

    Returns
    -------
    float
        The burning perimeter in physical length units. ``0.0`` if there are no
        contours (no burning surface).
    """
    total = 0.0
    for path in contours:
        if path.shape[0] < 2:
            continue
        segments = path[1:] - path[:-1]
        total += torch.linalg.vector_norm(segments, dim=1).sum().item()
    return total


def compute_port_area_2d(phi: torch.Tensor, h: float) -> float:
    """Open (bore) cross-sectional area: cells with ``phi > 0`` times ``h^2``.

    Parameters
    ----------
    phi:
        2D level-set field (positive in the open void).
    h:
        Grid spacing (square cells of area ``h^2``).

    Returns
    -------
    float
        Port area in physical area units.
    """
    open_cells = (phi > 0).sum().item()
    return open_cells * h * h


def compute_hydraulic_diameter(port_area: float, perimeter: float) -> float:
    """Hydraulic diameter ``D_h = 4 * A_port / P_wetted``.

    For a circular bore this is ``4 * pi r^2 / (2 pi r) = 2 r`` -- exactly the
    diameter, as expected.

    Parameters
    ----------
    port_area:
        Open cross-sectional area.
    perimeter:
        Wetted (burning) perimeter.

    Returns
    -------
    float
        The hydraulic diameter, or ``0.0`` when the perimeter is zero (no
        burning surface, e.g. at burnout) -- ``D_h`` is undefined there and 0 is
        a safe sentinel for downstream code.
    """
    if perimeter <= 0.0:
        return 0.0
    return 4.0 * port_area / perimeter
