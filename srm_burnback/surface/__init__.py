"""Burning-surface extraction and geometric metrics."""

from .extraction import extract_contour_2d
from .metrics import (
    compute_burning_perimeter_2d,
    compute_hydraulic_diameter,
    compute_port_area_2d,
)

__all__ = [
    "extract_contour_2d",
    "compute_burning_perimeter_2d",
    "compute_port_area_2d",
    "compute_hydraulic_diameter",
]
