"""Tests for burning-surface extraction and metrics (issue #8).

Run with:  pytest tests/test_surface.py

For a circular bore of radius r the analytical metrics are perimeter = 2*pi*r,
port area = pi*r^2, and hydraulic diameter = 2*r. Grid discretization gives a
few percent error, so tolerances are ~5%.
"""

import math

import pytest
import torch

from srm_burnback.geometry.BATESGrain import BATESGrain
from srm_burnback.surface.extraction import extract_contour_2d
from srm_burnback.surface.metrics import (
    compute_burning_perimeter_2d,
    compute_hydraulic_diameter,
    compute_port_area_2d,
)


def _bates_field(inner=0.5, outer=2.0, res=401):
    grain = BATESGrain(inner_radius=inner, outer_radius=outer)
    phi, coords, h = grain.initialize_grid(res)
    return grain, phi, coords, h


# --- #57: contour extraction -------------------------------------------------


def test_contour_is_circle_at_bore_radius():
    grain, phi, coords, h = _bates_field()
    contours = extract_contour_2d(phi, coords[0], coords[1])
    assert len(contours) == 1  # a BATES bore is a single closed circle
    pts = contours[0]
    r = torch.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    # Every contour point sits within one cell of the true bore radius.
    assert (r - grain.inner_radius).abs().max().item() < h
    assert r.mean().item() == pytest.approx(grain.inner_radius, abs=h)


def test_contour_empty_when_no_zero_crossing():
    # A field that is positive everywhere has no surface.
    phi = torch.ones(50, 50)
    axis = torch.linspace(-1, 1, 50)
    X, Y = torch.meshgrid(axis, axis, indexing="ij")
    assert extract_contour_2d(phi, X, Y) == []


def test_extract_rejects_3d():
    with pytest.raises(ValueError):
        extract_contour_2d(torch.zeros(4, 4, 4), torch.zeros(4, 4, 4), torch.zeros(4, 4, 4))


# --- #58: burning perimeter --------------------------------------------------


def test_perimeter_matches_2pir():
    grain, phi, coords, h = _bates_field()
    contours = extract_contour_2d(phi, coords[0], coords[1])
    perimeter = compute_burning_perimeter_2d(contours)
    expected = 2 * math.pi * grain.inner_radius
    assert perimeter == pytest.approx(expected, rel=0.05)


def test_perimeter_sums_multiple_contours():
    # Two separate bores -> perimeter is the sum of both circumferences.
    res, half = 401, 3.0
    axis = torch.linspace(-half, half, res)
    X, Y = torch.meshgrid(axis, axis, indexing="ij")
    r1 = torch.sqrt((X - 1.5) ** 2 + Y**2)
    r2 = torch.sqrt((X + 1.5) ** 2 + Y**2)
    phi = torch.maximum(0.5 - r1, 0.5 - r2)  # union of two radius-0.5 discs
    contours = extract_contour_2d(phi, X, Y)
    assert len(contours) == 2
    perimeter = compute_burning_perimeter_2d(contours)
    assert perimeter == pytest.approx(2 * (2 * math.pi * 0.5), rel=0.05)


def test_perimeter_empty_is_zero():
    assert compute_burning_perimeter_2d([]) == 0.0


# --- #59: port area ----------------------------------------------------------


def test_port_area_matches_pir2():
    grain, phi, coords, h = _bates_field()
    area = compute_port_area_2d(phi, h)
    expected = math.pi * grain.inner_radius**2
    assert area == pytest.approx(expected, rel=0.05)


def test_port_area_scales_with_radius():
    g1, phi1, _, h1 = _bates_field(inner=0.5)
    g2, phi2, _, h2 = _bates_field(inner=1.0)
    a1 = compute_port_area_2d(phi1, h1)
    a2 = compute_port_area_2d(phi2, h2)
    # Doubling the bore radius quadruples the area.
    assert a2 / a1 == pytest.approx(4.0, rel=0.05)


# --- #60: hydraulic diameter -------------------------------------------------


def test_hydraulic_diameter_is_2r():
    grain, phi, coords, h = _bates_field()
    contours = extract_contour_2d(phi, coords[0], coords[1])
    perimeter = compute_burning_perimeter_2d(contours)
    area = compute_port_area_2d(phi, h)
    dh = compute_hydraulic_diameter(area, perimeter)
    assert dh == pytest.approx(2 * grain.inner_radius, rel=0.05)


def test_hydraulic_diameter_zero_perimeter():
    # Burnout / no surface -> safe sentinel, not a divide-by-zero.
    assert compute_hydraulic_diameter(1.0, 0.0) == 0.0


# --- two grain sizes (catch hardcoded radii) ---------------------------------


@pytest.mark.parametrize("inner", [0.5, 1.0])
def test_all_metrics_consistent_across_sizes(inner):
    grain, phi, coords, h = _bates_field(inner=inner, outer=inner + 2.0)
    contours = extract_contour_2d(phi, coords[0], coords[1])
    perimeter = compute_burning_perimeter_2d(contours)
    area = compute_port_area_2d(phi, h)
    dh = compute_hydraulic_diameter(area, perimeter)
    assert perimeter == pytest.approx(2 * math.pi * inner, rel=0.05)
    assert area == pytest.approx(math.pi * inner**2, rel=0.05)
    assert dh == pytest.approx(2 * inner, rel=0.05)
