"""Unit tests for grain geometry SDFs.

Run with:  pytest tests/test_geometry.py
"""

import pytest
import torch

from srm_burnback.geometry.bates import BATESGrain


def make_grain() -> BATESGrain:
    """A standard test grain: bore 0.5, casing 2.0, length 6.0."""
    return BATESGrain(inner_radius=0.5, outer_radius=2.0, length=6.0)


def test_center_of_bore_is_positive():
    # A point on the central axis is in the open bore -> SDF must be positive.
    grain = make_grain()
    X = torch.tensor([0.0])
    Y = torch.tensor([0.0])
    phi = grain.signed_distance((X, Y))
    assert phi.item() > 0


def test_bore_wall_is_zero():
    # A point exactly on the bore wall (r == inner_radius) is the burning
    # surface -> SDF must be (approximately) zero.
    grain = make_grain()
    X = torch.tensor([0.5])
    Y = torch.tensor([0.0])
    phi = grain.signed_distance((X, Y))
    assert phi.item() == pytest.approx(0.0)


def test_inside_propellant_is_negative():
    # A point between the bore and the casing is solid propellant -> negative.
    grain = make_grain()
    X = torch.tensor([1.0])
    Y = torch.tensor([0.0])
    phi = grain.signed_distance((X, Y))
    assert phi.item() < 0


def test_outer_boundary_zero_at_casing():
    # The casing-wall SDF must be zero exactly at outer_radius.
    grain = make_grain()
    X = torch.tensor([2.0])
    Y = torch.tensor([0.0])
    dist = grain.outer_boundary_distance((X, Y))
    assert dist.item() == pytest.approx(0.0)


def test_3d_past_end_face_is_void():
    # In 3D, a point at a propellant radius but beyond the end face
    # (|z| > length/2) is in the open void -> positive.
    grain = make_grain()
    X = torch.tensor([1.0])
    Y = torch.tensor([0.0])
    Z = torch.tensor([3.5])  # length/2 == 3.0, so this is past the end
    phi = grain.signed_distance((X, Y, Z))
    assert phi.item() > 0


def test_3d_inside_propellant_is_negative():
    # In 3D, a point inside both the radius and length limits is propellant.
    grain = make_grain()
    X = torch.tensor([1.0])
    Y = torch.tensor([0.0])
    Z = torch.tensor([0.0])  # middle of the grain
    phi = grain.signed_distance((X, Y, Z))
    assert phi.item() < 0


# --- analytical validation oracle (uniform, non-erosive rate) ----------------


def test_analytical_radius():
    # r(t) = r_inner + burn_rate * t  (issue #32 'Done when' example).
    grain = BATESGrain(inner_radius=0.01, outer_radius=0.05, length=0.1)
    assert grain.analytical_radius(1.0, 0.005) == pytest.approx(0.015)


def test_analytical_burnout_time():
    # t = (r_outer - r_inner) / burn_rate.
    grain = BATESGrain(inner_radius=0.01, outer_radius=0.05, length=0.1)
    assert grain.analytical_burnout_time(0.005) == pytest.approx(8.0)


def test_radius_reaches_outer_at_burnout():
    # By construction the bore should arrive exactly at the casing at burnout.
    grain = BATESGrain(inner_radius=0.01, outer_radius=0.05, length=0.1)
    t_burnout = grain.analytical_burnout_time(0.005)
    assert grain.analytical_radius(t_burnout, 0.005) == pytest.approx(grain.outer_radius)


def test_analytical_perimeter_and_port_area():
    # Perimeter = 2*pi*r(t), port area = pi*r(t)**2 at t=0 (r == r_inner).
    grain = BATESGrain(inner_radius=0.01, outer_radius=0.05, length=0.1)
    assert grain.analytical_burning_perimeter(0.0, 0.005) == pytest.approx(2 * 3.141592653589793 * 0.01)
    assert grain.analytical_port_area(0.0, 0.005) == pytest.approx(3.141592653589793 * 0.01**2)
