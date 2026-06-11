"""Unit tests for grain geometry SDFs.

Run with:  pytest tests/test_geometry.py
"""

import pytest
import torch

from srm_burnback.geometry.BATESGrain import BATESGrain


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


# --- initialize_grid (issue #33 'Done when') ---------------------------------


def test_initialize_grid_shape():
    # `grain.initialize_grid(200)` returns a (200, 200) phi tensor with no
    # explicit domain_size (defaults to just past the casing wall).
    grain = make_grain()
    phi, coords, height = grain.initialize_grid(200)
    assert phi.shape == (200, 200)
    assert len(coords) == 2  # 2D grid by default
    assert isinstance(height, float)


def test_initialize_grid_sign_structure():
    # Negative values exist inside the propellant ring, positive in the bore.
    grain = make_grain()
    phi, _, _ = grain.initialize_grid(200)
    assert (phi > 0).any()  # bore (open void)
    assert (phi < 0).any()  # propellant
    # The exact centre cell is inside the bore -> positive.
    assert phi[100, 100].item() > 0


def test_initialize_grid_zero_set_matches_bore_radius():
    # The zero level set should sit at the initial bore radius. Walk a radial
    # line out from the centre and find where phi changes sign; that crossing
    # must land within one grid cell of inner_radius.
    grain = make_grain()
    resolution = 401  # odd -> a row passes exactly through the centre
    phi, coords, height = grain.initialize_grid(resolution)
    X = coords[0]
    mid = resolution // 2
    row = phi[mid]              # phi along the +/-x axis through the centre
    x_vals = X[:, mid]
    # First index past centre where phi crosses from + (bore) to - (propellant).
    right = row[mid:]
    crossing = (right[:-1] > 0) & (right[1:] <= 0)
    idx = mid + int(crossing.nonzero()[0].item())
    r_found = abs(x_vals[idx].item())
    assert r_found == pytest.approx(grain.inner_radius, abs=height)


def test_initialize_grid_3d():
    # Passing a length builds a full 3D grid, same call, one more axis.
    grain = make_grain()
    phi, coords, _ = grain.initialize_grid(60, length=grain.length)
    assert phi.shape == (60, 60, 60)
    assert len(coords) == 3


def test_initialize_grid_requires_domain_without_default():
    # A grain that doesn't define default_domain_size must be told the domain.
    grain = make_grain()
    # Explicit domain still works regardless of the default hook.
    phi, _, _ = grain.initialize_grid(50, domain_size=3.0)
    assert phi.shape == (50, 50)


def test_initialize_grid_dtype():
    # Issue #34 step 7: initialize_grid returns the right shape *and* dtype.
    grain = make_grain()
    phi, coords, _ = grain.initialize_grid(64)
    assert phi.dtype == torch.float32
    assert coords[0].dtype == torch.float32


# --- two grain sizes (issue #34: catch hardcoded values) ---------------------

# Each case: (inner_radius, outer_radius). Two distinct sizes so a test that
# accidentally hardcodes a radius fails on at least one of them.
GRAIN_SIZES = [(0.5, 2.0), (1.5, 4.0)]


@pytest.mark.parametrize("inner, outer", GRAIN_SIZES)
def test_center_sdf_equals_inner_radius(inner, outer):
    # Key detail: distance from the centre to the bore wall is exactly inner_radius.
    grain = BATESGrain(inner_radius=inner, outer_radius=outer)
    phi = grain.signed_distance((torch.tensor([0.0]), torch.tensor([0.0])))
    assert phi.item() == pytest.approx(inner)


@pytest.mark.parametrize("inner, outer", GRAIN_SIZES)
def test_bore_surface_is_zero(inner, outer):
    grain = BATESGrain(inner_radius=inner, outer_radius=outer)
    phi = grain.signed_distance((torch.tensor([inner]), torch.tensor([0.0])))
    assert phi.item() == pytest.approx(0.0)


@pytest.mark.parametrize("inner, outer", GRAIN_SIZES)
def test_propellant_interior_is_negative(inner, outer):
    grain = BATESGrain(inner_radius=inner, outer_radius=outer)
    # A radius midway between bore and casing is solid propellant.
    mid = (inner + outer) / 2
    phi = grain.signed_distance((torch.tensor([mid]), torch.tensor([0.0])))
    assert phi.item() < 0


@pytest.mark.parametrize("inner, outer", GRAIN_SIZES)
def test_casing_wall_is_zero(inner, outer):
    grain = BATESGrain(inner_radius=inner, outer_radius=outer)
    dist = grain.outer_boundary_distance((torch.tensor([outer]), torch.tensor([0.0])))
    assert dist.item() == pytest.approx(0.0)
