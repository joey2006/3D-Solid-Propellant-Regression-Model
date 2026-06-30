"""Tests for burn-rate models (issue #7).

Run with:  pytest tests/test_physics.py
"""

import math

import pytest
import torch

from srm_burnback.physics.base import BurnRateModel
from srm_burnback.physics.vieille import VieilleBurnRate


# --- #54: abstract base ------------------------------------------------------


def test_base_cannot_be_instantiated():
    # BurnRateModel is an interface -- instantiating it directly must fail.
    with pytest.raises(TypeError):
        BurnRateModel()


def test_vieille_is_a_burn_rate_model():
    assert isinstance(VieilleBurnRate(a=0.005, n=0.3), BurnRateModel)


# --- #55 / #56: Vieille's law ------------------------------------------------


def test_known_value_matches_hand_calculation():
    a, n, P = 0.005, 0.3, 7e6
    F = VieilleBurnRate(a, n).compute_burn_rate(P)
    assert F.item() == pytest.approx(a * P**n)


def test_burn_rate_is_physically_reasonable():
    # F = a*P^n is unit-agnostic: a and P must share a unit system. The typical
    # a ~ 0.005 is the (m/s)/MPa^n convention (openMotor-style), so the pressure
    # here is in MPa (7 MPa), NOT Pa. With P = 7e6 Pa and the same a the result
    # would be ~0.57 m/s -- unphysically fast -- which is just the wrong unit
    # pairing, not a wrong formula.
    F = VieilleBurnRate(a=0.005, n=0.3).compute_burn_rate(7.0)  # P in MPa
    assert 0.001 < F.item() < 0.05


def test_doubling_pressure_scales_by_two_to_the_n():
    model = VieilleBurnRate(a=0.005, n=0.3)
    F1 = model.compute_burn_rate(3e6).item()
    F2 = model.compute_burn_rate(6e6).item()
    assert F2 / F1 == pytest.approx(2.0**0.3)


def test_zero_pressure_gives_zero_rate():
    F = VieilleBurnRate(a=0.005, n=0.3).compute_burn_rate(0.0)
    assert F.item() == pytest.approx(0.0)


def test_negative_pressure_raises():
    with pytest.raises(ValueError):
        VieilleBurnRate(a=0.005, n=0.3).compute_burn_rate(-1.0)


def test_output_is_tensor():
    F = VieilleBurnRate(a=0.005, n=0.3).compute_burn_rate(7e6)
    assert isinstance(F, torch.Tensor)


def test_scalar_input_gives_scalar_tensor():
    # Uniform burn -> a 0-d tensor that broadcasts against any grid.
    F = VieilleBurnRate(a=0.005, n=0.3).compute_burn_rate(7e6)
    assert F.ndim == 0


def test_tensor_pressure_field_preserved():
    # A per-point pressure field returns a matching field (the hook erosive
    # models will use). Also exercises the non-negative check elementwise.
    model = VieilleBurnRate(a=0.005, n=0.3)
    P = torch.tensor([1e6, 4e6, 9e6])
    F = model.compute_burn_rate(P)
    assert F.shape == P.shape
    assert torch.allclose(F, 0.005 * P**0.3)


@pytest.mark.parametrize("a,n,P", [(0.003, 0.5, 5e6), (0.01, 0.4, 1e6), (0.005, 0.35, 1.2e7)])
def test_multiple_parameter_sets(a, n, P):
    F = VieilleBurnRate(a, n).compute_burn_rate(P)
    assert F.item() == pytest.approx(a * P**n)
