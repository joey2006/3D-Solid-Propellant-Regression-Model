"""Integration tests for the simulation runner (issue #9 / #62-#68).

Run with:  pytest tests/test_simulation.py

The gatekeeper check: a full BATES 2D burnback with a uniform Vieille rate
(the validation idealization -- real motors always burn erosively) must match
the closed-form answer r(t) = r0 + F*t. One full simulation is run once per
module and every assertion reads from it, so the suite stays fast.
"""

import math

import pytest
import torch

from srm_burnback.geometry.BATESGrain import BATESGrain
from srm_burnback.physics.vieille import VieilleBurnRate
from srm_burnback.simulation import (
    BurnbackSimulation,
    SimulationConfig,
    SimulationResults,
)

# The standard Phase 1 validation case (issue #68): metres, seconds, MPa.
INNER = 0.01
OUTER = 0.05
A_COEF = 0.005  # MPa convention
N_EXP = 0.35
PRESSURE = 7.0  # MPa
F_RATE = A_COEF * PRESSURE**N_EXP  # ~0.00988 m/s, uniform (validation only)


@pytest.fixture(scope="module")
def bates_run():
    """One full 200x200 BATES burnback shared by every test in this module."""
    grain = BATESGrain(inner_radius=INNER, outer_radius=OUTER)
    config = SimulationConfig(resolution=200, pressure=PRESSURE, max_time=10.0)
    sim = BurnbackSimulation(config, grain, VieilleBurnRate(a=A_COEF, n=N_EXP))
    results = sim.run()
    return grain, sim, results


# --- #62: SimulationConfig ----------------------------------------------------


def test_config_defaults_are_valid():
    cfg = SimulationConfig()
    assert cfg.resolution == 200
    assert 0.0 < cfg.cfl_factor < 1.0
    assert cfg.to_dict()["resolution"] == 200


def test_config_overrides():
    cfg = SimulationConfig(resolution=400, cfl_factor=0.3)
    assert cfg.resolution == 400
    assert cfg.cfl_factor == 0.3


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        SimulationConfig(cfl_factor=1.5)
    with pytest.raises(ValueError):
        SimulationConfig(resolution=1)
    with pytest.raises(ValueError):
        SimulationConfig(max_time=-1.0)


# --- #63: SimulationResults ---------------------------------------------------


def test_results_record_and_snapshots():
    res = SimulationResults()
    phi = torch.zeros(4, 4)
    res.record(0.0, 1.0, 2.0, 3.0, 0.5, phi=phi)
    res.record(0.1, 1.1, 2.1, 2.9, 0.5)  # no snapshot
    assert len(res) == 2
    assert res.times == [0.0, 0.1]
    assert len(res.phi_snapshots) == 1
    assert res.snapshot_times == [0.0]
    # Stored snapshot is a copy, not a live view.
    phi += 1.0
    assert res.phi_snapshots[0].abs().max().item() == 0.0
    d = res.to_dict()
    assert d["burning_perimeter"] == [1.0, 1.1]
    assert d["burned_out"] is False


# --- #64: initialize() ----------------------------------------------------------


def test_initialize_builds_grid_and_casing(bates_run):
    grain, sim, _ = bates_run
    assert sim.phi.shape == (200, 200)
    assert sim.h is not None and sim.h > 0
    # Casing SDF: negative inside the wall, positive outside.
    r = torch.sqrt(sim.coords[0] ** 2 + sim.coords[1] ** 2)
    assert (torch.sign(sim.phi_outer) == torch.sign(r - OUTER)).all()


# --- #65-#67: the full run ------------------------------------------------------


def test_simulation_reaches_burnout(bates_run):
    _, _, results = bates_run
    assert results.burned_out
    assert results.burnout_time is not None
    assert len(results) > 10  # a real time history was recorded


def test_burnout_time_matches_analytical(bates_run):
    grain, _, results = bates_run
    t_analytical = grain.analytical_burnout_time(F_RATE)
    assert results.burnout_time == pytest.approx(t_analytical, rel=0.05)


def test_no_propellant_remains_inside_casing(bates_run):
    _, sim, _ = bates_run
    inside = sim.phi_outer < 0
    assert (sim.phi[inside] >= 0).all()


def test_front_never_leaves_casing(bates_run):
    # #66: outside the wall phi must stay non-positive (no void past the wall).
    _, sim, _ = bates_run
    outside = sim.phi_outer > 0
    assert (sim.phi[outside] <= 0).all()


def test_initial_perimeter_matches_bore(bates_run):
    _, _, results = bates_run
    expected = 2.0 * math.pi * INNER
    assert results.burning_perimeter[0] == pytest.approx(expected, rel=0.05)


def test_intermediate_radius_within_1pct(bates_run):
    """Bore radius (from perimeter / 2*pi) matches r0 + F*t within 1%."""
    grain, _, results = bates_run
    times = results.times_tensor()
    perims = results.burning_perimeter_tensor()
    for t, p in zip(times.tolist(), perims.tolist()):
        r_ana = grain.analytical_radius(t, F_RATE)
        # Only intermediate times: skip the start-up and the final cell-width
        # of web where the discrete front merges with the wall.
        if r_ana < INNER + 2 * 0.0005 or r_ana > OUTER - 2 * 0.0005:
            continue
        r_sim = p / (2.0 * math.pi)
        assert r_sim == pytest.approx(r_ana, rel=0.01), f"at t={t:.4f}"


def test_metrics_recorded_every_step(bates_run):
    _, _, results = bates_run
    n = len(results)
    assert len(results.burning_perimeter) == n
    assert len(results.port_area) == n
    assert len(results.propellant_area) == n
    assert len(results.burn_rate) == n
    # Snapshots are sparse but present, and include the final state.
    assert 1 < len(results.phi_snapshots) < n
    assert results.snapshot_times[-1] == pytest.approx(results.times[-1])


def test_port_area_grows_monotonically(bates_run):
    _, _, results = bates_run
    areas = results.port_area_tensor()
    assert (areas[1:] >= areas[:-1]).all()


def test_run_respects_max_time():
    """No burnout before max_time -> the loop exits cleanly at max_time."""
    grain = BATESGrain(inner_radius=INNER, outer_radius=OUTER)
    config = SimulationConfig(resolution=64, pressure=PRESSURE, max_time=0.05)
    sim = BurnbackSimulation(config, grain, VieilleBurnRate(a=A_COEF, n=N_EXP))
    results = sim.run()
    assert not results.burned_out
    assert results.times[-1] == pytest.approx(0.05)
