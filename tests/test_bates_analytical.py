"""Formal BATES analytical validation (issue #10 / #72).

Run with:  pytest tests/test_bates_analytical.py

This is the gatekeeper for the whole project: a uniformly burning circle must
match r(t) = r0 + F*t. The validation criteria from issue #10, on a 200x200
grid:

* bore radius within 1% of r0 + F*t at every output step,
* burnout time within 1% of (r_outer - r_inner) / F,
* the contour stays circular (max radius - min radius < 1 grid spacing),
* burning perimeter within 2% of 2*pi*r(t) at every output step.

The uniform rate is the validation idealization -- real motors always burn
erosively and never produce this circular front.
"""

import math

import matplotlib

matplotlib.use("Agg")  # plotting smoke tests must not open windows

import pytest
import torch

from srm_burnback.geometry.BATESGrain import BATESGrain
from srm_burnback.physics.vieille import VieilleBurnRate
from srm_burnback.simulation import BurnbackSimulation, SimulationConfig
from srm_burnback.surface import extract_contour_2d
from srm_burnback.visualization import (
    animate_burnback,
    plot_burning_perimeter,
    plot_phi_field,
    plot_port_area,
)

INNER = 0.01
OUTER = 0.05
A_COEF = 0.005  # MPa convention
N_EXP = 0.35
PRESSURE = 7.0  # MPa
F_RATE = A_COEF * PRESSURE**N_EXP


@pytest.fixture(scope="module")
def bates_run():
    """The issue #10 validation run: 200x200, shared by all tests here.

    reinit_interval=20: for a uniform circular burn phi stays an exact SDF, so
    reinitialization contributes nearly pure drift; running it sparsely is the
    correct configuration for this case (real erosive sims keep the default).
    """
    grain = BATESGrain(inner_radius=INNER, outer_radius=OUTER)
    config = SimulationConfig(
        resolution=200, pressure=PRESSURE, max_time=10.0, reinit_interval=20
    )
    sim = BurnbackSimulation(config, grain, VieilleBurnRate(a=A_COEF, n=N_EXP))
    results = sim.run()
    return grain, sim, results


def _mid_burn_steps(results, h):
    """(t, perimeter) pairs excluding the final cell-width of web, where the
    discrete front merges with the casing wall."""
    return [
        (t, p)
        for t, p in zip(results.times, results.burning_perimeter)
        if INNER + F_RATE * t < OUTER - 2 * h
    ]


# --- issue #10 validation criteria ---------------------------------------------


def test_radius_within_1pct(bates_run):
    grain, sim, results = bates_run
    for t, p in _mid_burn_steps(results, sim.h):
        r_ana = grain.analytical_radius(t, F_RATE)
        r_sim = p / (2.0 * math.pi)
        assert r_sim == pytest.approx(r_ana, rel=0.01), f"at t={t:.4f}"


def test_burnout_time_within_1pct(bates_run):
    grain, _, results = bates_run
    assert results.burned_out
    t_ana = grain.analytical_burnout_time(F_RATE)
    assert results.burnout_time == pytest.approx(t_ana, rel=0.01)


def test_contour_stays_circular(bates_run):
    """Max radius minus min radius of the contour < 1 grid spacing, always."""
    _, sim, results = bates_run
    for t, phi in zip(results.snapshot_times, results.phi_snapshots):
        if INNER + F_RATE * t >= OUTER - 2 * sim.h:
            continue  # front merging with the wall at burnout
        contours = extract_contour_2d(phi, sim.coords[0].cpu(), sim.coords[1].cpu())
        assert len(contours) == 1, f"expected one contour at t={t:.4f}"
        r = torch.sqrt(contours[0][:, 0] ** 2 + contours[0][:, 1] ** 2)
        assert (r.max() - r.min()).item() < sim.h, f"at t={t:.4f}"


def test_perimeter_within_2pct(bates_run):
    grain, sim, results = bates_run
    for t, p in _mid_burn_steps(results, sim.h):
        expected = grain.analytical_burning_perimeter(t, F_RATE)
        assert p == pytest.approx(expected, rel=0.02), f"at t={t:.4f}"


def test_port_area_within_2pct(bates_run):
    grain, sim, results = bates_run
    for t, a in [
        (t, a)
        for t, a in zip(results.times, results.port_area)
        if INNER + F_RATE * t < OUTER - 2 * sim.h
    ]:
        expected = grain.analytical_port_area(t, F_RATE)
        assert a == pytest.approx(expected, rel=0.02), f"at t={t:.4f}"


# --- #69/#70: plotting and animation smoke tests --------------------------------


def test_metric_plots_render(bates_run):
    import matplotlib.pyplot as plt

    grain, _, results = bates_run
    perim_ana = [grain.analytical_burning_perimeter(t, F_RATE) for t in results.times]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot_burning_perimeter(results.times, results.burning_perimeter, ax1,
                           analytical=perim_ana)
    plot_port_area(results.times, results.port_area, ax2)
    assert len(ax1.lines) == 2  # simulated + analytical
    assert len(ax2.lines) == 1
    plt.close(fig)


def test_phi_field_plot_renders(bates_run):
    import matplotlib.pyplot as plt

    _, sim, results = bates_run
    fig, ax = plt.subplots()
    img = plot_phi_field(results.phi_snapshots[0], sim.coords, ax, show_casing=OUTER)
    assert img is not None
    plt.close(fig)


def test_animation_builds(bates_run):
    import matplotlib.pyplot as plt

    _, sim, results = bates_run
    # Just the first few frames -- this checks wiring, not rendering speed.
    anim = animate_burnback(
        results.phi_snapshots[:3], results.snapshot_times[:3], sim.coords,
        show_casing=OUTER,
    )
    assert anim is not None
    plt.close("all")


def test_animation_rejects_empty():
    with pytest.raises(ValueError):
        animate_burnback([], [], (torch.zeros(4, 4), torch.zeros(4, 4)))
