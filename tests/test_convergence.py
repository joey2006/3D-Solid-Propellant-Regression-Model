"""Grid and temporal convergence tests (issue #11 / #73-#75).

Run with:  pytest tests/test_convergence.py

Three properties prove the numerics are implemented correctly:

* **Spatial convergence** (#73): the burn-radius error decreases systematically
  as the grid is refined. The Godunov Hamiltonian here uses first-order
  one-sided differences, so the formal order is O(h) -- measured slope ~1.2.
  (The O(h^2) figure in the original issue text describes a higher-order
  ENO/WENO discretization, a possible future upgrade.)
* **Temporal convergence** (#74): TVD-RK3 is third order, O(dt^3). This cannot
  be seen against the analytical answer at fixed h (the spatial error floor
  dominates), so it is measured by *self-convergence*: fixed grid, shrinking
  dt, error taken against a tiny-dt reference run -- the shared spatial error
  cancels exactly, isolating the time integrator. The speed field must be
  spatially varying: a uniform circular burn keeps |grad phi| = 1, making the
  RHS time-independent and leaving RK3 nothing to converge on. Run in float64
  so O(dt^3) differences stay above roundoff.
* **Mass conservation** (#75): propellant area only ever decreases -- burning
  destroys propellant and nothing creates it.
"""

import math

import numpy as np
import pytest
import torch

from srm_burnback.geometry.BATESGrain import BATESGrain
from srm_burnback.levelset import tvd_rk3_step
from srm_burnback.physics.vieille import VieilleBurnRate
from srm_burnback.simulation import BurnbackSimulation, SimulationConfig

INNER = 0.01
OUTER = 0.05
A_COEF = 0.005  # MPa convention
N_EXP = 0.35
PRESSURE = 7.0  # MPa
F_RATE = A_COEF * PRESSURE**N_EXP


def run_spatial_study(resolutions=(50, 100, 200, 400)):
    """Max abs bore-radius error vs analytical at each resolution.

    Returns (grid spacings, max errors), both lists of floats. Shared by the
    test below and examples/convergence_study.py.
    """
    grain = BATESGrain(inner_radius=INNER, outer_radius=OUTER)
    hs, errs = [], []
    for res in resolutions:
        cfg = SimulationConfig(resolution=res, pressure=PRESSURE, reinit_interval=20)
        sim = BurnbackSimulation(cfg, grain, VieilleBurnRate(a=A_COEF, n=N_EXP))
        results = sim.run()
        err = max(
            abs(p / (2 * math.pi) - grain.analytical_radius(t, F_RATE))
            for t, p in zip(results.times, results.burning_perimeter)
            if INNER + 2 * sim.h < grain.analytical_radius(t, F_RATE) < OUTER - 2 * sim.h
        )
        hs.append(sim.h)
        errs.append(err)
    return hs, errs


def run_temporal_study(n_steps_list=(25, 50, 100, 200), n_ref=1000, resolution=100):
    """Self-convergence of TVD-RK3: fixed grid, shrinking dt, error against a
    tiny-dt reference. Returns (dts, sup-norm errors)."""
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        grain = BATESGrain(inner_radius=INNER, outer_radius=OUTER)
        phi0, coords, h = grain.initialize_grid(resolution)
        X, Y = coords
        L = grain.default_domain_size()
        # Spatially varying speed (erosive-like) so the RHS evolves in time.
        F = F_RATE * (1.0 + 0.3 * torch.sin(math.pi * X / L) * torch.cos(math.pi * Y / L))

        T = 0.5

        def advance(n_steps):
            phi = phi0.clone()
            dt = T / n_steps
            for _ in range(n_steps):
                phi, _ = tvd_rk3_step(phi, -F, h, dt=dt)
            return phi

        ref = advance(n_ref)
        dts, errs = [], []
        for n in n_steps_list:
            errs.append((advance(n) - ref).abs().max().item())
            dts.append(T / n)
        return dts, errs
    finally:
        torch.set_default_dtype(default_dtype)


def fit_loglog_slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


# --- #73: spatial convergence ---------------------------------------------------


@pytest.fixture(scope="module")
def spatial_study():
    return run_spatial_study()


def test_spatial_error_decreases_monotonically(spatial_study):
    _, errs = spatial_study
    assert all(e2 < e1 for e1, e2 in zip(errs, errs[1:]))


def test_spatial_convergence_at_least_first_order(spatial_study):
    hs, errs = spatial_study
    slope = fit_loglog_slope(hs, errs)
    # First-order one-sided differences: formal O(h), measured ~1.2.
    assert slope > 0.8, f"spatial slope {slope:.2f} -- not converging"


# --- #74: temporal convergence --------------------------------------------------


def test_temporal_convergence_third_order():
    dts, errs = run_temporal_study()
    slope = fit_loglog_slope(dts, errs)
    # TVD-RK3 is O(dt^3); allow a band around 3 for fit noise.
    assert 2.5 < slope < 3.5, f"temporal slope {slope:.2f}, expected ~3"


# --- #75: mass conservation -----------------------------------------------------


def test_propellant_area_never_increases():
    grain = BATESGrain(inner_radius=INNER, outer_radius=OUTER)
    cfg = SimulationConfig(resolution=100, pressure=PRESSURE)
    sim = BurnbackSimulation(cfg, grain, VieilleBurnRate(a=A_COEF, n=N_EXP))
    results = sim.run()
    areas = results.propellant_area_tensor()
    # Never increases at any step (burning only destroys propellant)...
    assert (areas[1:] <= areas[:-1]).all()
    # ...and strictly decreases while propellant remains.
    assert (areas[1:] < areas[:-1]).sum() >= 0.95 * (len(areas) - 1)
    assert areas[-1].item() == 0.0  # burnout: nothing left
