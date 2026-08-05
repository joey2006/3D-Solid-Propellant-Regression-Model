"""The main burnback simulation driver (issues #64-#67).

``BurnbackSimulation`` wires every Phase 1 component together: grid setup and
initial SDF (geometry), the speed field (burn-rate model), Godunov + TVD-RK3
advection (levelset), periodic reinitialization, casing enforcement, surface
metrics, and burnout detection. Each timestep is:

1. compute the speed field ``F`` from the burn-rate model;
2. advance phi one TVD-RK3 step (CFL-limited timestep);
3. enforce the casing boundary (the front cannot leave the motor wall);
4. reinitialize every ``reinit_interval`` steps (fix SDF drift);
5. extract surface metrics and record them;
6. stop on burnout (no propellant left inside the casing) or ``max_time``.

Dimension-agnostic: the loop itself never branches on dimension -- phi can be
2D or 3D and every levelset operation loops over ``range(phi.ndim)``. Only the
surface-metric extraction dispatches on dimension (contours in 2D; the 3D
marching-cubes path arrives in Phase 5, #14).

Sign conventions (this module is where they are finalized, see #7/#9)
---------------------------------------------------------------------
* phi > 0 in the void, < 0 in propellant, 0 on the burning surface.
* Burning grows the void, so phi must *rise* everywhere: the level-set speed
  passed to the integrator is ``-F`` (``d(phi)/dt = -(-F)|grad phi| = +F``).
* ``phi_outer = geometry.outer_boundary_distance`` is negative inside the
  casing, positive outside.

Casing enforcement (#66) -- corrected formula
---------------------------------------------
The burn front must stop at the casing wall. We clamp
``phi = min(phi, -phi_outer)``: ``-phi_outer`` is the signed distance *to the
wall from inside* (positive inside, negative outside), i.e. the SDF of the
casing interior. The clamp caps the void at the casing interior -- outside the
wall phi can never become positive, so the front is pinned exactly at the wall.

Crucially this clamp is a **no-op until the burn actually reaches the wall**
(everywhere in the propellant ``phi < -phi_outer`` already holds), so it never
injects a spurious zero contour. The formula ``max(phi, phi_outer)`` originally
written in issue #66 would instead plant a zero level set *on* the wall at
t = 0, which the PDE would advect inward as a phantom front burning from the
casing -- halving the burnout time. Same intent ("burn can't pass the wall"),
opposite clamp direction.

Uniform burn rate is a validation idealization only
---------------------------------------------------
This runner accepts any :class:`~srm_burnback.physics.base.BurnRateModel`. With
the Phase 1 Vieille model the speed is a uniform scalar -- good only for
checking the numerics against the closed-form BATES answer. Every real motor
has erosive burning (the grain burns faster aft than fore); actual motor
predictions must use the erosive Lenoir-Robillard model when it lands (#13).
"""

from __future__ import annotations

import math

import torch

from ..geometry.GrainGeometry import GrainGeometry
from ..levelset import compute_cfl_timestep, reinitialize, tvd_rk3_step
from ..physics.base import BurnRateModel
from ..surface import (
    compute_burning_perimeter_2d,
    compute_port_area_2d,
    extract_contour_2d,
)
from .config import SimulationConfig
from .results import SimulationResults


class BurnbackSimulation:
    """Runs a grain burnback from ignition to burnout.

    Parameters
    ----------
    config:
        All numerical/physical knobs (grid, CFL, pressure, ...).
    geometry:
        The grain: supplies the initial SDF and the casing SDF.
    burn_rate_model:
        Supplies the front speed ``F`` each step. Scalar (uniform Vieille --
        validation only) or a full spatial field (erosive, Phase 3).
    """

    def __init__(
        self,
        config: SimulationConfig,
        geometry: GrainGeometry,
        burn_rate_model: BurnRateModel,
    ) -> None:
        self.config = config
        self.geometry = geometry
        self.burn_rate_model = burn_rate_model

        # Populated by initialize().
        self.phi: torch.Tensor | None = None
        self.coords: tuple[torch.Tensor, ...] | None = None
        self.h: float | None = None
        self.phi_outer: torch.Tensor | None = None
        self.results: SimulationResults | None = None
        self.time: float = 0.0
        self.step: int = 0

    # --- setup (#64) --------------------------------------------------------

    def initialize(self) -> None:
        """Build the grid, the initial phi, and the static casing SDF."""
        cfg = self.config
        phi, coords, h = self.geometry.initialize_grid(
            cfg.resolution,
            domain_size=cfg.domain_size,
            length=cfg.length,
            device=cfg.device,
        )
        self.phi = phi
        self.coords = coords
        self.h = h
        # The casing never moves: evaluated once, then only read (see the
        # "two SDFs" note in GrainGeometry). Same device as phi.
        self.phi_outer = self.geometry.outer_boundary_distance(coords)
        self.results = SimulationResults()
        self.time = 0.0
        self.step = 0

    # --- per-step pieces (#66, #67) ------------------------------------------

    def _enforce_casing_boundary(self, phi: torch.Tensor) -> torch.Tensor:
        """Pin the burn front at the casing wall: ``phi = min(phi, -phi_outer)``.

        No-op until the front reaches the wall (see module docstring for why
        this direction, not issue #66's original ``max``).
        """
        return torch.minimum(phi, -self.phi_outer)

    def _propellant_remaining(self, phi: torch.Tensor) -> bool:
        """Any unburned propellant left *inside the casing*?

        Cells outside the casing are inert (phi is clamped negative out there
        by design), so they are excluded via the casing mask.
        """
        return bool(((phi < 0) & (self.phi_outer < 0)).any())

    def _collect_metrics(self, phi: torch.Tensor) -> tuple[float, float, float]:
        """(burning perimeter, port area, propellant area) for the current phi.

        2D uses the contour pipeline from #8. The 3D analogue (marching cubes
        surface area) is Phase 5 (#14); until then a 3D run records NaN for
        the surface metrics but still runs the full burnback.
        """
        h = self.h
        inside = self.phi_outer < 0
        propellant_area = float(((phi < 0) & inside).sum().item()) * h**phi.ndim

        if phi.ndim == 2:
            contours = extract_contour_2d(phi, self.coords[0], self.coords[1])
            perimeter = compute_burning_perimeter_2d(contours)
            port_area = compute_port_area_2d(phi, h)
        else:
            perimeter = math.nan
            port_area = math.nan
        return perimeter, port_area, propellant_area

    # --- the main loop (#65) --------------------------------------------------

    def run(self, on_step=None) -> SimulationResults:
        """Advance from t = 0 until burnout or ``max_time``; return the history.

        Parameters
        ----------
        on_step:
            Optional ``callable(step, time, metrics) -> bool`` invoked after
            every step. Returning ``False`` stops the run and returns the
            history so far.

            This exists so a UI can show progress and offer a stop button
            (#132) without reimplementing the loop. Cancellation is cooperative
            -- checked once per step rather than by killing a thread -- which is
            enough here because a burnback is a long sequence of cheap steps.
            (The winding number is the opposite: one long kernel that cannot be
            interrupted, which is why *that* is warned about in advance
            instead.)

            A stopped run is a valid, if short, result rather than an error.
        """
        if self.phi is None:
            self.initialize()
        cfg = self.config
        phi = self.phi
        h = self.h
        results = self.results

        # Record the initial state (with a snapshot) before any stepping.
        F = self.burn_rate_model.compute_burn_rate(cfg.pressure)
        perim, port, prop = self._collect_metrics(phi)
        results.record(self.time, perim, port, prop, float(F.abs().max()), phi=phi)

        while self.time < cfg.max_time:
            # 1. Speed field. Positive burn rate; the level-set speed is -F so
            #    the void grows (see sign conventions in the module docstring).
            F = self.burn_rate_model.compute_burn_rate(cfg.pressure)

            # 2. Advance one CFL-limited TVD-RK3 step.
            dt = compute_cfl_timestep(F, h, cfg.cfl_factor)
            # Don't overshoot max_time.
            dt = min(dt, cfg.max_time - self.time)
            phi, _ = tvd_rk3_step(phi, -F, h, dt=dt)

            # 3. The front stops at the casing wall.
            phi = self._enforce_casing_boundary(phi)

            self.time += dt
            self.step += 1

            # 4. Periodically restore phi to a true signed-distance field.
            if cfg.reinit_iterations > 0 and self.step % cfg.reinit_interval == 0:
                phi = reinitialize(phi, h, n_iterations=cfg.reinit_iterations)
                phi = self._enforce_casing_boundary(phi)

            # 5. Metrics every step; full phi snapshot every snapshot_interval.
            burned_out = not self._propellant_remaining(phi)
            snapshot = self.step % cfg.snapshot_interval == 0 or burned_out
            perim, port, prop = self._collect_metrics(phi)
            results.record(
                self.time, perim, port, prop, float(F.abs().max()),
                phi=phi if snapshot else None,
            )

            if cfg.verbose and self.step % 100 == 0:
                print(
                    f"step {self.step:5d}  t = {self.time:.4f}  "
                    f"propellant = {prop:.6g}"
                )

            if on_step is not None and on_step(self.step, self.time, {
                "burning_perimeter": perim,
                "port_area": port,
                "propellant_area": prop,
                "burn_rate": float(F.abs().max()),
            }) is False:
                results.stopped = True
                break

            # 6. Burnout: no propellant left anywhere inside the casing.
            if burned_out:
                results.burned_out = True
                results.burnout_time = self.time
                if cfg.verbose:
                    print(f"burnout at t = {self.time:.4f} ({self.step} steps)")
                break

        self.phi = phi
        return results
