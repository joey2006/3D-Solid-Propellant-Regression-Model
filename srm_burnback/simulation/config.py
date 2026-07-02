"""Simulation configuration (issue #62).

A single dataclass holds every knob the simulation runner needs, so a run is
fully described by one object: easy to print, easy to serialize, easy to sweep
in a convergence study. The defaults are tuned for the standard Phase 1
validation case -- a BATES 2D cross-section on a 200x200 grid.

Units
-----
Lengths, times and pressures are unit-agnostic here, exactly like the rest of
the package: the numbers only have to be *consistent with each other* and with
the burn-rate model's coefficients. The default ``pressure`` follows the same
convention the burn model uses (e.g. a Vieille ``a ~ 0.005`` is the MPa
convention, so feed pressure in MPa; see ``srm_burnback.physics.vieille``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class SimulationConfig:
    """All parameters of a burnback run in one place.

    Parameters
    ----------
    resolution:
        Grid points per axis (the grid is square/cubic).
    domain_size:
        Half-extent of the x/y axes. ``None`` uses the grain's
        ``default_domain_size()`` (casing radius plus a small margin).
    length:
        Axial extent for a 3D run. ``None`` runs a 2D cross-section --
        the Phase 1 validation configuration.
    cfl_factor:
        CFL safety factor in ``(0, 1)`` for the auto-computed timestep.
    max_time:
        Hard cap on simulated time; the run stops here even without burnout.
    reinit_interval:
        Reinitialize the SDF every N steps (fixes ``|grad phi|`` drift).
    reinit_iterations:
        Pseudo-time iterations per reinitialization (5-10 is plenty).
    pressure:
        Chamber pressure fed to the burn-rate model. Constant in Phase 1;
        computed from chamber conditions in Phase 6. Units must match the
        burn model's coefficient convention (MPa for the usual ``a ~ 0.005``).
    snapshot_interval:
        Store the full phi field every N steps (for animation); per-step
        scalar metrics are always stored every step.
    device:
        Torch device: ``"cpu"`` or ``"cuda"``.
    verbose:
        Print progress every ~100 steps.
    """

    resolution: int = 200
    domain_size: float | None = None
    length: float | None = None
    cfl_factor: float = 0.5
    max_time: float = 10.0
    reinit_interval: int = 5
    reinit_iterations: int = 5
    pressure: float = 7.0
    snapshot_interval: int = 10
    device: str = "cpu"
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.resolution < 3:
            raise ValueError("resolution must be at least 3 grid points")
        if not 0.0 < self.cfl_factor < 1.0:
            raise ValueError("cfl_factor must be in (0, 1)")
        if self.max_time <= 0.0:
            raise ValueError("max_time must be positive")
        if self.reinit_interval < 1 or self.reinit_iterations < 0:
            raise ValueError("reinit_interval >= 1 and reinit_iterations >= 0 required")
        if self.snapshot_interval < 1:
            raise ValueError("snapshot_interval must be >= 1")

    def to_dict(self) -> dict:
        """Plain-dict view for JSON/YAML serialization."""
        return asdict(self)
