"""Time-history storage for a burnback run (issue #63).

The runner produces two kinds of output with very different sizes:

* **per-step scalars** (time, burning perimeter, port area, propellant area,
  burn rate) -- tiny, so they are recorded at *every* step; these are the
  curves that get plotted and validated;
* **full phi snapshots** -- one whole grid each, so they are only stored every
  ``snapshot_interval`` steps (plus the final state); these become the frames
  of the burn animation.

Everything is plain lists so the class stays dumb storage; tensors views for
plotting come from :meth:`SimulationResults.times_tensor` and friends.
"""

from __future__ import annotations

import torch


class SimulationResults:
    """Accumulates the outputs of one simulation run."""

    def __init__(self) -> None:
        # Per-step scalar history (every timestep).
        self.times: list[float] = []
        self.burning_perimeter: list[float] = []
        self.port_area: list[float] = []
        self.propellant_area: list[float] = []
        self.burn_rate: list[float] = []

        # Sparse full-field history (every snapshot_interval steps).
        self.snapshot_times: list[float] = []
        self.phi_snapshots: list[torch.Tensor] = []

        # Filled in by the runner when the propellant is exhausted.
        self.burned_out: bool = False
        self.burnout_time: float | None = None
        #: True when a caller stopped the run early through ``run(on_step=...)``.
        #: A stopped run is a valid short history, not a failure.
        self.stopped: bool = False

    def record(
        self,
        time: float,
        perimeter: float,
        port_area: float,
        propellant_area: float,
        burn_rate: float,
        phi: torch.Tensor | None = None,
    ) -> None:
        """Record one step's scalars; pass ``phi`` only on snapshot steps.

        ``phi`` is cloned onto the CPU so the stored history is immune to the
        runner mutating the live field (and doesn't pin GPU memory).
        """
        self.times.append(time)
        self.burning_perimeter.append(perimeter)
        self.port_area.append(port_area)
        self.propellant_area.append(propellant_area)
        self.burn_rate.append(burn_rate)
        if phi is not None:
            self.snapshot_times.append(time)
            self.phi_snapshots.append(phi.detach().cpu().clone())

    # --- convenience views -------------------------------------------------

    def times_tensor(self) -> torch.Tensor:
        return torch.tensor(self.times)

    def burning_perimeter_tensor(self) -> torch.Tensor:
        return torch.tensor(self.burning_perimeter)

    def port_area_tensor(self) -> torch.Tensor:
        return torch.tensor(self.port_area)

    def propellant_area_tensor(self) -> torch.Tensor:
        return torch.tensor(self.propellant_area)

    def __len__(self) -> int:
        """Number of recorded steps."""
        return len(self.times)

    def to_dict(self) -> dict:
        """Serializable dict of the scalar history (snapshots excluded --
        they are bulky tensors; save them separately if needed)."""
        return {
            "times": list(self.times),
            "burning_perimeter": list(self.burning_perimeter),
            "port_area": list(self.port_area),
            "propellant_area": list(self.propellant_area),
            "burn_rate": list(self.burn_rate),
            "snapshot_times": list(self.snapshot_times),
            "burned_out": self.burned_out,
            "burnout_time": self.burnout_time,
            "stopped": self.stopped,
        }
