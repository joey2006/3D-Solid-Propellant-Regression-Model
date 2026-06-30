"""Abstract interface for burn-rate models.

A burn-rate model answers one question: how fast is the propellant surface
receding, i.e. what is the speed field ``F`` the level set is advected by? Every
model -- the uniform Vieille law here in Phase 1, the erosive Lenoir-Robillard
model in Phase 3 -- implements the same :meth:`BurnRateModel.compute_burn_rate`
method, so the simulation runner can swap models without changing any of its
own code.

Scalar vs. field
----------------
``compute_burn_rate`` may return either a single scalar (uniform burn -- the
same rate at every surface point) or a tensor matching the grid shape
(spatially varying burn). The downstream level-set machinery (``_rhs``,
``tvd_rk3_step``, ``reinitialize``) already accepts both, so a model upgrading
from scalar to field requires no change anywhere else.

Erosive burning is always physically present
--------------------------------------------
The uniform scalar rate (Vieille only) is a **validation/staging idealization**:
it exists to check the level-set numerics against a closed-form answer and to
let Phase 1 be built before the erosive physics lands. A real motor always has
erosive burning -- combustion gas accelerates toward the nozzle, axial mass flux
rises aft, and the local rate climbs toward the aft end -- so any actual motor
simulation must run the erosive model (Lenoir-Robillard). The shared ``**kwargs``
on the interface is what lets erosive models accept the extra inputs they need
(mass flux, hydraulic diameter, ...) that the uniform law ignores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BurnRateModel(ABC):
    """Interface every burn-rate model implements.

    Subclasses store their defining parameters (e.g. ``a`` and ``n`` for
    Vieille) and compute the surface regression speed ``F`` from the current
    conditions.
    """

    @abstractmethod
    def compute_burn_rate(
        self, pressure: float | torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Surface regression speed ``F``.

        Parameters
        ----------
        pressure:
            Chamber pressure (or a per-point pressure field). Units must be
            consistent with the model's coefficients.
        **kwargs:
            Extra inputs specific to richer models (e.g. axial mass flux and
            hydraulic diameter for Lenoir-Robillard). Ignored by uniform models.

        Returns
        -------
        torch.Tensor
            The burn rate ``F``. A 0-d tensor for a uniform rate, or a tensor
            matching the grid for a spatially varying rate.
        """
