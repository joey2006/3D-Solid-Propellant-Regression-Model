"""Vieille's burn-rate law (a.k.a. Saint-Robert's law): ``F = a * P^n``.

The simplest burn-rate model: the surface recedes at the same speed everywhere,
set only by the chamber pressure. ``a`` (the burn-rate coefficient) and ``n``
(the pressure exponent) are propellant properties; ``P`` is the chamber
pressure, treated as a fixed input in Phase 1 (it is computed from chamber
conditions only in Phase 6).

This produces a *uniform* rate, which is a validation/staging idealization only
(see :mod:`srm_burnback.physics.base`): it gives the closed-form BATES burn-back
a constant speed to check the numerics against. A real motor always burns faster
aft than fore; that erosive behavior arrives with Lenoir-Robillard in Phase 3.
"""

from __future__ import annotations

import torch

from .base import BurnRateModel


class VieilleBurnRate(BurnRateModel):
    """Uniform burn rate ``F = a * P^n``.

    Parameters
    ----------
    a:
        Burn-rate coefficient (units: length/time/pressure^n). For a composite
        propellant in SI, ``a ~ 0.005 m/s/Pa^n``.
    n:
        Pressure exponent, typically 0.3-0.7. Must satisfy stability ``n < 1``
        in a real motor, but the law itself is evaluated for whatever ``n`` is
        given.
    """

    def __init__(self, a: float, n: float) -> None:
        self.a = a
        self.n = n

    def compute_burn_rate(
        self, pressure: float | torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute ``F = a * P^n``.

        Parameters
        ----------
        pressure:
            Chamber pressure ``P >= 0``. May be a scalar or a tensor; the result
            matches its shape (a 0-d tensor for a scalar input -- the uniform
            case). ``P = 0`` gives ``F = 0``.
        **kwargs:
            Unused; accepted so the model honors the :class:`BurnRateModel`
            interface that erosive models extend.

        Raises
        ------
        ValueError
            If any pressure is negative (unphysical, and ``P^n`` is not real).
        """
        P = torch.as_tensor(pressure)
        if not torch.is_floating_point(P):
            # Promote integer/bool pressures so P**n is real-valued.
            P = P.to(torch.get_default_dtype())
        if (P < 0).any():
            raise ValueError("pressure must be non-negative")
        return self.a * P**self.n
