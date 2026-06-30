"""Burn-rate models: how fast the propellant surface recedes."""

from .base import BurnRateModel
from .vieille import VieilleBurnRate

__all__ = ["BurnRateModel", "VieilleBurnRate"]
