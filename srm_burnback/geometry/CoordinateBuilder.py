

from __future__ import annotations
import torch
from .GrainGeometry import Coords

def build_coords(
    resolution: int,
    domain_size: float,
    length: float | None = None,
    device: str = "cpu",
) -> tuple[Coords, float]:
 
    x_axis = torch.linspace(-domain_size, domain_size, resolution, device=device)
    axes = [x_axis, x_axis]  # x and y share the same span

    # Add the third dimension only when a length is given.
    if length is not None:
        axes.append(torch.linspace(-length / 2, length / 2, resolution, device=device))

    coords = torch.meshgrid(*axes, indexing="ij")
    height = (x_axis[1] - x_axis[0]).item()  # plain float, not a 0-d tensor
    return coords, height
