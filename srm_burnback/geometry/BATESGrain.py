"""BATES (tubular) grain geometry.

A BATES grain is a solid cylinder of propellant with a circular bore down the
centre. It is the simplest grain shape and the one we use to validate the whole
simulation, because its burn-back has a known analytical answer.
"""

from __future__ import annotations
import math
import torch
from .GrainGeometry import Coords, GrainGeometry


class BATESGrain(GrainGeometry):
    """A cylindrical grain with a circular central bore.

    Parameters
    ----------
    inner_radius:
        Radius of the bore (the open hole down the middle).
    outer_radius:
        Radius of the casing wall. The burn cannot advance past this.
    length:
        Axial length of the grain. Only used in 3D; ignored in 2D.
    """

    def __init__(self, inner_radius: float, outer_radius: float, length: float | None = None) -> None:
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.length = length

    def signed_distance(self, coords: Coords) -> torch.Tensor:
        X, Y = coords[0], coords[1]

        # Radial distance from the central axis at every grid point.
        r = torch.sqrt(X**2 + Y**2)

        # The bore SDF: positive inside the bore (r < inner_radius), negative in
        # the propellant, zero exactly on the bore wall.
        phi = self.inner_radius - r

        # In 3D the grain also has two end faces. A point is in the open void if
        # it is past either end (|z| > length/2). We combine the two surfaces by
        # taking the element-wise maximum, matching the formula in issue #2.
        if len(coords) == 3:
            if self.length is None:
                raise ValueError("length must be set for a 3D BATES grain")
            Z = coords[2]
            end_faces = torch.abs(Z) - self.length / 2
            phi = torch.maximum(phi, end_faces)

        return phi
    

    def outer_boundary_distance(self, coords: Coords) -> torch.Tensor:
        # The casing wall is the cylinder at outer_radius. Same sign convention:
        # negative inside the casing, positive outside, zero on the wall.
        X, Y = coords[0], coords[1]
        r = torch.sqrt(X**2 + Y**2)
        return r - self.outer_radius

    def default_domain_size(self) -> float:
        # Extend 5% past the casing so the outer wall isn't on the grid edge.
        return self.outer_radius * 1.05


    def analytical_radius(self, t: float, burn_rate: float) -> float:
        """Bore radius at time ``t`` under a uniform, non-erosive burn rate.

        ``r(t) = r_inner + burn_rate * t``. Idealisation for validation only.
        """
        return self.inner_radius + burn_rate * t

    def analytical_burnout_time(self, burn_rate: float) -> float:
        """Time for the bore to reach the casing wall (web fully burned).

        ``t = (r_outer - r_inner) / burn_rate``. Assumes a uniform, non-erosive
        burn rate -- idealisation for validation only.
        """
        return (self.outer_radius - self.inner_radius) / burn_rate

    def analytical_burning_perimeter(self, t: float, burn_rate: float) -> float:
        """Circumference of the expanding bore at time ``t``.

        ``2 * pi * r(t)``. Assumes a uniform, non-erosive burn rate --
        idealisation for validation only.
        """
        return 2.0 * math.pi * self.analytical_radius(t, burn_rate)

    def analytical_port_area(self, t: float, burn_rate: float) -> float:
        """Open cross-sectional (port) area of the bore at time ``t``.

        ``pi * r(t)**2``. Assumes a uniform, non-erosive burn rate --
        idealisation for validation only.
        """
        return math.pi * self.analytical_radius(t, burn_rate) ** 2
