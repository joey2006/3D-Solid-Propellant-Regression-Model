"""2D plotting helpers for grain geometry and the level-set field.

These render a φ field produced by :meth:`GrainGeometry.initialize_grid` as a
heatmap with the burning surface (the φ = 0 contour) overlaid -- the standard
diagnostic view for inspecting a burnback simulation.
"""

from __future__ import annotations

import torch
from matplotlib.axes import Axes


def plot_phi_field(
    phi: torch.Tensor,
    coords: tuple[torch.Tensor, ...],
    ax: Axes,
    *,
    show_casing: float | None = None,
    title: str | None = None,
):
    """Draw a 2D φ field on ``ax`` with its φ = 0 contour (the burning surface).

    Parameters
    ----------
    phi:
        2D signed-distance field; positive in the open void, negative in the
        propellant (as returned by ``initialize_grid``).
    coords:
        The ``(X, Y)`` coordinate grid φ was evaluated on.
    ax:
        Matplotlib axes to draw into.
    show_casing:
        If given, draw the casing wall as a circle of this radius.
    title:
        Optional axes title.

    Returns
    -------
    The image handle from ``imshow`` (useful for attaching a colorbar).
    """
    if phi.ndim != 2:
        raise ValueError(f"plot_phi_field expects a 2D field, got {phi.ndim}D")

    X, Y = coords[0], coords[1]
    # Convert to NumPy on the CPU so matplotlib can consume it.
    phi_np = phi.detach().cpu().numpy()
    x = X.detach().cpu().numpy()
    y = Y.detach().cpu().numpy()
    extent = (x.min(), x.max(), y.min(), y.max())

    ax.clear()
    # imshow expects [row, col] = [y, x], so transpose the ij-indexed field.
    img = ax.imshow(
        phi_np.T,
        origin="lower",
        extent=extent,
        cmap="RdBu",
        vmin=-abs(phi_np).max(),
        vmax=abs(phi_np).max(),
    )
    # The burning surface: where φ crosses zero.
    ax.contour(x, y, phi_np, levels=[0.0], colors="black", linewidths=2)

    if show_casing is not None:
        circle = plt_circle(show_casing)
        ax.add_patch(circle)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    return img


def plt_circle(radius: float):
    """A dashed grey circle patch for the casing wall."""
    from matplotlib.patches import Circle

    return Circle(
        (0.0, 0.0),
        radius,
        fill=False,
        linestyle="--",
        edgecolor="grey",
        linewidth=1.5,
    )
