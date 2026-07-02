"""2D plotting helpers for grain geometry and the level-set field.

These render a φ field produced by :meth:`GrainGeometry.initialize_grid` as a
heatmap with the burning surface (the φ = 0 contour) overlaid -- the standard
diagnostic view for inspecting a burnback simulation -- plus the metric
time-history plots (burning perimeter / port area vs time) and the burn
regression animation (issues #69, #70).

Styling: simulated data is always the same blue; analytical references are a
dashed neutral grey so the two can never be confused. Grids and spines are kept
recessive so the data carries the plot. The φ heatmap uses a diverging palette
(RdBu) because the field is signed -- propellant below zero, void above, with
the burning surface at the neutral midpoint.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from matplotlib.axes import Axes

# Fixed series colors (colorblind-safe, validated): blue for simulated data,
# neutral dashed grey for analytical references.
SIM_COLOR = "#2a78d6"
REF_COLOR = "#5f5e59"
GRID_COLOR = "#d5d4cd"


def _style_axes(ax: Axes) -> None:
    """Recessive grid and spines so the data stands out."""
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


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


# --- metric time histories (#69) ---------------------------------------------


def _plot_history(
    ax: Axes,
    times: Sequence[float],
    values: Sequence[float],
    ylabel: str,
    analytical: Sequence[float] | None,
    title: str | None,
) -> None:
    """Shared body of the two metric plots: simulated curve + optional
    analytical reference, one y-axis, legend only when both series exist."""
    ax.plot(times, values, color=SIM_COLOR, linewidth=2, label="simulated")
    if analytical is not None:
        ax.plot(
            times,
            analytical,
            color=REF_COLOR,
            linewidth=1.5,
            linestyle="--",
            label="analytical",
        )
        ax.legend(frameon=False)
    _style_axes(ax)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def plot_burning_perimeter(
    times: Sequence[float],
    perimeters: Sequence[float],
    ax: Axes,
    *,
    analytical: Sequence[float] | None = None,
    title: str | None = "Burning perimeter vs time",
) -> None:
    """Burning perimeter history, optionally against the analytical 2πr(t).

    ``analytical`` must be sampled at the same ``times`` as the simulated data.
    """
    _plot_history(ax, times, perimeters, "burning perimeter", analytical, title)


def plot_port_area(
    times: Sequence[float],
    areas: Sequence[float],
    ax: Axes,
    *,
    analytical: Sequence[float] | None = None,
    title: str | None = "Port area vs time",
) -> None:
    """Port (open bore) area history, optionally against the analytical πr(t)²."""
    _plot_history(ax, times, areas, "port area", analytical, title)


# --- burn regression animation (#70) ------------------------------------------


def animate_burnback(
    phi_snapshots: Sequence[torch.Tensor],
    snapshot_times: Sequence[float],
    coords: tuple[torch.Tensor, ...],
    *,
    show_casing: float | None = None,
    interval_ms: int = 200,
    save_path: str | None = None,
):
    """Animate the burn regression from the stored φ snapshots.

    Each frame redraws the φ heatmap with its zero contour (the burning
    surface), so watching the contour march outward is a direct visual check
    of the simulation. Returns the ``FuncAnimation``; keep a reference to it
    while displaying interactively (matplotlib animations are garbage-collected
    otherwise).

    Parameters
    ----------
    phi_snapshots, snapshot_times:
        As stored by :class:`~srm_burnback.simulation.SimulationResults`.
    coords:
        The ``(X, Y)`` grid the snapshots live on.
    show_casing:
        Casing radius to overlay as a dashed circle.
    interval_ms:
        Delay between frames.
    save_path:
        When given, write the animation to this file (``.gif`` via Pillow).
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    if len(phi_snapshots) == 0:
        raise ValueError("no phi snapshots to animate")

    fig, ax = plt.subplots(figsize=(6, 6))

    def draw(frame: int):
        plot_phi_field(
            phi_snapshots[frame],
            coords,
            ax,
            show_casing=show_casing,
            title=f"t = {snapshot_times[frame]:.3f}",
        )
        return []

    anim = FuncAnimation(
        fig, draw, frames=len(phi_snapshots), interval=interval_ms, blit=False
    )
    if save_path is not None:
        anim.save(save_path, writer=PillowWriter(fps=max(1, round(1000 / interval_ms))))
    return anim
