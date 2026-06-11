"""Interactive UI for the BATES initial grid (issue #33).

Run it:

    python examples/view_bates_grid.py

A matplotlib window opens showing the initial signed-distance field phi from
``BATESGrain.initialize_grid``: blue = open bore (phi > 0), red = propellant
(phi < 0), and the black contour is the burning surface (phi = 0). Drag the
sliders to change the bore radius, casing radius, and grid resolution and watch
the field rebuild live -- a direct visual check that the zero contour sits at
the bore wall and the casing is contained in the domain.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from srm_burnback.geometry.BATESGrain import BATESGrain
from srm_burnback.visualization.plot2d import plot_phi_field

# Initial parameters.
INNER0, OUTER0, RES0 = 0.5, 2.0, 200

fig, ax = plt.subplots(figsize=(6.5, 7.5))
plt.subplots_adjust(bottom=0.28)

# Colorbar created once against a first render, then reused.
grain = BATESGrain(inner_radius=INNER0, outer_radius=OUTER0)
phi, coords, height = grain.initialize_grid(RES0)
img = plot_phi_field(phi, coords, ax, show_casing=OUTER0, title="BATES initial phi")
cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04, label="phi")

# Sliders.
ax_inner = plt.axes([0.18, 0.16, 0.65, 0.03])
ax_outer = plt.axes([0.18, 0.10, 0.65, 0.03])
ax_res = plt.axes([0.18, 0.04, 0.65, 0.03])
s_inner = Slider(ax_inner, "bore radius", 0.1, 1.8, valinit=INNER0)
s_outer = Slider(ax_outer, "casing radius", 0.5, 4.0, valinit=OUTER0)
s_res = Slider(ax_res, "resolution", 50, 400, valinit=RES0, valstep=10)


def update(_):
    inner = s_inner.val
    outer = max(s_outer.val, inner + 0.05)  # keep casing outside the bore
    res = int(s_res.val)
    grain = BATESGrain(inner_radius=inner, outer_radius=outer)
    phi, coords, _ = grain.initialize_grid(res)
    plot_phi_field(phi, coords, ax, show_casing=outer, title="BATES initial phi")
    fig.canvas.draw_idle()


for s in (s_inner, s_outer, s_res):
    s.on_changed(update)

if __name__ == "__main__":
    plt.show()
