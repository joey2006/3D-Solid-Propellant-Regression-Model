"""Full BATES 2D burnback demo (issue #71).

Runs the complete Phase 1 validation case -- a BATES grain burning at a uniform
Vieille rate -- and produces every standard diagnostic:

* φ field snapshots (start / mid-burn / burnout),
* burning perimeter and port area vs time, against the analytical curves,
* an animated GIF of the burn regression,
* printed accuracy metrics vs the closed-form answer r(t) = r0 + F t.

Outputs land in ``examples/output/``.

NOTE: the uniform burn rate here is the *validation idealization* -- it exists
because a BATES circle has a closed-form answer to check the numerics against.
A real motor always burns erosively (faster aft than fore); actual motor
predictions must use the erosive model when it lands (Phase 3, #13).

Run with:  python examples/bates_2d.py
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed
import matplotlib.pyplot as plt

from srm_burnback.geometry.BATESGrain import BATESGrain
from srm_burnback.physics.vieille import VieilleBurnRate
from srm_burnback.simulation import BurnbackSimulation, SimulationConfig
from srm_burnback.visualization import (
    animate_burnback,
    plot_burning_perimeter,
    plot_phi_field,
    plot_port_area,
)

# --- the validation case: metres, seconds, MPa --------------------------------
INNER = 0.01   # bore radius
OUTER = 0.05   # casing radius
A_COEF = 0.005  # Vieille coefficient (MPa convention)
N_EXP = 0.35   # pressure exponent
PRESSURE = 7.0  # chamber pressure, MPa
F_RATE = A_COEF * PRESSURE**N_EXP  # ~0.00988 m/s, uniform (validation only)

OUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    grain = BATESGrain(inner_radius=INNER, outer_radius=OUTER)
    # reinit_interval=20: with a uniform circular burn phi stays an exact SDF,
    # so reinitialization is nearly pure drift here -- run it sparsely. Real
    # (erosive) simulations keep the default of 5.
    config = SimulationConfig(
        resolution=200, pressure=PRESSURE, max_time=10.0,
        reinit_interval=20, verbose=True,
    )
    sim = BurnbackSimulation(config, grain, VieilleBurnRate(a=A_COEF, n=N_EXP))

    print(f"BATES 2D: bore {INNER} -> casing {OUTER}, F = {F_RATE:.6f}")
    results = sim.run()

    # --- accuracy vs the analytical oracle ------------------------------------
    t_ana = grain.analytical_burnout_time(F_RATE)
    burnout_err = abs(results.burnout_time - t_ana) / t_ana
    radius_errs = []
    for t, p in zip(results.times, results.burning_perimeter):
        r_ana = grain.analytical_radius(t, F_RATE)
        if INNER < r_ana < OUTER - 2 * sim.h:  # skip the final cell at the wall
            radius_errs.append(abs(p / (2 * math.pi) - r_ana) / r_ana)

    print()
    print(f"burnout time:  simulated {results.burnout_time:.4f} s, "
          f"analytical {t_ana:.4f} s  ({burnout_err * 100:.2f}% error)")
    print(f"max bore-radius error vs r0 + F*t: {max(radius_errs) * 100:.3f}%")
    print(f"steps: {len(results)}, phi snapshots: {len(results.phi_snapshots)}")

    # --- φ field snapshots: start / mid-burn / burnout -------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    idx = [0, len(results.phi_snapshots) // 2, -1]
    for ax, i in zip(axes, idx):
        plot_phi_field(
            results.phi_snapshots[i], sim.coords, ax,
            show_casing=OUTER, title=f"t = {results.snapshot_times[i]:.3f} s",
        )
    fig.suptitle("BATES 2D burn regression (uniform rate -- validation case)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bates_2d_phi_fields.png", dpi=150)
    print(f"wrote {OUT_DIR / 'bates_2d_phi_fields.png'}")

    # --- metric histories vs analytical ---------------------------------------
    times = results.times
    perim_ana = [grain.analytical_burning_perimeter(min(t, t_ana), F_RATE) for t in times]
    area_ana = [grain.analytical_port_area(min(t, t_ana), F_RATE) for t in times]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_burning_perimeter(times, results.burning_perimeter, ax1, analytical=perim_ana)
    plot_port_area(times, results.port_area, ax2, analytical=area_ana)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bates_2d_metrics.png", dpi=150)
    print(f"wrote {OUT_DIR / 'bates_2d_metrics.png'}")

    # --- burn animation ---------------------------------------------------------
    gif = OUT_DIR / "bates_2d_burn.gif"
    animate_burnback(
        results.phi_snapshots, results.snapshot_times, sim.coords,
        show_casing=OUTER, interval_ms=200, save_path=str(gif),
    )
    print(f"wrote {gif}")


if __name__ == "__main__":
    main()
