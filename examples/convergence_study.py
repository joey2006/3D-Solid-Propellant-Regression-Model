"""Grid and temporal convergence study with publication-quality plots (#76).

Runs the same studies as ``tests/test_convergence.py`` (they share the study
functions) and renders the two log-log convergence plots for the validation
documentation / paper:

* spatial: max bore-radius error vs grid spacing h, with an O(h) reference
  (the Godunov Hamiltonian uses first-order one-sided differences);
* temporal: TVD-RK3 self-convergence error vs dt, with an O(dt^3) reference.

Outputs land in ``examples/output/``.

Run with:  python examples/convergence_study.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
from test_convergence import (  # noqa: E402  (shared study code)
    fit_loglog_slope,
    run_spatial_study,
    run_temporal_study,
)

SIM_COLOR = "#2a78d6"
REF_COLOR = "#5f5e59"
OUT_DIR = Path(__file__).parent / "output"


def _loglog_panel(ax, x, y, order, xlabel, ylabel, title, ref_label):
    """One convergence panel: measured points + dashed O(x^order) reference
    anchored at the finest point, slopes direct-labeled."""
    slope = fit_loglog_slope(x, y)
    ax.loglog(x, y, "o-", color=SIM_COLOR, linewidth=2, markersize=7,
              label=f"measured (slope {slope:.2f})")
    ref = [y[-1] * (xi / x[-1]) ** order for xi in x]
    ax.loglog(x, ref, "--", color=REF_COLOR, linewidth=1.5, label=ref_label)
    ax.grid(True, which="both", color="#d5d4cd", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    return slope


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    print("spatial study (50/100/200/400)...")
    hs, spatial_errs = run_spatial_study()
    print("temporal study (self-convergence, float64)...")
    dts, temporal_errs = run_temporal_study()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    s1 = _loglog_panel(
        ax1, hs, spatial_errs, 1,
        "$h$", "max bore-radius error",
        "Spatial convergence (Godunov, first order)",
        ref_label="$O(h)$ reference",
    )
    s2 = _loglog_panel(
        ax2, dts, temporal_errs, 3,
        r"$\Delta t$", r"$\|\phi - \phi_{ref}\|_\infty$",
        "Temporal convergence (TVD-RK3, third order)",
        ref_label=r"$O(\Delta t^3)$ reference",
    )
    fig.suptitle("BATES 2D convergence study")
    fig.tight_layout()
    out = OUT_DIR / "convergence.png"
    fig.savefig(out, dpi=150)

    print(f"spatial slope:  {s1:.2f}  (expect ~1: first-order Godunov)")
    print(f"temporal slope: {s2:.2f}  (expect ~3: TVD-RK3)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
