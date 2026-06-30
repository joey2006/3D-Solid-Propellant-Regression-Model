# Notebooks

Jupyter notebooks for **exploration and validation figures only** — not for
core logic.

## What goes here
- Visualizing burnback (φ contours evolving, |∇φ| drift, reinitialization effects)
- Tuning numerics interactively (CFL factor, reinit frequency, band width)
- Validation studies and their figures (BATES burnback vs. analytical, grid /
  temporal convergence log-log plots — feeds issues #10 and #16)
- One-off experiments and sanity checks

## What does NOT go here
- Anything importable that the simulator depends on. All real code lives in the
  `srm_burnback/` package and is covered by `pytest`. Notebooks **import** from
  `srm_burnback`; they never define simulation logic.
- Tests. Those stay as `.py` files under `tests/`.

## Conventions
- Keep notebooks thin: import from `srm_burnback`, call it, plot. If a helper
  grows useful, promote it into the package (with a test) rather than leaving it
  in a cell.
- **Clear outputs before committing** so notebooks diff cleanly in git. The
  easy way is `nbstripout` (`pip install nbstripout && nbstripout --install`),
  which strips outputs automatically on commit. `.ipynb_checkpoints/` is
  gitignored.
- Name by purpose, e.g. `bates_validation.ipynb`, `convergence_study.ipynb`.

## Running
From the repo root, with the package installed (`pip install -e .`):

```
jupyter lab
```
