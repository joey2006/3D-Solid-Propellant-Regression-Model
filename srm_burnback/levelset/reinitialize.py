"""Signed-distance reinitialization (Sussman-Smereka-Osher 1994).

As the level set is advected, phi drifts away from a true distance field --
``|grad phi|`` stops being 1, first far from the front and eventually near it,
which corrupts the Godunov gradients the scheme depends on. Reinitialization
repairs this by relaxing a *separate* PDE in fictitious time ``tau`` to steady
state:

    d(phi)/d(tau) + S(phi_0) * (|grad phi| - 1) = 0,

which drives ``|grad phi|`` back toward 1 while holding the zero level set (the
burning surface) in place. The interface is preserved because the smoothed sign
``S(phi_0)`` vanishes there, so the PDE has zero speed exactly on the surface.

    S(phi_0) = phi_0 / sqrt(phi_0^2 + h^2)

``phi_0`` is the field *before* reinitialization: the sign is frozen so the
characteristics always point outward from the original interface. The pseudo-
timestep ``d(tau) = 0.5 * h`` keeps the explicit update stable, and 5-10
iterations is plenty -- too many can let the contour drift.

Relation to the brute-force engine (#3)
---------------------------------------
:func:`srm_burnback.sdf.signed_distance_from_sign` also produces a distance
field, but by exact brute force (used for geometry import and as the accuracy
reference). This PDE version is the cheap, local per-step workhorse: it only
needs the current neighborhood and, in narrow-band mode, only touches cells near
the front. The full computational narrow-band/fast-sweep optimization is Phase 8
(#18); here the band crops the work to the interface's bounding box.

Dimension-agnostic: every operation is elementwise or loops over ``range(ndim)``
inside the Godunov call, so 2D and 3D share this code unchanged.
"""

from __future__ import annotations

import torch

from .godunov import godunov_gradient_magnitude


def _smoothed_sign(phi0: torch.Tensor, h: float) -> torch.Tensor:
    """Smoothed sign ``S(phi0) = phi0 / sqrt(phi0^2 + h^2)``.

    Transitions smoothly from -1 to +1 across the interface (spread over ~one
    cell), passing through 0 at ``phi0 = 0``. The smoothing avoids the
    discontinuity of a hard ``sign`` at the surface, which would destabilize the
    pseudo-time iteration. Always lies in ``[-1, 1]``.
    """
    return phi0 / torch.sqrt(phi0**2 + h**2)


def _reinit_iterations(
    phi: torch.Tensor, phi0: torch.Tensor, h: float, n_iterations: int
) -> torch.Tensor:
    """Run the reinitialization pseudo-time loop over a (sub)grid.

    ``phi0`` supplies the frozen sign; ``phi`` is the field being relaxed. The
    Godunov gradient is upwinded by ``sign(phi0)`` so information flows away from
    the interface, matching the direction of the reinitialization characteristics.
    """
    sign0 = torch.sign(phi0)
    smooth_sign = _smoothed_sign(phi0, h)
    dtau = 0.5 * h

    out = phi
    for _ in range(n_iterations):
        grad = godunov_gradient_magnitude(out, h, F_sign=sign0)
        out = out - dtau * smooth_sign * (grad - 1.0)
    return out


def reinitialize(
    phi: torch.Tensor,
    h: float,
    n_iterations: int = 5,
    band_width: float | None = None,
) -> torch.Tensor:
    """Restore ``phi`` to a signed-distance field, fixing numerical drift.

    Parameters
    ----------
    phi:
        The (possibly distorted) level-set field, 2D or 3D.
    h:
        Grid spacing.
    n_iterations:
        Number of pseudo-time steps. 5-10 restores ``|grad phi| ~ 1`` without
        letting the zero contour drift.
    band_width:
        When given, only cells within ``band_width * h`` of the interface are
        reinitialized; the rest keep their values. The work is cropped to that
        band's bounding box, so a localized front gives a real speedup. When
        ``None`` the whole grid is reinitialized.

    Returns
    -------
    torch.Tensor
        A corrected field, the same shape as ``phi``, with the zero level set
        unmoved.
    """
    phi0 = phi.clone()

    if band_width is None:
        return _reinit_iterations(phi.clone(), phi0, h, n_iterations)

    band = phi0.abs() < band_width * h
    if not band.any():
        return phi.clone()

    # Crop the heavy Godunov work to the band's bounding box. The band carries a
    # margin beyond the 3-cell stencil, so the replicate-padded crop edges sit
    # outside the band and never contaminate the cells we keep.
    idx = band.nonzero()
    lo = idx.amin(dim=0)
    hi = idx.amax(dim=0) + 1
    crop = tuple(slice(int(lo[d]), int(hi[d])) for d in range(phi.ndim))

    sub0 = phi0[crop]
    sub_band = band[crop]
    sub = _reinit_iterations(phi[crop].clone(), sub0, h, n_iterations)

    out = phi.clone()
    # Only the band cells inside the crop are updated; everything else untouched.
    out[crop] = torch.where(sub_band, sub, phi[crop])
    return out
