"""φ from a boundary: winding-number sign + closest-point distance (#158).

This is the shared front end for every non-analytical geometry -- uploaded
meshes (#138), extruded cross-sections (#139), CT scans (#123), the generic
``PolygonGrain`` (#122). It turns a *boundary* (an oriented polyline in 2D, an
oriented triangle mesh in 3D) into the volumetric field the level-set solver
evolves.

Two halves, computed in one pass
--------------------------------
An SDF needs a magnitude *and* a sign, and for arbitrary user-supplied geometry
neither is available in closed form.

**The sign comes from the generalized winding number.** The obvious method --
ray casting, counting crossing parity -- is only correct on a perfectly closed
surface. Real uploaded STLs have gaps, flipped faces and self-intersections,
and a ray fired through a gap escapes, inverting the parity for everything
behind it: one hole mis-signs a whole region. The generalized winding number
(Jacobson, Kavan & Sorkine 2013) instead integrates how much of the full
surround the boundary covers, so a hole *nudges* the value rather than flipping
it. On a watertight sphere ``w`` is 1.0 inside and 0.0 outside; with 20% of
faces deleted it degrades to ~0.67 / ~0.10 -- still cleanly separated by the
0.5 threshold. That graceful degradation is what makes "upload any object"
work at all.

**The distance comes from the primitives**, not from the grid. The tempting
shortcut is to hand a ±1 sign field to :func:`sdf.signed_distance_from_sign`
and let it locate the surface by interpolating between opposite-signed
neighbours. But that interpolation is ``frac = lo / (lo - hi)``, which for a
raw ±1 field is *exactly* 0.5 -- every crossing lands at a cell-edge midpoint
and all sub-grid information is destroyed before the distance code runs.
Measured on a sphere at 64³: 0.49 h max error that way, versus 0.34 h from real
magnitudes, and 6× worse in the mean.

That error is permanent. Reinitialization rebuilds distance from wherever φ
currently crosses zero, so it faithfully preserves a surface placed at
midpoints -- it repairs ``|∇φ|``, not surface position. A geometry error
introduced at import contaminates every burn-area and thrust number for the
whole run.

Taking distance from the primitives instead bounds the error by **tessellation
density**, which the user controls through their CAD export tolerance, rather
than by **grid spacing h**, which we impose. And it is nearly free: the winding
number already visits every (grid point, primitive) pair, and that traversal is
the expensive part. Closest-point-on-primitive rides along in the same pass.

This does not replace :func:`sdf.signed_distance_from_sign`. That is still
required for reinitialization, where φ carries real magnitudes and
reconstruction is accurate. Import and reinit have genuinely different
information available and correctly use different code.

Dimension-agnostic
------------------
2D takes oriented segments and accumulates the signed angle each subtends,
normalising by 2π. 3D takes oriented triangles and accumulates solid angle via
Van Oosterom--Strackee, normalising by 4π. The structure is identical; only the
per-primitive kernel differs, selected on ``vertices.shape[-1]``.
"""

from __future__ import annotations

import torch

#: ``w`` above this counts as inside the solid. Exactly halfway between the 1.0
#: a closed surface gives inside and the 0.0 it gives outside, which is what
#: leaves the most room for a damaged mesh to degrade without crossing over.
INSIDE_THRESHOLD = 0.5

#: Target (point, primitive) pairs per block. The kernels hold ~15 intermediates
#: of this size, so 2^22 pairs is roughly 250 MB of working set in float32 --
#: comfortable on an 8 GB card while still giving the GPU enough work per launch
#: to stay saturated.
PAIRS_PER_BLOCK = 1 << 22


def _blocks(n_points: int, n_primitives: int) -> tuple[int, int]:
    """Point- and primitive-block sizes that keep one block near the budget.

    Chunking both axes rather than one matters: a 256³ grid against a 50k-face
    mesh is 8.4e11 pairs. Materialising even one intermediate over a whole axis
    would be hundreds of gigabytes, so neither loop can be left whole.
    """
    primitive_block = max(1, min(n_primitives, 1024))
    point_block = max(1, min(n_points, PAIRS_PER_BLOCK // primitive_block))
    return point_block, primitive_block


def _closest_point_on_segment(
    points: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Closest point on segment ``ab`` to each point: clamped projection."""
    ab = b - a
    denominator = (ab * ab).sum(-1).clamp_min(torch.finfo(ab.dtype).tiny)
    t = ((points - a) * ab).sum(-1) / denominator
    return a + ab * t.clamp(0.0, 1.0).unsqueeze(-1)


def _closest_point_on_triangle(
    points: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Closest point on triangle ``abc`` to each point.

    The standard seven-region Voronoi case analysis (Ericson, *Real-Time
    Collision Detection*, §5.1.5): the closest point lies either inside the
    face, on one of three edges, or at one of three vertices. Written branchless
    -- every region is evaluated and selected by mask -- because this runs over
    hundreds of millions of pairs on the GPU, where divergent branches cost far
    more than the redundant arithmetic.
    """
    ab = b - a
    ac = c - a

    d1 = (ab * (points - a)).sum(-1)
    d2 = (ac * (points - a)).sum(-1)
    d3 = (ab * (points - b)).sum(-1)
    d4 = (ac * (points - b)).sum(-1)
    d5 = (ab * (points - c)).sum(-1)
    d6 = (ac * (points - c)).sum(-1)

    # Barycentric-style edge functions. Their signs identify the region.
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    tiny = torch.finfo(points.dtype).tiny

    # Default: the projection falls inside the face.
    denominator = (va + vb + vc)
    denominator = torch.where(denominator.abs() < tiny, torch.full_like(denominator, tiny), denominator)
    v = (vb / denominator).unsqueeze(-1)
    w = (vc / denominator).unsqueeze(-1)
    closest = a + ab * v + ac * w

    def _select(mask: torch.Tensor, value: torch.Tensor) -> None:
        nonlocal closest
        closest = torch.where(mask.unsqueeze(-1), value, closest)

    # Edges, then vertices: the vertex regions are subsets of the edge tests in
    # degenerate cases, so applying them last lets them win.
    edge_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = (d1 / (d1 - d3).clamp_min(tiny)).unsqueeze(-1)
    _select(edge_ab, a + ab * t)

    edge_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = (d2 / (d2 - d6).clamp_min(tiny)).unsqueeze(-1)
    _select(edge_ac, a + ac * t)

    edge_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = ((d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp_min(tiny)).unsqueeze(-1)
    _select(edge_bc, b + (c - b) * t)

    _select((d1 <= 0) & (d2 <= 0), a.expand_as(closest))
    _select((d3 >= 0) & (d4 <= d3), b.expand_as(closest))
    _select((d6 >= 0) & (d5 <= d6), c.expand_as(closest))

    return closest


def _segment_angle(
    points: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Signed angle segment ``ab`` subtends at each point (2D).

    Summed over a closed polyline this is 2π inside and 0 outside -- the planar
    winding number, and the exact 2D analogue of accumulated solid angle.
    """
    pa = a - points
    pb = b - points
    cross = pa[..., 0] * pb[..., 1] - pa[..., 1] * pb[..., 0]
    dot = (pa * pb).sum(-1)
    return torch.atan2(cross, dot)


def _triangle_solid_angle(
    points: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Signed solid angle triangle ``abc`` subtends at each point (3D).

    Van Oosterom & Strackee (1983): for the tetrahedron formed by the point and
    the triangle,

        tan(Ω/2) = |A B C| / (|A||B||C| + (A·B)|C| + (A·C)|B| + (B·C)|A|)

    Evaluated with ``atan2`` rather than ``atan`` so it stays correct across the
    full range instead of wrapping when the denominator turns negative -- which
    is precisely what happens for points close to the surface, the region we
    care most about getting right.
    """
    pa = a - points
    pb = b - points
    pc = c - points

    na = pa.norm(dim=-1)
    nb = pb.norm(dim=-1)
    nc = pc.norm(dim=-1)

    numerator = (pa * torch.linalg.cross(pb, pc, dim=-1)).sum(-1)
    denominator = (
        na * nb * nc
        + (pa * pb).sum(-1) * nc
        + (pa * pc).sum(-1) * nb
        + (pb * pc).sum(-1) * na
    )
    return 2.0 * torch.atan2(numerator, denominator)


def winding_number_and_distance(
    points: torch.Tensor,
    vertices: torch.Tensor,
    primitives: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized winding number and unsigned distance, in one traversal.

    Parameters
    ----------
    points:
        ``(N, D)`` query points. ``D`` is 2 or 3.
    vertices:
        ``(V, D)`` boundary vertices.
    primitives:
        ``(M, D)`` vertex indices -- segments ``(M, 2)`` in 2D, triangles
        ``(M, 3)`` in 3D. Orientation matters: they must be consistently wound
        so that the accumulated angle has a consistent sign.

    Returns
    -------
    (winding, distance):
        ``winding`` is ~1 strictly inside a closed boundary and ~0 outside,
        sliding smoothly between the two near a defect. ``distance`` is the
        unsigned distance to the nearest primitive.
    """
    if points.ndim != 2:
        raise ValueError(f"points must be (N, D), got {tuple(points.shape)}")
    dim = points.shape[-1]
    if dim not in (2, 3):
        raise ValueError(f"only 2D and 3D are supported, got D={dim}")
    if vertices.shape[-1] != dim:
        raise ValueError(
            f"vertices are {vertices.shape[-1]}D but points are {dim}D"
        )
    if primitives.shape[-1] != dim:
        raise ValueError(
            f"{dim}D expects {dim} indices per primitive, got "
            f"{primitives.shape[-1]}"
        )

    # The boundary follows the query points onto whatever device they are on.
    # A mesh arrives from trimesh on the CPU while the grid is typically built
    # on the GPU, and making every caller remember to move it is a footgun that
    # only shows up as a device mismatch deep inside a kernel.
    vertices = vertices.to(device=points.device, dtype=points.dtype)
    corners = vertices[primitives.to(device=points.device).long()]  # (M, D, D)

    n_points = points.shape[0]
    n_primitives = corners.shape[0]
    point_block, primitive_block = _blocks(n_points, n_primitives)

    angle = torch.zeros(n_points, dtype=points.dtype, device=points.device)
    squared = torch.full(
        (n_points,), float("inf"), dtype=points.dtype, device=points.device
    )

    for start in range(0, n_points, point_block):
        stop = min(start + point_block, n_points)
        block = points[start:stop].unsqueeze(1)  # (P, 1, D)

        angle_sum = torch.zeros(
            stop - start, dtype=points.dtype, device=points.device
        )
        best = torch.full(
            (stop - start,), float("inf"), dtype=points.dtype, device=points.device
        )

        for first in range(0, n_primitives, primitive_block):
            last = min(first + primitive_block, n_primitives)
            chunk = corners[first:last].unsqueeze(0)  # (1, C, D, D)

            if dim == 3:
                a, b, c = chunk[..., 0, :], chunk[..., 1, :], chunk[..., 2, :]
                angle_sum += _triangle_solid_angle(block, a, b, c).sum(-1)
                closest = _closest_point_on_triangle(block, a, b, c)
            else:
                a, b = chunk[..., 0, :], chunk[..., 1, :]
                angle_sum += _segment_angle(block, a, b).sum(-1)
                closest = _closest_point_on_segment(block, a, b)

            offset = closest - block
            best = torch.minimum(best, (offset * offset).sum(-1).amin(dim=-1))

        angle[start:stop] = angle_sum
        squared[start:stop] = best

    full_turn = 4.0 * torch.pi if dim == 3 else 2.0 * torch.pi
    return angle / full_turn, squared.clamp_min(0.0).sqrt()


def winding_and_split_distance(
    points: torch.Tensor,
    vertices: torch.Tensor,
    primitives: torch.Tensor,
    burning: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Winding number over *all* primitives, distance split by label (#176).

    Returns ``(winding, distance_to_burning, distance_to_inhibited)``.

    The split matters because the two halves need different inputs. **Sign has
    to come from the whole closed surface** -- the winding number measures how
    much of the surround the boundary covers, so feeding it only the burning
    faces would hand it an open patch and a meaningless answer. **Distance has
    to come from the burning faces only**, because the burning surface is what
    φ is supposed to measure distance to.

    Both distances fall out of the one traversal, which is the expensive part.
    """
    if burning.shape[0] != primitives.shape[0]:
        raise ValueError(
            f"burning mask has {burning.shape[0]} entries for "
            f"{primitives.shape[0]} primitives"
        )

    dim = points.shape[-1]
    vertices = vertices.to(device=points.device, dtype=points.dtype)
    corners = vertices[primitives.to(device=points.device).long()]
    burning = burning.to(device=points.device, dtype=torch.bool)

    n_points = points.shape[0]
    n_primitives = corners.shape[0]
    point_block, primitive_block = _blocks(n_points, n_primitives)

    angle = torch.zeros(n_points, dtype=points.dtype, device=points.device)
    infinity = float("inf")
    burn_squared = torch.full(
        (n_points,), infinity, dtype=points.dtype, device=points.device
    )
    inhibited_squared = torch.full(
        (n_points,), infinity, dtype=points.dtype, device=points.device
    )

    for start in range(0, n_points, point_block):
        stop = min(start + point_block, n_points)
        block = points[start:stop].unsqueeze(1)

        angle_sum = torch.zeros(
            stop - start, dtype=points.dtype, device=points.device
        )
        best_burn = torch.full(
            (stop - start,), infinity, dtype=points.dtype, device=points.device
        )
        best_inhibited = torch.full(
            (stop - start,), infinity, dtype=points.dtype, device=points.device
        )

        for first in range(0, n_primitives, primitive_block):
            last = min(first + primitive_block, n_primitives)
            chunk = corners[first:last].unsqueeze(0)
            labels = burning[first:last]

            if dim == 3:
                a, b, c = chunk[..., 0, :], chunk[..., 1, :], chunk[..., 2, :]
                angle_sum += _triangle_solid_angle(block, a, b, c).sum(-1)
                closest = _closest_point_on_triangle(block, a, b, c)
            else:
                a, b = chunk[..., 0, :], chunk[..., 1, :]
                angle_sum += _segment_angle(block, a, b).sum(-1)
                closest = _closest_point_on_segment(block, a, b)

            offset = closest - block
            squared = (offset * offset).sum(-1)
            # Masking with +inf rather than indexing keeps the tensor shape
            # fixed, so the two minima cost one extra reduction each instead of
            # a second traversal.
            masked = torch.where(labels, squared, torch.full_like(squared, infinity))
            best_burn = torch.minimum(best_burn, masked.amin(dim=-1))
            masked = torch.where(labels, torch.full_like(squared, infinity), squared)
            best_inhibited = torch.minimum(best_inhibited, masked.amin(dim=-1))

        angle[start:stop] = angle_sum
        burn_squared[start:stop] = best_burn
        inhibited_squared[start:stop] = best_inhibited

    full_turn = 4.0 * torch.pi if dim == 3 else 2.0 * torch.pi
    return (
        angle / full_turn,
        burn_squared.clamp_min(0.0).sqrt(),
        inhibited_squared.clamp_min(0.0).sqrt(),
    )


def phi_from_labelled_boundary(
    coords: tuple[torch.Tensor, ...],
    vertices: torch.Tensor,
    primitives: torch.Tensor,
    burning: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """φ measured to the burning surface, plus the casing field. ``(phi, phi_outer)``

    This is the physically correct import path. :func:`phi_from_boundary` gives
    the signed distance to the *object*, which is a different thing: its zero
    set includes the outer wall, so a solver advancing it would burn the grain
    inward from the casing as well as outward from the bore.

    The sign rule has three cases, and the third is the one that fixes it:

    * **inside the solid** -- propellant, ``φ = -distance to burning surface``.
    * **outside the solid, nearer a burning face** -- the bore or a slot, so
      genuinely open void: ``φ = +distance``.
    * **outside the solid, nearer an inhibited face** -- past the casing wall.
      This is *not* void the flame can reach, so φ stays **negative** and keeps
      decreasing outward. No second zero crossing appears at the wall, and the
      field matches what ``BATESGrain.signed_distance`` produces analytically.

    "Nearer a burning face" is decided by comparing the two distances rather
    than by identifying the closest face, which is the same test without having
    to carry face indices through the reduction.

    ``phi_outer`` is the casing: negative within the wall, positive beyond it,
    ready for the existing ``min(phi, -phi_outer)`` clamp. With no inhibited
    faces at all it is ``-inf`` everywhere, which the clamp correctly treats as
    "no casing".
    """
    shape = coords[0].shape
    points = torch.stack([c.reshape(-1) for c in coords], dim=-1)

    winding, to_burning, to_inhibited = winding_and_split_distance(
        points, vertices, primitives, burning
    )

    inside = winding > INSIDE_THRESHOLD
    open_void = ~inside & (to_burning <= to_inhibited)
    phi = torch.where(open_void, to_burning, -to_burning)

    # The casing surface, signed the same way as GrainGeometry's:
    # negative inside the wall, positive outside it.
    beyond_casing = ~inside & (to_inhibited < to_burning)
    phi_outer = torch.where(beyond_casing, to_inhibited, -to_inhibited)

    return phi.reshape(shape), phi_outer.reshape(shape)


def phi_from_labelled_mesh(
    coords: tuple[torch.Tensor, ...],
    mesh,
    ends: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """``(phi, phi_outer, labels)`` for a mesh, labelling its faces first.

    ``ends`` selects how the axial end faces are treated -- see
    :mod:`srm_burnback.geometry.surfaces`, which owns that decision.
    """
    from .surfaces import DEFAULT_ENDS, surface_labels

    labels = surface_labels(mesh, ends or DEFAULT_ENDS)

    device = coords[0].device
    vertices = torch.as_tensor(mesh.vertices, dtype=dtype, device=device)
    faces = torch.as_tensor(mesh.faces, dtype=torch.long, device=device)
    burning = torch.as_tensor(labels["burning"], dtype=torch.bool, device=device)

    phi, phi_outer = phi_from_labelled_boundary(coords, vertices, faces, burning)
    return phi, phi_outer, labels


def phi_from_boundary(
    coords: tuple[torch.Tensor, ...],
    vertices: torch.Tensor,
    primitives: torch.Tensor,
) -> torch.Tensor:
    """Signed distance field on ``coords`` from a boundary. The import path.

    Sign convention matches :class:`GrainGeometry`: **positive in the open
    void, negative inside the solid propellant**. So the winding number, which
    is ~1 *inside* the boundary, is inverted here -- the mesh describes the
    propellant, and the propellant is the negative region.

    Parameters
    ----------
    coords:
        ``(X, Y)`` or ``(X, Y, Z)``, each the same shape, as built by
        :func:`CoordinateBuilder.build_coords`.
    vertices, primitives:
        The boundary, as for :func:`winding_number_and_distance`.

    Returns
    -------
    torch.Tensor
        φ with the shape of one coordinate tensor.
    """
    shape = coords[0].shape
    points = torch.stack([c.reshape(-1) for c in coords], dim=-1)

    winding, distance = winding_number_and_distance(points, vertices, primitives)

    inside = winding > INSIDE_THRESHOLD
    phi = torch.where(inside, -distance, distance)
    return phi.reshape(shape)


def phi_from_mesh(
    coords: tuple[torch.Tensor, ...], mesh, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """:func:`phi_from_boundary` for a ``trimesh.Trimesh``, on the coords' device.

    Convenience only -- it exists so the desktop app and the examples do not
    each re-derive the tensor conversion.
    """
    device = coords[0].device
    vertices = torch.as_tensor(mesh.vertices, dtype=dtype, device=device)
    faces = torch.as_tensor(mesh.faces, dtype=torch.long, device=device)
    return phi_from_boundary(coords, vertices, faces)
