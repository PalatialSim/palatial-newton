# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read-side helpers for NewtonRodAPI USD assets.

Authoring lives in the external converter. This module resolves rod guide
geometry, material parameters, and centerline/radius fallbacks from a
converted USDA so the generic palatial loader can build an isotropic rod in
Newton.
"""
from __future__ import annotations

# `import newton` registers the bundled USD plugins (newton + newton_shell)
# via newton/_src/usd/__init__.py. Must precede any pxr.Usd usage in the
# same process.
import newton  # noqa: F401

from . import _resolvers  # noqa: F401  (kept for parity with shell.py init)

from dataclasses import dataclass
import math

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdShade


DEFAULTS = {
    # geometry
    "segmentCount": None,
    "radius": 0.01,
    # material
    "stretchStiffness": 1.0e5,
    "stretchDamping": 0.0,
    "compressStiffness": 1.0e5,
    "bendYStiffness": 0.0,
    "bendYDamping": 0.0,
    "bendZStiffness": 0.0,
    "bendZDamping": 0.0,
    "torsionStiffness": 0.0,
    "torsionDamping": 0.0,
}


@dataclass
class RodCenterlineSpec:
    """Resolved rod centerline and isotropic radius."""

    points: list[tuple[float, float, float]]
    radius: float
    guide_prim_path: str | None
    centerline_source_path: str | None
    radius_source_path: str | None


@dataclass
class _RigidBodyCandidate:
    prim_path: str
    name: str
    bbox_size: tuple[float, float, float]
    collider_centers: np.ndarray
    collider_point_groups: list[np.ndarray]
    visual_bbox_size: tuple[float, float, float] | None


def _has_rod_api(prim: Usd.Prim) -> bool:
    """True iff the prim carries NewtonRodAPI."""
    applied = set(prim.GetAppliedSchemas())
    raw = prim.GetMetadata("apiSchemas")
    if raw is not None:
        for items_attr in ("prependedItems", "appendedItems", "explicitItems"):
            for tok in getattr(raw, items_attr, []) or []:
                applied.add(str(tok))
    return "NewtonRodAPI" in applied


def _has_rod_material_api(prim: Usd.Prim) -> bool:
    """True iff the prim carries NewtonRodMaterialAPI."""
    applied = set(prim.GetAppliedSchemas())
    raw = prim.GetMetadata("apiSchemas")
    if raw is not None:
        for items_attr in ("prependedItems", "appendedItems", "explicitItems"):
            for tok in getattr(raw, items_attr, []) or []:
                applied.add(str(tok))
    return "NewtonRodMaterialAPI" in applied


def _find_rod_prim(stage: Usd.Stage) -> Usd.Prim | None:
    """Return the first rod guide prim on the stage."""
    found_intent = None
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.BasisCurves):
            continue
        if _has_rod_api(prim):
            return prim
        attr = prim.GetAttribute("newton:deformable:simulationIntent")
        if attr and attr.HasAuthoredValue() and str(attr.Get()) == "rod":
            found_intent = prim
            break
    return found_intent


def find_rod_prim_path(usd_path: str) -> str | None:
    """Return the prim path of the first rod guide on the stage."""
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        return None
    prim = _find_rod_prim(stage)
    return prim.GetPath().pathString if prim else None


def _bound_rod_material_prims(rod_prim: Usd.Prim) -> list[Usd.Prim]:
    """Return Material prims bound to the rod guide that carry NewtonRodMaterialAPI."""
    out: list[Usd.Prim] = []
    binding = UsdShade.MaterialBindingAPI(rod_prim)
    candidates = []
    for purpose in ("", "physics"):
        try:
            result = binding.ComputeBoundMaterial(purpose) if purpose else binding.ComputeBoundMaterial()
        except Exception:
            result = None
        if result is None:
            continue
        material = result[0] if isinstance(result, tuple) else result
        if material and material.GetPrim().IsValid():
            candidates.append(material.GetPrim())

    seen = set()
    for prim in candidates:
        path = prim.GetPath()
        if path in seen:
            continue
        seen.add(path)
        if _has_rod_material_api(prim):
            out.append(prim)
    return out


def _stage_units(stage: Usd.Stage) -> tuple[float, str]:
    """Return (meters_per_unit, up_axis)."""
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    up_axis = str(UsdGeom.GetStageUpAxis(stage) or "Z")
    return meters_per_unit, up_axis


def _to_newton_world(points: np.ndarray, meters_per_unit: float, up_axis: str) -> np.ndarray:
    """Convert USD world-space points into Newton world coordinates."""
    if points.size == 0:
        return points
    out = np.asarray(points, dtype=np.float64) * float(meters_per_unit)
    if up_axis == "Y":
        out = np.stack([out[:, 0], -out[:, 2], out[:, 1]], axis=1)
    return out


def _matrix_transform_points(matrix: Gf.Matrix4d, points: np.ndarray) -> np.ndarray:
    """Apply a USD transform matrix to a point cloud."""
    if points.size == 0:
        return points
    world = np.empty_like(points, dtype=np.float64)
    for i, point in enumerate(points):
        value = matrix.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
        world[i] = (float(value[0]), float(value[1]), float(value[2]))
    return world


def _read_basis_curve_points(
    stage: Usd.Stage,
    prim: Usd.Prim,
    *,
    meters_per_unit: float,
    up_axis: str,
    xform_cache: UsdGeom.XformCache,
) -> list[tuple[float, float, float]]:
    """Read the first authored rod curve in Newton world coordinates."""
    points_attr = prim.GetAttribute("points")
    if not (points_attr and points_attr.HasAuthoredValue()):
        return []
    curve_points = points_attr.Get()
    if curve_points is None:
        return []

    counts_attr = prim.GetAttribute("curveVertexCounts")
    counts = counts_attr.Get() if counts_attr and counts_attr.HasAuthoredValue() else None
    count = int(counts[0]) if counts else len(curve_points)
    if count <= 0:
        return []

    local = np.asarray(
        [(float(point[0]), float(point[1]), float(point[2])) for point in curve_points[:count]],
        dtype=np.float64,
    )
    matrix = xform_cache.GetLocalToWorldTransform(prim)
    world = _matrix_transform_points(matrix, local)
    world = _to_newton_world(world, meters_per_unit, up_axis)
    return [(float(point[0]), float(point[1]), float(point[2])) for point in world]


def _read_mesh_world_points(
    stage: Usd.Stage,
    prim: Usd.Prim,
    *,
    meters_per_unit: float,
    up_axis: str,
    xform_cache: UsdGeom.XformCache,
) -> np.ndarray:
    """Read mesh points in Newton world coordinates."""
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    if points is None:
        return np.empty((0, 3), dtype=np.float64)
    local = np.asarray([(float(point[0]), float(point[1]), float(point[2])) for point in points], dtype=np.float64)
    matrix = xform_cache.GetLocalToWorldTransform(prim)
    world = _matrix_transform_points(matrix, local)
    return _to_newton_world(world, meters_per_unit, up_axis)


def _collect_rigid_body_candidates(
    stage: Usd.Stage,
    *,
    meters_per_unit: float,
    up_axis: str,
    xform_cache: UsdGeom.XformCache,
) -> list[_RigidBodyCandidate]:
    """Collect rigid-body candidates that can act as centerline/radius sources."""
    candidates: list[_RigidBodyCandidate] = []
    for prim in stage.Traverse():
        applied = set(prim.GetAppliedSchemas())
        raw = prim.GetMetadata("apiSchemas")
        if raw is not None:
            for items_attr in ("prependedItems", "appendedItems", "explicitItems"):
                for tok in getattr(raw, items_attr, []) or []:
                    applied.add(str(tok))
        if "PhysicsRigidBodyAPI" not in applied:
            continue

        point_groups: list[np.ndarray] = []
        centers: list[np.ndarray] = []
        all_points: list[np.ndarray] = []
        visual_points: list[np.ndarray] = []
        for child in prim.GetChildren():
            if child.IsA(UsdGeom.Mesh):
                points = _read_mesh_world_points(
                    stage,
                    child,
                    meters_per_unit=meters_per_unit,
                    up_axis=up_axis,
                    xform_cache=xform_cache,
                )
                if points.size > 0:
                    visual_points.append(points)

        for child in prim.GetChildren():
            if not child.GetName().startswith("Colliders_"):
                continue
            for mesh_prim in child.GetChildren():
                if not mesh_prim.IsA(UsdGeom.Mesh):
                    continue
                points = _read_mesh_world_points(
                    stage,
                    mesh_prim,
                    meters_per_unit=meters_per_unit,
                    up_axis=up_axis,
                    xform_cache=xform_cache,
                )
                if points.size == 0:
                    continue
                point_groups.append(points)
                centers.append(points.mean(axis=0))
                all_points.append(points)

        if len(point_groups) < 2:
            continue

        merged = np.concatenate(all_points, axis=0)
        bbox_size = tuple(float(value) for value in (merged.max(axis=0) - merged.min(axis=0)))
        visual_bbox_size = None
        if visual_points:
            merged_visual = np.concatenate(visual_points, axis=0)
            visual_bbox_size = tuple(float(value) for value in (merged_visual.max(axis=0) - merged_visual.min(axis=0)))
        candidates.append(
            _RigidBodyCandidate(
                prim_path=prim.GetPath().pathString,
                name=prim.GetName(),
                bbox_size=bbox_size,
                collider_centers=np.asarray(centers, dtype=np.float64),
                collider_point_groups=point_groups,
                visual_bbox_size=visual_bbox_size,
            )
        )
    return candidates


def _complete_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Return pairwise Euclidean distances for a point set."""
    diff = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diff, axis=2)


def _order_polyline_centers(points: np.ndarray) -> np.ndarray:
    """Order unordered collider centers along a path using an MST traversal."""
    if len(points) <= 2:
        return np.asarray(points, dtype=np.float64)

    dists = _complete_distance_matrix(points)
    count = len(points)
    in_tree = np.zeros(count, dtype=bool)
    in_tree[0] = True
    adjacency = [[] for _ in range(count)]

    while int(in_tree.sum()) < count:
        best_i = -1
        best_j = -1
        best_d = float("inf")
        for i in range(count):
            if not in_tree[i]:
                continue
            for j in range(count):
                if in_tree[j] or i == j:
                    continue
                dist = float(dists[i, j])
                if dist < best_d:
                    best_d = dist
                    best_i = i
                    best_j = j
        if best_i < 0 or best_j < 0:
            break
        adjacency[best_i].append(best_j)
        adjacency[best_j].append(best_i)
        in_tree[best_j] = True

    endpoints = [index for index, neighbors in enumerate(adjacency) if len(neighbors) == 1]
    if len(endpoints) != 2:
        farthest = np.unravel_index(np.argmax(dists), dists.shape)
        order = np.argsort(np.dot(points - points[farthest[0]], points[farthest[1]] - points[farthest[0]]))
        return points[order]

    ordered: list[int] = []
    previous = -1
    current = endpoints[0]
    while True:
        ordered.append(current)
        next_nodes = [node for node in adjacency[current] if node != previous]
        if not next_nodes:
            break
        previous, current = current, next_nodes[0]
    return points[np.asarray(ordered, dtype=int)]


def _endpoint_cost(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """Compute an endpoint-matching cost between an ordered polyline and guide endpoints."""
    return float(np.linalg.norm(points[0] - start) + np.linalg.norm(points[-1] - end))


def _orient_centers_to_guide(points: np.ndarray, guide_points: np.ndarray | None) -> np.ndarray:
    """Reverse an ordered centerline when needed to match guide endpoints."""
    if guide_points is None or len(guide_points) < 2 or len(points) < 2:
        return points
    direct = _endpoint_cost(points, guide_points[0], guide_points[-1])
    flipped = _endpoint_cost(points[::-1], guide_points[0], guide_points[-1])
    return points if direct <= flipped else points[::-1]


def _segment_centers_to_nodes(centers: np.ndarray) -> np.ndarray:
    """Convert ordered segment centers into polyline node positions."""
    if len(centers) < 2:
        return np.asarray(centers, dtype=np.float64)

    nodes = [centers[0] - 0.5 * (centers[1] - centers[0])]
    for i in range(1, len(centers)):
        nodes.append(0.5 * (centers[i - 1] + centers[i]))
    nodes.append(centers[-1] + 0.5 * (centers[-1] - centers[-2]))
    return np.asarray(nodes, dtype=np.float64)


def _polyline_lengths(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Return cumulative arc lengths for a polyline."""
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64), 0.0
    if len(points) == 1:
        return np.zeros(1, dtype=np.float64), 0.0
    seg = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    return cumulative, float(cumulative[-1])


def _resample_polyline(points: np.ndarray, point_count: int) -> np.ndarray:
    """Resample a polyline to a fixed number of points."""
    if point_count <= 0:
        return np.empty((0, 3), dtype=np.float64)
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    if len(points) == 1:
        return np.repeat(points, point_count, axis=0)

    cumulative, total = _polyline_lengths(points)
    if total <= 1.0e-12:
        return np.repeat(points[:1], point_count, axis=0)

    samples = np.linspace(0.0, total, point_count)
    out = np.empty((point_count, 3), dtype=np.float64)
    seg_index = 0
    for i, target in enumerate(samples):
        while seg_index + 1 < len(cumulative) and cumulative[seg_index + 1] < target:
            seg_index += 1
        if seg_index + 1 >= len(points):
            out[i] = points[-1]
            continue
        start_len = cumulative[seg_index]
        end_len = cumulative[seg_index + 1]
        if end_len <= start_len:
            out[i] = points[seg_index]
            continue
        alpha = (target - start_len) / (end_len - start_len)
        out[i] = (1.0 - alpha) * points[seg_index] + alpha * points[seg_index + 1]
    return out


def _candidate_cross_section(candidate: _RigidBodyCandidate) -> float:
    """Return a rough half-width used for candidate ranking."""
    dims = sorted(float(value) for value in candidate.bbox_size)
    if len(dims) < 2:
        return float("inf")
    return 0.5 * dims[1]


def _candidate_keyword_rank(candidate: _RigidBodyCandidate, *, centerline: bool) -> int:
    """Return a keyword preference rank for a rigid-body candidate."""
    lower = candidate.name.lower()
    if centerline:
        if "path" in lower:
            return 0
        if "centerline" in lower or "guide" in lower:
            return 1
        if "jacket" in lower:
            return 3
        return 2
    if "jacket" in lower:
        return 0
    if "cable" in lower and "path" not in lower:
        return 1
    if "path" in lower or "guide" in lower:
        return 3
    return 2


def _select_centerline_candidate(
    candidates: list[_RigidBodyCandidate],
    guide_points: np.ndarray | None,
) -> _RigidBodyCandidate | None:
    """Choose the best rigid-body candidate for centerline extraction."""
    if not candidates:
        return None

    def _score(candidate: _RigidBodyCandidate) -> tuple[float, float, float]:
        ordered = _order_polyline_centers(candidate.collider_centers)
        ordered = _orient_centers_to_guide(ordered, guide_points)
        endpoint_cost = 0.0
        if guide_points is not None and len(guide_points) >= 2:
            endpoint_cost = _endpoint_cost(ordered, guide_points[0], guide_points[-1])
        return (
            float(_candidate_keyword_rank(candidate, centerline=True)),
            float(_candidate_cross_section(candidate)),
            float(endpoint_cost),
        )

    return min(candidates, key=_score)


def _select_radius_candidate(
    candidates: list[_RigidBodyCandidate],
    guide_points: np.ndarray | None,
) -> _RigidBodyCandidate | None:
    """Choose the best rigid-body candidate for radius estimation."""
    if not candidates:
        return None

    def _score(candidate: _RigidBodyCandidate) -> tuple[float, float, float]:
        ordered = _order_polyline_centers(candidate.collider_centers)
        ordered = _orient_centers_to_guide(ordered, guide_points)
        endpoint_cost = 0.0
        if guide_points is not None and len(guide_points) >= 2:
            endpoint_cost = _endpoint_cost(ordered, guide_points[0], guide_points[-1])
        return (
            float(_candidate_keyword_rank(candidate, centerline=False)),
            float(-_candidate_cross_section(candidate)),
            float(endpoint_cost),
        )

    return min(candidates, key=_score)


def _nearest_segment_index(points: np.ndarray, sample: np.ndarray) -> int:
    """Return the index of the nearest centerline segment to a sample point."""
    if len(points) < 2:
        return 0
    midpoints = 0.5 * (points[1:] + points[:-1])
    return int(np.argmin(np.linalg.norm(midpoints - sample, axis=1)))


def _estimate_radius_from_point_groups(point_groups: list[np.ndarray], points: np.ndarray) -> float | None:
    """Estimate an isotropic rod radius from collider point clouds."""
    if len(point_groups) == 0 or len(points) < 2:
        return None

    radii = []
    for group in point_groups:
        if group.size == 0:
            continue
        centroid = group.mean(axis=0)
        seg_index = _nearest_segment_index(points, centroid)
        p0 = points[seg_index]
        p1 = points[seg_index + 1]
        tangent = p1 - p0
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1.0e-12:
            continue
        tangent /= tangent_norm
        rel = group - p0
        projected = rel @ tangent
        closest = p0 + np.outer(projected, tangent)
        radial = np.linalg.norm(group - closest, axis=1)
        if radial.size == 0:
            continue
        radii.append(float(np.quantile(radial, 0.9)))

    if not radii:
        return None
    return float(np.median(np.asarray(radii, dtype=np.float64)))


def _estimate_radius_from_visual_bbox(candidate: _RigidBodyCandidate) -> float | None:
    """Estimate a rod radius from a rigid body's visible mesh thickness."""
    if candidate.visual_bbox_size is None:
        return None
    dims = [float(value) for value in candidate.visual_bbox_size if float(value) > 1.0e-9]
    if not dims:
        return None
    return 0.5 * min(dims)


def _read_centerline_spec(stage: Usd.Stage, rod_prim: Usd.Prim, segment_count: int | None) -> RodCenterlineSpec:
    """Resolve the rod centerline and isotropic radius from guide + helper meshes."""
    meters_per_unit, up_axis = _stage_units(stage)
    xform_cache = UsdGeom.XformCache()

    guide_points_list = _read_basis_curve_points(
        stage,
        rod_prim,
        meters_per_unit=meters_per_unit,
        up_axis=up_axis,
        xform_cache=xform_cache,
    )
    guide_points = np.asarray(guide_points_list, dtype=np.float64)

    widths_attr = rod_prim.GetAttribute("widths")
    widths = []
    if widths_attr and widths_attr.HasAuthoredValue():
        widths = [float(width) for width in widths_attr.Get() or []]
    width_radius = 0.5 * float(np.median(np.asarray(widths, dtype=np.float64))) if widths else None

    candidates = _collect_rigid_body_candidates(
        stage,
        meters_per_unit=meters_per_unit,
        up_axis=up_axis,
        xform_cache=xform_cache,
    )
    centerline_candidate = _select_centerline_candidate(candidates, guide_points if len(guide_points) >= 2 else None)
    radius_candidate = _select_radius_candidate(candidates, guide_points if len(guide_points) >= 2 else None)

    if len(guide_points) > 2:
        centerline = guide_points
        centerline_source_path = rod_prim.GetPath().pathString
    elif centerline_candidate is not None:
        ordered_centers = _order_polyline_centers(centerline_candidate.collider_centers)
        ordered_centers = _orient_centers_to_guide(ordered_centers, guide_points if len(guide_points) >= 2 else None)
        centerline = _segment_centers_to_nodes(ordered_centers)
        if len(guide_points) >= 2:
            centerline[0] = guide_points[0]
            centerline[-1] = guide_points[-1]
        centerline_source_path = centerline_candidate.prim_path
    elif len(guide_points) >= 2:
        centerline = guide_points
        centerline_source_path = rod_prim.GetPath().pathString
    else:
        raise RuntimeError(f"No usable rod centerline found for {rod_prim.GetPath().pathString}")

    if segment_count is not None and segment_count >= 2:
        centerline = _resample_polyline(centerline, segment_count + 1)

    radius = None
    radius_source_path = None
    if radius_candidate is not None:
        radius = _estimate_radius_from_visual_bbox(radius_candidate)
        if radius is None:
            radius = _estimate_radius_from_point_groups(radius_candidate.collider_point_groups, centerline)
        if radius is not None:
            radius_source_path = radius_candidate.prim_path
    if radius is None and centerline_candidate is not None:
        radius = _estimate_radius_from_visual_bbox(centerline_candidate)
        if radius is None:
            radius = _estimate_radius_from_point_groups(centerline_candidate.collider_point_groups, centerline)
        if radius is not None:
            radius_source_path = centerline_candidate.prim_path
    if radius is None and width_radius is not None:
        radius = width_radius
        radius_source_path = rod_prim.GetPath().pathString
    if radius is None:
        radius = float(DEFAULTS["radius"])

    return RodCenterlineSpec(
        points=[(float(point[0]), float(point[1]), float(point[2])) for point in centerline],
        radius=float(radius),
        guide_prim_path=rod_prim.GetPath().pathString,
        centerline_source_path=centerline_source_path,
        radius_source_path=radius_source_path,
    )


def _average_nonempty(values: list[float]) -> float:
    """Return the arithmetic mean of finite values, or 0.0 when empty."""
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0
    return float(sum(finite) / len(finite))


def read_rod_params(usd_path: str) -> dict:
    """Resolve rod geometry + material parameters into a normalized dict."""
    out = dict(DEFAULTS)
    out["points"] = []
    out["guidePrimPath"] = None
    out["centerlineSourcePath"] = None
    out["radiusSourcePath"] = None

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        return out

    rod_prim = _find_rod_prim(stage)
    if not rod_prim:
        return out

    materials = _bound_rod_material_prims(rod_prim)
    sources = [rod_prim, *materials]

    def _walk(*names, default):
        for prim in sources:
            for name in names:
                attr = prim.GetAttribute(name)
                if attr and attr.HasAuthoredValue():
                    return attr.Get()
        return default

    segment_count_value = _walk("newton:rod:segmentCount", default=None)
    segment_count = None if segment_count_value is None else int(segment_count_value)

    out["segmentCount"] = segment_count
    out["stretchStiffness"] = float(_walk("newton:rod:stretchStiffness", default=DEFAULTS["stretchStiffness"]))
    out["stretchDamping"] = float(_walk("newton:rod:stretchDamping", default=DEFAULTS["stretchDamping"]))
    out["compressStiffness"] = float(_walk("newton:rod:compressStiffness", default=DEFAULTS["compressStiffness"]))
    out["bendYStiffness"] = float(_walk("newton:rod:bendYStiffness", default=DEFAULTS["bendYStiffness"]))
    out["bendYDamping"] = float(_walk("newton:rod:bendYDamping", default=DEFAULTS["bendYDamping"]))
    out["bendZStiffness"] = float(_walk("newton:rod:bendZStiffness", default=DEFAULTS["bendZStiffness"]))
    out["bendZDamping"] = float(_walk("newton:rod:bendZDamping", default=DEFAULTS["bendZDamping"]))
    out["torsionStiffness"] = float(_walk("newton:rod:torsionStiffness", default=DEFAULTS["torsionStiffness"]))
    out["torsionDamping"] = float(_walk("newton:rod:torsionDamping", default=DEFAULTS["torsionDamping"]))

    centerline = _read_centerline_spec(stage, rod_prim, segment_count=segment_count)
    out["points"] = centerline.points
    out["radius"] = centerline.radius
    out["guidePrimPath"] = centerline.guide_prim_path
    out["centerlineSourcePath"] = centerline.centerline_source_path
    out["radiusSourcePath"] = centerline.radius_source_path

    widths_attr = rod_prim.GetAttribute("widths")
    out["widths"] = [float(width) for width in (widths_attr.Get() or [])] if widths_attr and widths_attr.HasAuthoredValue() else []

    out["bendStiffness"] = _average_nonempty(
        [out["bendYStiffness"], out["bendZStiffness"], out["torsionStiffness"]]
    )
    out["bendDamping"] = _average_nonempty(
        [out["bendYDamping"], out["bendZDamping"], out["torsionDamping"]]
    )

    intent_attr = rod_prim.GetAttribute("newton:deformable:simulationIntent")
    out["intent"] = str(intent_attr.Get()) if intent_attr and intent_attr.HasAuthoredValue() else "rod"
    return out
