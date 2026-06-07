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
from ..usd import utils as _usd_material_utils
from .usd_utils import (
    has_api_schema,
    matrix_transform_points,
    read_mesh_world_points,
    stage_units,
    to_newton_world,
)

from dataclasses import dataclass
import math
import os

import numpy as np
from pxr import Usd, UsdGeom, UsdShade


DEFAULTS = {
    # geometry
    "segmentCount": None,
    "radius": 0.01,
    "frameDefinition": "parallelTransport",
    "closed": False,
    "crossSectionType": "roundSolid",
    "width": 0.01,
    "thickness": 0.002,
    "length": 1.0,
    "dropHeight": 0.3,
    "twistTotal": 0.0,
    # material
    "density": 1000.0,
    "stretchStiffness": 1.0e5,
    "stretchDamping": 0.0,
    "compressStiffness": 1.0e5,
    "compressDamping": 0.0,
    "bendStiffness": 0.0,
    "bendDamping": 0.0,
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
    return has_api_schema(prim, "NewtonRodAPI")


def _has_rod_material_api(prim: Usd.Prim) -> bool:
    """True iff the prim carries NewtonRodMaterialAPI."""
    return has_api_schema(prim, "NewtonRodMaterialAPI")


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


def _geometry_source_prims(rod_prim: Usd.Prim) -> list[Usd.Prim]:
    """Return the rod guide plus ancestors that may carry rod attrs."""
    out: list[Usd.Prim] = []
    seen = set()
    prim = rod_prim
    while prim and prim.IsValid():
        path = prim.GetPath()
        if path not in seen:
            seen.add(path)
            out.append(prim)
        prim = prim.GetParent()
    return out


def _authored_attribute_value(sources: list[Usd.Prim], *names: str) -> tuple[object | None, str | None]:
    """Return the first authored attribute value plus the prim path that authored it."""
    for prim in sources:
        for name in names:
            attr = prim.GetAttribute(name)
            if attr and attr.HasAuthoredValue():
                return attr.Get(), prim.GetPath().pathString
    return None, None


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


def _coerce_color_triplet(value) -> tuple[float, float, float] | None:
    """Coerce a USD color value (Vec3 or per-point array) to an RGB tuple."""
    if value is None:
        return None
    try:
        if hasattr(value, "__len__") and len(value) > 0 and hasattr(value[0], "__len__"):
            value = value[0]
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError, IndexError):
        return None


def _sample_texture_mean_rgb(
    stage: Usd.Stage,
    texture_path: str,
) -> tuple[float, float, float] | None:
    """Return the alpha-weighted mean RGB of a diffuse texture, or None on failure.

    Tries the resolved texture path first, then falls back to ``<usda_dir>/<name>``
    and ``<usda_dir>/textures/<name>`` since some exports record absolute paths
    that no longer exist on the consumer's machine.
    """
    from ..utils.texture import load_texture_from_file

    candidates: list[str] = []
    if texture_path:
        candidates.append(texture_path)
    root_layer = stage.GetRootLayer() if stage else None
    base_dir = os.path.dirname(root_layer.realPath) if root_layer and root_layer.realPath else ""
    if base_dir and texture_path:
        name = os.path.basename(texture_path)
        if name:
            candidates.append(os.path.join(base_dir, name))
            candidates.append(os.path.join(base_dir, "textures", name))

    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        image = load_texture_from_file(path)
        if image is None or image.size == 0:
            continue
        rgb = image[..., :3].astype(np.float32) / 255.0
        if image.ndim == 3 and image.shape[-1] == 4:
            alpha = image[..., 3].astype(np.float32) / 255.0
            total = float(alpha.sum())
            if total < 1.0:
                continue
            weighted = (rgb * alpha[..., None]).reshape(-1, 3).sum(axis=0) / total
        else:
            weighted = rgb.reshape(-1, 3).mean(axis=0)
        return (float(weighted[0]), float(weighted[1]), float(weighted[2]))
    return None


def _resolve_display_color(
    stage: Usd.Stage,
    rod_prim: Usd.Prim,
    centerline_source_path: str | None,
) -> tuple[float, float, float] | None:
    """Resolve a single RGB display color for the rod from bound USD materials.

    Preference order:
      1. Flat color on the material bound to the rod guide curve.
      2. Flat color on the material bound to any mesh descendant in the
         centerline source's ancestor chain (the cable-side rigid body).
      3. ``primvars:displayColor`` authored on those mesh descendants.
      4. Alpha-weighted mean RGB of the diffuse texture bound to those meshes.

    Returns None when no color can be resolved (the loader then falls back to
    a neutral grey).
    """
    props = _usd_material_utils.resolve_material_properties_for_prim(rod_prim)
    color = _coerce_color_triplet(props.get("color"))
    if color is not None:
        return color

    # Walk from the centerline source (often the guide curve itself) up the
    # parent chain so we visit the enclosing rigid-body Xform that owns the
    # cable Mesh + bound material.
    visited: set[str] = set()
    search_roots: list[Usd.Prim] = []
    if centerline_source_path:
        cursor = stage.GetPrimAtPath(centerline_source_path)
        while cursor and cursor.IsValid() and not cursor.IsPseudoRoot():
            path = cursor.GetPath().pathString
            if path in visited:
                break
            visited.add(path)
            search_roots.append(cursor)
            cursor = cursor.GetParent()

    texture_paths: list[str] = []
    for root in search_roots:
        for descendant in Usd.PrimRange(root):
            if not descendant.IsA(UsdGeom.Mesh):
                continue
            props = _usd_material_utils.resolve_material_properties_for_prim(descendant)
            color = _coerce_color_triplet(props.get("color"))
            if color is not None:
                return color
            primvar = UsdGeom.PrimvarsAPI(descendant).GetPrimvar("displayColor")
            if primvar:
                color = _coerce_color_triplet(primvar.Get())
                if color is not None:
                    return color
            texture = props.get("texture")
            if texture and texture not in texture_paths:
                texture_paths.append(texture)

    for texture in texture_paths:
        color = _sample_texture_mean_rgb(stage, texture)
        if color is not None:
            return color
    return None


def _resolve_existing_texture_path(stage: Usd.Stage, texture_path: str) -> str | None:
    """Return the first on-disk path matching ``texture_path`` or its basename.

    USD assets sometimes bake absolute texture paths (e.g. ``/opt/...``) that
    do not exist on the consumer machine; in that case we fall back to
    ``<usda_dir>/<name>`` and ``<usda_dir>/textures/<name>``.
    """
    if not texture_path:
        return None
    candidates: list[str] = [texture_path]
    root_layer = stage.GetRootLayer() if stage else None
    base_dir = os.path.dirname(root_layer.realPath) if root_layer and root_layer.realPath else ""
    if base_dir:
        name = os.path.basename(texture_path)
        if name:
            candidates.append(os.path.join(base_dir, name))
            candidates.append(os.path.join(base_dir, "textures", name))
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _resolve_diffuse_texture_path(
    stage: Usd.Stage,
    rod_prim: Usd.Prim,
    centerline_source_path: str | None,
) -> str | None:
    """Resolve an on-disk diffuse texture path bound to the rod's cable mesh."""
    props = _usd_material_utils.resolve_material_properties_for_prim(rod_prim)
    texture = props.get("texture")
    if texture:
        resolved = _resolve_existing_texture_path(stage, texture)
        if resolved is not None:
            return resolved

    visited: set[str] = set()
    search_roots: list[Usd.Prim] = []
    if centerline_source_path:
        cursor = stage.GetPrimAtPath(centerline_source_path)
        while cursor and cursor.IsValid() and not cursor.IsPseudoRoot():
            path = cursor.GetPath().pathString
            if path in visited:
                break
            visited.add(path)
            search_roots.append(cursor)
            cursor = cursor.GetParent()

    for root in search_roots:
        for descendant in Usd.PrimRange(root):
            if not descendant.IsA(UsdGeom.Mesh):
                continue
            props = _usd_material_utils.resolve_material_properties_for_prim(descendant)
            texture = props.get("texture")
            if texture:
                resolved = _resolve_existing_texture_path(stage, texture)
                if resolved is not None:
                    return resolved
    return None


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
    world = matrix_transform_points(matrix, local)
    world = to_newton_world(world, meters_per_unit, up_axis)
    return [(float(point[0]), float(point[1]), float(point[2])) for point in world]


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
                points = read_mesh_world_points(
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
                points = read_mesh_world_points(
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


def _synthesize_straight_centerline(
    *,
    segment_count: int,
    length: float,
    drop_height: float,
) -> np.ndarray:
    """Create a straight fallback centerline along +X at the requested height."""
    if segment_count < 1:
        raise RuntimeError(f"Straight rod fallback requires segmentCount >= 1 (got {segment_count})")
    if length <= 0.0:
        raise RuntimeError(f"Straight rod fallback requires positive length (got {length})")

    x = np.linspace(0.0, float(length), int(segment_count) + 1, dtype=np.float64)
    y = np.zeros_like(x)
    z = np.full_like(x, float(drop_height))
    return np.stack((x, y, z), axis=1)


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


def _read_centerline_spec(
    stage: Usd.Stage,
    rod_prim: Usd.Prim,
    *,
    segment_count: int | None,
    fallback_length: float,
    fallback_drop_height: float,
) -> RodCenterlineSpec:
    """Resolve the rod centerline and isotropic radius from guide + helper meshes."""
    meters_per_unit, up_axis = stage_units(stage)
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
        fallback_segments = int(segment_count) if segment_count is not None else 10
        centerline = _synthesize_straight_centerline(
            segment_count=fallback_segments,
            length=float(fallback_length),
            drop_height=float(fallback_drop_height),
        )
        centerline_source_path = rod_prim.GetPath().pathString

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


def _canonical_or_compat_value(
    values: dict[str, float],
    authored: dict[str, bool],
    canonical_key: str,
    compat_key: str,
) -> float:
    """Use canonical isotropic value, or one authored compatibility value."""
    if authored.get(canonical_key, False):
        return float(values[canonical_key])
    if authored.get(compat_key, False):
        return float(values[compat_key])
    return float(values[canonical_key])


def read_rod_params(usd_path: str) -> dict:
    """Resolve rod geometry + material parameters into a normalized dict."""
    out = dict(DEFAULTS)
    out["points"] = []
    out["guidePrimPath"] = None
    out["centerlineSourcePath"] = None
    out["radiusSourcePath"] = None
    out["effectiveDensity"] = float(DEFAULTS["density"])
    out["axialStiffness"] = float(DEFAULTS["stretchStiffness"])
    out["axialDamping"] = float(DEFAULTS["stretchDamping"])

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        return out

    rod_prim = _find_rod_prim(stage)
    if not rod_prim:
        return out

    geometry_sources = _geometry_source_prims(rod_prim)
    materials = _bound_rod_material_prims(rod_prim)
    material_sources: list[Usd.Prim] = []
    seen_material_sources = set()
    for prim in [*materials, *geometry_sources]:
        path = prim.GetPath()
        if path in seen_material_sources:
            continue
        seen_material_sources.add(path)
        material_sources.append(prim)

    authored: dict[str, bool] = {}

    def _walk_geometry(key: str, *names: str, default):
        value, _path = _authored_attribute_value(geometry_sources, *names)
        authored[key] = value is not None
        return default if value is None else value

    def _walk_material(key: str, *names: str, default):
        value, _path = _authored_attribute_value(material_sources, *names)
        authored[key] = value is not None
        return default if value is None else value

    out["frameDefinition"] = str(
        _walk_geometry("frameDefinition", "newton:rod:frameDefinition", default=DEFAULTS["frameDefinition"])
    )
    out["closed"] = bool(
        _walk_geometry("closed", "newton:rod:closed", "newton:rod:isClosed", default=DEFAULTS["closed"])
    )
    out["crossSectionType"] = str(
        _walk_geometry("crossSectionType", "newton:rod:crossSectionType", default=DEFAULTS["crossSectionType"])
    )
    out["width"] = float(_walk_geometry("width", "newton:rod:width", default=DEFAULTS["width"]))
    out["thickness"] = float(_walk_geometry("thickness", "newton:rod:thickness", default=DEFAULTS["thickness"]))
    out["length"] = float(_walk_geometry("length", "newton:rod:length", default=DEFAULTS["length"]))
    out["dropHeight"] = float(_walk_geometry("dropHeight", "newton:rod:dropHeight", default=DEFAULTS["dropHeight"]))
    out["twistTotal"] = float(_walk_geometry("twistTotal", "newton:rod:twistTotal", default=DEFAULTS["twistTotal"]))

    segment_count_value = _walk_geometry("segmentCount", "newton:rod:segmentCount", default=None)
    segment_count = None if segment_count_value is None else int(segment_count_value)

    out["segmentCount"] = segment_count
    out["density"] = float(_walk_material("density", "newton:rod:density", default=DEFAULTS["density"]))
    out["stretchStiffness"] = float(
        _walk_material("stretchStiffness", "newton:rod:stretchStiffness", default=DEFAULTS["stretchStiffness"])
    )
    out["stretchDamping"] = float(
        _walk_material("stretchDamping", "newton:rod:stretchDamping", default=DEFAULTS["stretchDamping"])
    )
    out["compressStiffness"] = float(
        _walk_material("compressStiffness", "newton:rod:compressStiffness", default=DEFAULTS["compressStiffness"])
    )
    out["compressDamping"] = float(
        _walk_material("compressDamping", "newton:rod:compressDamping", default=DEFAULTS["compressDamping"])
    )
    out["bendStiffness"] = float(
        _walk_material("bendStiffness", "newton:rod:bendStiffness", default=DEFAULTS["bendStiffness"])
    )
    out["bendDamping"] = float(
        _walk_material("bendDamping", "newton:rod:bendDamping", default=DEFAULTS["bendDamping"])
    )

    centerline = _read_centerline_spec(
        stage,
        rod_prim,
        segment_count=segment_count,
        fallback_length=float(out["length"]),
        fallback_drop_height=float(out["dropHeight"]),
    )
    out["points"] = centerline.points
    out["guidePrimPath"] = centerline.guide_prim_path
    out["centerlineSourcePath"] = centerline.centerline_source_path

    widths_attr = rod_prim.GetAttribute("widths")
    out["widths"] = [float(width) for width in (widths_attr.Get() or [])] if widths_attr and widths_attr.HasAuthoredValue() else []

    explicit_radius_value, explicit_radius_path = _authored_attribute_value(geometry_sources, "newton:rod:radius")
    if explicit_radius_value is not None:
        out["radius"] = float(explicit_radius_value)
        out["radiusSourcePath"] = explicit_radius_path
    elif str(out["crossSectionType"]) == "flatRect" and float(out["thickness"]) > 0.0:
        out["radius"] = 0.5 * float(out["thickness"])
        out["radiusSourcePath"] = rod_prim.GetPath().pathString
    else:
        out["radius"] = centerline.radius
        out["radiusSourcePath"] = centerline.radius_source_path

    material_values = {
        "stretchStiffness": float(out["stretchStiffness"]),
        "stretchDamping": float(out["stretchDamping"]),
        "compressStiffness": float(out["compressStiffness"]),
        "compressDamping": float(out["compressDamping"]),
        "bendStiffness": float(out["bendStiffness"]),
        "bendDamping": float(out["bendDamping"]),
    }
    out["axialStiffness"] = _canonical_or_compat_value(
        material_values,
        authored,
        "stretchStiffness",
        "compressStiffness",
    )
    out["axialDamping"] = _canonical_or_compat_value(
        material_values,
        authored,
        "stretchDamping",
        "compressDamping",
    )

    out["effectiveDensity"] = float(out["density"])
    if (
        str(out["crossSectionType"]) == "flatRect"
        and float(out["width"]) > 0.0
        and float(out["thickness"]) > 0.0
        and float(out["radius"]) > 0.0
    ):
        rect_area = float(out["width"]) * float(out["thickness"])
        circular_area = math.pi * float(out["radius"]) * float(out["radius"])
        if circular_area > 1.0e-12:
            out["effectiveDensity"] = float(out["density"]) * rect_area / circular_area

    intent_attr = rod_prim.GetAttribute("newton:deformable:simulationIntent")
    out["intent"] = str(intent_attr.Get()) if intent_attr and intent_attr.HasAuthoredValue() else "rod"

    out["displayColor"] = _resolve_display_color(stage, rod_prim, out.get("centerlineSourcePath"))
    out["diffuseTexturePath"] = _resolve_diffuse_texture_path(
        stage, rod_prim, out.get("centerlineSourcePath")
    )
    return out
