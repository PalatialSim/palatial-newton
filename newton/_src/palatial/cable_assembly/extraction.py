# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""USD extraction for v1 Power cable assembly assets."""

from __future__ import annotations

from math import isfinite
from pathlib import Path

# Importing newton registers the bundled USD schema plugins before pxr.Usd use.
import newton as _newton  # noqa: F401
import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

from .constants import (
    GEOMETRY_SCOPE_PATH,
    PCA_CABLE_EPS,
    POWER_IEC_REQUIRED_PRIM_NAMES,
    POWER_NEMA_REQUIRED_PRIM_NAMES,
    POWER_REQUIRED_PRIM_NAMES,
)
from .types import (
    CableExtraction,
    ConnectorEndpointExtraction,
    EndpointId,
    ExtractedPrim,
    Point3,
    PowerCableAssemblyExtraction,
)
from .utils import (
    as_vec3,
    bounds,
    combine_bounds,
    midpoint,
    point_is_finite,
    transform_point,
    triangulate_face_indices,
)


def extract_power_cable_assembly(usd_path: str) -> PowerCableAssemblyExtraction:
    """Extract the v1 Power cable assembly from a USD stage."""

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise RuntimeError(f"Cannot open USD: {usd_path}")
    _validate_stage_contract(stage, usd_path)

    geometry = stage.GetPrimAtPath(GEOMETRY_SCOPE_PATH)
    if not geometry or not geometry.IsValid():
        raise RuntimeError(f"Power cable assembly requires {GEOMETRY_SCOPE_PATH}")

    prim_by_name = {
        child.GetName(): _build_extracted_prim(child)
        for child in geometry.GetChildren()
        if child.GetName() in POWER_REQUIRED_PRIM_NAMES
    }
    missing = tuple(name for name in POWER_REQUIRED_PRIM_NAMES if name not in prim_by_name)
    if missing:
        raise RuntimeError(f"Power cable assembly missing required prims: {', '.join(missing)}")

    cable = _extract_cable(prim_by_name["Power_Cable_Body"])
    endpoints = (
        _build_endpoint("iec", POWER_IEC_REQUIRED_PRIM_NAMES, prim_by_name),
        _build_endpoint("nema", POWER_NEMA_REQUIRED_PRIM_NAMES, prim_by_name),
    )
    return PowerCableAssemblyExtraction(
        source_path=Path(usd_path).resolve(),
        cable=cable,
        endpoints=endpoints,
    )


def _validate_stage_contract(stage: Usd.Stage, usd_path: str) -> None:
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    if abs(meters_per_unit - 1.0) > 1.0e-9:
        raise RuntimeError(f"Power cable assembly {usd_path} must use metersPerUnit=1.0")

    up_axis = str(UsdGeom.GetStageUpAxis(stage) or "Y")
    if up_axis != "Y":
        raise RuntimeError(f"Power cable assembly {usd_path} must use upAxis='Y'")


def _build_extracted_prim(prim: Usd.Prim) -> ExtractedPrim:
    mesh_prim = _single_mesh_child(prim)
    mesh = UsdGeom.Mesh(mesh_prim)
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if points is None or counts is None or indices is None:
        raise RuntimeError(f"Assembly mesh {mesh_prim.GetPath()} is missing points or face data")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_transform = xform_cache.GetLocalToWorldTransform(mesh_prim)
    world_points = tuple(transform_point(world_transform, point) for point in points)
    if len(world_points) < 3:
        raise RuntimeError(f"Assembly mesh {mesh_prim.GetPath()} must contain at least three points")

    bounds_min, bounds_max = bounds(world_points)
    triangle_indices = triangulate_face_indices(counts, indices)
    if len(triangle_indices) < 3:
        raise RuntimeError(f"Assembly mesh {mesh_prim.GetPath()} must contain at least one triangle")

    return ExtractedPrim(
        prim_name=prim.GetName(),
        prim_path=prim.GetPath().pathString,
        mesh_path=mesh_prim.GetPath().pathString,
        world_bounds_min=bounds_min,
        world_bounds_max=bounds_max,
        world_centroid=midpoint(bounds_min, bounds_max),
        world_points=world_points,
        triangle_vertex_indices=triangle_indices,
    )


def _single_mesh_child(prim: Usd.Prim) -> Usd.Prim:
    meshes = [child for child in prim.GetChildren() if child.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise RuntimeError(f"Assembly prim {prim.GetPath()} must contain exactly one Mesh child")
    return meshes[0]


def _extract_cable(prim: ExtractedPrim) -> CableExtraction:
    try:
        centerline_points, radius, length = _extract_cable_from_world_points_pca(prim)
    except ValueError:
        centerline_points, radius, length = _extract_cable_from_world_bounds(prim)

    if not all(point_is_finite(point) for point in centerline_points):
        raise RuntimeError("Power cable centerline points must be finite")
    if not isfinite(radius) or radius <= 0.0:
        raise RuntimeError(f"Power cable radius must be positive and finite, got {radius}")
    if not isfinite(length) or length <= 0.0:
        raise RuntimeError(f"Power cable length must be positive and finite, got {length}")

    return CableExtraction(
        centerline_points=(as_vec3(centerline_points[0]), as_vec3(centerline_points[1])),
        radius=radius,
        length=length,
    )


def _extract_cable_from_world_points_pca(
    prim: ExtractedPrim,
) -> tuple[tuple[Point3, Point3], float, float]:
    points = np.asarray(prim.world_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Power cable PCA requires finite Nx3 points")

    mean = points.mean(axis=0)
    centered = points - mean
    covariance = centered.T @ centered / float(len(points))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.isfinite(eigenvalues).all() or not np.isfinite(eigenvectors).all():
        raise ValueError("Power cable PCA eigensolution must be finite")

    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    dominant_axis_index = int(np.argmax(np.abs(axis)))
    if axis[dominant_axis_index] < 0.0:
        axis = -axis
    axis_norm = float(np.linalg.norm(axis))
    if not isfinite(axis_norm) or axis_norm <= PCA_CABLE_EPS:
        raise ValueError("Power cable PCA axis must have positive length")
    axis = axis / axis_norm

    projections = centered @ axis
    t_min = float(np.min(projections))
    t_max = float(np.max(projections))
    length = t_max - t_min
    if not isfinite(length) or length <= PCA_CABLE_EPS:
        raise ValueError("Power cable PCA length must be positive")

    radial_vectors = centered - np.outer(projections, axis)
    radial_distances = np.linalg.norm(radial_vectors, axis=1)
    radius = float(np.max(radial_distances))
    if not isfinite(radius) or radius <= PCA_CABLE_EPS:
        raise ValueError("Power cable PCA radius must be positive")

    start = mean + t_min * axis
    end = mean + t_max * axis
    return (
        (float(start[0]), float(start[1]), float(start[2])),
        (float(end[0]), float(end[1]), float(end[2])),
    ), radius, length


def _extract_cable_from_world_bounds(prim: ExtractedPrim) -> tuple[tuple[Point3, Point3], float, float]:
    extents = tuple(
        prim.world_bounds_max[index] - prim.world_bounds_min[index]
        for index in range(3)
    )
    major_axis_index = int(np.argmax(np.asarray(extents, dtype=np.float64)))
    center = midpoint(prim.world_bounds_min, prim.world_bounds_max)
    start = list(center)
    end = list(center)
    start[major_axis_index] = prim.world_bounds_min[major_axis_index]
    end[major_axis_index] = prim.world_bounds_max[major_axis_index]
    minor_axis_indices = tuple(index for index in range(3) if index != major_axis_index)
    radius = max(extents[index] * 0.5 for index in minor_axis_indices)
    length = extents[major_axis_index]
    return (tuple(start), tuple(end)), float(radius), float(length)


def _build_endpoint(
    endpoint_id: EndpointId,
    prim_names: tuple[str, ...],
    prim_by_name: dict[str, ExtractedPrim],
) -> ConnectorEndpointExtraction:
    prims = tuple(prim_by_name[name] for name in prim_names)
    bounds_min, bounds_max = combine_bounds(prims)
    return ConnectorEndpointExtraction(
        endpoint_id=endpoint_id,
        prim_names=prim_names,
        prims=prims,
        world_bounds_min=bounds_min,
        world_bounds_max=bounds_max,
        anchor_point=as_vec3(midpoint(bounds_min, bounds_max)),
    )
