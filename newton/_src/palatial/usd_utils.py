# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared USD geometry helpers for palatial loaders."""

from __future__ import annotations

import newton  # noqa: F401

import numpy as np
from pxr import Gf, Usd, UsdGeom


def has_api_schema(prim: Usd.Prim, schema_name: str) -> bool:
    """Return True when ``prim`` carries the requested applied schema."""
    applied = set(prim.GetAppliedSchemas())
    raw = prim.GetMetadata("apiSchemas")
    if raw is not None:
        for items_attr in ("prependedItems", "appendedItems", "explicitItems"):
            for tok in getattr(raw, items_attr, []) or []:
                applied.add(str(tok))
    return schema_name in applied


def stage_units(stage: Usd.Stage) -> tuple[float, str]:
    """Return ``(meters_per_unit, up_axis)`` for a stage."""
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    up_axis = str(UsdGeom.GetStageUpAxis(stage) or "Z")
    return meters_per_unit, up_axis


def to_newton_world(points: np.ndarray, meters_per_unit: float, up_axis: str) -> np.ndarray:
    """Convert USD world-space points into Newton world coordinates."""
    if points.size == 0:
        return points
    out = np.asarray(points, dtype=np.float64) * float(meters_per_unit)
    if up_axis == "Y":
        out = np.stack([out[:, 0], -out[:, 2], out[:, 1]], axis=1)
    return out


def matrix_transform_points(matrix: Gf.Matrix4d, points: np.ndarray) -> np.ndarray:
    """Apply a USD transform matrix to a point cloud."""
    if points.size == 0:
        return points
    world = np.empty_like(points, dtype=np.float64)
    for i, point in enumerate(points):
        value = matrix.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
        world[i] = (float(value[0]), float(value[1]), float(value[2]))
    return world


def read_mesh_world_points(
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
    world = matrix_transform_points(matrix, local)
    return to_newton_world(world, meters_per_unit, up_axis)
