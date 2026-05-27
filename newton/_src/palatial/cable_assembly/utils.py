# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Small geometry utilities shared by Power cable assembly modules."""

from __future__ import annotations

from math import isfinite

import warp as wp
from pxr import Gf

from .types import ExtractedPrim, Point3


def transform_point(matrix: Gf.Matrix4d, point: object) -> Point3:
    """Transform a point-like value with a USD matrix."""

    transformed = matrix.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
    return (float(transformed[0]), float(transformed[1]), float(transformed[2]))


def triangulate_face_indices(counts: object, indices: object) -> tuple[int, ...]:
    """Triangulate USD polygon face indices using fan triangulation."""

    face_counts = [int(count) for count in counts]
    face_indices = [int(index) for index in indices]
    out: list[int] = []
    cursor = 0
    for count in face_counts:
        if count < 3:
            cursor += count
            continue
        first = face_indices[cursor]
        for offset in range(1, count - 1):
            out.extend((first, face_indices[cursor + offset], face_indices[cursor + offset + 1]))
        cursor += count
    return tuple(out)


def bounds(points: tuple[Point3, ...]) -> tuple[Point3, Point3]:
    """Return axis-aligned bounds for point tuples."""

    return (
        tuple(min(point[index] for point in points) for index in range(3)),
        tuple(max(point[index] for point in points) for index in range(3)),
    )


def combine_bounds(prims: tuple[ExtractedPrim, ...]) -> tuple[Point3, Point3]:
    """Return combined axis-aligned bounds for extracted prims."""

    if not prims:
        raise RuntimeError("Cannot combine bounds for an empty prim set")
    return (
        tuple(min(prim.world_bounds_min[index] for prim in prims) for index in range(3)),
        tuple(max(prim.world_bounds_max[index] for prim in prims) for index in range(3)),
    )


def midpoint(point_a: Point3, point_b: Point3) -> Point3:
    """Return the midpoint between two point tuples."""

    return tuple((point_a[index] + point_b[index]) * 0.5 for index in range(3))


def as_vec3(point: object) -> wp.vec3:
    """Convert a point-like value to ``wp.vec3``."""

    return wp.vec3(float(point[0]), float(point[1]), float(point[2]))


def point_is_finite(point: object) -> bool:
    """Return whether every point component is finite."""

    return all(isfinite(float(point[index])) for index in range(3))


def distance_squared(point_a: object, point_b: object) -> float:
    """Return squared Euclidean distance between two point-like values."""

    dx = float(point_a[0]) - float(point_b[0])
    dy = float(point_a[1]) - float(point_b[1])
    dz = float(point_a[2]) - float(point_b[2])
    return dx * dx + dy * dy + dz * dz
