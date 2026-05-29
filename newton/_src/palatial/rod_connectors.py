# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid connector planning/helpers for palatial rod assets."""

from __future__ import annotations

# `import newton` registers the bundled USD plugins via
# newton/_src/usd/__init__.py. Must precede any pxr.Usd usage.
import newton

from dataclasses import dataclass
import math
import re
from typing import Any

import numpy as np
from pxr import Usd, UsdGeom
import warp as wp

from .usd_utils import (
    has_api_schema,
    matrix_transform_points,
    read_mesh_world_points,
    stage_units,
    to_newton_world,
)

_TransformParts = tuple[wp.vec3, wp.quat]


@dataclass
class _UsdJointLink:
    """Joint prim metadata used to import rigid subsets around a rod."""

    prim_path: str
    body0_path: str
    body1_path: str


@dataclass
class _RodAttachmentComponent:
    """A rigid connector assembly that should attach to one rod endpoint."""

    root_body_path: str
    body_paths: set[str]
    joint_paths: set[str]
    endpoint_name: str
    relative_xform: _TransformParts
    parent_xform: _TransformParts
    child_xform: _TransformParts
    proxy_bounds: tuple[wp.vec3, wp.vec3] | None


def _normalize_vector(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Return a normalized vector, or the fallback when degenerate."""
    norm = float(np.linalg.norm(values))
    if norm <= 1.0e-12:
        return np.asarray(fallback, dtype=np.float64)
    return np.asarray(values, dtype=np.float64) / norm


def _quat_conjugate_np(q: np.ndarray) -> np.ndarray:
    """Return the conjugate of an ``(x, y, z, w)`` quaternion."""
    return np.asarray((-q[0], -q[1], -q[2], q[3]), dtype=np.float64)


def _quat_mul_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two ``(x, y, z, w)`` quaternions."""
    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    return np.asarray(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dtype=np.float64,
    )


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector by an ``(x, y, z, w)`` quaternion."""
    q = np.asarray(q, dtype=np.float64)
    v_quat = np.asarray((float(v[0]), float(v[1]), float(v[2]), 0.0), dtype=np.float64)
    return _quat_mul_np(_quat_mul_np(q, v_quat), _quat_conjugate_np(q))[:3]


def _quat_between_vectors_np(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
    """Return a robust quaternion that rotates one direction vector onto another."""
    eps = 1.0e-8
    a = _normalize_vector(from_vec, np.asarray((0.0, 0.0, 1.0), dtype=np.float64))
    b = _normalize_vector(to_vec, a)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))

    if dot >= 1.0 - eps:
        return np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)

    if dot <= -1.0 + eps:
        helper = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        if abs(float(a[0])) >= 0.9:
            helper = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        axis = _normalize_vector(np.cross(a, helper), np.asarray((0.0, 0.0, 1.0), dtype=np.float64))
        return np.asarray((axis[0], axis[1], axis[2], 0.0), dtype=np.float64)

    axis = np.cross(a, b)
    s = math.sqrt(2.0 * (1.0 + dot))
    inv_s = 1.0 / s
    return _normalize_vector(
        np.asarray((axis[0] * inv_s, axis[1] * inv_s, axis[2] * inv_s, 0.5 * s), dtype=np.float64),
        np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64),
    )


def _component_outward_axis(
    root_points: np.ndarray,
    *,
    anchor_world: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    """Estimate the connector's outward axis from the mouth toward the far end."""
    if root_points.size == 0:
        return _normalize_vector(fallback, np.asarray((1.0, 0.0, 0.0), dtype=np.float64))

    deltas = root_points - anchor_world[None, :]
    distances = np.linalg.norm(deltas, axis=1)
    if len(distances) == 0:
        return _normalize_vector(fallback, np.asarray((1.0, 0.0, 0.0), dtype=np.float64))
    tip = root_points[int(np.argmax(distances))]
    return _normalize_vector(tip - anchor_world, fallback)


def _collect_rigid_body_paths(stage: Usd.Stage) -> set[str]:
    """Return all rigid-body prim paths on a stage."""
    out: set[str] = set()
    for prim in stage.Traverse():
        if has_api_schema(prim, "PhysicsRigidBodyAPI"):
            out.add(prim.GetPath().pathString)
    return out


def _collect_joint_links(stage: Usd.Stage, rigid_body_paths: set[str]) -> list[_UsdJointLink]:
    """Return joint links between rigid-body prims on a stage."""
    out: list[_UsdJointLink] = []
    for prim in stage.Traverse():
        rel0 = prim.GetRelationship("physics:body0")
        rel1 = prim.GetRelationship("physics:body1")
        if rel0 is None and rel1 is None:
            continue
        targets0 = rel0.GetTargets() if rel0 else []
        targets1 = rel1.GetTargets() if rel1 else []
        if not targets0 or not targets1:
            continue
        body0_path = str(targets0[0])
        body1_path = str(targets1[0])
        if body0_path not in rigid_body_paths or body1_path not in rigid_body_paths:
            continue
        out.append(
            _UsdJointLink(
                prim_path=prim.GetPath().pathString,
                body0_path=body0_path,
                body1_path=body1_path,
            )
        )
    return out


def _rigid_transform_to_newton(
    matrix: Any,
    *,
    meters_per_unit: float,
    up_axis: str,
) -> wp.transform:
    """Convert a USD rigid transform matrix into Newton world coordinates."""
    basis = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ],
        dtype=np.float64,
    )
    transformed = matrix_transform_points(matrix, basis)
    transformed = to_newton_world(transformed, meters_per_unit, up_axis)

    origin = transformed[0]
    x_axis = _normalize_vector(transformed[1] - origin, np.asarray((1.0, 0.0, 0.0), dtype=np.float64))
    y_axis = transformed[2] - origin
    y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
    y_axis = _normalize_vector(y_axis, np.asarray((0.0, 1.0, 0.0), dtype=np.float64))
    z_axis = np.cross(x_axis, y_axis)
    z_axis = _normalize_vector(z_axis, np.asarray((0.0, 0.0, 1.0), dtype=np.float64))
    y_axis = _normalize_vector(np.cross(z_axis, x_axis), np.asarray((0.0, 1.0, 0.0), dtype=np.float64))

    rot_np = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32, copy=False)
    rot = wp.quat_from_matrix(wp.mat33(*rot_np.flatten().tolist()))
    return wp.transform(wp.vec3(float(origin[0]), float(origin[1]), float(origin[2])), rot)


def _read_rigid_body_points(
    stage: Usd.Stage,
    prim: Usd.Prim,
    *,
    meters_per_unit: float,
    up_axis: str,
    xform_cache: UsdGeom.XformCache,
    prefer_colliders: bool,
) -> np.ndarray:
    """Read representative rigid-body points in Newton world coordinates."""
    point_sets: list[np.ndarray] = []

    if prefer_colliders:
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
                if points.size > 0:
                    point_sets.append(points)
        if point_sets:
            return np.concatenate(point_sets, axis=0)

    for child in prim.GetChildren():
        if not child.IsA(UsdGeom.Mesh):
            continue
        points = read_mesh_world_points(
            stage,
            child,
            meters_per_unit=meters_per_unit,
            up_axis=up_axis,
            xform_cache=xform_cache,
        )
        if points.size > 0:
            point_sets.append(points)

    if point_sets:
        return np.concatenate(point_sets, axis=0)
    return np.empty((0, 3), dtype=np.float64)


def _component_endpoint_name(points: np.ndarray, component_points: np.ndarray) -> str:
    """Choose the closer rod endpoint for a rigid component."""
    if component_points.size == 0 or len(points) < 2:
        return "start"
    centroid = component_points.mean(axis=0)
    start_dist = float(np.linalg.norm(centroid - points[0]))
    end_dist = float(np.linalg.norm(centroid - points[-1]))
    return "start" if start_dist <= end_dist else "end"


def _component_proxy_bounds(component_points: np.ndarray, root_world: wp.transform) -> tuple[wp.vec3, wp.vec3] | None:
    """Return a connector contact proxy box in root-body local space."""
    if component_points.size == 0:
        return None

    root_to_local = wp.transform_inverse(root_world)
    local_points = []
    for point in component_points:
        p_local = wp.transform_point(root_to_local, wp.vec3(float(point[0]), float(point[1]), float(point[2])))
        local_points.append((float(p_local[0]), float(p_local[1]), float(p_local[2])))

    local_np = np.asarray(local_points, dtype=np.float64)
    local_np = local_np[np.isfinite(local_np).all(axis=1)]
    if local_np.size == 0:
        return None

    bounds_min = local_np.min(axis=0)
    bounds_max = local_np.max(axis=0)
    center = 0.5 * (bounds_min + bounds_max)
    half_extents = np.maximum(0.5 * (bounds_max - bounds_min), 1.0e-4)
    return (
        wp.vec3(float(center[0]), float(center[1]), float(center[2])),
        wp.vec3(float(half_extents[0]), float(half_extents[1]), float(half_extents[2])),
    )


def _component_mouth_anchor(
    root_points: np.ndarray,
    *,
    tangent: np.ndarray,
    endpoint_name: str,
    rod_radius: float,
    fallback: np.ndarray,
) -> np.ndarray:
    """Return the centered connector mouth point nearest the rod endpoint."""
    if root_points.size == 0:
        return np.asarray(fallback, dtype=np.float64)

    valid_points = np.asarray(root_points, dtype=np.float64)
    valid_points = valid_points[np.isfinite(valid_points).all(axis=1)]
    if valid_points.size == 0:
        return np.asarray(fallback, dtype=np.float64)

    projections = valid_points @ tangent
    extreme_projection = float(np.max(projections) if endpoint_name == "start" else np.min(projections))
    span = float(np.max(projections) - np.min(projections))
    base_band = max(float(rod_radius), 1.0e-4)
    target_count = min(4, len(valid_points))
    mouth_mask = np.zeros(len(valid_points), dtype=bool)

    for multiplier in (1.0, 2.0, 4.0):
        band = min(base_band * multiplier, span) if span > 0.0 else base_band * multiplier
        if endpoint_name == "start":
            mouth_mask = projections >= extreme_projection - band
        else:
            mouth_mask = projections <= extreme_projection + band
        if int(mouth_mask.sum()) >= target_count:
            break

    if not mouth_mask.any():
        mouth_mask[int(np.argmax(projections) if endpoint_name == "start" else np.argmin(projections))] = True

    mouth_points = valid_points[mouth_mask]
    mouth_projections = projections[mouth_mask]
    transverse_points = mouth_points - mouth_projections[:, None] * tangent[None, :]
    transverse_center = transverse_points.mean(axis=0)
    return transverse_center + extreme_projection * tangent


def _component_relative_xform(
    *,
    stage: Usd.Stage,
    root_body_path: str,
    endpoint_name: str,
    points: np.ndarray,
    quaternions: list[wp.quat],
    rod_radius: float,
    meters_per_unit: float,
    up_axis: str,
    xform_cache: UsdGeom.XformCache,
) -> tuple[_TransformParts, _TransformParts, _TransformParts]:
    """Return the root-body transform that aligns a connector mouth/axis to a rod end."""
    root_prim = stage.GetPrimAtPath(root_body_path)
    root_world = _rigid_transform_to_newton(
        xform_cache.GetLocalToWorldTransform(root_prim),
        meters_per_unit=meters_per_unit,
        up_axis=up_axis,
    )

    root_points = _read_rigid_body_points(
        stage,
        root_prim,
        meters_per_unit=meters_per_unit,
        up_axis=up_axis,
        xform_cache=xform_cache,
        prefer_colliders=True,
    )
    if len(points) < 2:
        tangent = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    elif endpoint_name == "start":
        tangent = _normalize_vector(points[1] - points[0], np.asarray((1.0, 0.0, 0.0), dtype=np.float64))
    else:
        tangent = _normalize_vector(points[-1] - points[-2], np.asarray((1.0, 0.0, 0.0), dtype=np.float64))

    anchor_world = _component_mouth_anchor(
        root_points,
        tangent=tangent,
        endpoint_name=endpoint_name,
        rod_radius=rod_radius,
        fallback=np.asarray(root_world.p, dtype=np.float64),
    )
    desired_axis_world = -tangent if endpoint_name == "start" else tangent
    current_axis_world = _component_outward_axis(
        root_points,
        anchor_world=anchor_world,
        fallback=desired_axis_world,
    )
    align_delta = _quat_between_vectors_np(current_axis_world, desired_axis_world)
    root_quat_np = np.asarray(
        (float(root_world.q[0]), float(root_world.q[1]), float(root_world.q[2]), float(root_world.q[3])),
        dtype=np.float64,
    )
    desired_root_quat_np = _quat_mul_np(align_delta, root_quat_np)

    root_to_local = wp.transform_inverse(root_world)
    anchor_local_wp = wp.transform_point(
        root_to_local,
        wp.vec3(float(anchor_world[0]), float(anchor_world[1]), float(anchor_world[2])),
    )
    anchor_local = np.asarray(
        (float(anchor_local_wp[0]), float(anchor_local_wp[1]), float(anchor_local_wp[2])),
        dtype=np.float64,
    )

    endpoint_world = points[0] if endpoint_name == "start" else points[-1]
    root_translation = np.asarray(endpoint_world, dtype=np.float64) - _quat_rotate_np(
        desired_root_quat_np,
        anchor_local,
    )
    desired_root_world = wp.transform(
        wp.vec3(float(root_translation[0]), float(root_translation[1]), float(root_translation[2])),
        wp.quat(
            float(desired_root_quat_np[0]),
            float(desired_root_quat_np[1]),
            float(desired_root_quat_np[2]),
            float(desired_root_quat_np[3]),
        ),
    )

    if endpoint_name == "start":
        parent_world = wp.transform(wp.vec3(*points[0].tolist()), quaternions[0])
        parent_anchor = wp.transform_identity()
    else:
        parent_world = wp.transform(wp.vec3(*points[-2].tolist()), quaternions[-1])
        segment_length = float(np.linalg.norm(points[-1] - points[-2]))
        parent_anchor = wp.transform(wp.vec3(0.0, 0.0, segment_length), wp.quat_identity())
    endpoint_frame_world = parent_world * parent_anchor
    child_anchor = wp.transform_inverse(desired_root_world) * endpoint_frame_world
    relative = wp.transform_inverse(parent_world) * desired_root_world
    return (relative.p, relative.q), (parent_anchor.p, parent_anchor.q), (child_anchor.p, child_anchor.q)


def _collect_component_paths(
    root_path: str,
    *,
    adjacency: dict[str, set[str]],
    joint_lookup: dict[frozenset[str], set[str]],
    blocked_paths: set[str],
) -> tuple[set[str], set[str]]:
    """Return rigid-body and joint paths for a connector component."""
    stack = [root_path]
    body_paths: set[str] = set()
    joint_paths: set[str] = set()

    while stack:
        current = stack.pop()
        if current in body_paths or current in blocked_paths:
            continue
        body_paths.add(current)
        for neighbor in adjacency.get(current, ()):
            if neighbor in blocked_paths:
                continue
            joint_paths.update(joint_lookup.get(frozenset((current, neighbor)), set()))
            if neighbor not in body_paths:
                stack.append(neighbor)

    return body_paths, joint_paths


def _subset_ignore_paths(
    *,
    all_body_paths: set[str],
    all_joint_paths: set[str],
    keep_body_paths: set[str],
    keep_joint_paths: set[str],
    extra_ignored_paths: list[str],
) -> list[str]:
    """Build ignore regexes that keep only a selected rigid subset."""
    ignore_paths = [re.escape(path) for path in extra_ignored_paths if path]
    for path in sorted(all_body_paths):
        if path not in keep_body_paths:
            ignore_paths.append(re.escape(path))
    for path in sorted(all_joint_paths):
        if path not in keep_joint_paths:
            ignore_paths.append(re.escape(path))
    return ignore_paths


def _reparent_existing_joint(
    builder: Any,
    *,
    joint_idx: int,
    parent_body: int,
    parent_xform: wp.transform,
    child_xform: wp.transform,
    label: str,
) -> None:
    """Rewrite an imported base joint so it attaches to a rod body instead of world."""
    child_body = int(builder.joint_child[joint_idx])
    old_parent = int(builder.joint_parent[joint_idx])

    builder.joint_parent[joint_idx] = parent_body
    builder.joint_X_p[joint_idx] = parent_xform
    builder.joint_X_c[joint_idx] = child_xform
    builder.joint_label[joint_idx] = label
    builder.joint_collision_filter_parent[joint_idx] = True

    child_parents = builder.joint_parents.get(child_body, [])
    builder.joint_parents[child_body] = [
        (parent_body if idx == joint_idx else parent, idx) for parent, idx in child_parents
    ]

    if old_parent in builder.joint_children:
        builder.joint_children[old_parent] = [
            (child, idx) for child, idx in builder.joint_children[old_parent] if idx != joint_idx
        ]
        if not builder.joint_children[old_parent]:
            del builder.joint_children[old_parent]
    builder.joint_children.setdefault(parent_body, []).append((child_body, joint_idx))


def _disable_shape_collisions(builder: Any, shape_indices: range | list[int]) -> None:
    """Disable imported connector collisions for the minimal rod path."""
    collision_bits = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES)
    for shape_idx in shape_indices:
        builder.shape_flags[shape_idx] = int(builder.shape_flags[shape_idx]) & ~collision_bits
        builder.shape_collision_group[shape_idx] = 0


def filter_body_self_collisions(builder: Any, body_indices: list[int]) -> None:
    """Disable collisions among shapes attached to the given bodies."""
    shape_indices: list[int] = []
    for body_idx in body_indices:
        shape_indices.extend(int(shape_idx) for shape_idx in builder.body_shapes.get(int(body_idx), ()))

    for i, shape_a in enumerate(shape_indices):
        for shape_b in shape_indices[i + 1:]:
            builder.add_shape_collision_filter_pair(shape_a, shape_b)


def _filter_shapes_against_bodies(builder: Any, shape_indices: list[int], body_indices: list[int]) -> None:
    """Disable collisions between selected shapes and all shapes on selected bodies."""
    body_shape_indices: list[int] = []
    for body_idx in body_indices:
        body_shape_indices.extend(int(shape_idx) for shape_idx in builder.body_shapes.get(int(body_idx), ()))

    for shape_idx in shape_indices:
        for body_shape_idx in body_shape_indices:
            if shape_idx == body_shape_idx:
                continue
            builder.add_shape_collision_filter_pair(shape_idx, body_shape_idx)


def shift_transform_translation(xform: wp.transform, offset: np.ndarray) -> wp.transform:
    """Return an equivalent local transform after moving its body origin by offset."""
    p = np.asarray((float(xform.p[0]), float(xform.p[1]), float(xform.p[2])), dtype=np.float64) - offset
    return wp.transform(wp.vec3(float(p[0]), float(p[1]), float(p[2])), xform.q)


def _shape_local_points(builder: Any, shape_idx: int) -> np.ndarray:
    """Return representative shape points in its body-local frame."""
    xform = builder.shape_transform[shape_idx]
    scale = builder.shape_scale[shape_idx]
    src = builder.shape_source[shape_idx]
    vertices = getattr(src, "vertices", None)
    if vertices is not None:
        points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        scaled = points * np.asarray((float(scale[0]), float(scale[1]), float(scale[2])), dtype=np.float64)
    else:
        hx, hy, hz = float(scale[0]), float(scale[1]), float(scale[2])
        scaled = np.asarray(
            [
                (sx * hx, sy * hy, sz * hz)
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )

    local_points = []
    for point in scaled:
        p_local = wp.transform_point(xform, wp.vec3(float(point[0]), float(point[1]), float(point[2])))
        local_points.append((float(p_local[0]), float(p_local[1]), float(p_local[2])))
    return np.asarray(local_points, dtype=np.float64)


def recenter_body_frames(builder: Any, body_indices: list[int]) -> dict[int, np.ndarray]:
    """Move imported body origins to local geometry centers while preserving world geometry."""
    offsets: dict[int, np.ndarray] = {}
    local_coms: dict[int, np.ndarray] = {}
    for body_idx in body_indices:
        body_points: list[np.ndarray] = []
        shape_indices = list(builder.body_shapes.get(int(body_idx), ()))
        collider_shape_indices = [
            int(shape_idx)
            for shape_idx in shape_indices
            if "/Colliders" in str(builder.shape_label[int(shape_idx)])
            or "/Collider" in str(builder.shape_label[int(shape_idx)])
        ]
        for shape_idx in collider_shape_indices or shape_indices:
            points = _shape_local_points(builder, int(shape_idx))
            if points.size > 0:
                body_points.append(points)
        if not body_points:
            continue

        points_np = np.concatenate(body_points, axis=0)
        points_np = points_np[np.isfinite(points_np).all(axis=1)]
        if points_np.size == 0:
            continue

        offset = 0.5 * (points_np.min(axis=0) + points_np.max(axis=0))
        if float(np.linalg.norm(offset)) <= 1.0e-8:
            continue
        offsets[int(body_idx)] = offset
        local_coms[int(body_idx)] = points_np.mean(axis=0) - offset

    for body_idx, offset in offsets.items():
        old_body_xform = builder.body_q[body_idx]
        new_body_p = wp.transform_point(old_body_xform, wp.vec3(float(offset[0]), float(offset[1]), float(offset[2])))
        builder.body_q[body_idx] = wp.transform(new_body_p, old_body_xform.q)

        local_com = local_coms[body_idx]
        builder.body_com[body_idx] = wp.vec3(
            float(local_com[0]),
            float(local_com[1]),
            float(local_com[2]),
        )

        for shape_idx in builder.body_shapes.get(body_idx, ()):
            builder.shape_transform[int(shape_idx)] = shift_transform_translation(
                builder.shape_transform[int(shape_idx)],
                offset,
            )

    for joint_idx in range(len(builder.joint_type)):
        parent_body = int(builder.joint_parent[joint_idx])
        child_body = int(builder.joint_child[joint_idx])
        if parent_body in offsets:
            builder.joint_X_p[joint_idx] = shift_transform_translation(
                builder.joint_X_p[joint_idx],
                offsets[parent_body],
            )
        if child_body in offsets:
            builder.joint_X_c[joint_idx] = shift_transform_translation(
                builder.joint_X_c[joint_idx],
                offsets[child_body],
            )

    return offsets


def hide_oversized_connector_visuals(builder: Any, body_indices: list[int]) -> None:
    """Hide polluted connector visuals and show their clean collider fallback."""
    visible_bit = int(newton.ShapeFlags.VISIBLE)
    for body_idx in body_indices:
        shape_indices = [int(shape_idx) for shape_idx in builder.body_shapes.get(int(body_idx), ())]
        collider_shape_indices = [
            shape_idx
            for shape_idx in shape_indices
            if "/Colliders" in str(builder.shape_label[shape_idx])
            or "/Collider" in str(builder.shape_label[shape_idx])
        ]
        if not collider_shape_indices:
            continue

        collider_points = [
            points
            for shape_idx in collider_shape_indices
            if (points := _shape_local_points(builder, shape_idx)).size > 0
        ]
        if not collider_points:
            continue
        collider_np = np.concatenate(collider_points, axis=0)
        collider_extent = float(np.max(collider_np.max(axis=0) - collider_np.min(axis=0)))
        if collider_extent <= 0.0:
            continue

        hidden_visual = False
        for shape_idx in shape_indices:
            if shape_idx in collider_shape_indices:
                continue
            if not int(builder.shape_flags[shape_idx]) & visible_bit:
                continue
            visual_points = _shape_local_points(builder, shape_idx)
            if visual_points.size == 0:
                continue
            visual_extent = float(np.max(visual_points.max(axis=0) - visual_points.min(axis=0)))
            if visual_extent > max(collider_extent * 6.0, collider_extent + 0.25):
                builder.shape_flags[shape_idx] = int(builder.shape_flags[shape_idx]) & ~visible_bit
                hidden_visual = True

        if hidden_visual:
            for shape_idx in collider_shape_indices:
                builder.shape_flags[shape_idx] = int(builder.shape_flags[shape_idx]) | visible_bit
                builder.shape_color[shape_idx] = (1.0, 1.0, 1.0)


def plan_rod_rigid_imports(
    usd_path: str,
    params: dict,
    points: np.ndarray,
    quaternions: list[wp.quat],
) -> tuple[list[_RodAttachmentComponent], set[str], set[str], set[str], set[str]]:
    """Split rigid USD content into rod-attached connector components and leftovers."""
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        return [], set(), set(), set(), set()

    rigid_body_paths = _collect_rigid_body_paths(stage)
    all_joint_links = _collect_joint_links(stage, rigid_body_paths)
    if not rigid_body_paths:
        return [], set(), set(), set(), set()

    helper_paths = {
        str(path)
        for path in (
            params.get("guidePrimPath"),
            params.get("centerlineSourcePath"),
            params.get("radiusSourcePath"),
        )
        if path
    }
    helper_body_paths = rigid_body_paths & helper_paths

    adjacency: dict[str, set[str]] = {path: set() for path in rigid_body_paths}
    joint_lookup: dict[frozenset[str], set[str]] = {}
    for link in all_joint_links:
        adjacency.setdefault(link.body0_path, set()).add(link.body1_path)
        adjacency.setdefault(link.body1_path, set()).add(link.body0_path)
        joint_lookup.setdefault(frozenset((link.body0_path, link.body1_path)), set()).add(link.prim_path)

    meters_per_unit, up_axis = stage_units(stage)
    xform_cache = UsdGeom.XformCache()
    source_path = str(params.get("radiusSourcePath") or "")

    source_roots: list[str] = []
    for link in all_joint_links:
        if link.body0_path == source_path and link.body1_path not in helper_body_paths:
            source_roots.append(link.body1_path)
        elif link.body1_path == source_path and link.body0_path not in helper_body_paths:
            source_roots.append(link.body0_path)

    components: list[_RodAttachmentComponent] = []
    visited_bodies: set[str] = set()
    for root_path in source_roots:
        if root_path in visited_bodies or root_path in helper_body_paths:
            continue

        body_paths, joint_paths = _collect_component_paths(
            root_path,
            adjacency=adjacency,
            joint_lookup=joint_lookup,
            blocked_paths=helper_body_paths,
        )
        if not body_paths:
            continue

        component_points: list[np.ndarray] = []
        for body_path in sorted(body_paths):
            prim = stage.GetPrimAtPath(body_path)
            points_world = _read_rigid_body_points(
                stage,
                prim,
                meters_per_unit=meters_per_unit,
                up_axis=up_axis,
                xform_cache=xform_cache,
                prefer_colliders=True,
            )
            if points_world.size > 0:
                component_points.append(points_world)
        merged_points = (
            np.concatenate(component_points, axis=0)
            if component_points
            else np.empty((0, 3), dtype=np.float64)
        )
        root_prim = stage.GetPrimAtPath(root_path)
        root_world = _rigid_transform_to_newton(
            xform_cache.GetLocalToWorldTransform(root_prim),
            meters_per_unit=meters_per_unit,
            up_axis=up_axis,
        )
        proxy_bounds = _component_proxy_bounds(merged_points, root_world)
        endpoint_name = _component_endpoint_name(points, merged_points)
        relative_xform, parent_xform, child_xform = _component_relative_xform(
            stage=stage,
            root_body_path=root_path,
            endpoint_name=endpoint_name,
            points=points,
            quaternions=quaternions,
            rod_radius=float(params["radius"]),
            meters_per_unit=meters_per_unit,
            up_axis=up_axis,
            xform_cache=xform_cache,
        )
        components.append(
            _RodAttachmentComponent(
                root_body_path=root_path,
                body_paths=body_paths,
                joint_paths=joint_paths,
                endpoint_name=endpoint_name,
                relative_xform=relative_xform,
                parent_xform=parent_xform,
                child_xform=child_xform,
                proxy_bounds=proxy_bounds,
            )
        )
        visited_bodies.update(body_paths)

    remaining_body_paths = rigid_body_paths - helper_body_paths - visited_bodies
    remaining_joint_paths = {
        link.prim_path
        for link in all_joint_links
        if link.body0_path in remaining_body_paths and link.body1_path in remaining_body_paths
    }
    all_joint_paths = {link.prim_path for link in all_joint_links}
    return components, remaining_body_paths, remaining_joint_paths, rigid_body_paths, all_joint_paths


def attach_rod_connector_component(
    builder: Any,
    *,
    component: _RodAttachmentComponent,
    usd_path: str,
    all_body_paths: set[str],
    all_joint_paths: set[str],
    extra_ignored_paths: list[str],
    rod_bodies: list[int],
    positions: list[wp.vec3],
    quaternions: list[wp.quat],
) -> tuple[int, list[int]]:
    """Import one connector component and attach it to a rod endpoint."""
    ignore_paths = _subset_ignore_paths(
        all_body_paths=all_body_paths,
        all_joint_paths=all_joint_paths,
        keep_body_paths=set(component.body_paths),
        keep_joint_paths=set(component.joint_paths),
        extra_ignored_paths=extra_ignored_paths,
    )
    parent_body = rod_bodies[0] if component.endpoint_name == "start" else rod_bodies[-1]
    parent_world = (
        wp.transform(positions[0], quaternions[0])
        if component.endpoint_name == "start"
        else wp.transform(positions[-2], quaternions[-1])
    )
    root_relative_xform = wp.transform(*component.relative_xform)
    world_xform = parent_world * root_relative_xform
    parent_anchor_xform = wp.transform(*component.parent_xform)
    child_anchor_xform = wp.transform(*component.child_xform)
    parent_articulation = builder._find_articulation_for_body(parent_body)
    joints_before = len(builder.joint_type)
    articulations_before = len(builder.articulation_start)
    shapes_before = len(builder.shape_flags)
    result = builder.add_usd(
        usd_path,
        xform=(world_xform.p, world_xform.q),
        floating=False,
        skip_mesh_approximation=True,
        enable_self_collisions=False,
        hide_collision_shapes=True,
        ignore_paths=ignore_paths,
    )
    component_shape_indices = range(shapes_before, len(builder.shape_flags))
    path_body_map = result.get("path_body_map", {})
    root_body = path_body_map.get(component.root_body_path)
    if root_body is None:
        raise RuntimeError(f"Failed to import rod connector root body: {component.root_body_path}")
    component_body_indices = [
        int(body_idx)
        for body_path in sorted(component.body_paths)
        if (body_idx := path_body_map.get(body_path)) is not None
    ]
    body_frame_offsets = recenter_body_frames(builder, component_body_indices)
    root_body_offset = body_frame_offsets.get(int(root_body))
    if root_body_offset is not None:
        child_anchor_xform = shift_transform_translation(child_anchor_xform, root_body_offset)
    hide_oversized_connector_visuals(builder, component_body_indices)
    new_joint_indices = list(range(joints_before, len(builder.joint_type)))
    root_joint_idx = next(
        (
            joint_idx
            for joint_idx in new_joint_indices
            if int(builder.joint_child[joint_idx]) == int(root_body)
            and int(builder.joint_parent[joint_idx]) == -1
        ),
        None,
    )
    if root_joint_idx is None:
        root_joint_idx = builder.add_joint_fixed(
            parent=parent_body,
            child=int(root_body),
            parent_xform=parent_anchor_xform,
            child_xform=child_anchor_xform,
            label=f"{component.root_body_path}__rod_attach",
        )
        new_joint_indices.append(root_joint_idx)
    else:
        if int(builder.joint_type[root_joint_idx]) != int(newton.JointType.FIXED):
            raise RuntimeError(
                f"Rod connector root joint must be fixed, got {builder.joint_type[root_joint_idx]}: "
                f"{component.root_body_path}"
            )
        _reparent_existing_joint(
            builder,
            joint_idx=root_joint_idx,
            parent_body=parent_body,
            parent_xform=parent_anchor_xform,
            child_xform=child_anchor_xform,
            label=f"{component.root_body_path}__rod_attach",
        )

    if parent_articulation is not None:
        for joint_idx in new_joint_indices:
            builder.joint_articulation[joint_idx] = parent_articulation
    if len(builder.articulation_start) > articulations_before:
        del builder.articulation_start[articulations_before:]
        del builder.articulation_label[articulations_before:]
        del builder.articulation_world[articulations_before:]
    _disable_shape_collisions(builder, component_shape_indices)
    if component.proxy_bounds is not None:
        proxy_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            gap=0.0,
            has_shape_collision=True,
            has_particle_collision=False,
            is_visible=False,
            mu=1.0,
            collision_group=1,
        )
        proxy_center, proxy_half_extents = component.proxy_bounds
        if root_body_offset is not None:
            proxy_center = wp.vec3(
                float(proxy_center[0]) - float(root_body_offset[0]),
                float(proxy_center[1]) - float(root_body_offset[1]),
                float(proxy_center[2]) - float(root_body_offset[2]),
            )
        proxy_shape_idx = builder.add_shape_box(
            int(root_body),
            xform=wp.transform(proxy_center, wp.quat_identity()),
            hx=float(proxy_half_extents[0]),
            hy=float(proxy_half_extents[1]),
            hz=float(proxy_half_extents[2]),
            cfg=proxy_cfg,
            label=f"{component.root_body_path}__contact_proxy",
        )
        _filter_shapes_against_bodies(builder, [proxy_shape_idx], rod_bodies)
    return int(parent_body), component_body_indices


def import_remaining_rigid_content(
    builder: Any,
    *,
    usd_path: str,
    all_body_paths: set[str],
    all_joint_paths: set[str],
    remaining_body_paths: set[str],
    remaining_joint_paths: set[str],
    extra_ignored_paths: list[str],
) -> None:
    """Import any non-connector rigid content that remains on the stage."""
    if not (remaining_body_paths or remaining_joint_paths):
        return
    ignore_paths = _subset_ignore_paths(
        all_body_paths=all_body_paths,
        all_joint_paths=all_joint_paths,
        keep_body_paths=set(remaining_body_paths),
        keep_joint_paths=set(remaining_joint_paths),
        extra_ignored_paths=extra_ignored_paths,
    )
    builder.add_usd(
        usd_path,
        skip_mesh_approximation=True,
        ignore_paths=ignore_paths,
    )
