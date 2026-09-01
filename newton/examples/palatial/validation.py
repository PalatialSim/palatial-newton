# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Semantic validation helpers for Palatial Newton playback examples."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "palatial.newton-validation.v1"


def _quat_rotate(quaternion: np.ndarray, points: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    xyz = quaternion[:3]
    scalar = float(quaternion[3])
    first_cross = np.cross(np.broadcast_to(xyz, points.shape), points)
    second_cross = np.cross(np.broadcast_to(xyz, points.shape), first_cross)
    return points + 2.0 * (scalar * first_cross + second_cross)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    return _quat_rotate(transform[3:7], points) + transform[:3]


def compute_world_shape_points(
    body_q: np.ndarray,
    shape_body: np.ndarray,
    shape_transform: np.ndarray,
    shape_aabb_lower: np.ndarray,
    shape_aabb_upper: np.ndarray,
    *,
    shape_flags: np.ndarray | None = None,
    collide_shapes_flag: int | None = None,
) -> np.ndarray:
    """Return world-space AABB corners for body-bound collision shapes.

    Static and visual-only shapes are excluded so neither a support plane nor
    a render bound can decide collision penetration.  ``shape_flags`` remains
    optional for callers that provide collision-only arrays themselves.
    """
    body_q = np.asarray(body_q, dtype=np.float64)
    shape_body = np.asarray(shape_body, dtype=np.int64).reshape(-1)
    shape_transform = np.asarray(shape_transform, dtype=np.float64)
    shape_aabb_lower = np.asarray(shape_aabb_lower, dtype=np.float64)
    shape_aabb_upper = np.asarray(shape_aabb_upper, dtype=np.float64)
    if not (shape_body.shape[0] == shape_transform.shape[0] == shape_aabb_lower.shape[0] == shape_aabb_upper.shape[0]):
        raise ValueError("shape arrays must have matching leading dimensions")
    if shape_flags is None:
        if collide_shapes_flag is not None:
            raise ValueError("collide_shapes_flag requires shape_flags")
        collision_enabled = np.ones(shape_body.shape[0], dtype=bool)
    else:
        flags = np.asarray(shape_flags, dtype=np.int64).reshape(-1)
        if flags.shape[0] != shape_body.shape[0]:
            raise ValueError("shape_flags must contain one value per shape")
        if collide_shapes_flag is None:
            raise ValueError("shape_flags requires collide_shapes_flag")
        collision_enabled = (flags & int(collide_shapes_flag)) != 0

    world_points: list[np.ndarray] = []
    for index, body_index in enumerate(shape_body):
        if body_index < 0 or not collision_enabled[index]:
            continue
        if body_index >= body_q.shape[0]:
            raise ValueError("shape body index is outside body_q")
        lower = shape_aabb_lower[index]
        upper = shape_aabb_upper[index]
        corners = np.array(
            [(x, y, z) for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])],
            dtype=np.float64,
        )
        shape_points = _transform_points(corners, shape_transform[index])
        world_points.append(_transform_points(shape_points, body_q[body_index]))
    if not world_points:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(world_points, axis=0)


def clearance_translation(
    world_min_z: float,
    clearance: float,
    *,
    support_plane_z: float = 0.0,
) -> float:
    """Return the Z translation that places an asset at exact clearance."""
    values = (world_min_z, clearance, support_plane_z)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("clearance inputs must be finite")
    if clearance < 0.0:
        raise ValueError("clearance must be non-negative")
    return float(support_plane_z) + float(clearance) - float(world_min_z)


def translate_free_roots(
    joint_q: np.ndarray,
    *,
    joint_type: np.ndarray,
    joint_q_start: np.ndarray,
    delta_z: float,
    free_joint_type: int,
) -> tuple[np.ndarray, int]:
    """Translate FREE roots without changing descendant joint coordinates."""
    translated = np.asarray(joint_q).copy()
    joint_type = np.asarray(joint_type).reshape(-1)
    joint_q_start = np.asarray(joint_q_start).reshape(-1)
    if joint_q_start.shape[0] < joint_type.shape[0]:
        raise ValueError("joint_q_start must contain every joint offset")
    count = 0
    for joint_kind, start in zip(
        joint_type,
        joint_q_start[: joint_type.shape[0]],
        strict=True,
    ):
        if int(joint_kind) != int(free_joint_type):
            continue
        z_index = int(start) + 2
        if z_index >= translated.shape[0]:
            raise ValueError("FREE joint translation is outside joint_q")
        translated[z_index] += float(delta_z)
        count += 1
    return translated, count


def resolve_persistent_rigid_placement(
    *,
    translation_z: float,
    support_plane_z: float,
    joint_count: int,
    translated_free_joint_count: int,
    tolerance_m: float = 1.0e-6,
) -> tuple[float, float]:
    """Resolve rigid placement without applying a transient body transform.

    A jointed rigid asset without a FREE root has an authored fixed world
    placement. For that case, preserve the articulation and relocate the
    support plane so the authored asset keeps the requested clearance.
    """
    if not math.isfinite(float(translation_z)):
        raise ValueError("rigid placement translation must be finite")
    if not math.isfinite(float(support_plane_z)):
        raise ValueError("rigid placement support plane must be finite")
    if tolerance_m < 0.0:
        raise ValueError("rigid placement tolerance must be non-negative")
    if (
        int(joint_count) > 0
        and int(translated_free_joint_count) == 0
        and abs(float(translation_z)) > float(tolerance_m)
    ):
        return 0.0, float(support_plane_z) - float(translation_z)
    return float(translation_z), float(support_plane_z)


def relocate_ground_plane_transform(
    shape_transform: np.ndarray,
    *,
    shape_body: np.ndarray,
    shape_type: np.ndarray,
    shape_label: list[str],
    plane_type: int,
    support_plane_z: float,
) -> np.ndarray:
    """Return shape transforms with the generated global ground relocated."""
    transforms = np.asarray(shape_transform).copy()
    bodies = np.asarray(shape_body).reshape(-1)
    types = np.asarray(shape_type).reshape(-1)
    if transforms.ndim != 2 or transforms.shape[1] != 7:
        raise ValueError("shape_transform must have shape (N, 7)")
    if not (transforms.shape[0] == bodies.shape[0] == types.shape[0] == len(shape_label)):
        raise ValueError("ground plane shape arrays must have matching lengths")
    if not math.isfinite(float(support_plane_z)):
        raise ValueError("support plane height must be finite")
    matches = [
        index
        for index, label in enumerate(shape_label)
        if label == "ground_plane" and int(bodies[index]) == -1 and int(types[index]) == int(plane_type)
    ]
    if len(matches) != 1:
        raise RuntimeError("Newton model must contain exactly one generated ground plane")
    transforms[matches[0], 2] = float(support_plane_z)
    return transforms


class NewtonValidationTracker:
    """Collect deterministic trajectory metrics for a recorded playback."""

    def __init__(
        self,
        initial_points: np.ndarray,
        *,
        point_radii: np.ndarray | None = None,
        support_plane_z: float = 0.0,
        trajectory_envelope_radius_m: float | None = None,
        displacement_limit_m: float | None = None,
        speed_limit_m_s: float | None = None,
        initial_penetration_tolerance_m: float = 1.0e-4,
        support_penetration_tolerance_m: float = 1.0e-2,
    ) -> None:
        points = np.asarray(initial_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError("initial_points must contain at least one 3D point")
        if point_radii is None:
            radii = np.zeros(points.shape[0], dtype=np.float64)
        else:
            radii = np.asarray(point_radii, dtype=np.float64).reshape(-1)
            if radii.shape[0] != points.shape[0]:
                raise ValueError("point_radii must contain one radius per point")
            if not np.isfinite(radii).all() or np.any(radii < 0.0):
                raise ValueError("point_radii must be finite and non-negative")
        self._point_radii = radii
        self._initial_center = points.mean(axis=0)
        extent = points.max(axis=0) - points.min(axis=0)
        diagonal = max(float(np.linalg.norm(extent)), 1.0e-3)
        self.support_plane_z = float(support_plane_z)
        self.trajectory_envelope_radius_m = float(
            trajectory_envelope_radius_m if trajectory_envelope_radius_m is not None else max(2.0, 4.0 * diagonal)
        )
        self.displacement_limit_m = float(
            displacement_limit_m if displacement_limit_m is not None else max(5.0, 20.0 * diagonal)
        )
        self.speed_limit_m_s = float(speed_limit_m_s if speed_limit_m_s is not None else max(50.0, 100.0 * diagonal))
        self.initial_penetration_tolerance_m = float(initial_penetration_tolerance_m)
        self.support_penetration_tolerance_m = float(support_penetration_tolerance_m)
        self.initial_penetration_m = max(
            0.0,
            self.support_plane_z - float((points[:, 2] - self._point_radii).min()),
        )
        self.max_support_penetration_m = self.initial_penetration_m
        self.sample_count = 0
        self.centroid_in_envelope_samples = 0
        self.finite_transforms = True
        self.max_displacement_m = 0.0
        self.max_linear_speed_m_s = 0.0

    def observe(self, points: np.ndarray, linear_velocities: np.ndarray) -> None:
        """Record one rendered/simulated frame's geometry and velocity state."""
        points = np.asarray(points, dtype=np.float64)
        velocities = np.asarray(linear_velocities, dtype=np.float64)
        self.sample_count += 1
        finite = bool(np.isfinite(points).all() and np.isfinite(velocities).all())
        self.finite_transforms = self.finite_transforms and finite
        if not finite or points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            return
        if points.shape[0] != self._point_radii.shape[0]:
            self.finite_transforms = False
            return
        center = points.mean(axis=0)
        displacement = float(np.linalg.norm(center - self._initial_center))
        self.max_displacement_m = max(self.max_displacement_m, displacement)
        support_penetration = max(
            0.0,
            self.support_plane_z - float((points[:, 2] - self._point_radii).min()),
        )
        self.max_support_penetration_m = max(
            self.max_support_penetration_m,
            support_penetration,
        )
        if velocities.size:
            speeds = np.linalg.norm(velocities.reshape(-1, 3), axis=1)
            self.max_linear_speed_m_s = max(
                self.max_linear_speed_m_s,
                float(speeds.max(initial=0.0)),
            )
        if displacement <= self.trajectory_envelope_radius_m:
            self.centroid_in_envelope_samples += 1

    def report(self) -> dict[str, object]:
        """Return the versioned semantic report consumed by the provider."""
        centroid_in_envelope_sample_ratio = (
            float(self.centroid_in_envelope_samples) / float(self.sample_count) if self.sample_count else 0.0
        )
        trajectory_stable = bool(
            self.max_displacement_m <= self.displacement_limit_m and self.max_linear_speed_m_s <= self.speed_limit_m_s
        )
        failures: list[str] = []
        if self.initial_penetration_m > self.initial_penetration_tolerance_m:
            failures.append("initial_support_penetration")
        elif self.max_support_penetration_m > self.support_penetration_tolerance_m:
            failures.append("trajectory_support_penetration")
        if not self.finite_transforms:
            failures.append("non_finite_transform")
        if self.max_displacement_m > self.displacement_limit_m:
            failures.append("trajectory_displacement_exceeded")
        if self.max_linear_speed_m_s > self.speed_limit_m_s:
            failures.append("trajectory_speed_exceeded")
        if centroid_in_envelope_sample_ratio < 0.8:
            failures.append("trajectory_left_envelope")
        if self.sample_count == 0:
            failures.append("no_trajectory_samples")
        passed = not failures
        return {
            "schemaVersion": SCHEMA_VERSION,
            "status": "passed" if passed else "failed",
            "summary": {
                "passed": passed,
                "initial_penetration_m": self.initial_penetration_m,
                "max_support_penetration_m": self.max_support_penetration_m,
                "finite_transforms": self.finite_transforms,
                "trajectory_stable": trajectory_stable,
                "sample_count": self.sample_count,
                "max_displacement_m": self.max_displacement_m,
                "max_linear_speed_m_s": self.max_linear_speed_m_s,
                "centroid_in_envelope_sample_ratio": (centroid_in_envelope_sample_ratio),
            },
            "thresholds": {
                "initial_penetration_tolerance_m": (self.initial_penetration_tolerance_m),
                "support_penetration_tolerance_m": (self.support_penetration_tolerance_m),
                "trajectory_envelope_radius_m": (self.trajectory_envelope_radius_m),
                "centroid_in_envelope_sample_ratio": 0.8,
                "displacement_limit_m": self.displacement_limit_m,
                "speed_limit_m_s": self.speed_limit_m_s,
            },
            "trajectoryEnvelopeMethod": "centroid_displacement_v1",
            "supportExtentMethod": "point_collision_radius_v1",
            "failures": failures,
        }

    def write(self, path: str | Path) -> dict[str, object]:
        """Write and return the semantic report."""
        report = self.report()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
