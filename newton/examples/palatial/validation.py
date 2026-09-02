# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Semantic validation helpers for Palatial Newton playback examples."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CollisionSupportSample:
    """Measured and proxy collision support extents for one body state."""

    surface_min_z: float
    aabb_proxy_min_z: float
    exact_shape_count: int
    aabb_fallback_shape_count: int


class CollisionSupportSampler:
    """Measure dynamic collision surfaces against a world-space support plane.

    Mesh-backed shapes use their scaled source vertices. Shapes without source
    vertices fall back to their oriented collision AABB. The independent AABB
    result is retained so reports expose approximation error instead of hiding
    it inside a single penetration number.
    """

    def __init__(
        self,
        *,
        shape_body: np.ndarray,
        shape_transform: np.ndarray,
        shape_aabb_lower: np.ndarray,
        shape_aabb_upper: np.ndarray,
        shape_local_vertices: list[np.ndarray | None] | tuple[np.ndarray | None, ...] | None = None,
        shape_flags: np.ndarray | None = None,
        collide_shapes_flag: int | None = None,
    ) -> None:
        bodies = np.asarray(shape_body, dtype=np.int64).reshape(-1)
        transforms = np.asarray(shape_transform, dtype=np.float64)
        lower = np.asarray(shape_aabb_lower, dtype=np.float64)
        upper = np.asarray(shape_aabb_upper, dtype=np.float64)
        shape_count = bodies.shape[0]
        if not (shape_count == transforms.shape[0] == lower.shape[0] == upper.shape[0]):
            raise ValueError("shape arrays must have matching leading dimensions")
        if transforms.ndim != 2 or transforms.shape[1] != 7:
            raise ValueError("shape_transform must have shape (N, 7)")
        if lower.shape != (shape_count, 3) or upper.shape != (shape_count, 3):
            raise ValueError("shape AABBs must have shape (N, 3)")
        if shape_local_vertices is None:
            vertices: list[np.ndarray | None] = [None] * shape_count
        else:
            if len(shape_local_vertices) != shape_count:
                raise ValueError("shape_local_vertices must contain one entry per shape")
            vertices = []
            for value in shape_local_vertices:
                if value is None:
                    vertices.append(None)
                    continue
                points = np.asarray(value, dtype=np.float64)
                if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
                    raise ValueError("shape-local vertices must contain 3D points")
                if not np.isfinite(points).all():
                    raise ValueError("shape-local vertices must be finite")
                vertices.append(points.copy())
        if shape_flags is None:
            if collide_shapes_flag is not None:
                raise ValueError("collide_shapes_flag requires shape_flags")
            collision_enabled = np.ones(shape_count, dtype=bool)
        else:
            flags = np.asarray(shape_flags, dtype=np.int64).reshape(-1)
            if flags.shape[0] != shape_count:
                raise ValueError("shape_flags must contain one value per shape")
            if collide_shapes_flag is None:
                raise ValueError("shape_flags requires collide_shapes_flag")
            collision_enabled = (flags & int(collide_shapes_flag)) != 0

        self._shapes: list[tuple[int, np.ndarray, np.ndarray, np.ndarray | None]] = []
        for index, body_index in enumerate(bodies):
            if body_index < 0 or not collision_enabled[index]:
                continue
            corners = np.array(
                [
                    (x, y, z)
                    for x in (lower[index, 0], upper[index, 0])
                    for y in (lower[index, 1], upper[index, 1])
                    for z in (lower[index, 2], upper[index, 2])
                ],
                dtype=np.float64,
            )
            self._shapes.append((int(body_index), transforms[index].copy(), corners, vertices[index]))
        self.exact_shape_count = sum(vertices is not None for _, _, _, vertices in self._shapes)
        self.aabb_fallback_shape_count = len(self._shapes) - self.exact_shape_count

    @property
    def support_extent_method(self) -> str:
        """Return the fidelity of the collision-surface measurement."""
        if self.exact_shape_count and not self.aabb_fallback_shape_count:
            return "collision_mesh_vertices_v2"
        if self.exact_shape_count:
            return "collision_mesh_vertices_with_aabb_fallback_v2"
        return "collision_shape_aabb_corners_v1"

    @property
    def shape_count(self) -> int:
        """Return the number of sampled dynamic collision shapes."""
        return len(self._shapes)

    def sample(self, body_q: np.ndarray) -> CollisionSupportSample:
        """Return exact-or-fallback and pure-AABB minimum world Z extents."""
        bodies = np.asarray(body_q, dtype=np.float64)
        if bodies.ndim != 2 or bodies.shape[1] != 7:
            raise ValueError("body_q must have shape (N, 7)")
        if not self._shapes:
            raise ValueError("no dynamic collision shapes are available")

        surface_min_z = math.inf
        proxy_min_z = math.inf
        for body_index, shape_transform, corners, local_vertices in self._shapes:
            if body_index >= bodies.shape[0]:
                raise ValueError("shape body index is outside body_q")
            proxy_points = _transform_points(corners, shape_transform)
            proxy_points = _transform_points(proxy_points, bodies[body_index])
            shape_proxy_min_z = float(proxy_points[:, 2].min())
            proxy_min_z = min(proxy_min_z, shape_proxy_min_z)
            if local_vertices is None:
                shape_surface_min_z = shape_proxy_min_z
            else:
                surface_points = _transform_points(local_vertices, shape_transform)
                surface_points = _transform_points(surface_points, bodies[body_index])
                shape_surface_min_z = float(surface_points[:, 2].min())
            surface_min_z = min(surface_min_z, shape_surface_min_z)

        return CollisionSupportSample(
            surface_min_z=surface_min_z,
            aabb_proxy_min_z=proxy_min_z,
            exact_shape_count=self.exact_shape_count,
            aabb_fallback_shape_count=self.aabb_fallback_shape_count,
        )


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
        settling_window_s: float = 0.5,
        initial_support_min_z: float | None = None,
        initial_support_proxy_min_z: float | None = None,
        support_extent_method: str = "point_collision_radius_v1",
        support_proxy_extent_method: str | None = None,
        support_exact_shape_count: int = 0,
        support_aabb_fallback_shape_count: int = 0,
        solver_name: str = "unspecified",
        body_type: str = "unknown",
        frames_per_second: float | None = None,
        substeps: int | None = None,
        solver_iterations: int | None = None,
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
        self.settling_window_s = float(settling_window_s)
        if not math.isfinite(self.settling_window_s) or self.settling_window_s <= 0.0:
            raise ValueError("settling_window_s must be finite and positive")
        self.support_extent_method = str(support_extent_method)
        if not self.support_extent_method:
            raise ValueError("support_extent_method must not be empty")
        self.support_proxy_extent_method = str(support_proxy_extent_method or self.support_extent_method)
        self.support_exact_shape_count = int(support_exact_shape_count)
        self.support_aabb_fallback_shape_count = int(support_aabb_fallback_shape_count)
        if self.support_exact_shape_count < 0 or self.support_aabb_fallback_shape_count < 0:
            raise ValueError("support shape counts must be non-negative")
        self.solver_name = str(solver_name)
        self.body_type = str(body_type)
        self.frames_per_second = None if frames_per_second is None else float(frames_per_second)
        if self.frames_per_second is not None and (
            not math.isfinite(self.frames_per_second) or self.frames_per_second <= 0.0
        ):
            raise ValueError("frames_per_second must be finite and positive")
        self.substeps = None if substeps is None else int(substeps)
        self.solver_iterations = None if solver_iterations is None else int(solver_iterations)
        if self.substeps is not None and self.substeps <= 0:
            raise ValueError("substeps must be positive")
        if self.solver_iterations is not None and self.solver_iterations <= 0:
            raise ValueError("solver_iterations must be positive")
        point_support_min_z = float((points[:, 2] - self._point_radii).min())
        support_min_z = point_support_min_z if initial_support_min_z is None else float(initial_support_min_z)
        support_proxy_min_z = (
            support_min_z if initial_support_proxy_min_z is None else float(initial_support_proxy_min_z)
        )
        if not math.isfinite(support_min_z) or not math.isfinite(support_proxy_min_z):
            raise ValueError("initial support extents must be finite")
        self.initial_penetration_m = max(
            0.0,
            self.support_plane_z - support_min_z,
        )
        self.max_support_penetration_m = self.initial_penetration_m
        self.max_support_penetration_proxy_m = max(
            0.0,
            self.support_plane_z - support_proxy_min_z,
        )
        self.final_support_penetration_m = self.initial_penetration_m
        self.peak_support_penetration_sample: int | None = None
        self.peak_support_penetration_time_s: float | None = None
        self._support_penetration_samples: list[tuple[float, float]] = []
        self.sample_count = 0
        self.centroid_in_envelope_samples = 0
        self.finite_transforms = True
        self.max_displacement_m = 0.0
        self.max_linear_speed_m_s = 0.0

    def observe(
        self,
        points: np.ndarray,
        linear_velocities: np.ndarray,
        *,
        support_min_z: float | None = None,
        support_proxy_min_z: float | None = None,
        sample_time_s: float | None = None,
    ) -> None:
        """Record one rendered/simulated frame's geometry and velocity state."""
        points = np.asarray(points, dtype=np.float64)
        velocities = np.asarray(linear_velocities, dtype=np.float64)
        sample_index = self.sample_count
        self.sample_count += 1
        if sample_time_s is None:
            sample_time_s = (
                float(sample_index) / self.frames_per_second
                if self.frames_per_second is not None
                else float(sample_index)
            )
        sample_time_s = float(sample_time_s)
        finite = bool(np.isfinite(points).all() and np.isfinite(velocities).all() and math.isfinite(sample_time_s))
        self.finite_transforms = self.finite_transforms and finite
        if not finite or points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            return
        if points.shape[0] != self._point_radii.shape[0]:
            self.finite_transforms = False
            return
        center = points.mean(axis=0)
        displacement = float(np.linalg.norm(center - self._initial_center))
        self.max_displacement_m = max(self.max_displacement_m, displacement)
        point_support_min_z = float((points[:, 2] - self._point_radii).min())
        measured_support_min_z = point_support_min_z if support_min_z is None else float(support_min_z)
        proxy_min_z = measured_support_min_z if support_proxy_min_z is None else float(support_proxy_min_z)
        if not math.isfinite(measured_support_min_z) or not math.isfinite(proxy_min_z):
            self.finite_transforms = False
            return
        support_penetration = max(0.0, self.support_plane_z - measured_support_min_z)
        support_penetration_proxy = max(0.0, self.support_plane_z - proxy_min_z)
        if self.peak_support_penetration_sample is None or support_penetration > self.max_support_penetration_m:
            self.peak_support_penetration_sample = sample_index
            self.peak_support_penetration_time_s = sample_time_s
        self.max_support_penetration_m = max(self.max_support_penetration_m, support_penetration)
        self.max_support_penetration_proxy_m = max(
            self.max_support_penetration_proxy_m,
            support_penetration_proxy,
        )
        self.final_support_penetration_m = support_penetration
        self._support_penetration_samples.append((sample_time_s, support_penetration))
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
        if self._support_penetration_samples:
            final_time_s = self._support_penetration_samples[-1][0]
            settling_start_s = final_time_s - self.settling_window_s
            settled_samples = [
                penetration
                for sample_time, penetration in self._support_penetration_samples
                if sample_time >= settling_start_s
            ]
        else:
            settled_samples = []
        settled_max_support_penetration_m = max(settled_samples, default=self.final_support_penetration_m)
        advisories: list[str] = []
        if self.initial_penetration_m > self.initial_penetration_tolerance_m:
            advisories.append("initial_support_penetration")
        elif self.max_support_penetration_m > self.support_penetration_tolerance_m:
            advisories.append("trajectory_support_penetration")
        failures: list[str] = []
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
            "simulation": {
                "solver": self.solver_name,
                "body_type": self.body_type,
                "frames_per_second": self.frames_per_second,
                "substeps": self.substeps,
                "solver_iterations": self.solver_iterations,
            },
            "summary": {
                "passed": passed,
                "initial_penetration_m": self.initial_penetration_m,
                "max_support_penetration_m": self.max_support_penetration_m,
                "max_transient_support_penetration_m": (self.max_support_penetration_m),
                "max_support_penetration_proxy_m": (self.max_support_penetration_proxy_m),
                "peak_support_penetration_sample": (self.peak_support_penetration_sample),
                "peak_support_penetration_time_s": (self.peak_support_penetration_time_s),
                "final_support_penetration_m": (self.final_support_penetration_m),
                "settled_max_support_penetration_m": (settled_max_support_penetration_m),
                "settled_sample_count": len(settled_samples),
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
            "supportExtentMethod": self.support_extent_method,
            "supportProxyExtentMethod": self.support_proxy_extent_method,
            "supportPenetration": {
                "kind": "collision_geometry_vs_support_plane",
                "acceptance": "advisory",
                "support_plane_z_m": self.support_plane_z,
                "settling_window_s": self.settling_window_s,
                "exact_shape_count": self.support_exact_shape_count,
                "aabb_fallback_shape_count": (self.support_aabb_fallback_shape_count),
            },
            "advisories": advisories,
            "failures": failures,
        }

    def write(self, path: str | Path) -> dict[str, object]:
        """Write and return the semantic report."""
        report = self.report()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
