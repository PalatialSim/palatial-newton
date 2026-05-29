# Example: load a Newton-ready USDA produced by an external converter and
# simulate it. Modeled after newton/examples/cloth/example_cloth_hanging.py.
#
# The converted USDA already encodes:
#   - solver name (newton:solver)
#   - timestep (newton:timeStepsPerSecond)
#   - solver tuning (newton:solver:iterations, substeps, ...)
#   - body type (newton:bodyType)
#   - cloth material (newton:cloth:triKe / triKa / triKd / edgeKe / edgeKd)
#   - cloth density / drop height / friction / restitution
#
# So this example does NOT hand-pick a solver or stiffness — it reads the
# bundle from the asset and steps. Same code works for a rigid mug, a rod
# cable, or a cloth t-shirt.
#
# Usage:
#   python -m newton.examples.palatial.example_palatial_load <converted.usda>
#       [--steps 600] [--gui] [--substeps N]

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import re
import sys

# IMPORTANT: import newton stack BEFORE any pxr.Usd usage in the same process.
import warp as wp  # noqa: F401
import newton
import numpy as np

from newton.palatial import load


def _to_newton_points(points, meters_per_unit: float, up_axis: str):
    import numpy as _np

    pts = _np.asarray(points, dtype=_np.float32) * float(meters_per_unit)
    if up_axis == "Y":
        pts = _np.stack([pts[:, 0], -pts[:, 2], pts[:, 1]], axis=1)
    return pts


def _read_usd_geometry_bounds(usd_path: str):
    try:
        from pxr import Gf, Usd, UsdGeom  # noqa: PLC0415
    except Exception:
        return None

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        return None

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    up_axis = str(UsdGeom.GetStageUpAxis(stage) or "Z")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    bounds = []
    for prim in stage.Traverse():
        if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.BasisCurves)):
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        if box.IsEmpty():
            continue
        mn = box.GetMin()
        mx = box.GetMax()
        corners = [
            (mn[0], mn[1], mn[2]),
            (mn[0], mn[1], mx[2]),
            (mn[0], mx[1], mn[2]),
            (mn[0], mx[1], mx[2]),
            (mx[0], mn[1], mn[2]),
            (mx[0], mn[1], mx[2]),
            (mx[0], mx[1], mn[2]),
            (mx[0], mx[1], mx[2]),
        ]
        bounds.append(_to_newton_points(corners, meters_per_unit, up_axis))

    if not bounds:
        return None

    import numpy as _np

    merged = _np.concatenate(bounds, axis=0)
    return merged.min(axis=0), merged.max(axis=0)


@wp.kernel
def _spin_kinematic_bodies_kernel(
    body_indices: wp.array[wp.int32],
    twist_rates: wp.array[float],
    world_twist_axes: wp.array[wp.vec3],
    local_pivot_points: wp.array[wp.vec3],
    world_pivot_points: wp.array[wp.vec3],
    dt: float,
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
    body_qd0: wp.array[wp.spatial_vector],
    body_qd1: wp.array[wp.spatial_vector],
):
    """Rotate selected kinematic bodies about a fixed world-space centerline."""
    tid = wp.tid()
    body_id = body_indices[tid]

    xform = body_q0[body_id]
    rot = wp.transform_get_rotation(xform)

    angle = twist_rates[tid] * dt
    if angle != 0.0:
        dq = wp.quat_from_axis_angle(world_twist_axes[tid], angle)
        rot = wp.mul(dq, rot)

    pos = world_pivot_points[tid] - wp.quat_rotate(rot, local_pivot_points[tid])
    twisted = wp.transform(pos, rot)
    zero_velocity = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    body_q0[body_id] = twisted
    body_q1[body_id] = twisted
    body_qd0[body_id] = zero_velocity
    body_qd1[body_id] = zero_velocity


@wp.kernel
def _copy_selected_body_q_kernel(
    body_indices: wp.array[wp.int32],
    source_q: wp.array[wp.transform],
    target_q: wp.array[wp.transform],
):
    """Copy selected body transforms between Warp transform arrays."""
    tid = wp.tid()
    body_id = body_indices[tid]
    target_q[body_id] = source_q[body_id]


@wp.kernel
def _propagate_kinematic_cluster_bodies_kernel(
    body_indices: wp.array[wp.int32],
    root_slots: wp.array[wp.int32],
    root_body_indices: wp.array[wp.int32],
    local_positions: wp.array[wp.vec3],
    local_rotations: wp.array[wp.quat],
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
    body_qd0: wp.array[wp.spatial_vector],
    body_qd1: wp.array[wp.spatial_vector],
):
    """Rigidly propagate kinematic connector cluster members from their root bodies."""
    tid = wp.tid()
    body_id = body_indices[tid]
    root_body_id = root_body_indices[root_slots[tid]]

    root_xform = body_q0[root_body_id]
    root_pos = wp.transform_get_translation(root_xform)
    root_rot = wp.transform_get_rotation(root_xform)

    world_pos = root_pos + wp.quat_rotate(root_rot, local_positions[tid])
    world_rot = wp.mul(root_rot, local_rotations[tid])
    xform = wp.transform(world_pos, world_rot)
    zero_velocity = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    body_q0[body_id] = xform
    body_q1[body_id] = xform
    body_qd0[body_id] = zero_velocity
    body_qd1[body_id] = zero_velocity


def _find_longest_rod_body_chain(body_labels: list[str]) -> list[int]:
    """Return the longest contiguous rod body chain inferred from body labels."""
    pattern = re.compile(r"^(?P<prefix>.+)_edge_body_(?P<index>\d+)$")
    groups: dict[str, list[tuple[int, int]]] = {}

    for body_idx, label in enumerate(body_labels):
        match = pattern.match(str(label))
        if not match:
            continue
        prefix = match.group("prefix")
        index = int(match.group("index"))
        groups.setdefault(prefix, []).append((index, body_idx))

    best_chain: list[int] = []
    for entries in groups.values():
        ordered = [body_idx for _index, body_idx in sorted(entries)]
        if len(ordered) > len(best_chain):
            best_chain = ordered
    return best_chain


def _infer_rod_endpoint_bodies(body_labels: list[str]) -> list[int]:
    """Return the first/last body ids of the longest inferred rod body chain."""
    rod_chain = _find_longest_rod_body_chain(body_labels)
    if len(rod_chain) < 2:
        return []
    return [int(rod_chain[0]), int(rod_chain[-1])]


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by an ``(x, y, z, w)`` quaternion."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    uv = np.cross(q[:3], v)
    uuv = np.cross(q[:3], uv)
    return v + 2.0 * (q[3] * uv + uuv)


def _normalize_axis(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Return a normalized 3D direction, or the fallback if degenerate."""
    norm = float(np.linalg.norm(values))
    if norm <= 1.0e-12:
        fallback = np.asarray(fallback, dtype=np.float64)
        fallback_norm = float(np.linalg.norm(fallback))
        if fallback_norm <= 1.0e-12:
            return np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        return fallback / fallback_norm
    return np.asarray(values, dtype=np.float64) / norm


def _transform_point_np(body_q_row: np.ndarray, local_point: np.ndarray) -> np.ndarray:
    """Transform a local point by a body pose row ``[px,py,pz,qx,qy,qz,qw]``."""
    return np.asarray(body_q_row[:3], dtype=np.float64) + _quat_rotate_np(
        np.asarray(body_q_row[3:7], dtype=np.float64),
        local_point,
    )


@dataclass
class _RodTwistTargetSpec:
    """One prescribed rod-end twist target."""

    body_index: int
    rod_body_index: int | None
    attach_joint_index: int
    local_axis: np.ndarray
    local_pivot: np.ndarray
    world_axis: np.ndarray
    world_pivot: np.ndarray


@dataclass
class _RigidClusterSpec:
    """Bodies that should move as one rigid connector assembly."""

    root_body_index: int
    member_body_indices: list[int]
    member_local_positions: list[np.ndarray]
    member_local_rotations: list[np.ndarray]
    internal_joint_indices: list[int]


def _infer_rod_twist_targets(model, body_q: np.ndarray | None = None) -> list[_RodTwistTargetSpec]:
    """Choose which bodies to twist and the fixed pivot/axis data to use.

    Prefer connector root bodies attached through ``__rod_attach`` joints so the
    visible connectors are what actually spin in the example. Each target stores
    the connector-local attachment point plus a fixed world-space pivot/axis so
    the connector stays clamped while twisting around the cable centerline while
    the rod endpoint remains dynamic and attached through the fixed joint.
    """
    if body_q is None:
        body_q = model.body_q.numpy()

    body_labels = list(model.body_label)
    rod_endpoints = _infer_rod_endpoint_bodies(body_labels)
    if len(rod_endpoints) < 2:
        return []

    targets_by_endpoint: dict[int, _RodTwistTargetSpec] = {}
    joint_labels = list(getattr(model, "joint_label", []))
    if joint_labels:
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_X_p = model.joint_X_p.numpy()
        joint_X_c = model.joint_X_c.numpy()
        for joint_idx, joint_label in enumerate(joint_labels):
            if not str(joint_label).endswith("__rod_attach"):
                continue
            rod_body = int(joint_parent[joint_idx])
            connector_body = int(joint_child[joint_idx])
            if rod_body not in rod_endpoints:
                continue
            connector_body_q = np.asarray(body_q[connector_body], dtype=np.float64)
            child_joint_q = np.asarray(joint_X_c[joint_idx, 3:7], dtype=np.float64)
            local_axis = _normalize_axis(
                _quat_rotate_np(child_joint_q, np.asarray((0.0, 0.0, 1.0), dtype=np.float64)),
                np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
            )
            local_pivot = np.asarray(joint_X_c[joint_idx, :3], dtype=np.float64)
            world_axis = _normalize_axis(
                _quat_rotate_np(connector_body_q[3:7], local_axis),
                local_axis,
            )
            world_pivot = _transform_point_np(connector_body_q, local_pivot)
            targets_by_endpoint[rod_body] = _RodTwistTargetSpec(
                body_index=connector_body,
                rod_body_index=rod_body,
                attach_joint_index=int(joint_idx),
                local_axis=local_axis,
                local_pivot=local_pivot,
                world_axis=world_axis,
                world_pivot=world_pivot,
            )

    target_specs: list[_RodTwistTargetSpec] = []
    for endpoint_body in rod_endpoints:
        target = targets_by_endpoint.get(endpoint_body)
        if target is None:
            endpoint_body_q = np.asarray(body_q[endpoint_body], dtype=np.float64)
            local_axis = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
            local_pivot = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
            target = _RodTwistTargetSpec(
                body_index=int(endpoint_body),
                rod_body_index=int(endpoint_body),
                attach_joint_index=-1,
                local_axis=local_axis,
                local_pivot=local_pivot,
                world_axis=_normalize_axis(
                    _quat_rotate_np(endpoint_body_q[3:7], local_axis),
                    local_axis,
                ),
                world_pivot=_transform_point_np(endpoint_body_q, local_pivot),
            )
        target_specs.append(target)
    return target_specs


def _quat_conjugate_np(q: np.ndarray) -> np.ndarray:
    """Return quaternion conjugate for ``(x, y, z, w)`` storage."""
    return np.asarray((-q[0], -q[1], -q[2], q[3]), dtype=np.float64)


def _quat_mul_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two quaternions in ``(x, y, z, w)`` storage."""
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


def _infer_rigid_connector_clusters(
    model,
    body_q: np.ndarray,
    twist_targets: list[_RodTwistTargetSpec],
) -> list[_RigidClusterSpec]:
    """Infer connector body clusters hanging off twist-root bodies."""
    root_set = {int(target.body_index) for target in twist_targets}
    joint_labels = list(getattr(model, "joint_label", []))
    if not joint_labels:
        return []

    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    adjacency: dict[int, list[tuple[int, int]]] = {}
    for joint_idx, joint_label in enumerate(joint_labels):
        if str(joint_label).endswith("__rod_attach"):
            continue
        parent = int(joint_parent[joint_idx])
        child = int(joint_child[joint_idx])
        if parent < 0 or child < 0:
            continue
        adjacency.setdefault(parent, []).append((child, joint_idx))
        adjacency.setdefault(child, []).append((parent, joint_idx))

    cluster_specs: list[_RigidClusterSpec] = []
    for target in twist_targets:
        root_body_index = int(target.body_index)
        if target.attach_joint_index < 0:
            cluster_specs.append(
                _RigidClusterSpec(
                    root_body_index=root_body_index,
                    member_body_indices=[],
                    member_local_positions=[],
                    member_local_rotations=[],
                    internal_joint_indices=[],
                )
            )
            continue

        visited = {root_body_index}
        member_body_indices: list[int] = []
        member_local_positions: list[np.ndarray] = []
        member_local_rotations: list[np.ndarray] = []
        internal_joint_indices: list[int] = []
        stack = [
            (neighbor, int(joint_idx))
            for neighbor, joint_idx in adjacency.get(root_body_index, ())
            if neighbor not in root_set
        ]
        root_pose = np.asarray(body_q[int(root_body_index)], dtype=np.float64)
        root_pos = root_pose[:3]
        root_rot = root_pose[3:7]
        root_rot_inv = _quat_conjugate_np(root_rot)

        while stack:
            current, incoming_joint_idx = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            internal_joint_indices.append(int(incoming_joint_idx))
            current_pose = np.asarray(body_q[int(current)], dtype=np.float64)
            local_pos = _quat_rotate_np(root_rot_inv, current_pose[:3] - root_pos)
            local_rot = _quat_mul_np(root_rot_inv, current_pose[3:7])
            member_body_indices.append(int(current))
            member_local_positions.append(local_pos)
            member_local_rotations.append(local_rot)
            for neighbor, joint_idx in adjacency.get(current, ()):
                if neighbor in visited or neighbor in root_set:
                    continue
                stack.append((neighbor, int(joint_idx)))

        cluster_specs.append(
            _RigidClusterSpec(
                root_body_index=int(root_body_index),
                member_body_indices=member_body_indices,
                member_local_positions=member_local_positions,
                member_local_rotations=member_local_rotations,
                internal_joint_indices=internal_joint_indices,
            )
        )
    return cluster_specs


def _set_bodies_kinematic(model, body_indices: list[int]) -> None:
    """Zero out mass and inertia for selected bodies on a finalized model."""
    if not body_indices:
        return

    body_mass = model.body_mass.numpy().copy()
    body_inv_mass = model.body_inv_mass.numpy().copy()
    body_inertia = model.body_inertia.numpy().copy()
    body_inv_inertia = model.body_inv_inertia.numpy().copy()

    for body_idx in body_indices:
        body_mass[int(body_idx)] = 0.0
        body_inv_mass[int(body_idx)] = 0.0
        body_inertia[int(body_idx)] = 0.0
        body_inv_inertia[int(body_idx)] = 0.0

    model.body_mass.assign(body_mass)
    model.body_inv_mass.assign(body_inv_mass)
    model.body_inertia.assign(body_inertia)
    model.body_inv_inertia.assign(body_inv_inertia)


class Example:
    def __init__(self, viewer, usd_path: str, substeps: int | None = None,
                 drop_height: float = 0.0, device: str | None = None,
                 solver_iterations: int | None = None,
                 rod_twist_rate: float = 0.0,
                 rod_twist_end_time: float | None = None,
                 rod_stretch_stiffness: float | None = None,
                 rod_stretch_damping: float | None = None,
                 rod_bend_stiffness: float | None = None,
                 rod_bend_damping: float | None = None,
                 zero_gravity: bool = False,
                 gravity_scale: float = 1.0,
                 cloth_particle_radius: float = 0.008,
                 soft_contact_ke: float = 100.0,
                 soft_contact_kd: float = 2e-3,
                 soft_contact_mu: float = 1.0,
                 soft_contact_max: int = 1_000_000,
                 cloth_body_contact_margin: float = 0.01,
                 bending_ke: float = 1e-4,
                 bending_kd: float = 1e-3,
                 vbd_particle_edge_contact_buffer_size: int = 64,
                 vbd_particle_collision_detection_interval: int = -1,
                 vbd_rigid_contact_k_start: float | None = None,
                 joint_q_overrides: dict[int, float] | None = None,
                 joint_target_overrides: dict[int, float] | None = None,
                 joint_target_ke: float | None = None,
                 joint_target_kd: float | None = None,
                 list_joints: bool = False,
                 rotate_x_deg: float = 0.0,
                 rotate_y_deg: float = 0.0,
                 rotate_z_deg: float = 0.0):
        self.viewer = viewer

        # One call: parse USDA, build model, build solver with baked params.
        bundle = load(usd_path, device=device)
        self.bundle = bundle
        self.model = bundle.model
        self.solver = bundle.solver
        self.state_0 = bundle.state_in
        self.state_1 = bundle.state_out
        self.control = bundle.control

        # Frame timing comes from newton:timeStepsPerSecond in the USDA.
        self.fps = bundle.fps
        self.frame_dt = 1.0 / float(self.fps)

        # Substeps: caller override > USDA newton:solver:substeps > default 1.
        if substeps is not None:
            self.sim_substeps = max(1, int(substeps))
        else:
            self.sim_substeps = max(1, int(bundle.solver_params.get("substeps", 1)))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        def _sync_body_pose_teleport() -> None:
            if int(self.model.body_count) <= 0 or self.state_0.body_q is None:
                return
            if self.state_1.body_q is not None:
                wp.copy(self.state_1.body_q, self.state_0.body_q)
            if self.state_0.body_qd is not None:
                self.state_0.body_qd.zero_()
            if self.state_1.body_qd is not None:
                self.state_1.body_qd.zero_()
            body_q_prev = getattr(self.solver, "body_q_prev", None)
            if body_q_prev is not None:
                wp.copy(body_q_prev, self.state_0.body_q)

        self._rod_twist_body_indices = None
        self._rod_twist_rates = None
        self._rod_twist_local_axes = None
        self._rod_twist_local_pivots = None
        self._rod_twist_world_axes = None
        self._rod_twist_world_pivots = None
        self._rod_twist_all_body_indices = None
        self._rod_twist_cluster_body_indices = None
        self._rod_twist_cluster_root_slots = None
        self._rod_twist_cluster_local_positions = None
        self._rod_twist_cluster_local_rotations = None
        self._rod_twist_device = None
        self._rod_twist_elapsed = 0.0
        self._rod_twist_end_time = math.inf if rod_twist_end_time is None else float(rod_twist_end_time)

        def _apply_rod_twist_boundary(dt: float) -> None:
            if (
                self._rod_twist_body_indices is None
                or self._rod_twist_rates is None
                or self._rod_twist_local_axes is None
                or self._rod_twist_local_pivots is None
                or self._rod_twist_world_axes is None
                or self._rod_twist_world_pivots is None
            ):
                return
            if self._rod_twist_device is None:
                raise RuntimeError("rod twist boundary is missing a Warp launch device")
            wp.launch(
                kernel=_spin_kinematic_bodies_kernel,
                dim=self._rod_twist_body_indices.shape[0],
                inputs=[
                    self._rod_twist_body_indices,
                    self._rod_twist_rates,
                    self._rod_twist_world_axes,
                    self._rod_twist_local_pivots,
                    self._rod_twist_world_pivots,
                    float(dt),
                    self.state_0.body_q,
                    self.state_1.body_q,
                    self.state_0.body_qd,
                    self.state_1.body_qd,
                ],
                device=self._rod_twist_device,
            )
            if self._rod_twist_cluster_body_indices is not None and self._rod_twist_cluster_body_indices.shape[0] > 0:
                wp.launch(
                    kernel=_propagate_kinematic_cluster_bodies_kernel,
                    dim=self._rod_twist_cluster_body_indices.shape[0],
                    inputs=[
                        self._rod_twist_cluster_body_indices,
                        self._rod_twist_cluster_root_slots,
                        self._rod_twist_body_indices,
                        self._rod_twist_cluster_local_positions,
                        self._rod_twist_cluster_local_rotations,
                        self.state_0.body_q,
                        self.state_1.body_q,
                        self.state_0.body_qd,
                        self.state_1.body_qd,
                    ],
                    device=self._rod_twist_device,
                )
            body_q_prev = getattr(self.solver, "body_q_prev", None)
            if body_q_prev is not None:
                if body_q_prev.device != self._rod_twist_device:
                    wp.copy(body_q_prev, self.state_0.body_q)
                else:
                    wp.launch(
                        kernel=_copy_selected_body_q_kernel,
                        dim=self._rod_twist_all_body_indices.shape[0],
                        inputs=[self._rod_twist_all_body_indices, self.state_0.body_q, body_q_prev],
                        device=self._rod_twist_device,
                    )
        self._apply_rod_twist_boundary = _apply_rod_twist_boundary

        def _rebuild_solver_with_params(params: dict) -> None:
            import inspect as _inspect

            alias = {
                "iterations": "iterations",
                "substeps": "substeps",
                "particleEnableSelfContact": "particle_enable_self_contact",
                "particleSelfContactRadius": "particle_self_contact_radius",
                "particleSelfContactMargin": "particle_self_contact_margin",
            }
            try:
                accepted = set(_inspect.signature(type(self.solver).__init__).parameters)
            except (TypeError, ValueError):
                accepted = set()
            kwargs = {}
            for key, value in params.items():
                py_key = alias.get(key, key)
                if py_key in accepted:
                    kwargs[py_key] = value
            self.solver = type(self.solver)(self.model, **kwargs)
            bundle.solver = self.solver
            bundle.solver_params = dict(params)
            _sync_body_pose_teleport()

        solver_params = dict(bundle.solver_params)
        current_iterations = int(solver_params.get("iterations", getattr(self.solver, "iterations", 0) or 0))
        target_iterations = solver_iterations
        if target_iterations is not None:
            if target_iterations <= 0:
                raise ValueError(f"solver_iterations must be > 0, got {target_iterations}")
            solver_params["iterations"] = int(target_iterations)
            _rebuild_solver_with_params(solver_params)
            if int(target_iterations) != current_iterations:
                print(f"  solver-iterations: {current_iterations} -> {int(target_iterations)}")

        if zero_gravity:
            self.model.set_gravity((0.0, 0.0, 0.0))
        elif gravity_scale != 1.0:
            if not math.isfinite(gravity_scale):
                raise ValueError(f"gravity_scale must be finite, got {gravity_scale}")
            if gravity_scale < 0.0:
                raise ValueError(f"gravity_scale must be >= 0, got {gravity_scale}")
            gravity = self.model.gravity.numpy() * float(gravity_scale)
            self.model.set_gravity(gravity)
            print(f"  gravity-scale: x{gravity_scale:g}")

        # ---- Optional rotation of the asset around world axes (degrees) ----
        # Runs BEFORE drop_height so the lift always ends up along world +z,
        # regardless of which axis the user rotates around.
        if rotate_x_deg or rotate_y_deg or rotate_z_deg:
            import math as _math
            import numpy as _np

            def _axis_quat(axis, deg):
                a = _math.radians(deg) * 0.5
                s, c = _math.sin(a), _math.cos(a)
                return _np.array([
                    s if axis == 0 else 0.0,
                    s if axis == 1 else 0.0,
                    s if axis == 2 else 0.0,
                    c,
                ], dtype=_np.float32)

            def _qmul(a, b):
                ax, ay, az, aw = a
                bx, by, bz, bw = b
                return _np.array([
                    aw*bx + ax*bw + ay*bz - az*by,
                    aw*by - ax*bz + ay*bw + az*bx,
                    aw*bz + ax*by - ay*bx + az*bw,
                    aw*bw - ax*bx - ay*by - az*bz,
                ], dtype=_np.float32)

            def _qrot_vec(q, v):
                # rotate vec3 v by quat (qx,qy,qz,qw)
                qx, qy, qz, qw = q
                # cross(q.xyz, v) * 2
                t = 2.0 * _np.cross([qx, qy, qz], v)
                return v + qw * t + _np.cross([qx, qy, qz], t)

            qrot = _np.array([0.0, 0.0, 0.0, 1.0], dtype=_np.float32)
            if rotate_x_deg: qrot = _qmul(_axis_quat(0, rotate_x_deg), qrot)
            if rotate_y_deg: qrot = _qmul(_axis_quat(1, rotate_y_deg), qrot)
            if rotate_z_deg: qrot = _qmul(_axis_quat(2, rotate_z_deg), qrot)

            n_p = int(self.model.particle_count)
            n_b = int(self.model.body_count)
            n_j = int(self.model.joint_count)
            applied_to = []

            # Cloth particles: rotate each particle around origin.
            if bundle.body_type == "cloth" and n_p > 0:
                pq = self.state_0.particle_q.numpy().copy()
                for i in range(pq.shape[0]):
                    pq[i] = _qrot_vec(qrot, pq[i])
                self.state_0.particle_q.assign(pq)
                applied_to.append(f"particle_q×{n_p}")

            # Rigid articulated: rotate FREE root joint quaternion(s).
            elif bundle.body_type == "rigid" and n_j > 0:
                free = int(newton.JointType.FREE)
                jtypes = self.model.joint_type.numpy()
                jq_start = self.model.joint_q_start.numpy()
                jq = self.state_0.joint_q.numpy().copy()
                n_rot = 0
                for i, t in enumerate(jtypes):
                    if int(t) == free:
                        s = int(jq_start[i])
                        # Rotate translation around origin too.
                        p = _np.array([jq[s], jq[s+1], jq[s+2]], dtype=_np.float32)
                        p = _qrot_vec(qrot, p)
                        jq[s], jq[s+1], jq[s+2] = p
                        cur = jq[s+3:s+7]
                        nq = _qmul(qrot, cur)
                        jq[s+3:s+7] = nq
                        n_rot += 1
                if n_rot:
                    self.state_0.joint_q.assign(jq)
                    applied_to.append(f"FREE root joint(s)×{n_rot}")

            # Rods and plain rigid bodies: rotate body poses directly.
            elif n_b > 0:
                bq = self.state_0.body_q.numpy().copy()
                for i in range(bq.shape[0]):
                    p = _qrot_vec(qrot, bq[i, 0:3])
                    cur = bq[i, 3:7]
                    nq = _qmul(qrot, cur)
                    bq[i, 0:3] = p
                    bq[i, 3:7] = nq
                self.state_0.body_q.assign(bq)
                applied_to.append(f"body_q×{n_b}")

            print(f"  rotate: x={rotate_x_deg} y={rotate_y_deg} z={rotate_z_deg} deg → {applied_to or 'no targets'}")
            if n_b > 0:
                _sync_body_pose_teleport()

        # Drop height: lift the asset by drop_height meters in z.
        # For articulated assets with a floating base, world pose lives in
        # joint_q of the FREE root joint — body_q is derived from it via FK
        # at every step, so lifting body_q alone gets overwritten.
        # For cloth, lift particle_q directly (one row per particle).
        if drop_height:
            import numpy as _np
            n_bodies = int(self.model.body_count)
            n_joints = int(self.model.joint_count)
            n_particles = int(self.model.particle_count)
            n_lifted_free = 0

            if bundle.body_type == "rigid":
                # 1. Lift FREE root joints in joint_q (articulated assets).
                #    A FREE joint has 7 coords: [px, py, pz, qx, qy, qz, qw].
                if n_joints > 0:
                    jq = self.state_0.joint_q.numpy().copy()
                    jtypes = self.model.joint_type.numpy()
                    jq_start = self.model.joint_q_start.numpy()
                    free = int(newton.JointType.FREE)  # = 4
                    for i, t in enumerate(jtypes):
                        if int(t) == free:
                            jq[int(jq_start[i]) + 2] += float(drop_height)
                            n_lifted_free += 1
                    if n_lifted_free:
                        self.state_0.joint_q.assign(jq)
                        # Re-derive body_q from the new joint_q via forward kinematics.
                        newton.eval_fk(self.model, self.state_0.joint_q,
                                       self.state_0.joint_qd, self.state_0)
                        _sync_body_pose_teleport()

                # 2. Otherwise (no free joints, plain rigid): lift body_q directly.
                elif n_bodies > 0:
                    q = self.state_0.body_q.numpy().copy()
                    q[:, 2] += float(drop_height)
                    self.state_0.body_q.assign(q)
                    _sync_body_pose_teleport()

            elif bundle.body_type == "rod" and n_bodies > 0:
                q = self.state_0.body_q.numpy().copy()
                q[:, 2] += float(drop_height)
                self.state_0.body_q.assign(q)
                _sync_body_pose_teleport()

            elif bundle.body_type == "cloth" and n_particles > 0:
                pq = self.state_0.particle_q.numpy().copy()
                pq[:, 2] += float(drop_height)
                self.state_0.particle_q.assign(pq)

            print(
                f"  drop: +{drop_height}m  body={bundle.body_type}  "
                f"(free_joints={n_lifted_free}, bodies={n_bodies}, "
                f"joints={n_joints}, particles={n_particles})"
            )

        # Cloth-friendly soft-contact defaults (overridden by viewer/asset later).
        if bundle.body_type == "cloth":
            self.cloth_particle_radius = cloth_particle_radius
            self.model.soft_contact_ke = soft_contact_ke
            self.model.soft_contact_kd = soft_contact_kd
            self.model.soft_contact_mu = soft_contact_mu
            self.model.soft_contact_max = soft_contact_max
            import numpy as _np
            n_particles = int(self.model.particle_count)
            if n_particles > 0:
                self.model.particle_radius.assign(
                    _np.full(n_particles, self.cloth_particle_radius, dtype=_np.float32)
                )
            self.model.cloth_body_contact_margin = cloth_body_contact_margin
            self.model.bending_ke = bending_ke
            self.model.bending_kd = bending_kd

            # VBD solver extras (no-op for other solvers).
            if type(self.solver).__name__ == "SolverVBD":
                self.solver.particle_edge_contact_buffer_size = vbd_particle_edge_contact_buffer_size
                self.solver.particle_collision_detection_interval = vbd_particle_collision_detection_interval
                self.solver.rigid_contact_k_start = (
                    soft_contact_ke if vbd_rigid_contact_k_start is None
                    else vbd_rigid_contact_k_start
                )

        if bundle.body_type == "rod" and hasattr(self.model, "joint_target_ke"):
            gains_changed = False
            ke = self.model.joint_target_ke.numpy().copy()
            kd = self.model.joint_target_kd.numpy().copy() if hasattr(self.model, "joint_target_kd") else None
            if rod_stretch_stiffness is not None:
                ke[0::2] = float(rod_stretch_stiffness)
                gains_changed = True
            if rod_bend_stiffness is not None:
                ke[1::2] = float(rod_bend_stiffness)
                gains_changed = True
            if kd is not None and rod_stretch_damping is not None:
                kd[0::2] = float(rod_stretch_damping)
                gains_changed = True
            if kd is not None and rod_bend_damping is not None:
                kd[1::2] = float(rod_bend_damping)
                gains_changed = True
            if gains_changed:
                self.model.joint_target_ke.assign(ke)
                if kd is not None:
                    self.model.joint_target_kd.assign(kd)
                _rebuild_solver_with_params(dict(bundle.solver_params))
                print(
                    "  rod-gains: "
                    f"stretch_ke={rod_stretch_stiffness} "
                    f"stretch_kd={rod_stretch_damping} "
                    f"bend_ke={rod_bend_stiffness} "
                    f"bend_kd={rod_bend_damping}"
                )

        # Diagnostic: confirm gravity + per-particle mass non-zero.
        try:
            import numpy as _np
            n_p = int(self.model.particle_count)
            grav = getattr(self.model, "gravity", None)
            if n_p > 0:
                inv_m = self.model.particle_inv_mass.numpy()
                pq0 = self.state_0.particle_q.numpy()
                n_pinned = int((_np.asarray(inv_m) == 0.0).sum())
                print(
                    f"  diag: gravity={grav}  particles={n_p}  "
                    f"pinned(inv_mass==0)={n_pinned}  "
                    f"inv_mass[min,mean,max]=[{inv_m.min():.3e},{inv_m.mean():.3e},{inv_m.max():.3e}]  "
                    f"z[min,mean,max]=[{pq0[:,2].min():.3f},{pq0[:,2].mean():.3f},{pq0[:,2].max():.3f}]"
                )
            else:
                n_b = int(self.model.body_count)
                if n_b > 0:
                    inv_m = self.model.body_inv_mass.numpy()
                    n_pinned = int((_np.asarray(inv_m) == 0.0).sum())
                    # Mass = 1/inv_mass (skip pinned bodies in stats).
                    safe = _np.asarray(inv_m, dtype=float)
                    nz = safe[safe > 0.0]
                    if nz.size:
                        m = 1.0 / nz
                        m_stat = f"mass[min,mean,max]=[{m.min():.3f},{m.mean():.3f},{m.max():.3f}]kg"
                    else:
                        m_stat = "mass=N/A (all pinned)"
                    print(
                        f"  diag: gravity={grav}  bodies={n_b}  "
                        f"static(inv_mass==0)={n_pinned}  {m_stat}"
                    )
                    # Per-body breakdown (small rigid sets are usually tiny).
                    if n_b <= 16:
                        for i in range(n_b):
                            mi = (1.0 / inv_m[i]) if inv_m[i] > 0 else float("inf")
                            print(f"    body[{i}] mass={mi:.4f}kg  inv_mass={float(inv_m[i]):.4f}")
        except Exception as _e:
            print(f"  diag: print failed: {_e}")

        # ---- Sync body_q with joint_q via FK so rigid articulations don't snap on first step ----
        if bundle.body_type == "rigid" and int(self.model.joint_count) > 0:
            try:
                newton.eval_fk(self.model, self.state_0.joint_q,
                               self.state_0.joint_qd, self.state_0)
                jq = self.state_0.joint_q.numpy()
                print(f"  init joint_q={jq.tolist()}")
            except Exception as _e:
                print(f"  eval_fk skipped: {_e}")

        # ---- Joint introspection / overrides ----
        n_joints = int(self.model.joint_count)
        self.joint_target_overrides: dict[int, float] = dict(joint_target_overrides or {})
        if bundle.body_type == "rigid" and n_joints > 0:
            import numpy as _np
            jtypes = self.model.joint_type.numpy()
            jq_start = self.model.joint_q_start.numpy()
            jqd_start = self.model.joint_qd_start.numpy()
            type_names = {
                0: "PRISMATIC", 1: "REVOLUTE", 2: "BALL",
                3: "FIXED", 4: "FREE", 5: "DISTANCE",
                6: "D6",
            }
            if list_joints:
                print("  joints:")
                for i in range(n_joints):
                    tn = type_names.get(int(jtypes[i]), str(int(jtypes[i])))
                    print(f"    [{i}] type={tn}  q_start={int(jq_start[i])}  qd_start={int(jqd_start[i])}")

            # Apply --joint-q overrides on initial state.
            if joint_q_overrides:
                jq = self.state_0.joint_q.numpy().copy()
                for i, v in joint_q_overrides.items():
                    if 0 <= i < n_joints:
                        jq[int(jq_start[i])] = float(v)
                    else:
                        print(f"  warn: joint index {i} out of range (n_joints={n_joints})")
                self.state_0.joint_q.assign(jq)
                newton.eval_fk(self.model, self.state_0.joint_q,
                               self.state_0.joint_qd, self.state_0)

            # Optional PD gain bump. MuJoCo bakes ke/kd at solver init, so
            # rebuild the solver if we touch them.
            gains_changed = False
            if joint_target_ke is not None and hasattr(self.model, "joint_target_ke"):
                ke = self.model.joint_target_ke.numpy().copy()
                ke[:] = float(joint_target_ke)
                self.model.joint_target_ke.assign(ke)
                gains_changed = True
            if joint_target_kd is not None and hasattr(self.model, "joint_target_kd"):
                kd = self.model.joint_target_kd.numpy().copy()
                kd[:] = float(joint_target_kd)
                self.model.joint_target_kd.assign(kd)
                gains_changed = True
            if gains_changed:
                self.solver = type(self.solver)(self.model)
                print(f"  rebuilding solver with new PD gains: ke={joint_target_ke} kd={joint_target_kd}")

            # Bake joint_target overrides into control. MuJoCo expects
            # joint_target_pos (sized by joint_qd_count, indexed by qd_start);
            # older code paths use joint_target (sized by joint_q_count).
            if self.joint_target_overrides and self.control is not None:
                target_attr = None
                for name in ("joint_target_pos", "joint_target"):
                    if hasattr(self.control, name):
                        target_attr = name
                        break
                if target_attr is not None:
                    arr = getattr(self.control, target_attr)
                    jt = arr.numpy().copy()
                    starts = jqd_start if target_attr == "joint_target_pos" else jq_start
                    for i, v in self.joint_target_overrides.items():
                        if 0 <= i < n_joints:
                            idx = int(starts[i])
                            if 0 <= idx < jt.shape[0]:
                                jt[idx] = float(v)
                    arr.assign(jt)

        if abs(float(rod_twist_rate)) > 0.0:
            if bundle.body_type != "rod":
                print(f"  warn: ignoring --rod-twist-rate for non-rod body_type={bundle.body_type}")
            else:
                initial_body_q = self.state_0.body_q.numpy()
                twist_targets = _infer_rod_twist_targets(self.model, initial_body_q)
                if len(twist_targets) < 2:
                    raise RuntimeError("Could not infer rod endpoint bodies from model.body_label")
                if self.state_0.body_q is None or self.state_1.body_q is None:
                    raise RuntimeError("Rod twist boundary requires rigid body transforms")
                if self.state_0.body_qd is None or self.state_1.body_qd is None:
                    raise RuntimeError("Rod twist boundary requires rigid body velocities")
                self._rod_twist_device = self.state_0.body_q.device
                twist_body_indices = [target.body_index for target in twist_targets]
                cluster_specs = _infer_rigid_connector_clusters(self.model, initial_body_q, twist_targets)
                cluster_body_indices = [
                    body_idx
                    for cluster in cluster_specs
                    for body_idx in cluster.member_body_indices
                ]
                all_kinematic_body_indices = [*twist_body_indices, *cluster_body_indices]
                _set_bodies_kinematic(self.model, all_kinematic_body_indices)
                if getattr(self.model, "joint_enabled", None) is not None:
                    joint_enabled = self.model.joint_enabled.numpy().copy()
                    for cluster in cluster_specs:
                        for joint_idx in cluster.internal_joint_indices:
                            joint_enabled[int(joint_idx)] = False
                    self.model.joint_enabled.assign(joint_enabled)
                # Some solvers cache inverse masses / joint topology at construction time.
                # Rebuild after marking kinematic bodies and disabling internal connector joints
                # so the prescribed boundary is solved against the updated model state.
                _rebuild_solver_with_params(solver_params)
                self._rod_twist_body_indices = wp.array(
                    np.asarray(twist_body_indices, dtype=np.int32),
                    dtype=wp.int32,
                    device=self._rod_twist_device,
                )
                self._rod_twist_all_body_indices = wp.array(
                    np.asarray(all_kinematic_body_indices, dtype=np.int32),
                    dtype=wp.int32,
                    device=self._rod_twist_device,
                )
                self._rod_twist_rates = wp.array(
                    np.asarray((-float(rod_twist_rate), float(rod_twist_rate)), dtype=np.float32),
                    dtype=wp.float32,
                    device=self._rod_twist_device,
                )
                self._rod_twist_local_axes = wp.array(
                    np.asarray([target.local_axis for target in twist_targets], dtype=np.float32),
                    dtype=wp.vec3,
                    device=self._rod_twist_device,
                )
                self._rod_twist_local_pivots = wp.array(
                    np.asarray([target.local_pivot for target in twist_targets], dtype=np.float32),
                    dtype=wp.vec3,
                    device=self._rod_twist_device,
                )
                self._rod_twist_world_axes = wp.array(
                    np.asarray([target.world_axis for target in twist_targets], dtype=np.float32),
                    dtype=wp.vec3,
                    device=self._rod_twist_device,
                )
                self._rod_twist_world_pivots = wp.array(
                    np.asarray([target.world_pivot for target in twist_targets], dtype=np.float32),
                    dtype=wp.vec3,
                    device=self._rod_twist_device,
                )
                cluster_root_slots = [
                    root_slot
                    for root_slot, cluster in enumerate(cluster_specs)
                    for _body_idx in cluster.member_body_indices
                ]
                cluster_local_positions = [
                    local_pos
                    for cluster in cluster_specs
                    for local_pos in cluster.member_local_positions
                ]
                cluster_local_rotations = [
                    local_rot
                    for cluster in cluster_specs
                    for local_rot in cluster.member_local_rotations
                ]
                self._rod_twist_cluster_body_indices = wp.array(
                    np.asarray(cluster_body_indices, dtype=np.int32),
                    dtype=wp.int32,
                    device=self._rod_twist_device,
                )
                self._rod_twist_cluster_root_slots = wp.array(
                    np.asarray(cluster_root_slots, dtype=np.int32),
                    dtype=wp.int32,
                    device=self._rod_twist_device,
                )
                self._rod_twist_cluster_local_positions = wp.array(
                    np.asarray(cluster_local_positions, dtype=np.float32).reshape((-1, 3)),
                    dtype=wp.vec3,
                    device=self._rod_twist_device,
                )
                self._rod_twist_cluster_local_rotations = wp.array(
                    np.asarray(cluster_local_rotations, dtype=np.float32).reshape((-1, 4)),
                    dtype=wp.quat,
                    device=self._rod_twist_device,
                )
                _apply_rod_twist_boundary(0.0)
                print(
                    "  rod-twist: "
                    f"bodies={twist_body_indices} "
                    f"labels={[self.model.body_label[i] for i in twist_body_indices]} "
                    f"cluster_members={len(cluster_body_indices)} "
                    f"pivots={[target.world_pivot.tolist() for target in twist_targets]} "
                    f"rate={float(rod_twist_rate):.3f}rad/s "
                    f"end_time={self._rod_twist_end_time if math.isfinite(self._rod_twist_end_time) else 'inf'}"
                )

        self.contacts = self.model.collide(self.state_0)
        self.viewer.set_model(self.model)

        print(
            f"[load] usd={usd_path}  body={bundle.body_type}  "
            f"solver={bundle.solver_name}  fps={self.fps}  substeps={self.sim_substeps}  "
            f"params={bundle.solver_params}"
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            if self._rod_twist_body_indices is not None:
                twist_dt = 0.0
                if self._rod_twist_elapsed < self._rod_twist_end_time:
                    remaining = self._rod_twist_end_time - self._rod_twist_elapsed
                    twist_dt = self.sim_dt if not math.isfinite(remaining) else min(self.sim_dt, max(0.0, remaining))
                self._apply_rod_twist_boundary(twist_dt)
                self._rod_twist_elapsed += self.sim_dt
            self.viewer.apply_forces(self.state_0)
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            if self._rod_twist_body_indices is not None:
                self._apply_rod_twist_boundary(0.0)

    def step(self):
        self._frame_idx = getattr(self, "_frame_idx", -1) + 1
        self.simulate()
        self.sim_time += self.frame_dt
        # Per-frame debug: always print first 5 frames, then every 30.
        try:
            if self._frame_idx < 5 or self._frame_idx % 30 == 0:
                if int(self.model.particle_count) > 0:
                    z = self.state_0.particle_q.numpy()[:, 2]
                    print(f"  step#{self._frame_idx} t={self.sim_time:.3f}s  z[min,mean,max]=[{z.min():.3f},{z.mean():.3f},{z.max():.3f}]")
                elif int(self.model.body_count) > 0:
                    z = self.state_0.body_q.numpy()[:, 2]
                    print(f"  step#{self._frame_idx} t={self.sim_time:.3f}s  body_z[min,max]=[{z.min():.3f},{z.max():.3f}]")
        except Exception:
            pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="example_load_converted")
    p.add_argument("usd", help="Path to a converted *.newton.usda")
    p.add_argument("--steps", type=int, default=600,
                   help="Frames to simulate when not in GUI mode")
    p.add_argument("--substeps", type=int, default=None,
                   help="Override newton:solver:substeps from the USDA")
    p.add_argument("--solver-iterations", type=int, default=None,
                   help="Override newton:solver:iterations from the USDA")
    p.add_argument("--gui", action="store_true",
                   help="Open ViewerGL and run until the window is closed")
    p.add_argument("--drop-height", type=float, default=0.0,
                   help="Lift rigid or rod bodies by this many meters in z before sim "
                        "(ignored for cloth — cloth drop height is baked into the USDA)")
    p.add_argument("--rod-twist-rate", type=float, default=0.0,
                   help="For rod bundles only: counter-twist both rod endpoints at this angular speed in rad/s. "
                        "0 disables the boundary condition.")
    p.add_argument("--rod-twist-end-time", type=float, default=None,
                   help="For rod bundles only: stop the prescribed endpoint twist after this many seconds. "
                        "Default is no time limit.")
    p.add_argument("--rod-stretch-stiffness", type=float, default=None,
                   help="For rod bundles only: override axial stretch stiffness.")
    p.add_argument("--rod-stretch-damping", type=float, default=None,
                   help="For rod bundles only: override axial stretch damping.")
    p.add_argument("--rod-bend-stiffness", type=float, default=None,
                   help="For rod bundles only: override isotropic bend/twist stiffness.")
    p.add_argument("--rod-bend-damping", type=float, default=None,
                   help="For rod bundles only: override isotropic bend/twist damping.")
    p.add_argument("--device", default=None,
                   help="Warp device, e.g. 'cuda:0' or 'cpu' (default: GPU if available)")
    p.add_argument("--zero-gravity", action="store_true",
                   help="Override the loaded scene gravity with (0, 0, 0)")
    p.add_argument("--gravity-scale", type=float, default=1.0,
                   help="Scale the loaded scene gravity by this factor "
                        "(ignored when --zero-gravity is set)")

    # Optional physics JSON: overrides --cloth-particle-radius from
    # solver.vbd_particle_self_contact_radius and --bending-ke from
    # cloth.bend_stiffness when present.
    p.add_argument("--physics-json", default=None,
                   help="Path to a physics JSON. If given, "
                        "solver.vbd_particle_self_contact_radius overrides "
                        "--cloth-particle-radius and cloth.bend_stiffness "
                        "overrides --bending-ke.")

    # Cloth tuning (only applied when bundle.body_type == "cloth").
    p.add_argument("--cloth-particle-radius", type=float, default=0.008)
    p.add_argument("--soft-contact-ke", type=float, default=100.0)
    p.add_argument("--soft-contact-kd", type=float, default=2e-3)
    p.add_argument("--soft-contact-mu", type=float, default=1.0)
    p.add_argument("--soft-contact-max", type=int, default=1_000_000)
    p.add_argument("--cloth-body-contact-margin", type=float, default=0.01)
    p.add_argument("--bending-ke", type=float, default=1e-4)
    p.add_argument("--bending-kd", type=float, default=1e-3)

    # VBD-only knobs (ignored for other solvers).
    p.add_argument("--vbd-particle-edge-contact-buffer-size", type=int, default=64)
    p.add_argument("--vbd-particle-collision-detection-interval", type=int, default=-1)
    p.add_argument("--vbd-rigid-contact-k-start", type=float, default=None,
                   help="Defaults to --soft-contact-ke when omitted")

    # Joint inspection / driving (rigid articulated assets).
    p.add_argument("--list-joints", action="store_true",
                   help="Print joint table on load (index, type, q_start, qd_start)")
    p.add_argument("--joint-q", default=None,
                   help='Initial joint positions, e.g. "1=0.78,2=-0.3" (radians for revolute, meters for prismatic). Index is joint number, sets first DOF.')
    p.add_argument("--joint-target", default=None,
                   help='Constant PD targets, same syntax as --joint-q. Held every step.')
    p.add_argument("--joint-target-ke", type=float, default=None,
                   help="Override model.joint_target_ke for ALL DOFs (PD position gain)")
    p.add_argument("--joint-target-kd", type=float, default=None,
                   help="Override model.joint_target_kd for ALL DOFs (PD velocity gain)")
    p.add_argument("--rotate-x", type=float, default=0.0, help="Rotate asset around X (degrees)")
    p.add_argument("--rotate-y", type=float, default=0.0, help="Rotate asset around Y (degrees)")
    p.add_argument("--rotate-z", type=float, default=0.0, help="Rotate asset around Z (degrees)")

    # mp4 recording (front-facing camera).
    p.add_argument("--record-mp4", default=None,
                   help="Output mp4 path. Uses ViewerGL (headless unless --gui) "
                        "and pipes frames to ffmpeg with a front-facing camera.")
    p.add_argument("--mp4-fps", type=int, default=60,
                   help="Output mp4 framerate (default 60)")
    p.add_argument("--top-view", action="store_true",
                   help="Place camera straight above the asset looking down")
    p.add_argument("--no-auto-camera", action="store_true",
                   help="Skip the front-view auto-framer so the viewer keeps its default "
                        "camera (useful when rotating the asset and you want the view "
                        "to stay constant).")
    args = p.parse_args(argv)

    # Apply physics-json overrides for cloth and rod tuning.
    if args.physics_json:
        import json
        with open(args.physics_json) as _f:
            _phys = json.load(_f)
        _r = _phys.get("solver", {}).get("vbd_particle_self_contact_radius")
        if _r is not None:
            args.cloth_particle_radius = float(_r)
        _b = _phys.get("cloth", {}).get("bend_stiffness")
        if _b is not None:
            args.bending_ke = float(_b)
        _s = _phys.get("sim_substeps")
        if _s is None:
            _s = _phys.get("solver", {}).get("sim_substeps")
        _rod = None
        _scene = None
        if isinstance(_phys.get("newton"), dict):
            _rod = _phys["newton"].get("rod")
            _scene = _phys["newton"].get("simulation_scene")
        if _rod is None:
            for _part in _phys.get("parts", []) or []:
                _newton = _part.get("newton", {}) if isinstance(_part, dict) else {}
                if isinstance(_newton.get("rod"), dict):
                    _rod = _newton["rod"]
                    _scene = _newton.get("simulation_scene")
                    break
        if _s is None and isinstance(_scene, dict):
            _s = _scene.get("sim_substeps")
        if _s is not None and args.substeps is None:
            args.substeps = int(_s)
        if (
            args.solver_iterations is None
            and isinstance(_scene, dict)
            and _scene.get("max_solver_iterations") is not None
        ):
            args.solver_iterations = int(_scene["max_solver_iterations"])
        if isinstance(_rod, dict):
            if _rod.get("stretch_stiffness") is not None:
                args.rod_stretch_stiffness = float(_rod["stretch_stiffness"])
            if _rod.get("stretch_damping") is not None:
                args.rod_stretch_damping = float(_rod["stretch_damping"])
            if _rod.get("bend_stiffness") is not None:
                args.rod_bend_stiffness = float(_rod["bend_stiffness"])
            if _rod.get("bend_damping") is not None:
                args.rod_bend_damping = float(_rod["bend_damping"])
        print(
            f"  physics-json: {args.physics_json}  "
            f"cloth_particle_radius={args.cloth_particle_radius}  "
            f"bending_ke={args.bending_ke}  "
            f"substeps={args.substeps}  "
            f"solver_iterations={args.solver_iterations}  "
            f"rod_bend_stiffness={args.rod_bend_stiffness}  "
            f"rod_bend_damping={args.rod_bend_damping}"
        )

    def _parse_joint_kv(s: str | None) -> dict[int, float]:
        if not s:
            return {}
        out: dict[int, float] = {}
        for tok in s.split(","):
            tok = tok.strip()
            if not tok:
                continue
            k, _, v = tok.partition("=")
            out[int(k)] = float(v)
        return out

    from newton import viewer as v
    if args.record_mp4:
        viewer = v.ViewerGL(headless=not args.gui)
    elif args.gui:
        viewer = v.ViewerGL(headless=False)
    else:
        viewer = v.ViewerNull()

    ex = Example(viewer, args.usd, substeps=args.substeps,
                 drop_height=args.drop_height, device=args.device,
                 solver_iterations=args.solver_iterations,
                 rod_twist_rate=args.rod_twist_rate,
                 rod_twist_end_time=args.rod_twist_end_time,
                 rod_stretch_stiffness=args.rod_stretch_stiffness,
                 rod_stretch_damping=args.rod_stretch_damping,
                 rod_bend_stiffness=args.rod_bend_stiffness,
                 rod_bend_damping=args.rod_bend_damping,
                 zero_gravity=args.zero_gravity,
                 gravity_scale=args.gravity_scale,
                 cloth_particle_radius=args.cloth_particle_radius,
                 soft_contact_ke=args.soft_contact_ke,
                 soft_contact_kd=args.soft_contact_kd,
                 soft_contact_mu=args.soft_contact_mu,
                 soft_contact_max=args.soft_contact_max,
                 cloth_body_contact_margin=args.cloth_body_contact_margin,
                 bending_ke=args.bending_ke,
                 bending_kd=args.bending_kd,
                 vbd_particle_edge_contact_buffer_size=args.vbd_particle_edge_contact_buffer_size,
                 vbd_particle_collision_detection_interval=args.vbd_particle_collision_detection_interval,
                 vbd_rigid_contact_k_start=args.vbd_rigid_contact_k_start,
                 joint_q_overrides=_parse_joint_kv(args.joint_q),
                 joint_target_overrides=_parse_joint_kv(args.joint_target),
                 joint_target_ke=args.joint_target_ke,
                 joint_target_kd=args.joint_target_kd,
                 list_joints=args.list_joints,
                 rotate_x_deg=args.rotate_x,
                 rotate_y_deg=args.rotate_y,
                 rotate_z_deg=args.rotate_z)

    usd_bounds = _read_usd_geometry_bounds(args.usd)

    def _camera_bounds():
        import numpy as _np

        if usd_bounds is not None:
            return usd_bounds
        n_p = int(ex.model.particle_count)
        if n_p > 0:
            pts = ex.state_0.particle_q.numpy()
        else:
            pts = ex.state_0.body_q.numpy()[:, 0:3]
        return _np.asarray(pts.min(axis=0), dtype=float), _np.asarray(pts.max(axis=0), dtype=float)

    # ---- Top-down camera (ViewerGL only) ----
    if args.top_view and hasattr(ex.viewer, "set_camera"):
        bounds_min, bounds_max = _camera_bounds()
        cx = float(0.5 * (bounds_min[0] + bounds_max[0]))
        cy = float(0.5 * (bounds_min[1] + bounds_max[1]))
        zmax = float(bounds_max[2])
        ext_x = float(bounds_max[0] - bounds_min[0])
        ext_y = float(bounds_max[1] - bounds_min[1])
        height = zmax + max(ext_x, ext_y, 1.0) * 1.5
        # pitch -90 looks straight down, yaw 0 keeps +Y up in image.
        ex.viewer.set_camera(wp.vec3(cx, cy, height), -90.0, 0.0)
        print(f"  top-view camera: pos=({cx:.3f},{cy:.3f},{height:.3f})")

    # ---- Front-facing auto-camera for GUI / mp4 (ViewerGL only) ----
    elif (args.record_mp4 or args.gui) and not args.no_auto_camera and hasattr(ex.viewer, "set_camera"):
        bounds_min, bounds_max = _camera_bounds()
        cx = float(0.5 * (bounds_min[0] + bounds_max[0]))
        cy = float(0.5 * (bounds_min[1] + bounds_max[1]))
        cz_mid = float(0.5 * (bounds_min[2] + bounds_max[2]))
        ext_x = float(bounds_max[0] - bounds_min[0])
        ext_y = float(bounds_max[1] - bounds_min[1])
        ext_z = float(bounds_max[2] - bounds_min[2])
        dist = max(ext_x, ext_y, ext_z, 1.0) * 2.2
        # Z-up: camera in -Y, looking toward +Y → yaw=90, pitch=0.
        ex.viewer.set_camera(wp.vec3(cx, cy - dist, cz_mid), 0.0, 90.0)
        print(f"  front-view camera: pos=({cx:.3f},{cy - dist:.3f},{cz_mid:.3f})")

    # ---- mp4 recorder (ffmpeg subprocess) ----
    ffmpeg_proc = None
    if args.record_mp4:
        import subprocess, shutil
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not on PATH; cannot record mp4")
        ex.render()
        frame = ex.viewer.get_frame()
        h, w, _ = frame.shape
        # libx264 requires even dimensions; crop to nearest even size.
        enc_w = w - (w % 2)
        enc_h = h - (h % 2)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{enc_w}x{enc_h}", "-r", str(args.mp4_fps),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "20", "-preset", "fast",
            args.record_mp4,
        ]
        ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        ffmpeg_proc.stdin.write(frame.numpy()[:enc_h, :enc_w, :].tobytes())
        print(f"  recording mp4: {args.record_mp4}  size={enc_w}x{enc_h}  fps={args.mp4_fps}")

    # Decimation so 1s sim == 1s video: write an mp4 frame every physics_per_video
    # physics steps. --steps still counts physics steps (unchanged semantics).
    # With sim_fps=240 and --mp4-fps=60 → 1 mp4 frame per 4 sim steps.
    physics_per_video = (
        max(1, int(round(float(ex.fps) / float(args.mp4_fps))))
        if ffmpeg_proc is not None
        else 1
    )
    if ffmpeg_proc is not None:
        print(f"  physics/video decimation: 1 mp4 frame per {physics_per_video} sim step(s) "
              f"(sim_fps={ex.fps}, mp4_fps={args.mp4_fps})")

    def _viewer_running() -> bool:
        is_running = getattr(viewer, "is_running", True)
        if callable(is_running):
            try:
                return bool(is_running())
            except Exception:
                return True
        return bool(is_running)

    i = 1 if ffmpeg_proc is not None else 0
    use_step_count = bool(args.record_mp4) or not args.gui
    try:
        while True:
            if args.gui and not _viewer_running():
                break
            if use_step_count:
                if i >= args.steps:
                    break
            else:
                if not _viewer_running():
                    break
            ex.step()
            ex.render()
            if ffmpeg_proc is not None and i % physics_per_video == 0:
                ffmpeg_proc.stdin.write(ex.viewer.get_frame().numpy()[:enc_h, :enc_w, :].tobytes())
            i += 1
    finally:
        if ffmpeg_proc is not None:
            try:
                ffmpeg_proc.stdin.close()
                ffmpeg_proc.wait(timeout=30)
            except Exception:
                ffmpeg_proc.kill()
        try:
            viewer.close()
        except Exception:
            pass

    print(f"[done] frames={i}  sim_time={ex.sim_time:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
