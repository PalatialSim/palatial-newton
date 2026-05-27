# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton ModelBuilder adapter for canonical Power cable assemblies."""

from __future__ import annotations

import newton
import numpy as np
import warp as wp

from .constants import (
    DEFAULT_ASSEMBLY_ROD_SEGMENT_COUNT,
    DEFAULT_CONNECTOR_MASS,
    DEFAULT_CONTACT_KD,
    DEFAULT_CONTACT_KE,
    DEFAULT_CONTACT_KF,
    DEFAULT_CONTACT_MU,
    DEFAULT_ROD_BEND_STIFFNESS,
    DEFAULT_ROD_DAMPING,
    DEFAULT_ROD_DENSITY,
    DEFAULT_ROD_STRETCH_STIFFNESS,
    DEFAULT_ROD_TORSION_STIFFNESS,
)
from .types import CanonicalConnectorBody, CanonicalPowerCableAssembly, PowerCableAssemblyBuildResult

_IDENTITY_QUAT = wp.quat(0.0, 0.0, 0.0, 1.0)


def build_power_cable_assembly(
    builder: newton.ModelBuilder,
    assembly: CanonicalPowerCableAssembly,
) -> PowerCableAssemblyBuildResult:
    """Populate ``builder`` with a canonical Power cable assembly."""

    from newton import utils as newton_utils

    rod_cfg = builder.default_shape_cfg.copy()
    rod_cfg.density = DEFAULT_ROD_DENSITY
    _apply_contact_config(rod_cfg)

    rod_positions = _interpolate_centerline(
        assembly.cable.centerline_points,
        DEFAULT_ASSEMBLY_ROD_SEGMENT_COUNT,
    )
    rod_quaternions = newton_utils.create_parallel_transport_cable_quaternions(rod_positions)
    rod_body_ids_raw, rod_joint_ids_raw = builder.add_rod(
        positions=rod_positions,
        quaternions=rod_quaternions,
        radius=assembly.cable.radius,
        cfg=rod_cfg,
        stretch_stiffness=DEFAULT_ROD_STRETCH_STIFFNESS,
        stretch_damping=DEFAULT_ROD_DAMPING,
        bend_y_stiffness=DEFAULT_ROD_BEND_STIFFNESS,
        bend_y_damping=DEFAULT_ROD_DAMPING,
        bend_z_stiffness=DEFAULT_ROD_BEND_STIFFNESS,
        bend_z_damping=DEFAULT_ROD_DAMPING,
        torsion_stiffness=DEFAULT_ROD_TORSION_STIFFNESS,
        torsion_damping=DEFAULT_ROD_DAMPING,
        label="power_cable_rod",
    )
    rod_body_ids = tuple(int(body_id) for body_id in rod_body_ids_raw)
    rod_joint_ids = tuple(int(joint_id) for joint_id in rod_joint_ids_raw)
    if not rod_body_ids:
        raise RuntimeError("Power cable assembly builder created no rod bodies")

    endpoint_to_rod_body = {
        assembly.cable.connector_ids_by_endpoint[0]: rod_body_ids[0],
        assembly.cable.connector_ids_by_endpoint[1]: rod_body_ids[-1],
    }
    endpoint_to_rod_local_position = {
        assembly.cable.connector_ids_by_endpoint[0]: wp.vec3(0.0, 0.0, 0.0),
        assembly.cable.connector_ids_by_endpoint[1]: wp.vec3(0.0, 0.0, _segment_length(rod_positions)),
    }

    connector_body_ids: list[int] = []
    connector_shape_ids: list[int] = []
    attachment_joint_ids: list[int] = []
    connector_cfg = builder.default_shape_cfg.copy()
    _apply_contact_config(connector_cfg)
    for connector in assembly.connector_bodies:
        body_id = int(
            builder.add_body(
                xform=wp.transform(connector.anchor_world_position, _IDENTITY_QUAT),
                mass=DEFAULT_CONNECTOR_MASS,
                label=f"power_{connector.endpoint_id}_connector_body",
                lock_inertia=True,
            )
        )
        connector_body_ids.append(body_id)
        shape_id = _add_connector_mesh_shape(
            builder,
            body_id=body_id,
            connector=connector,
            cfg=connector_cfg,
        )
        connector_shape_ids.append(shape_id)
        joint_id = int(
            builder.add_joint_fixed(
                parent=endpoint_to_rod_body[connector.endpoint_id],
                child=body_id,
                parent_xform=wp.transform(endpoint_to_rod_local_position[connector.endpoint_id], _IDENTITY_QUAT),
                child_xform=wp.transform(connector.attachment_site.local_position, _IDENTITY_QUAT),
                label=f"power_{connector.endpoint_id}_attachment_fixed_joint",
            )
        )
        attachment_joint_ids.append(joint_id)

    return PowerCableAssemblyBuildResult(
        rod_body_ids=rod_body_ids,
        rod_joint_ids=rod_joint_ids,
        connector_body_ids=(connector_body_ids[0], connector_body_ids[1]),
        connector_shape_ids=(connector_shape_ids[0], connector_shape_ids[1]),
        attachment_joint_ids=(attachment_joint_ids[0], attachment_joint_ids[1]),
    )


def _add_connector_mesh_shape(
    builder: newton.ModelBuilder,
    *,
    body_id: int,
    connector: CanonicalConnectorBody,
    cfg: newton.ModelBuilder.ShapeConfig,
) -> int:
    vertices = np.asarray(
        [(float(point[0]), float(point[1]), float(point[2])) for point in connector.mesh.local_points],
        dtype=np.float32,
    )
    indices = np.asarray(connector.mesh.triangle_vertex_indices, dtype=np.int32)
    mesh = newton.Mesh(vertices=vertices, indices=indices, compute_inertia=False)
    return int(
        builder.add_shape_mesh(
            body=body_id,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), _IDENTITY_QUAT),
            mesh=mesh,
            cfg=cfg,
            label=f"power_{connector.endpoint_id}_connector_mesh_shape",
        )
    )


def _apply_contact_config(shape_cfg: object) -> None:
    setattr(shape_cfg, "ke", DEFAULT_CONTACT_KE)
    setattr(shape_cfg, "kd", DEFAULT_CONTACT_KD)
    setattr(shape_cfg, "kf", DEFAULT_CONTACT_KF)
    setattr(shape_cfg, "mu", DEFAULT_CONTACT_MU)


def _interpolate_centerline(points: tuple[wp.vec3, wp.vec3], segment_count: int) -> list[wp.vec3]:
    start, end = points
    if segment_count < 2:
        raise ValueError("segment_count must be >= 2")
    return [
        start + (end - start) * (float(index) / float(segment_count))
        for index in range(segment_count + 1)
    ]


def _segment_length(points: list[wp.vec3]) -> float:
    if len(points) < 2:
        raise ValueError("at least two points are required")
    return float(wp.length(points[1] - points[0]))
