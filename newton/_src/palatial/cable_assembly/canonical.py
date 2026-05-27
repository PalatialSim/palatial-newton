# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Canonical Power cable assembly construction."""

from __future__ import annotations

import warp as wp

from .types import (
    CanonicalCable,
    CanonicalConnectorBody,
    CanonicalPowerCableAssembly,
    ConnectorAttachmentSite,
    ConnectorEndpointExtraction,
    ConnectorMergedMesh,
    EndpointId,
    PowerCableAssemblyExtraction,
)
from .utils import as_vec3, distance_squared


def build_power_canonical_assembly(
    extraction: PowerCableAssemblyExtraction,
) -> CanonicalPowerCableAssembly:
    """Build the canonical simulation-facing Power assembly."""

    connector_ids_by_endpoint = _assign_cable_endpoints(extraction)
    cable = CanonicalCable(
        centerline_points=extraction.cable.centerline_points,
        radius=extraction.cable.radius,
        length=extraction.cable.length,
        connector_ids_by_endpoint=connector_ids_by_endpoint,
    )
    connector_bodies = tuple(
        _build_connector_body(
            endpoint,
            cable_attachment_world_point=extraction.cable.centerline_points[
                connector_ids_by_endpoint.index(endpoint.endpoint_id)
            ],
        )
        for endpoint in extraction.endpoints
    )
    return CanonicalPowerCableAssembly(
        cable=cable,
        connector_bodies=(connector_bodies[0], connector_bodies[1]),
    )


def _assign_cable_endpoints(extraction: PowerCableAssemblyExtraction) -> tuple[EndpointId, EndpointId]:
    cable_points = extraction.cable.centerline_points
    endpoints = extraction.endpoints
    permutations: tuple[tuple[int, int], ...] = ((0, 1), (1, 0))

    def assignment_cost(assignment: tuple[int, int]) -> float:
        return sum(
            distance_squared(endpoints[index].anchor_point, cable_points[cable_point_index])
            for index, cable_point_index in enumerate(assignment)
        )

    best_assignment = min(permutations, key=lambda assignment: (assignment_cost(assignment), assignment))
    connector_ids: list[EndpointId | None] = [None, None]
    for endpoint_index, cable_point_index in enumerate(best_assignment):
        connector_ids[cable_point_index] = endpoints[endpoint_index].endpoint_id
    if connector_ids[0] is None or connector_ids[1] is None:
        raise RuntimeError("Failed to assign both Power cable endpoints")
    return (connector_ids[0], connector_ids[1])


def _build_connector_body(
    endpoint: ConnectorEndpointExtraction,
    *,
    cable_attachment_world_point: wp.vec3,
) -> CanonicalConnectorBody:
    mesh = _build_connector_merged_mesh(endpoint, anchor_point=endpoint.anchor_point)
    site = ConnectorAttachmentSite(
        local_position=cable_attachment_world_point - endpoint.anchor_point,
        world_position=cable_attachment_world_point,
    )
    return CanonicalConnectorBody(
        endpoint_id=endpoint.endpoint_id,
        anchor_world_position=endpoint.anchor_point,
        mesh=mesh,
        attachment_site=site,
        source_prim_names=endpoint.prim_names,
    )


def _build_connector_merged_mesh(
    endpoint: ConnectorEndpointExtraction,
    *,
    anchor_point: wp.vec3,
) -> ConnectorMergedMesh:
    local_points: list[wp.vec3] = []
    triangle_vertex_indices: list[int] = []
    point_offset = 0
    for prim in endpoint.prims:
        local_points.extend(as_vec3(point) - anchor_point for point in prim.world_points)
        triangle_vertex_indices.extend(point_offset + index for index in prim.triangle_vertex_indices)
        point_offset += len(prim.world_points)

    return ConnectorMergedMesh(
        source_prim_names=endpoint.prim_names,
        local_points=tuple(local_points),
        triangle_vertex_indices=tuple(triangle_vertex_indices),
    )
