# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Data records used by the Power cable assembly pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import warp as wp

Point3: TypeAlias = tuple[float, float, float]
EndpointId = Literal["iec", "nema"]


@dataclass(frozen=True)
class ExtractedPrim:
    """World-space mesh record for one assembly source prim."""

    prim_name: str
    prim_path: str
    mesh_path: str
    world_bounds_min: Point3
    world_bounds_max: Point3
    world_centroid: Point3
    world_points: tuple[Point3, ...]
    triangle_vertex_indices: tuple[int, ...]


@dataclass(frozen=True)
class CableExtraction:
    """Straight cable approximation extracted from the source cable mesh."""

    centerline_points: tuple[wp.vec3, wp.vec3]
    radius: float
    length: float


@dataclass(frozen=True)
class ConnectorEndpointExtraction:
    """Grouped connector source prims for one cable endpoint."""

    endpoint_id: EndpointId
    prim_names: tuple[str, ...]
    prims: tuple[ExtractedPrim, ...]
    world_bounds_min: Point3
    world_bounds_max: Point3
    anchor_point: wp.vec3


@dataclass(frozen=True)
class PowerCableAssemblyExtraction:
    """Normalized extraction record for the v1 Power cable assembly."""

    source_path: Path
    cable: CableExtraction
    endpoints: tuple[ConnectorEndpointExtraction, ConnectorEndpointExtraction]


@dataclass(frozen=True)
class ConnectorAttachmentSite:
    """Connector-local attachment site for the cable endpoint."""

    local_position: wp.vec3
    world_position: wp.vec3


@dataclass(frozen=True)
class ConnectorMergedMesh:
    """Merged connector mesh in connector-body local space."""

    source_prim_names: tuple[str, ...]
    local_points: tuple[wp.vec3, ...]
    triangle_vertex_indices: tuple[int, ...]


@dataclass(frozen=True)
class CanonicalConnectorBody:
    """Canonical connector body consumed by the Newton builder adapter."""

    endpoint_id: EndpointId
    anchor_world_position: wp.vec3
    mesh: ConnectorMergedMesh
    attachment_site: ConnectorAttachmentSite
    source_prim_names: tuple[str, ...]


@dataclass(frozen=True)
class CanonicalCable:
    """Canonical straight cable linked to connector endpoint ids."""

    centerline_points: tuple[wp.vec3, wp.vec3]
    radius: float
    length: float
    connector_ids_by_endpoint: tuple[EndpointId, EndpointId]


@dataclass(frozen=True)
class CanonicalPowerCableAssembly:
    """Canonical v1 Power cable assembly."""

    cable: CanonicalCable
    connector_bodies: tuple[CanonicalConnectorBody, CanonicalConnectorBody]


@dataclass(frozen=True)
class PowerCableAssemblyBuildResult:
    """Inspectable ids created while populating the Newton builder."""

    rod_body_ids: tuple[int, ...]
    rod_joint_ids: tuple[int, ...]
    connector_body_ids: tuple[int, int]
    connector_shape_ids: tuple[int, int]
    attachment_joint_ids: tuple[int, int]
