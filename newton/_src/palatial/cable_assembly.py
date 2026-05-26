# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Power cable assembly loader helpers for Palatial USD assets."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal, TypeAlias

import newton
import numpy as np
import warp as wp
from pxr import Gf, Usd, UsdGeom


Point3: TypeAlias = tuple[float, float, float]
EndpointId = Literal["iec", "nema"]

POWER_CABLE_REQUIRED_PRIM_NAMES: tuple[str, ...] = ("Power_Cable_Body",)
POWER_IEC_REQUIRED_PRIM_NAMES: tuple[str, ...] = (
    "Pow_IEC_Strain",
    "Pow_IEC_Body",
    "Pow_IEC_Recess",
    "Pow_IEC_Slot0",
    "Pow_IEC_Slot1",
    "Pow_IEC_ESlot",
)
POWER_NEMA_REQUIRED_PRIM_NAMES: tuple[str, ...] = (
    "Pow_NEMA_Strain",
    "Pow_NEMA_Body",
    "Pow_NEMA_Face",
    "Pow_NEMA_Hot",
    "Pow_NEMA_Neut",
    "Pow_NEMA_Gnd",
    "Pow_NEMA_FS0",
    "Pow_NEMA_FS1",
    "Pow_NEMA_GS",
)
POWER_REQUIRED_PRIM_NAMES: tuple[str, ...] = (
    *POWER_CABLE_REQUIRED_PRIM_NAMES,
    *POWER_IEC_REQUIRED_PRIM_NAMES,
    *POWER_NEMA_REQUIRED_PRIM_NAMES,
)

DEFAULT_ASSEMBLY_FPS = 60
DEFAULT_ASSEMBLY_SUBSTEPS = 10
DEFAULT_ASSEMBLY_ITERATIONS = 2
DEFAULT_ASSEMBLY_ROD_SEGMENT_COUNT = 60
DEFAULT_CONNECTOR_MASS = 1.0e-1
DEFAULT_ROD_DENSITY = 1000.0
DEFAULT_ROD_STRETCH_STIFFNESS = 1.0e6
DEFAULT_ROD_BEND_STIFFNESS = 1.0e2
DEFAULT_ROD_TORSION_STIFFNESS = 1.0e2
DEFAULT_ROD_DAMPING = 1.0e-1
DEFAULT_CONTACT_KE = 1.0e4
DEFAULT_CONTACT_KD = 1.0e-1
DEFAULT_CONTACT_KF = 1.0e3
DEFAULT_CONTACT_MU = 1.0

_GEOMETRY_SCOPE_PATH = "/World/Geometry"
_PCA_CABLE_EPS = 1.0e-9
_IDENTITY_QUAT = wp.quat(0.0, 0.0, 0.0, 1.0)


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


def is_power_cable_assembly_stage(stage: Usd.Stage) -> bool:
    """Return whether the stage matches the v1 Power cable assembly shape."""

    geometry = stage.GetPrimAtPath(_GEOMETRY_SCOPE_PATH)
    if not geometry or not geometry.IsValid():
        return False
    child_names = {child.GetName() for child in geometry.GetChildren()}
    return set(POWER_REQUIRED_PRIM_NAMES).issubset(child_names)


def build_power_cable_assembly_model(usd_path: str, *, device: str | None = None) -> newton.Model:
    """Build a Newton model for a v1 Power cable assembly USD."""

    extraction = extract_power_cable_assembly(usd_path)
    assembly = build_power_canonical_assembly(extraction)

    with wp.ScopedDevice(device) if device else wp.ScopedDevice(wp.get_preferred_device()):
        builder = newton.ModelBuilder()
        build_power_cable_assembly(builder, assembly)
        _apply_default_runtime_scene(builder)
        try:
            builder.color()
        except Exception:
            pass
        return builder.finalize()


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


def extract_power_cable_assembly(usd_path: str) -> PowerCableAssemblyExtraction:
    """Extract the v1 Power cable assembly from a USD stage."""

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise RuntimeError(f"Cannot open USD: {usd_path}")
    _validate_stage_contract(stage, usd_path)

    geometry = stage.GetPrimAtPath(_GEOMETRY_SCOPE_PATH)
    if not geometry or not geometry.IsValid():
        raise RuntimeError(f"Power cable assembly requires {_GEOMETRY_SCOPE_PATH}")

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
    world_points = tuple(_transform_point(world_transform, point) for point in points)
    if len(world_points) < 3:
        raise RuntimeError(f"Assembly mesh {mesh_prim.GetPath()} must contain at least three points")

    bounds_min, bounds_max = _bounds(world_points)
    triangle_indices = _triangulate_face_indices(counts, indices)
    if len(triangle_indices) < 3:
        raise RuntimeError(f"Assembly mesh {mesh_prim.GetPath()} must contain at least one triangle")

    return ExtractedPrim(
        prim_name=prim.GetName(),
        prim_path=prim.GetPath().pathString,
        mesh_path=mesh_prim.GetPath().pathString,
        world_bounds_min=bounds_min,
        world_bounds_max=bounds_max,
        world_centroid=_midpoint(bounds_min, bounds_max),
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

    if not all(_point_is_finite(point) for point in centerline_points):
        raise RuntimeError("Power cable centerline points must be finite")
    if not isfinite(radius) or radius <= 0.0:
        raise RuntimeError(f"Power cable radius must be positive and finite, got {radius}")
    if not isfinite(length) or length <= 0.0:
        raise RuntimeError(f"Power cable length must be positive and finite, got {length}")

    return CableExtraction(
        centerline_points=(_as_vec3(centerline_points[0]), _as_vec3(centerline_points[1])),
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
    if not isfinite(axis_norm) or axis_norm <= _PCA_CABLE_EPS:
        raise ValueError("Power cable PCA axis must have positive length")
    axis = axis / axis_norm

    projections = centered @ axis
    t_min = float(np.min(projections))
    t_max = float(np.max(projections))
    length = t_max - t_min
    if not isfinite(length) or length <= _PCA_CABLE_EPS:
        raise ValueError("Power cable PCA length must be positive")

    radial_vectors = centered - np.outer(projections, axis)
    radial_distances = np.linalg.norm(radial_vectors, axis=1)
    radius = float(np.max(radial_distances))
    if not isfinite(radius) or radius <= _PCA_CABLE_EPS:
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
    center = _midpoint(prim.world_bounds_min, prim.world_bounds_max)
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
    bounds_min, bounds_max = _combine_bounds(prims)
    return ConnectorEndpointExtraction(
        endpoint_id=endpoint_id,
        prim_names=prim_names,
        prims=prims,
        world_bounds_min=bounds_min,
        world_bounds_max=bounds_max,
        anchor_point=_as_vec3(_midpoint(bounds_min, bounds_max)),
    )


def _assign_cable_endpoints(extraction: PowerCableAssemblyExtraction) -> tuple[EndpointId, EndpointId]:
    cable_points = extraction.cable.centerline_points
    endpoints = extraction.endpoints
    permutations: tuple[tuple[int, int], ...] = ((0, 1), (1, 0))

    def assignment_cost(assignment: tuple[int, int]) -> float:
        return sum(
            _distance_squared(endpoints[index].anchor_point, cable_points[cable_point_index])
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
        local_points.extend(_as_vec3(point) - anchor_point for point in prim.world_points)
        triangle_vertex_indices.extend(point_offset + index for index in prim.triangle_vertex_indices)
        point_offset += len(prim.world_points)

    return ConnectorMergedMesh(
        source_prim_names=endpoint.prim_names,
        local_points=tuple(local_points),
        triangle_vertex_indices=tuple(triangle_vertex_indices),
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


def _apply_default_runtime_scene(builder: newton.ModelBuilder) -> None:
    builder.add_ground_plane()
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(1.0, 0.0, 2.0), _IDENTITY_QUAT),
        hx=0.5,
        hy=0.5,
        hz=0.5,
        label="power_reference_box",
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


def _transform_point(matrix: Gf.Matrix4d, point: object) -> Point3:
    transformed = matrix.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
    return (float(transformed[0]), float(transformed[1]), float(transformed[2]))


def _triangulate_face_indices(counts: object, indices: object) -> tuple[int, ...]:
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


def _bounds(points: tuple[Point3, ...]) -> tuple[Point3, Point3]:
    return (
        tuple(min(point[index] for point in points) for index in range(3)),
        tuple(max(point[index] for point in points) for index in range(3)),
    )


def _combine_bounds(prims: tuple[ExtractedPrim, ...]) -> tuple[Point3, Point3]:
    if not prims:
        raise RuntimeError("Cannot combine bounds for an empty prim set")
    return (
        tuple(min(prim.world_bounds_min[index] for prim in prims) for index in range(3)),
        tuple(max(prim.world_bounds_max[index] for prim in prims) for index in range(3)),
    )


def _midpoint(point_a: Point3, point_b: Point3) -> Point3:
    return tuple((point_a[index] + point_b[index]) * 0.5 for index in range(3))


def _as_vec3(point: object) -> wp.vec3:
    return wp.vec3(float(point[0]), float(point[1]), float(point[2]))


def _point_is_finite(point: object) -> bool:
    return all(isfinite(float(point[index])) for index in range(3))


def _distance_squared(point_a: object, point_b: object) -> float:
    dx = float(point_a[0]) - float(point_b[0])
    dy = float(point_a[1]) - float(point_b[1])
    dz = float(point_a[2]) - float(point_b[2])
    return dx * dx + dy * dy + dz * dz


__all__ = [
    "DEFAULT_ASSEMBLY_FPS",
    "DEFAULT_ASSEMBLY_ITERATIONS",
    "DEFAULT_ASSEMBLY_SUBSTEPS",
    "POWER_REQUIRED_PRIM_NAMES",
    "build_power_cable_assembly_model",
    "is_power_cable_assembly_stage",
]
