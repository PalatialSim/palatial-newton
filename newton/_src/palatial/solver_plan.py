# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Build Newton solvers from Palatial soft-body solver plans."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM, SolverCoupledProxy

from .load import _build_solver


@dataclass(frozen=True)
class SolverPlanPartEntities:
    """Newton entity indices belonging to one Palatial part.

    Args:
        bodies: Global body indices owned by the part.
        particles: Global particle indices owned by the part.
        joints: Global joint indices owned by the part.
        shapes: Global shape indices owned by the part.
    """

    bodies: Sequence[int] = ()
    particles: Sequence[int] = ()
    joints: Sequence[int] = ()
    shapes: Sequence[int] = ()


_GLOBAL_PARAMETER_KEYS = {
    "gravitational_acceleration",
    "gravity_enabled",
    "max_solver_iterations",
    "sim_substeps",
    "simulation_steps_per_second",
}

_PARAMETER_ALIASES = {
    "particleConservativeBoundRelaxation": "particle_conservative_bound_relaxation",
    "particleEnableSelfContact": "particle_enable_self_contact",
    "particleRestShapeContactExclusionRadius": "particle_rest_shape_contact_exclusion_radius",
    "particleSelfContactMargin": "particle_self_contact_margin",
    "particleSelfContactRadius": "particle_self_contact_radius",
    "particleTopologicalContactFilterThreshold": "particle_topological_contact_filter_threshold",
    "particleVertexContactBufferSize": "particle_vertex_contact_buffer_size",
    "particle_self_contact_enabled": "particle_enable_self_contact",
    "rigid_contact_constraint_weighting": "rigid_contact_con_weighting",
}


def _solver_plan(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    if "soft_body_spec" in spec:
        soft_body_spec = spec["soft_body_spec"]
        if not isinstance(soft_body_spec, Mapping):
            raise TypeError("soft_body_spec must be an object")
        spec = soft_body_spec
    if "solver_plan" in spec:
        plan = spec["solver_plan"]
        if not isinstance(plan, Mapping):
            raise TypeError("solver_plan must be an object")
        return plan
    if "mode" in spec and "assignments" in spec and "couplings" in spec:
        return spec
    raise ValueError("expected a physics result, soft_body_spec, or solver_plan")


def _entity_indices(
    part_ids: Sequence[str],
    part_entities: Mapping[str, SolverPlanPartEntities],
) -> SolverPlanPartEntities:
    fields = ("bodies", "particles", "joints", "shapes")
    merged: dict[str, list[int]] = {field: [] for field in fields}
    for part_id in part_ids:
        if part_id not in part_entities:
            raise ValueError(f"solver plan references unknown part {part_id!r}")
        entities = part_entities[part_id]
        for field in fields:
            merged[field].extend(int(index) for index in getattr(entities, field))
    return SolverPlanPartEntities(**{field: tuple(dict.fromkeys(merged[field])) for field in fields})


def _solver_parameters(solver_name: str, parameters: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    prefix = f"{solver_name}_"
    normalized: dict[str, Any] = {}
    for key, value in parameters.items():
        if key in _GLOBAL_PARAMETER_KEYS:
            continue
        normalized_key = key[len(prefix) :] if key.startswith(prefix) else key
        if normalized_key == "substeps":
            continue
        normalized[_PARAMETER_ALIASES.get(normalized_key, normalized_key)] = value
    if "iterations" not in normalized and "max_solver_iterations" in parameters:
        normalized["iterations"] = parameters["max_solver_iterations"]
    substeps = int(parameters.get(f"{solver_name}_substeps", parameters.get("sim_substeps", 1)))
    if substeps < 1:
        raise ValueError(f"sim_substeps must be >= 1, got {substeps}")
    return normalized, substeps


def _entries(
    assignments: Sequence[Mapping[str, Any]],
    part_entities: Mapping[str, SolverPlanPartEntities],
) -> tuple[list[SolverCoupled.Entry], dict[str, SolverPlanPartEntities]]:
    entries = []
    entities_by_assignment = {}
    for assignment in assignments:
        name = str(assignment["id"])
        solver_name = str(assignment["solver"])
        entities = _entity_indices(assignment["part_ids"], part_entities)
        parameters, substeps = _solver_parameters(solver_name, assignment.get("parameters", {}))
        entries.append(
            SolverCoupled.Entry(
                name=name,
                solver=lambda view, name=solver_name, params=parameters: _build_solver(name, view, params),
                bodies=entities.bodies,
                particles=entities.particles,
                joints=entities.joints,
                shapes=entities.shapes,
                substeps=substeps,
            )
        )
        entities_by_assignment[name] = entities
    return entries, entities_by_assignment


def _merged_coupling_parameters(couplings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for coupling in couplings:
        for key, value in coupling.get("parameters", {}).items():
            if key in merged and merged[key] != value:
                raise ValueError(f"coupling parameter {key!r} has conflicting values")
            merged[key] = value
    return merged


def _proxy_solver(
    model: Any,
    entries: Sequence[SolverCoupled.Entry],
    couplings: Sequence[Mapping[str, Any]],
    entities_by_assignment: Mapping[str, SolverPlanPartEntities],
    collision_pipeline_factory: Callable[[Any], Any] | None,
) -> SolverCoupledProxy:
    proxies = []
    iteration_values = {
        int(parameters.get("iterations", parameters.get("proxy_iterations", 1)))
        for coupling in couplings
        for parameters in [coupling.get("parameters", {})]
    }
    if len(iteration_values) != 1:
        raise ValueError("proxy coupling iteration counts must match")
    for coupling in couplings:
        source = str(coupling["from"])
        destination = str(coupling["to"])
        entities = entities_by_assignment[source]
        local = dict(coupling.get("parameters", {}))
        proxies.append(
            SolverCoupledProxy.Proxy(
                source=source,
                destination=destination,
                bodies=entities.bodies,
                particles=entities.particles,
                joints=entities.joints if bool(local.get("proxy_joints_enabled", False)) else (),
                mass_scale=float(local.get("mass_scale", 1.0)),
                mode=str(local.get("mode", "lagged")),
                proxy_relaxation=float(local.get("proxy_relaxation", 1.0)),
                proxy_relaxation_mode=str(local.get("proxy_relaxation_mode", "fixed")),
                proxy_relaxation_min=float(local.get("proxy_relaxation_min", 0.1)),
                proxy_relaxation_max=float(local.get("proxy_relaxation_max", 1.0)),
                collision_pipeline=collision_pipeline_factory,
                collide_interval=local.get("collide_interval", 1 if collision_pipeline_factory else None),
            )
        )
    return SolverCoupledProxy(
        model=model,
        entries=entries,
        coupling=SolverCoupledProxy.Config(
            proxies=proxies,
            iterations=iteration_values.pop(),
        ),
    )


def _admm_solver(
    model: Any,
    entries: Sequence[SolverCoupled.Entry],
    couplings: Sequence[Mapping[str, Any]],
) -> SolverCoupledADMM:
    parameters = _merged_coupling_parameters(couplings)
    contact_pairs = [
        SolverCoupledADMM.ContactPair(source=str(coupling["from"]), destination=str(coupling["to"]))
        for coupling in couplings
        if bool(coupling.get("parameters", {}).get("contacts_enabled", False))
    ]
    return SolverCoupledADMM(
        model=model,
        entries=entries,
        coupling=SolverCoupledADMM.Config(
            iterations=int(parameters.get("iterations", parameters.get("admm_iterations", 5))),
            rho=float(parameters.get("rho", 1.0)),
            gamma=float(parameters.get("gamma", 0.0)),
            baumgarte=float(parameters.get("baumgarte", 0.0)),
            joint_stiffness=float(parameters.get("joint_stiffness", 1.0e4)),
            joint_damping=float(parameters.get("joint_damping", 0.0)),
            joint_angular_stiffness=float(parameters.get("joint_angular_stiffness", 1.0e4)),
            joint_angular_damping=float(parameters.get("joint_angular_damping", 0.0)),
            joint_proximal_bodies=bool(parameters.get("joint_proximal_bodies", True)),
            joint_proximal_mass_scale=float(parameters.get("joint_proximal_mass_scale", 1.0)),
            contact_pairs=contact_pairs,
        ),
    )


def solver_from_plan(
    model: Any,
    spec: Mapping[str, Any],
    part_entities: Mapping[str, SolverPlanPartEntities],
    *,
    collision_pipeline_factory: Callable[[Any], Any] | None = None,
) -> Any:
    """Build a single or coupled solver from a Palatial soft-body solver plan.

    ``part_entities`` is the converter-owned bridge from stable Palatial part
    identifiers to indices in the finalized Newton model. For proxy coupling,
    ``from`` is the solver whose entities are mirrored into the ``to`` solver.
    ADMM plans derive attachments and cross-solver joints from model metadata;
    set ``contacts_enabled`` on a coupling to also create an ADMM contact pair.

    Args:
        model: Finalized Newton model containing all mixed-body entities.
        spec: Full physics result, ``soft_body_spec``, or ``solver_plan`` object.
        part_entities: Part identifier to Newton entity-index mapping.
        collision_pipeline_factory: Optional destination-view collision
            pipeline factory for proxy contact refreshes.

    Returns:
        Constructed Newton solver.
    """

    plan = _solver_plan(spec)
    mode = str(plan["mode"])
    assignments = list(plan["assignments"])
    couplings = list(plan["couplings"])
    assignment_ids = [str(assignment["id"]) for assignment in assignments]
    if len(assignment_ids) != len(set(assignment_ids)):
        raise ValueError("solver assignment ids must be unique")

    planned_parts = [str(part_id) for assignment in assignments for part_id in assignment["part_ids"]]
    if len(planned_parts) != len(set(planned_parts)):
        raise ValueError("each part must have exactly one solver assignment")
    if set(planned_parts) != set(part_entities):
        raise ValueError("solver plan and part entity mapping must contain the same part ids")

    if mode == "single":
        if len(assignments) != 1 or couplings:
            raise ValueError("single solver plans require one assignment and no couplings")
        assignment = assignments[0]
        parameters, _ = _solver_parameters(str(assignment["solver"]), assignment.get("parameters", {}))
        return _build_solver(str(assignment["solver"]), model, parameters)
    if mode != "coupled":
        raise ValueError(f"unknown solver plan mode {mode!r}")
    if len(assignments) < 2 or not couplings:
        raise ValueError("coupled solver plans require at least two assignments and one coupling")

    known_assignments = set(assignment_ids)
    methods = set()
    for coupling in couplings:
        source = str(coupling["from"])
        destination = str(coupling["to"])
        if source not in known_assignments or destination not in known_assignments:
            raise ValueError("coupling endpoints must reference solver assignment ids")
        if source == destination:
            raise ValueError("coupling endpoints must differ")
        methods.add(str(coupling["method"]))
    if len(methods) != 1:
        raise ValueError("one coupled solver plan cannot mix proxy and ADMM methods")

    entries, entities_by_assignment = _entries(assignments, part_entities)
    method = methods.pop()
    if method == "proxy":
        return _proxy_solver(
            model,
            entries,
            couplings,
            entities_by_assignment,
            collision_pipeline_factory,
        )
    if method == "admm":
        return _admm_solver(model, entries, couplings)
    raise ValueError(f"unsupported coupling method {method!r}")


__all__ = ["SolverPlanPartEntities", "solver_from_plan"]
