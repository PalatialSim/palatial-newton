from __future__ import annotations

import newton

from pxr import Usd, UsdGeom

from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
import warp as wp
from .rod_connectors import (
    _normalize_vector,
    attach_rod_connector_component,
    filter_body_self_collisions,
    import_remaining_rigid_content,
    plan_rod_rigid_imports,
)


def _build_solver(name: str, model: Any, params: dict) -> Any:
    """Construct a solver, forwarding only kwargs the solver accepts."""
    import inspect as _ins
    classes = {
        "mujoco":        getattr(newton.solvers, "SolverMuJoCo",       None),
        "xpbd":          getattr(newton.solvers, "SolverXPBD",         None),
        "featherstone":  getattr(newton.solvers, "SolverFeatherstone", None),
        "vbd":           getattr(newton.solvers, "SolverVBD",          None),
        "semi_implicit": getattr(newton.solvers, "SolverSemiImplicit", None),
        "style3d":       getattr(newton.solvers, "SolverStyle3D",      None),
    }
    cls = classes.get(name)
    if cls is None:
        raise RuntimeError(f"Solver '{name}' not available in this Newton build")

    try:
        sig = _ins.signature(cls.__init__)
        accepted = set(sig.parameters.keys())
    except (TypeError, ValueError):
        accepted = set()

    kwargs = {}
    for k, v in params.items():
        py = _SOLVER_PARAM_ALIAS.get(k, k)
        if py in accepted:
            kwargs[py] = v
    return cls(model, **kwargs)


def _untint_textured_shapes(builder: Any) -> None:
    """Force textured imported shapes to white fallback vertex colors."""
    for i, src in enumerate(builder.shape_source):
        if src is not None and getattr(src, "texture", None) is not None:
            builder.shape_color[i] = (1.0, 1.0, 1.0)

@dataclass
class _RodBuildResult:
    """Rod model plus the curved input pose used as the initial state."""

    model: Any
    initial_body_q: Any


@dataclass
class NewtonBundle:
    """Everything needed to step the converted asset in Newton."""
    usd_path: str
    body_type: str
    solver_name: str
    fps: int
    model: Any
    solver: Any
    state_in: Any
    state_out: Any
    control: Any
    solver_params: dict

    @property
    def dt(self) -> float:
        return 1.0 / float(self.fps)


# USD-schema canonical (camelCase) -> solver kwarg (snake_case).
_SOLVER_PARAM_ALIAS = {
    "iterations":                          "iterations",
    "substeps":                            "substeps",
    "particleEnableSelfContact":           "particle_enable_self_contact",
    "particleSelfContactRadius":           "particle_self_contact_radius",
    "particleSelfContactMargin":           "particle_self_contact_margin",
    "particleConservativeBoundRelaxation": "particle_conservative_bound_relaxation",
    "particleRestShapeContactExclusionRadius":   "particle_rest_shape_contact_exclusion_radius",
    "particleTopologicalContactFilterThreshold": "particle_topological_contact_filter_threshold",
    "particleVertexContactBufferSize":           "particle_vertex_contact_buffer_size",
}


def _dedupe_solver_params(params: dict) -> dict:
    """Drop snake_case duplicates when the canonical camelCase form is present."""
    out = dict(params)
    for camel, snake in _SOLVER_PARAM_ALIAS.items():
        if camel == snake:
            continue
        if camel in out and snake in out and out[camel] != out[snake]:
            print(
                f"  warn: solver param conflict — {snake}={out[snake]!r} "
                f"ignored, using canonical {camel}={out[camel]!r}"
            )
            out.pop(snake, None)
        elif camel in out and snake in out:
            out.pop(snake, None)
    return out


def _read_scene_params(stage: Usd.Stage) -> tuple[str, int, dict]:
    """Return ``(solver_name, fps, solver_params)`` from the first PhysicsScene."""
    solver = "mujoco"
    fps = 240
    solver_params: dict = {}
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsScene":
            continue
        a = prim.GetAttribute("newton:solver")
        if not (a and a.HasAuthoredValue()):
            a = prim.GetAttribute("palatial:solver")
        if a and a.HasAuthoredValue():
            solver = str(a.Get())
        a = prim.GetAttribute("newton:timeStepsPerSecond")
        if a and a.HasAuthoredValue():
            fps = int(a.Get())
        for attr in prim.GetAttributes():
            name = attr.GetName()
            for pfx in ("newton:solver:", "palatial:solver:"):
                if name.startswith(pfx) and attr.HasAuthoredValue():
                    solver_params.setdefault(name[len(pfx):], attr.Get())
                    break
        break
    return solver, fps, _dedupe_solver_params(solver_params)


def _detect_body_type(stage: Usd.Stage) -> str:
    """Return ``'cloth'``/``'rod'`` when detected, else ``'rigid'``."""
    rod_found = False
    for prim in stage.Traverse():
        applied = set(prim.GetAppliedSchemas())
        raw = prim.GetMetadata("apiSchemas")
        if raw is not None:
            for items_attr in ("prependedItems", "appendedItems", "explicitItems"):
                for tok in getattr(raw, items_attr, []) or []:
                    applied.add(str(tok))
        if "NewtonShellAPI" in applied or "NewtonClothAPI" in applied:
            return "cloth"
        a = prim.GetAttribute("newton:deformable:simulationIntent")
        if a and a.HasAuthoredValue() and str(a.Get()) in ("cloth", "shell"):
            return "cloth"
        a = prim.GetAttribute("newton:bodyType")
        if a and a.HasAuthoredValue() and str(a.Get()) == "cloth":
            return "cloth"
        if "NewtonRodAPI" in applied:
            rod_found = True
        a = prim.GetAttribute("newton:deformable:simulationIntent")
        if a and a.HasAuthoredValue() and str(a.Get()) == "rod":
            rod_found = True
    return "rod" if rod_found else "rigid"


def _build_rigid(usd_path: str, *, device: str | None = None,
                 fix_base: bool = False,
                 solver_name: str | None = None) -> Any:
    """Build a rigid model via Newton's USD parser."""
    with wp.ScopedDevice(device) if device else wp.ScopedDevice(wp.get_preferred_device()):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        bodies_before = len(builder.body_mass)

        if fix_base:
            # Patch add_joint_free to emit fixed joints so parse_usd's
            # floating-base bodies are anchored to world.
            counter = [0]
            def _free_to_fixed(child, parent_xform=None, child_xform=None,
                               parent=-1, label=None, **kw):
                counter[0] += 1
                k = label or kw.pop("key", None) or f"fixed_base_{counter[0]}"
                jid = builder.add_joint_fixed(
                    parent=parent, child=child,
                    parent_xform=parent_xform,
                    child_xform=child_xform,
                    label=k,
                )
                return jid
            builder.add_joint_free = _free_to_fixed
            builder.add_free_joints_to_floating_bodies = lambda *a, **kw: None

        builder.add_usd(usd_path, skip_mesh_approximation=True)

        _untint_textured_shapes(builder)

        if fix_base:
            new_bodies = range(bodies_before, len(builder.body_mass))
            connected = set(builder.joint_child)
            for b in new_bodies:
                if b in connected:
                    continue
                if builder.body_mass[b] <= 0:
                    continue
                j = builder.add_joint_fixed(parent=-1, child=b,
                                            label=f"fixed_base_orphan_{b}")
                builder.add_articulation([j], label=f"articulation_orphan_{b}")

        if solver_name == "vbd":
            try:
                builder.color()
            except Exception:
                pass

        return builder.finalize()


def _build_cloth(usd_path: str, *, device: str | None = None,
                 solver_name: str | None = None,
                 table: dict | None = None) -> Any:
    """Build a cloth model from params + mesh data baked into the USD."""
    import inspect as _ins
    from .cloth import _extract_first_mesh, find_cloth_prim_path
    from .shell import find_shell_prim_path, read_shell_params

    cloth_path = find_shell_prim_path(usd_path) or find_cloth_prim_path(usd_path)
    if not cloth_path:
        raise RuntimeError(f"No cloth/shell prim found in {usd_path}")

    p = read_shell_params(usd_path)

    verts, tri = _extract_first_mesh(usd_path)

    with wp.ScopedDevice(device) if device else wp.ScopedDevice(wp.get_preferred_device()):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        table_pos: tuple[float, float, float] | None = None
        table_size: tuple[float, float, float] | None = None
        if table is not None:
            table_pos = tuple(float(v) for v in table.get("pos", (0.0, 0.0, 0.1)))
            table_size = tuple(float(v) for v in table.get("size", (1.0, 1.0, 0.1)))
            table_shape_idx = builder.add_shape_box(
                -1,
                wp.transform(
                    wp.vec3(table_pos[0], table_pos[1], table_pos[2]),
                    wp.quat_identity(),
                ),
                hx=table_size[0],
                hy=table_size[1],
                hz=table_size[2],
            )
            if table_shape_idx is None:
                table_shape_idx = len(builder.shape_color) - 1
            builder.shape_color[int(table_shape_idx)] = (1.0, 1.0, 1.0)

        cloth_pos_z = float(p["dropHeight"])
        cloth_rot = wp.quat(0.0, 0.0, 0.0, 1.0)
        cloth_scale = 1.0
        target_x = 0.0
        target_y = 0.0
        bottom_z: float | None = None
        if table is not None and table_pos is not None and table_size is not None:
            table_top_z = table_pos[2] + table_size[2]
            margin = float(table.get("margin", 0.01))
            target_x = float(table_pos[0])
            target_y = float(table_pos[1])
            bottom_z = table_top_z + margin
            cloth_rot = table.get(
                "rot",
                wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -float(np.pi) / 2.0),
            )
            cloth_scale = float(table.get("cloth_scale", 1.0))

        # Replicate add_cloth_mesh's scale + rot in NumPy to compute the
        # world-space AABB and centre the asset on the target.
        verts_np = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        if verts_np.size > 0:
            qx, qy, qz, qw = (
                float(cloth_rot[0]),
                float(cloth_rot[1]),
                float(cloth_rot[2]),
                float(cloth_rot[3]),
            )
            q_xyz = np.asarray((qx, qy, qz), dtype=np.float64)
            scaled = verts_np * cloth_scale
            cross1 = np.cross(q_xyz, scaled) + qw * scaled
            rotated = scaled + 2.0 * np.cross(q_xyz, cross1)
            bbox_min = rotated.min(axis=0)
            bbox_max = rotated.max(axis=0)
            center_xy = 0.5 * (bbox_min[:2] + bbox_max[:2])
            cloth_pos_x = target_x - float(center_xy[0])
            cloth_pos_y = target_y - float(center_xy[1])
            if bottom_z is not None:
                cloth_pos_z = max(cloth_pos_z, bottom_z - float(bbox_min[2]))
        else:
            cloth_pos_x = target_x
            cloth_pos_y = target_y
            if bottom_z is not None:
                cloth_pos_z = max(cloth_pos_z, bottom_z)

        candidates = {
            "pos":      wp.vec3(cloth_pos_x, cloth_pos_y, cloth_pos_z),
            "rot":      cloth_rot,
            "scale":    cloth_scale,
            "vel":      wp.vec3(0.0, 0.0, 0.0),
            "vertices": verts,
            "indices":  tri,
            "density":  float(p["density"]),
            "tri_ke":   float(p["triStiffness"]),
            "tri_ka":   float(p["triAreaStiffness"]),
            "tri_kd":   float(p["triDamping"]),
            "tri_drag": float(p["triDrag"]),
            "tri_lift": float(p["triLift"]),
            "edge_ke":  float(p["bendStiffness"]),
            "edge_kd":  float(p["bendDamping"]),
            "particle_radius":   float(p["particleRadius"]),
            "add_bending_edges": bool(p["addBendingEdges"]),
        }
        if p.get("style3dTriAnisoKe") is not None:
            candidates["tri_aniso_ke"]  = p["style3dTriAnisoKe"]
        if p.get("style3dEdgeAnisoKe") is not None:
            candidates["edge_aniso_ke"] = p["style3dEdgeAnisoKe"]

        try:
            accepted = set(_ins.signature(builder.add_cloth_mesh).parameters)
        except (TypeError, ValueError):
            accepted = set(candidates)
        kwargs = {k: v for k, v in candidates.items()
                  if k in accepted and v is not None}
        builder.add_cloth_mesh(**kwargs)

        # SolverVBD needs particle graph coloring; harmless for XPBD/Style3D.
        try:
            builder.color()
        except Exception:
            pass

        if solver_name == "style3d":
            s3d = getattr(newton.solvers, "SolverStyle3D", None)
            reg = getattr(s3d, "register_custom_attributes", None) if s3d else None
            if reg is not None:
                reg(builder)

        return builder.finalize()

def _copy_transform(xform: wp.transform) -> wp.transform:
    """Return a value copy of a Warp transform."""
    return wp.transform(
        wp.vec3(float(xform.p[0]), float(xform.p[1]), float(xform.p[2])),
        wp.quat(float(xform.q[0]), float(xform.q[1]), float(xform.q[2]), float(xform.q[3])),
    )


def _set_rod_zero_curvature_rest_poses(
    builder: Any,
    *,
    rod_bodies: list[int],
    points: np.ndarray,
    attached_components: list[tuple[int, list[int]]],
    closed: bool,
    twist_total: float,
) -> list[wp.transform]:
    """Make the rod's rest pose straight while preserving the curved initial pose."""
    initial_body_q = [_copy_transform(xform) for xform in builder.body_q]
    rest_body_q = [_copy_transform(xform) for xform in builder.body_q]

    if not rod_bodies:
        return initial_body_q
    if closed:
        return initial_body_q

    segment_lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
    fallback_tangent = points[1] - points[0] if len(points) > 1 else np.asarray((1.0, 0.0, 0.0))
    rest_direction = _normalize_vector(points[-1] - points[0], fallback_tangent)
    distances = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    rest_points_np = points[0] + distances[:, None] * rest_direction[None, :]
    rest_positions = [
        wp.vec3(float(point[0]), float(point[1]), float(point[2]))
        for point in rest_points_np
    ]
    rest_quaternions = newton.utils.create_parallel_transport_cable_quaternions(
        rest_positions,
        twist_total=float(twist_total),
    )

    for i, body_idx in enumerate(rod_bodies):
        rest_body_q[int(body_idx)] = wp.transform(rest_positions[i], rest_quaternions[i])

    for parent_body, body_indices in attached_components:
        rest_from_initial = rest_body_q[int(parent_body)] * wp.transform_inverse(initial_body_q[int(parent_body)])
        for body_idx in body_indices:
            rest_body_q[int(body_idx)] = rest_from_initial * initial_body_q[int(body_idx)]

    builder.body_q[:] = rest_body_q
    return initial_body_q

def _build_rod_textured_tube(
    builder: "newton.ModelBuilder",
    *,
    rod_bodies: list[int],
    points: list[tuple[float, float, float]],
    radius: float,
    texture_path: str,
    color: tuple[float, float, float] | None,
    radial_segments: int,
    label: str,
) -> None:
    """Hide rod capsules and attach a textured cylindrical tube mesh per segment."""
    points_np = np.asarray(points, dtype=np.float64)
    if points_np.shape[0] != len(rod_bodies) + 1:
        return
    seg_lengths = np.linalg.norm(np.diff(points_np, axis=0), axis=1)
    total_length = float(seg_lengths.sum())
    if total_length <= 0.0:
        return
    arc_starts = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    v_fracs = arc_starts / total_length

    n = max(int(radial_segments), 3)
    thetas = np.linspace(0.0, 2.0 * np.pi, n + 1, dtype=np.float64)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    u_coords = thetas / (2.0 * np.pi)

    visual_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        has_shape_collision=False,
        has_particle_collision=False,
        collision_group=-1,
    )

    for body_id in rod_bodies:
        for shape_idx in builder.body_shapes.get(body_id, []):
            builder.shape_flags[shape_idx] &= ~int(newton.ShapeFlags.VISIBLE)

    identity_xform = wp.transform()
    for i, body_id in enumerate(rod_bodies):
        seg_len = float(seg_lengths[i])
        if seg_len <= 0.0:
            continue
        v0 = float(v_fracs[i])
        v1 = float(v_fracs[i + 1])

        # Duplicate the theta=2*pi seam so U wraps cleanly from 1.0 to 0.0.
        ring0 = np.stack([radius * cos_t, radius * sin_t, np.zeros(n + 1)], axis=1)
        ring1 = np.stack([radius * cos_t, radius * sin_t, np.full(n + 1, seg_len)], axis=1)
        vertices = np.vstack([ring0, ring1]).astype(np.float32)

        normals_xy = np.stack([cos_t, sin_t, np.zeros(n + 1)], axis=1)
        normals = np.vstack([normals_xy, normals_xy]).astype(np.float32)

        uvs0 = np.stack([u_coords, np.full(n + 1, v0)], axis=1)
        uvs1 = np.stack([u_coords, np.full(n + 1, v1)], axis=1)
        uvs = np.vstack([uvs0, uvs1]).astype(np.float32)

        indices = np.empty(n * 6, dtype=np.int32)
        ring_stride = n + 1
        for k in range(n):
            a = k
            b = k + 1
            c = k + ring_stride
            d = k + 1 + ring_stride
            base = k * 6
            indices[base + 0] = a
            indices[base + 1] = c
            indices[base + 2] = b
            indices[base + 3] = b
            indices[base + 4] = c
            indices[base + 5] = d

        mesh = newton.Mesh(
            vertices=vertices,
            indices=indices,
            normals=normals,
            uvs=uvs,
            color=color,
            texture=texture_path,
            compute_inertia=False,
        )
        builder.add_shape_mesh(
            body=int(body_id),
            xform=identity_xform,
            mesh=mesh,
            cfg=visual_cfg,
            label=f"{label}_tube_{i}",
        )


def _build_rod(
    usd_path: str,
    *,
    device: str | None = None,
    textured_tube: bool = False,
    tube_radial_segments: int = 12,
) -> _RodBuildResult:
    """Build an isotropic rod model from a ``NewtonRodAPI``-authored USDA."""
    from .rod import read_rod_params

    params = read_rod_params(usd_path)
    frame_definition = str(params.get("frameDefinition") or "parallelTransport")
    if frame_definition not in ("parallelTransport", "parallel_transport"):
        raise RuntimeError(
            f"Rod asset {usd_path} must declare newton:rod:frameDefinition='parallelTransport' "
            f"(got {frame_definition!r})"
        )
    points = params.get("points") or []
    if len(points) < 3:
        raise RuntimeError(f"Rod asset requires at least 3 centerline points: {usd_path}")

    positions = [wp.vec3(float(x), float(y), float(z)) for x, y, z in points]
    quaternions = newton.utils.create_parallel_transport_cable_quaternions(
        positions,
        twist_total=float(params["twistTotal"]),
    )
    points_np = np.asarray(points, dtype=np.float64)
    label = (params.get("guidePrimPath") or "rod").rsplit("/", maxsplit=1)[-1]
    extra_ignored_paths = [
        str(path)
        for path in (
            params.get("guidePrimPath"),
            params.get("centerlineSourcePath"),
            params.get("radiusSourcePath"),
        )
        if path
    ]
    components, remaining_body_paths, remaining_joint_paths, all_body_paths, all_joint_paths = plan_rod_rigid_imports(
        usd_path,
        params,
        points_np,
        quaternions,
    )

    with wp.ScopedDevice(device) if device else wp.ScopedDevice(wp.get_preferred_device()):
        builder = newton.ModelBuilder()
        builder.rigid_gap = max(float(params["radius"]) * 0.25, 1.0e-4)
        builder.add_ground_plane()
        rod_cfg = newton.ModelBuilder.ShapeConfig(density=float(params["effectiveDensity"]))
        rod_bodies, _ = builder.add_rod(
            positions=positions,
            quaternions=quaternions,
            radius=float(params["radius"]),
            cfg=rod_cfg,
            stretch_stiffness=float(params["axialStiffness"]),
            stretch_damping=float(params["axialDamping"]),
            bend_stiffness=float(params["bendStiffness"]),
            bend_damping=float(params["bendDamping"]),
            closed=bool(params["closed"]),
            label=label,
        )
        filter_body_self_collisions(builder, rod_bodies)
        rod_color = params.get("displayColor") or (0.5, 0.5, 0.5)
        for body_id in rod_bodies:
            for shape_idx in builder.body_shapes.get(body_id, []):
                builder.shape_color[shape_idx] = rod_color

        texture_path = params.get("diffuseTexturePath")
        if textured_tube and texture_path and not bool(params["closed"]):
            _build_rod_textured_tube(
                builder,
                rod_bodies=rod_bodies,
                points=points,
                radius=float(params["radius"]),
                texture_path=str(texture_path),
                color=rod_color,
                radial_segments=int(tube_radial_segments),
                label=label,
            )
        attached_component_bodies: list[tuple[int, list[int]]] = []

        if bool(params["closed"]) and components:
            raise RuntimeError(
                f"Closed rod asset {usd_path} cannot auto-attach endpoint connector components"
            )

        if not bool(params["closed"]):
            for component in components:
                attached_component_bodies.append(
                    attach_rod_connector_component(
                        builder,
                        component=component,
                        usd_path=usd_path,
                        all_body_paths=all_body_paths,
                        all_joint_paths=all_joint_paths,
                        extra_ignored_paths=extra_ignored_paths,
                        rod_bodies=rod_bodies,
                        positions=positions,
                        quaternions=quaternions,
                    )
                )
                _untint_textured_shapes(builder)
        import_remaining_rigid_content(
            builder,
            usd_path=usd_path,
            all_body_paths=all_body_paths,
            all_joint_paths=all_joint_paths,
            remaining_body_paths=remaining_body_paths,
            remaining_joint_paths=remaining_joint_paths,
            extra_ignored_paths=extra_ignored_paths,
        )
        if remaining_body_paths or remaining_joint_paths:
            _untint_textured_shapes(builder)

        initial_body_q = _set_rod_zero_curvature_rest_poses(
            builder,
            rod_bodies=rod_bodies,
            points=points_np,
            attached_components=attached_component_bodies,
            closed=bool(params["closed"]),
            twist_total=float(params["twistTotal"]),
        )
        builder.color()
        model = builder.finalize()
        return _RodBuildResult(
            model=model,
            initial_body_q=wp.array(initial_body_q, dtype=wp.transform, device=model.body_q.device),
        )


def _scene_pins_solver(stage: Usd.Stage) -> bool:
    """True when the stage explicitly authors a solver choice."""
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsScene":
            continue
        a = prim.GetAttribute("newton:solver")
        if not (a and a.HasAuthoredValue()):
            a = prim.GetAttribute("palatial:solver")
        return bool(a and a.HasAuthoredValue())
    return False


def load(usd_path: str, *, solver_override: str | None = None,
         device: str | None = None, fix_base: bool = False,
         table: dict | None = None,
         rod_textured_tube: bool = False,
         rod_tube_radial_segments: int = 12,
         solver_param_overrides: dict | None = None,
         on_model: Callable[[Any], None] | None = None) -> NewtonBundle:
    """Read a converted USD and return a ready-to-step :class:`NewtonBundle`.

    See ``docs/palatial_package.md`` for parameter semantics.

    Args:
        solver_param_overrides: Extra kwargs forwarded to the solver
            constructor. Keys accept either USDA camelCase or solver
            snake_case; caller-supplied values win over scene- and
            shell-level USDA attributes. Use this for VBD knobs the
            schema does not author (e.g.
            ``particle_rest_shape_contact_exclusion_radius``) or to feed
            values parsed from a companion physics JSON.
        on_model: Invoked once with the finalized :class:`~newton.Model`
            immediately before the solver is constructed. Use this when a
            solver bakes model state at ``__init__`` time (e.g. MuJoCo
            actuator type and drive gains).
    """
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise RuntimeError(f"Cannot open USD: {usd_path}")
    solver_name, fps, solver_params = _read_scene_params(stage)
    body_type = _detect_body_type(stage)

    scene_pinned = _scene_pins_solver(stage)
    if body_type == "cloth" and not scene_pinned and not solver_override:
        style3d_used = False
        for prim in stage.Traverse():
            for n in ("newton:shell:style3d:triAnisoKe",
                      "newton:shell:style3d:edgeAnisoKe"):
                a = prim.GetAttribute(n)
                if a and a.HasAuthoredValue():
                    style3d_used = True
                    break
            if style3d_used:
                break
        if style3d_used and getattr(newton.solvers, "SolverStyle3D", None):
            solver_name = "style3d"
        elif getattr(newton.solvers, "SolverVBD", None):
            solver_name = "vbd"
        else:
            solver_name = "xpbd"
    elif body_type == "rod" and not scene_pinned and not solver_override:
        if getattr(newton.solvers, "SolverVBD", None):
            solver_name = "vbd"
        else:
            solver_name = "xpbd"
    del stage

    if solver_override:
        solver_name = solver_override

    if device is None:
        device = str(wp.get_preferred_device())

    # Shell-level VBD knobs supersede any scene-level equivalents.
    if body_type == "cloth":
        from .shell import read_shell_params as _read_shell_params
        _shell = _read_shell_params(usd_path)
        _shell_to_solver = {
            "vbdSelfContactRadius":           "particleSelfContactRadius",
            "vbdSelfContactMargin":           "particleSelfContactMargin",
            "vbdConservativeBoundRelaxation": "particleConservativeBoundRelaxation",
        }
        for sk, ck in _shell_to_solver.items():
            v = _shell.get(sk)
            if v is None:
                continue
            sn = _SOLVER_PARAM_ALIAS.get(ck, ck)
            solver_params.pop(ck, None)
            solver_params.pop(sn, None)
            solver_params[ck] = v

        # Anti-pinch VBD self-contact defaults for dense garment meshes.
        # The converter authors particleSelfContact{Radius,Margin} but not
        # these three knobs. Left at SolverVBD defaults
        # (rest_excl=0.0, topo_filter=2, vertex_buffer=32) rest-adjacent
        # triangles self-collide and erupt into sharp tent-spikes. Inject
        # sane defaults so the bare load() path drapes cleanly; any value
        # authored in the USDA still wins via setdefault.
        if solver_name == "vbd":
            solver_params.setdefault("particleEnableSelfContact", True)
            solver_params.setdefault("particleRestShapeContactExclusionRadius", 0.005)
            solver_params.setdefault("particleTopologicalContactFilterThreshold", 1)
            solver_params.setdefault("particleVertexContactBufferSize", 64)

    rod_initial_body_q = None
    if body_type == "cloth":
        model = _build_cloth(usd_path, device=device, solver_name=solver_name, table=table)
    elif body_type == "rod":
        rod_result = _build_rod(
            usd_path,
            device=device,
            textured_tube=rod_textured_tube,
            tube_radial_segments=rod_tube_radial_segments,
        )
        model = rod_result.model
        rod_initial_body_q = rod_result.initial_body_q
    else:
        model = _build_rigid(usd_path, device=device, fix_base=fix_base,
                             solver_name=solver_name)

    # Caller-supplied overrides win over scene/shell-level USDA attrs.
    # Pop both the alias form and the literal form so an override under
    # either name displaces the USDA-authored value cleanly; then dedupe
    # to keep the dict canonical for downstream consumers.
    if solver_param_overrides:
        _snake_to_camel = {v: k for k, v in _SOLVER_PARAM_ALIAS.items() if k != v}
        for key in solver_param_overrides:
            solver_params.pop(key, None)
            if key in _snake_to_camel:
                solver_params.pop(_snake_to_camel[key], None)
            elif key in _SOLVER_PARAM_ALIAS:
                solver_params.pop(_SOLVER_PARAM_ALIAS[key], None)
        solver_params.update(solver_param_overrides)
        solver_params = _dedupe_solver_params(solver_params)

    # Pre-solver model hook (e.g. set MuJoCo joint drive mode/gains so the
    # solver bakes them at __init__).
    if on_model is not None:
        on_model(model)

    solver = _build_solver(solver_name, model, solver_params)
    state_in = model.state()
    state_out = model.state()
    if rod_initial_body_q is not None:
        wp.copy(state_in.body_q, rod_initial_body_q)
        wp.copy(state_out.body_q, rod_initial_body_q)
        if state_in.body_qd is not None:
            state_in.body_qd.zero_()
        if state_out.body_qd is not None:
            state_out.body_qd.zero_()
        body_q_prev = getattr(solver, "body_q_prev", None)
        if body_q_prev is not None:
            wp.copy(body_q_prev, rod_initial_body_q)

    return NewtonBundle(
        usd_path=usd_path,
        body_type=body_type,
        solver_name=solver_name,
        fps=fps,
        model=model,
        solver=solver,
        state_in=state_in,
        state_out=state_out,
        control=model.control(),
        solver_params=solver_params,
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(prog="framework.load",
                                description="Inspect what would be loaded from a converted USD.")
    p.add_argument("usd")
    a = p.parse_args()
    b = load(a.usd)
    print(f"usd        : {b.usd_path}")
    print(f"body type  : {b.body_type}")
    print(f"solver     : {b.solver_name}  ({type(b.solver).__name__})")
    print(f"fps / dt   : {b.fps}  /  {b.dt:.6f}")
    if b.solver_params:
        print("solver params:")
        for k, v in b.solver_params.items():
            print(f"  {k}: {v}")
    print(f"particles  : {int(b.model.particle_count)}")
    print(f"shapes     : {int(b.model.shape_count)}")
    print(f"bodies     : {int(b.model.body_count)}")
