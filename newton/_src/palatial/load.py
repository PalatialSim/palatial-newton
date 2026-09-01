"""Generic loader for Newton-ready USD assets produced by `framework.convert`.

Reads solver name + simulation params from the converted USD's PhysicsScene and
body-type markers, then constructs a Newton Model + Solver. Caller does not
need to know whether the asset is rigid, cloth, or rod, or which solver was
baked in — that information is read from the file.

Convention (authored by framework.convert):

    /<root>/physicsScene
        apiSchemas        += NewtonSceneAPI [, NewtonXpbdSceneAPI | NewtonKaminoSceneAPI]
        custom token       newton:solver               = "<name>"
        uniform int        newton:timeStepsPerSecond   = <fps>
        gravityDirection, gravityMagnitude  (UsdPhysics.Scene)

    rigid: standard UsdPhysics rigid body / collision / mass / material APIs
           (consumed by newton.utils.parse_usd via add_usd).
    cloth: first UsdGeom.Mesh has
           custom token  newton:bodyType            = "cloth"
           custom float  newton:cloth:density
           custom float  newton:cloth:friction
           custom float  newton:cloth:restitution
           custom float  newton:cloth:dropHeight

Usage:
    from framework.load import load
    bundle = load("/path/to/asset.newton.usda")
    model, solver, fps = bundle.model, bundle.solver, bundle.fps
    state_in, state_out, control = bundle.state_in, bundle.state_out, bundle.control
    dt = 1.0 / fps
    while running:
        contacts = model.collide(state_in)
        solver.step(state_in, state_out, control, contacts, dt)
        state_in, state_out = state_out, state_in
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp
from pxr import Usd, UsdPhysics  # noqa: TID253

# `import newton` registers the newton_usd_schemas plugin (NewtonSceneAPI,
# NewtonXpbdSceneAPI, ...) AND the bundled schema-extension plugin
# (NewtonShellAPI, NewtonClothAPI, NewtonDeformableAPI,
# NewtonShellMaterialAPI, NewtonRodAPI, NewtonRodMaterialAPI) via
# newton/_src/usd/__init__.py. Must precede any pxr.Usd usage in the
# same process.
import newton

from .rod_connectors import (
    _normalize_vector,
    attach_rod_connector_component,
    filter_body_self_collisions,
    import_remaining_rigid_content,
    plan_rod_rigid_imports,
)


def _build_solver(name: str, model: Any, params: dict) -> Any:
    """Construct a solver, forwarding only kwargs the solver actually accepts."""
    import inspect as _ins  # noqa: PLC0415

    classes = {
        "mujoco": getattr(newton.solvers, "SolverMuJoCo", None),
        "xpbd": getattr(newton.solvers, "SolverXPBD", None),
        "featherstone": getattr(newton.solvers, "SolverFeatherstone", None),
        "vbd": getattr(newton.solvers, "SolverVBD", None),
        "semi_implicit": getattr(newton.solvers, "SolverSemiImplicit", None),
        "style3d": getattr(newton.solvers, "SolverStyle3D", None),
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


def _synchronize_newton_contact_capacity(model: Any, solver_name: str, solver_params: dict) -> None:
    """Keep Newton and MuJoCo contact buffers aligned for Newton contacts."""
    if solver_name != "mujoco":
        return

    use_mujoco_contacts = solver_params.get(
        "useMujocoContacts",
        solver_params.get("use_mujoco_contacts"),
    )
    if use_mujoco_contacts is not False:
        return

    authored_capacity = int(solver_params.get("nconmax", 0) or 0)
    if authored_capacity <= 0:
        return

    current_capacity = int(getattr(model, "rigid_contact_max", 0) or 0)
    model.rigid_contact_max = max(current_capacity, authored_capacity)


def _untint_textured_shapes(builder: Any) -> None:
    """Force textured imported shapes to use white fallback vertex colors."""
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
    body_type: str  # "rigid", "cloth", or "rod"
    solver_name: str
    fps: int
    model: Any
    solver: Any
    state_in: Any
    state_out: Any
    control: Any
    solver_params: dict  # raw newton:solver:* attrs from the USD

    @property
    def dt(self) -> float:
        return 1.0 / float(self.fps)


# USD-schema canonical (camelCase) name -> solver constructor kwarg (snake_case).
# Centralized so _read_scene_params can dedupe duplicates and _build_solver can
# translate names without diverging.
_SOLVER_PARAM_ALIAS = {
    "iterations": "iterations",
    "substeps": "substeps",
    "particleEnableSelfContact": "particle_enable_self_contact",
    "particleSelfContactRadius": "particle_self_contact_radius",
    "particleSelfContactMargin": "particle_self_contact_margin",
    "particleConservativeBoundRelaxation": "particle_conservative_bound_relaxation",
}


def _dedupe_solver_params(params: dict) -> dict:
    """Drop snake_case duplicates when the canonical camelCase form is present.

    Some converters write both ``newton:solver:particleEnableSelfContact`` and
    ``newton:solver:particle_enable_self_contact``. They end up in the same
    dict, then ``_build_solver`` lets the last one win in iteration order,
    which is non-deterministic between asset versions. Prefer the canonical
    USD-schema name and warn when a duplicate is dropped.
    """
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
            # Same value, just remove the snake_case duplicate silently.
            out.pop(snake, None)
    return out


def _read_scene_params(stage: Usd.Stage) -> tuple[str, int, dict]:
    """Return (solver_name, fps, solver_params) from the first PhysicsScene."""
    solver = "mujoco"
    fps = 240
    solver_params: dict = {}
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsScene":
            continue
        # Prefer canonical newton:solver namespace; fall back to legacy palatial:solver.
        a = prim.GetAttribute("newton:solver")
        if not (a and a.HasAuthoredValue()):
            a = prim.GetAttribute("palatial:solver")
        if a and a.HasAuthoredValue():
            solver = str(a.Get())
        a = prim.GetAttribute("newton:timeStepsPerSecond")
        if a and a.HasAuthoredValue():
            fps = int(a.Get())
        # Pick up every newton:solver:<key> (or legacy palatial:solver:<key>) custom attr.
        for attr in prim.GetAttributes():
            name = attr.GetName()
            for pfx in ("newton:solver:", "palatial:solver:"):
                if name.startswith(pfx) and attr.HasAuthoredValue():
                    solver_params.setdefault(name[len(pfx) :], attr.Get())
                    break
        break
    return solver, fps, _dedupe_solver_params(solver_params)


def _detect_body_type(stage: Usd.Stage) -> str:
    """Return 'cloth'/'rod' when detected, else 'rigid'.

    Three positive signals (any of):
      - new schema: NewtonShellAPI / NewtonClothAPI in applied schemas
      - new schema: newton:deformable:simulationIntent in {cloth, shell}
      - legacy: newton:bodyType="cloth"
    """
    rod_found = False
    for prim in stage.Traverse():
        applied = set(prim.GetAppliedSchemas())
        # Also walk raw apiSchemas listOp so we still see tokens when this
        # plugin happens not to be loaded in the current process.
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


def _has_movable_rigid_joint(usd_path: str) -> bool:
    """Whether the rigid asset needs collisions between articulated bodies."""
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        return False
    return any(
        prim.IsA(UsdPhysics.Joint) and not prim.IsA(UsdPhysics.FixedJoint)
        for prim in stage.Traverse()
    )


def _build_rigid(
    usd_path: str, *, device: str | None = None, fix_base: bool = False, solver_name: str | None = None
) -> Any:
    """Use Newton's USD parser for rigid assets.

    fix_base: if True, anchor every floating root body to world via a
    `add_joint_fixed(parent=-1, child=body)` joint instead of the FREE
    joints that `parse_usd` adds by default. This mirrors the basic_joints
    example pattern and produces a true mujoco fixed-base articulation.
    """
    with wp.ScopedDevice(device) if device else wp.ScopedDevice(wp.get_preferred_device()):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        bodies_before = len(builder.body_mass)

        if fix_base:
            # parse_usd inlines 4 calls to add_joint_free for floating-base
            # bodies (lines 1212/1222/1273/1327 of import_usd.py), plus a
            # post-pass via add_free_joints_to_floating_bodies. Redirect all
            # of them to add_joint_fixed(parent=-1, child=...) so the root
            # is anchored to world like import_mjcf does for fixed_base.
            counter = [0]

            def _free_to_fixed(child, parent_xform=None, child_xform=None, parent=-1, label=None, **kw):
                counter[0] += 1
                k = label or kw.pop("key", None) or f"fixed_base_{counter[0]}"
                jid = builder.add_joint_fixed(
                    parent=parent,
                    child=child,
                    parent_xform=parent_xform,
                    child_xform=child_xform,
                    label=k,
                )
                # Note: prior versions of import_usd did
                # `builder.joint_q[-7:] = art_xform` after add_joint_free, so
                # we used to pad 7 zeros here to absorb that slice write.
                # Current parse_usd populates joint_q via JointDofConfig and
                # never writes a 7-slice, so padding now causes a length
                # mismatch in builder.finalize()._validate_structure().
                return jid

            builder.add_joint_free = _free_to_fixed
            builder.add_free_joints_to_floating_bodies = lambda *a, **kw: None

        # SimReady ships pre-decomposed convex pieces; tell parser to trust them.
        # Fixed part boundaries do not represent independently moving bodies.
        # Collapsing them avoids redundant internal contacts while preserving
        # authored movable articulations.
        builder.add_usd(
            usd_path,
            skip_mesh_approximation=True,
            collapse_fixed_joints=True,
            enable_self_collisions=_has_movable_rigid_joint(usd_path),
        )

        # Workaround for textured USD imports being tinted by palette colors.
        _untint_textured_shapes(builder)

        if fix_base:
            # Catch any body that escaped all the inline routes (e.g. orphaned
            # body added without articulation) and anchor it as FIXED too.
            new_bodies = range(bodies_before, len(builder.body_mass))
            connected = set(builder.joint_child)
            for b in new_bodies:
                if b in connected:
                    continue
                if builder.body_mass[b] <= 0:
                    continue
                j = builder.add_joint_fixed(parent=-1, child=b, label=f"fixed_base_orphan_{b}")
                builder.add_articulation([j], label=f"articulation_orphan_{b}")

        # SolverVBD requires per-body graph coloring before finalize() when
        # any rigid bodies are present (mirrors the cloth/rod paths above).
        if solver_name == "vbd":
            try:
                builder.color()
            except Exception:
                pass

        return builder.finalize()


def _build_cloth(
    usd_path: str, *, device: str | None = None, solver_name: str | None = None, table: dict | None = None
) -> Any:
    """Build a cloth model using params + mesh data baked into the USD.

    Reads both new (`newton:shell:*` on mesh + bound Material) and legacy
    (`newton:cloth:*` on mesh) attribute namespaces. Forwards only kwargs
    that the installed `add_cloth_mesh` accepts.

    table: optional dict with keys ``pos`` (vec3 in meters, default
        ``(0.0, 0.0, 0.1)``) and ``size`` (vec3 half-extents in meters,
        default ``(1.0, 1.0, 0.1)``). When provided, a static box shape
        is added under the cloth, the cloth's rest orientation is baked
        to lay flat (-pi/2 around X), and the cloth's lowest rotated+
        scaled vertex is positioned exactly ``margin`` meters above the
        table top so it lands right on the surface (instead of free-
        falling). Optional keys ``rot`` (wp.quat, overrides the default
        flat rotation), ``margin`` (meters of clearance between the
        cloth's lowest vertex and the table top, default 0.01), and
        ``cloth_scale`` (float, uniform mesh scale applied via
        ``add_cloth_mesh``; useful for garments authored at full real-
        world size that need to be shrunk to fit the table — e.g. a
        ~1.9 m gown on an 0.8 m table needs cloth_scale ~ 0.4).
    """
    import inspect as _ins  # noqa: PLC0415

    from .cloth import _extract_first_mesh, find_cloth_prim_path  # noqa: PLC0415
    from .shell import find_shell_prim_path, read_shell_params  # noqa: PLC0415

    cloth_path = find_shell_prim_path(usd_path) or find_cloth_prim_path(usd_path)
    if not cloth_path:
        raise RuntimeError(f"No cloth/shell prim found in {usd_path}")

    # `read_shell_params` already merges new + legacy attrs (see shell.py)
    # and walks bound Material prims when reading material attrs.
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
            # Render the table as pure white (RGB). Falls back to indexing
            # the last appended shape when add_shape_box doesn't return an
            # index in older builder versions.
            if table_shape_idx is None:
                table_shape_idx = len(builder.shape_color) - 1
            builder.shape_color[int(table_shape_idx)] = (1.0, 1.0, 1.0)

        # Default cloth drop pose. When a table is configured, bake a
        # gown-style flat rotation (-pi/2 around X) into the rest mesh and
        # lift the spawn z above the table top so the asset drops cleanly.
        # The cloth is ALWAYS centered so its rotated+scaled AABB center in
        # X/Y is aligned with scene origin (or table_pos when --add-table
        # is on), i.e. the asset's geometric centroid lives at scene 0,0.
        # Without this, the cloth_pos translation would place the mesh's
        # local origin (often chest/waist of the gown, not its centroid) at
        # 0,0, which makes the gown drape asymmetrically.
        cloth_pos_z = float(p["dropHeight"])
        cloth_rot = wp.quat(0.0, 0.0, 0.0, 1.0)
        cloth_scale = 1.0
        target_x = 0.0
        target_y = 0.0
        bottom_z: float | None = None  # set when a table is active
        if table is not None and table_pos is not None and table_size is not None:
            table_top_z = table_pos[2] + table_size[2]
            # Default to a 1 cm gap so the cloth starts right above the
            # table top and settles immediately (vs. free-falling).
            margin = float(table.get("margin", 0.01))
            target_x = float(table_pos[0])
            target_y = float(table_pos[1])
            bottom_z = table_top_z + margin
            cloth_rot = table.get(
                "rot",
                wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -float(np.pi) / 2.0),
            )
            cloth_scale = float(table.get("cloth_scale", 1.0))

        # add_cloth_mesh applies scale -> rot -> translation, so replicate
        # scale+rot in NumPy to compute the world-space AABB and pick a
        # translation that centers the asset.
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
            # v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
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

        # Candidate kwargs covering both old and current Newton signatures.
        candidates = {
            "pos": wp.vec3(cloth_pos_x, cloth_pos_y, cloth_pos_z),
            "rot": cloth_rot,
            "scale": cloth_scale,
            "vel": wp.vec3(0.0, 0.0, 0.0),
            "vertices": verts,
            "indices": tri,
            "density": float(p["density"]),
            "tri_ke": float(p["triStiffness"]),
            "tri_ka": float(p["triAreaStiffness"]),
            "tri_kd": float(p["triDamping"]),
            "tri_drag": float(p["triDrag"]),
            "tri_lift": float(p["triLift"]),
            "edge_ke": float(p["bendStiffness"]),
            "edge_kd": float(p["bendDamping"]),
            "particle_radius": float(p["particleRadius"]),
            "add_bending_edges": bool(p["addBendingEdges"]),
        }
        # Style3D-only anisotropic stiffness; pass tuples when authored.
        if p.get("style3dTriAnisoKe") is not None:
            candidates["tri_aniso_ke"] = p["style3dTriAnisoKe"]
        if p.get("style3dEdgeAnisoKe") is not None:
            candidates["edge_aniso_ke"] = p["style3dEdgeAnisoKe"]

        try:
            accepted = set(_ins.signature(builder.add_cloth_mesh).parameters)
        except (TypeError, ValueError):
            accepted = set(candidates)
        kwargs = {k: v for k, v in candidates.items() if k in accepted and v is not None}
        builder.add_cloth_mesh(**kwargs)

        # SolverVBD requires a particle graph coloring before finalize().
        # Harmless for XPBD/Style3D, so always do it for cloth.
        try:
            builder.color()
        except Exception:
            pass

        # Style3D injects extra per-particle/per-edge custom attributes
        # (e.g. rest-state buffers) that must exist on the model at finalize
        # time. Register them on the builder before finalize when style3d
        # is the chosen solver.
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
    rest_positions = [wp.vec3(float(point[0]), float(point[1]), float(point[2])) for point in rest_points_np]
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
    builder: newton.ModelBuilder,
    *,
    rod_bodies: list[int],
    points: list[tuple[float, float, float]],
    radius: float,
    texture_path: str,
    color: tuple[float, float, float] | None,
    radial_segments: int,
    label: str,
) -> None:
    """Hide rod capsules and attach a textured cylindrical tube mesh per segment.

    For each rod body, builds a closed cylindrical strip whose axis runs along
    local +Z from the segment start to its end (matching the body frame produced
    by :meth:`ModelBuilder.add_rod`). UVs are laid out with U = theta/(2*pi)
    around the tube and V increasing monotonically along the rod (arc-length
    fraction) so the diffuse texture is applied without seams along the cable.
    """
    points_np = np.asarray(points, dtype=np.float64)
    if points_np.shape[0] != len(rod_bodies) + 1:
        # Should not happen for non-closed rods, but bail out safely.
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

    # Hide the existing capsule shapes from rendering (collisions stay intact).
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

        # Two rings of n+1 vertices each (the seam at theta=2*pi is duplicated
        # so the U coordinate wraps cleanly from 1.0 back to 0.0 visually).
        ring0 = np.stack([radius * cos_t, radius * sin_t, np.zeros(n + 1)], axis=1)
        ring1 = np.stack([radius * cos_t, radius * sin_t, np.full(n + 1, seg_len)], axis=1)
        vertices = np.vstack([ring0, ring1]).astype(np.float32)

        normals_xy = np.stack([cos_t, sin_t, np.zeros(n + 1)], axis=1)
        normals = np.vstack([normals_xy, normals_xy]).astype(np.float32)

        uvs0 = np.stack([u_coords, np.full(n + 1, v0)], axis=1)
        uvs1 = np.stack([u_coords, np.full(n + 1, v1)], axis=1)
        uvs = np.vstack([uvs0, uvs1]).astype(np.float32)

        # Triangle indices: for each radial quad form two triangles. Winding
        # is CCW when viewed from outside (normals point radially outward).
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
    """Build an isotropic rod model from a NewtonRodAPI-authored USDA."""
    from .rod import read_rod_params  # noqa: PLC0415

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
        # Paint every rod segment with one color sourced from the cable's USD
        # material (falls back to neutral grey) so the rod renders uniformly
        # instead of with per-shape palette colors.
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
            raise RuntimeError(f"Closed rod asset {usd_path} cannot auto-attach endpoint connector components")

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
    """Return True when the stage explicitly authors a solver choice."""
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsScene":
            continue
        a = prim.GetAttribute("newton:solver")
        if not (a and a.HasAuthoredValue()):
            a = prim.GetAttribute("palatial:solver")
        return bool(a and a.HasAuthoredValue())
    return False


def load(
    usd_path: str,
    *,
    solver_override: str | None = None,
    device: str | None = None,
    fix_base: bool = False,
    table: dict | None = None,
    rod_textured_tube: bool = False,
    rod_tube_radial_segments: int = 12,
) -> NewtonBundle:
    """Read converted USD, build Newton model + solver, return ready-to-step bundle.

    device: warp device string ("cuda:0", "cpu", ...). Defaults to GPU if
    available (wp.get_preferred_device()).
    table: cloth-only; optional dict with ``pos`` and ``size`` (half-extents)
    in meters. When provided, a static box is added under the cloth so the
    asset can drop onto it.
    rod_textured_tube: rod-only; when True and the USD binds a diffuse texture
    to the cable mesh, hide the underlying capsule shapes and render the rod
    as a swept textured cylinder. Has no effect on closed rods or when no
    on-disk texture can be resolved.
    rod_tube_radial_segments: rod-only; number of radial segments used for the
    swept tube when ``rod_textured_tube`` is enabled.
    """
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise RuntimeError(f"Cannot open USD: {usd_path}")
    solver_name, fps, solver_params = _read_scene_params(stage)
    body_type = _detect_body_type(stage)

    # Defensive: if the scene didn't pin a solver, pick a sensible default
    # for deformable assets.
    scene_pinned = _scene_pins_solver(stage)
    if body_type == "cloth" and not scene_pinned and not solver_override:
        style3d_used = False
        for prim in stage.Traverse():
            for n in ("newton:shell:style3d:triAnisoKe", "newton:shell:style3d:edgeAnisoKe"):
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

    # Forward shell-level VBD knobs (authored on the cloth Material as
    # `newton:shell:vbd:*`) into solver_params. Shell-level is the
    # canonical source for these particle self-contact tunables; any
    # scene-level `newton:solver:*` equivalent is silently superseded.
    if body_type == "cloth":
        from .shell import read_shell_params as _read_shell_params  # noqa: PLC0415

        _shell = _read_shell_params(usd_path)
        _shell_to_solver = {
            "vbdSelfContactRadius": "particleSelfContactRadius",
            "vbdSelfContactMargin": "particleSelfContactMargin",
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
        model = _build_rigid(usd_path, device=device, fix_base=fix_base, solver_name=solver_name)

    _synchronize_newton_contact_capacity(model, solver_name, solver_params)
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

    p = argparse.ArgumentParser(prog="framework.load", description="Inspect what would be loaded from a converted USD.")
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
