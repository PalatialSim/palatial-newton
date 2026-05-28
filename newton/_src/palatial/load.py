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

# `import newton` registers the newton_usd_schemas plugin (NewtonSceneAPI,
# NewtonXpbdSceneAPI, ...) AND the bundled schema-extension plugin
# (NewtonShellAPI, NewtonClothAPI, NewtonDeformableAPI,
# NewtonShellMaterialAPI, NewtonRodAPI, NewtonRodMaterialAPI) via
# newton/_src/usd/__init__.py. Must precede any pxr.Usd usage in the
# same process.
import newton

from pxr import Usd, UsdGeom

from dataclasses import dataclass
import re
from typing import Any
import numpy as np
import warp as wp

_TransformParts = tuple[wp.vec3, wp.quat]


def _build_solver(name: str, model: Any, params: dict) -> Any:
    """Construct a solver, forwarding only kwargs the solver actually accepts."""
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

    # Translate USD attr names -> solver constructor kwargs.
    alias = {
        "iterations":                "iterations",
        "substeps":                  "substeps",
        "particleEnableSelfContact": "particle_enable_self_contact",
        "particleSelfContactRadius": "particle_self_contact_radius",
        "particleSelfContactMargin": "particle_self_contact_margin",
    }
    try:
        sig = _ins.signature(cls.__init__)
        accepted = set(sig.parameters.keys())
    except (TypeError, ValueError):
        accepted = set()

    kwargs = {}
    for k, v in params.items():
        py = alias.get(k, k)
        if py in accepted:
            kwargs[py] = v
    return cls(model, **kwargs)


def _untint_textured_shapes(builder: Any) -> None:
    """Force textured imported shapes to use white fallback vertex colors."""
    for i, src in enumerate(builder.shape_source):
        if src is not None and getattr(src, "texture", None) is not None:
            builder.shape_color[i] = (1.0, 1.0, 1.0)


@dataclass
class _UsdJointLink:
    """Joint prim metadata used to import rigid subsets around a rod."""

    prim_path: str
    body0_path: str
    body1_path: str


@dataclass
class _RodAttachmentComponent:
    """A rigid connector assembly that should attach to one rod endpoint."""

    root_body_path: str
    body_paths: set[str]
    joint_paths: set[str]
    endpoint_name: str
    relative_xform: _TransformParts
    parent_xform: _TransformParts
    child_xform: _TransformParts
    proxy_bounds: dict[str, tuple[wp.vec3, wp.vec3]]


@dataclass
class _RodBuildResult:
    """Rod model plus the curved input pose used as the initial state."""

    model: Any
    initial_body_q: Any


@dataclass
class NewtonBundle:
    """Everything needed to step the converted asset in Newton."""
    usd_path: str
    body_type: str           # "rigid", "cloth", or "rod"
    solver_name: str
    fps: int
    model: Any
    solver: Any
    state_in: Any
    state_out: Any
    control: Any
    solver_params: dict      # raw newton:solver:* attrs from the USD

    @property
    def dt(self) -> float:
        return 1.0 / float(self.fps)


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
                    solver_params.setdefault(name[len(pfx):], attr.Get())
                    break
        break
    return solver, fps, solver_params


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


def _build_rigid(usd_path: str, *, device: str | None = None,
                 fix_base: bool = False) -> Any:
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
        builder.add_usd(usd_path, skip_mesh_approximation=True)

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
                j = builder.add_joint_fixed(parent=-1, child=b,
                                            label=f"fixed_base_orphan_{b}")
                builder.add_articulation([j], label=f"articulation_orphan_{b}")
        return builder.finalize()


def _build_cloth(usd_path: str, *, device: str | None = None,
                 solver_name: str | None = None) -> Any:
    """Build a cloth model using params + mesh data baked into the USD.

    Reads both new (`newton:shell:*` on mesh + bound Material) and legacy
    (`newton:cloth:*` on mesh) attribute namespaces. Forwards only kwargs
    that the installed `add_cloth_mesh` accepts.
    """
    import inspect as _ins
    from .cloth import _extract_first_mesh, find_cloth_prim_path
    from .shell import find_shell_prim_path, read_shell_params

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

        # Candidate kwargs covering both old and current Newton signatures.
        candidates = {
            "pos":      wp.vec3(0.0, 0.0, float(p["dropHeight"])),
            "rot":      wp.quat(0.0, 0.0, 0.0, 1.0),
            "scale":    1.0,
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
        # Style3D-only anisotropic stiffness; pass tuples when authored.
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


def _has_api_schema(prim: Usd.Prim, schema_name: str) -> bool:
    """Return True when a prim carries the requested applied schema."""
    applied = set(prim.GetAppliedSchemas())
    raw = prim.GetMetadata("apiSchemas")
    if raw is not None:
        for items_attr in ("prependedItems", "appendedItems", "explicitItems"):
            for tok in getattr(raw, items_attr, []) or []:
                applied.add(str(tok))
    return schema_name in applied


def _collect_rigid_body_paths(stage: Usd.Stage) -> set[str]:
    """Return all rigid-body prim paths on a stage."""
    out: set[str] = set()
    for prim in stage.Traverse():
        if _has_api_schema(prim, "PhysicsRigidBodyAPI"):
            out.add(prim.GetPath().pathString)
    return out


def _collect_joint_links(stage: Usd.Stage, rigid_body_paths: set[str]) -> list[_UsdJointLink]:
    """Return joint links between rigid-body prims on a stage."""
    out: list[_UsdJointLink] = []
    for prim in stage.Traverse():
        rel0 = prim.GetRelationship("physics:body0")
        rel1 = prim.GetRelationship("physics:body1")
        if rel0 is None and rel1 is None:
            continue
        targets0 = rel0.GetTargets() if rel0 else []
        targets1 = rel1.GetTargets() if rel1 else []
        if not targets0 or not targets1:
            continue
        body0_path = str(targets0[0])
        body1_path = str(targets1[0])
        if body0_path not in rigid_body_paths or body1_path not in rigid_body_paths:
            continue
        out.append(
            _UsdJointLink(
                prim_path=prim.GetPath().pathString,
                body0_path=body0_path,
                body1_path=body1_path,
            )
        )
    return out


def _normalize_vector(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Return a normalized vector, or the fallback when degenerate."""
    norm = float(np.linalg.norm(values))
    if norm <= 1.0e-12:
        return np.asarray(fallback, dtype=np.float64)
    return np.asarray(values, dtype=np.float64) / norm


def _rigid_transform_to_newton(
    matrix: Any,
    *,
    meters_per_unit: float,
    up_axis: str,
    to_newton_world: Any,
    matrix_transform_points: Any,
) -> wp.transform:
    """Convert a USD rigid transform matrix into Newton world coordinates."""
    basis = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ],
        dtype=np.float64,
    )
    transformed = matrix_transform_points(matrix, basis)
    transformed = to_newton_world(transformed, meters_per_unit, up_axis)

    origin = transformed[0]
    x_axis = _normalize_vector(transformed[1] - origin, np.asarray((1.0, 0.0, 0.0), dtype=np.float64))
    y_axis = transformed[2] - origin
    y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
    y_axis = _normalize_vector(y_axis, np.asarray((0.0, 1.0, 0.0), dtype=np.float64))
    z_axis = np.cross(x_axis, y_axis)
    z_axis = _normalize_vector(z_axis, np.asarray((0.0, 0.0, 1.0), dtype=np.float64))
    y_axis = _normalize_vector(np.cross(z_axis, x_axis), np.asarray((0.0, 1.0, 0.0), dtype=np.float64))

    rot_np = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32, copy=False)
    rot = wp.quat_from_matrix(wp.mat33(*rot_np.flatten().tolist()))
    return wp.transform(wp.vec3(float(origin[0]), float(origin[1]), float(origin[2])), rot)


def _read_rigid_body_points(
    stage: Usd.Stage,
    prim: Usd.Prim,
    *,
    meters_per_unit: float,
    up_axis: str,
    xform_cache: UsdGeom.XformCache,
    read_mesh_world_points: Any,
    prefer_colliders: bool,
) -> np.ndarray:
    """Read representative rigid-body points in Newton world coordinates."""
    point_sets: list[np.ndarray] = []

    if prefer_colliders:
        for child in prim.GetChildren():
            if not child.GetName().startswith("Colliders_"):
                continue
            for mesh_prim in child.GetChildren():
                if not mesh_prim.IsA(UsdGeom.Mesh):
                    continue
                points = read_mesh_world_points(
                    stage,
                    mesh_prim,
                    meters_per_unit=meters_per_unit,
                    up_axis=up_axis,
                    xform_cache=xform_cache,
                )
                if points.size > 0:
                    point_sets.append(points)
        if point_sets:
            return np.concatenate(point_sets, axis=0)

    for child in prim.GetChildren():
        if not child.IsA(UsdGeom.Mesh):
            continue
        points = read_mesh_world_points(
            stage,
            child,
            meters_per_unit=meters_per_unit,
            up_axis=up_axis,
            xform_cache=xform_cache,
        )
        if points.size > 0:
            point_sets.append(points)

    if point_sets:
        return np.concatenate(point_sets, axis=0)
    return np.empty((0, 3), dtype=np.float64)


def _component_endpoint_name(points: np.ndarray, component_points: np.ndarray) -> str:
    """Choose the closer rod endpoint for a rigid component."""
    if component_points.size == 0 or len(points) < 2:
        return "start"
    centroid = component_points.mean(axis=0)
    start_dist = float(np.linalg.norm(centroid - points[0]))
    end_dist = float(np.linalg.norm(centroid - points[-1]))
    return "start" if start_dist <= end_dist else "end"


def _component_proxy_bounds(component_points: np.ndarray, root_world: wp.transform) -> tuple[wp.vec3, wp.vec3] | None:
    """Return a connector contact proxy box in root-body local space."""
    if component_points.size == 0:
        return None

    root_to_local = wp.transform_inverse(root_world)
    local_points = []
    for point in component_points:
        p_local = wp.transform_point(root_to_local, wp.vec3(float(point[0]), float(point[1]), float(point[2])))
        local_points.append((float(p_local[0]), float(p_local[1]), float(p_local[2])))

    local_np = np.asarray(local_points, dtype=np.float64)
    local_np = local_np[np.isfinite(local_np).all(axis=1)]
    if local_np.size == 0:
        return None

    bounds_min = local_np.min(axis=0)
    bounds_max = local_np.max(axis=0)
    center = 0.5 * (bounds_min + bounds_max)
    half_extents = np.maximum(0.5 * (bounds_max - bounds_min), 1.0e-4)
    return (
        wp.vec3(float(center[0]), float(center[1]), float(center[2])),
        wp.vec3(float(half_extents[0]), float(half_extents[1]), float(half_extents[2])),
    )


def _component_mouth_anchor(
    root_points: np.ndarray,
    *,
    tangent: np.ndarray,
    endpoint_name: str,
    rod_radius: float,
    fallback: np.ndarray,
) -> np.ndarray:
    """Return the centered connector mouth point nearest the rod endpoint."""
    if root_points.size == 0:
        return np.asarray(fallback, dtype=np.float64)

    valid_points = np.asarray(root_points, dtype=np.float64)
    valid_points = valid_points[np.isfinite(valid_points).all(axis=1)]
    if valid_points.size == 0:
        return np.asarray(fallback, dtype=np.float64)

    projections = valid_points @ tangent
    extreme_projection = float(np.max(projections) if endpoint_name == "start" else np.min(projections))
    span = float(np.max(projections) - np.min(projections))
    base_band = max(float(rod_radius), 1.0e-4)
    target_count = min(4, len(valid_points))
    mouth_mask = np.zeros(len(valid_points), dtype=bool)

    for multiplier in (1.0, 2.0, 4.0):
        band = min(base_band * multiplier, span) if span > 0.0 else base_band * multiplier
        if endpoint_name == "start":
            mouth_mask = projections >= extreme_projection - band
        else:
            mouth_mask = projections <= extreme_projection + band
        if int(mouth_mask.sum()) >= target_count:
            break

    if not mouth_mask.any():
        mouth_mask[int(np.argmax(projections) if endpoint_name == "start" else np.argmin(projections))] = True

    mouth_points = valid_points[mouth_mask]
    mouth_projections = projections[mouth_mask]
    transverse_points = mouth_points - mouth_projections[:, None] * tangent[None, :]
    transverse_center = transverse_points.mean(axis=0)
    return transverse_center + extreme_projection * tangent


def _component_relative_xform(
    *,
    stage: Usd.Stage,
    root_body_path: str,
    endpoint_name: str,
    points: np.ndarray,
    quaternions: list[wp.quat],
    rod_radius: float,
    meters_per_unit: float,
    up_axis: str,
    xform_cache: UsdGeom.XformCache,
    to_newton_world: Any,
    matrix_transform_points: Any,
    read_mesh_world_points: Any,
) -> tuple[_TransformParts, _TransformParts, _TransformParts]:
    """Return the root-body transform that aligns a connector mouth to a rod end."""
    root_prim = stage.GetPrimAtPath(root_body_path)
    root_world = _rigid_transform_to_newton(
        xform_cache.GetLocalToWorldTransform(root_prim),
        meters_per_unit=meters_per_unit,
        up_axis=up_axis,
        to_newton_world=to_newton_world,
        matrix_transform_points=matrix_transform_points,
    )

    root_points = _read_rigid_body_points(
        stage,
        root_prim,
        meters_per_unit=meters_per_unit,
        up_axis=up_axis,
        xform_cache=xform_cache,
        read_mesh_world_points=read_mesh_world_points,
        prefer_colliders=True,
    )
    if len(points) < 2:
        tangent = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    elif endpoint_name == "start":
        tangent = _normalize_vector(points[1] - points[0], np.asarray((1.0, 0.0, 0.0), dtype=np.float64))
    else:
        tangent = _normalize_vector(points[-1] - points[-2], np.asarray((1.0, 0.0, 0.0), dtype=np.float64))

    anchor_world = _component_mouth_anchor(
        root_points,
        tangent=tangent,
        endpoint_name=endpoint_name,
        rod_radius=rod_radius,
        fallback=np.asarray(root_world.p, dtype=np.float64),
    )

    endpoint_world = points[0] if endpoint_name == "start" else points[-1]
    root_translation = np.asarray(root_world.p, dtype=np.float64) + (endpoint_world - anchor_world)
    desired_root_world = wp.transform(
        wp.vec3(float(root_translation[0]), float(root_translation[1]), float(root_translation[2])),
        root_world.q,
    )

    if endpoint_name == "start":
        parent_world = wp.transform(wp.vec3(*points[0].tolist()), quaternions[0])
        parent_anchor = wp.transform_identity()
    else:
        parent_world = wp.transform(wp.vec3(*points[-2].tolist()), quaternions[-1])
        segment_length = float(np.linalg.norm(points[-1] - points[-2]))
        parent_anchor = wp.transform(wp.vec3(0.0, 0.0, segment_length), wp.quat_identity())
    endpoint_frame_world = parent_world * parent_anchor
    child_anchor = wp.transform_inverse(desired_root_world) * endpoint_frame_world
    relative = wp.transform_inverse(parent_world) * desired_root_world
    return (relative.p, relative.q), (parent_anchor.p, parent_anchor.q), (child_anchor.p, child_anchor.q)


def _collect_component_paths(
    root_path: str,
    *,
    adjacency: dict[str, set[str]],
    joint_lookup: dict[frozenset[str], set[str]],
    blocked_paths: set[str],
) -> tuple[set[str], set[str]]:
    """Return rigid-body and joint paths for a connector component."""
    stack = [root_path]
    body_paths: set[str] = set()
    joint_paths: set[str] = set()

    while stack:
        current = stack.pop()
        if current in body_paths or current in blocked_paths:
            continue
        body_paths.add(current)
        for neighbor in adjacency.get(current, ()):
            if neighbor in blocked_paths:
                continue
            joint_paths.update(joint_lookup.get(frozenset((current, neighbor)), set()))
            if neighbor not in body_paths:
                stack.append(neighbor)

    return body_paths, joint_paths


def _subset_ignore_paths(
    *,
    all_body_paths: set[str],
    all_joint_paths: set[str],
    keep_body_paths: set[str],
    keep_joint_paths: set[str],
    extra_ignored_paths: list[str],
) -> list[str]:
    """Build ignore regexes that keep only a selected rigid subset."""
    ignore_paths = [re.escape(path) for path in extra_ignored_paths if path]
    for path in sorted(all_body_paths):
        if path not in keep_body_paths:
            ignore_paths.append(re.escape(path))
    for path in sorted(all_joint_paths):
        if path not in keep_joint_paths:
            ignore_paths.append(re.escape(path))
    return ignore_paths


def _reparent_existing_joint(
    builder: Any,
    *,
    joint_idx: int,
    parent_body: int,
    parent_xform: wp.transform,
    child_xform: wp.transform,
    label: str,
) -> None:
    """Rewrite an imported base joint so it attaches to a rod body instead of world."""
    child_body = int(builder.joint_child[joint_idx])
    old_parent = int(builder.joint_parent[joint_idx])

    builder.joint_parent[joint_idx] = parent_body
    builder.joint_X_p[joint_idx] = parent_xform
    builder.joint_X_c[joint_idx] = child_xform
    builder.joint_label[joint_idx] = label
    builder.joint_collision_filter_parent[joint_idx] = True

    child_parents = builder.joint_parents.get(child_body, [])
    builder.joint_parents[child_body] = [
        (parent_body if idx == joint_idx else parent, idx) for parent, idx in child_parents
    ]

    if old_parent in builder.joint_children:
        builder.joint_children[old_parent] = [
            (child, idx) for child, idx in builder.joint_children[old_parent] if idx != joint_idx
        ]
        if not builder.joint_children[old_parent]:
            del builder.joint_children[old_parent]
    builder.joint_children.setdefault(parent_body, []).append((child_body, joint_idx))


def _disable_shape_collisions(builder: Any, shape_indices: range | list[int]) -> None:
    """Disable imported connector collisions for the minimal rod path."""
    collision_bits = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES)
    for shape_idx in shape_indices:
        builder.shape_flags[shape_idx] = int(builder.shape_flags[shape_idx]) & ~collision_bits
        builder.shape_collision_group[shape_idx] = 0


def _filter_body_self_collisions(builder: Any, body_indices: list[int]) -> None:
    """Disable collisions among shapes attached to the given bodies."""
    shape_indices: list[int] = []
    for body_idx in body_indices:
        shape_indices.extend(int(shape_idx) for shape_idx in builder.body_shapes.get(int(body_idx), ()))

    for i, shape_a in enumerate(shape_indices):
        for shape_b in shape_indices[i + 1:]:
            builder.add_shape_collision_filter_pair(shape_a, shape_b)


def _filter_shapes_against_bodies(builder: Any, shape_indices: list[int], body_indices: list[int]) -> None:
    """Disable collisions between selected shapes and all shapes on selected bodies."""
    body_shape_indices: list[int] = []
    for body_idx in body_indices:
        body_shape_indices.extend(int(shape_idx) for shape_idx in builder.body_shapes.get(int(body_idx), ()))

    for shape_idx in shape_indices:
        for body_shape_idx in body_shape_indices:
            if shape_idx == body_shape_idx:
                continue
            builder.add_shape_collision_filter_pair(shape_idx, body_shape_idx)


def _shift_transform_translation(xform: wp.transform, offset: np.ndarray) -> wp.transform:
    """Return an equivalent local transform after moving its body origin by offset."""
    p = np.asarray((float(xform.p[0]), float(xform.p[1]), float(xform.p[2])), dtype=np.float64) - offset
    return wp.transform(wp.vec3(float(p[0]), float(p[1]), float(p[2])), xform.q)


def _shape_local_points(builder: Any, shape_idx: int) -> np.ndarray:
    """Return representative shape points in its body-local frame."""
    xform = builder.shape_transform[shape_idx]
    scale = builder.shape_scale[shape_idx]
    src = builder.shape_source[shape_idx]
    vertices = getattr(src, "vertices", None)
    if vertices is not None:
        points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        scaled = points * np.asarray((float(scale[0]), float(scale[1]), float(scale[2])), dtype=np.float64)
    else:
        hx, hy, hz = float(scale[0]), float(scale[1]), float(scale[2])
        scaled = np.asarray(
            [
                (sx * hx, sy * hy, sz * hz)
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )

    local_points = []
    for point in scaled:
        p_local = wp.transform_point(xform, wp.vec3(float(point[0]), float(point[1]), float(point[2])))
        local_points.append((float(p_local[0]), float(p_local[1]), float(p_local[2])))
    return np.asarray(local_points, dtype=np.float64)


def _recenter_body_frames(builder: Any, body_indices: list[int]) -> dict[int, np.ndarray]:
    """Move imported body origins to local geometry centers while preserving world geometry."""
    offsets: dict[int, np.ndarray] = {}
    local_coms: dict[int, np.ndarray] = {}
    for body_idx in body_indices:
        body_points: list[np.ndarray] = []
        shape_indices = list(builder.body_shapes.get(int(body_idx), ()))
        collider_shape_indices = [
            int(shape_idx)
            for shape_idx in shape_indices
            if "/Colliders" in str(builder.shape_label[int(shape_idx)])
            or "/Collider" in str(builder.shape_label[int(shape_idx)])
        ]
        for shape_idx in collider_shape_indices or shape_indices:
            points = _shape_local_points(builder, int(shape_idx))
            if points.size > 0:
                body_points.append(points)
        if not body_points:
            continue

        points_np = np.concatenate(body_points, axis=0)
        points_np = points_np[np.isfinite(points_np).all(axis=1)]
        if points_np.size == 0:
            continue

        offset = 0.5 * (points_np.min(axis=0) + points_np.max(axis=0))
        if float(np.linalg.norm(offset)) <= 1.0e-8:
            continue
        offsets[int(body_idx)] = offset
        local_coms[int(body_idx)] = points_np.mean(axis=0) - offset

    for body_idx, offset in offsets.items():
        old_body_xform = builder.body_q[body_idx]
        new_body_p = wp.transform_point(old_body_xform, wp.vec3(float(offset[0]), float(offset[1]), float(offset[2])))
        builder.body_q[body_idx] = wp.transform(new_body_p, old_body_xform.q)

        local_com = local_coms[body_idx]
        builder.body_com[body_idx] = wp.vec3(
            float(local_com[0]),
            float(local_com[1]),
            float(local_com[2]),
        )

        for shape_idx in builder.body_shapes.get(body_idx, ()):
            builder.shape_transform[int(shape_idx)] = _shift_transform_translation(
                builder.shape_transform[int(shape_idx)],
                offset,
            )

    for joint_idx in range(len(builder.joint_type)):
        parent_body = int(builder.joint_parent[joint_idx])
        child_body = int(builder.joint_child[joint_idx])
        if parent_body in offsets:
            builder.joint_X_p[joint_idx] = _shift_transform_translation(
                builder.joint_X_p[joint_idx],
                offsets[parent_body],
            )
        if child_body in offsets:
            builder.joint_X_c[joint_idx] = _shift_transform_translation(
                builder.joint_X_c[joint_idx],
                offsets[child_body],
            )

    return offsets


def _hide_oversized_connector_visuals(builder: Any, body_indices: list[int]) -> None:
    """Hide polluted connector visuals and show their clean collider fallback."""
    visible_bit = int(newton.ShapeFlags.VISIBLE)
    for body_idx in body_indices:
        shape_indices = [int(shape_idx) for shape_idx in builder.body_shapes.get(int(body_idx), ())]
        collider_shape_indices = [
            shape_idx
            for shape_idx in shape_indices
            if "/Colliders" in str(builder.shape_label[shape_idx])
            or "/Collider" in str(builder.shape_label[shape_idx])
        ]
        if not collider_shape_indices:
            continue

        collider_points = [
            points
            for shape_idx in collider_shape_indices
            if (points := _shape_local_points(builder, shape_idx)).size > 0
        ]
        if not collider_points:
            continue
        collider_np = np.concatenate(collider_points, axis=0)
        collider_extent = float(np.max(collider_np.max(axis=0) - collider_np.min(axis=0)))
        if collider_extent <= 0.0:
            continue

        hidden_visual = False
        for shape_idx in shape_indices:
            if shape_idx in collider_shape_indices:
                continue
            if not int(builder.shape_flags[shape_idx]) & visible_bit:
                continue
            visual_points = _shape_local_points(builder, shape_idx)
            if visual_points.size == 0:
                continue
            visual_extent = float(np.max(visual_points.max(axis=0) - visual_points.min(axis=0)))
            if visual_extent > max(collider_extent * 6.0, collider_extent + 0.25):
                builder.shape_flags[shape_idx] = int(builder.shape_flags[shape_idx]) & ~visible_bit
                hidden_visual = True

        if hidden_visual:
            for shape_idx in collider_shape_indices:
                builder.shape_flags[shape_idx] = int(builder.shape_flags[shape_idx]) | visible_bit
                builder.shape_color[shape_idx] = (1.0, 1.0, 1.0)


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
) -> list[wp.transform]:
    """Make the rod's rest pose straight while preserving the curved initial pose."""
    initial_body_q = [_copy_transform(xform) for xform in builder.body_q]
    rest_body_q = [_copy_transform(xform) for xform in builder.body_q]

    if not rod_bodies:
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
    rest_quaternions = newton.utils.create_parallel_transport_cable_quaternions(rest_positions)

    for i, body_idx in enumerate(rod_bodies):
        rest_body_q[int(body_idx)] = wp.transform(rest_positions[i], rest_quaternions[i])

    for parent_body, body_indices in attached_components:
        rest_from_initial = rest_body_q[int(parent_body)] * wp.transform_inverse(initial_body_q[int(parent_body)])
        for body_idx in body_indices:
            rest_body_q[int(body_idx)] = rest_from_initial * initial_body_q[int(body_idx)]

    builder.body_q[:] = rest_body_q
    return initial_body_q


def _plan_rod_rigid_imports(
    usd_path: str,
    params: dict,
    points: np.ndarray,
    quaternions: list[wp.quat],
) -> tuple[list[_RodAttachmentComponent], set[str], set[str], set[str], set[str]]:
    """Split rigid USD content into rod-attached connector components and leftovers."""
    from .rod import _matrix_transform_points, _read_mesh_world_points, _stage_units, _to_newton_world

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        return [], set(), set(), set(), set()

    rigid_body_paths = _collect_rigid_body_paths(stage)
    all_joint_links = _collect_joint_links(stage, rigid_body_paths)
    if not rigid_body_paths:
        return [], set(), set(), set(), set()

    helper_paths = {
        str(path)
        for path in (
            params.get("guidePrimPath"),
            params.get("centerlineSourcePath"),
            params.get("radiusSourcePath"),
        )
        if path
    }
    helper_body_paths = rigid_body_paths & helper_paths

    adjacency: dict[str, set[str]] = {path: set() for path in rigid_body_paths}
    joint_lookup: dict[frozenset[str], set[str]] = {}
    for link in all_joint_links:
        adjacency.setdefault(link.body0_path, set()).add(link.body1_path)
        adjacency.setdefault(link.body1_path, set()).add(link.body0_path)
        joint_lookup.setdefault(frozenset((link.body0_path, link.body1_path)), set()).add(link.prim_path)

    meters_per_unit, up_axis = _stage_units(stage)
    xform_cache = UsdGeom.XformCache()
    source_path = str(params.get("radiusSourcePath") or "")

    source_roots: list[str] = []
    for link in all_joint_links:
        if link.body0_path == source_path and link.body1_path not in helper_body_paths:
            source_roots.append(link.body1_path)
        elif link.body1_path == source_path and link.body0_path not in helper_body_paths:
            source_roots.append(link.body0_path)

    components: list[_RodAttachmentComponent] = []
    visited_bodies: set[str] = set()
    for root_path in source_roots:
        if root_path in visited_bodies or root_path in helper_body_paths:
            continue

        body_paths, joint_paths = _collect_component_paths(
            root_path,
            adjacency=adjacency,
            joint_lookup=joint_lookup,
            blocked_paths=helper_body_paths,
        )
        if not body_paths:
            continue

        component_points: list[np.ndarray] = []
        proxy_bounds: dict[str, tuple[wp.vec3, wp.vec3]] = {}
        for body_path in sorted(body_paths):
            prim = stage.GetPrimAtPath(body_path)
            points_world = _read_rigid_body_points(
                stage,
                prim,
                meters_per_unit=meters_per_unit,
                up_axis=up_axis,
                xform_cache=xform_cache,
                read_mesh_world_points=_read_mesh_world_points,
                prefer_colliders=True,
            )
            if points_world.size > 0:
                component_points.append(points_world)
                body_world = _rigid_transform_to_newton(
                    xform_cache.GetLocalToWorldTransform(prim),
                    meters_per_unit=meters_per_unit,
                    up_axis=up_axis,
                    to_newton_world=_to_newton_world,
                    matrix_transform_points=_matrix_transform_points,
                )
                bounds = _component_proxy_bounds(points_world, body_world)
                if bounds is not None:
                    proxy_bounds[body_path] = bounds
        merged_points = (
            np.concatenate(component_points, axis=0)
            if component_points
            else np.empty((0, 3), dtype=np.float64)
        )
        endpoint_name = _component_endpoint_name(points, merged_points)
        relative_xform, parent_xform, child_xform = _component_relative_xform(
            stage=stage,
            root_body_path=root_path,
            endpoint_name=endpoint_name,
            points=points,
            quaternions=quaternions,
            rod_radius=float(params["radius"]),
            meters_per_unit=meters_per_unit,
            up_axis=up_axis,
            xform_cache=xform_cache,
            to_newton_world=_to_newton_world,
            matrix_transform_points=_matrix_transform_points,
            read_mesh_world_points=_read_mesh_world_points,
        )
        components.append(
            _RodAttachmentComponent(
                root_body_path=root_path,
                body_paths=body_paths,
                joint_paths=joint_paths,
                endpoint_name=endpoint_name,
                relative_xform=relative_xform,
                parent_xform=parent_xform,
                child_xform=child_xform,
                proxy_bounds=proxy_bounds,
            )
        )
        visited_bodies.update(body_paths)

    remaining_body_paths = rigid_body_paths - helper_body_paths - visited_bodies
    remaining_joint_paths = {
        link.prim_path
        for link in all_joint_links
        if link.body0_path in remaining_body_paths and link.body1_path in remaining_body_paths
    }
    all_joint_paths = {link.prim_path for link in all_joint_links}
    return components, remaining_body_paths, remaining_joint_paths, rigid_body_paths, all_joint_paths


def _build_rod(usd_path: str, *, device: str | None = None) -> _RodBuildResult:
    """Build an isotropic rod model from a NewtonRodAPI-authored USDA."""
    from .rod import read_rod_params

    params = read_rod_params(usd_path)
    points = params.get("points") or []
    if len(points) < 3:
        raise RuntimeError(f"Rod asset requires at least 3 centerline points: {usd_path}")

    positions = [wp.vec3(float(x), float(y), float(z)) for x, y, z in points]
    quaternions = newton.utils.create_parallel_transport_cable_quaternions(positions)
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
    components, remaining_body_paths, remaining_joint_paths, all_body_paths, all_joint_paths = _plan_rod_rigid_imports(
        usd_path,
        params,
        points_np,
        quaternions,
    )

    with wp.ScopedDevice(device) if device else wp.ScopedDevice(wp.get_preferred_device()):
        builder = newton.ModelBuilder()
        builder.rigid_gap = max(float(params["radius"]) * 0.25, 1.0e-4)
        builder.add_ground_plane()
        rod_bodies, _ = builder.add_rod(
            positions=positions,
            quaternions=quaternions,
            radius=float(params["radius"]),
            stretch_stiffness=float(params["stretchStiffness"]),
            stretch_damping=float(params["stretchDamping"]),
            bend_stiffness=float(params["bendStiffness"]),
            bend_damping=float(params["bendDamping"]),
            label=label,
        )
        _filter_body_self_collisions(builder, rod_bodies)
        attached_component_bodies: list[tuple[int, list[int]]] = []

        for component in components:
            ignore_paths = _subset_ignore_paths(
                all_body_paths=all_body_paths,
                all_joint_paths=all_joint_paths,
                keep_body_paths=set(component.body_paths),
                keep_joint_paths=set(component.joint_paths),
                extra_ignored_paths=extra_ignored_paths,
            )
            parent_body = rod_bodies[0] if component.endpoint_name == "start" else rod_bodies[-1]
            parent_world = (
                wp.transform(positions[0], quaternions[0])
                if component.endpoint_name == "start"
                else wp.transform(positions[-2], quaternions[-1])
            )
            root_relative_xform = wp.transform(*component.relative_xform)
            world_xform = parent_world * root_relative_xform
            parent_anchor_xform = wp.transform(*component.parent_xform)
            child_anchor_xform = wp.transform(*component.child_xform)
            parent_articulation = builder._find_articulation_for_body(parent_body)
            joints_before = len(builder.joint_type)
            articulations_before = len(builder.articulation_start)
            shapes_before = len(builder.shape_flags)
            result = builder.add_usd(
                usd_path,
                xform=(world_xform.p, world_xform.q),
                floating=False,
                skip_mesh_approximation=True,
                enable_self_collisions=False,
                hide_collision_shapes=True,
                ignore_paths=ignore_paths,
            )
            component_shape_indices = range(shapes_before, len(builder.shape_flags))
            path_body_map = result.get("path_body_map", {})
            root_body = path_body_map.get(component.root_body_path)
            if root_body is None:
                raise RuntimeError(f"Failed to import rod connector root body: {component.root_body_path}")
            component_body_indices = [
                int(body_idx)
                for body_path in sorted(component.body_paths)
                if (body_idx := path_body_map.get(body_path)) is not None
            ]
            body_frame_offsets = _recenter_body_frames(builder, component_body_indices)
            root_body_offset = body_frame_offsets.get(int(root_body))
            if root_body_offset is not None:
                child_anchor_xform = _shift_transform_translation(child_anchor_xform, root_body_offset)
            _hide_oversized_connector_visuals(builder, component_body_indices)
            attached_component_bodies.append((int(parent_body), component_body_indices))
            joints_after = len(builder.joint_type)
            new_joint_indices = list(range(joints_before, joints_after))
            root_joint_idx = next(
                (
                    joint_idx
                    for joint_idx in new_joint_indices
                    if int(builder.joint_child[joint_idx]) == int(root_body)
                    and int(builder.joint_parent[joint_idx]) == -1
                ),
                None,
            )
            if root_joint_idx is None:
                root_joint_idx = builder.add_joint_fixed(
                    parent=parent_body,
                    child=int(root_body),
                    parent_xform=parent_anchor_xform,
                    child_xform=child_anchor_xform,
                    label=f"{component.root_body_path}__rod_attach",
                )
                new_joint_indices.append(root_joint_idx)
            else:
                if int(builder.joint_type[root_joint_idx]) != int(newton.JointType.FIXED):
                    raise RuntimeError(
                        f"Rod connector root joint must be fixed, got {builder.joint_type[root_joint_idx]}: "
                        f"{component.root_body_path}"
                    )
                _reparent_existing_joint(
                    builder,
                    joint_idx=root_joint_idx,
                    parent_body=parent_body,
                    parent_xform=parent_anchor_xform,
                    child_xform=child_anchor_xform,
                    label=f"{component.root_body_path}__rod_attach",
                )

            if parent_articulation is not None:
                for joint_idx in new_joint_indices:
                    builder.joint_articulation[joint_idx] = parent_articulation
            if len(builder.articulation_start) > articulations_before:
                del builder.articulation_start[articulations_before:]
                del builder.articulation_label[articulations_before:]
                del builder.articulation_world[articulations_before:]
            _untint_textured_shapes(builder)
            _disable_shape_collisions(builder, component_shape_indices)
            if component.proxy_bounds:
                proxy_cfg = newton.ModelBuilder.ShapeConfig(
                    density=0.0,
                    gap=0.0,
                    has_shape_collision=True,
                    has_particle_collision=False,
                    is_visible=False,
                    mu=1.0,
                    collision_group=1,
                )
                proxy_shape_indices: list[int] = []
                for body_path in sorted(component.proxy_bounds):
                    body_idx = path_body_map.get(body_path)
                    if body_idx is None:
                        continue
                    proxy_center, proxy_half_extents = component.proxy_bounds[body_path]
                    body_offset = body_frame_offsets.get(int(body_idx))
                    if body_offset is not None:
                        proxy_center = wp.vec3(
                            float(proxy_center[0]) - float(body_offset[0]),
                            float(proxy_center[1]) - float(body_offset[1]),
                            float(proxy_center[2]) - float(body_offset[2]),
                        )
                    proxy_shape_indices.append(
                        builder.add_shape_box(
                            int(body_idx),
                            xform=wp.transform(proxy_center, wp.quat_identity()),
                            hx=float(proxy_half_extents[0]),
                            hy=float(proxy_half_extents[1]),
                            hz=float(proxy_half_extents[2]),
                            cfg=proxy_cfg,
                            label=f"{body_path}__contact_proxy",
                        )
                    )
                for i, shape_a in enumerate(proxy_shape_indices):
                    for shape_b in proxy_shape_indices[i + 1:]:
                        builder.add_shape_collision_filter_pair(shape_a, shape_b)
                _filter_shapes_against_bodies(builder, proxy_shape_indices, rod_bodies)

        if remaining_body_paths or remaining_joint_paths:
            ignore_paths = _subset_ignore_paths(
                all_body_paths=all_body_paths,
                all_joint_paths=all_joint_paths,
                keep_body_paths=set(remaining_body_paths),
                keep_joint_paths=set(remaining_joint_paths),
                extra_ignored_paths=extra_ignored_paths,
            )
            builder.add_usd(
                usd_path,
                skip_mesh_approximation=True,
                ignore_paths=ignore_paths,
            )
            _untint_textured_shapes(builder)

        initial_body_q = _set_rod_zero_curvature_rest_poses(
            builder,
            rod_bodies=rod_bodies,
            points=points_np,
            attached_components=attached_component_bodies,
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


def load(usd_path: str, *, solver_override: str | None = None,
         device: str | None = None, fix_base: bool = False) -> NewtonBundle:
    """Read converted USD, build Newton model + solver, return ready-to-step bundle.

    device: warp device string ("cuda:0", "cpu", ...). Defaults to GPU if
    available (wp.get_preferred_device()).
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

    rod_initial_body_q = None
    if body_type == "cloth":
        model = _build_cloth(usd_path, device=device, solver_name=solver_name)
    elif body_type == "rod":
        rod_result = _build_rod(usd_path, device=device)
        model = rod_result.model
        rod_initial_body_q = rod_result.initial_body_q
    else:
        model = _build_rigid(usd_path, device=device, fix_base=fix_base)

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
