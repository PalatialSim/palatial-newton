"""Generic loader for Newton-ready USD assets produced by `framework.convert`.

Reads solver name + simulation params from the converted USD's PhysicsScene and
body-type markers, then constructs a Newton Model + Solver. Caller does not
need to know whether the asset is rigid or cloth, or which solver was baked in
— that information is read from the file.

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
# NewtonXpbdSceneAPI, ...) AND the bundled `newton_shell` plugin
# (NewtonShellAPI, NewtonClothAPI, NewtonRodAPI, NewtonDeformableAPI,
# NewtonShellMaterialAPI, NewtonRodMaterialAPI) via
# newton/_src/usd/__init__.py. Must precede any pxr.Usd usage in the same
# process.
import newton

from pxr import Gf, Usd, UsdGeom

from dataclasses import dataclass
from typing import Any
import warp as wp


def _build_solver(name: str, model: Any, params: dict) -> Any:
    """Construct a solver, forwarding only kwargs the solver actually accepts."""
    import inspect as _ins

    classes = {
        "mujoco":        getattr(newton.solvers, "SolverMuJoCo",       None),
        "xpbd":          getattr(newton.solvers, "SolverXPBD",         None),
        "featherstone":  getattr(newton.solvers, "SolverFeatherstone", None),
        "vbd":           getattr(newton.solvers, "SolverVBD",          None),
        "vbd_palatial":  getattr(newton.solvers, "SolverVBDPalatial",  None),
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


@dataclass
class NewtonBundle:
    """Everything needed to step the converted asset in Newton."""
    usd_path: str
    body_type: str           # "rigid", "cloth", or "cable"
    solver_name: str
    fps: int
    model: Any
    solver: Any
    state_in: Any
    state_out: Any
    control: Any
    solver_params: dict      # raw newton:solver:* attrs from the USD
    scene_kind: str | None = None

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


def _detect_scene_kind(stage: Usd.Stage) -> str:
    """Return the narrow scene kind for the Palatial loader.

    Three positive signals (any of):
      - v1 source assembly: Power cable assembly under /World/Geometry
      - new schema: NewtonShellAPI / NewtonClothAPI in applied schemas
      - new schema: NewtonRodAPI in applied schemas
      - new schema: newton:deformable:simulationIntent in {cloth, shell}
      - new schema: newton:deformable:simulationIntent == rod
      - legacy: newton:bodyType="cloth"
    """
    from .cable_assembly import is_power_cable_assembly_stage

    if is_power_cable_assembly_stage(stage):
        return "cable_assembly"

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
        if "NewtonRodAPI" in applied:
            return "cable"
        a = prim.GetAttribute("newton:deformable:simulationIntent")
        if a and a.HasAuthoredValue():
            intent = str(a.Get())
            if intent in ("cloth", "shell"):
                return "cloth"
            if intent == "rod":
                return "cable"
        a = prim.GetAttribute("newton:bodyType")
        if a and a.HasAuthoredValue() and str(a.Get()) == "cloth":
            return "cloth"
    return "rigid"


def _detect_body_type(stage: Usd.Stage) -> str:
    """Return the coarse body family: 'cloth', 'cable', or 'rigid'."""
    scene_kind = _detect_scene_kind(stage)
    return "cable" if scene_kind == "cable_assembly" else scene_kind


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

        # Workaround for Newton bug: ModelBuilder._shape_palette_color() is
        # used as the fallback display color when a USD material authors its
        # diffuseColor as a *texture connection* (UsdPreviewSurface +
        # UsdUVTexture) rather than a scalar value. The viewer's fragment
        # shader then computes `albedo = ObjectColor * texture`, tinting the
        # diffuse texture with the synthetic per-shape palette (cyan/green/
        # yellow). Override the color to white for any shape whose source
        # mesh has a texture, so the texture renders untinted.
        for i, src in enumerate(builder.shape_source):
            if src is not None and getattr(src, "texture", None) is not None:
                builder.shape_color[i] = (1.0, 1.0, 1.0)

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


def _build_cable(usd_path: str, *, device: str | None = None) -> Any:
    """Build a cable model from NewtonRodAPI metadata and a BasisCurves centerline."""
    from newton import utils as newton_utils

    from .cable import (
        create_cable_quaternions,
        extract_cable_points,
        find_cable_prim_path,
        get_cable_reference_transform_matrix,
        read_cable_params,
    )

    cable_path = find_cable_prim_path(usd_path)
    if not cable_path:
        raise RuntimeError(f"No cable or rod prim found in {usd_path}")

    params = read_cable_params(usd_path)
    points = extract_cable_points(usd_path, world_space=True)
    if str(params["frameDefinition"]) not in ("parallelTransport", "parallel_transport"):
        raise RuntimeError(
            f"Cable asset {usd_path} must declare newton:rod:frameDefinition='parallelTransport' "
            f"(got {params['frameDefinition']!r})"
        )
    if int(params["verticesPerSegment"]) != 2:
        raise RuntimeError(
            f"Cable asset {usd_path} must declare newton:rod:verticesPerSegment=2 for linear centerlines "
            f"(got {params['verticesPerSegment']})"
        )
    if not points:
        segment_count = int(params["segmentCount"])
        local_points = newton_utils.create_straight_cable_points(
            start=wp.vec3(0.0, 0.0, float(params["dropHeight"])),
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=float(params["length"]),
            num_segments=segment_count,
        )
        reference_transform = get_cable_reference_transform_matrix(usd_path)
        points = [
            wp.vec3(
                float(transformed[0]),
                float(transformed[1]),
                float(transformed[2]),
            )
            for point in local_points
            for transformed in (
                reference_transform.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))),
            )
        ]

    if len(points) < 2:
        raise RuntimeError(f"Cable asset {usd_path} must provide at least 2 centerline points")

    quaternions = create_cable_quaternions(
        points,
        cross_section_type=str(params["crossSectionType"]),
        twist_total=float(params["twistTotal"]),
    )
    cfg = newton.ModelBuilder.ShapeConfig(density=float(params["density"]))

    with wp.ScopedDevice(device) if device else wp.ScopedDevice(wp.get_preferred_device()):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        shape_type = "box" if str(params["crossSectionType"]) == "flatRect" else "capsule"
        builder.add_rod(
            positions=points,
            quaternions=quaternions,
            radius=float(params["radius"]),
            cfg=cfg,
            stretch_stiffness=float(params["stretchStiffness"]),
            stretch_damping=float(params["stretchDamping"]),
            bend_y_stiffness=float(params["bendYStiffness"]),
            bend_y_damping=float(params["bendYDamping"]),
            bend_z_stiffness=float(params["bendZStiffness"]),
            bend_z_damping=float(params["bendZDamping"]),
            torsion_stiffness=float(params["torsionStiffness"]),
            torsion_damping=float(params["torsionDamping"]),
            closed=bool(params["closed"]),
            label="cable",
            shape_type=shape_type,
            width=float(params["width"]),
            thickness=float(params["thickness"]),
        )
        try:
            builder.color()
        except Exception:
            pass
        return builder.finalize()


def _build_cable_assembly(usd_path: str, *, device: str | None = None) -> Any:
    """Build a cable assembly model from a v1 Power assembly source USDA."""
    from .cable_assembly import build_power_cable_assembly_model

    return build_power_cable_assembly_model(usd_path, device=device)


def _scene_has_authored_solver(stage: Usd.Stage) -> bool:
    """Return whether the first PhysicsScene pins a solver."""
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsScene":
            continue
        a = prim.GetAttribute("newton:solver")
        if not (a and a.HasAuthoredValue()):
            a = prim.GetAttribute("palatial:solver")
        return bool(a and a.HasAuthoredValue())
    return False


def _scene_has_authored_fps(stage: Usd.Stage) -> bool:
    """Return whether the first PhysicsScene pins timeStepsPerSecond."""
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsScene":
            continue
        a = prim.GetAttribute("newton:timeStepsPerSecond")
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
    scene_kind = _detect_scene_kind(stage)
    body_type = "cable" if scene_kind == "cable_assembly" else scene_kind

    # Defensive: if the scene didn't pin a solver but the asset is cloth,
    # pick a sensible default. VBD by default; Style3D when any
    # newton:shell:style3d:* attr is authored (those are Style3D-only).
    scene_pinned = _scene_has_authored_solver(stage)
    fps_pinned = _scene_has_authored_fps(stage)
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
    if body_type == "cable":
        cable_solver_name = solver_override or solver_name
        if scene_kind == "cable_assembly" and not fps_pinned:
            from .cable_assembly import (
                DEFAULT_ASSEMBLY_FPS,
                DEFAULT_ASSEMBLY_ITERATIONS,
                DEFAULT_ASSEMBLY_SUBSTEPS,
            )

            fps = DEFAULT_ASSEMBLY_FPS
            solver_params.setdefault("iterations", DEFAULT_ASSEMBLY_ITERATIONS)
            solver_params.setdefault("substeps", DEFAULT_ASSEMBLY_SUBSTEPS)
        if not scene_pinned and not solver_override:
            if scene_kind == "cable_assembly" and getattr(newton.solvers, "SolverVBDPalatial", None):
                solver_name = "vbd_palatial"
            elif getattr(newton.solvers, "SolverVBD", None):
                solver_name = "vbd"
            else:
                raise RuntimeError(
                    "Cable assets require SolverVBD or SolverVBDPalatial, but this Newton build does not provide it"
                )
        elif cable_solver_name not in ("vbd", "vbd_palatial"):
            raise RuntimeError(
                f"Cable assets require SolverVBD because cable joint runtime is only supported by VBD "
                f"(got solver '{cable_solver_name}')"
            )
    del stage

    if solver_override:
        solver_name = solver_override

    if device is None:
        device = str(wp.get_preferred_device())

    if body_type == "cloth":
        model = _build_cloth(usd_path, device=device, solver_name=solver_name)
    elif scene_kind == "cable_assembly":
        model = _build_cable_assembly(usd_path, device=device)
    elif body_type == "cable":
        model = _build_cable(usd_path, device=device)
    else:
        model = _build_rigid(usd_path, device=device, fix_base=fix_base)

    solver = _build_solver(solver_name, model, solver_params)

    return NewtonBundle(
        usd_path=usd_path,
        body_type=body_type,
        solver_name=solver_name,
        fps=fps,
        model=model,
        solver=solver,
        state_in=model.state(),
        state_out=model.state(),
        control=model.control(),
        solver_params=solver_params,
        scene_kind=scene_kind,
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
    print(f"scene kind : {b.scene_kind}")
    print(f"solver     : {b.solver_name}  ({type(b.solver).__name__})")
    print(f"fps / dt   : {b.fps}  /  {b.dt:.6f}")
    if b.solver_params:
        print("solver params:")
        for k, v in b.solver_params.items():
            print(f"  {k}: {v}")
    print(f"particles  : {int(b.model.particle_count)}")
    print(f"shapes     : {int(b.model.shape_count)}")
    print(f"bodies     : {int(b.model.body_count)}")
