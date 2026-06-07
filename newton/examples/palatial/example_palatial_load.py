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
# Rod assets get the lightweight cable-twist flow: body 0 is pinned kinematic
# and the rest of the cable falls under gravity. Drop height is configurable
# via --drop-height and `sim_substeps` is read from --physics-json. For the
# heavy isotropic-rod twist demo (endpoint twist boundary, connector clusters,
# rod gain overrides, textured tube), use example_palatial_isotropic.py.
#
# Usage:
#   python -m newton.examples.palatial.example_palatial_load <converted.usda>
#       [--steps 600] [--gui] [--substeps N]

from __future__ import annotations

import argparse
import math
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
                 contact_update_interval: int | None = None,
                 cuda_graph: bool = True,
                 step_diagnostics: bool = False,
                 zero_gravity: bool = False,
                 gravity_scale: float = 1.0,
                 cloth_particle_radius: float | None = None,
                 soft_contact_ke: float = 100.0,
                 soft_contact_kd: float = 2e-3,
                 soft_contact_mu: float = 0.25,
                 soft_contact_max: int = 1_000_000,
                 cloth_body_contact_margin: float = 0.01,
                 bending_ke: float | None = None,
                 bending_kd: float | None = None,
                 vbd_particle_edge_contact_buffer_size: int = 64,
                 vbd_particle_collision_detection_interval: int = -1,
                 vbd_rigid_contact_k_start: float | None = None,
                 vbd_particle_vertex_contact_buffer_size: int = 16,
                 vbd_particle_topological_contact_filter_threshold: int = 1,
                 vbd_particle_rest_shape_contact_exclusion_radius: float = 0.005,
                 preset: bool = False,
                 cloth_zero_edge_rest_angle: bool = False,
                 use_collision_pipeline: bool = False,
                 joint_q_overrides: dict[int, float] | None = None,
                 joint_target_overrides: dict[int, float] | None = None,
                 joint_target_ke: float | None = None,
                 joint_target_kd: float | None = None,
                 list_joints: bool = False,
                 rotate_x_deg: float = 0.0,
                 rotate_y_deg: float = 0.0,
                 rotate_z_deg: float = 0.0,
                 table: dict | None = None):
        self.viewer = viewer
        self.graph = None
        self._cuda_graph_requested = bool(cuda_graph)
        self._step_diagnostics = bool(step_diagnostics)
        self._table = table

        # One call: parse USDA, build model, build solver with baked params.
        bundle = load(
            usd_path,
            device=device,
            table=table,
        )
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

        def _rebuild_solver_with_params(params: dict) -> None:
            import inspect as _inspect
            from newton._src.palatial.load import (
                _SOLVER_PARAM_ALIAS as _alias,
                _dedupe_solver_params as _dedupe,
            )

            params = _dedupe(params)
            try:
                accepted = set(_inspect.signature(type(self.solver).__init__).parameters)
            except (TypeError, ValueError):
                accepted = set()
            kwargs = {}
            for key, value in params.items():
                py_key = _alias.get(key, key)
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

        #Optional rotation of the asset around world axes
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
            # When unset, keep USDA's newton:shell:particleRadius (loaded
            # into model.particle_radius by the loader).
            if cloth_particle_radius is not None:
                import numpy as _np
                n_particles = int(self.model.particle_count)
                if n_particles > 0:
                    self.model.particle_radius.assign(
                        _np.full(n_particles, float(cloth_particle_radius), dtype=_np.float32)
                    )
            self.model.cloth_body_contact_margin = cloth_body_contact_margin

            # VBD/Style3D read per-edge bend params from
            # model.edge_bending_properties ([edge_count, 2] = [ke, kd]).
            # model.bending_ke is a dead attribute. Only touch columns the
            # user explicitly set; the rest keep USDA's bendStiffness /
            # bendDamping forwarded via add_cloth_mesh.
            if (bending_ke is not None or bending_kd is not None) and getattr(
                self.model, "edge_bending_properties", None
            ) is not None:
                _bp = self.model.edge_bending_properties.numpy().copy()
                if _bp.size > 0:
                    if bending_ke is not None:
                        _bp[:, 0] = float(bending_ke)
                    if bending_kd is not None:
                        _bp[:, 1] = float(bending_kd)
                    self.model.edge_bending_properties.assign(_bp)
                    print(
                        "  cloth: edge_bending_properties overridden "
                        f"(bend_ke={bending_ke}, bend_kd={bending_kd}, "
                        f"edges={_bp.shape[0]})"
                    )

            # Flatten the cloth rest shape so the asset drapes naturally on a
            # surface instead of curling toward its authored rest angle.
            # This matches example_cloth_gown_franka.
            if cloth_zero_edge_rest_angle and hasattr(self.model, "edge_rest_angle"):
                try:
                    self.model.edge_rest_angle.zero_()
                    print("  cloth: edge_rest_angle zeroed (flat rest shape)")
                except Exception as _e:
                    print(f"  cloth: edge_rest_angle.zero_() failed: {_e}")

            # VBD solver extras (no-op for other solvers).
            if type(self.solver).__name__ == "SolverVBD":
                self.solver.particle_edge_contact_buffer_size = vbd_particle_edge_contact_buffer_size
                self.solver.particle_collision_detection_interval = vbd_particle_collision_detection_interval
                self.solver.rigid_contact_k_start = (
                    soft_contact_ke if vbd_rigid_contact_k_start is None
                    else vbd_rigid_contact_k_start
                )

                # VBD __init__-only knobs need a solver rebuild. Buffer
                # knobs are always applied; self-contact overrides only
                # under preset.
                import inspect as _ins
                from newton._src.palatial.load import (
                    _SOLVER_PARAM_ALIAS,
                    _dedupe_solver_params,
                )
                vbd_overrides: dict = {
                    "particle_vertex_contact_buffer_size": int(vbd_particle_vertex_contact_buffer_size),
                    "particle_topological_contact_filter_threshold": int(vbd_particle_topological_contact_filter_threshold),
                    "particle_rest_shape_contact_exclusion_radius": float(vbd_particle_rest_shape_contact_exclusion_radius),
                }
                if preset:
                    vbd_overrides.update({
                        "particle_enable_self_contact": True,
                        "particle_self_contact_radius": 0.002,
                        "particle_self_contact_margin": 0.002,
                    })
                # Strip USDA equivalents (camel + snake) before injecting.
                base = dict(bundle.solver_params)
                _snake_to_camel = {v: k for k, v in _SOLVER_PARAM_ALIAS.items() if k != v}
                for snake_k in vbd_overrides:
                    base.pop(snake_k, None)
                    camel = _snake_to_camel.get(snake_k)
                    if camel:
                        base.pop(camel, None)
                base.update(vbd_overrides)
                base = _dedupe_solver_params(base)
                try:
                    accepted = set(_ins.signature(type(self.solver).__init__).parameters)
                except (TypeError, ValueError):
                    accepted = set(base)
                kwargs_vbd = {}
                for k, v in base.items():
                    py_key = _SOLVER_PARAM_ALIAS.get(k, k)
                    if py_key in accepted:
                        kwargs_vbd[py_key] = v
                self.solver = type(self.solver)(self.model, **kwargs_vbd)
                bundle.solver = self.solver
                bundle.solver_params = base
                # Re-apply the per-solver knobs that live outside __init__.
                self.solver.particle_edge_contact_buffer_size = vbd_particle_edge_contact_buffer_size
                self.solver.particle_collision_detection_interval = vbd_particle_collision_detection_interval
                self.solver.rigid_contact_k_start = (
                    soft_contact_ke if vbd_rigid_contact_k_start is None
                    else vbd_rigid_contact_k_start
                )
                if preset:
                    print(f"  gown-preset: VBD self-contact applied {vbd_overrides}")
                else:
                    print(f"  cloth: VBD buffer knobs applied {vbd_overrides}")

        # Rod assets: pin body 0 kinematic so one end is clamped while gravity
        # drops the rest of the cable (mirrors example_palatial_cable_twist).
        # The solver caches body_inv_mass_effective at construction time via
        # Solver._init_kinematic_state(); after we zero the per-body mass
        # arrays here we MUST refresh that cache or the solver will keep
        # integrating body 0 with its original mass and the clamp will fail.
        if bundle.body_type == "rod" and int(self.model.body_count) > 0:
            _set_bodies_kinematic(self.model, [0])
            if hasattr(self.solver, "_refresh_kinematic_state"):
                self.solver._refresh_kinematic_state()
            _sync_body_pose_teleport()
            print("  rod: pinned body 0 kinematic (clamp + free-fall drop)")

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

        if contact_update_interval is None:
            contact_update_interval = (
                min(10, self.sim_substeps)
                if bundle.body_type == "rod" and hasattr(self.solver, "set_rigid_history_update")
                else 1
            )
        if int(contact_update_interval) <= 0:
            raise ValueError(f"contact_update_interval must be > 0, got {contact_update_interval}")
        self.contact_update_interval = int(contact_update_interval)
        if self.contact_update_interval > 1 and not hasattr(self.solver, "set_rigid_history_update"):
            print(
                "  warn: contact-update-interval > 1 requires solver.set_rigid_history_update(); "
                "falling back to 1"
            )
            self.contact_update_interval = 1

        self.collision_pipeline = None
        if use_collision_pipeline and bundle.body_type == "cloth":
            try:
                self.collision_pipeline = newton.CollisionPipeline(
                    self.model,
                    soft_contact_margin=float(cloth_body_contact_margin),
                )
                self.contacts = self.collision_pipeline.contacts()
                self.collision_pipeline.collide(self.state_0, self.contacts)
                print(
                    "  cloth: using explicit CollisionPipeline "
                    f"(soft_contact_margin={cloth_body_contact_margin})"
                )
            except Exception as _e:
                print(f"  cloth: CollisionPipeline setup failed ({_e}); falling back to model.contacts()")
                self.collision_pipeline = None

        if self.collision_pipeline is None:
            self.contacts = self.model.contacts()
            self.model.collide(self.state_0, self.contacts)
        self.viewer.set_model(self.model)

        print(
            f"[load] usd={usd_path}  body={bundle.body_type}  "
            f"solver={bundle.solver_name}  fps={self.fps}  substeps={self.sim_substeps}  "
            f"contact_update_interval={self.contact_update_interval}  "
            f"params={bundle.solver_params}"
        )
        self.capture()

    def capture(self):
        self.graph = None
        if not self._cuda_graph_requested:
            return
        solver_device = getattr(self.solver, "device", None)
        if solver_device is None or not solver_device.is_cuda:
            return
        try:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
            print("  cuda-graph: enabled")
        except Exception as exc:
            self.graph = None
            print(f"  cuda-graph: disabled ({exc})")

    def simulate(self):
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            refresh_contacts = (substep % self.contact_update_interval) == 0
            if refresh_contacts:
                if self.collision_pipeline is not None:
                    self.collision_pipeline.collide(self.state_0, self.contacts)
                else:
                    self.model.collide(self.state_0, self.contacts)
            if hasattr(self.solver, "set_rigid_history_update"):
                self.solver.set_rigid_history_update(refresh_contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self._frame_idx = getattr(self, "_frame_idx", -1) + 1
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        if not self._step_diagnostics:
            return
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
    p.add_argument("--contact-update-interval", type=int, default=None,
                   help="Refresh contacts every N simulation substeps. "
                        "Default: rods use one refresh per rendered frame, other assets use 1.")
    p.add_argument("--no-cuda-graph", action="store_true",
                   help="Disable CUDA graph capture and run the Python substep loop each frame.")
    p.add_argument("--step-diagnostics", action="store_true",
                   help="Print per-frame z diagnostics. Disabled by default because it syncs GPU state.")
    p.add_argument("--gui", action="store_true",
                   help="Open ViewerGL and run until the window is closed")
    p.add_argument("--drop-height", type=float, default=0.0,
                   help="Lift rigid or rod bodies by this many meters in z before sim "
                        "(ignored for cloth — cloth drop height is baked into the USDA)")
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
    p.add_argument("--cloth-particle-radius", type=float, default=None,
                   help="Override model.particle_radius. When unset, the loader's "
                        "USDA-derived value (newton:shell:particleRadius) wins.")
    p.add_argument("--soft-contact-ke", type=float, default=100.0)
    p.add_argument("--soft-contact-kd", type=float, default=2e-3)
    p.add_argument("--soft-contact-mu", type=float, default=0.25)
    p.add_argument("--soft-contact-max", type=int, default=1_000_000)
    p.add_argument("--cloth-body-contact-margin", type=float, default=0.01)
    p.add_argument("--bending-ke", type=float, default=None,
                   help="Override edge bend stiffness for ALL edges (overwrites "
                        "model.edge_bending_properties[:,0]). Omit to keep the "
                        "USDA-authored bendStiffness value.")
    p.add_argument("--bending-kd", type=float, default=None,
                   help="Override edge bend damping for ALL edges (overwrites "
                        "model.edge_bending_properties[:,1]). Omit to keep the "
                        "USDA-authored bendDamping value.")

    # VBD-only knobs (ignored for other solvers).
    p.add_argument("--vbd-particle-edge-contact-buffer-size", type=int, default=64)
    p.add_argument("--vbd-particle-collision-detection-interval", type=int, default=-1)
    p.add_argument("--vbd-rigid-contact-k-start", type=float, default=None,
                   help="Defaults to --soft-contact-ke when omitted")
    # VBD __init__-only knobs. Defaults match the gown preset; these always
    # overwrite USDA-authored equivalents (which is intentional — we want a
    # predictable contact-buffer sizing across assets).
    p.add_argument("--vbd-particle-vertex-contact-buffer-size", type=int, default=16)
    p.add_argument("--vbd-particle-topological-contact-filter-threshold", type=int, default=1)
    p.add_argument("--vbd-particle-rest-shape-contact-exclusion-radius", type=float, default=0.005)
    # particle_enable_self_contact / radius / margin remain preset-only knobs
    # because the USDA usually authors them and we don't want to silently
    # clobber the asset's authored values from a CLI default. Use --gown-preset
    # to apply the gown self-contact recipe.
    p.add_argument("--cloth-zero-edge-rest-angle", action="store_true", default=True,
                   help="Zero edge rest angles so the cloth drapes flat (matches gown_franka).")
    p.add_argument("--use-collision-pipeline", action="store_true", default=True,
                   help="Build an explicit newton.CollisionPipeline so soft_contact_margin\n"
                        "is honored (matches example_cloth_gown_franka).")
    p.add_argument("--preset", action="store_true", default=False,
                   help="Cloth + table preset that mirrors example_cloth_gown_franka in meter scale:\n"
                        "sets soft-contact stiffness, bending, self-contact, particle radius,\n"
                        "explicit CollisionPipeline, and flat rest shape. Overrides values you did\n"
                        "NOT set on the command line; values you pass explicitly win.")

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
    p.add_argument("--rotate-x", type=float, default=None,
                   help="Rotate asset around X (degrees). Defaults to 90 when the "
                        "loaded USDA is detected as a rod, else 0.")
    p.add_argument("--rotate-y", type=float, default=0.0, help="Rotate asset around Y (degrees)")
    p.add_argument("--rotate-z", type=float, default=0.0, help="Rotate asset around Z (degrees)")

    # Cloth-only: add a static table under the cloth so it drops onto a surface.
    p.add_argument("--add-table", action=argparse.BooleanOptionalAction, default=None,
                   help="Cloth only: add a static box (table) under the asset for the drop. "
                        "Also overrides the camera. "
                        "Defaults to ON when the loaded USDA is detected as cloth; pass "
                        "--no-add-table to disable.")
    p.add_argument("--table-size", default="1.0,1.0,0.1",
                   help="Table half-extents in meters as 'hx,hy,hz' ")
    p.add_argument("--table-pos", default="0.0,0.0,0.1",
                   help="Table center position in meters as 'x,y,z' ")
    p.add_argument("--cloth-scale", type=float, default=None,
                   help="Uniform mesh scale applied to the cloth via add_cloth_mesh.")

    # mp4 recording (front-facing camera).
    p.add_argument("--record-mp4", default=None,
                   help="Output mp4 path. Uses ViewerGL (headless unless --gui) "
                        "and pipes frames to ffmpeg with a front-facing camera.")
    p.add_argument("--mp4-fps", type=int, default=60,
                   help="Output mp4 framerate (default 60)")
    p.add_argument("--top-view", action="store_true",
                   help="Place camera straight above the asset looking down")
    p.add_argument("--front-view", action="store_true",
                   help="Force the front-facing auto camera (default for rod assets; "
                        "overrides --top-view if both are passed).")
    p.add_argument("--cable-axis", choices=["x", "y", "z"], default="x",
                   help="Rod centerline axis hint (informational only; the asset's "
                        "authored BasisCurves centerline is always used).")
    p.add_argument("--no-auto-camera", action="store_true",
                   help="Skip the front-view auto-framer so the viewer keeps its default "
                        "camera (useful when rotating the asset and you want the view "
                        "to stay constant).")
    args = p.parse_args(argv)

    # ----  preset for cloth-on-table drops --------------------
    # Mirrors example_cloth_gown_franka in meter scalee.
    if args.preset:
        _defaults = {a.dest: a.default for a in p._actions}

        def _maybe(name: str, value):
            cur = getattr(args, name, None)
            if cur == _defaults.get(name):
                setattr(args, name, value)

        _maybe("cloth_particle_radius", 0.008)            
        _maybe("cloth_body_contact_margin", 0.008)        
        _maybe("soft_contact_ke", 1.0e4)
        _maybe("soft_contact_kd", 1.0e-2)
        _maybe("soft_contact_mu", 0.25)                  
        _maybe("bending_ke", 5.0e-3)                      
        _maybe("bending_kd", 1.0e-5)                      
        _maybe("vbd_particle_edge_contact_buffer_size", 20)
        _maybe("cloth_zero_edge_rest_angle", True)
        _maybe("use_collision_pipeline", True)
        _maybe("add_table", True)
        if args.substeps is None:
            args.substeps = 10
        if args.solver_iterations is None:
            args.solver_iterations = 32
        print(
            "  gown-preset: applied (only fields not set on CLI were overwritten)"
        )

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
        _scene = None
        if isinstance(_phys.get("newton"), dict):
            _scene = _phys["newton"].get("simulation_scene")
        # Walk parts to fill in simulation_scene when it isn't at the top
        # level (cloth-only assets author it under each part).
        for _part in _phys.get("parts", []) or []:
            if _scene is not None:
                break
            _newton = _part.get("newton", {}) if isinstance(_part, dict) else {}
            if isinstance(_newton.get("simulation_scene"), dict):
                _scene = _newton["simulation_scene"]
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
        print(
            f"  physics-json: {args.physics_json}  "
            f"cloth_particle_radius={args.cloth_particle_radius}  "
            f"bending_ke={args.bending_ke}  "
            f"substeps={args.substeps}  "
            f"solver_iterations={args.solver_iterations}"
        )

    def _parse_vec3(s: str) -> tuple[float, float, float]:
        parts = [float(x) for x in s.split(",")]
        if len(parts) != 3:
            raise ValueError(f"expected 3 comma-separated floats, got {s!r}")
        return (parts[0], parts[1], parts[2])

   
    _bt: str | None = None
    if args.add_table is None:
        try:
            from pxr import Usd as _Usd
            from newton._src.palatial.load import _detect_body_type as _detect
            _stage = _Usd.Stage.Open(args.usd)
            _bt = _detect(_stage) if _stage is not None else "rigid"
        except Exception as _e:
            print(f"  warn: body-type auto-detect failed ({_e}); defaulting to rigid")
            _bt = "rigid"
        args.add_table = (_bt == "cloth")
        if args.add_table:
            print("  --add-table: auto-enabled (detected cloth asset)")
        if _bt == "rod" and not args.top_view:
            args.front_view = True
            print("  --front-view: auto-enabled (detected rod asset)")
    if args.rotate_x is None:
        args.rotate_x = 0.0

    table_cfg: dict | None = None
    if args.add_table:
        # When --add-table is on, keep the cloth at real-world size (1.0).
        # The default --table-size is sized to hold a full ~1.9 m gown.
        cloth_scale = args.cloth_scale if args.cloth_scale is not None else 1.0
        table_cfg = {
            "pos": _parse_vec3(args.table_pos),
            "size": _parse_vec3(args.table_size),
            "cloth_scale": float(cloth_scale),
        }

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
                 preset=args.preset,
                 vbd_particle_vertex_contact_buffer_size=args.vbd_particle_vertex_contact_buffer_size,
                 vbd_particle_topological_contact_filter_threshold=args.vbd_particle_topological_contact_filter_threshold,
                 vbd_particle_rest_shape_contact_exclusion_radius=args.vbd_particle_rest_shape_contact_exclusion_radius,
                 cloth_zero_edge_rest_angle=args.cloth_zero_edge_rest_angle,
                 use_collision_pipeline=args.use_collision_pipeline,
                 drop_height=args.drop_height, device=args.device,
                 solver_iterations=args.solver_iterations,
                 contact_update_interval=args.contact_update_interval,
                 cuda_graph=not args.no_cuda_graph,
                 step_diagnostics=args.step_diagnostics,
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
                 rotate_z_deg=args.rotate_z,
                 table=table_cfg)

    usd_bounds = _read_usd_geometry_bounds(args.usd)

    def _camera_bounds():
        # Prefer live simulation state so the camera reflects post-drop
        # particle positions and the actual Z-up axis, instead of the raw USD geometry bounds 
        import numpy as _np

        n_p = int(ex.model.particle_count)
        if n_p > 0:
            pts = ex.state_0.particle_q.numpy()
        elif int(ex.model.body_count) > 0:
            pts = ex.state_0.body_q.numpy()[:, 0:3]
        elif usd_bounds is not None:
            return usd_bounds
        else:
            return _np.zeros(3, dtype=float), _np.ones(3, dtype=float)
        return _np.asarray(pts.min(axis=0), dtype=float), _np.asarray(pts.max(axis=0), dtype=float)

    # Cloth + table: camera angles on an origin-centered table
    
    if (
        args.add_table
        and ex.bundle.body_type == "cloth"
        and (args.record_mp4 or args.gui)
        and not args.no_auto_camera
        and hasattr(ex.viewer, "set_camera")
    ):
        # With pitch=0 (horizontal look) the screen's vertical axis is world Z,
        # so set camera.z to the midpoint between the cloth's spawn top
        # (b_max[2] at state_0) and the table top (where it settles).
        # X distance scales with the asset's longest XY extent so big assets
        # are framed further out and small ones are zoomed in.
        b_min, b_max = _camera_bounds()
        spawn_top_z = float(b_max[2])
        if table_cfg is not None:
            table_top_z = float(table_cfg["pos"][2] + table_cfg["size"][2])
        else:
            table_top_z = float(b_min[2])
        mid_z = 0.75 * (spawn_top_z + table_top_z)
        ext_x = float(b_max[0] - b_min[0])
        ext_y = float(b_max[1] - b_min[1])
        reference_extent = 2.0  # ~1.9 m gown after the -pi/2 X rotation
        scale = max(max(ext_x, ext_y) / reference_extent, 0.5)
        cam_x = -3.6 * scale
        pos = wp.vec3(cam_x, 0.0, mid_z)
        ex.viewer.set_camera(pos, -25.0, 0.0)
        print(
            f"  table camera: pos=({cam_x:.3f},0.000,{mid_z:.3f}) "
            f"pitch=-25 yaw=0  (spawn_top={spawn_top_z:.3f} table_top={table_top_z:.3f}, "
            f"scale={scale:.2f})"
        )

    if ex.bundle.body_type == "rod" and args.cable_axis != "x":
        print(f"  cable-axis={args.cable_axis!r}: hint ignored — using authored "
              f"BasisCurves centerline from the USDA.")

    if args.add_table and ex.bundle.body_type == "cloth":
        # Already set above; skip the front/top auto-framers.
        pass
    elif args.top_view and not args.front_view and hasattr(ex.viewer, "set_camera"):
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
        # Frame by the AABB diagonal so the asset occupies the same fraction
        # of the frame regardless of physical size (fixed FOV → constant
        # angular size). Rod uses a tighter multiplier for a zoomed-in feel;
        # cloth needs a wider frame to show drape.
        diag = math.sqrt(ext_x * ext_x + ext_y * ext_y + ext_z * ext_z)
        diag = max(diag, 1e-3)
        if ex.bundle.body_type == "cloth":
            dist_mult = 2.8
        elif ex.bundle.body_type == "rod":
            dist_mult = 1.8
        else:
            dist_mult = 2.2
        dist = diag * dist_mult
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
