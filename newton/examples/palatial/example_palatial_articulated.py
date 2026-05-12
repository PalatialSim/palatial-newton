# Example: load a converted articulated USDA and drive its joints.
#
# Mirrors the structure of newton/examples/basic/example_basic_joints.py, but
# the asset comes from the palatial converter instead of being authored
# programmatically with ModelBuilder.
#
# Usage:
#   python -m newton.examples.palatial.example_palatial_articulated <converted.usda> [--gui]
#       [--substeps N]
#       [--drive-joint i]                    # which joint to drive (default: first revolute)
#       [--drive-amplitude RAD]
#       [--drive-frequency HZ]
#       [--joint-target-ke FLOAT --joint-target-kd FLOAT]
#       [--steps 600]                        # for headless mode
#
#   # Record a textured USD playback for usdview / Composer / Blender:
#   python -m newton.examples.palatial.example_palatial_articulated <converted.usda> \
#       --use-usd-viewer --usd-out out.usda --steps 600
#
# Driving uses control.joint_target updated every frame with a sine wave.
# Set --drive-amplitude 0 to disable driving and just watch the asset settle.
from __future__ import annotations

import argparse
import math
import sys

# Newton stack must import before any pxr.Usd usage in the same process.
import warp as wp  # noqa: F401
import newton

from newton.palatial import load


JOINT_TYPE_NAMES = {
    0: "PRISMATIC", 1: "REVOLUTE", 2: "BALL",
    3: "FIXED", 4: "FREE", 5: "DISTANCE", 6: "D6",
}


class Example:
    def __init__(self, viewer, usd_path: str, *, substeps: int | None = None,
                 device: str | None = None,
                 drive_joint: int | None = None,
                 drive_amplitude: float = 0.7,
                 drive_frequency: float = 0.5,
                 joint_target_ke: float | None = None,
                 joint_target_kd: float | None = None,
                 rotate_x_deg: float = 0.0,
                 rotate_y_deg: float = 0.0,
                 rotate_z_deg: float = 0.0):
        self.viewer = viewer

        # Articulated assets always anchor their root via a FIXED joint to
        # world (basic_joints pattern). MuJoCo then sees fixed_base=True.
        bundle = load(usd_path, device=device, fix_base=True)
        self.bundle = bundle
        self.model = bundle.model
        self.solver = bundle.solver
        self.state_0 = bundle.state_in
        self.state_1 = bundle.state_out
        self.control = bundle.control

        if int(self.model.particle_count) > 0:
            raise RuntimeError("This example is for rigid articulated assets only "
                               "(asset has cloth particles).")

        # Sim timing comes from the bundle (newton:timeStepsPerSecond).
        self.fps = bundle.fps
        self.frame_dt = 1.0 / float(self.fps)
        self.sim_substeps = max(1, int(substeps) if substeps is not None
                                else int(bundle.solver_params.get("substeps", 1)))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # Print joint table (basic_joints style).
        n_joints = int(self.model.joint_count)
        if n_joints == 0:
            raise RuntimeError("Asset has no joints — nothing to articulate.")

        jtypes = self.model.joint_type.numpy()
        jq_start = self.model.joint_q_start.numpy()
        jqd_start = self.model.joint_qd_start.numpy()
        print("  joints:")
        revolutes = []
        for i in range(n_joints):
            t = int(jtypes[i])
            tn = JOINT_TYPE_NAMES.get(t, str(t))
            print(f"    [{i}] type={tn}  q_start={int(jq_start[i])}  qd_start={int(jqd_start[i])}")
            if t in (0, 1):  # PRISMATIC or REVOLUTE
                revolutes.append(i)

        # Pick driven joint: caller override > first revolute/prismatic > none.
        if drive_joint is not None:
            self.drive_joint = int(drive_joint)
        elif revolutes:
            self.drive_joint = revolutes[0]
        else:
            self.drive_joint = -1
        self.drive_amp = float(drive_amplitude)
        self.drive_freq = float(drive_frequency)
        if self.drive_joint >= 0 and self.drive_amp != 0.0:
            print(f"  driving joint [{self.drive_joint}] sine amp={self.drive_amp} freq={self.drive_freq}Hz")

        # Track whether we mutate any model attribute that MuJoCo bakes in
        # at solver construction (joint_target_ke/kd, body_inv_mass, ...).
        # We rebuild once at the end if so.
        model_dirty = False

        # Optional PD gain bump so target tracking is visible.
        if joint_target_ke is not None and hasattr(self.model, "joint_target_ke"):
            ke = self.model.joint_target_ke.numpy().copy()
            ke[:] = float(joint_target_ke)
            self.model.joint_target_ke.assign(ke)
            model_dirty = True
        if joint_target_kd is not None and hasattr(self.model, "joint_target_kd"):
            kd = self.model.joint_target_kd.numpy().copy()
            kd[:] = float(joint_target_kd)
            self.model.joint_target_kd.assign(kd)
            model_dirty = True

        # Optional asset rotation. With fix_base, root joint is FIXED-to-world
        # (parent=-1) so rotating joint_q won't move the chain. Instead rotate
        # joint_X_p of the world-anchor joints (FIXED with parent=-1, or FREE
        # if fix_base was off). Then eval_fk re-derives body_q from there.
        if rotate_x_deg or rotate_y_deg or rotate_z_deg:
            free_t = int(newton.JointType.FREE)
            fixed_t = int(newton.JointType.FIXED)
            jtypes = self.model.joint_type.numpy()
            jparents = self.model.joint_parent.numpy()

            def _axis_quat(axis, deg):
                a = math.radians(deg) * 0.5
                s, c = math.sin(a), math.cos(a)
                return (s if axis == 0 else 0.0,
                        s if axis == 1 else 0.0,
                        s if axis == 2 else 0.0,
                        c)

            def _qmul(a, b):
                ax, ay, az, aw = a
                bx, by, bz, bw = b
                return (
                    aw*bx + ax*bw + ay*bz - az*by,
                    aw*by - ax*bz + ay*bw + az*bx,
                    aw*bz + ax*by - ay*bx + az*bw,
                    aw*bw - ax*bx - ay*by - az*bz,
                )

            qrot = (0.0, 0.0, 0.0, 1.0)
            if rotate_x_deg:
                qrot = _qmul(_axis_quat(0, rotate_x_deg), qrot)
            if rotate_y_deg:
                qrot = _qmul(_axis_quat(1, rotate_y_deg), qrot)
            if rotate_z_deg:
                qrot = _qmul(_axis_quat(2, rotate_z_deg), qrot)

            # Rotate joint_X_p of every world-anchor joint. wp.transform layout
            # is (px,py,pz, qx,qy,qz,qw). Pre-multiply translation by qrot and
            # compose orientation as qrot * q_existing.
            xp = self.model.joint_X_p.numpy().copy()
            n_rot = 0
            for i in range(int(self.model.joint_count)):
                if int(jparents[i]) != -1:
                    continue
                t = int(jtypes[i])
                if t != fixed_t and t != free_t:
                    continue
                px, py, pz = float(xp[i, 0]), float(xp[i, 1]), float(xp[i, 2])
                qx, qy, qz, qw = (float(xp[i, 3]), float(xp[i, 4]),
                                  float(xp[i, 5]), float(xp[i, 6]))
                # rotate translation by qrot
                vx, vy, vz, vw = qrot
                # quat-rotate point: q * (0,p) * conj(q)
                tx = (1 - 2*(vy*vy + vz*vz))*px + 2*(vx*vy - vz*vw)*py + 2*(vx*vz + vy*vw)*pz
                ty = 2*(vx*vy + vz*vw)*px + (1 - 2*(vx*vx + vz*vz))*py + 2*(vy*vz - vx*vw)*pz
                tz = 2*(vx*vz - vy*vw)*px + 2*(vy*vz + vx*vw)*py + (1 - 2*(vx*vx + vy*vy))*pz
                nq = _qmul(qrot, (qx, qy, qz, qw))
                xp[i, 0], xp[i, 1], xp[i, 2] = tx, ty, tz
                xp[i, 3], xp[i, 4], xp[i, 5], xp[i, 6] = nq
                n_rot += 1
            if n_rot:
                self.model.joint_X_p.assign(xp)
                model_dirty = True
                print(f"  rotate: x={rotate_x_deg} y={rotate_y_deg} z={rotate_z_deg} deg "
                      f"(applied to {n_rot} world-anchor joint(s))")
            else:
                print(f"  rotate: no world-anchor joint found — rotation not applied")

        # Rebuild solver if any baked-at-construction model attribute was
        # mutated (PD gains, etc.). fix_base is handled at builder time so
        # no anchor is needed here — the FREE root joint never exists.
        if model_dirty:
            self.solver = type(self.solver)(self.model)
            print(f"  rebuilt solver after model edits")

        # Sync body_q with joint_q so non-MuJoCo solvers (XPBD/Featherstone) don't snap.
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self.contacts = self.model.collide(self.state_0)
        self.viewer.set_model(self.model)

        print(
            f"[load] usd={usd_path}  bodies={int(self.model.body_count)}  "
            f"joints={n_joints}  solver={bundle.solver_name}  "
            f"fps={self.fps}  substeps={self.sim_substeps}"
        )

    def _set_drive_target(self):
        """Update PD position target for the driven joint to a sine value.

        Newton's MuJoCo solver reads `control.joint_target_pos` (and
        `joint_target_vel`); some other solvers may use a single
        `joint_target` attribute. Try the modern name first, fall back."""
        if self.drive_joint < 0 or self.drive_amp == 0.0:
            return
        target = self.drive_amp * math.sin(2.0 * math.pi * self.drive_freq * self.sim_time)
        target_attr = None
        for name in ("joint_target_pos", "joint_target"):
            if self.control is not None and hasattr(self.control, name):
                target_attr = name
                break
        if target_attr is None:
            return
        arr = getattr(self.control, target_attr)
        v = arr.numpy().copy()
        # joint_target_pos is sized by joint_qd_count (DOFs), indexed by qd_start.
        # legacy joint_target may be sized by joint_q_count, indexed by q_start.
        if target_attr == "joint_target_pos":
            idx = int(self.model.joint_qd_start.numpy()[self.drive_joint])
        else:
            idx = int(self.model.joint_q_start.numpy()[self.drive_joint])
        if 0 <= idx < v.shape[0]:
            v[idx] = target
            arr.assign(v)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self._set_drive_target()
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="example_palatial_articulated")
    p.add_argument("usd", help="Path to a converted *.newton.usda (rigid articulated)")
    p.add_argument("--gui", action="store_true")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--substeps", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--drive-joint", type=int, default=None,
                   help="Joint index to drive (default: first revolute/prismatic). -1 disables.")
    p.add_argument("--drive-amplitude", type=float, default=0.7,
                   help="Sine amplitude in radians (revolute) or meters (prismatic). 0 disables.")
    p.add_argument("--drive-frequency", type=float, default=0.5, help="Sine frequency in Hz")
    p.add_argument("--joint-target-ke", type=float, default=None)
    p.add_argument("--joint-target-kd", type=float, default=None)
    p.add_argument("--rotate-x", type=float, default=0.0, help="Rotate asset around X (degrees)")
    p.add_argument("--rotate-y", type=float, default=0.0, help="Rotate asset around Y (degrees)")
    p.add_argument("--rotate-z", type=float, default=0.0, help="Rotate asset around Z (degrees)")
    p.add_argument("--use-usd-viewer", action="store_true",
                   help="Record sim to a USD file (textured playback for usdview).")
    p.add_argument("--usd-out", type=str, default=None,
                   help="Output USD path. Default: <input>.replay.usda next to the asset.")
    p.add_argument("--usd-fps", type=int, default=60,
                   help="Recording FPS for the USD viewer (default 60).")
    p.add_argument("--usd-num-frames", type=int, default=None,
                   help="Frame cap for the USD recording. Default: matches --steps.")
    p.add_argument("--usd-up-axis", type=str, default="Z", choices=("X", "Y", "Z"),
                   help="USD up axis for the recording (default Z, matches converter).")
    p.add_argument("--usd-textured", action="store_true",
                   help="After recording, also produce a textured replay USDA that\n"
                        "references the source IsaacSim asset (with materials/textures)\n"
                        "and overrides body transforms from the recording. Open this\n"
                        "in usdview / Composer for true PBR playback.")
    p.add_argument("--usd-source-asset", type=str, default=None,
                   help="Path to the source IsaacSim_asset_*.usd to reference for\n"
                        "--usd-textured. Default: auto-detect next to the converter input.")
    args = p.parse_args(argv)

    from newton import viewer as v

    # Build the recorder if requested.
    rec_viewer = None
    usd_out = None
    if args.use_usd_viewer:
        usd_out = args.usd_out
        if usd_out is None:
            base = args.usd
            for sfx in (".newton.usda", ".newton.usdc", ".newton.usd",
                        ".usda", ".usdc", ".usd"):
                if base.endswith(sfx):
                    base = base[: -len(sfx)]
                    break
            usd_out = base + ".replay.usda"
        num_frames = args.usd_num_frames if args.usd_num_frames is not None else args.steps
        print(f"[viewer] ViewerUSD -> {usd_out}  fps={args.usd_fps}  num_frames={num_frames}  up={args.usd_up_axis}")
        rec_viewer = v.ViewerUSD(usd_out, fps=args.usd_fps, up_axis=args.usd_up_axis,
                                 num_frames=num_frames)

    # Build the live viewer (GL window) if --gui, else a Null viewer.
    live_viewer = v.ViewerGL(headless=False) if args.gui else v.ViewerNull()

    # Fan out to both when both are requested; otherwise just use the recorder
    # (no --gui) or just the live viewer (no --use-usd-viewer).
    if args.use_usd_viewer and args.gui:
        viewer = _FanoutViewer(live_viewer, rec_viewer)
        print("[viewer] fanout: ViewerGL + ViewerUSD running together")
    elif args.use_usd_viewer:
        viewer = rec_viewer
    else:
        viewer = live_viewer

    ex = Example(
        viewer, args.usd,
        substeps=args.substeps, device=args.device,
        drive_joint=args.drive_joint,
        drive_amplitude=args.drive_amplitude,
        drive_frequency=args.drive_frequency,
        joint_target_ke=args.joint_target_ke,
        joint_target_kd=args.joint_target_kd,
        rotate_x_deg=args.rotate_x,
        rotate_y_deg=args.rotate_y,
        rotate_z_deg=args.rotate_z,
    )

    def _viewer_running() -> bool:
        """viewer.is_running is an attribute on ViewerGL but a method on
        ViewerUSD/ViewerNull/ViewerBase/_FanoutViewer. Normalize."""
        ir = getattr(viewer, "is_running", True)
        if callable(ir):
            try:
                return bool(ir())
            except Exception:
                return True
        return bool(ir)

    i = 0
    try:
        while True:
            # Stop if the live viewer was closed (GL window X), the recorder
            # frame cap was hit, or we've reached --steps.
            if not _viewer_running():
                break
            if i >= args.steps:
                break
            ex.step()
            ex.render()
            i += 1
    finally:
        try:
            viewer.close()
        except Exception:
            pass

    print(f"[done] frames={i}  sim_time={ex.sim_time:.3f}s")

    # --- Optional textured replay ---------------------------------------
    if args.use_usd_viewer and args.usd_textured:
        try:
            tex_out = _write_textured_replay(
                converter_input=args.usd,
                recording_path=usd_out,
                source_asset=args.usd_source_asset,
                fps=args.usd_fps,
                up_axis=args.usd_up_axis,
                bundle=ex.bundle,
            )
            print(f"[textured] {tex_out}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[textured] FAILED: {e}")

    return 0


class _FanoutViewer:
    """Tiny adapter that fans every viewer method call across two viewers.

    The example only touches a small subset of ViewerBase: set_model,
    begin_frame / end_frame, log_state, log_contacts, apply_forces, close,
    is_running. We forward each one explicitly so behaviour is obvious.

    apply_forces is GL-only (it reads keyboard input for interactive forces).
    is_running ANDs both viewers, so closing the GL window OR hitting the USD
    recorder's num_frames cap stops the loop.
    """

    def __init__(self, live, recorder):
        self._live = live
        self._rec = recorder

    def set_model(self, model, *args, **kwargs):
        self._live.set_model(model, *args, **kwargs)
        self._rec.set_model(model, *args, **kwargs)

    def begin_frame(self, time):
        self._live.begin_frame(time)
        self._rec.begin_frame(time)

    def log_state(self, state):
        self._live.log_state(state)
        self._rec.log_state(state)

    def log_contacts(self, contacts, state):
        self._live.log_contacts(contacts, state)
        self._rec.log_contacts(contacts, state)

    def end_frame(self):
        self._live.end_frame()
        self._rec.end_frame()

    def apply_forces(self, state):
        # Only the live viewer reads keyboard for interactive forces; the
        # recorder's apply_forces is a no-op anyway, but skip to avoid surprise.
        if hasattr(self._live, "apply_forces"):
            self._live.apply_forces(state)

    def close(self):
        # Close recorder first so its file is flushed even if GL.close() throws.
        try:
            self._rec.close()
        finally:
            self._live.close()

    def is_running(self) -> bool:
        # GL exposes attribute; USD exposes method.
        live_ir = getattr(self._live, "is_running", True)
        if callable(live_ir):
            try: live_ok = bool(live_ir())
            except Exception: live_ok = True
        else:
            live_ok = bool(live_ir)
        rec_ir = getattr(self._rec, "is_running", True)
        if callable(rec_ir):
            try: rec_ok = bool(rec_ir())
            except Exception: rec_ok = True
        else:
            rec_ok = bool(rec_ir)
        return live_ok and rec_ok

    def __getattr__(self, name):
        live_attr = getattr(self._live, name, None)
        rec_attr = getattr(self._rec, name, None)
        if callable(live_attr) and callable(rec_attr):
            def _both(*a, **kw):
                self._live and live_attr(*a, **kw)
                self._rec and rec_attr(*a, **kw)
            return _both
        if callable(live_attr):
            return live_attr
        if callable(rec_attr):
            return rec_attr
        if live_attr is not None:
            return live_attr
        if rec_attr is not None:
            return rec_attr
        raise AttributeError(name)


def _write_textured_replay(converter_input: str, recording_path: str,
                           source_asset: str | None,
                           fps: int, up_axis: str, bundle) -> str:
    """Produce a textured replay USDA next to the recording.

    Mechanic: create a new layer that references the source IsaacSim_asset_*.usd
    (which carries UsdShade materials + textures). For each rigid body, author
    time-sampled translate/orient xformOps on its body Xform prim using the
    transforms recorded by ViewerUSD on /root/model/shapes/shape_<i>/instance_0.
    Open the resulting file in usdview / Composer for textured playback.
    """
    import os as _os
    from pxr import Usd, UsdGeom, Gf

    # Resolve source asset path.
    if source_asset is None:
        in_dir = _os.path.dirname(_os.path.abspath(converter_input))
        candidates = [f for f in _os.listdir(in_dir) if f.startswith("IsaacSim_asset_") and f.endswith(".usd")]
        if not candidates:
            raise FileNotFoundError(
                f"--usd-textured: no IsaacSim_asset_*.usd found next to {converter_input}; "
                "pass --usd-source-asset explicitly."
            )
        source_asset = _os.path.join(in_dir, candidates[0])
    source_asset = _os.path.abspath(source_asset)

    rec_stage = Usd.Stage.Open(recording_path)
    if rec_stage is None:
        raise RuntimeError(f"could not open recording: {recording_path}")

    # Map each shape index -> body prim path on the source asset, using the
    # bundle's model.shape_body and model.body_label. Skip ground (-1).
    # Newton renamed body_key -> body_label in the 0.2.x line; fall back to
    # the old attribute name for forward/backward compatibility.
    model = bundle.model
    shape_body = model.shape_body.numpy()
    body_keys = list(
        getattr(model, "body_label", None)
        or getattr(model, "body_key", None)
        or []
    )
    if not body_keys:
        raise RuntimeError(
            "model has no body_label/body_key array; cannot map shapes back to source prims."
        )

    # Build the output stage: defaultPrim is /World referenced from the source.
    out_path = _os.path.splitext(recording_path)[0] + ".textured.usda"
    out_stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(out_stage, {"X": UsdGeom.Tokens.x, "Y": UsdGeom.Tokens.y,
                                       "Z": UsdGeom.Tokens.z}[up_axis.upper()])
    UsdGeom.SetStageMetersPerUnit(out_stage, 1.0)
    out_stage.SetFramesPerSecond(float(fps))
    out_stage.SetTimeCodesPerSecond(float(fps))
    out_stage.SetStartTimeCode(rec_stage.GetStartTimeCode())
    out_stage.SetEndTimeCode(rec_stage.GetEndTimeCode())

    # Reference the source asset under /World.
    world = out_stage.OverridePrim("/World")
    world.GetReferences().AddReference(source_asset)
    out_stage.SetDefaultPrim(world)

    # For each non-ground shape, copy time samples from the recording's instance
    # prim onto the source body's Xform.
    n_authored = 0
    for shape_idx in range(len(shape_body)):
        body_idx = int(shape_body[shape_idx])
        if body_idx < 0:
            continue  # ground / world-anchored
        if body_idx >= len(body_keys):
            continue
        body_path = body_keys[body_idx]
        if not body_path or not body_path.startswith("/"):
            continue
        rec_inst = rec_stage.GetPrimAtPath(f"/root/model/shapes/shape_{shape_idx}/instance_0")
        if not rec_inst:
            continue
        rec_xf = UsdGeom.Xformable(rec_inst)
        rec_ops = {o.GetOpName(): o for o in rec_xf.GetOrderedXformOps()}
        rec_t = rec_ops.get("xformOp:translate")
        rec_r = rec_ops.get("xformOp:orient")
        if rec_t is None or rec_r is None:
            continue

        # Define an override on the source body Xform and add fresh ops.
        body_prim = out_stage.OverridePrim(body_path)
        body_xf = UsdGeom.Xformable(body_prim)
        body_xf.ClearXformOpOrder()
        op_t = body_xf.AddTranslateOp()
        op_r = body_xf.AddOrientOp()

        # Copy time samples 1:1.
        t_samples = rec_t.GetTimeSamples()
        for ts in t_samples:
            v = rec_t.Get(ts)
            if v is not None:
                op_t.Set(Gf.Vec3d(float(v[0]), float(v[1]), float(v[2])), ts)
        r_samples = rec_r.GetTimeSamples()
        for ts in r_samples:
            q = rec_r.Get(ts)
            if q is not None:
                op_r.Set(Gf.Quatf(q.GetReal(), q.GetImaginary()[0],
                                  q.GetImaginary()[1], q.GetImaginary()[2]), ts)
        n_authored += 1

    out_stage.GetRootLayer().Save()
    print(f"[textured] referenced source : {source_asset}")
    print(f"[textured] authored xforms on : {n_authored} body prim(s)")
    return out_path


if __name__ == "__main__":
    sys.exit(main())
