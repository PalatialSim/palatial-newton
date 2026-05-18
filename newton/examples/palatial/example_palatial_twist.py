# Example: load a Newton-ready cloth USDA produced by an external converter
# and twist it by pinning two opposite edges and rotating them in opposite
# directions, mimicking newton/examples/cloth/example_cloth_twist.py.
#
# Unlike the upstream example which hardcodes a 50x50 grid, this one picks the
# two edges by bounding box along --twist-axis, so it works on any cloth mesh
# baked into a converted USDA (square_cloth, t-shirt, ...).
#
# Same load(usd) flow as example_palatial_load.py — solver / fps / substeps /
# cloth material come from the USDA. Cloth-only; rigid bundles are rejected.
#
# Usage:
#   python -m newton.examples.palatial.example_palatial_twist
#       <converted_cloth.usda> [--gui] [--steps N] [--substeps N]
#       [--twist-axis x|y|z] [--angular-velocity RAD_PER_SEC]
#       [--end-time SECONDS] [--edge-thickness METERS] [--gravity 0|1]

from __future__ import annotations

import argparse
import math
import sys

# Newton/Warp must be imported before pxr in the same process.
import warp as wp
import numpy as np

import newton
from newton import ParticleFlags

from newton.palatial import load


@wp.kernel
def initialize_rotation(
    vertex_indices_to_rot: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    rot_centers: wp.array(dtype=wp.vec3),
    rot_axes: wp.array(dtype=wp.vec3),
    t: wp.array(dtype=float),
    # output
    roots: wp.array(dtype=wp.vec3),
    roots_to_ps: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    v_index = vertex_indices_to_rot[tid]

    p = pos[v_index]
    rot_center = rot_centers[tid]
    rot_axis = rot_axes[tid]
    op = p - rot_center

    root = rot_center + wp.dot(op, rot_axis) * rot_axis
    root_to_p = p - root

    roots[tid] = root
    roots_to_ps[tid] = root_to_p

    if tid == 0:
        t[0] = 0.0


@wp.kernel
def apply_rotation(
    vertex_indices_to_rot: wp.array(dtype=wp.int32),
    rot_axes: wp.array(dtype=wp.vec3),
    roots: wp.array(dtype=wp.vec3),
    roots_to_ps: wp.array(dtype=wp.vec3),
    t: wp.array(dtype=float),
    angular_velocity: float,
    dt: float,
    end_time: float,
    # output
    pos_0: wp.array(dtype=wp.vec3),
    pos_1: wp.array(dtype=wp.vec3),
):
    cur_t = t[0]
    if cur_t > end_time:
        return

    tid = wp.tid()
    v_index = vertex_indices_to_rot[tid]

    rot_axis = rot_axes[tid]
    ux = rot_axis[0]
    uy = rot_axis[1]
    uz = rot_axis[2]

    theta = cur_t * angular_velocity
    c = wp.cos(theta)
    s = wp.sin(theta)
    one_c = 1.0 - c

    R = wp.mat33(
        c + ux * ux * one_c,        ux * uy * one_c - uz * s,   ux * uz * one_c + uy * s,
        uy * ux * one_c + uz * s,   c + uy * uy * one_c,        uy * uz * one_c - ux * s,
        uz * ux * one_c - uy * s,   uz * uy * one_c + ux * s,   c + uz * uz * one_c,
    )

    root = roots[tid]
    root_to_p = roots_to_ps[tid]
    p_rot = root + R * root_to_p

    pos_0[v_index] = p_rot
    pos_1[v_index] = p_rot

    if tid == 0:
        t[0] = cur_t + dt


def _select_edge_indices(particle_q_np: np.ndarray, axis: int,
                         edge_thickness: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (low_side_idx, high_side_idx) — particles within `edge_thickness`
    of the bounding-box min/max along `axis`."""
    coords = particle_q_np[:, axis]
    lo, hi = float(coords.min()), float(coords.max())
    low_mask = coords <= lo + edge_thickness
    high_mask = coords >= hi - edge_thickness
    return np.nonzero(low_mask)[0], np.nonzero(high_mask)[0]


class Example:
    def __init__(self, viewer, usd_path: str, *,
                 substeps: int | None = None,
                 twist_axis: int = 1,
                 angular_velocity: float = math.pi / 3,
                 end_time: float = 10.0,
                 edge_thickness: float = 0.02,
                 disable_gravity: bool = True,
                 device: str | None = None,
                 drop_height: float = 0.0,
                 cloth_particle_radius: float = 0.008,
                 soft_contact_ke: float = 100.0,
                 soft_contact_kd: float = 2e-3,
                 soft_contact_mu: float = 1.0,
                 soft_contact_max: int = 1_000_000,
                 cloth_body_contact_margin: float = 0.01,
                 bending_ke: float = 1e-4,
                 bending_kd: float = 1e-3,
                 vbd_particle_edge_contact_buffer_size: int = 64,
                 vbd_particle_collision_detection_interval: int = -1,
                 vbd_rigid_contact_k_start: float | None = None,
                 drop_frames: int = 0):
        self.viewer = viewer

        bundle = load(usd_path, device=device)
        if bundle.body_type != "cloth":
            raise RuntimeError(
                f"example_load_twist needs a cloth USDA, got body_type={bundle.body_type!r}"
            )
        self.bundle = bundle
        self.model = bundle.model
        self.solver = bundle.solver
        self.state_0 = bundle.state_in
        self.state_1 = bundle.state_out
        self.control = bundle.control

        # Cache original gravity so we can switch between drop (gravity on)
        # and twist (respect --gravity flag) phases.
        g = self.model.gravity
        if hasattr(g, "numpy"):
            self._gravity_original = g.numpy().copy()
        else:
            self._gravity_original = None
        self._gravity_zeros = (
            np.zeros_like(self._gravity_original)
            if self._gravity_original is not None else None
        )
        self._disable_gravity_after_drop = bool(disable_gravity)

        # Twist phase starts immediately if no drop is requested.
        self.drop_frames = int(max(0, drop_frames))
        self.frame_idx = 0
        self.twisting = (self.drop_frames == 0)

        # Apply initial gravity. During drop phase gravity must be ON regardless
        # of --gravity (otherwise nothing falls).
        if self.twisting and self._disable_gravity_after_drop:
            self._set_gravity(False)
        else:
            self._set_gravity(True)

        # Optional drop: lift particles in z before the twist begins.
        if drop_height:
            pq = self.state_0.particle_q.numpy().copy()
            pq[:, 2] += float(drop_height)
            self.state_0.particle_q.assign(pq)

        # Cloth contact / material tuning (mirrors example_load_converted).
        self.model.soft_contact_ke = soft_contact_ke
        self.model.soft_contact_kd = soft_contact_kd
        self.model.soft_contact_mu = soft_contact_mu
        self.model.soft_contact_max = soft_contact_max
        self.model.cloth_body_contact_margin = cloth_body_contact_margin
        self.model.bending_ke = bending_ke
        self.model.bending_kd = bending_kd
        n_p = int(self.model.particle_count)
        if n_p > 0:
            self.model.particle_radius.assign(
                np.full(n_p, cloth_particle_radius, dtype=np.float32)
            )

        # VBD-only knobs (no-op for other solvers).
        if type(self.solver).__name__ == "SolverVBD":
            self.solver.particle_edge_contact_buffer_size = vbd_particle_edge_contact_buffer_size
            self.solver.particle_collision_detection_interval = vbd_particle_collision_detection_interval
            self.solver.rigid_contact_k_start = (
                soft_contact_ke if vbd_rigid_contact_k_start is None
                else vbd_rigid_contact_k_start
            )

        # Frame timing from the USDA.
        self.fps = bundle.fps
        self.frame_dt = 1.0 / float(self.fps)
        if substeps is not None:
            self.sim_substeps = max(1, int(substeps))
        else:
            self.sim_substeps = max(1, int(bundle.solver_params.get("substeps", 10)))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.angular_velocity = float(angular_velocity)
        self.end_time = float(end_time)

        # Pick the two opposite edges by bounding box.
        n_particles = int(self.model.particle_count)
        if n_particles == 0:
            raise RuntimeError("Cloth model has zero particles")
        pq = self.state_0.particle_q.numpy()
        low_idx, high_idx = _select_edge_indices(pq, twist_axis, edge_thickness)
        if len(low_idx) == 0 or len(high_idx) == 0:
            raise RuntimeError(
                f"Empty edge selection (axis={twist_axis}, thickness={edge_thickness}). "
                f"Try a larger --edge-thickness."
            )

        rot_indices = np.concatenate([low_idx, high_idx]).astype(np.int32)

        # Rotation axes: one side spins +axis, the other -axis (counter-twist).
        axis_vec = np.zeros(3, dtype=np.float32)
        axis_vec[twist_axis] = 1.0
        rot_axes = np.empty((rot_indices.shape[0], 3), dtype=np.float32)
        rot_axes[: len(low_idx)] = -axis_vec
        rot_axes[len(low_idx):] = axis_vec

        # Rotation centers: per-side centroid projected onto the twist axis
        # passing through the cloth midpoint along the perpendicular plane.
        # Simple choice: use centroid of each edge (matches cloth_twist).
        low_center = pq[low_idx].mean(axis=0)
        high_center = pq[high_idx].mean(axis=0)
        rot_centers = np.empty((rot_indices.shape[0], 3), dtype=np.float32)
        rot_centers[: len(low_idx)] = low_center
        rot_centers[len(low_idx):] = high_center

        # Deactivate the pinned particles (kernel writes positions directly).
        flags = self.model.particle_flags.numpy()
        for i in rot_indices:
            flags[int(i)] = flags[int(i)] & ~int(ParticleFlags.ACTIVE)
        self.model.particle_flags = wp.array(flags)

        self.rot_point_indices = wp.array(rot_indices, dtype=int)
        self.rot_axes = wp.array(rot_axes, dtype=wp.vec3)
        self.rot_centers = wp.array(rot_centers, dtype=wp.vec3)
        self.t = wp.zeros((1,), dtype=float)
        self.roots = wp.zeros_like(self.rot_centers)
        self.roots_to_ps = wp.zeros_like(self.rot_centers)

        wp.launch(
            kernel=initialize_rotation,
            dim=self.rot_point_indices.shape[0],
            inputs=[self.rot_point_indices, self.state_0.particle_q,
                    self.rot_centers, self.rot_axes, self.t],
            outputs=[self.roots, self.roots_to_ps],
        )

        self.contacts = self.model.collide(self.state_0)
        self.viewer.set_model(self.model)

        print(
            f"[twist] usd={usd_path}  solver={bundle.solver_name}  "
            f"fps={self.fps}  substeps={self.sim_substeps}\n"
            f"        particles={n_particles}  pinned={len(rot_indices)} "
            f"(low={len(low_idx)}, high={len(high_idx)})  "
            f"axis={'xyz'[twist_axis]}  ang_vel={self.angular_velocity:.3f} rad/s  "
            f"end_time={self.end_time}s  drop_frames={self.drop_frames}  "
            f"twist_gravity={'off' if disable_gravity else 'on'}"
        )

    def _set_gravity(self, on: bool):
        """Write either the original gravity or zeros into model.gravity."""
        if self._gravity_original is None:
            return
        target = self._gravity_original if on else self._gravity_zeros
        g = self.model.gravity
        if hasattr(g, "assign"):
            g.assign(target)

    def _start_twist(self):
        """Switch from drop phase to twist phase."""
        # Re-init rotation roots from current particle positions, since the
        # cloth may have moved during the drop. Pinned particles themselves
        # are inactive so they haven't moved, but recapture for safety.
        wp.launch(
            kernel=initialize_rotation,
            dim=self.rot_point_indices.shape[0],
            inputs=[self.rot_point_indices, self.state_0.particle_q,
                    self.rot_centers, self.rot_axes, self.t],
            outputs=[self.roots, self.roots_to_ps],
        )
        if self._disable_gravity_after_drop:
            self._set_gravity(False)
        self.twisting = True
        print(f"  drop done at frame {self.frame_idx}; twist begins")

    def simulate(self):
        self.model.collide(self.state_0, self.contacts)
        # SolverVBD has rebuild_bvh; harmless to skip elsewhere.
        if hasattr(self.solver, "rebuild_bvh"):
            self.solver.rebuild_bvh(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            if self.twisting:
                wp.launch(
                    kernel=apply_rotation,
                    dim=self.rot_point_indices.shape[0],
                    inputs=[self.rot_point_indices, self.rot_axes,
                            self.roots, self.roots_to_ps, self.t,
                            self.angular_velocity, self.sim_dt, self.end_time],
                    outputs=[self.state_0.particle_q, self.state_1.particle_q],
                )

            self.solver.step(self.state_0, self.state_1, self.control,
                             self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # Phase transition: drop -> twist
        if not self.twisting and self.frame_idx >= self.drop_frames:
            self._start_twist()
        self.simulate()
        self.sim_time += self.frame_dt
        self.frame_idx += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="example_load_twist")
    p.add_argument("usd", help="Path to a converted *.newton.usda (cloth body type)")
    p.add_argument("--steps", type=int, default=600,
                   help="Frames to simulate when not in GUI mode")
    p.add_argument("--substeps", type=int, default=None,
                   help="Override newton:solver:substeps from the USDA")
    p.add_argument("--gui", action="store_true",
                   help="Open ViewerGL and run until the window is closed")
    p.add_argument("--device", default=None,
                   help="Warp device, e.g. 'cuda:0' or 'cpu'")
    p.add_argument("--twist-axis", choices=["x", "y", "z"], default="y",
                   help="World axis along which to pick the two opposite edges")
    p.add_argument("--angular-velocity", type=float, default=math.pi / 3,
                   help="Twist angular velocity (rad/s)")
    p.add_argument("--end-time", type=float, default=10.0,
                   help="Stop twisting after this many seconds (cloth keeps relaxing)")
    p.add_argument("--edge-thickness", type=float, default=0.02,
                   help="Thickness (m) of the bounding-box slab used to pick each edge")
    p.add_argument("--gravity", action="store_true",
                   help="Keep gravity on during twist phase (default off, like example_cloth_twist). "
                        "Gravity is always ON during the drop phase.")
    p.add_argument("--drop-height", type=float, default=0.0,
                   help="Lift particles by this many meters in z before sim starts")
    p.add_argument("--drop-frames", type=int, default=0,
                   help="Frames to drop under gravity (with edges pinned) before twisting. "
                        "0 = start twisting immediately (default).")
    p.add_argument("--record-mp4", default=None,
                   help="Path to output .mp4. Uses ViewerGL (headless unless --gui) "
                        "and pipes frames to ffmpeg.")
    p.add_argument("--mp4-fps", type=int, default=60,
                   help="Output mp4 framerate (default 60)")
    p.add_argument("--top-view", action="store_true",
                   help="Place camera straight above the cloth looking down")

    # Optional physics JSON: overrides --cloth-particle-radius from
    # solver.vbd_particle_self_contact_radius and --bending-ke from
    # cloth.bend_stiffness when present.
    p.add_argument("--physics-json", default=None,
                   help="Path to a physics JSON. If given, "
                        "solver.vbd_particle_self_contact_radius overrides "
                        "--cloth-particle-radius and cloth.bend_stiffness "
                        "overrides --bending-ke.")

    # Cloth tuning (same defaults as example_load_converted).
    p.add_argument("--cloth-particle-radius", type=float, default=0.008)
    p.add_argument("--soft-contact-ke", type=float, default=100.0)
    p.add_argument("--soft-contact-kd", type=float, default=2e-3)
    p.add_argument("--soft-contact-mu", type=float, default=1.0)
    p.add_argument("--soft-contact-max", type=int, default=1_000_000)
    p.add_argument("--cloth-body-contact-margin", type=float, default=0.01)
    p.add_argument("--bending-ke", type=float, default=1e-4)
    p.add_argument("--bending-kd", type=float, default=1e-3)

    # VBD-only knobs.
    p.add_argument("--vbd-particle-edge-contact-buffer-size", type=int, default=64)
    p.add_argument("--vbd-particle-collision-detection-interval", type=int, default=-1)
    p.add_argument("--vbd-rigid-contact-k-start", type=float, default=None,
                   help="Defaults to --soft-contact-ke when omitted")

    args = p.parse_args(argv)

    # Apply physics-json overrides for cloth-particle-radius and bending-ke.
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
        print(
            f"  physics-json: {args.physics_json}  "
            f"cloth_particle_radius={args.cloth_particle_radius}  "
            f"bending_ke={args.bending_ke}"
        )

    axis = {"x": 0, "y": 1, "z": 2}[args.twist_axis]

    from newton import viewer as v
    if args.record_mp4:
        # ViewerGL renders to a framebuffer we can read back via get_frame().
        viewer = v.ViewerGL(headless=not args.gui)
    elif args.gui:
        viewer = v.ViewerGL(headless=False)
    else:
        viewer = v.ViewerNull()

    ex = Example(
        viewer, args.usd,
        substeps=args.substeps,
        twist_axis=axis,
        angular_velocity=args.angular_velocity,
        end_time=args.end_time,
        edge_thickness=args.edge_thickness,
        disable_gravity=not args.gravity,
        device=args.device,
        drop_height=args.drop_height,
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
        drop_frames=args.drop_frames,
    )

    # ---- Top-down camera (ViewerGL only) ----
    if args.top_view and hasattr(ex.viewer, "set_camera"):
        pq = ex.state_0.particle_q.numpy()
        cx, cy = float(pq[:, 0].mean()), float(pq[:, 1].mean())
        zmax = float(pq[:, 2].max())
        ext_x = float(pq[:, 0].max() - pq[:, 0].min())
        ext_y = float(pq[:, 1].max() - pq[:, 1].min())
        height = zmax + max(ext_x, ext_y, 1.0) * 1.5
        # pitch -90 looks straight down, yaw 0 keeps +Y up in image.
        ex.viewer.set_camera(wp.vec3(cx, cy, height), -90.0, 0.0)
        print(f"  top-view camera: pos=({cx:.3f},{cy:.3f},{height:.3f})")

    # ---- mp4 recorder (ffmpeg subprocess) ----
    ffmpeg_proc = None
    if args.record_mp4:
        import subprocess, shutil
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not on PATH; cannot record mp4")
        # Render one frame so the framebuffer size is known.
        ex.render()
        frame = ex.viewer.get_frame()
        h, w, _ = frame.shape
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(args.mp4_fps),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "20", "-preset", "fast",
            args.record_mp4,
        ]
        ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        # First frame goes to ffmpeg too.
        ffmpeg_proc.stdin.write(frame.numpy().tobytes())
        print(f"  recording mp4: {args.record_mp4}  size={w}x{h}  fps={args.mp4_fps}")

    i = 1 if ffmpeg_proc is not None else 0
    use_step_count = bool(args.record_mp4) or not args.gui
    try:
        while True:
            if use_step_count:
                if i >= args.steps:
                    break
            else:
                if not getattr(viewer, "is_running", True):
                    break
            ex.step()
            ex.render()
            if ffmpeg_proc is not None:
                ffmpeg_proc.stdin.write(ex.viewer.get_frame().numpy().tobytes())
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
