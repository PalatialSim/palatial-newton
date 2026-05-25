# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Load a Palatial cable USDA and simulate it with optional anchored twist.

This is the cable-specific sibling of ``example_palatial_load.py``. It consumes
the public ``newton.palatial.load()`` entry point, requires a cable or rod USDA,
and optionally anchors the first segment and spins it about the cable axis to
demonstrate twist propagation through the loaded cable bundle.
"""

from __future__ import annotations

import argparse
import inspect
import sys

import numpy as np
import warp as wp

import newton

from newton.palatial import extract_cable_points, load, read_cable_params


@wp.kernel
def spin_first_capsules_kernel(
    body_indices: wp.array[wp.int32],
    twist_rates: wp.array[float],
    dt: float,
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
):
    """Apply continuous twist to the anchored cable segment."""
    tid = wp.tid()
    body_id = body_indices[tid]

    transform = body_q0[body_id]
    pos = wp.transform_get_translation(transform)
    rot = wp.transform_get_rotation(transform)

    # Cable rod capsules are authored with their long axis along local +Z.
    axis_world = wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0))
    angle = twist_rates[tid] * dt
    dq = wp.quat_from_axis_angle(axis_world, angle)
    rot_new = wp.mul(dq, rot)

    updated = wp.transform(pos, rot_new)
    body_q0[body_id] = updated
    body_q1[body_id] = updated


def _filter_solver_kwargs(solver_cls: type, solver_params: dict[str, object]) -> dict[str, object]:
    """Keep only solver kwargs accepted by the current constructor."""
    alias = {
        "iterations": "iterations",
        "substeps": "substeps",
        "particleEnableSelfContact": "particle_enable_self_contact",
        "particleSelfContactRadius": "particle_self_contact_radius",
        "particleSelfContactMargin": "particle_self_contact_margin",
    }
    try:
        accepted = set(inspect.signature(solver_cls.__init__).parameters)
    except (TypeError, ValueError):
        accepted = set()

    kwargs: dict[str, object] = {}
    for name, value in solver_params.items():
        py_name = alias.get(name, name)
        if py_name in accepted:
            kwargs[py_name] = value
    return kwargs


def _set_front_camera(viewer, points: np.ndarray) -> None:
    """Frame a front-facing camera from the current cable bounds."""
    cx = float(points[:, 0].mean())
    cy = float(points[:, 1].mean())
    cz_mid = float((points[:, 2].min() + points[:, 2].max()) * 0.5)
    ext_x = float(points[:, 0].max() - points[:, 0].min())
    ext_y = float(points[:, 1].max() - points[:, 1].min())
    ext_z = float(points[:, 2].max() - points[:, 2].min())
    dist = max(ext_x, ext_y, ext_z, 1.0) * 2.2
    viewer.set_camera(wp.vec3(cx, cy - dist, cz_mid), 0.0, 90.0)
    print(f"  front-view camera: pos=({cx:.3f},{cy - dist:.3f},{cz_mid:.3f})")


def _set_top_camera(viewer, points: np.ndarray) -> None:
    """Frame a top-down camera from the current cable bounds."""
    cx = float(points[:, 0].mean())
    cy = float(points[:, 1].mean())
    zmax = float(points[:, 2].max())
    ext_x = float(points[:, 0].max() - points[:, 0].min())
    ext_y = float(points[:, 1].max() - points[:, 1].min())
    height = zmax + max(ext_x, ext_y, 1.0) * 1.5
    viewer.set_camera(wp.vec3(cx, cy, height), -90.0, 0.0)
    print(f"  top-view camera: pos=({cx:.3f},{cy:.3f},{height:.3f})")


class Example:
    def __init__(
        self,
        viewer,
        usd_path: str,
        *,
        substeps: int | None = None,
        device: str | None = None,
        solver_override: str | None = None,
        anchor_first: bool = True,
        spin_rate: float = 0.5,
        extra_drop_height: float = 0.0,
    ):
        self.viewer = viewer
        self.usd_path = usd_path

        bundle = load(usd_path, solver_override=solver_override, device=device)
        if bundle.body_type != "cable":
            raise RuntimeError(
                f"example_palatial_cable needs a cable or rod USDA, got body_type={bundle.body_type!r}"
            )

        self.bundle = bundle
        self.model = bundle.model
        self.solver = bundle.solver
        self.state_0 = bundle.state_in
        self.state_1 = bundle.state_out
        self.control = bundle.control
        self.contacts = bundle.model.contacts()

        self.cable_params = read_cable_params(usd_path)
        self.centerline_points = extract_cable_points(usd_path, world_space=True)
        self._solver_kwargs = _filter_solver_kwargs(type(self.solver), bundle.solver_params)

        self.anchor_first = bool(anchor_first)
        self.spin_rate = float(spin_rate)
        if abs(self.spin_rate) > 0.0 and not self.anchor_first:
            raise RuntimeError("--spin-rate requires --anchor-first so the driven segment stays kinematic")

        body_count = int(self.model.body_count)
        if body_count <= 0:
            raise RuntimeError("Cable bundle has zero rigid bodies")
        self.anchor_body_index = 0
        self.tip_body_index = body_count - 1

        if extra_drop_height:
            self._shift_cable_z(float(extra_drop_height))

        if self.anchor_first:
            self._make_body_kinematic(self.anchor_body_index)
            self._rebuild_solver()

        self.fps = bundle.fps
        self.frame_dt = 1.0 / float(self.fps)
        if substeps is not None:
            self.sim_substeps = max(1, int(substeps))
        else:
            self.sim_substeps = max(1, int(bundle.solver_params.get("substeps", 1)))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        body_device = self.state_0.body_q.device if self.state_0.body_q is not None else self.model.device
        self.spin_body_indices = wp.array([self.anchor_body_index], dtype=wp.int32, device=body_device)
        self.spin_rates = wp.array([self.spin_rate], dtype=wp.float32, device=body_device)

        self.model.collide(self.state_0, self.contacts)
        self.viewer.set_model(self.model)

        self._print_bundle_summary(extra_drop_height)

    def _rebuild_solver(self) -> None:
        self.solver = type(self.solver)(self.model, **self._solver_kwargs)

    def _shift_cable_z(self, dz: float) -> None:
        """Lift every cable segment by ``dz`` in world +Z."""
        for state in (self.state_0, self.state_1):
            body_q = state.body_q.numpy().copy()
            body_q[:, 2] += dz
            state.body_q.assign(body_q)

    def _make_body_kinematic(self, body_index: int) -> None:
        """Freeze one cable segment by zeroing its mass and inertia."""
        body_mass = self.model.body_mass.numpy().copy()
        body_inv_mass = self.model.body_inv_mass.numpy().copy()
        body_inertia = self.model.body_inertia.numpy().copy()
        body_inv_inertia = self.model.body_inv_inertia.numpy().copy()

        body_mass[body_index] = 0.0
        body_inv_mass[body_index] = 0.0
        body_inertia[body_index] = np.zeros((3, 3), dtype=np.float32)
        body_inv_inertia[body_index] = np.zeros((3, 3), dtype=np.float32)

        self.model.body_mass.assign(body_mass)
        self.model.body_inv_mass.assign(body_inv_mass)
        self.model.body_inertia.assign(body_inertia)
        self.model.body_inv_inertia.assign(body_inv_inertia)

    def _print_bundle_summary(self, extra_drop_height: float) -> None:
        params = self.cable_params
        print(
            f"[cable] usd={self.usd_path}  solver={self.bundle.solver_name}  "
            f"fps={self.fps}  substeps={self.sim_substeps}  bodies={int(self.model.body_count)}  "
            f"joints={int(self.model.joint_count)}"
        )
        print(
            f"        cross_section={params['crossSectionType']}  radius={float(params['radius']):.4f}m  "
            f"length={float(params['length']):.4f}m  segments={int(params['segmentCount'])}  "
            f"anchor_first={self.anchor_first}  spin_rate={self.spin_rate:.3f}rad/s  "
            f"extra_drop={extra_drop_height:.3f}m"
        )
        print(
            "        stiffness: "
            f"stretch={float(params['stretchStiffness']):.3g}  "
            f"bend_y={float(params['bendYStiffness']):.3g}  "
            f"bend_z={float(params['bendZStiffness']):.3g}  "
            f"torsion={float(params['torsionStiffness']):.3g}"
        )
        print(
            "        damping: "
            f"stretch={float(params['stretchDamping']):.3g}  "
            f"bend_y={float(params['bendYDamping']):.3g}  "
            f"bend_z={float(params['bendZDamping']):.3g}  "
            f"torsion={float(params['torsionDamping']):.3g}"
        )
        if self.centerline_points:
            print(
                f"        authored centerline points={len(self.centerline_points)}  "
                f"dropHeight={float(params['dropHeight']):.3f}m  twistTotal={float(params['twistTotal']):.3f}rad"
            )

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            if abs(self.spin_rate) > 0.0:
                wp.launch(
                    kernel=spin_first_capsules_kernel,
                    dim=self.spin_body_indices.shape[0],
                    inputs=[self.spin_body_indices, self.spin_rates, self.sim_dt],
                    outputs=[self.state_0.body_q, self.state_1.body_q],
                    device=self.spin_body_indices.device,
                )

            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="example_palatial_cable")
    parser.add_argument("usd", help="Path to a converted cable *.newton.usda")
    parser.add_argument("--steps", type=int, default=600, help="Frames to simulate when not in GUI mode")
    parser.add_argument("--substeps", type=int, default=None, help="Override newton:solver:substeps from the USDA")
    parser.add_argument("--gui", action="store_true", help="Open ViewerGL and run until the window is closed")
    parser.add_argument("--device", default=None, help="Warp device, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument(
        "--solver-override",
        choices=("vbd", "vbd_palatial"),
        default=None,
        help="Override the solver baked into the USDA",
    )
    parser.add_argument(
        "--anchor-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze the first cable segment so the bundle hangs from one end",
    )
    parser.add_argument(
        "--spin-rate",
        type=float,
        default=0.5,
        help="Angular speed [rad/s] applied to the anchored first segment around the cable axis",
    )
    parser.add_argument(
        "--extra-drop-height",
        type=float,
        default=0.0,
        help="Additional world-space lift [m] applied to the cable before simulation",
    )
    parser.add_argument(
        "--record-mp4",
        default=None,
        help="Output mp4 path. Uses ViewerGL (headless unless --gui) and pipes frames to ffmpeg.",
    )
    parser.add_argument("--mp4-fps", type=int, default=60, help="Output mp4 framerate (default 60)")
    parser.add_argument("--top-view", action="store_true", help="Place camera straight above the cable looking down")
    parser.add_argument(
        "--no-auto-camera",
        action="store_true",
        help="Skip the recording auto-camera so the viewer keeps its default camera",
    )
    args = parser.parse_args(argv)

    from newton import viewer as v

    if args.record_mp4:
        viewer = v.ViewerGL(headless=not args.gui)
    elif args.gui:
        viewer = v.ViewerGL(headless=False)
    else:
        viewer = v.ViewerNull()

    example = Example(
        viewer,
        args.usd,
        substeps=args.substeps,
        device=args.device,
        solver_override=args.solver_override,
        anchor_first=args.anchor_first,
        spin_rate=args.spin_rate,
        extra_drop_height=args.extra_drop_height,
    )

    body_points = example.state_0.body_q.numpy()[:, 0:3]
    if args.top_view and hasattr(example.viewer, "set_camera"):
        _set_top_camera(example.viewer, body_points)
    elif args.record_mp4 and not args.no_auto_camera and hasattr(example.viewer, "set_camera"):
        _set_front_camera(example.viewer, body_points)

    ffmpeg_proc = None
    if args.record_mp4:
        import shutil
        import subprocess

        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not on PATH; cannot record mp4")
        example.render()
        frame = example.viewer.get_frame()
        height, width, _ = frame.shape
        enc_width = width - (width % 2)
        enc_height = height - (height % 2)
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{enc_width}x{enc_height}",
            "-r",
            str(args.mp4_fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-preset",
            "fast",
            args.record_mp4,
        ]
        ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        ffmpeg_proc.stdin.write(frame.numpy()[:enc_height, :enc_width, :].tobytes())
        print(f"  recording mp4: {args.record_mp4}  size={enc_width}x{enc_height}  fps={args.mp4_fps}")

    frame_index = 1 if ffmpeg_proc is not None else 0
    use_step_count = bool(args.record_mp4) or not args.gui
    try:
        while True:
            if use_step_count:
                if frame_index >= args.steps:
                    break
            else:
                if not getattr(viewer, "is_running", True):
                    break
            example.step()
            example.render()
            if ffmpeg_proc is not None:
                ffmpeg_proc.stdin.write(example.viewer.get_frame().numpy()[:enc_height, :enc_width, :].tobytes())
            frame_index += 1
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

    print(f"[done] frames={frame_index}  sim_time={example.sim_time:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
