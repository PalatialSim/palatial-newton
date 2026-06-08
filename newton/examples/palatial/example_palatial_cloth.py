# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np

import newton
import newton.examples
from newton.palatial import load

from ._common import add_palatial_args


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer

        bundle = load(args.usd_path, device=args.device)
        self.bundle = bundle
        self.model = bundle.model
        # Solver is built by load() from the asset's authored newton:solver:*
        # params plus cloth self-contact defaults. We use it as-is.
        self.solver = bundle.solver
        self.control = bundle.control

        self.fps = bundle.fps
        self.frame_dt = 1.0 / float(self.fps)
       
        if getattr(args, "substeps", None) is not None:
            self.sim_substeps = max(1, int(args.substeps))
        else:
            self.sim_substeps = max(1, int(bundle.solver_params.get("substeps", 1)))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.state_0 = bundle.state_in
        self.state_1 = bundle.state_out

        # Cloth assets should have newton:soft_contact:* params authored for
        # good ground interaction out of the box, but CLI overrides are here for experimentation.
        if args.soft_contact_ke is not None:
            self.model.soft_contact_ke = float(args.soft_contact_ke)
        if args.soft_contact_kd is not None:
            self.model.soft_contact_kd = float(args.soft_contact_kd)
        print(
            f"  soft_contact: ke={self.model.soft_contact_ke} "
            f"kd={self.model.soft_contact_kd} mu={self.model.soft_contact_mu}"
        )

        self._pose_cloth(args)

        self.contacts = self.model.contacts()
        self.model.collide(self.state_0, self.contacts)

        self.viewer.set_model(self.model)
        print(
            f"[palatial_cloth] {args.usd_path}\n"
            f"  body_type={bundle.body_type} solver={bundle.solver_name} "
            f"fps={self.fps} substeps={self.sim_substeps} "
            f"particles={self.model.particle_count}"
        )

    def _pose_cloth(self, args):
        """Rotate the cloth about world axes, then drop it by a height.

        Mirrors the rotated-drop in example_palatial_load, cloth-only: every
        particle is rotated around the origin by the x/y/z euler angles, then
        translated up in +z so gravity pulls it down onto the ground.
        """
        if self.model.particle_count == 0:
            return

        def axis_quat(axis, deg):
            a = math.radians(deg) * 0.5
            s, c = math.sin(a), math.cos(a)
            return np.array(
                [s if axis == 0 else 0.0,
                 s if axis == 1 else 0.0,
                 s if axis == 2 else 0.0,
                 c],
                dtype=np.float32,
            )

        def qmul(a, b):
            ax, ay, az, aw = a
            bx, by, bz, bw = b
            return np.array([
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ], dtype=np.float32)

        rx, ry, rz = args.rotate_x, args.rotate_y, args.rotate_z
        if rx or ry or rz:
            q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            if rx:
                q = qmul(axis_quat(0, rx), q)
            if ry:
                q = qmul(axis_quat(1, ry), q)
            if rz:
                q = qmul(axis_quat(2, rz), q)
            # Rotate all particles around the origin: v + 2w(qxv) + 2qx(qxv).
            pq = self.state_0.particle_q.numpy().copy()
            qv = q[:3]
            t = 2.0 * np.cross(np.broadcast_to(qv, pq.shape), pq)
            pq = pq + q[3] * t + np.cross(np.broadcast_to(qv, t.shape), t)
            self.state_0.particle_q.assign(pq.astype(np.float32))
            print(f"  rotate: x={rx} y={ry} z={rz} deg -> particle_q x{pq.shape[0]}")

        if args.drop_height:
            pq = self.state_0.particle_q.numpy().copy()
            pq[:, 2] += float(args.drop_height)
            self.state_0.particle_q.assign(pq)
            print(f"  drop: +{args.drop_height}m -> particle_q x{pq.shape[0]}")

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    add_palatial_args(parser)
    parser.add_argument("--rotate-x", type=float, default=0.0,
                        help="Rotate the cloth this many degrees about the world X axis before dropping.")
    parser.add_argument("--rotate-y", type=float, default=0.0,
                        help="Rotate the cloth this many degrees about the world Y axis before dropping.")
    parser.add_argument("--rotate-z", type=float, default=0.0,
                        help="Rotate the cloth this many degrees about the world Z axis before dropping.")
    parser.add_argument("--drop-height", type=float, default=0.2,
                        help="Lift the cloth this many meters in +Z before it falls (default: 0.2).")
    parser.add_argument("--substeps", type=int, default=1,
                        help="Solver substeps per frame (default: 1 = real-time on a fast GPU). "
                             "Raise for extra stability at the cost of slower-than-real GUI playback; "
                             "mp4 recording stays real-time via fps decimation regardless.")
    parser.add_argument("--soft-contact-ke", type=float, default=1e5,
                        help="Soft-contact normal stiffness vs rigid/ground (default: 1e5). "
                             "At 1e4 the denim slowly compresses through the floor; 1e5 holds it.")
    parser.add_argument("--soft-contact-kd", type=float, default=1.0,
                        help="Soft-contact normal damping vs rigid/ground (default: 1.0). "
                             "Pairs with the stiff ke to stop ground seepage; only helps when ke is high.")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
