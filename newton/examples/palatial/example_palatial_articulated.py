# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Palatial Articulated
#
# Loads a converter-produced articulated USDA (a jointed mechanism such as
# a microwave door, drawer, or hinge) and drives ONE joint with a sine
# wave so you can see the articulation move. Every other joint holds at
# zero. Pass --drive-amplitude 0 to just watch the asset settle.
#
# Joint targets use Newton's DOF layout (control.joint_target_q indexed by
# joint_qd_start), which is the default in this Newton version.
#
# Run (interactive window):
#     python -m newton.examples palatial_articulated /path/to/asset.newton.usda
#
#     # drive the second joint harder:
#     python -m newton.examples palatial_articulated /path/to/asset.newton.usda \
#         --drive-joint 1 --drive-amplitude 0.7 --drive-frequency 0.5
###########################################################################

import math

import numpy as np

import newton
import newton.examples
from newton.palatial import load

from ._common import add_palatial_args, build_contacts, rebuild_solver


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer

        bundle = load(args.usd_path, device=args.device, fix_base=args.fix_base)
        self.bundle = bundle
        self.model = bundle.model
        self.solver = bundle.solver
        self.control = bundle.control

        self.fps = bundle.fps
        self.frame_dt = 1.0 / float(self.fps)
        self.sim_substeps = max(1, int(bundle.solver_params.get("substeps", 8)))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.state_0 = bundle.state_in
        self.state_1 = bundle.state_out
        self.contacts = build_contacts(self.model, self.state_0)

        # Pick the joint to drive: caller override, else the first
        # revolute/prismatic joint, else nothing.
        jtypes = self.model.joint_type.numpy()
        jqd_start = self.model.joint_qd_start.numpy()
        drivable = [
            i for i, t in enumerate(jtypes)
            if t in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC)
        ]
        if args.drive_joint is not None:
            self.drive_joint = int(args.drive_joint)
        elif drivable:
            self.drive_joint = drivable[0]
        else:
            self.drive_joint = -1

        # DOF index for the driven joint's first axis.
        self.drive_dof = int(jqd_start[self.drive_joint]) if self.drive_joint >= 0 else -1
        self.drive_amp = float(args.drive_amplitude)
        self.drive_freq = float(args.drive_frequency)

        # Make the driven joint a POSITION servo with real authority.
        #
        # Converter assets often bake a joint as a VELOCITY actuator with
        # joint_target_ke=0 (no position stiffness). Writing a position
        # target into such a joint does nothing, so the asset looks frozen.
        # We therefore (1) switch the driven joint into POSITION mode and
        # (2) give it a usable ke/kd when the asset baked none. Because
        # MuJoCo bakes actuator type + gains at solver-construction time,
        # these edits only take effect if we rebuild the solver afterwards.
        if self.drive_joint >= 0 and self.drive_amp != 0.0:
            self._arm_position_drive(args)

        self.viewer.set_model(self.model)
        print(
            f"[palatial_articulated] {args.usd_path}\n"
            f"  body_type={bundle.body_type} solver={bundle.solver_name} "
            f"fps={self.fps} substeps={self.sim_substeps} "
            f"bodies={self.model.body_count} joints={self.model.joint_count}"
        )
        if self.drive_joint >= 0 and self.drive_amp != 0.0:
            print(f"  driving joint[{self.drive_joint}] (dof {self.drive_dof}) "
                  f"sine amp={self.drive_amp} freq={self.drive_freq}Hz")
        else:
            print("  no joint driven (settle only)")

    def _arm_position_drive(self, args):
        """Force the driven joint into a POSITION servo and rebuild the solver.

        Sets joint_target_mode=POSITION on the driven DOF and applies a
        sensible ke/kd (CLI override wins; otherwise default when the asset
        baked ke<=0). Then rebuilds the solver so MuJoCo reconstructs the
        actuator as a position servo. Falls back to the original solver if
        the rebuild is unavailable.
        """
        dof = self.drive_dof

        # Switch the driven DOF to POSITION mode if the model exposes modes.
        mode_arr = getattr(self.model, "joint_target_mode", None)
        if mode_arr is not None and hasattr(newton, "JointTargetMode"):
            mode = mode_arr.numpy().copy()
            if 0 <= dof < mode.shape[0]:
                mode[dof] = int(newton.JointTargetMode.POSITION)
                mode_arr.assign(mode)

        # Position stiffness: CLI override, else a default when unauthored.
        if hasattr(self.model, "joint_target_ke"):
            ke = self.model.joint_target_ke.numpy().copy()
            if args.joint_target_ke is not None:
                ke[dof] = float(args.joint_target_ke)
            elif dof < ke.shape[0] and ke[dof] <= 0.0:
                ke[dof] = 300.0
            self.model.joint_target_ke.assign(ke)

        # Damping: CLI override, else a calm default that won't lock the joint.
        if hasattr(self.model, "joint_target_kd"):
            kd = self.model.joint_target_kd.numpy().copy()
            if args.joint_target_kd is not None:
                kd[dof] = float(args.joint_target_kd)
            elif dof < kd.shape[0] and kd[dof] > 100.0:
                kd[dof] = 10.0
            self.model.joint_target_kd.assign(kd)

        # Rebuild so the solver picks up the new actuator type + gains.
        new_solver = rebuild_solver(
            self.model, self.bundle.solver_name, self.bundle.solver_params
        )
        if new_solver is not None:
            self.solver = new_solver

    def _apply_drive(self):
        if self.drive_dof < 0 or self.drive_amp == 0.0:
            return
        target = self.control.joint_target_q
        if target is None:
            return
        tq = target.numpy().copy()
        tq[self.drive_dof] = self.drive_amp * math.sin(2.0 * math.pi * self.drive_freq * self.sim_time)
        target.assign(tq)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self._apply_drive()
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
    parser.add_argument("--drive-joint", type=int, default=None,
                        help="Index of the joint to drive (default: first revolute/prismatic).")
    parser.add_argument("--drive-amplitude", type=float, default=0.7,
                        help="Sine amplitude in rad (revolute) or m (prismatic). 0 disables driving.")
    parser.add_argument("--drive-frequency", type=float, default=0.5,
                        help="Sine frequency in Hz.")
    parser.add_argument("--joint-target-ke", type=float, default=None,
                        help="Override position-drive stiffness on all joints.")
    parser.add_argument("--joint-target-kd", type=float, default=None,
                        help="Override position-drive damping on all joints.")
    parser.add_argument("--fix-base", dest="fix_base", action="store_true", default=True,
                        help="Anchor the root body to the world (default for articulated assets).")
    parser.add_argument("--no-fix-base", dest="fix_base", action="store_false",
                        help="Let the root body fall freely instead of anchoring it.")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
