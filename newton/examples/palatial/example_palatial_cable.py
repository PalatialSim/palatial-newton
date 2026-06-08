# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Palatial Cable
#
# Loads a converter-produced cable/rod USDA and lets it hang. Cable assets
# are built as a chain of rigid segments connected by an isotropic rod
# (stretch + bend) model, with any connector hardware (plugs, housings)
# rigidly attached to the cable ends.
#
# To make the dynamics easy to see, this example pins the first cable
# segment in place (one clamped end) and lets gravity pull the rest of
# the chain down, like holding a cable by one plug.
#
# Run (interactive window):
#     python -m newton.examples palatial_cable /path/to/cable.newton.usda
#
# Run (headless, write usd):
#     python -m newton.examples palatial_cable /path/to/cable.newton.usda \
#         --viewer usd --output-path out.usd
###########################################################################

import newton
import newton.examples
from newton.palatial import load

from ._common import add_palatial_args, build_contacts, pin_body_kinematic


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer

        bundle = load(args.usd_path, device=args.device)
        self.bundle = bundle
        self.model = bundle.model
        self.solver = bundle.solver
        self.control = bundle.control

        self.fps = bundle.fps
        self.frame_dt = 1.0 / float(self.fps)
        self.sim_substeps = max(1, int(bundle.solver_params.get("substeps", 10)))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.state_0 = bundle.state_in
        self.state_1 = bundle.state_out

        # Lift the whole asset by a constant 0.4 m along the up axis (Z in
        # Newton world space) so the pinned end hangs above the ground and
        # the rest of the chain has room to swing.
        self._lift_asset(0.4)

        # Clamp one end: zero the first segment's inverse mass so it stays
        # put while the rest of the cable swings under gravity. Refresh the
        # solver's cached kinematic state if it exposes the hook.
        if self.model.body_count > 0:
            pin_body_kinematic(self.model, 0)
            if hasattr(self.solver, "_refresh_kinematic_state"):
                self.solver._refresh_kinematic_state()
            print("  cable: pinned segment 0 (one clamped end, free hang)")

        self.contacts = build_contacts(self.model, self.state_0)

        self.viewer.set_model(self.model)
        print(
            f"[palatial_cable] {args.usd_path}\n"
            f"  body_type={bundle.body_type} solver={bundle.solver_name} "
            f"fps={self.fps} substeps={self.sim_substeps} "
            f"segments={self.model.body_count} joints={self.model.joint_count}"
        )

    def _lift_asset(self, dz: float) -> None:
        # VBD uses ``model.body_q`` as the *structural rest pose* and clones
        # it into ``solver.body_q_prev`` at construction time. Lifting only
        # the State buffers makes the rest pose disagree with the current
        # state, so the joint/rod constraints yank the chain back to the
        # original Z on step 1 (the "drop from the sky"). We have to lift
        # the rest pose and the solver's cached prev-pose as well.
        targets = [
            getattr(self.state_0, "body_q", None),
            getattr(self.state_1, "body_q", None),
            getattr(self.model, "body_q", None),
            getattr(self.solver, "body_q_prev", None),
        ]
        for arr in targets:
            if arr is None or arr.shape[0] == 0:
                continue
            q = arr.numpy().copy()
            q[:, 2] += dz
            arr.assign(q)

        for state in (self.state_0, self.state_1):
            particle_q = getattr(state, "particle_q", None)
            if particle_q is not None and particle_q.shape[0] > 0:
                p = particle_q.numpy().copy()
                p[:, 2] += dz
                particle_q.assign(p)

        # Also lift any world-anchored joints (parent == -1). Their
        # ``joint_X_p`` is an absolute world pose; without shifting it,
        # the constraint solver tries to drag the chain back to the
        # original anchor on step 1.
        joint_X_p = getattr(self.model, "joint_X_p", None)
        joint_parent = getattr(self.model, "joint_parent", None)
        if joint_X_p is not None and joint_parent is not None and joint_X_p.shape[0] > 0:
            xp = joint_X_p.numpy().copy()
            parents = joint_parent.numpy()
            mask = parents == -1
            if mask.any():
                xp[mask, 2] += dz
                joint_X_p.assign(xp)

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
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
