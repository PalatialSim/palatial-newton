# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Palatial Rigid
#
# Loads a converter-produced rigid-body USDA and drops it on the ground.
# The whole scene (solver choice, fps, substeps, masses, materials) is
# baked into the USDA by the palatial converter, so this script just:
#
#     1. load() the bundle
#     2. step the solver each frame
#     3. log state to the viewer
#
# Run (interactive window):
#     python -m newton.examples palatial_rigid /path/to/asset.newton.usda
#
# Run (headless, write usd):
#     python -m newton.examples palatial_rigid /path/to/asset.newton.usda \
#         --viewer usd --output-path out.usd
###########################################################################

import newton
import newton.examples
from newton.palatial import load

from ._common import add_palatial_args, build_contacts


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer

        # One call parses the USDA and builds model + solver with the
        # parameters the converter baked in.
        bundle = load(args.usd_path, device=args.device)
        self.bundle = bundle
        self.model = bundle.model
        self.solver = bundle.solver
        self.control = bundle.control

        self.fps = bundle.fps
        self.frame_dt = 1.0 / float(self.fps)
        self.sim_substeps = max(1, int(bundle.solver_params.get("substeps", 4)))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.state_0 = bundle.state_in
        self.state_1 = bundle.state_out
        self.contacts = build_contacts(self.model, self.state_0)

        self.viewer.set_model(self.model)
        print(
            f"[palatial_rigid] {args.usd_path}\n"
            f"  body_type={bundle.body_type} solver={bundle.solver_name} "
            f"fps={self.fps} substeps={self.sim_substeps} "
            f"bodies={self.model.body_count} shapes={self.model.shape_count}"
        )

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
