# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Palatial USD-to-Newton bundle loader and helpers.

Example
-------
    import newton
    from newton.palatial import load

    bundle = load("/path/to/asset.newton.usda")
    model, solver = bundle.model, bundle.solver
    dt = bundle.dt
    while running:
        contacts = model.collide(bundle.state_in)
        solver.step(bundle.state_in, bundle.state_out,
                    bundle.control, contacts, dt)
        bundle.state_in, bundle.state_out = bundle.state_out, bundle.state_in
"""

from ._src.palatial.cloth import find_cloth_prim_path
from ._src.palatial.load import NewtonBundle, load
from ._src.palatial.rod import find_rod_prim_path, read_rod_params
from ._src.palatial.shell import find_shell_prim_path, read_shell_params
from ._src.palatial.solver_plan import SolverPlanPartEntities, solver_from_plan

__all__ = [
    "NewtonBundle",
    "SolverPlanPartEntities",
    "find_cloth_prim_path",
    "find_rod_prim_path",
    "find_shell_prim_path",
    "load",
    "read_rod_params",
    "read_shell_params",
    "solver_from_plan",
]
