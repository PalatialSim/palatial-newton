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

The returned :class:`NewtonBundle` also includes ``scene_kind`` so callers
can distinguish plain cable bundles from assembly bundles.
"""

from ._src.palatial.load import NewtonBundle, load
from ._src.palatial.cable import (
    extract_cable_points,
    find_cable_centerline_prim_path,
    find_cable_prim_path,
    read_cable_params,
)
from ._src.palatial.shell import find_shell_prim_path, read_shell_params
from ._src.palatial.cloth import find_cloth_prim_path

__all__ = [
    "NewtonBundle",
    "load",
    "extract_cable_points",
    "find_cable_centerline_prim_path",
    "find_cable_prim_path",
    "read_cable_params",
    "read_shell_params",
    "find_shell_prim_path",
    "find_cloth_prim_path",
]
