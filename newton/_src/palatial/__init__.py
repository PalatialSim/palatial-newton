# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Palatial USD-to-Newton loader / converter helpers.

Reads `*.newton.usda` files (authored by an external converter) and
returns a `NewtonBundle(model, solver, fps, state_in, state_out, control,
body_type, solver_params, scene_kind)` ready to step. Same entry point works
for rigid, cloth, simple cable, and cable-assembly assets; the body type and
scene kind are detected from the USDA.

Public surface is re-exported via `newton.palatial`.
"""
from .load import NewtonBundle, load
from .cable import extract_cable_points, find_cable_centerline_prim_path, find_cable_prim_path, read_cable_params
from .shell import find_shell_prim_path, read_shell_params
from .cloth import find_cloth_prim_path

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
