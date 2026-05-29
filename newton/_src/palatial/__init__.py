# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Palatial USD-to-Newton loader / converter helpers.

Reads `*.newton.usda` files (authored by an external converter) and
returns a `NewtonBundle(model, solver, fps, state_in, state_out, control,
body_type, solver_params)` ready to step. Same entry point works for
rigid, cloth, and rod assets; the body type is detected from the USDA.

Public surface is re-exported via `newton.palatial`.
"""
from .load import NewtonBundle, load
from .shell import find_shell_prim_path, read_shell_params
from .cloth import find_cloth_prim_path
from .rod import find_rod_prim_path, read_rod_params

__all__ = [
    "NewtonBundle",
    "load",
    "read_shell_params",
    "find_shell_prim_path",
    "find_cloth_prim_path",
    "find_rod_prim_path",
    "read_rod_params",
]
