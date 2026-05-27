# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Power cable assembly stage detection."""

from __future__ import annotations

# Importing newton registers the bundled USD schema plugins before pxr.Usd use.
import newton as _newton  # noqa: F401
from pxr import Usd

from .constants import GEOMETRY_SCOPE_PATH, POWER_REQUIRED_PRIM_NAMES


def is_power_cable_assembly_stage(stage: Usd.Stage) -> bool:
    """Return whether the stage matches the v1 Power cable assembly shape."""

    geometry = stage.GetPrimAtPath(GEOMETRY_SCOPE_PATH)
    if not geometry or not geometry.IsValid():
        return False
    child_names = {child.GetName() for child in geometry.GetChildren()}
    return set(POWER_REQUIRED_PRIM_NAMES).issubset(child_names)
