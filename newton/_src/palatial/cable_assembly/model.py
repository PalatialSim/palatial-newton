# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Power cable assembly model construction."""

from __future__ import annotations

import newton
import warp as wp

from .builder import build_power_cable_assembly
from .canonical import build_power_canonical_assembly
from .extraction import extract_power_cable_assembly
from .scene import apply_default_runtime_scene


def build_power_cable_assembly_model(usd_path: str, *, device: str | None = None) -> newton.Model:
    """Build a Newton model for a v1 Power cable assembly USD."""

    extraction = extract_power_cable_assembly(usd_path)
    assembly = build_power_canonical_assembly(extraction)

    with wp.ScopedDevice(device) if device else wp.ScopedDevice(wp.get_preferred_device()):
        builder = newton.ModelBuilder()
        build_power_cable_assembly(builder, assembly)
        apply_default_runtime_scene(builder)
        try:
            builder.color()
        except Exception:
            pass
        return builder.finalize()
