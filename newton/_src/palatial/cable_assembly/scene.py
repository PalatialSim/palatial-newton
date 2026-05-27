# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reference runtime scene helpers for Power cable assemblies."""

from __future__ import annotations

import newton
import warp as wp

_IDENTITY_QUAT = wp.quat(0.0, 0.0, 0.0, 1.0)


def apply_default_runtime_scene(builder: newton.ModelBuilder) -> None:
    """Add the reference ground plane and static obstacle used by v1 assembly examples."""

    builder.add_ground_plane()
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(1.0, 0.0, 2.0), _IDENTITY_QUAT),
        hx=0.5,
        hy=0.5,
        hz=0.5,
        label="power_reference_box",
    )
