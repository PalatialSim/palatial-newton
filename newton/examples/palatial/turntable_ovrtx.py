# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Looping hero-camera orbit for Newton OVRTX render scripting.

Pass this file to a Newton example with ``--ovrtx-script``. It binds the
renderer-owned camera once, then writes one transform per rendered frame.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from newton.ovrtx import camera_matrix

CAMERA_PATH = "/NewtonCamera"
TARGET = (0.0, 0.0, 0.02)
RADIUS = 0.42
HEIGHT = 0.10
ORBIT_AMPLITUDE_DEGREES = 25.0
LOOP_SECONDS = 4.0


class _TurntableState:
    camera_binding: Any = None


_STATE = _TurntableState()


def on_stage_open(context: Any) -> None:
    """Bind the renderer-owned camera's synthetic OVStage transform."""
    _STATE.camera_binding = context.bind_attribute(
        [CAMERA_PATH],
        "omni:xform",
        semantic=context.ovstage.AttributeSemantic.MATRIX,
    )


def before_frame(context: Any) -> None:
    """Sweep smoothly around the front view and return to a seamless loop."""
    if _STATE.camera_binding is None:
        raise RuntimeError("Turntable camera binding was not initialized")
    time_seconds = 0.0 if context.time_seconds is None else context.time_seconds
    phase = 2.0 * math.pi * time_seconds / LOOP_SECONDS
    angle = math.pi / 2.0 + math.radians(ORBIT_AMPLITUDE_DEGREES) * math.sin(phase)
    position = (RADIUS * math.cos(angle), RADIUS * math.sin(angle), HEIGHT)
    transform = camera_matrix(position, TARGET)[np.newaxis, ...]
    _STATE.camera_binding.write(transform)


def on_stage_close(context: Any) -> None:
    """Release the native query before OVStage is destroyed."""
    del context
    if _STATE.camera_binding is not None:
        _STATE.camera_binding.close()
        _STATE.camera_binding = None
