# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""OVRTX-backed USD recording viewer."""

from __future__ import annotations

from pathlib import Path

from ...ovrtx import render_usd
from ..core.types import override
from .viewer_usd import ViewerUSD


class ViewerOVRTX(ViewerUSD):
    """Record a Newton simulation to USD and render its final state with OVRTX.

    OVRTX owns rendering while :class:`ViewerUSD` remains the simulation-side
    exporter. This keeps Newton's USD output portable and lets OVRTX re-evaluate
    its time-sampled transforms at the requested simulation time.
    """

    def __init__(
        self,
        output_path: str,
        render_output_path: str,
        *,
        width: int = 1280,
        height: int = 720,
        camera_position: tuple[float, float, float] = (3.0, -3.0, 2.5),
        camera_target: tuple[float, float, float] = (0.0, 0.0, 0.5),
        render_mode: str = "RealTimePathTracing",
        warmup_frames: int = 40,
        fps: int = 60,
        up_axis: str = "Z",
        num_frames: int | None = 100,
    ):
        """Initialize the USD recorder and OVRTX render configuration.

        Args:
            output_path: Intermediate Newton USD recording path.
            render_output_path: Final OVRTX LDR PNG path.
            width: Render width in pixels.
            height: Render height in pixels.
            camera_position: Camera position in stage metres.
            camera_target: Point at which the camera is aimed in stage metres.
            render_mode: OVRTX camera render mode.
            warmup_frames: Number of renderer steps before PNG output.
            fps: Simulation and USD time-sampling frequency.
            up_axis: USD up axis, either ``"Y"`` or ``"Z"``.
            num_frames: Maximum number of simulation frames to record.
        """
        super().__init__(output_path=output_path, fps=fps, up_axis=up_axis, num_frames=num_frames)
        self.render_output_path = Path(render_output_path).expanduser().resolve()
        self.width = width
        self.height = height
        self.camera_position = camera_position
        self.camera_target = camera_target
        self.render_mode = render_mode
        self.warmup_frames = warmup_frames
        self._closed = False

    @override
    def close(self):
        """Save the recording and render its last authored USD timecode."""
        if self._closed:
            return
        self._closed = True
        super().close()
        output = render_usd(
            self.output_path,
            self.render_output_path,
            width=self.width,
            height=self.height,
            camera_position=self.camera_position,
            camera_target=self.camera_target,
            render_mode=self.render_mode,
            warmup_frames=self.warmup_frames,
            usd_time_code=self._frame_index,
            fps=self.fps,
        )
        print(f"OVRTX render saved in: {output}")
