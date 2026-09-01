# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""OVRTX-backed USD recording viewer."""

from __future__ import annotations

from pathlib import Path

from ...ovrtx import render_usd
from ..core.types import override
from .viewer_usd import ViewerUSD


class ViewerOVRTX(ViewerUSD):
    """Record a Newton simulation as USD and render it through OVRTX.

    :class:`ViewerUSD` authors the complete, portable simulation stage. One
    attached OVRTX session then evaluates that stage's timeline. The render
    target is first-class: an image captures the final frame, a video renders
    the animation, and a suffix-free path receives a PNG frame sequence.
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
        samples_per_frame: int = 1,
        frame_step: int = 1,
        script_path: str | None = None,
        video_codec: str = "libx264",
        video_crf: int = 18,
        video_preset: str = "medium",
        fps: int = 60,
        up_axis: str = "Z",
        num_frames: int | None = 100,
    ):
        """Initialize the USD recorder and OVRTX render configuration.

        Args:
            output_path: Intermediate Newton USD recording path.
            render_output_path: OVRTX image, video, or frame-directory target.
            width: Render width in pixels.
            height: Render height in pixels.
            camera_position: Camera position in stage metres.
            camera_target: Point at which the camera is aimed in stage metres.
            render_mode: OVRTX camera render mode.
            warmup_frames: Renderer steps before the first captured frame.
            samples_per_frame: Renderer steps for each subsequent frame.
            frame_step: USD timeline sampling stride for video or frame output.
            script_path: Optional Python render-lifecycle script.
            video_codec: FFmpeg video codec for video targets.
            video_crf: FFmpeg constant-rate-factor quality value.
            video_preset: FFmpeg encoder preset.
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
        self.samples_per_frame = samples_per_frame
        self.frame_step = frame_step
        self.script_path = script_path
        self.video_codec = video_codec
        self.video_crf = video_crf
        self.video_preset = video_preset
        self._closed = False

    @override
    def close(self):
        """Save the stage and render the target from its authored timeline."""
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
            samples_per_frame=self.samples_per_frame,
            frame_start=0,
            frame_end=self._frame_index,
            frame_step=self.frame_step,
            fps=self.fps,
            script_path=self.script_path,
            video_codec=self.video_codec,
            video_crf=self.video_crf,
            video_preset=self.video_preset,
        )
        print(f"OVRTX output saved in: {output}")
