# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for selecting OVRTX from Newton's shared examples CLI."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from newton.examples import create_parser, init


class TestOVRTXCli(unittest.TestCase):
    """Verify argument parsing without importing the optional GPU runtime."""

    def test_parser_accepts_ovrtx_viewer(self):
        args = create_parser().parse_args(["--viewer", "ovrtx"])

        self.assertEqual(args.viewer, "ovrtx")
        self.assertEqual(args.ovrtx_output_path, "output.png")
        self.assertEqual(args.ovrtx_width, 1280)
        self.assertEqual(args.ovrtx_height, 720)
        self.assertEqual(args.ovrtx_camera_position, (3.0, -3.0, 2.5))
        self.assertEqual(args.ovrtx_camera_target, (0.0, 0.0, 0.5))
        self.assertEqual(args.ovrtx_render_mode, "RealTimePathTracing")
        self.assertEqual(args.ovrtx_warmup_frames, 4)
        self.assertEqual(args.ovrtx_samples_per_frame, 1)
        self.assertEqual(args.ovrtx_render_every, 1)
        self.assertIsNone(args.ovrtx_script)
        self.assertEqual(args.ovrtx_video_codec, "libx264")
        self.assertEqual(args.ovrtx_video_crf, 18)
        self.assertEqual(args.ovrtx_video_preset, "medium")

    def test_init_constructs_ovrtx_viewer_with_arguments(self):
        parser = create_parser()
        argv = [
            "newton-example",
            "--viewer",
            "ovrtx",
            "--output-path",
            "simulation.usd",
            "--ovrtx-output-path",
            "render.png",
            "--ovrtx-width",
            "800",
            "--ovrtx-height",
            "600",
            "--ovrtx-camera-position",
            "4",
            "-2",
            "3",
            "--ovrtx-camera-target",
            "0",
            "1",
            "0.5",
            "--ovrtx-render-mode",
            "Minimal",
            "--ovrtx-warmup-frames",
            "4",
            "--ovrtx-samples-per-frame",
            "2",
            "--ovrtx-frame-step",
            "3",
            "--ovrtx-script",
            "render_scene.py",
            "--ovrtx-video-codec",
            "h264_nvenc",
            "--ovrtx-video-crf",
            "20",
            "--ovrtx-video-preset",
            "fast",
        ]
        with patch.object(sys, "argv", argv), patch("newton.viewer.ViewerOVRTX") as viewer_class:
            viewer, args = init(parser)

        self.assertIs(viewer, viewer_class.return_value)
        self.assertEqual(args.viewer, "ovrtx")
        call = viewer_class.call_args
        self.assertEqual(call.kwargs["output_path"], "simulation.usd")
        self.assertEqual(call.kwargs["render_output_path"], "render.png")
        self.assertEqual(call.kwargs["camera_position"], (4.0, -2.0, 3.0))
        self.assertEqual(call.kwargs["camera_target"], (0.0, 1.0, 0.5))
        self.assertEqual(call.kwargs["fps"], 60)
        self.assertEqual(call.kwargs["num_frames"], 100)
        config = call.kwargs["config"]
        self.assertEqual(config.width, 800)
        self.assertEqual(config.height, 600)
        self.assertEqual(config.render_mode, "Minimal")
        self.assertEqual(config.render_every, 3)
        self.assertEqual(config.warmup_frames, 4)
        self.assertEqual(config.samples_per_frame, 2)
        self.assertEqual(config.script_path, "render_scene.py")
        self.assertEqual(config.video_codec, "h264_nvenc")
        self.assertEqual(config.video_crf, 20)
        self.assertEqual(config.video_preset, "fast")
