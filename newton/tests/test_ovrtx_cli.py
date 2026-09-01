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
        self.assertEqual(args.ovrtx_warmup_frames, 40)

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
        ]
        with patch.object(sys, "argv", argv), patch("newton.viewer.ViewerOVRTX") as viewer_class:
            viewer, args = init(parser)

        self.assertIs(viewer, viewer_class.return_value)
        self.assertEqual(args.viewer, "ovrtx")
        viewer_class.assert_called_once_with(
            output_path="simulation.usd",
            render_output_path="render.png",
            width=800,
            height=600,
            camera_position=(4.0, -2.0, 3.0),
            camera_target=(0.0, 1.0, 0.5),
            render_mode="Minimal",
            warmup_frames=4,
            num_frames=100,
        )
