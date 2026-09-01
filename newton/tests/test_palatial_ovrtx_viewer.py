# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for first-class OVRTX recording in the Palatial playback example."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.viewer.viewer_ovrtx import ViewerOVRTX
from newton.examples.palatial.example_palatial_load import _create_viewer, create_parser

try:
    from pxr import UsdGeom, UsdShade
except ImportError:
    UsdGeom = UsdShade = None


class TestPalatialOVRTXViewer(unittest.TestCase):
    def test_palatial_cli_routes_recording_to_ovrtx_at_simulation_rate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / "validation.mp4"
            args = create_parser().parse_args(
                [
                    "scene.usd",
                    "--viewer",
                    "ovrtx",
                    "--record-mp4",
                    str(video),
                    "--mp4-fps",
                    "60",
                    "--steps",
                    "120",
                ]
            )

            with patch("newton.viewer.ViewerOVRTX") as viewer_class:
                viewer = _create_viewer(args, fps=240)

        self.assertIs(viewer, viewer_class.return_value)
        call = viewer_class.call_args
        self.assertEqual(call.kwargs["output_path"], str(video.with_suffix(".usd").resolve()))
        self.assertEqual(call.kwargs["render_output_path"], str(video.resolve()))
        self.assertEqual(call.kwargs["fps"], 240)
        self.assertEqual(call.kwargs["num_frames"], 120)
        self.assertEqual(call.kwargs["config"].render_every, 4)

    def test_ovrtx_requires_the_existing_record_mp4_contract(self):
        args = create_parser().parse_args(["scene.usd", "--viewer", "ovrtx"])

        with self.assertRaisesRegex(ValueError, "requires --record-mp4"):
            _create_viewer(args, fps=60)

    def test_parser_accepts_explicit_camera_look_at(self):
        args = create_parser().parse_args(
            [
                "scene.usd",
                "--camera-position",
                "0",
                "-0.4",
                "0.08",
                "--camera-target",
                "0",
                "0.04",
                "0.02",
            ]
        )

        self.assertEqual(args.camera_position, [0.0, -0.4, 0.08])
        self.assertEqual(args.camera_target, [0.0, 0.04, 0.02])

    @unittest.skipIf(UsdGeom is None, "USD Python bindings are not installed")
    def test_textured_mesh_authors_uv_material_and_portable_texture(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            texture = root / "albedo.ppm"
            texture.write_bytes(b"P6\n2 2\n255\n" + bytes([240, 40, 20]) * 4)
            output = root / "recording.usd"
            viewer = ViewerOVRTX(str(output), num_frames=1)
            points = wp.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], dtype=wp.vec3)
            indices = wp.array([0, 1, 2], dtype=wp.int32)
            uvs = wp.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], dtype=wp.vec2)

            viewer.log_mesh("/geometry/textured", points, indices, uvs=uvs, texture=str(texture))
            viewer._author_render_stage()

            mesh = viewer.stage.GetPrimAtPath("/root/geometry/textured")
            st = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
            self.assertTrue(st)
            self.assertEqual(st.GetInterpolation(), UsdGeom.Tokens.vertex)

            shader = UsdShade.Shader(viewer.stage.GetPrimAtPath("/Materials/Mesh_0/BaseColorTexture"))
            asset = shader.GetInput("file").Get()
            self.assertEqual(asset.path, "recording_textures/mesh_000_albedo.ppm")
            self.assertTrue((root / asset.path).is_file())
            viewer.close()

    @unittest.skipIf(UsdShade is None, "USD Python bindings are not installed")
    def test_transparent_mesh_authors_preview_surface_opacity_and_ior(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "recording.usd"
            viewer = ViewerOVRTX(str(output), num_frames=1)
            points = wp.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], dtype=wp.vec3)
            indices = wp.array([0, 1, 2], dtype=wp.int32)

            viewer.log_mesh(
                "/geometry/lens",
                points,
                indices,
                color=(1.0, 0.0, 0.0),
                roughness=0.08,
                opacity=0.38,
                ior=1.49,
            )
            viewer._author_render_stage()

            shader = UsdShade.Shader(viewer.stage.GetPrimAtPath("/Materials/Mesh_0/PreviewSurface"))
            self.assertAlmostEqual(shader.GetInput("opacity").Get(), 0.38, places=6)
            self.assertAlmostEqual(shader.GetInput("ior").Get(), 1.49, places=6)
            np.testing.assert_allclose(shader.GetInput("diffuseColor").Get(), (1.0, 0.0, 0.0), atol=1e-6)
            self.assertFalse(viewer.stage.GetPrimAtPath("/Materials/Mesh_0/DisplayColor"))
            viewer.close()


if __name__ == "__main__":
    unittest.main()
