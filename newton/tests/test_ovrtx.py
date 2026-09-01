# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the optional OVRTX rendering path."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from newton import ovrtx as newton_ovrtx

try:
    from pxr import Sdf
except ImportError:
    Sdf = None


class TestOVRTXRendering(unittest.TestCase):
    """Keep the OVRTX adapter testable without an RTX GPU runtime."""

    def test_compose_render_stage_adds_camera_and_product(self):
        stage = newton_ovrtx._compose_render_stage(
            Path("/tmp/scene.usd"),
            "/Render/Newton",
            640,
            480,
            (3.0, -3.0, 2.5),
            (0.0, 0.0, 0.5),
            "RealTimePathTracing",
        )

        self.assertIn("@/tmp/scene.usd@", stage)
        self.assertIn('def Camera "NewtonCamera"', stage)
        self.assertIn('def RenderProduct "Newton"', stage)
        self.assertIn("uniform int2 resolution = (640, 480)", stage)
        self.assertIn('custom token omni:rtx:rendermode = "RealTimePathTracing"', stage)
        self.assertIn('uniform string sourceName = "LdrColor"', stage)

    @unittest.skipIf(Sdf is None, "USD Python bindings are not installed")
    def test_composed_render_stage_is_valid_usda(self):
        stage_text = newton_ovrtx._compose_render_stage(
            Path("/tmp/scene.usd"),
            "/Render/Newton",
            640,
            480,
            (3.0, -3.0, 2.5),
            (0.0, 0.0, 0.5),
            "RealTimePathTracing",
        )

        layer = Sdf.Layer.CreateAnonymous()
        self.assertTrue(layer.ImportFromString(stage_text))
        self.assertIsNotNone(layer.GetPrimAtPath("/NewtonCamera"))
        self.assertIsNotNone(layer.GetPrimAtPath("/Render/Newton"))

    def test_compose_render_stage_rejects_invalid_product_path(self):
        with self.assertRaisesRegex(ValueError, "form '/Render/<name>'"):
            newton_ovrtx._compose_render_stage(
                Path("/tmp/scene.usd"),
                "/Products/Newton",
                640,
                480,
                (3.0, -3.0, 2.5),
                (0.0, 0.0, 0.5),
                "RealTimePathTracing",
            )

    def test_render_usd_requires_existing_source(self):
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            newton_ovrtx.render_usd("/definitely/not/a/newton-recording.usd", "out.png")

    @patch("newton.ovrtx.np.from_dlpack", return_value=__import__("numpy").zeros((2, 2, 4), dtype="uint8"))
    @patch("newton.ovrtx._compose_render_stage", return_value="#usda 1.0")
    def test_render_usd_uses_attached_ovstage_lifecycle(self, compose_stage, from_dlpack):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "scene.usd"
            source.touch()
            output = Path(temp_dir) / "render.png"

            fake_image = MagicMock()
            fake_pillow = MagicMock()
            fake_pillow.Image.fromarray.return_value = fake_image
            fake_frame = MagicMock()
            fake_frame.render_vars = {"LdrColor": MagicMock()}
            fake_product = MagicMock(frames=[fake_frame])
            fake_renderer = MagicMock()
            fake_renderer.step.return_value = {"/Render/Newton": fake_product}
            fake_ovrtx = MagicMock()
            fake_ovrtx.Renderer.return_value = fake_renderer
            fake_ovrtx.Device.CPU = "cpu"
            fake_stage = MagicMock()
            fake_ovstage = MagicMock()
            fake_ovstage.Stage.return_value = fake_stage
            fake_ovstage.Scope.ALL = "all"

            with patch.dict(
                "sys.modules",
                {"ovrtx": fake_ovrtx, "ovstage": fake_ovstage, "PIL": fake_pillow},
            ):
                rendered = newton_ovrtx.render_usd(source, output, usd_time_code=75, warmup_frames=3)

        self.assertEqual(rendered, output.resolve())
        fake_renderer.attach_ovstage.assert_called_once_with(fake_stage)
        fake_ovstage.population.open_usd_from_string.assert_called_once()
        fake_ovstage.population.update_from_usd_time_async.assert_called_once_with(fake_stage, ordinal=2, time_code=75)
        fake_stage.advance_write_floor.assert_any_call(1, "all")
        fake_stage.advance_write_floor.assert_any_call(2, "all")
        self.assertEqual(fake_renderer.step.call_count, 3)
        fake_renderer.step.assert_called_with(render_products={"/Render/Newton"}, delta_time=1.0 / 60.0, ordinal=2)
        fake_frame.render_vars["LdrColor"].map.assert_called_once_with(device="cpu")
        fake_image.save.assert_called_once_with(output.resolve())
        fake_renderer.detach_ovstage.assert_called_once()
        fake_stage.destroy.assert_called_once()
        fake_renderer.destroy.assert_called_once()
        compose_stage.assert_called_once()
        from_dlpack.assert_called_once()
