# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the optional OVRTX stage renderer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from newton import ovrtx as newton_ovrtx
from newton._src.viewer.viewer_ovrtx import _transforms_to_matrices
from newton._src.viewer.viewer_usd import _usd_add_xform, _usd_set_xform

try:
    from pxr import Sdf, Usd, UsdGeom
except ImportError:
    Sdf = Usd = UsdGeom = None


class _RenderProductSetOutputs:
    """Match OVRTX's native indexing contract (which deliberately has no get)."""

    def __init__(self, values):
        self._values = values

    def __getitem__(self, key):
        return self._values[key]


def _fake_runtime(width: int = 2, height: int = 2):
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    mapped = MagicMock()
    ldr_color = MagicMock()
    ldr_color.map.return_value = mapped
    frame = MagicMock()
    frame.render_vars = {"LdrColor": ldr_color}
    product = MagicMock(frames=[frame])
    renderer = MagicMock()
    renderer.step.return_value = _RenderProductSetOutputs({"/Render/Newton": product})
    ovrtx = MagicMock()
    ovrtx.Renderer.return_value = renderer
    ovrtx.Device.CPU = "cpu"
    stage = MagicMock()
    ovstage = MagicMock()
    ovstage.Stage.return_value = stage
    ovstage.Scope.ALL = "all"
    return pixels, mapped, renderer, stage, ovrtx, ovstage


class TestOVRTXRendering(unittest.TestCase):
    """Exercise the public stage seam without requiring an RTX GPU."""

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

    def test_config_uses_exact_ovrtx_aov_names_without_an_output_mode(self):
        config = newton_ovrtx.OVRTXConfig(render_vars=("rgb", "depth", "normals", "semantic_segmentation"))

        self.assertFalse(hasattr(config, "output"))
        self.assertEqual(newton_ovrtx.RENDER_VARS["depth"], "DepthSD")
        self.assertEqual(newton_ovrtx.RENDER_VARS["normals"], "NormalSD")
        self.assertEqual(newton_ovrtx.RENDER_VARS["semantic_segmentation"], "SemanticSegmentation")

    @unittest.skipIf(Usd is None, "USD Python bindings are not installed")
    def test_live_transform_matrix_matches_usd_xform_ops(self):
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/Instance")
        _usd_add_xform(prim)
        xform = np.array([[1.0, 2.0, 3.0, 0.2, 0.3, 0.4, 0.84]], dtype=np.float64)
        xform[:, 3:] /= np.linalg.norm(xform[:, 3:], axis=1, keepdims=True)
        scales = np.array([[2.0, 3.0, 4.0]], dtype=np.float64)
        _usd_set_xform(prim, xform[0, :3], xform[0, 3:], scales[0], 0.0)

        expected = np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation(0.0), dtype=np.float64)
        np.testing.assert_allclose(_transforms_to_matrices(xform, scales)[0], expected, atol=1e-6)

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

    @patch("newton.ovrtx.np.from_dlpack")
    @patch("newton.ovrtx._save_image")
    def test_render_usd_reuses_attached_stage_and_converts_frames_to_seconds(self, save_image, from_dlpack):
        pixels, mapped, renderer, stage, fake_ovrtx, fake_ovstage = _fake_runtime()
        from_dlpack.return_value = pixels

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "scene.usd"
            source.touch()
            output = Path(temp_dir) / "render.png"
            with patch.dict("sys.modules", {"ovrtx": fake_ovrtx, "ovstage": fake_ovstage}):
                rendered = newton_ovrtx.render_usd(
                    source,
                    output,
                    width=2,
                    height=2,
                    usd_time_code=75,
                    warmup_frames=3,
                )

        self.assertEqual(rendered, output.resolve())
        renderer.attach_ovstage.assert_called_once_with(stage)
        fake_ovstage.population.open_usd_from_string.assert_called_once()
        fake_ovstage.population.update_from_usd_time_async.assert_called_once_with(
            stage,
            ordinal=2,
            time_code=1.25,
        )
        stage.advance_write_floor.assert_any_call(1, "all")
        stage.advance_write_floor.assert_any_call(2, "all")
        self.assertEqual(renderer.step.call_count, 3)
        renderer.step.assert_called_with(render_products={"/Render/Newton"}, delta_time=1.0 / 60.0, ordinal=2)
        mapped.unmap.assert_called_once()
        save_image.assert_called_once()
        self.assertEqual(save_image.call_args.args[0], output.resolve())
        np.testing.assert_array_equal(save_image.call_args.args[1], pixels)
        renderer.detach_ovstage.assert_called_once()
        stage.destroy.assert_called_once()
        renderer.destroy.assert_called_once()

    @patch("newton.ovrtx.np.from_dlpack")
    def test_stage_script_receives_lifecycle_and_can_transform_pixels(self, from_dlpack):
        pixels, _, _, _, fake_ovrtx, fake_ovstage = _fake_runtime()
        from_dlpack.return_value = pixels

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "scene.usd"
            source.touch()
            script = root / "render_script.py"
            script.write_text(
                "events = []\n"
                "def compose_stage(source_path, default_stage):\n"
                "    events.append(('compose', source_path.name))\n"
                "    return default_stage\n"
                "def on_stage_open(context):\n"
                "    events.append(('open', context.ordinal))\n"
                "    with context.mutation() as ordinal:\n"
                "        events.append(('mutation', ordinal))\n"
                "def before_frame(context):\n"
                "    events.append(('before', context.frame_index, context.time_seconds))\n"
                "def after_frame(context, pixels):\n"
                "    events.append(('after', context.frame_index))\n"
                "    result = pixels.copy()\n"
                "    result[..., 0] = 7\n"
                "    return result\n"
                "def on_stage_close(context):\n"
                "    events.append(('close', context.ordinal))\n",
                encoding="utf-8",
            )

            with patch.dict("sys.modules", {"ovrtx": fake_ovrtx, "ovstage": fake_ovstage}):
                backend = newton_ovrtx.OVRTXStage(source, width=2, height=2, script_path=script)
                with backend:
                    frame = backend.render_frame(30, steps=1)
                    script_module = backend._script

            self.assertEqual(frame.time_seconds, 0.5)
            self.assertTrue(np.all(frame.pixels[..., 0] == 7))
            self.assertEqual(
                script_module.events,
                [
                    ("compose", "scene.usd"),
                    ("open", 1),
                    ("mutation", 2),
                    ("before", 30, 0.5),
                    ("after", 30),
                    ("close", 3),
                ],
            )

    def test_render_target_is_inferred_without_an_output_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "scene.usd"
            source.touch()
            backend = newton_ovrtx.OVRTXStage(source, width=2, height=2)
            frame = newton_ovrtx.RenderedFrame(5, 5 / 60.0, np.zeros((2, 2, 4), dtype=np.uint8))

            with (
                patch.object(backend, "open"),
                patch.object(backend, "render_frame", return_value=frame) as render_frame,
                patch("newton.ovrtx._save_image") as save_image,
            ):
                output = backend.render(root / "final.png", frame_start=0, frame_end=5)

            self.assertEqual(output, (root / "final.png").resolve())
            render_frame.assert_called_once_with(5, steps=40)
            save_image.assert_called_once()

            with (
                patch.object(backend, "open"),
                patch.object(backend, "_render_video") as render_video,
            ):
                backend.render(root / "movie.mp4", frame_start=2, frame_end=6, frame_step=2)

            render_video.assert_called_once()
            self.assertEqual(list(render_video.call_args.args[1]), [2, 4, 6])

    def test_video_streams_raw_frames_to_ffmpeg_without_temporary_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "scene.usd"
            source.touch()
            output = root / "movie.mp4"
            output.write_bytes(b"encoded-video")
            backend = newton_ovrtx.OVRTXStage(source, width=2, height=2, warmup_frames=3)
            pixels = np.zeros((2, 2, 4), dtype=np.uint8)

            process = MagicMock()
            process.stdin = MagicMock()
            process.stderr.read.return_value = b""
            process.wait.return_value = 0
            process.poll.return_value = 0
            with (
                patch("newton.ovrtx.shutil.which", return_value="/usr/bin/ffmpeg"),
                patch("newton.ovrtx.subprocess.Popen", return_value=process) as popen,
                patch.object(
                    backend,
                    "render_frame",
                    side_effect=lambda index, steps: newton_ovrtx.RenderedFrame(index, index / 60.0, pixels),
                ) as render_frame,
            ):
                backend._render_video(output, range(0, 3), codec="libx264", crf=18, preset="medium")

            self.assertEqual(render_frame.call_args_list[0].kwargs["steps"], 3)
            self.assertEqual(render_frame.call_args_list[1].kwargs["steps"], 1)
            self.assertEqual(process.stdin.write.call_count, 3)
            self.assertTrue(all(len(call.args[0]) == pixels.nbytes for call in process.stdin.write.call_args_list))
            command = popen.call_args.args[0]
            self.assertIn("rawvideo", command)
            self.assertNotIn(".png", " ".join(command))
