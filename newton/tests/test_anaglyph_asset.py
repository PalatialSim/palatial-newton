# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton.examples.palatial.author_anaglyph_glasses import author_anaglyph_glasses

try:
    from pxr import Usd, UsdGeom, UsdShade
except ImportError:
    Usd = UsdGeom = UsdShade = None


@unittest.skipIf(Usd is None, "USD Python bindings are not installed")
class TestAnaglyphAssetAuthoring(unittest.TestCase):
    def test_splits_two_lenses_and_binds_subtle_red_green_materials(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "source.usda"
            output_path = root / "anaglyph.usda"
            stage = Usd.Stage.CreateNew(str(source_path))
            UsdGeom.Xform.Define(stage, "/World")
            UsdGeom.Xform.Define(stage, "/World/part_3")
            source = UsdGeom.Mesh.Define(stage, "/World/part_3/part_3_NewtonVisual")
            source.CreatePointsAttr(
                [
                    (-2.0, 0.0, 0.0),
                    (-1.0, 0.0, 0.0),
                    (-1.5, 1.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (2.0, 0.0, 0.0),
                    (1.5, 1.0, 0.0),
                ]
            )
            source.CreateFaceVertexCountsAttr([3, 3])
            source.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 4, 5])
            stage.GetRootLayer().Save()

            author_anaglyph_glasses(source_path, output_path)

            result = Usd.Stage.Open(str(output_path))
            red_mesh = UsdGeom.Mesh.Get(result, "/World/part_3/AnaglyphLensRed")
            green_mesh = UsdGeom.Mesh.Get(result, "/World/part_3/AnaglyphLensGreen")
            self.assertTrue(red_mesh)
            self.assertTrue(green_mesh)
            self.assertEqual(len(red_mesh.GetFaceVertexCountsAttr().Get()), 1)
            self.assertEqual(len(green_mesh.GetFaceVertexCountsAttr().Get()), 1)
            self.assertEqual(
                UsdGeom.Imageable.Get(result, "/World/part_3/part_3_NewtonVisual").ComputeVisibility(),
                UsdGeom.Tokens.invisible,
            )

            red_shader = UsdShade.Shader.Get(result, "/World/Looks/AnaglyphRed/PreviewSurface")
            green_shader = UsdShade.Shader.Get(result, "/World/Looks/AnaglyphGreen/PreviewSurface")
            frame_shader = UsdShade.Shader.Get(result, "/World/Looks/material_plastic/PreviewSurface")
            np.testing.assert_allclose(red_shader.GetInput("diffuseColor").Get(), (0.95, 0.02, 0.015), atol=1e-6)
            np.testing.assert_allclose(green_shader.GetInput("diffuseColor").Get(), (0.01, 0.78, 0.035), atol=1e-6)
            np.testing.assert_allclose(frame_shader.GetInput("diffuseColor").Get(), (0.004, 0.005, 0.006), atol=1e-6)
            self.assertAlmostEqual(red_shader.GetInput("opacity").Get(), 0.06, places=6)
            self.assertAlmostEqual(green_shader.GetInput("roughness").Get(), 0.04, places=6)
            self.assertAlmostEqual(green_shader.GetInput("ior").Get(), 1.49, places=6)
            self.assertAlmostEqual(frame_shader.GetInput("roughness").Get(), 0.32, places=6)


if __name__ == "__main__":
    unittest.main()
