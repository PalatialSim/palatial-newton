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
    def test_splits_two_lenses_and_binds_transparent_red_cyan_materials(self):
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
            cyan_mesh = UsdGeom.Mesh.Get(result, "/World/part_3/AnaglyphLensCyan")
            self.assertTrue(red_mesh)
            self.assertTrue(cyan_mesh)
            self.assertEqual(len(red_mesh.GetFaceVertexCountsAttr().Get()), 1)
            self.assertEqual(len(cyan_mesh.GetFaceVertexCountsAttr().Get()), 1)
            self.assertEqual(
                UsdGeom.Imageable.Get(result, "/World/part_3/part_3_NewtonVisual").ComputeVisibility(),
                UsdGeom.Tokens.invisible,
            )

            red_shader = UsdShade.Shader.Get(result, "/World/Looks/AnaglyphRed/PreviewSurface")
            cyan_shader = UsdShade.Shader.Get(result, "/World/Looks/AnaglyphCyan/PreviewSurface")
            np.testing.assert_allclose(red_shader.GetInput("diffuseColor").Get(), (1.0, 0.005, 0.005), atol=1e-6)
            np.testing.assert_allclose(cyan_shader.GetInput("diffuseColor").Get(), (0.005, 0.9, 1.0), atol=1e-6)
            self.assertAlmostEqual(red_shader.GetInput("opacity").Get(), 0.38, places=6)
            self.assertAlmostEqual(cyan_shader.GetInput("ior").Get(), 1.49, places=6)


if __name__ == "__main__":
    unittest.main()
