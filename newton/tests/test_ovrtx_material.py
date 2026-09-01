# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import newton
from newton.ovrtx import OVRTXMaterial, author_material

try:
    from pxr import Sdf, Usd, UsdGeom, UsdShade
except ImportError:
    Sdf = Usd = UsdGeom = UsdShade = None


@unittest.skipIf(Usd is None, "USD Python bindings are not installed")
class TestOVRTXMaterial(unittest.TestCase):
    def test_authors_openpbr_and_preview_surface_from_one_material(self):
        stage = Usd.Stage.CreateInMemory()
        spec = OVRTXMaterial(
            color=(1.0, 0.12, 0.2),
            roughness=0.03,
            ior=1.53,
            preview_opacity=0.04,
            transmission=1.0,
            transmission_color=(1.0, 0.82, 0.84),
            coat=0.16,
            coat_color=(0.12, 0.95, 0.48),
            coat_roughness=0.05,
            thin_film=0.14,
            thin_film_thickness=0.45,
            thin_film_ior=1.4,
        )

        material = author_material(stage, Sdf.Path("/Looks/Lens"), spec)

        preview = UsdShade.Shader.Get(stage, "/Looks/Lens/PreviewSurface")
        openpbr = UsdShade.Shader.Get(stage, "/Looks/Lens/OpenPBR")
        self.assertTrue(material)
        self.assertEqual(preview.GetIdAttr().Get(), "UsdPreviewSurface")
        self.assertAlmostEqual(preview.GetInput("opacity").Get(), 0.04, places=6)
        self.assertEqual(openpbr.GetIdAttr().Get(), "ND_open_pbr_surface_surfaceshader")
        self.assertAlmostEqual(openpbr.GetInput("transmission_weight").Get(), 1.0, places=6)
        self.assertAlmostEqual(openpbr.GetInput("coat_weight").Get(), 0.16, places=6)
        self.assertAlmostEqual(openpbr.GetInput("thin_film_weight").Get(), 0.14, places=6)
        self.assertAlmostEqual(openpbr.GetInput("thin_film_thickness").Get(), 0.45, places=6)
        self.assertEqual(
            material.GetSurfaceOutput("mtlx").GetConnectedSource()[0].GetPath(),
            openpbr.GetPath(),
        )
        self.assertEqual(
            material.GetSurfaceOutput().GetConnectedSource()[0].GetPath(),
            preview.GetPath(),
        )

    def test_rejects_out_of_range_material_weights(self):
        with self.assertRaisesRegex(ValueError, "thin_film"):
            OVRTXMaterial(thin_film=1.1)

    def test_connects_one_staged_texture_to_both_render_contexts(self):
        stage = Usd.Stage.CreateInMemory()

        author_material(
            stage,
            "/Looks/Textured",
            OVRTXMaterial(),
            base_color_texture=Sdf.AssetPath("textures/albedo.png"),
        )

        preview_texture = UsdShade.Shader.Get(stage, "/Looks/Textured/BaseColorTexture")
        materialx_texture = UsdShade.Shader.Get(stage, "/Looks/Textured/BaseColorImage")
        preview = UsdShade.Shader.Get(stage, "/Looks/Textured/PreviewSurface")
        openpbr = UsdShade.Shader.Get(stage, "/Looks/Textured/OpenPBR")
        self.assertEqual(preview_texture.GetInput("file").Get().path, "textures/albedo.png")
        self.assertEqual(materialx_texture.GetInput("file").Get().path, "textures/albedo.png")
        self.assertEqual(materialx_texture.GetInput("file").GetAttr().GetColorSpace(), "sRGB")
        self.assertEqual(
            preview.GetInput("diffuseColor").GetConnectedSource()[0].GetPath(),
            preview_texture.GetPath(),
        )
        self.assertEqual(
            openpbr.GetInput("base_color").GetConnectedSource()[0].GetPath(),
            materialx_texture.GetPath(),
        )

    def test_openpbr_survives_usd_import_on_newton_mesh(self):
        stage = Usd.Stage.CreateInMemory()
        mesh_prim = UsdGeom.Mesh.Define(stage, "/Lens")
        mesh_prim.CreatePointsAttr([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
        mesh_prim.CreateFaceVertexCountsAttr([3])
        mesh_prim.CreateFaceVertexIndicesAttr([0, 1, 2])
        spec = OVRTXMaterial(
            color=(0.01, 0.9, 0.02),
            preview_opacity=0.03,
            transmission=1.0,
            transmission_color=(0.82, 0.98, 0.82),
            coat=0.16,
            thin_film=0.14,
            thin_film_thickness=0.45,
        )
        material = author_material(stage, "/Looks/Lens", spec)
        UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim()).Bind(material)

        mesh = newton.usd.get_mesh(mesh_prim.GetPrim())

        self.assertIsInstance(mesh.visual_material, newton.OpenPBRMaterial)
        self.assertAlmostEqual(mesh.visual_material.transmission, 1.0, places=6)
        self.assertAlmostEqual(mesh.visual_material.thin_film, 0.14, places=6)
        self.assertAlmostEqual(mesh.visual_material.thin_film_thickness, 0.45, places=6)
        self.assertAlmostEqual(mesh.opacity, 0.03, places=6)


if __name__ == "__main__":
    unittest.main()
