# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import newton  # noqa: F401

from newton.palatial import (
    extract_cable_points,
    find_cable_centerline_prim_path,
    find_cable_prim_path,
    load,
    read_cable_params,
)
from newton.tests.unittest_utils import USD_AVAILABLE

if USD_AVAILABLE:
    try:
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
        from pxr import Usd as _Usd

        Usd: Any = _Usd
    except (ImportError, ModuleNotFoundError):
        Usd = None  # type: ignore[assignment]
else:
    Usd = None  # type: ignore[assignment]


def _create_custom_attribute(prim: Any, name: str, type_name: Any, value: object) -> None:
    prim.CreateAttribute(name, type_name, custom=True).Set(value)


def _author_test_cable_stage(output_path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(output_path))

    root = UsdGeom.Xform.Define(stage, "/Cable").GetPrim()
    stage.SetDefaultPrim(root)
    root.ApplyAPI("NewtonDeformableAPI")
    root.ApplyAPI("NewtonRodAPI")
    UsdShade.MaterialBindingAPI.Apply(root)
    _create_custom_attribute(root, "newton:deformable:enabled", Sdf.ValueTypeNames.Bool, True)
    _create_custom_attribute(root, "newton:deformable:simulationIntent", Sdf.ValueTypeNames.Token, "rod")
    _create_custom_attribute(root, "newton:rod:frameDefinition", Sdf.ValueTypeNames.Token, "parallelTransport")
    _create_custom_attribute(root, "newton:rod:closed", Sdf.ValueTypeNames.Bool, False)
    _create_custom_attribute(root, "newton:rod:crossSectionType", Sdf.ValueTypeNames.Token, "flatRect")
    _create_custom_attribute(root, "newton:rod:width", Sdf.ValueTypeNames.Float, 0.012)
    _create_custom_attribute(root, "newton:rod:thickness", Sdf.ValueTypeNames.Float, 0.004)
    _create_custom_attribute(root, "newton:rod:segmentCount", Sdf.ValueTypeNames.Int, 3)

    centerline = UsdGeom.BasisCurves.Define(stage, "/Cable/Centerline")
    centerline_prim = centerline.GetPrim()
    centerline_prim.ApplyAPI("NewtonRodAPI")
    centerline.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
    centerline.CreateCurveVertexCountsAttr().Set([4])
    centerline.CreatePointsAttr().Set(
        [
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(0.5, 0.0, 0.0),
            Gf.Vec3f(1.0, 0.0, 0.0),
            Gf.Vec3f(1.5, 0.0, 0.0),
        ]
    )
    _create_custom_attribute(centerline_prim, "newton:rod:length", Sdf.ValueTypeNames.Float, 1.5)

    material = UsdShade.Material.Define(stage, "/Materials/CableMaterial")
    material_prim = material.GetPrim()
    material_prim.ApplyAPI("NewtonRodMaterialAPI")
    _create_custom_attribute(material_prim, "newton:rod:density", Sdf.ValueTypeNames.Float, 500.0)
    _create_custom_attribute(material_prim, "newton:rod:stretchStiffness", Sdf.ValueTypeNames.Float, 321.0)
    _create_custom_attribute(material_prim, "newton:rod:stretchDamping", Sdf.ValueTypeNames.Float, 0.1)
    _create_custom_attribute(material_prim, "newton:rod:compressStiffness", Sdf.ValueTypeNames.Float, 654.0)
    _create_custom_attribute(material_prim, "newton:rod:compressDamping", Sdf.ValueTypeNames.Float, 0.05)
    _create_custom_attribute(material_prim, "newton:rod:bendYStiffness", Sdf.ValueTypeNames.Float, 12.0)
    _create_custom_attribute(material_prim, "newton:rod:bendYDamping", Sdf.ValueTypeNames.Float, 0.2)
    _create_custom_attribute(material_prim, "newton:rod:bendZStiffness", Sdf.ValueTypeNames.Float, 20.0)
    _create_custom_attribute(material_prim, "newton:rod:bendZDamping", Sdf.ValueTypeNames.Float, 0.4)
    _create_custom_attribute(material_prim, "newton:rod:torsionStiffness", Sdf.ValueTypeNames.Float, 9.0)
    _create_custom_attribute(material_prim, "newton:rod:torsionDamping", Sdf.ValueTypeNames.Float, 0.6)
    UsdShade.MaterialBindingAPI(root).Bind(material)

    scene = UsdPhysics.Scene.Define(stage, "/physicsScene").GetPrim()
    _create_custom_attribute(scene, "newton:timeStepsPerSecond", Sdf.ValueTypeNames.Int, 120)

    stage.Save()


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestPalatialCable(unittest.TestCase):
    """Tests for rod or cable schema registration and palatial loading."""

    def test_registered_rod_schemas_apply_to_expected_prims(self):
        stage = Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(stage, "/Cable").GetPrim()
        centerline = UsdGeom.BasisCurves.Define(stage, "/Cable/Centerline").GetPrim()
        material = UsdShade.Material.Define(stage, "/Materials/CableMaterial").GetPrim()

        root.ApplyAPI("NewtonDeformableAPI")
        root.ApplyAPI("NewtonRodAPI")
        centerline.ApplyAPI("NewtonRodAPI")
        material.ApplyAPI("NewtonRodMaterialAPI")

        self.assertTrue(root.HasAPI("NewtonDeformableAPI"))
        self.assertTrue(root.HasAPI("NewtonRodAPI"))
        self.assertTrue(centerline.HasAPI("NewtonRodAPI"))
        self.assertTrue(material.HasAPI("NewtonRodMaterialAPI"))
        self.assertTrue(root.GetAttribute("newton:rod:frameDefinition").IsValid())
        self.assertTrue(material.GetAttribute("newton:rod:bendYStiffness").IsValid())

    def test_cable_helpers_resolve_params_and_points(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            usd_path = Path(tmp_dir) / "test_cable.usda"
            _author_test_cable_stage(usd_path)

            self.assertEqual(find_cable_prim_path(str(usd_path)), "/Cable")
            self.assertEqual(find_cable_centerline_prim_path(str(usd_path)), "/Cable/Centerline")

            params = read_cable_params(str(usd_path))
            self.assertEqual(params["intent"], "rod")
            self.assertEqual(params["crossSectionType"], "flatRect")
            self.assertAlmostEqual(float(params["radius"]), 0.002)
            self.assertAlmostEqual(float(params["density"]), 500.0)
            self.assertAlmostEqual(float(params["stretchStiffness"]), 321.0)
            self.assertAlmostEqual(float(params["bendYStiffness"]), 12.0)
            self.assertAlmostEqual(float(params["bendZStiffness"]), 20.0)
            self.assertAlmostEqual(float(params["torsionDamping"]), 0.6)

            points = extract_cable_points(str(usd_path))
            self.assertEqual(len(points), 4)
            self.assertAlmostEqual(float(points[-1][0]), 1.5)

    def test_load_builds_cable_bundle(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            usd_path = Path(tmp_dir) / "test_cable.usda"
            _author_test_cable_stage(usd_path)

            bundle = load(str(usd_path), device="cpu")

            self.assertEqual(bundle.body_type, "cable")
            self.assertEqual(bundle.solver_name, "vbd")
            self.assertEqual(bundle.fps, 120)
            self.assertEqual(bundle.model.body_count, 3)
            self.assertEqual(bundle.model.joint_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
