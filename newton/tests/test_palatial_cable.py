# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import newton  # noqa: F401
import numpy as np
import warp as wp

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


def _apply_transform_ops(
    prim: Any,
    *,
    translate: tuple[float, float, float] | None = None,
    rotate_xyz: tuple[float, float, float] | None = None,
) -> None:
    xformable = UsdGeom.Xformable(prim)
    if translate is not None:
        xformable.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if rotate_xyz is not None:
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(*rotate_xyz))


def _author_test_cable_stage(
    output_path: Path,
    *,
    root_translate: tuple[float, float, float] | None = None,
    root_rotate_xyz: tuple[float, float, float] | None = None,
    centerline_translate: tuple[float, float, float] | None = None,
    centerline_rotate_xyz: tuple[float, float, float] | None = None,
) -> None:
    stage = Usd.Stage.CreateNew(str(output_path))

    root = UsdGeom.Xform.Define(stage, "/Cable").GetPrim()
    stage.SetDefaultPrim(root)
    _apply_transform_ops(root, translate=root_translate, rotate_xyz=root_rotate_xyz)
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
    _create_custom_attribute(root, "newton:rod:dropHeight", Sdf.ValueTypeNames.Float, 0.7)
    _create_custom_attribute(root, "newton:rod:twistTotal", Sdf.ValueTypeNames.Float, 0.25)

    centerline = UsdGeom.BasisCurves.Define(stage, "/Cable/Centerline")
    centerline_prim = centerline.GetPrim()
    _apply_transform_ops(centerline_prim, translate=centerline_translate, rotate_xyz=centerline_rotate_xyz)
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


def _author_test_fallback_cable_stage(
    output_path: Path,
    *,
    root_translate: tuple[float, float, float] | None = None,
    root_rotate_xyz: tuple[float, float, float] | None = None,
    centerline_translate: tuple[float, float, float] | None = None,
    centerline_rotate_xyz: tuple[float, float, float] | None = None,
) -> None:
    stage = Usd.Stage.CreateNew(str(output_path))

    root = UsdGeom.Xform.Define(stage, "/Cable").GetPrim()
    stage.SetDefaultPrim(root)
    _apply_transform_ops(root, translate=root_translate, rotate_xyz=root_rotate_xyz)
    root.ApplyAPI("NewtonDeformableAPI")
    root.ApplyAPI("NewtonRodAPI")
    UsdShade.MaterialBindingAPI.Apply(root)
    _create_custom_attribute(root, "newton:deformable:enabled", Sdf.ValueTypeNames.Bool, True)
    _create_custom_attribute(root, "newton:deformable:simulationIntent", Sdf.ValueTypeNames.Token, "rod")
    _create_custom_attribute(root, "newton:rod:frameDefinition", Sdf.ValueTypeNames.Token, "parallelTransport")
    _create_custom_attribute(root, "newton:rod:crossSectionType", Sdf.ValueTypeNames.Token, "roundSolid")
    _create_custom_attribute(root, "newton:rod:radius", Sdf.ValueTypeNames.Float, 0.01)
    _create_custom_attribute(root, "newton:rod:segmentCount", Sdf.ValueTypeNames.Int, 3)
    _create_custom_attribute(root, "newton:rod:length", Sdf.ValueTypeNames.Float, 1.5)
    _create_custom_attribute(root, "newton:rod:dropHeight", Sdf.ValueTypeNames.Float, 0.7)
    _create_custom_attribute(root, "newton:rod:twistTotal", Sdf.ValueTypeNames.Float, 0.5)

    centerline = UsdGeom.BasisCurves.Define(stage, "/Cable/Centerline")
    centerline_prim = centerline.GetPrim()
    _apply_transform_ops(centerline_prim, translate=centerline_translate, rotate_xyz=centerline_rotate_xyz)
    centerline_prim.ApplyAPI("NewtonRodAPI")
    centerline.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
    centerline.CreateCurveVertexCountsAttr().Set([4])

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


def _author_test_legacy_cable_stage(output_path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(output_path))

    root = UsdGeom.Xform.Define(stage, "/Cable").GetPrim()
    stage.SetDefaultPrim(root)
    root.ApplyAPI("NewtonDeformableAPI")
    root.ApplyAPI("NewtonRodAPI")
    _create_custom_attribute(root, "newton:deformable:enabled", Sdf.ValueTypeNames.Bool, True)
    _create_custom_attribute(root, "newton:deformable:simulationIntent", Sdf.ValueTypeNames.Token, "rod")
    _create_custom_attribute(root, "newton:rod:frameDefinition", Sdf.ValueTypeNames.Token, "parallelTransport")
    _create_custom_attribute(root, "newton:rod:isClosed", Sdf.ValueTypeNames.Bool, True)
    _create_custom_attribute(root, "newton:rod:verticesPerSegment", Sdf.ValueTypeNames.Int, 2)
    _create_custom_attribute(root, "newton:rod:length", Sdf.ValueTypeNames.Float, 1.5)
    _create_custom_attribute(root, "newton:rod:segmentCount", Sdf.ValueTypeNames.Int, 3)
    _create_custom_attribute(root, "newton:rod:radius", Sdf.ValueTypeNames.Float, 0.02)
    _create_custom_attribute(root, "newton:rodMaterial:density", Sdf.ValueTypeNames.Float, 700.0)
    _create_custom_attribute(root, "newton:rodMaterial:stretchStiffness", Sdf.ValueTypeNames.Float, 456.0)
    _create_custom_attribute(root, "newton:rodMaterial:bendStiffness", Sdf.ValueTypeNames.Float, 33.0)
    _create_custom_attribute(root, "newton:rodMaterial:damping", Sdf.ValueTypeNames.Float, 0.8)

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

    scene = UsdPhysics.Scene.Define(stage, "/physicsScene").GetPrim()
    _create_custom_attribute(scene, "newton:timeStepsPerSecond", Sdf.ValueTypeNames.Int, 120)

    stage.Save()


def _assert_same_quaternion(
    test: unittest.TestCase,
    actual_xyzw: np.ndarray,
    expected_xyzw: np.ndarray,
    *,
    atol: float = 1.0e-6,
) -> None:
    if float(np.dot(actual_xyzw, expected_xyzw)) < 0.0:
        expected_xyzw = -expected_xyzw
    np.testing.assert_allclose(actual_xyzw, expected_xyzw, atol=atol, rtol=0.0)


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
        self.assertTrue(root.GetAttribute("newton:rod:dropHeight").IsValid())
        self.assertTrue(root.GetAttribute("newton:rod:twistTotal").IsValid())
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
            self.assertAlmostEqual(float(params["dropHeight"]), 0.7)
            self.assertAlmostEqual(float(params["twistTotal"]), 0.25)
            self.assertAlmostEqual(float(params["bendYStiffness"]), 12.0)
            self.assertAlmostEqual(float(params["bendZStiffness"]), 20.0)
            self.assertAlmostEqual(float(params["torsionDamping"]), 0.6)

            points = extract_cable_points(str(usd_path))
            self.assertEqual(len(points), 4)
            self.assertAlmostEqual(float(points[-1][0]), 1.5)

    def test_cable_helpers_support_legacy_compat_attrs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            usd_path = Path(tmp_dir) / "test_cable_legacy.usda"
            _author_test_legacy_cable_stage(usd_path)

            params = read_cable_params(str(usd_path))

            self.assertTrue(bool(params["closed"]))
            self.assertEqual(int(params["verticesPerSegment"]), 2)
            self.assertAlmostEqual(float(params["density"]), 700.0)
            self.assertAlmostEqual(float(params["stretchStiffness"]), 456.0)
            self.assertAlmostEqual(float(params["bendYStiffness"]), 33.0)
            self.assertAlmostEqual(float(params["bendZStiffness"]), 33.0)
            self.assertAlmostEqual(float(params["torsionStiffness"]), 33.0)
            self.assertAlmostEqual(float(params["stretchDamping"]), 0.8)
            self.assertAlmostEqual(float(params["bendYDamping"]), 0.8)
            self.assertAlmostEqual(float(params["bendZDamping"]), 0.8)
            self.assertAlmostEqual(float(params["torsionDamping"]), 0.8)

    def test_extract_cable_points_world_space_applies_hierarchy_transforms(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            usd_path = Path(tmp_dir) / "test_cable_world_points.usda"
            _author_test_cable_stage(
                usd_path,
                root_translate=(1.0, 2.0, 3.0),
                root_rotate_xyz=(0.0, 0.0, 90.0),
                centerline_translate=(0.25, 0.5, 0.0),
            )

            local_points = extract_cable_points(str(usd_path))
            world_points = extract_cable_points(str(usd_path), world_space=True)

            self.assertEqual(len(world_points), 4)
            self.assertAlmostEqual(float(local_points[-1][0]), 1.5)

            stage = Usd.Stage.Open(str(usd_path))
            centerline_prim = stage.GetPrimAtPath("/Cable/Centerline")
            matrix = UsdGeom.Xformable(centerline_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            expected_last = matrix.Transform(Gf.Vec3d(1.5, 0.0, 0.0))

            self.assertAlmostEqual(float(world_points[0][0]), 0.5)
            self.assertAlmostEqual(float(world_points[0][1]), 2.25)
            self.assertAlmostEqual(float(world_points[0][2]), 3.0)
            self.assertAlmostEqual(float(world_points[-1][0]), float(expected_last[0]))
            self.assertAlmostEqual(float(world_points[-1][1]), float(expected_last[1]))
            self.assertAlmostEqual(float(world_points[-1][2]), float(expected_last[2]))

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

    def test_load_uses_world_space_authored_centerline_points(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            usd_path = Path(tmp_dir) / "test_cable_transformed.usda"
            _author_test_cable_stage(
                usd_path,
                root_translate=(1.0, 2.0, 3.0),
                root_rotate_xyz=(0.0, 0.0, 90.0),
                centerline_translate=(0.25, 0.5, 0.0),
            )

            bundle = load(str(usd_path), device="cpu")
            body_q = bundle.model.body_q.numpy()
            expected_points = extract_cable_points(str(usd_path), world_space=True)
            expected_quaternions = newton.utils.create_parallel_transport_cable_quaternions(
                expected_points,
                twist_total=0.25,
            )

            self.assertEqual(bundle.model.body_count, 3)
            for body_index in range(bundle.model.body_count):
                np.testing.assert_allclose(
                    body_q[body_index, :3],
                    np.array(expected_points[body_index], dtype=np.float32),
                    atol=1.0e-6,
                    rtol=0.0,
                )
                _assert_same_quaternion(
                    self,
                    body_q[body_index, 3:7],
                    np.array(expected_quaternions[body_index], dtype=np.float32),
                )

    def test_load_fallback_uses_drop_height_and_twist_total(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            usd_path = Path(tmp_dir) / "test_cable_fallback.usda"
            _author_test_fallback_cable_stage(usd_path)

            bundle = load(str(usd_path), device="cpu")
            body_q = bundle.model.body_q.numpy()

            expected_points = newton.utils.create_straight_cable_points(
                start=wp.vec3(0.0, 0.0, 0.7),
                direction=wp.vec3(1.0, 0.0, 0.0),
                length=1.5,
                num_segments=3,
            )
            expected_quaternions = newton.utils.create_parallel_transport_cable_quaternions(
                expected_points,
                twist_total=0.5,
            )

            self.assertEqual(bundle.model.body_count, 3)
            for body_index in range(bundle.model.body_count):
                np.testing.assert_allclose(
                    body_q[body_index, :3],
                    np.array(expected_points[body_index], dtype=np.float32),
                    atol=1.0e-6,
                    rtol=0.0,
                )
                _assert_same_quaternion(
                    self,
                    body_q[body_index, 3:7],
                    np.array(expected_quaternions[body_index], dtype=np.float32),
                )

    def test_load_fallback_applies_reference_transform(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            usd_path = Path(tmp_dir) / "test_cable_fallback_transformed.usda"
            _author_test_fallback_cable_stage(
                usd_path,
                root_translate=(1.0, 2.0, 3.0),
                root_rotate_xyz=(0.0, 0.0, 90.0),
                centerline_translate=(0.25, 0.5, 0.0),
            )

            bundle = load(str(usd_path), device="cpu")
            body_q = bundle.model.body_q.numpy()

            stage = Usd.Stage.Open(str(usd_path))
            centerline_prim = stage.GetPrimAtPath("/Cable/Centerline")
            matrix = UsdGeom.Xformable(centerline_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            local_points = newton.utils.create_straight_cable_points(
                start=wp.vec3(0.0, 0.0, 0.7),
                direction=wp.vec3(1.0, 0.0, 0.0),
                length=1.5,
                num_segments=3,
            )
            expected_points = [
                wp.vec3(*matrix.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))))
                for point in local_points
            ]
            expected_quaternions = newton.utils.create_parallel_transport_cable_quaternions(
                expected_points,
                twist_total=0.5,
            )

            self.assertEqual(bundle.model.body_count, 3)
            for body_index in range(bundle.model.body_count):
                np.testing.assert_allclose(
                    body_q[body_index, :3],
                    np.array(expected_points[body_index], dtype=np.float32),
                    atol=1.0e-6,
                    rtol=0.0,
                )
                _assert_same_quaternion(
                    self,
                    body_q[body_index, 3:7],
                    np.array(expected_quaternions[body_index], dtype=np.float32),
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
