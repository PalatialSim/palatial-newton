# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import newton  # noqa: F401
import numpy as np
import warp as wp

from newton.examples.palatial.cable_presets import (
    get_anisotropic_cable_preset,
    list_anisotropic_cable_presets,
)
import newton.viewer as viewer
from newton.examples.palatial.example_palatial_cable import Example, _resolve_input_usd
from newton.examples.palatial.generate_palatial_cable_usd import author_cable_usd
from newton.palatial import (
    create_cable_quaternions,
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


def _assert_close_param(
    test: unittest.TestCase,
    params: dict[str, object],
    key: str,
    expected: object,
) -> None:
    actual = params[key]
    if isinstance(expected, str):
        test.assertEqual(str(actual), expected)
    elif isinstance(expected, int):
        test.assertEqual(int(actual), expected)
    else:
        test.assertAlmostEqual(float(actual), float(expected), places=6)


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
            self.assertAlmostEqual(float(params["compressStiffness"]), 654.0)
            self.assertAlmostEqual(float(params["compressDamping"]), 0.05)
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
            self.assertAlmostEqual(float(params["compressStiffness"]), 1.0e5)
            self.assertAlmostEqual(float(params["compressDamping"]), 0.0)
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
            self.assertEqual(
                bundle.model.joint_type.numpy().tolist(),
                [int(newton.JointType.ANISOTROPIC_CABLE)] * 2,
            )
            shape_types = bundle.model.shape_type.numpy().tolist()
            self.assertEqual(shape_types.count(int(newton.GeoType.BOX)), 3)
            self.assertEqual(shape_types.count(int(newton.GeoType.CAPSULE)), 0)
            self.assertEqual(bundle.model.joint_dof_dim.numpy().tolist(), [[1, 3], [1, 3]])
            np.testing.assert_allclose(
                bundle.model.joint_target_ke.numpy(),
                np.array([321.0, 12.0, 20.0, 9.0, 321.0, 12.0, 20.0, 9.0], dtype=np.float32),
                rtol=0.0,
                atol=1.0e-6,
            )
            np.testing.assert_allclose(
                bundle.model.joint_target_kd.numpy(),
                np.array([0.1, 0.2, 0.4, 0.6, 0.1, 0.2, 0.4, 0.6], dtype=np.float32),
                rtol=0.0,
                atol=1.0e-6,
            )

    def test_load_accepts_solver_vbd_palatial_alias(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            usd_path = Path(tmp_dir) / "test_cable_vbd_palatial.usda"
            _author_test_cable_stage(usd_path)

            stage = Usd.Stage.Open(str(usd_path))
            scene = stage.GetPrimAtPath("/physicsScene")
            scene.CreateAttribute("newton:solver", Sdf.ValueTypeNames.Token, custom=True).Set("vbd_palatial")
            stage.Save()

            bundle = load(str(usd_path), device="cpu")

            self.assertEqual(bundle.solver_name, "vbd_palatial")
            self.assertIs(bundle.solver.__class__, newton.solvers.SolverVBDPalatial)

    def test_load_accepts_solver_override_vbd_palatial_alias(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            usd_path = Path(tmp_dir) / "test_cable_vbd_palatial_override.usda"
            _author_test_cable_stage(usd_path)

            bundle = load(str(usd_path), solver_override="vbd_palatial", device="cpu")

            self.assertEqual(bundle.solver_name, "vbd_palatial")
            self.assertIs(bundle.solver.__class__, newton.solvers.SolverVBDPalatial)

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
            expected_quaternions = create_cable_quaternions(
                expected_points,
                cross_section_type="flatRect",
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
            expected_quaternions = create_cable_quaternions(
                expected_points,
                cross_section_type="roundSolid",
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
            expected_quaternions = create_cable_quaternions(
                expected_points,
                cross_section_type="roundSolid",
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

    def test_generator_authors_default_flat_rect_asset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            usd_path = Path(tmp_dir) / "generated_flat_rect.newton.usda"

            authored_path = author_cable_usd(usd_path)
            self.assertEqual(authored_path, usd_path.resolve())

            params = read_cable_params(str(authored_path))
            self.assertEqual(params["crossSectionType"], "flatRect")
            self.assertAlmostEqual(float(params["width"]), 0.012)
            self.assertAlmostEqual(float(params["thickness"]), 0.004)
            self.assertAlmostEqual(float(params["radius"]), 0.002)

            points = extract_cable_points(str(authored_path))
            self.assertEqual(len(points), 17)
            self.assertAlmostEqual(float(points[0][2]), 0.302, places=6)
            self.assertAlmostEqual(float(points[-1][0]), 1.5, places=6)
            stage = Usd.Stage.Open(str(authored_path))
            surface = UsdGeom.Mesh.Get(stage, "/Cable/Surface")
            self.assertTrue(surface)
            self.assertEqual(len(surface.GetPointsAttr().Get()), 170)
            self.assertEqual(len(surface.GetFaceVertexCountsAttr().Get()), 160)

            bundle = load(str(authored_path), device="cpu")
            self.assertEqual(bundle.body_type, "cable")
            self.assertEqual(bundle.solver_name, "vbd")
            self.assertEqual(bundle.fps, 120)
            self.assertEqual(bundle.model.body_count, 16)
            self.assertEqual(
                bundle.model.joint_type.numpy().tolist(),
                [int(newton.JointType.ANISOTROPIC_CABLE)] * 15,
            )
            shape_types = bundle.model.shape_type.numpy().tolist()
            self.assertEqual(shape_types.count(int(newton.GeoType.BOX)), 16)
            self.assertEqual(shape_types.count(int(newton.GeoType.CAPSULE)), 0)

    def test_create_cable_quaternions_rolls_flat_rect_ribbon_face_up(self):
        points = newton.utils.create_straight_cable_points(
            start=wp.vec3(0.0, 0.0, 0.0),
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=1.0,
            num_segments=2,
        )

        quaternion = create_cable_quaternions(
            points,
            cross_section_type="flatRect",
            twist_total=0.0,
        )[0]

        width_axis = wp.quat_rotate(quaternion, wp.vec3(1.0, 0.0, 0.0))
        thickness_axis = wp.quat_rotate(quaternion, wp.vec3(0.0, 1.0, 0.0))
        tangent_axis = wp.quat_rotate(quaternion, wp.vec3(0.0, 0.0, 1.0))

        np.testing.assert_allclose(
            np.array(width_axis, dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            atol=1.0e-6,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.array(thickness_axis, dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            atol=1.0e-6,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.array(tangent_axis, dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            atol=1.0e-6,
            rtol=0.0,
        )

    def test_generator_can_author_round_asset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            usd_path = Path(tmp_dir) / "generated_round.newton.usda"

            author_cable_usd(
                usd_path,
                cross_section_type="roundSolid",
                radius=0.01,
                length=2.0,
                segment_count=6,
                drop_height=0.4,
                twist_total=0.25,
                solver="vbd_palatial",
                solver_iterations=3,
                solver_substeps=4,
            )

            params = read_cable_params(str(usd_path))
            self.assertEqual(params["crossSectionType"], "roundSolid")
            self.assertAlmostEqual(float(params["radius"]), 0.01)
            self.assertAlmostEqual(float(params["length"]), 2.0)
            self.assertAlmostEqual(float(params["dropHeight"]), 0.4)
            self.assertAlmostEqual(float(params["twistTotal"]), 0.25)

            points = extract_cable_points(str(usd_path))
            self.assertEqual(len(points), 7)
            self.assertAlmostEqual(float(points[0][2]), 0.41, places=6)
            self.assertAlmostEqual(float(points[-1][0]), 2.0, places=6)
            stage = Usd.Stage.Open(str(usd_path))
            surface = UsdGeom.Mesh.Get(stage, "/Cable/Surface")
            self.assertTrue(surface)
            self.assertEqual(len(surface.GetPointsAttr().Get()), 56)
            self.assertEqual(len(surface.GetFaceVertexCountsAttr().Get()), 48)

            bundle = load(str(usd_path), device="cpu")
            self.assertEqual(bundle.solver_name, "vbd_palatial")
            self.assertIs(bundle.solver.__class__, newton.solvers.SolverVBDPalatial)
            self.assertEqual(bundle.model.body_count, 6)
            self.assertEqual(
                bundle.model.joint_type.numpy().tolist(),
                [int(newton.JointType.ANISOTROPIC_CABLE)] * 5,
            )
            shape_types = bundle.model.shape_type.numpy().tolist()
            self.assertEqual(shape_types.count(int(newton.GeoType.CAPSULE)), 6)

    def test_named_anisotropic_presets_generate_loadable_bundles(self):
        for preset_name in list_anisotropic_cable_presets():
            with self.subTest(preset=preset_name), tempfile.TemporaryDirectory() as tmp_dir:
                usd_path = Path(tmp_dir) / f"{preset_name}.newton.usda"
                preset = get_anisotropic_cable_preset(preset_name)

                author_cable_usd(usd_path, **preset)
                params = read_cable_params(str(usd_path))
                bundle = load(str(usd_path), device="cpu")

                _assert_close_param(self, params, "crossSectionType", preset["cross_section_type"])
                _assert_close_param(self, params, "length", preset["length"])
                _assert_close_param(self, params, "segmentCount", preset["segment_count"])
                _assert_close_param(self, params, "dropHeight", preset["drop_height"])
                _assert_close_param(self, params, "twistTotal", preset["twist_total"])
                _assert_close_param(self, params, "stretchStiffness", preset["stretch_stiffness"])
                _assert_close_param(self, params, "bendYStiffness", preset["bend_y_stiffness"])
                _assert_close_param(self, params, "bendZStiffness", preset["bend_z_stiffness"])
                _assert_close_param(self, params, "torsionStiffness", preset["torsion_stiffness"])
                if str(preset["cross_section_type"]) == "flatRect":
                    _assert_close_param(self, params, "width", preset["width"])
                    _assert_close_param(self, params, "thickness", preset["thickness"])
                else:
                    _assert_close_param(self, params, "radius", preset["radius"])

                self.assertEqual(bundle.body_type, "cable")
                self.assertEqual(bundle.solver_name, preset["solver"])
                self.assertEqual(bundle.fps, int(preset["fps"]))
                self.assertEqual(bundle.solver_params.get("iterations"), int(preset["solver_iterations"]))
                self.assertEqual(bundle.solver_params.get("substeps"), int(preset["solver_substeps"]))
                self.assertTrue(
                    all(
                        joint_type == int(newton.JointType.ANISOTROPIC_CABLE)
                        for joint_type in bundle.model.joint_type.numpy().tolist()
                    )
                )
                stiffness_values = (
                    float(params["bendYStiffness"]),
                    float(params["bendZStiffness"]),
                    float(params["torsionStiffness"]),
                )
                self.assertGreater(max(stiffness_values) - min(stiffness_values), 0.0)

    def test_unknown_anisotropic_preset_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown anisotropic cable preset"):
            get_anisotropic_cable_preset("not_a_real_preset")

    def test_example_can_generate_default_flat_rect_asset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch(
                "newton.examples.palatial.example_palatial_cable.tempfile.gettempdir",
                return_value=tmp_dir,
            ):
                usd_path = Path(
                    _resolve_input_usd(
                        None,
                        substeps=3,
                        solver_override="vbd_palatial",
                    )
                )

            self.assertTrue(usd_path.exists())
            params = read_cable_params(str(usd_path))
            self.assertEqual(params["crossSectionType"], "flatRect")
            self.assertAlmostEqual(float(params["width"]), 0.012)
            self.assertAlmostEqual(float(params["thickness"]), 0.004)

            bundle = load(str(usd_path), device="cpu")
            self.assertEqual(bundle.solver_name, "vbd_palatial")
            self.assertEqual(
                bundle.model.joint_type.numpy().tolist(),
                [int(newton.JointType.ANISOTROPIC_CABLE)] * 15,
            )

    def test_example_extra_drop_height_does_not_launch_unanchored_cable(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            usd_path = Path(tmp_dir) / "free_drop.newton.usda"
            author_cable_usd(
                usd_path,
                cross_section_type="flatRect",
                length=1.2,
                segment_count=12,
                drop_height=0.55,
                width=0.04,
                thickness=0.01,
                stretch_stiffness=1.0e5,
                stretch_damping=0.05,
                compress_stiffness=1.0e5,
                compress_damping=0.05,
                bend_y_stiffness=8.0e2,
                bend_y_damping=0.1,
                bend_z_stiffness=1.6e3,
                bend_z_damping=0.1,
                torsion_stiffness=4.0e2,
                torsion_damping=0.05,
                solver="vbd_palatial",
                solver_substeps=6,
            )

            example = Example(
                viewer.ViewerNull(),
                str(usd_path),
                device="cpu",
                solver_override="vbd_palatial",
                substeps=6,
                anchor_first=False,
                spin_rate=0.0,
                extra_drop_height=0.25,
                obstacle_box=True,
                obstacle_box_hx=0.14,
                obstacle_box_hy=0.10,
                obstacle_box_hz=0.08,
            )

            z0 = float(example.state_0.body_q.numpy()[:, 2].mean())
            example.step()
            z1 = float(example.state_0.body_q.numpy()[:, 2].mean())

            self.assertTrue(np.isfinite(z1))
            self.assertLess(abs(z1 - z0), 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
