# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import tempfile
import textwrap
import unittest
import math
from pathlib import Path

import newton
import numpy as np
import warp as wp
from newton.examples.palatial.example_palatial_load import (
    _find_longest_rod_body_chain,
    _infer_rod_endpoint_bodies,
    _infer_rod_twist_targets,
)
from newton.palatial import find_rod_prim_path, load, read_rod_params
from newton.tests.unittest_utils import USD_AVAILABLE


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_conj(q):
    return (-q[0], -q[1], -q[2], q[3])


def _quat_angle(q) -> float:
    norm = math.sqrt(sum(float(v) * float(v) for v in q))
    w = abs(float(q[3])) / norm if norm > 0.0 else 1.0
    w = max(-1.0, min(1.0, w))
    return 2.0 * math.acos(w)


def _quat_rotate(q, v):
    qx, qy, qz, qw = q
    vec = np.asarray(v, dtype=np.float64)
    xyz = np.asarray((qx, qy, qz), dtype=np.float64)
    uv = np.cross(xyz, vec)
    uuv = np.cross(xyz, uv)
    return vec + 2.0 * (float(qw) * uv + uuv)


def _transform_point(body_q_row, local_point):
    return np.asarray(body_q_row[:3], dtype=np.float64) + _quat_rotate(body_q_row[3:7], local_point)


def _joint_relative_quat(body_q, joint_X_p, joint_X_c, parent: int, child: int):
    q_wp = _quat_mul(body_q[parent, 3:7], joint_X_p[3:7])
    q_wc = _quat_mul(body_q[child, 3:7], joint_X_c[3:7])
    return _quat_mul(_quat_conj(q_wp), q_wc)


def _format_point(point: tuple[float, float, float]) -> str:
    return f"({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f})"


def _point_cloud(center: tuple[float, float, float], radius: float, along: float = 0.004) -> list[tuple[float, float, float]]:
    x, y, z = center
    return [
        (x - along, y - radius, z - radius),
        (x + along, y - radius, z - radius),
        (x + along, y + radius, z - radius),
        (x - along, y + radius, z - radius),
        (x - along, y - radius, z + radius),
        (x + along, y - radius, z + radius),
        (x + along, y + radius, z + radius),
        (x - along, y + radius, z + radius),
    ]


def _visual_mesh_points(center: tuple[float, float, float], radius: float) -> list[tuple[float, float, float]]:
    along = max(2.5 * radius, 0.01)
    return _point_cloud(center, radius=radius, along=along)


def _mesh_block(name: str, points: list[tuple[float, float, float]]) -> str:
    point_list = ", ".join(_format_point(point) for point in points)
    face_counts = "4, 4, 4, 4, 4, 4"
    face_indices = "0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7"
    return textwrap.indent(
        textwrap.dedent(
            f"""
            def Mesh "{name}"
            {{
                point3f[] points = [{point_list}]
                int[] faceVertexCounts = [{face_counts}]
                int[] faceVertexIndices = [{face_indices}]
            }}
            """
        ).strip(),
        "        ",
    )


def _rigid_body_block(
    name: str,
    centers: list[tuple[float, float, float]],
    radius: float,
    visual_points: list[tuple[float, float, float]] | None = None,
) -> str:
    if visual_points is None:
        visual_points = _visual_mesh_points(centers[0], radius=radius)
    meshes = []
    for i, center in enumerate(centers):
        meshes.append(_mesh_block(f"Collider_{i}", _point_cloud(center, radius=radius)))
    mesh_text = "\n".join(meshes)
    return textwrap.dedent(
        f"""
        def Xform "{name}" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI"]
        )
        {{
            {_mesh_block(name, visual_points).strip()}

            def Xform "Colliders_{name}"
            {{
{mesh_text}
            }}
        }}
        """
    ).strip()


def _fixed_joint_block(name: str, body0: str, body1: str) -> str:
    return textwrap.dedent(
        f"""
        def PhysicsFixedJoint "{name}"
        {{
            rel physics:body0 = <{body0}>
            rel physics:body1 = <{body1}>
            point3f physics:localPos0 = (0, 0, 0)
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot0 = (1, 0, 0, 0)
            quatf physics:localRot1 = (1, 0, 0, 0)
        }}
        """
    ).strip()


def _rod_test_stage() -> str:
    ordered_centers = [
        (0.10, 0.00, 0.00),
        (0.28, 0.10, 0.02),
        (0.48, 0.18, 0.04),
        (0.70, 0.12, 0.06),
        (0.90, 0.02, 0.08),
    ]
    shuffled_centers = [ordered_centers[3], ordered_centers[1], ordered_centers[4], ordered_centers[0], ordered_centers[2]]

    return textwrap.dedent(
        f"""
        #usda 1.0
        (
            defaultPrim = "World"
            metersPerUnit = 1
            upAxis = "Z"
        )

        def Xform "World"
        {{
            def PhysicsScene "physicsScene"
            {{
                int newton:timeStepsPerSecond = 120
            }}

            def Material "RodMaterial" (
                prepend apiSchemas = ["NewtonRodMaterialAPI"]
            )
            {{
                float newton:rod:stretchStiffness = 1234
                float newton:rod:stretchDamping = 5.5
                float newton:rod:compressStiffness = 2345
                float newton:rod:bendStiffness = 20
                float newton:rod:bendDamping = 2.5
            }}

            def BasisCurves "RodGuide" (
                prepend apiSchemas = ["NewtonRodAPI", "MaterialBindingAPI"]
            )
            {{
                token type = "linear"
                int[] curveVertexCounts = [2]
                point3f[] points = [(0.000000, 0.000000, 0.000000), (1.000000, 0.000000, 0.100000)]
                float[] widths = [0.020000, 0.020000]
                int newton:rod:segmentCount = 5
                token newton:deformable:simulationIntent = "rod"
                rel material:binding = </World/RodMaterial>
            }}

            {_rigid_body_block("CablePath", shuffled_centers, radius=0.012)}

            {_rigid_body_block("CableJacket", shuffled_centers, radius=0.032)}

            {_rigid_body_block("ConnectorShell", [(1.200000, 0.000000, 0.100000)], radius=0.050)}
        }}
        """
    ).strip() + "\n"


def _rod_attachment_stage() -> str:
    right_boot_visual_points = _point_cloud((0.75, 0.00, 0.08), radius=0.030, along=0.40)
    right_boot_block = _rigid_body_block(
        "RightStrainReliefBoot",
        [(1.08, 0.00, 0.08)],
        radius=0.030,
        visual_points=right_boot_visual_points,
    )

    return textwrap.dedent(
        f"""
        #usda 1.0
        (
            defaultPrim = "World"
            metersPerUnit = 1
            upAxis = "Z"
        )

        def Xform "World"
        {{
            def PhysicsScene "physicsScene"
            {{
                int newton:timeStepsPerSecond = 120
            }}

            def Material "RodMaterial" (
                prepend apiSchemas = ["NewtonRodMaterialAPI"]
            )
            {{
                float newton:rod:stretchStiffness = 1234
                float newton:rod:stretchDamping = 5.5
                float newton:rod:bendStiffness = 20
                float newton:rod:bendDamping = 2.5
            }}

            def BasisCurves "RodGuide" (
                prepend apiSchemas = ["NewtonRodAPI", "MaterialBindingAPI"]
            )
            {{
                token type = "linear"
                int[] curveVertexCounts = [2]
                point3f[] points = [(0.000000, 0.000000, 0.000000), (1.000000, 0.000000, 0.100000)]
                float[] widths = [0.020000, 0.020000]
                int newton:rod:segmentCount = 5
                token newton:deformable:simulationIntent = "rod"
                rel material:binding = </World/RodMaterial>
            }}

            {_rigid_body_block("CablePath", [(0.10, 0.00, 0.00), (0.30, 0.10, 0.02), (0.50, 0.18, 0.04), (0.72, 0.12, 0.06), (0.90, 0.02, 0.08)], radius=0.012)}

            {_rigid_body_block("CableJacket", [(0.10, 0.00, 0.00), (0.30, 0.10, 0.02), (0.50, 0.18, 0.04), (0.72, 0.12, 0.06), (0.90, 0.02, 0.08)], radius=0.032)}

            {_rigid_body_block("LeftStrainReliefBoot", [(-0.08, 0.00, 0.08)], radius=0.030)}
            {_rigid_body_block("LeftPlugShell", [(-0.16, 0.00, 0.08)], radius=0.040)}
            {right_boot_block}
            {_rigid_body_block("RightPlugShell", [(1.16, 0.00, 0.08)], radius=0.040)}
            {_rigid_body_block("ConnectorShell", [(1.300000, 0.000000, 0.100000)], radius=0.050)}

            def Scope "Joints"
            {{
                {_fixed_joint_block("joint_cablejacket__leftstrainreliefboot", "/World/CableJacket", "/World/LeftStrainReliefBoot")}
                {_fixed_joint_block("joint_leftstrainreliefboot__leftplugshell", "/World/LeftStrainReliefBoot", "/World/LeftPlugShell")}
                {_fixed_joint_block("joint_cablejacket__rightstrainreliefboot", "/World/CableJacket", "/World/RightStrainReliefBoot")}
                {_fixed_joint_block("joint_rightstrainreliefboot__rightplugshell", "/World/RightStrainReliefBoot", "/World/RightPlugShell")}
            }}
        }}
        """
    ).strip() + "\n"


def _straight_rod_stage() -> str:
    return textwrap.dedent(
        """
        #usda 1.0
        (
            defaultPrim = "World"
            metersPerUnit = 1
            upAxis = "Z"
        )

        def Xform "World"
        {
            def PhysicsScene "physicsScene"
            {
                int newton:timeStepsPerSecond = 90
            }

            def Material "RodMaterial" (
                prepend apiSchemas = ["NewtonRodMaterialAPI"]
            )
            {
                float newton:rod:stretchStiffness = 999
                float newton:rod:stretchDamping = 1.5
                float newton:rod:bendStiffness = 5
                float newton:rod:bendDamping = 0.2
            }

            def BasisCurves "StraightRod" (
                prepend apiSchemas = ["NewtonRodAPI", "MaterialBindingAPI"]
            )
            {
                token type = "linear"
                int[] curveVertexCounts = [2]
                point3f[] points = [(0.000000, 0.000000, 0.000000), (1.000000, 0.000000, 0.000000)]
                float[] widths = [0.040000, 0.040000]
                int newton:rod:segmentCount = 4
                token newton:deformable:simulationIntent = "rod"
                rel material:binding = </World/RodMaterial>
            }
        }
        """
    ).strip() + "\n"


def _schema_attr_rod_stage() -> str:
    return textwrap.dedent(
        """
        #usda 1.0
        (
            defaultPrim = "World"
            metersPerUnit = 1
            upAxis = "Z"
        )

        def Xform "World"
        {
            def PhysicsScene "physicsScene"
            {
                int newton:timeStepsPerSecond = 90
            }

            def Material "RodMaterial" (
                prepend apiSchemas = ["NewtonRodMaterialAPI"]
            )
            {
                float newton:rod:density = 640
                float newton:rod:stretchStiffness = 100
                float newton:rod:stretchDamping = 0.2
                float newton:rod:compressStiffness = 300
                float newton:rod:compressDamping = 0.6
                float newton:rod:bendStiffness = 6
                float newton:rod:bendDamping = 0.3
            }

            def BasisCurves "FallbackRod" (
                prepend apiSchemas = ["NewtonRodAPI", "MaterialBindingAPI"]
            )
            {
                token type = "linear"
                int[] curveVertexCounts = [0]
                token newton:rod:frameDefinition = "parallelTransport"
                bool newton:rod:closed = false
                token newton:rod:crossSectionType = "flatRect"
                float newton:rod:radius = 0.03
                float newton:rod:width = 0.12
                float newton:rod:thickness = 0.02
                int newton:rod:segmentCount = 3
                float newton:rod:length = 1.5
                float newton:rod:dropHeight = 0.7
                float newton:rod:twistTotal = 0.45
                token newton:deformable:simulationIntent = "rod"
                rel material:binding = </World/RodMaterial>
            }
        }
        """
    ).strip() + "\n"


def _closed_rod_stage() -> str:
    points = [
        (0.30, 0.00, 0.40),
        (0.00, 0.30, 0.40),
        (-0.30, 0.00, 0.40),
        (0.00, -0.30, 0.40),
        (0.30, 0.00, 0.40),
    ]
    return textwrap.dedent(
        f"""
        #usda 1.0
        (
            defaultPrim = "World"
            metersPerUnit = 1
            upAxis = "Z"
        )

        def Xform "World"
        {{
            def PhysicsScene "physicsScene"
            {{
                int newton:timeStepsPerSecond = 90
            }}

            def Material "RodMaterial" (
                prepend apiSchemas = ["NewtonRodMaterialAPI"]
            )
            {{
                float newton:rod:density = 250
                float newton:rod:stretchStiffness = 200
                float newton:rod:compressDamping = 0.4
                float newton:rod:bendStiffness = 9
                float newton:rod:bendDamping = 0.4
            }}

            def BasisCurves "ClosedRod" (
                prepend apiSchemas = ["NewtonRodAPI", "MaterialBindingAPI"]
            )
            {{
                token type = "linear"
                int[] curveVertexCounts = [5]
                point3f[] points = [{", ".join(_format_point(point) for point in points)}]
                bool newton:rod:closed = true
                float newton:rod:radius = 0.02
                int newton:rod:segmentCount = 4
                token newton:deformable:simulationIntent = "rod"
                rel material:binding = </World/RodMaterial>
            }}
        }}
        """
    ).strip() + "\n"


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestPalatialRod(unittest.TestCase):
    def test_registered_rod_schema_restores_expected_attrs(self):
        from pxr import Usd, UsdGeom, UsdShade

        stage = Usd.Stage.CreateInMemory()
        rod = UsdGeom.BasisCurves.Define(stage, "/RodGuide").GetPrim()
        material = UsdShade.Material.Define(stage, "/RodMaterial").GetPrim()

        rod.ApplyAPI("NewtonRodAPI")
        material.ApplyAPI("NewtonRodMaterialAPI")

        for attr_name in (
            "newton:rod:frameDefinition",
            "newton:rod:closed",
            "newton:rod:crossSectionType",
            "newton:rod:radius",
            "newton:rod:width",
            "newton:rod:thickness",
            "newton:rod:segmentCount",
            "newton:rod:length",
            "newton:rod:dropHeight",
            "newton:rod:twistTotal",
        ):
            self.assertTrue(rod.GetAttribute(attr_name).IsValid(), attr_name)

        for attr_name in (
            "newton:rod:density",
            "newton:rod:stretchStiffness",
            "newton:rod:stretchDamping",
            "newton:rod:compressStiffness",
            "newton:rod:compressDamping",
            "newton:rod:bendStiffness",
            "newton:rod:bendDamping",
        ):
            self.assertTrue(material.GetAttribute(attr_name).IsValid(), attr_name)

    def test_find_longest_rod_body_chain(self):
        labels = [
            "/World/LeftStrainReliefBoot",
            "ShortRod_edge_body_0",
            "ShortRod_edge_body_1",
            "LongRod_edge_body_0",
            "LongRod_edge_body_1",
            "LongRod_edge_body_2",
            "/World/RightStrainReliefBoot",
        ]
        self.assertEqual(_find_longest_rod_body_chain(labels), [3, 4, 5])
        self.assertEqual(_infer_rod_endpoint_bodies(labels), [3, 5])

    def test_find_and_read_rod_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "rod_asset.usda"
            usd_path.write_text(_rod_test_stage(), encoding="utf-8")

            self.assertEqual(find_rod_prim_path(str(usd_path)), "/World/RodGuide")

            params = read_rod_params(str(usd_path))
            self.assertEqual(params["guidePrimPath"], "/World/RodGuide")
            self.assertEqual(params["centerlineSourcePath"], "/World/CablePath")
            self.assertEqual(params["radiusSourcePath"], "/World/CableJacket")
            self.assertEqual(params["segmentCount"], 5)
            self.assertEqual(params["intent"], "rod")
            self.assertEqual(params["frameDefinition"], "parallelTransport")
            self.assertFalse(params["closed"])
            self.assertEqual(params["crossSectionType"], "roundSolid")
            self.assertEqual(len(params["points"]), 6)
            self.assertEqual(len(params["widths"]), 2)
            self.assertAlmostEqual(params["widths"][0], 0.02, places=6)
            self.assertAlmostEqual(params["widths"][1], 0.02, places=6)
            self.assertAlmostEqual(params["points"][0][0], 0.0, places=6)
            self.assertAlmostEqual(params["points"][0][1], 0.0, places=6)
            self.assertAlmostEqual(params["points"][0][2], 0.0, places=6)
            self.assertAlmostEqual(params["points"][-1][0], 1.0, places=6)
            self.assertAlmostEqual(params["points"][-1][1], 0.0, places=6)
            self.assertAlmostEqual(params["points"][-1][2], 0.1, places=6)
            self.assertGreater(max(abs(point[1]) for point in params["points"]), 0.05)
            self.assertGreater(params["radius"], 0.02)
            self.assertLess(params["radius"], 0.07)
            self.assertAlmostEqual(params["stretchStiffness"], 1234.0)
            self.assertAlmostEqual(params["stretchDamping"], 5.5)
            self.assertAlmostEqual(params["compressStiffness"], 2345.0)
            self.assertAlmostEqual(params["axialStiffness"], 1234.0)
            self.assertAlmostEqual(params["axialDamping"], 5.5)
            self.assertAlmostEqual(params["bendStiffness"], 20.0)
            self.assertAlmostEqual(params["bendDamping"], 2.5)

    def test_read_restored_schema_attrs_drive_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "schema_attr_rod.usda"
            usd_path.write_text(_schema_attr_rod_stage(), encoding="utf-8")

            params = read_rod_params(str(usd_path))

            self.assertEqual(params["guidePrimPath"], "/World/FallbackRod")
            self.assertEqual(params["centerlineSourcePath"], "/World/FallbackRod")
            self.assertEqual(params["radiusSourcePath"], "/World/FallbackRod")
            self.assertEqual(params["frameDefinition"], "parallelTransport")
            self.assertFalse(params["closed"])
            self.assertEqual(params["crossSectionType"], "flatRect")
            self.assertEqual(params["segmentCount"], 3)
            self.assertAlmostEqual(params["radius"], 0.03)
            self.assertAlmostEqual(params["width"], 0.12)
            self.assertAlmostEqual(params["thickness"], 0.02)
            self.assertAlmostEqual(params["length"], 1.5)
            self.assertAlmostEqual(params["dropHeight"], 0.7)
            self.assertAlmostEqual(params["twistTotal"], 0.45)
            self.assertEqual(len(params["points"]), 4)
            self.assertAlmostEqual(params["points"][0][0], 0.0)
            self.assertAlmostEqual(params["points"][0][2], 0.7)
            self.assertAlmostEqual(params["points"][-1][0], 1.5)
            self.assertAlmostEqual(params["points"][-1][2], 0.7)
            self.assertAlmostEqual(params["density"], 640.0)
            self.assertAlmostEqual(params["effectiveDensity"], 640.0 * (0.12 * 0.02) / (math.pi * 0.03 * 0.03))
            self.assertAlmostEqual(params["stretchStiffness"], 100.0)
            self.assertAlmostEqual(params["compressStiffness"], 300.0)
            self.assertAlmostEqual(params["compressDamping"], 0.6)
            self.assertAlmostEqual(params["axialStiffness"], 100.0)
            self.assertAlmostEqual(params["axialDamping"], 0.2)
            self.assertAlmostEqual(params["bendStiffness"], 6.0)
            self.assertAlmostEqual(params["bendDamping"], 0.3)

    def test_load_restored_schema_attrs_drive_isotropic_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "schema_attr_rod.usda"
            usd_path.write_text(_schema_attr_rod_stage(), encoding="utf-8")

            bundle = load(str(usd_path), device="cpu")

            self.assertEqual(bundle.body_type, "rod")
            self.assertEqual(int(bundle.model.body_count), 3)
            self.assertEqual(int(bundle.model.joint_count), 2)

            body_q = bundle.state_in.body_q.numpy()
            for body_index, expected_x in enumerate((0.0, 0.5, 1.0)):
                self.assertAlmostEqual(float(body_q[body_index, 0]), expected_x, places=6)
                self.assertAlmostEqual(float(body_q[body_index, 1]), 0.0, places=6)
                self.assertAlmostEqual(float(body_q[body_index, 2]), 0.7, places=6)

            expected_points = [wp.vec3(0.0, 0.0, 0.7), wp.vec3(0.5, 0.0, 0.7), wp.vec3(1.0, 0.0, 0.7), wp.vec3(1.5, 0.0, 0.7)]
            expected_quaternions = newton.utils.create_parallel_transport_cable_quaternions(
                expected_points,
                twist_total=0.45,
            )
            for body_index in range(3):
                expected = np.array(expected_quaternions[body_index], dtype=np.float32)
                actual = body_q[body_index, 3:7]
                if float(np.dot(actual, expected)) < 0.0:
                    expected = -expected
                np.testing.assert_allclose(actual, expected, atol=1.0e-6, rtol=0.0)

            np.testing.assert_allclose(
                bundle.model.joint_target_ke.numpy(),
                np.array([100.0, 6.0, 100.0, 6.0], dtype=np.float32),
                atol=1.0e-6,
                rtol=0.0,
            )
            np.testing.assert_allclose(
                bundle.model.joint_target_kd.numpy(),
                np.array([0.2, 0.3, 0.2, 0.3], dtype=np.float32),
                atol=1.0e-6,
                rtol=0.0,
            )

            effective_density = 640.0 * (0.12 * 0.02) / (math.pi * 0.03 * 0.03)
            expected_mass = effective_density * ((4.0 / 3.0) * math.pi * 0.03**3 + math.pi * 0.03**2 * 0.5)
            self.assertAlmostEqual(float(bundle.model.body_mass.numpy()[0]), expected_mass, places=5)

    def test_load_closed_rod_uses_closed_flag_and_compat_damping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "closed_rod.usda"
            usd_path.write_text(_closed_rod_stage(), encoding="utf-8")

            params = read_rod_params(str(usd_path))
            self.assertTrue(params["closed"])
            self.assertAlmostEqual(params["axialDamping"], 0.4)

            bundle = load(str(usd_path), device="cpu")

            self.assertEqual(int(bundle.model.body_count), 4)
            self.assertEqual(int(bundle.model.joint_count), 4)
            np.testing.assert_allclose(
                bundle.model.joint_target_ke.numpy(),
                np.array([200.0, 9.0, 200.0, 9.0, 200.0, 9.0, 200.0, 9.0], dtype=np.float32),
                atol=1.0e-6,
                rtol=0.0,
            )
            np.testing.assert_allclose(
                bundle.model.joint_target_kd.numpy(),
                np.full(8, 0.4, dtype=np.float32),
                atol=1.0e-6,
                rtol=0.0,
            )

    def test_read_straight_rod_without_helper_meshes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "straight_rod.usda"
            usd_path.write_text(_straight_rod_stage(), encoding="utf-8")

            params = read_rod_params(str(usd_path))
            self.assertEqual(params["guidePrimPath"], "/World/StraightRod")
            self.assertEqual(params["centerlineSourcePath"], "/World/StraightRod")
            self.assertEqual(params["radiusSourcePath"], "/World/StraightRod")
            self.assertEqual(params["segmentCount"], 4)
            self.assertEqual(len(params["points"]), 5)
            self.assertAlmostEqual(params["radius"], 0.02, places=6)
            self.assertAlmostEqual(params["points"][0][0], 0.0, places=6)
            self.assertAlmostEqual(params["points"][1][0], 0.25, places=6)
            self.assertAlmostEqual(params["points"][2][0], 0.5, places=6)
            self.assertAlmostEqual(params["points"][3][0], 0.75, places=6)
            self.assertAlmostEqual(params["points"][4][0], 1.0, places=6)
            self.assertTrue(all(abs(point[1]) < 1.0e-6 for point in params["points"]))
            self.assertTrue(all(abs(point[2]) < 1.0e-6 for point in params["points"]))
            self.assertAlmostEqual(params["stretchStiffness"], 999.0)
            self.assertAlmostEqual(params["bendStiffness"], 5.0)
            self.assertAlmostEqual(params["bendDamping"], 0.2)

    def test_load_builds_rod_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "rod_attachment_asset.usda"
            usd_path.write_text(_rod_attachment_stage(), encoding="utf-8")

            bundle = load(str(usd_path), device="cpu")

            self.assertEqual(bundle.body_type, "rod")
            self.assertEqual(bundle.fps, 120)
            expected_solver = "vbd" if getattr(newton.solvers, "SolverVBD", None) else "xpbd"
            self.assertEqual(bundle.solver_name, expected_solver)
            self.assertEqual(int(bundle.model.body_count), 10)
            self.assertEqual(int(bundle.model.joint_count), 8)
            self.assertEqual(bundle.state_in.body_q.shape[0], 10)

            label_to_idx = {label: i for i, label in enumerate(bundle.model.body_label)}
            left_root = label_to_idx["/World/LeftStrainReliefBoot"]
            right_root = label_to_idx["/World/RightStrainReliefBoot"]
            rod_start = label_to_idx["RodGuide_edge_body_0"]
            rod_end = label_to_idx["RodGuide_edge_body_4"]
            body_q = bundle.state_in.body_q.numpy()
            rest_body_q = bundle.model.body_q.numpy()
            params = read_rod_params(str(usd_path))
            self.assertAlmostEqual(float(body_q[rod_start, 0]), params["points"][0][0], places=6)
            self.assertAlmostEqual(float(body_q[rod_start, 1]), params["points"][0][1], places=6)
            self.assertAlmostEqual(float(body_q[rod_start, 2]), params["points"][0][2], places=6)
            self.assertAlmostEqual(float(body_q[rod_end, 0]), params["points"][-2][0], places=6)
            self.assertAlmostEqual(float(body_q[rod_end, 1]), params["points"][-2][1], places=6)
            self.assertAlmostEqual(float(body_q[rod_end, 2]), params["points"][-2][2], places=6)

            parent = bundle.model.joint_parent.numpy()
            child = bundle.model.joint_child.numpy()
            joint_type = bundle.model.joint_type.numpy()
            joint_X_p = bundle.model.joint_X_p.numpy()
            joint_X_c = bundle.model.joint_X_c.numpy()
            pairs = {(int(parent[i]), int(child[i])) for i in range(int(bundle.model.joint_count))}
            pair_types = {
                (int(parent[i]), int(child[i])): int(joint_type[i])
                for i in range(int(bundle.model.joint_count))
            }
            self.assertIn((rod_start, left_root), pairs)
            self.assertIn((rod_end, right_root), pairs)
            self.assertNotIn((-1, left_root), pairs)
            self.assertNotIn((-1, right_root), pairs)
            self.assertEqual(pair_types[(rod_start, left_root)], int(newton.JointType.FIXED))
            self.assertEqual(pair_types[(rod_end, right_root)], int(newton.JointType.FIXED))
            for joint_idx, joint_label in enumerate(bundle.model.joint_label):
                if joint_label.endswith("__rod_attach"):
                    self.assertLess(max(abs(float(v)) for v in joint_X_c[joint_idx, :3]), 0.2)
            twist_targets = _infer_rod_twist_targets(bundle.model, body_q)
            self.assertEqual([target.body_index for target in twist_targets], [left_root, right_root])
            self.assertEqual(len(twist_targets), 2)
            for target in twist_targets:
                self.assertTrue(np.isfinite(target.local_axis).all())
                self.assertTrue(np.isfinite(target.local_pivot).all())
                self.assertTrue(np.isfinite(target.world_axis).all())
                self.assertTrue(np.isfinite(target.world_pivot).all())
                self.assertGreater(float(np.linalg.norm(target.local_axis)), 0.9)
                self.assertGreater(float(np.linalg.norm(target.world_axis)), 0.9)
                self.assertLess(
                    float(np.linalg.norm(_transform_point(body_q[target.body_index], target.local_pivot) - target.world_pivot)),
                    1.0e-6,
                )

            max_rest_cable_angle = 0.0
            max_initial_cable_angle = 0.0
            for joint_idx, joint_label in enumerate(bundle.model.joint_label):
                if "RodGuide_cable_" not in joint_label:
                    continue
                parent_idx = int(parent[joint_idx])
                child_idx = int(child[joint_idx])
                rest_relative = _joint_relative_quat(
                    rest_body_q,
                    joint_X_p[joint_idx],
                    joint_X_c[joint_idx],
                    parent_idx,
                    child_idx,
                )
                initial_relative = _joint_relative_quat(
                    body_q,
                    joint_X_p[joint_idx],
                    joint_X_c[joint_idx],
                    parent_idx,
                    child_idx,
                )
                max_rest_cable_angle = max(max_rest_cable_angle, _quat_angle(rest_relative))
                max_initial_cable_angle = max(max_initial_cable_angle, _quat_angle(initial_relative))
            self.assertLess(max_rest_cable_angle, 1.0e-5)
            self.assertGreater(max_initial_cable_angle, 0.05)

            for joint_idx, joint_label in enumerate(bundle.model.joint_label):
                if not joint_label.endswith("__rod_attach"):
                    continue
                parent_idx = int(parent[joint_idx])
                child_idx = int(child[joint_idx])
                rest_relative = _joint_relative_quat(
                    rest_body_q,
                    joint_X_p[joint_idx],
                    joint_X_c[joint_idx],
                    parent_idx,
                    child_idx,
                )
                initial_relative = _joint_relative_quat(
                    body_q,
                    joint_X_p[joint_idx],
                    joint_X_c[joint_idx],
                    parent_idx,
                    child_idx,
                )
                error = _quat_mul(initial_relative, _quat_conj(rest_relative))
                self.assertLess(_quat_angle(error), 1.0e-4)

            rod_shape_indices = [
                shape_idx
                for shape_idx, shape_label in enumerate(bundle.model.shape_label)
                if "RodGuide_edge_capsule_" in shape_label
            ]
            self.assertEqual(len(rod_shape_indices), 5)
            filter_pairs = set(bundle.model.shape_collision_filter_pairs)
            for i, shape_a in enumerate(rod_shape_indices):
                for shape_b in rod_shape_indices[i + 1:]:
                    self.assertIn((min(shape_a, shape_b), max(shape_a, shape_b)), filter_pairs)

            flags = bundle.model.shape_flags.numpy()
            shape_body = bundle.model.shape_body.numpy()
            shape_scale = bundle.model.shape_scale.numpy()
            visible_bit = int(newton.ShapeFlags.VISIBLE)
            collide_bit = int(newton.ShapeFlags.COLLIDE_SHAPES)
            proxy_indices = [
                shape_idx
                for shape_idx, shape_label in enumerate(bundle.model.shape_label)
                if shape_label.endswith("__contact_proxy")
            ]
            self.assertEqual(len(proxy_indices), 2)
            for shape_idx in proxy_indices:
                self.assertTrue(flags[shape_idx] & collide_bit)
                self.assertFalse(flags[shape_idx] & int(newton.ShapeFlags.COLLIDE_PARTICLES))
                self.assertFalse(flags[shape_idx] & visible_bit)
                self.assertTrue(all(float(axis) > 0.0 for axis in shape_scale[shape_idx]))

            right_boot_visual = next(
                shape_idx
                for shape_idx, shape_label in enumerate(bundle.model.shape_label)
                if shape_label == "/World/RightStrainReliefBoot/RightStrainReliefBoot"
            )
            right_boot_colliders = [
                shape_idx
                for shape_idx, shape_label in enumerate(bundle.model.shape_label)
                if shape_label.startswith("/World/RightStrainReliefBoot/Colliders_")
            ]
            self.assertFalse(flags[right_boot_visual] & visible_bit)
            self.assertTrue(right_boot_colliders)
            self.assertTrue(any(flags[shape_idx] & visible_bit for shape_idx in right_boot_colliders))
            for shape_idx in right_boot_colliders:
                self.assertFalse(flags[shape_idx] & collide_bit)

            attached_shape_names = (
                "LeftStrainReliefBoot",
                "LeftPlugShell",
                "RightStrainReliefBoot",
                "RightPlugShell",
            )
            for shape_idx, shape_label in enumerate(bundle.model.shape_label):
                if shape_idx in proxy_indices:
                    continue
                if any(name in shape_label for name in attached_shape_names):
                    self.assertFalse(flags[shape_idx] & int(newton.ShapeFlags.COLLIDE_SHAPES))

            rod_start_shapes = [
                shape_idx for shape_idx, body_idx in enumerate(shape_body) if int(body_idx) == rod_start
            ]
            rod_end_shapes = [shape_idx for shape_idx, body_idx in enumerate(shape_body) if int(body_idx) == rod_end]
            left_proxy = next(
                shape_idx
                for shape_idx in proxy_indices
                if bundle.model.shape_label[shape_idx] == "/World/LeftStrainReliefBoot__contact_proxy"
            )
            right_proxy = next(
                shape_idx
                for shape_idx in proxy_indices
                if bundle.model.shape_label[shape_idx] == "/World/RightStrainReliefBoot__contact_proxy"
            )
            self.assertEqual(int(shape_body[left_proxy]), left_root)
            self.assertEqual(int(shape_body[right_proxy]), right_root)
            for rod_shape in rod_start_shapes:
                self.assertIn((min(rod_shape, left_proxy), max(rod_shape, left_proxy)), filter_pairs)
            for rod_shape in rod_end_shapes:
                self.assertIn((min(rod_shape, right_proxy), max(rod_shape, right_proxy)), filter_pairs)

            self.assertLess(float(body_q[left_root, 0]), params["points"][0][0])
            self.assertGreater(float(body_q[right_root, 0]), params["points"][-1][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
