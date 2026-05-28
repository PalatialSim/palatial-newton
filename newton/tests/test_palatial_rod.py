# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import tempfile
import textwrap
import unittest
from pathlib import Path

import newton
from newton.palatial import find_rod_prim_path, load, read_rod_params
from newton.tests.unittest_utils import USD_AVAILABLE


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
) -> str:
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
                float newton:rod:bendYStiffness = 12
                float newton:rod:bendYDamping = 1
                float newton:rod:bendZStiffness = 18
                float newton:rod:bendZDamping = 2
                float newton:rod:torsionStiffness = 30
                float newton:rod:torsionDamping = 5
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
                float newton:rod:bendYStiffness = 4
                float newton:rod:bendYDamping = 0.1
                float newton:rod:bendZStiffness = 5
                float newton:rod:bendZDamping = 0.2
                float newton:rod:torsionStiffness = 6
                float newton:rod:torsionDamping = 0.3
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


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestPalatialRod(unittest.TestCase):
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
            self.assertAlmostEqual(params["bendStiffness"], 20.0)
            self.assertAlmostEqual(params["bendDamping"], 8.0 / 3.0)

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
            usd_path = Path(tmpdir) / "rod_asset.usda"
            usd_path.write_text(_rod_test_stage(), encoding="utf-8")

            bundle = load(str(usd_path), device="cpu")

            self.assertEqual(bundle.body_type, "rod")
            self.assertEqual(bundle.fps, 120)
            expected_solver = "vbd" if getattr(newton.solvers, "SolverVBD", None) else "xpbd"
            self.assertEqual(bundle.solver_name, expected_solver)
            self.assertEqual(int(bundle.model.body_count), 6)
            self.assertEqual(int(bundle.model.joint_count), 4)
            self.assertEqual(bundle.state_in.body_q.shape[0], 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
