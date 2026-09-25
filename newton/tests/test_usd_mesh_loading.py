# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for USD mesh extraction from stage, path, URL, and prim sources."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import newton
import newton.usd
from newton.tests.unittest_utils import USD_AVAILABLE, assert_np_equal


def _create_referenced_mesh_stage(tmpdir: str) -> Path:
    """Create a USD stage with a referenced translated triangle mesh."""
    from pxr import Gf, Usd, UsdGeom

    asset_path = Path(tmpdir) / "asset.usda"
    asset_stage = Usd.Stage.CreateNew(str(asset_path))
    mesh = UsdGeom.Mesh.Define(asset_stage, "/Asset/Triangle")
    UsdGeom.Xformable(mesh.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(1.0, 0.0, 0.0),
            Gf.Vec3f(0.0, 1.0, 0.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    asset_stage.GetRootLayer().Save()

    stage_path = Path(tmpdir) / "marker.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    marker = UsdGeom.Xform.Define(stage, "/Marker")
    marker.GetPrim().GetReferences().AddReference("./asset.usda", "/Asset")
    stage.GetRootLayer().Save()
    return stage_path


def _define_triangle_mesh(stage, path="/Triangle"):
    """Define a simple triangle mesh prim on a USD stage."""
    from pxr import Gf, UsdGeom

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(1.0, 0.0, 0.0),
            Gf.Vec3f(0.0, 1.0, 0.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    return mesh


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUsdMeshHelpers(unittest.TestCase):
    """Tests for loading Newton meshes from USD source variants."""

    def test_concave_usd_native_import_preserves_corner_attributes(self):
        """Import concave USD files into native models without overlapping triangles or lost corner data."""
        from pxr import Gf, Sdf, Usd, UsdGeom

        # A valid concave outline, with a collinear corner as in authored key tops.
        outline = np.array([(3, 2, 0), (2, 2, 0), (2, 1, 0), (0, 1, 0), (0, 0, 0), (2, 0, 0), (3, 0, 0)])
        with tempfile.TemporaryDirectory() as directory:
            for face_normals in (False, True):
                for preserve_uvs in (False, True):
                    for left_handed in (False, True):
                        with self.subTest(normals=face_normals, preserve_uvs=preserve_uvs, left_handed=left_handed):
                            path = Path(directory) / f"concave-{face_normals}-{preserve_uvs}-{left_handed}.usda"
                            stage = Usd.Stage.CreateNew(str(path))
                            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
                            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
                            shape = UsdGeom.Mesh.Define(stage, "/Mesh")
                            shape.CreatePointsAttr([Gf.Vec3f(*map(float, point)) for point in outline])
                            shape.CreateFaceVertexCountsAttr([len(outline)])
                            shape.CreateFaceVertexIndicesAttr(list(range(len(outline))))
                            if left_handed:
                                shape.CreateOrientationAttr("leftHanded")
                            if face_normals:
                                shape.CreateNormalsAttr(
                                    [Gf.Vec3f(float(p[0]) / 10, float(p[1]) / 10, 1) for p in outline]
                                )
                                shape.SetNormalsInterpolation("faceVarying")
                            uv = UsdGeom.PrimvarsAPI(shape).CreatePrimvar(
                                "st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying"
                            )
                            uv.Set([Gf.Vec2f(float(p[0]) / 3, float(p[1]) / 2) for p in outline])
                            convex = UsdGeom.Mesh.Define(stage, "/Convex")
                            convex.CreatePointsAttr([(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])
                            convex.CreateFaceVertexCountsAttr([4])
                            convex.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
                            degenerate_outlines = {
                                "RepeatedCorner": [
                                    (0, 0, 2),
                                    (1, 0, 2),
                                    (1, 0, 2),
                                    (1, 1, 2),
                                    (0.4, 0.4, 2),
                                    (0, 1, 2),
                                ],
                                "ZeroArea": [(0, 0, 3), (0, 0, 4), (0.01, 1, 3), (0.01, 1, 4)],
                            }
                            for name, boundary in degenerate_outlines.items():
                                degenerate = UsdGeom.Mesh.Define(stage, f"/{name}")
                                degenerate.CreatePointsAttr(boundary)
                                degenerate.CreateFaceVertexCountsAttr([len(boundary)])
                                degenerate.CreateFaceVertexIndicesAttr(list(range(len(boundary))))
                            stage.GetRootLayer().Save()
                            imported_stage = Usd.Stage.Open(str(path))
                            convex_mesh = newton.usd.get_mesh(
                                imported_stage.GetPrimAtPath("/Convex"), compute_inertia=False
                            )
                            assert_np_equal(convex_mesh.indices, np.array([0, 1, 2, 0, 2, 3]))
                            for name, boundary in degenerate_outlines.items():
                                with self.assertLogs("newton", level="WARNING"):
                                    degenerate_mesh = newton.usd.get_mesh(
                                        imported_stage.GetPrimAtPath(f"/{name}"), compute_inertia=False
                                    )
                                expected_fan = [[0, corner, corner + 1] for corner in range(1, len(boundary) - 1)]
                                assert_np_equal(degenerate_mesh.indices, np.asarray(expected_fan).reshape(-1))
                            mesh, corners = newton.usd.get_mesh(
                                imported_stage.GetPrimAtPath("/Mesh"),
                                load_uvs=True,
                                load_normals=True,
                                preserve_facevarying_uvs=preserve_uvs,
                                return_uv_indices=True,
                                compute_inertia=False,
                            )
                            vertices = np.asarray(mesh.vertices)
                            indices = np.asarray(mesh.indices).reshape(-1, 3)
                            triangles = vertices[indices]
                            areas = (
                                np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])[:, 2] / 2
                            )
                            self.assertTrue(np.all(areas < 0) if left_handed else np.all(areas > 0))
                            self.assertAlmostEqual(float(abs(areas).sum()), 4.0)
                            # UVs encode source position, so any mismapped polygon corner fails.
                            assert_np_equal(
                                np.asarray(mesh.uvs)[corners],
                                vertices[indices.reshape(-1), :2] / np.array([3, 2], dtype=np.float32),
                            )
                            if face_normals:
                                expected_normals = np.column_stack((vertices[:, :2] / 10, np.ones(len(vertices))))
                                expected_normals /= np.linalg.norm(expected_normals, axis=1, keepdims=True)
                                assert_np_equal(np.asarray(mesh.normals), expected_normals, tol=1e-6)
                            builder = newton.ModelBuilder()
                            builder.add_usd(str(path))
                            model = builder.finalize(device="cpu")
                            self.assertEqual(model.shape_count, 4)
                            stage = None
                            path.unlink()

    def test_get_mesh_accepts_usd_file_with_reference(self):
        """Load a mesh from a USD file containing a referenced asset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stage_path = _create_referenced_mesh_stage(tmpdir)

            mesh = newton.usd.get_mesh(stage_path, root_path="/Marker", compute_inertia=False)

        self.assertIsInstance(mesh, newton.Mesh)
        assert_np_equal(
            mesh.vertices,
            np.array(
                [
                    [1.0, 2.0, 3.0],
                    [2.0, 2.0, 3.0],
                    [1.0, 3.0, 3.0],
                ],
                dtype=np.float32,
            ),
        )
        assert_np_equal(mesh.indices, np.array([0, 1, 2], dtype=np.int32))

    def test_get_mesh_accepts_usd_stage_handle(self):
        """Load a mesh from an already-open USD stage handle."""
        from pxr import Usd

        with tempfile.TemporaryDirectory() as tmpdir:
            stage_path = _create_referenced_mesh_stage(tmpdir)
            stage = Usd.Stage.Open(str(stage_path), Usd.Stage.LoadAll)

            mesh = newton.usd.get_mesh(stage, root_path="/Marker", compute_inertia=False)

        self.assertIsInstance(mesh, newton.Mesh)
        self.assertEqual(len(mesh.vertices), 3)
        self.assertEqual(len(mesh.indices), 3)

    def test_get_mesh_rejects_http_urls(self):
        """Reject cleartext HTTP USD asset URLs."""
        with self.assertRaisesRegex(ValueError, "HTTP USD URLs are not supported"):
            newton.usd.get_mesh("http://example.com/marker.usda", compute_inertia=False)

    def test_get_mesh_prim_source_keeps_authored_units(self):
        """Keep authored coordinates when loading a single mesh prim."""
        from pxr import Gf, Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
        mesh = UsdGeom.Mesh.Define(stage, "/Triangle")
        mesh.CreatePointsAttr(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(100.0, 0.0, 0.0),
                Gf.Vec3f(0.0, 100.0, 0.0),
            ]
        )
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])

        result = newton.usd.get_mesh(mesh.GetPrim(), compute_inertia=False)

        assert_np_equal(result.vertices[1], np.array([100.0, 0.0, 0.0], dtype=np.float32))

    def test_get_mesh_accepts_legacy_prim_keyword(self):
        """Keep ``prim=`` working for compatibility with the existing API."""
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        mesh_prim = _define_triangle_mesh(stage).GetPrim()

        mesh = newton.usd.get_mesh(prim=mesh_prim, compute_inertia=False)

        self.assertIsInstance(mesh, newton.Mesh)
        assert_np_equal(mesh.indices, np.array([0, 1, 2], dtype=np.int32))

    def test_mesh_create_from_usd_accepts_legacy_prim_keyword(self):
        """Keep ``Mesh.create_from_usd(prim=...)`` working."""
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        mesh_prim = _define_triangle_mesh(stage).GetPrim()

        mesh = newton.Mesh.create_from_usd(prim=mesh_prim, compute_inertia=False)

        self.assertIsInstance(mesh, newton.Mesh)
        assert_np_equal(mesh.vertices[1], np.array([1.0, 0.0, 0.0], dtype=np.float32))

    def test_get_mesh_rejects_source_and_legacy_prim_keyword(self):
        """Reject ambiguous calls that provide both source names."""
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        mesh_prim = _define_triangle_mesh(stage).GetPrim()

        with self.assertRaisesRegex(TypeError, "received both 'source' and legacy 'prim'"):
            newton.usd.get_mesh(mesh_prim, prim=mesh_prim, compute_inertia=False)

    def test_mesh_create_from_usd_rejects_source_and_legacy_prim_keyword(self):
        """Reject ambiguous factory calls that provide both source names."""
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        mesh_prim = _define_triangle_mesh(stage).GetPrim()

        with self.assertRaisesRegex(TypeError, "received both 'source' and legacy 'prim'"):
            newton.Mesh.create_from_usd(mesh_prim, prim=mesh_prim, compute_inertia=False)

    def test_get_mesh_merges_multiple_mesh_prims(self):
        """Merge multiple mesh prims under a selected root."""
        from pxr import Gf, Usd, UsdGeom

        with tempfile.TemporaryDirectory() as tmpdir:
            stage_path = Path(tmpdir) / "multi.usda"
            stage = Usd.Stage.CreateNew(str(stage_path))
            UsdGeom.Xform.Define(stage, "/Root")
            for name, tx in (("A", 0.0), ("B", 2.0)):
                mesh = UsdGeom.Mesh.Define(stage, f"/Root/{name}")
                UsdGeom.Xformable(mesh.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(tx, 0.0, 0.0))
                mesh.CreatePointsAttr(
                    [
                        Gf.Vec3f(0.0, 0.0, 0.0),
                        Gf.Vec3f(1.0, 0.0, 0.0),
                        Gf.Vec3f(0.0, 1.0, 0.0),
                    ]
                )
                mesh.CreateFaceVertexCountsAttr([3])
                mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
            stage.GetRootLayer().Save()

            mesh = newton.usd.get_mesh(stage_path, root_path="/Root", compute_inertia=False)

        self.assertEqual(len(mesh.vertices), 6)
        assert_np_equal(mesh.indices, np.array([0, 1, 2, 3, 4, 5], dtype=np.int32))
        assert_np_equal(mesh.vertices[3:], np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]]))

    def test_get_mesh_rejects_preserved_facevarying_uvs_for_merged_sources(self):
        """Reject merged-source loads that request face-varying UV preservation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stage_path = _create_referenced_mesh_stage(tmpdir)

            with self.assertRaisesRegex(ValueError, "preserve_facevarying_uvs is not supported"):
                newton.usd.get_mesh(
                    stage_path,
                    root_path="/Marker",
                    preserve_facevarying_uvs=True,
                    compute_inertia=False,
                )

    def test_get_mesh_applies_root_relative_transform_and_stage_units(self):
        """Apply root-relative transforms and authored stage units."""
        from pxr import Gf, Usd, UsdGeom

        with tempfile.TemporaryDirectory() as tmpdir:
            stage_path = Path(tmpdir) / "units.usda"
            stage = Usd.Stage.CreateNew(str(stage_path))
            UsdGeom.SetStageMetersPerUnit(stage, 0.01)
            root = UsdGeom.Xform.Define(stage, "/Root")
            root.AddTranslateOp().Set(Gf.Vec3d(1000.0, 0.0, 0.0))
            mesh = UsdGeom.Mesh.Define(stage, "/Root/Triangle")
            UsdGeom.Xformable(mesh.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(100.0, 0.0, 0.0))
            mesh.CreatePointsAttr(
                [
                    Gf.Vec3f(0.0, 0.0, 0.0),
                    Gf.Vec3f(100.0, 0.0, 0.0),
                    Gf.Vec3f(0.0, 100.0, 0.0),
                ]
            )
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
            stage.GetRootLayer().Save()

            mesh = newton.usd.get_mesh(stage_path, root_path="/Root", compute_inertia=False)

        assert_np_equal(
            mesh.vertices,
            np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32),
        )

    def test_get_mesh_flips_winding_for_negative_scale(self):
        """Flip triangle winding when a merged transform mirrors handedness."""
        from pxr import Gf, Usd, UsdGeom

        with tempfile.TemporaryDirectory() as tmpdir:
            stage_path = Path(tmpdir) / "mirror.usda"
            stage = Usd.Stage.CreateNew(str(stage_path))
            UsdGeom.Xform.Define(stage, "/Root")
            mesh = UsdGeom.Mesh.Define(stage, "/Root/Triangle")
            UsdGeom.Xformable(mesh.GetPrim()).AddScaleOp().Set(Gf.Vec3d(-1.0, 1.0, 1.0))
            mesh.CreatePointsAttr(
                [
                    Gf.Vec3f(0.0, 0.0, 0.0),
                    Gf.Vec3f(1.0, 0.0, 0.0),
                    Gf.Vec3f(0.0, 1.0, 0.0),
                ]
            )
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
            stage.GetRootLayer().Save()

            mesh = newton.usd.get_mesh(stage_path, root_path="/Root", compute_inertia=False)

        assert_np_equal(mesh.indices, np.array([0, 2, 1], dtype=np.int32))
        assert_np_equal(mesh.vertices[1], np.array([-1.0, 0.0, 0.0], dtype=np.float32))

    def test_get_mesh_transforms_normals_with_rotation(self):
        """Transform authored normals with the same row-vector convention as points."""
        from pxr import Gf, Usd, UsdGeom

        with tempfile.TemporaryDirectory() as tmpdir:
            stage_path = Path(tmpdir) / "normals.usda"
            stage = Usd.Stage.CreateNew(str(stage_path))
            UsdGeom.Xform.Define(stage, "/Root")
            mesh = UsdGeom.Mesh.Define(stage, "/Root/Triangle")
            UsdGeom.Xformable(mesh.GetPrim()).AddRotateZOp().Set(90.0)
            mesh.CreatePointsAttr(
                [
                    Gf.Vec3f(0.0, 0.0, 0.0),
                    Gf.Vec3f(1.0, 0.0, 0.0),
                    Gf.Vec3f(0.0, 1.0, 0.0),
                ]
            )
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
            mesh.CreateNormalsAttr([Gf.Vec3f(1.0, 0.0, 0.0)] * 3)
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            stage.GetRootLayer().Save()

            mesh = newton.usd.get_mesh(stage_path, root_path="/Root", load_normals=True, compute_inertia=False)

        self.assertIsNotNone(mesh.normals)
        np.testing.assert_allclose(mesh.normals[0], np.array([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
