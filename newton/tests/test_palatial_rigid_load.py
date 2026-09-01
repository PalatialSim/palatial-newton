# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pxr import Gf, Usd, UsdGeom, UsdPhysics

import newton.palatial


class TestPalatialRigidLoad(unittest.TestCase):
    def _write_asset(self, directory: str, joint_schema):
        usd_path = Path(directory) / "parts.usda"
        stage = Usd.Stage.CreateNew(str(usd_path))
        UsdPhysics.Scene.Define(stage, "/World/physicsScene")
        asset = UsdGeom.Xform.Define(stage, "/World/asset")
        UsdPhysics.ArticulationRootAPI.Apply(asset.GetPrim())

        for name, offset in (("body", 0.0), ("lid", 0.45)):
            cube = UsdGeom.Cube.Define(stage, f"/World/asset/{name}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, offset))
            UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            mass = UsdPhysics.MassAPI.Apply(cube.GetPrim())
            mass.CreateMassAttr(1.0)

        joint = joint_schema.Define(stage, "/World/asset/body_to_lid")
        joint.CreateBody0Rel().SetTargets(["/World/asset/body"])
        joint.CreateBody1Rel().SetTargets(["/World/asset/lid"])
        stage.GetRootLayer().Save()
        return usd_path

    def _load_with_import_options(self, usd_path: Path):
        original_add_usd = newton.ModelBuilder.add_usd
        options = {}

        def capture_options(builder, *args, **kwargs):
            options.update(kwargs)
            return original_add_usd(builder, *args, **kwargs)

        with (
            mock.patch.object(newton.ModelBuilder, "add_usd", new=capture_options),
            mock.patch("newton._src.palatial.load._build_solver", return_value=object()),
        ):
            bundle = newton.palatial.load(str(usd_path), solver_override="xpbd", device="cpu")
        return bundle, options

    def test_fixed_parts_load_as_one_compound_rigid_body(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = self._write_asset(tmpdir, UsdPhysics.FixedJoint)
            bundle, options = self._load_with_import_options(usd_path)

            self.assertEqual(bundle.model.body_count, 1)
            self.assertFalse(options["enable_self_collisions"])

    def test_movable_parts_keep_articulation_self_collision_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = self._write_asset(tmpdir, UsdPhysics.RevoluteJoint)
            bundle, options = self._load_with_import_options(usd_path)

            self.assertEqual(bundle.model.body_count, 2)
            self.assertTrue(options["enable_self_collisions"])


if __name__ == "__main__":
    unittest.main()
