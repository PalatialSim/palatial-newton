# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pxr import Gf, Usd, UsdGeom, UsdPhysics

import newton.palatial

from newton._src.palatial.load import _synchronize_newton_contact_capacity


class TestPalatialLoad(unittest.TestCase):
    def _write_rigid_parts(self, directory: str, joint_schema):
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
            UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(1.0)

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
            bundle, options = self._load_with_import_options(
                self._write_rigid_parts(tmpdir, UsdPhysics.FixedJoint)
            )

            self.assertEqual(bundle.model.body_count, 1)
            self.assertFalse(options["enable_self_collisions"])

    def test_movable_parts_keep_articulation_self_collision_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle, options = self._load_with_import_options(
                self._write_rigid_parts(tmpdir, UsdPhysics.RevoluteJoint)
            )

            self.assertEqual(bundle.model.body_count, 2)
            self.assertTrue(options["enable_self_collisions"])

    def test_synchronizes_newton_and_mujoco_contact_capacity(self):
        model = SimpleNamespace(rigid_contact_max=1160)
        solver_params = {
            "nconmax": 4096,
            "use_mujoco_contacts": False,
        }

        _synchronize_newton_contact_capacity(model, "mujoco", solver_params)

        self.assertEqual(model.rigid_contact_max, 4096)

    def test_preserves_capacity_for_native_mujoco_contacts(self):
        model = SimpleNamespace(rigid_contact_max=1160)
        solver_params = {
            "nconmax": 4096,
            "use_mujoco_contacts": True,
        }

        _synchronize_newton_contact_capacity(model, "mujoco", solver_params)

        self.assertEqual(model.rigid_contact_max, 1160)


if __name__ == "__main__":
    unittest.main()
