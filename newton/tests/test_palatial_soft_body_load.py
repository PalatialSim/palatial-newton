# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
import unittest

from newton.palatial import load
from newton.solvers import SolverVBD
from newton.solvers.experimental.coupled import SolverCoupledProxy

_ASSET = os.path.join(os.path.dirname(__file__), "assets", "deformables_mixed.usda")


class TestPalatialSoftBodyLoad(unittest.TestCase):
    """Test automatic loading of an embedded Palatial mixed-body contract."""

    def setUp(self):
        """Author stable part tags and a coupled plan onto the native mixed fixture."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        self.temp_dir = tempfile.TemporaryDirectory()
        self.usd_path = os.path.join(self.temp_dir.name, "palatial_mixed.usda")
        stage = Usd.Stage.Open(_ASSET)
        rigid = UsdGeom.Cube.Define(stage, "/World/Rigid")
        rigid.CreateSizeAttr(0.2)
        UsdPhysics.RigidBodyAPI.Apply(rigid.GetPrim())
        UsdPhysics.CollisionAPI.Apply(rigid.GetPrim())

        parts = [
            ("rigid", "/World/Rigid", "rigid", "rigid"),
            ("cable_a", "/World/CableA", "soft", "rod"),
            ("cable_b", "/World/CableB", "soft", "rod"),
            ("cloth", "/World/Cloth", "soft", "surface"),
            ("soft_a", "/World/SoftA", "soft", "volume"),
            ("soft_b", "/World/SoftB", "soft", "volume"),
        ]
        part_specs = []
        for index, (part_id, path, role, representation) in enumerate(parts):
            prim = stage.GetPrimAtPath(path)
            prim.CreateAttribute("palatial:partId", Sdf.ValueTypeNames.String, custom=True).Set(part_id)
            prim.CreateAttribute("palatial:physicsRole", Sdf.ValueTypeNames.Token, custom=True).Set(role)
            prim.CreateAttribute("palatial:representation", Sdf.ValueTypeNames.Token, custom=True).Set(representation)
            part_specs.append(
                {
                    "part_id": part_id,
                    "source_part_index": index,
                    "physics_role": role,
                    "representation": representation,
                    "material": {},
                    "properties": {},
                }
            )

        soft_ids = [part_id for part_id, _path, role, _representation in parts if role == "soft"]
        spec = {
            "schema_version": "1.0",
            "body_type": "soft_bodies",
            "parts": part_specs,
            "attachments": [],
            "solver_plan": {
                "mode": "coupled",
                "assignments": [
                    {"id": "rigid_solver", "solver": "semi_implicit", "part_ids": ["rigid"]},
                    {
                        "id": "soft_solver",
                        "solver": "vbd",
                        "part_ids": soft_ids,
                        "parameters": {"vbd_iterations": 4},
                    },
                ],
                "couplings": [
                    {
                        "from": "rigid_solver",
                        "to": "soft_solver",
                        "method": "proxy",
                        "parameters": {"proxy_iterations": 1},
                    }
                ],
            },
            "inference": {"source": "physics_vlm"},
        }
        scene = stage.GetPrimAtPath("/PhysicsScene")
        scene.CreateAttribute("palatial:softBodySpec", Sdf.ValueTypeNames.String, custom=True).Set(
            json.dumps(spec, sort_keys=True, separators=(",", ":"))
        )
        stage.Export(self.usd_path)

    def tearDown(self):
        """Remove the temporary authored stage."""
        self.temp_dir.cleanup()

    def test_load_builds_coupled_solver_and_realized_part_ownership(self):
        """Consume the embedded plan and importer ranges without name inference."""
        bundle = load(self.usd_path, device="cpu")

        self.assertEqual(bundle.body_type, "mixed")
        self.assertIsInstance(bundle.solver, SolverCoupledProxy)
        self.assertEqual(bundle.solver.entry_names(), ("rigid_solver", "soft_solver"))
        self.assertEqual(set(bundle.part_entities), {"rigid", "cable_a", "cable_b", "cloth", "soft_a", "soft_b"})
        self.assertEqual(len(bundle.part_entities["rigid"].bodies), 1)
        self.assertEqual(len(bundle.part_entities["cable_a"].bodies), 3)
        self.assertEqual(len(bundle.part_entities["cable_b"].bodies), 3)
        self.assertEqual(len(bundle.part_entities["cloth"].particles), 4)
        self.assertEqual(len(bundle.part_entities["soft_a"].particles), 4)
        self.assertEqual(len(bundle.part_entities["soft_b"].particles), 4)

    def test_explicit_solver_and_parameter_overrides_replace_authored_plan(self):
        """Let callers deliberately force one solver and its settings at runtime."""
        bundle = load(
            self.usd_path,
            device="cpu",
            solver_override="vbd",
            solver_param_overrides={"vbd_iterations": 7},
        )

        self.assertIsInstance(bundle.solver, SolverVBD)
        self.assertEqual(bundle.solver.iterations, 7)
        self.assertEqual(bundle.solver_plan["mode"], "single")
        self.assertEqual(bundle.solver_plan["assignments"][0]["part_ids"], list(bundle.part_entities))


if __name__ == "__main__":
    unittest.main()
