# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp

import newton
from newton.palatial import SolverPlanPartEntities, solver_from_plan
from newton.solvers import SolverVBD, SolverXPBD
from newton.solvers.experimental.coupled import SolverCoupledADMM, SolverCoupledProxy


class TestPalatialSolverPlan(unittest.TestCase):
    """Test Palatial solver-plan construction."""

    def setUp(self):
        """Build a two-part model and stable part-to-entity mapping."""
        builder = newton.ModelBuilder()
        rigid_body = builder.add_body(xform=wp.transform(p=wp.vec3(-0.5, 0.0, 1.0)))
        rigid_joint = builder.joint_count - 1
        rigid_shape = builder.add_shape_box(rigid_body, hx=0.1, hy=0.1, hz=0.1)
        soft_body = builder.add_body(xform=wp.transform(p=wp.vec3(0.5, 0.0, 1.0)))
        soft_joint = builder.joint_count - 1
        soft_shape = builder.add_shape_box(soft_body, hx=0.1, hy=0.1, hz=0.1)
        particle_start = builder.particle_count
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 2.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=1,
            dim_y=1,
            dim_z=1,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=1000.0,
            k_mu=1.0e4,
            k_lambda=1.0e4,
            k_damp=100.0,
        )
        builder.color()
        self.model = builder.finalize(device="cpu")
        self.part_entities = {
            "rigid": SolverPlanPartEntities(bodies=[rigid_body], joints=[rigid_joint], shapes=[rigid_shape]),
            "soft": SolverPlanPartEntities(
                bodies=[soft_body],
                particles=range(particle_start, builder.particle_count),
                joints=[soft_joint],
                shapes=[soft_shape],
            ),
        }

    def test_build_single_solver_with_prefixed_parameters(self):
        """Build one solver and translate its prefixed iteration setting."""
        plan = {
            "mode": "single",
            "assignments": [
                {
                    "id": "xpbd_0",
                    "solver": "xpbd",
                    "part_ids": ["rigid", "soft"],
                    "parameters": {"xpbd_iterations": 7},
                }
            ],
            "couplings": [],
        }

        solver = solver_from_plan(self.model, plan, self.part_entities)

        self.assertIsInstance(solver, SolverXPBD)
        self.assertEqual(solver.iterations, 7)

    def test_translate_vbd_schema_parameter_names(self):
        """Translate VBD schema names into the native constructor contract."""
        plan = {
            "mode": "single",
            "assignments": [
                {
                    "id": "vbd_0",
                    "solver": "vbd",
                    "part_ids": ["rigid", "soft"],
                    "parameters": {
                        "vbd_iterations": 9,
                        "vbd_particle_self_contact_enabled": True,
                        "vbd_particle_self_contact_radius": 0.02,
                    },
                }
            ],
            "couplings": [],
        }

        solver = solver_from_plan(self.model, plan, self.part_entities)

        self.assertIsInstance(solver, SolverVBD)
        self.assertEqual(solver.iterations, 9)
        self.assertTrue(solver.particle_enable_self_contact)
        self.assertEqual(solver.particle_self_contact_radius, 0.02)

    def test_build_proxy_coupled_solver(self):
        """Build directed proxy coupling with explicit mixed-body ownership."""
        plan = self._coupled_plan("proxy")
        plan["couplings"][0]["parameters"] = {"proxy_iterations": 2, "mass_scale": 0.75}

        solver = solver_from_plan(self.model, {"solver_plan": plan}, self.part_entities)

        self.assertIsInstance(solver, SolverCoupledProxy)
        self.assertEqual(solver.entry_names(), ("rigid_solver", "soft_solver"))

    def test_build_admm_coupled_solver(self):
        """Build ADMM coupling for model-derived attachments and joints."""
        plan = self._coupled_plan("admm")
        plan["couplings"][0]["parameters"] = {"admm_iterations": 3, "rho": 20.0}

        solver = solver_from_plan(
            self.model,
            {"soft_body_spec": {"solver_plan": plan}},
            self.part_entities,
        )

        self.assertIsInstance(solver, SolverCoupledADMM)
        self.assertEqual(solver.entry_names(), ("rigid_solver", "soft_solver"))

    def test_reject_mixed_coupling_methods(self):
        """Reject a plan that needs two incompatible coupling wrappers."""
        plan = self._coupled_plan("proxy")
        plan["couplings"].append({"from": "soft_solver", "to": "rigid_solver", "method": "admm"})

        with self.assertRaisesRegex(ValueError, "cannot mix proxy and ADMM"):
            solver_from_plan(self.model, plan, self.part_entities)

    def test_reject_incomplete_part_mapping(self):
        """Reject runtime ownership that omits a planned part."""
        with self.assertRaisesRegex(ValueError, "same part ids"):
            solver_from_plan(
                self.model,
                self._coupled_plan("proxy"),
                {"rigid": self.part_entities["rigid"]},
            )

    @staticmethod
    def _coupled_plan(method):
        return {
            "mode": "coupled",
            "assignments": [
                {
                    "id": "rigid_solver",
                    "solver": "semi_implicit",
                    "part_ids": ["rigid"],
                },
                {
                    "id": "soft_solver",
                    "solver": "xpbd",
                    "part_ids": ["soft"],
                    "parameters": {"xpbd_iterations": 4},
                },
            ],
            "couplings": [{"from": "rigid_solver", "to": "soft_solver", "method": method}],
        }


if __name__ == "__main__":
    unittest.main()
