# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import newton
import warp as wp

from newton._src.solvers.vbd import particle_vbd_kernels as particle_vbd_kernels_stock
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    evaluate_angular_constraint_force_hessian as evaluate_angular_constraint_force_hessian_stock,
    evaluate_anisotropic_angular_constraint_force_hessian as evaluate_anisotropic_angular_constraint_force_hessian_stock,
)
from newton._src.solvers.vbd import tri_mesh_collision as tri_mesh_collision_stock
from newton._src.solvers.vbd_palatial import particle_vbd_kernels as particle_vbd_kernels_palatial
from newton._src.solvers.vbd_palatial.rigid_vbd_kernels import (
    evaluate_angular_constraint_force_hessian as evaluate_angular_constraint_force_hessian_palatial,
    evaluate_anisotropic_angular_constraint_force_hessian as evaluate_anisotropic_angular_constraint_force_hessian_palatial,
)
from newton._src.solvers.vbd_palatial.solver_vbd import SolverVBD as SolverVBDPalatialInternal
from newton._src.solvers.vbd_palatial import tri_mesh_collision as tri_mesh_collision_palatial


class TestSolverVBDPalatial(unittest.TestCase):
    """Tests for the SolverVBDPalatial compatibility surface."""

    @staticmethod
    def _build_anisotropic_rod_model() -> newton.Model:
        builder = newton.ModelBuilder()
        positions, quaternions = newton.utils.create_straight_cable_points_and_quaternions(
            start=wp.vec3(0.0, 0.0, 0.0),
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=1.5,
            num_segments=3,
        )
        builder.add_rod(
            positions=positions,
            quaternions=quaternions,
            stretch_stiffness=123.0,
            stretch_damping=0.1,
            bend_y_stiffness=12.0,
            bend_y_damping=0.2,
            bend_z_stiffness=20.0,
            bend_z_damping=0.4,
            torsion_stiffness=9.0,
            torsion_damping=0.6,
        )
        builder.color()
        return builder.finalize(device="cpu")

    def test_public_alias_reuses_stock_solver_class(self):
        self.assertIs(newton.solvers.SolverVBDPalatial, newton.solvers.SolverVBD)
        self.assertIs(SolverVBDPalatialInternal, newton.solvers.SolverVBD)

    def test_internal_kernel_module_path_reuses_stock_symbols(self):
        self.assertIs(
            evaluate_angular_constraint_force_hessian_palatial,
            evaluate_angular_constraint_force_hessian_stock,
        )
        self.assertIs(
            evaluate_anisotropic_angular_constraint_force_hessian_palatial,
            evaluate_anisotropic_angular_constraint_force_hessian_stock,
        )
        self.assertEqual(
            particle_vbd_kernels_palatial.NUM_THREADS_PER_COLLISION_PRIMITIVE,
            particle_vbd_kernels_stock.NUM_THREADS_PER_COLLISION_PRIMITIVE,
        )
        self.assertIs(
            tri_mesh_collision_palatial.TriMeshCollisionInfo,
            tri_mesh_collision_stock.TriMeshCollisionInfo,
        )

    def test_public_alias_instantiates_on_anisotropic_rod_model(self):
        model = self._build_anisotropic_rod_model()

        solver = newton.solvers.SolverVBDPalatial(model, iterations=2)

        self.assertIsInstance(solver, newton.solvers.SolverVBD)
        self.assertEqual(solver.joint_constraint_dim.numpy().tolist(), [4, 4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
