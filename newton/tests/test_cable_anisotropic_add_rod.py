# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton


class TestCableAnisotropicAddRod(unittest.TestCase):
    """Tests for the phase-A anisotropic cable surface on ModelBuilder.add_rod()."""

    @staticmethod
    def _straight_points_and_quaternions() -> tuple[list[wp.vec3], list[wp.quat]]:
        return newton.utils.create_straight_cable_points_and_quaternions(
            start=wp.vec3(0.0, 0.0, 0.0),
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=1.5,
            num_segments=3,
        )

    def test_add_rod_collapses_anisotropic_stiffness_and_damping(self):
        builder = newton.ModelBuilder()
        positions, quaternions = self._straight_points_and_quaternions()

        rod_bodies, rod_joints = builder.add_rod(
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

        self.assertEqual(len(rod_bodies), 3)
        self.assertEqual(len(rod_joints), 2)
        self.assertEqual(builder.joint_target_ke[-4:], [123.0, 16.0, 123.0, 16.0])
        self.assertEqual(builder.joint_target_kd[-4:], [0.6, 0.6, 0.6, 0.6])

    def test_add_rod_uses_explicit_isotropic_bend_stiffness_when_present(self):
        builder = newton.ModelBuilder()
        positions, quaternions = self._straight_points_and_quaternions()

        builder.add_rod(
            positions=positions,
            quaternions=quaternions,
            stretch_stiffness=50.0,
            stretch_damping=0.1,
            bend_stiffness=7.5,
            bend_damping=0.3,
            bend_y_stiffness=12.0,
            bend_y_damping=0.2,
            bend_z_stiffness=20.0,
            bend_z_damping=0.4,
            torsion_stiffness=9.0,
            torsion_damping=0.25,
        )

        self.assertEqual(builder.joint_target_ke[-4:], [50.0, 7.5, 50.0, 7.5])
        self.assertEqual(builder.joint_target_kd[-4:], [0.4, 0.4, 0.4, 0.4])

    def test_add_joint_cable_rejects_anisotropic_kwargs(self):
        builder = newton.ModelBuilder()
        parent = builder.add_link()
        child = builder.add_link()

        with self.assertRaisesRegex(ValueError, "add_joint_anisotropic_cable"):
            builder.add_joint_cable(
                parent=parent,
                child=child,
                bend_y_stiffness=12.0,
            )

    def test_add_joint_anisotropic_cable_uses_frozen_slot_order(self):
        builder = newton.ModelBuilder()
        parent = builder.add_link()
        child = builder.add_link()

        joint_index = builder.add_joint_anisotropic_cable(
            parent=parent,
            child=child,
            stretch_stiffness=123.0,
            stretch_damping=0.1,
            bend_y_stiffness=12.0,
            bend_y_damping=0.2,
            bend_z_stiffness=20.0,
            bend_z_damping=0.4,
            torsion_stiffness=9.0,
            torsion_damping=0.6,
        )

        self.assertEqual(
            builder.joint_type[joint_index],
            newton.JointType.ANISOTROPIC_CABLE,
        )
        self.assertEqual(builder.joint_dof_dim[joint_index], (1, 3))
        self.assertEqual(builder.joint_qd_start[joint_index], 0)
        self.assertEqual(builder.joint_dof_count, 4)
        self.assertEqual(builder.joint_coord_count, 4)
        self.assertEqual(builder.joint_target_ke[:4], [123.0, 12.0, 20.0, 9.0])
        self.assertEqual(builder.joint_target_kd[:4], [0.1, 0.2, 0.4, 0.6])

    def test_add_rod_anisotropic_builds_anisotropic_joint_chain(self):
        builder = newton.ModelBuilder()
        positions, quaternions = self._straight_points_and_quaternions()

        rod_bodies, rod_joints = builder.add_rod_anisotropic(
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
            label="anisotropic",
        )

        self.assertEqual(len(rod_bodies), 3)
        self.assertEqual(len(rod_joints), 2)
        self.assertEqual(
            [builder.joint_type[joint_index] for joint_index in rod_joints],
            [newton.JointType.ANISOTROPIC_CABLE, newton.JointType.ANISOTROPIC_CABLE],
        )
        self.assertEqual(
            builder.joint_target_ke[-8:],
            [123.0, 12.0, 20.0, 9.0, 123.0, 12.0, 20.0, 9.0],
        )
        self.assertEqual(
            builder.joint_target_kd[-8:],
            [0.1, 0.2, 0.4, 0.6, 0.1, 0.2, 0.4, 0.6],
        )

        model = builder.finalize(device="cpu")
        self.assertEqual(
            model.joint_type.numpy().tolist(),
            [int(newton.JointType.ANISOTROPIC_CABLE)] * 2,
        )
        self.assertEqual(model.joint_dof_dim.numpy().tolist(), [[1, 3], [1, 3]])

    def test_solver_vbd_initializes_anisotropic_constraint_slots(self):
        builder = newton.ModelBuilder()
        positions, quaternions = self._straight_points_and_quaternions()

        builder.add_rod_anisotropic(
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

        model = builder.finalize(device="cpu")
        solver = newton.solvers.SolverVBD(model, iterations=2)

        self.assertEqual(solver.joint_constraint_dim.numpy().tolist(), [4, 4])
        self.assertEqual(solver.joint_constraint_start.numpy().tolist(), [0, 4])
        np.testing.assert_allclose(
            solver.joint_penalty_k_max.numpy(),
            np.array([123.0, 12.0, 20.0, 9.0, 123.0, 12.0, 20.0, 9.0], dtype=np.float32),
            rtol=0.0,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            solver.joint_penalty_kd.numpy(),
            np.array([0.1, 0.2, 0.4, 0.6, 0.1, 0.2, 0.4, 0.6], dtype=np.float32),
            rtol=0.0,
            atol=1.0e-6,
        )

    def test_solver_vbd_steps_anisotropic_rod(self):
        builder = newton.ModelBuilder()
        positions, quaternions = self._straight_points_and_quaternions()

        builder.add_ground_plane()
        builder.add_rod_anisotropic(
            positions=positions,
            quaternions=quaternions,
            radius=0.02,
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

        model = builder.finalize(device="cpu")
        solver = newton.solvers.SolverVBD(model, iterations=2)
        state0 = model.state()
        state1 = model.state()
        control = model.control()
        contacts = model.collide(state0)

        solver.step(state0, state1, control, contacts, 1.0 / 120.0)

        self.assertTrue(np.isfinite(state1.body_q.numpy()).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
