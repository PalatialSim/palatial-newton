# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton


def _numpy_to_transform(arr: np.ndarray) -> wp.transform:
    return wp.transform(
        wp.vec3(float(arr[0]), float(arr[1]), float(arr[2])),
        wp.quat(float(arr[3]), float(arr[4]), float(arr[5]), float(arr[6])),
    )


def _joint_world_frames(model: newton.Model, body_q: wp.array, joint_id: int) -> tuple[wp.transform, wp.transform]:
    joint_parent = model.joint_parent.numpy().tolist()
    joint_child = model.joint_child.numpy().tolist()
    joint_X_p = model.joint_X_p.numpy()
    joint_X_c = model.joint_X_c.numpy()
    body_q_np = body_q.numpy()

    parent = int(joint_parent[joint_id])
    child = int(joint_child[joint_id])
    X_pj = _numpy_to_transform(joint_X_p[joint_id])
    X_cj = _numpy_to_transform(joint_X_c[joint_id])
    if parent >= 0:
        X_wp = _numpy_to_transform(body_q_np[parent]) * X_pj
    else:
        X_wp = X_pj
    X_wc = _numpy_to_transform(body_q_np[child]) * X_cj
    return X_wp, X_wc


def _joint_kappa(model: newton.Model, body_q: wp.array, joint_id: int) -> np.ndarray:
    body_q_rest = model.body_q.numpy()
    X_wp, X_wc = _joint_world_frames(model, body_q, joint_id)

    parent = int(model.joint_parent.numpy()[joint_id])
    child = int(model.joint_child.numpy()[joint_id])
    X_pj = _numpy_to_transform(model.joint_X_p.numpy()[joint_id])
    X_cj = _numpy_to_transform(model.joint_X_c.numpy()[joint_id])
    if parent >= 0:
        X_wp_rest = _numpy_to_transform(body_q_rest[parent]) * X_pj
    else:
        X_wp_rest = X_pj
    X_wc_rest = _numpy_to_transform(body_q_rest[child]) * X_cj

    q_wp = wp.transform_get_rotation(X_wp)
    q_wc = wp.transform_get_rotation(X_wc)
    q_wp_rest = wp.transform_get_rotation(X_wp_rest)
    q_wc_rest = wp.transform_get_rotation(X_wc_rest)
    q_rel = wp.normalize(wp.mul(wp.quat_inverse(q_wp), q_wc))
    q_rel_rest = wp.normalize(wp.mul(wp.quat_inverse(q_wp_rest), q_wc_rest))
    q_align = wp.normalize(wp.mul(q_rel, wp.quat_inverse(q_rel_rest)))
    if q_align[3] < 0.0:
        q_align = wp.quat(-q_align[0], -q_align[1], -q_align[2], -q_align[3])
    axis, angle = wp.quat_to_axis_angle(q_align)
    kappa = axis * angle
    return np.array([float(kappa[0]), float(kappa[1]), float(kappa[2])], dtype=np.float64)


class TestCableAnisotropicAddRod(unittest.TestCase):
    """Tests for anisotropic cable support on ModelBuilder.add_rod()."""

    @staticmethod
    def _straight_points_and_quaternions() -> tuple[list[wp.vec3], list[wp.quat]]:
        return newton.utils.create_straight_cable_points_and_quaternions(
            start=wp.vec3(0.0, 0.0, 0.0),
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=1.5,
            num_segments=3,
        )

    def test_add_rod_uses_anisotropic_joint_chain_when_per_axis_args_are_authored(self):
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

    def test_add_rod_uses_isotropic_bend_as_fallback_for_unspecified_angular_channels(self):
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
            bend_z_stiffness=None,
            bend_z_damping=None,
            torsion_stiffness=9.0,
            torsion_damping=0.25,
        )

        self.assertEqual(
            builder.joint_target_ke[-8:],
            [50.0, 12.0, 7.5, 9.0, 50.0, 12.0, 7.5, 9.0],
        )
        self.assertEqual(
            builder.joint_target_kd[-8:],
            [0.1, 0.2, 0.3, 0.25, 0.1, 0.2, 0.3, 0.25],
        )

    def test_add_rod_uses_isotropic_cable_when_only_isotropic_args_are_authored(self):
        builder = newton.ModelBuilder()
        positions, quaternions = self._straight_points_and_quaternions()

        _rod_bodies, rod_joints = builder.add_rod(
            positions=positions,
            quaternions=quaternions,
            stretch_stiffness=50.0,
            stretch_damping=0.1,
            bend_stiffness=7.5,
            bend_damping=0.3,
        )

        self.assertEqual(
            [builder.joint_type[joint_index] for joint_index in rod_joints],
            [newton.JointType.CABLE, newton.JointType.CABLE],
        )
        self.assertEqual(builder.joint_target_ke[-4:], [50.0, 7.5, 50.0, 7.5])
        self.assertEqual(builder.joint_target_kd[-4:], [0.1, 0.3, 0.1, 0.3])

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
        builder.add_rod(
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

    def test_solver_vbd_respects_per_axis_bend_stiffness(self):
        def run_case(
            *,
            rotation_axis: wp.vec3,
            bend_y_stiffness: float,
            bend_z_stiffness: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            builder = newton.ModelBuilder(gravity=0.0)
            positions = [
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(1.0, 0.0, 0.0),
                wp.vec3(2.0, 0.0, 0.0),
            ]
            _points, quaternions = self._straight_points_and_quaternions()
            _rod_bodies, rod_joints = builder.add_rod(
                positions=positions,
                quaternions=quaternions[:2],
                stretch_stiffness=1.0e5,
                stretch_damping=0.0,
                bend_y_stiffness=bend_y_stiffness,
                bend_y_damping=0.0,
                bend_z_stiffness=bend_z_stiffness,
                bend_z_damping=0.0,
                torsion_stiffness=0.0,
                torsion_damping=0.0,
                wrap_in_articulation=True,
            )
            builder.color()

            model = builder.finalize(device="cpu")
            solver = newton.solvers.SolverVBD(model, iterations=1)
            state0 = model.state()
            state1 = model.state()
            control = model.control()
            contacts = model.contacts()

            body_q_np = state0.body_q.numpy()
            body_qd_np = state0.body_qd.numpy()
            child_body = int(model.joint_child.numpy()[rod_joints[0]])
            child_pos = body_q_np[child_body, :3].copy()
            child_rot_initial = wp.quat(
                float(body_q_np[child_body, 3]),
                float(body_q_np[child_body, 4]),
                float(body_q_np[child_body, 5]),
                float(body_q_np[child_body, 6]),
            )
            child_rot_offset = wp.quat_from_axis_angle(rotation_axis, 0.1)
            child_rot = wp.normalize(wp.mul(child_rot_offset, child_rot_initial))
            body_q_np[child_body] = np.array(
                [child_pos[0], child_pos[1], child_pos[2], child_rot[0], child_rot[1], child_rot[2], child_rot[3]],
                dtype=body_q_np.dtype,
            )
            state0.body_q = wp.array(body_q_np, dtype=wp.transform, device="cpu")
            state0.body_qd = wp.array(np.zeros_like(body_qd_np), dtype=wp.spatial_vector, device="cpu")

            kappa_initial = _joint_kappa(model, state0.body_q, rod_joints[0])
            model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, 1.0 / 120.0)
            kappa_final = _joint_kappa(model, state1.body_q, rod_joints[0])
            return kappa_initial, kappa_final

        # For a rod aligned along world +X, rotating the child around world +Z excites
        # the first transverse curvature channel and rotating around world +Y excites
        # the second one. Use a single zero-gravity step so the comparison stays local
        # to the authored anisotropic stiffness slots instead of long-horizon dynamics.
        initial_bend_y, final_bend_y = run_case(
            rotation_axis=wp.vec3(0.0, 0.0, 1.0),
            bend_y_stiffness=200.0,
            bend_z_stiffness=5.0,
        )
        initial_bend_y_ref, final_bend_y_ref = run_case(
            rotation_axis=wp.vec3(0.0, 0.0, 1.0),
            bend_y_stiffness=5.0,
            bend_z_stiffness=200.0,
        )
        np.testing.assert_allclose(initial_bend_y, initial_bend_y_ref, rtol=0.0, atol=1.0e-6)
        self.assertGreater(abs(initial_bend_y[0]), 1.0e-3)
        self.assertLess(abs(initial_bend_y[1]), 1.0e-6)
        self.assertLess(abs(initial_bend_y[2]), 1.0e-6)
        self.assertLess(abs(final_bend_y[0]), abs(final_bend_y_ref[0]))

        initial_bend_z, final_bend_z = run_case(
            rotation_axis=wp.vec3(0.0, 1.0, 0.0),
            bend_y_stiffness=200.0,
            bend_z_stiffness=5.0,
        )
        initial_bend_z_ref, final_bend_z_ref = run_case(
            rotation_axis=wp.vec3(0.0, 1.0, 0.0),
            bend_y_stiffness=5.0,
            bend_z_stiffness=200.0,
        )
        np.testing.assert_allclose(initial_bend_z, initial_bend_z_ref, rtol=0.0, atol=1.0e-6)
        self.assertGreater(abs(initial_bend_z[1]), 1.0e-3)
        self.assertLess(abs(initial_bend_z[0]), 1.0e-6)
        self.assertLess(abs(initial_bend_z[2]), 1.0e-6)
        self.assertLess(abs(final_bend_z_ref[1]), abs(final_bend_z[1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
