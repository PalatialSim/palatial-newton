# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    evaluate_angular_constraint_force_hessian,
    evaluate_anisotropic_angular_constraint_force_hessian,
)


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


@wp.kernel
def _eval_stock_cable_angular_kernel(
    q_wp: wp.array[wp.quat],
    q_wc: wp.array[wp.quat],
    q_wp_rest: wp.array[wp.quat],
    q_wc_rest: wp.array[wp.quat],
    q_wp_prev: wp.array[wp.quat],
    q_wc_prev: wp.array[wp.quat],
    k_eff: float,
    damping: float,
    torque_out: wp.array[wp.vec3],
    hessian_out: wp.array[wp.mat33],
):
    tau, H_aa, _kappa, _J = evaluate_angular_constraint_force_hessian(
        q_wp[0],
        q_wc[0],
        q_wp_rest[0],
        q_wc_rest[0],
        q_wp_prev[0],
        q_wc_prev[0],
        True,
        k_eff,
        wp.identity(3, float),
        wp.vec3(0.0),
        wp.vec3(0.0),
        wp.vec3(0.0),
        wp.vec3(0.0),
        0.0,
        damping,
        1.0,
    )
    torque_out[0] = tau
    hessian_out[0] = H_aa


@wp.kernel
def _eval_anisotropic_cable_angular_kernel(
    q_wp: wp.array[wp.quat],
    q_wc: wp.array[wp.quat],
    q_wp_rest: wp.array[wp.quat],
    q_wc_rest: wp.array[wp.quat],
    q_wp_prev: wp.array[wp.quat],
    q_wc_prev: wp.array[wp.quat],
    k_eff: wp.vec3,
    damping: wp.vec3,
    torque_out: wp.array[wp.vec3],
    hessian_out: wp.array[wp.mat33],
):
    tau, H_aa, _kappa, _J = evaluate_anisotropic_angular_constraint_force_hessian(
        q_wp[0],
        q_wc[0],
        q_wp_rest[0],
        q_wc_rest[0],
        q_wp_prev[0],
        q_wc_prev[0],
        True,
        k_eff,
        wp.vec3(0.0),
        wp.vec3(0.0),
        0.0,
        damping,
        1.0,
    )
    torque_out[0] = tau
    hessian_out[0] = H_aa


class TestCableAnisotropicAddRod(unittest.TestCase):
    """Tests for anisotropic cable support on ModelBuilder.add_rod()."""

    _CHANNEL_ISOLATION_TOL = 1.0e-4
    _ISOTROPIC_EQUIVALENCE_TOL = 1.0e-6

    @staticmethod
    def _straight_points_and_quaternions() -> tuple[list[wp.vec3], list[wp.quat]]:
        return newton.utils.create_straight_cable_points_and_quaternions(
            start=wp.vec3(0.0, 0.0, 0.0),
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=1.5,
            num_segments=3,
        )

    @staticmethod
    def _identity_quat() -> wp.quat:
        return wp.quat(0.0, 0.0, 0.0, 1.0)

    @staticmethod
    def _axis_angle_quat(axis: tuple[float, float, float], angle: float) -> wp.quat:
        axis_length = math.sqrt(sum(component * component for component in axis))
        axis_vec = tuple(component / axis_length for component in axis)
        half_angle = 0.5 * angle
        sin_half = math.sin(half_angle)
        return wp.quat(
            axis_vec[0] * sin_half,
            axis_vec[1] * sin_half,
            axis_vec[2] * sin_half,
            math.cos(half_angle),
        )

    def _evaluate_stock(
        self,
        *,
        child_rotation: wp.quat,
        previous_child_rotation: wp.quat,
        k_eff: float,
        damping: float,
    ) -> tuple[tuple[float, float, float], tuple[tuple[float, ...], ...]]:
        q_wp = wp.array([self._identity_quat()], dtype=wp.quat, device="cpu")
        q_wc = wp.array([child_rotation], dtype=wp.quat, device="cpu")
        q_wp_rest = wp.array([self._identity_quat()], dtype=wp.quat, device="cpu")
        q_wc_rest = wp.array([self._identity_quat()], dtype=wp.quat, device="cpu")
        q_wp_prev = wp.array([self._identity_quat()], dtype=wp.quat, device="cpu")
        q_wc_prev = wp.array([previous_child_rotation], dtype=wp.quat, device="cpu")
        torque_out = wp.zeros(1, dtype=wp.vec3, device="cpu")
        hessian_out = wp.zeros(1, dtype=wp.mat33, device="cpu")

        wp.launch(
            _eval_stock_cable_angular_kernel,
            dim=1,
            inputs=[
                q_wp,
                q_wc,
                q_wp_rest,
                q_wc_rest,
                q_wp_prev,
                q_wc_prev,
                k_eff,
                damping,
                torque_out,
                hessian_out,
            ],
            device="cpu",
        )

        torque_np = torque_out.numpy()[0]
        hessian_np = hessian_out.numpy()[0]
        return (
            (float(torque_np[0]), float(torque_np[1]), float(torque_np[2])),
            tuple(
                tuple(float(hessian_np[row, col]) for col in range(3))
                for row in range(3)
            ),
        )

    def _evaluate_anisotropic(
        self,
        *,
        child_rotation: wp.quat,
        previous_child_rotation: wp.quat,
        stiffness: tuple[float, float, float],
        damping: tuple[float, float, float],
    ) -> tuple[tuple[float, float, float], tuple[tuple[float, ...], ...]]:
        q_wp = wp.array([self._identity_quat()], dtype=wp.quat, device="cpu")
        q_wc = wp.array([child_rotation], dtype=wp.quat, device="cpu")
        q_wp_rest = wp.array([self._identity_quat()], dtype=wp.quat, device="cpu")
        q_wc_rest = wp.array([self._identity_quat()], dtype=wp.quat, device="cpu")
        q_wp_prev = wp.array([self._identity_quat()], dtype=wp.quat, device="cpu")
        q_wc_prev = wp.array([previous_child_rotation], dtype=wp.quat, device="cpu")
        torque_out = wp.zeros(1, dtype=wp.vec3, device="cpu")
        hessian_out = wp.zeros(1, dtype=wp.mat33, device="cpu")

        wp.launch(
            _eval_anisotropic_cable_angular_kernel,
            dim=1,
            inputs=[
                q_wp,
                q_wc,
                q_wp_rest,
                q_wc_rest,
                q_wp_prev,
                q_wc_prev,
                wp.vec3(*stiffness),
                wp.vec3(*damping),
                torque_out,
                hessian_out,
            ],
            device="cpu",
        )

        torque_np = torque_out.numpy()[0]
        hessian_np = hessian_out.numpy()[0]
        return (
            (float(torque_np[0]), float(torque_np[1]), float(torque_np[2])),
            tuple(
                tuple(float(hessian_np[row, col]) for col in range(3))
                for row in range(3)
            ),
        )

    def _assert_channel_isolated(
        self,
        torque: tuple[float, float, float],
        *,
        active_axis: int,
        expected_scale: float,
    ) -> None:
        self.assertGreater(abs(torque[active_axis]), 0.0)
        leakage_limit = abs(expected_scale) * self._CHANNEL_ISOLATION_TOL
        for axis, component in enumerate(torque):
            if axis == active_axis:
                continue
            self.assertLessEqual(abs(component), leakage_limit)

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

    def test_anisotropic_kernel_pure_bend_y_excites_only_bend_y_channel(self):
        angle = 0.125
        torque, _hessian = self._evaluate_anisotropic(
            child_rotation=self._axis_angle_quat((1.0, 0.0, 0.0), angle),
            previous_child_rotation=self._identity_quat(),
            stiffness=(5.0, 2.0, 1.0),
            damping=(0.0, 0.0, 0.0),
        )
        self._assert_channel_isolated(torque, active_axis=0, expected_scale=5.0 * angle)

    def test_anisotropic_kernel_pure_bend_z_excites_only_bend_z_channel(self):
        angle = 0.125
        torque, _hessian = self._evaluate_anisotropic(
            child_rotation=self._axis_angle_quat((0.0, 1.0, 0.0), angle),
            previous_child_rotation=self._identity_quat(),
            stiffness=(5.0, 2.0, 1.0),
            damping=(0.0, 0.0, 0.0),
        )
        self._assert_channel_isolated(torque, active_axis=1, expected_scale=2.0 * angle)

    def test_anisotropic_kernel_pure_torsion_excites_only_torsion_channel(self):
        angle = 0.125
        torque, _hessian = self._evaluate_anisotropic(
            child_rotation=self._axis_angle_quat((0.0, 0.0, 1.0), angle),
            previous_child_rotation=self._identity_quat(),
            stiffness=(5.0, 2.0, 1.0),
            damping=(0.0, 0.0, 0.0),
        )
        self._assert_channel_isolated(torque, active_axis=2, expected_scale=1.0 * angle)

    def test_anisotropic_kernel_equal_channels_match_stock_isotropic(self):
        child_rotation = self._axis_angle_quat((1.0, 1.0, 1.0), 0.09)
        previous_child_rotation = self._axis_angle_quat((1.0, 1.0, 1.0), 0.03)

        stock_torque, stock_hessian = self._evaluate_stock(
            child_rotation=child_rotation,
            previous_child_rotation=previous_child_rotation,
            k_eff=3.0,
            damping=0.2,
        )
        anisotropic_torque, anisotropic_hessian = self._evaluate_anisotropic(
            child_rotation=child_rotation,
            previous_child_rotation=previous_child_rotation,
            stiffness=(3.0, 3.0, 3.0),
            damping=(0.2, 0.2, 0.2),
        )

        for stock_component, anisotropic_component in zip(stock_torque, anisotropic_torque):
            self.assertAlmostEqual(stock_component, anisotropic_component, delta=self._ISOTROPIC_EQUIVALENCE_TOL)

        for stock_row, anisotropic_row in zip(stock_hessian, anisotropic_hessian):
            for stock_value, anisotropic_value in zip(stock_row, anisotropic_row):
                self.assertAlmostEqual(stock_value, anisotropic_value, delta=self._ISOTROPIC_EQUIVALENCE_TOL)

    def test_anisotropic_kernel_damping_isolated_per_axis(self):
        current_rotation = self._axis_angle_quat((0.0, 0.0, 1.0), 0.15)
        previous_rotation = self._axis_angle_quat((0.0, 0.0, 1.0), 0.05)
        torque_no_damping, hessian_no_damping = self._evaluate_anisotropic(
            child_rotation=current_rotation,
            previous_child_rotation=previous_rotation,
            stiffness=(4.0, 6.0, 8.0),
            damping=(0.0, 0.0, 0.0),
        )
        torque_with_damping, hessian_with_damping = self._evaluate_anisotropic(
            child_rotation=current_rotation,
            previous_child_rotation=previous_rotation,
            stiffness=(4.0, 6.0, 8.0),
            damping=(0.0, 0.0, 0.5),
        )

        self.assertAlmostEqual(torque_no_damping[0], torque_with_damping[0], delta=1.0e-8)
        self.assertAlmostEqual(torque_no_damping[1], torque_with_damping[1], delta=1.0e-8)
        self.assertGreater(abs(torque_with_damping[2] - torque_no_damping[2]), 1.0e-6)
        self.assertAlmostEqual(hessian_no_damping[0][0], hessian_with_damping[0][0], delta=1.0e-8)
        self.assertAlmostEqual(hessian_no_damping[1][1], hessian_with_damping[1][1], delta=1.0e-8)
        self.assertGreater(abs(hessian_with_damping[2][2] - hessian_no_damping[2][2]), 1.0e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
