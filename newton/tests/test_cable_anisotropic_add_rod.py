# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
