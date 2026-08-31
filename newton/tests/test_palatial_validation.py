# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from newton.examples.palatial.example_palatial_load import Example
from newton.examples.palatial.validation import (
    NewtonValidationTracker,
    clearance_translation,
    compute_world_shape_points,
    relocate_ground_plane_transform,
    resolve_persistent_rigid_placement,
    translate_free_roots,
)


class TestPalatialValidation(unittest.TestCase):
    def test_drop_height_is_clearance_above_support_plane(self):
        body_q = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        shape_body = np.array([0])
        shape_transform = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        shape_aabb_lower = np.array([[-0.2, -0.3, -0.455]])
        shape_aabb_upper = np.array([[0.2, 0.3, 0.4]])

        before = compute_world_shape_points(
            body_q,
            shape_body,
            shape_transform,
            shape_aabb_lower,
            shape_aabb_upper,
        )
        delta_z = clearance_translation(float(before[:, 2].min()), 0.2)
        translated = body_q.copy()
        translated[:, 2] += delta_z
        after = compute_world_shape_points(
            translated,
            shape_body,
            shape_transform,
            shape_aabb_lower,
            shape_aabb_upper,
        )

        self.assertAlmostEqual(delta_z, 0.655)
        self.assertAlmostEqual(float(after[:, 2].min()), 0.2)

    def test_articulated_clearance_moves_only_free_roots(self):
        joint_q = np.array([0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 1.0, 0.015, -0.02])
        translated, count = translate_free_roots(
            joint_q,
            joint_type=np.array([4, 1, 2]),
            # Newton stores a trailing sentinel offset after the last joint.
            joint_q_start=np.array([0, 7, 8, 9]),
            delta_z=0.7,
            free_joint_type=4,
        )

        self.assertEqual(count, 1)
        self.assertAlmostEqual(translated[2], 0.6)
        np.testing.assert_array_equal(translated[7:], joint_q[7:])
        np.testing.assert_array_equal(
            translated[[0, 1, 3, 4, 5, 6]],
            joint_q[[0, 1, 3, 4, 5, 6]],
        )

    def test_fixed_root_articulation_relocates_support_without_moving_asset(self):
        translation_z, support_plane_z = resolve_persistent_rigid_placement(
            translation_z=0.35,
            support_plane_z=0.0,
            joint_count=3,
            translated_free_joint_count=0,
        )

        self.assertEqual(translation_z, 0.0)
        self.assertAlmostEqual(support_plane_z, -0.35)
        self.assertAlmostEqual(-0.15 - support_plane_z, 0.2)

        floating_translation_z, floating_support_plane_z = resolve_persistent_rigid_placement(
            translation_z=0.35,
            support_plane_z=0.0,
            joint_count=3,
            translated_free_joint_count=1,
        )
        self.assertAlmostEqual(floating_translation_z, 0.35)
        self.assertEqual(floating_support_plane_z, 0.0)

        transforms = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        relocated = relocate_ground_plane_transform(
            transforms,
            shape_body=np.array([-1, -1, 0]),
            shape_type=np.array([7, 7, 1]),
            shape_label=["ground_plane", "authored_plane", "link_shape"],
            plane_type=7,
            support_plane_z=support_plane_z,
        )

        self.assertAlmostEqual(relocated[0, 2], -0.35)
        np.testing.assert_array_equal(relocated[1:], transforms[1:])
        self.assertEqual(transforms[0, 2], 0.0)

    def test_runaway_trajectory_fails_even_when_frames_exist(self):
        initial = np.array([[-0.2, -0.3, 0.2], [0.2, 0.3, 1.0]])
        tracker = NewtonValidationTracker(
            initial,
            support_plane_z=0.0,
            trajectory_envelope_radius_m=2.0,
            displacement_limit_m=5.0,
            speed_limit_m_s=50.0,
        )
        tracker.observe(initial, np.zeros((1, 3)))
        tracker.observe(
            initial + np.array([0.0, 0.0, 100.0]),
            np.array([[0.0, 0.0, 100.0]]),
        )

        report = tracker.report()

        self.assertEqual(report["schemaVersion"], "palatial.newton-validation.v1")
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["summary"]["passed"])
        self.assertTrue(report["summary"]["finite_transforms"])
        self.assertFalse(report["summary"]["trajectory_stable"])
        self.assertEqual(report["summary"]["sample_count"], 2)
        self.assertAlmostEqual(
            report["summary"]["centroid_in_envelope_sample_ratio"],
            0.5,
        )
        self.assertNotIn("visible_frame_ratio", report["summary"])
        self.assertIn("trajectory_displacement_exceeded", report["failures"])
        self.assertIn("trajectory_speed_exceeded", report["failures"])
        self.assertIn("trajectory_left_envelope", report["failures"])

    def test_floor_aligned_finite_settling_trajectory_passes(self):
        initial = np.array([[-0.2, -0.3, 0.2], [0.2, 0.3, 1.0]])
        tracker = NewtonValidationTracker(initial, support_plane_z=0.0)
        tracker.observe(initial, np.zeros((1, 3)))
        tracker.observe(
            initial + np.array([0.05, 0.0, -0.19]),
            np.array([[0.1, 0.0, -0.2]]),
        )

        report = tracker.report()

        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["summary"]["passed"])
        self.assertEqual(report["failures"], [])
        self.assertAlmostEqual(report["summary"]["initial_penetration_m"], 0.0)
        self.assertAlmostEqual(
            report["summary"]["centroid_in_envelope_sample_ratio"],
            1.0,
        )

    def test_trajectory_penetration_fails_after_clear_spawn(self):
        initial = np.array([[-0.2, -0.3, 0.2], [0.2, 0.3, 1.0]])
        tracker = NewtonValidationTracker(initial, support_plane_z=0.0)
        tracker.observe(initial, np.zeros((1, 3)))
        tracker.observe(
            initial + np.array([0.0, 0.0, -0.25]),
            np.zeros((1, 3)),
        )

        report = tracker.report()

        self.assertEqual(report["status"], "failed")
        self.assertAlmostEqual(report["summary"]["initial_penetration_m"], 0.0)
        self.assertAlmostEqual(report["summary"]["max_support_penetration_m"], 0.05)
        self.assertIn("trajectory_support_penetration", report["failures"])

    def test_small_contact_penetration_stays_within_trajectory_tolerance(self):
        initial = np.array([[-0.2, -0.3, 0.2], [0.2, 0.3, 1.0]])
        tracker = NewtonValidationTracker(initial, support_plane_z=0.0)
        tracker.observe(initial, np.zeros((1, 3)))
        tracker.observe(
            initial + np.array([0.0, 0.0, -0.205]),
            np.zeros((1, 3)),
        )

        report = tracker.report()

        self.assertEqual(report["status"], "passed")
        self.assertAlmostEqual(report["summary"]["max_support_penetration_m"], 0.005)
        self.assertEqual(report["failures"], [])

    def test_cloth_support_penetration_uses_particle_collision_radii(self):
        centers = np.array([[0.0, 0.0, 0.015], [0.1, 0.0, 0.03]])
        tracker = NewtonValidationTracker(
            centers,
            support_plane_z=0.0,
            point_radii=np.array([0.02, 0.005]),
        )
        tracker.observe(centers, np.zeros((2, 3)))

        report = tracker.report()

        self.assertEqual(report["status"], "failed")
        self.assertAlmostEqual(report["summary"]["initial_penetration_m"], 0.005)
        self.assertAlmostEqual(
            report["summary"]["max_support_penetration_m"],
            0.005,
        )
        self.assertIn("initial_support_penetration", report["failures"])

    def test_example_finalizes_and_asserts_semantic_report(self):
        initial = np.array([[-0.2, -0.3, 0.2], [0.2, 0.3, 1.0]])
        tracker = NewtonValidationTracker(
            initial,
            support_plane_z=0.0,
            displacement_limit_m=1.0,
        )
        tracker.observe(initial, np.zeros((1, 3)))
        tracker.observe(initial + np.array([0.0, 0.0, 2.0]), np.zeros((1, 3)))

        with TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "simulation_report.json"
            example = Example.__new__(Example)
            example._validation_tracker = tracker
            example._validation_report_path = report_path

            report = example.test_final(assert_valid=False)

            self.assertEqual(report["status"], "failed")
            self.assertTrue(report_path.is_file())
            with self.assertRaisesRegex(AssertionError, "semantic validation failed"):
                example.test_final()


if __name__ == "__main__":
    unittest.main()
