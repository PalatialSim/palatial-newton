# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import gc
import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.geometry import GaussianSkinning
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _fixture(device):
    asset = newton.Gaussian(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        scales=np.full((2, 3), 0.1, dtype=np.float32),
    )
    template = newton.ModelBuilder()
    template.add_particle(pos=wp.vec3(0.0), vel=wp.vec3(0.0), mass=1.0)
    template.add_shape_gaussian(-1, gaussian=asset)
    builder = newton.ModelBuilder()
    builder.add_world(template)
    builder.add_world(template)
    model = builder.finalize(device=device)
    skinning = GaussianSkinning(
        model,
        shape_ids=np.array([0, 1]),
        node_indices=np.array([[0, 1], [0, 1]]),
        node_weights=np.array([[1.0, 0.0], [0.25, 0.75]]),
        node_count=2,
    )
    return asset, model, skinning


def test_skinning_isolation_bounds_and_reset(test, device):
    _, model, skinning = _fixture(device)
    state = model.state()
    physical_before = state.particle_q.numpy().copy()
    nodes = np.zeros((2, 2, 7), dtype=np.float32)
    nodes[:, :, 6] = 1.0
    nodes[0, :, 0] = [3.0, 7.0]
    nodes[1, :, 1] = [-2.0, 2.0]
    transforms = wp.array(nodes, dtype=wp.transform, device=device)
    skinning.update(transforms, state)
    result = skinning.transforms.numpy()
    np.testing.assert_allclose(result[:, :, :3], [[[3, 0, 0], [7, 0, 0]], [[0, -2, 0], [1, 1, 0]]], atol=1.0e-6)
    np.testing.assert_array_equal(state.particle_q.numpy(), physical_before)
    bounds = model.bvh_shape_bounds.numpy()
    test.assertLess(bounds[0, 0, 0], 3.0)
    test.assertGreater(bounds[0, 1, 0], 7.0)
    retained = result[1].copy()
    nodes[0, :, :3] = 0.0
    transforms.assign(nodes)
    skinning.update(transforms, state)
    np.testing.assert_allclose(skinning.transforms.numpy()[0, :, :3], [[0, 0, 0], [1, 0, 0]], atol=1.0e-6)
    np.testing.assert_array_equal(skinning.transforms.numpy()[1], retained)
    test.assertLess(model.bvh_shape_bounds.numpy()[0, 1, 0], 2.0)


def test_skinning_shared_appearance_and_lifetime(test, device):
    asset, model, skinning = _fixture(device)
    a, b = skinning._views
    for field in ("scales", "opacities", "sh_coeffs"):
        test.assertEqual(getattr(a, field).ptr, getattr(b, field).ptr)
    test.assertNotEqual(a.transforms.ptr, b.transforms.ptr)
    template_ptr = model._gaussian_keep_alive[0][0].transforms.ptr
    test.assertNotEqual(a.transforms.ptr, template_ptr)
    # Re-finalizing a source must not free resources used by an older model.
    old_bvh = model._gaussian_keep_alive[0][1]
    asset.finalize(device=device)
    gc.collect()
    test.assertNotEqual(asset.bvh.id, old_bvh.id)
    test.assertEqual(model._gaussian_keep_alive[0][1].id, old_bvh.id)
    nodes = wp.array(np.tile([0, 0, 0, 0, 0, 0, 1], (2, 2, 1)), dtype=wp.transform, device=device)
    skinning.update(nodes, model.state())
    np.testing.assert_allclose(skinning.transforms.numpy()[:, :, 0], [[0, 1], [0, 1]])


def test_skinning_rotation_equivariance(test, device):
    _, model, skinning = _fixture(device)
    nodes = np.zeros((2, 2, 7), dtype=np.float32)
    nodes[:, :, 5:7] = np.sqrt(0.5)
    nodes[:, 1, 3:] *= -1.0
    transforms = wp.array(nodes, dtype=wp.transform, device=device)
    skinning.update(transforms, model.state())
    result = skinning.transforms.numpy()
    np.testing.assert_allclose(result[:, 1, :3], [[0, 1, 0], [0, 1, 0]], atol=1.0e-6)
    np.testing.assert_allclose(np.abs(result[:, :, 5:7]), np.sqrt(0.5), atol=1.0e-6)


def test_skinning_deformed_bounds_remain_visible(test, device):
    _, model, skinning = _fixture(device)
    state = model.state()
    sensor = newton.sensors.SensorTiledCamera(model)
    rays = sensor.utils.compute_camera_rays_pinhole(32, 24, camera_fovs=math.radians(45.0))
    cameras = wp.array(
        [[wp.transform(wp.vec3(3.0, 0.0, 2.0), wp.quat_identity())] * 2], dtype=wp.transform, device=device
    )
    depth = sensor.utils.create_depth_image_output(32, 24, camera_count=1)
    sensor.update(state, cameras, rays, depth_image=depth)
    test.assertEqual(np.count_nonzero(depth.numpy() > 0.0), 0)
    nodes = np.zeros((2, 2, 7), dtype=np.float32)
    nodes[:, :, 6] = 1.0
    nodes[0, :, 0] = 3.0
    skinning.update(wp.array(nodes, dtype=wp.transform, device=device), state)
    sensor.update(state, cameras, rays, depth_image=depth)
    image = depth.numpy()
    test.assertGreater(np.count_nonzero(image[0] > 0.0), 0)
    test.assertEqual(np.count_nonzero(image[1] > 0.0), 0)


def test_skinning_cuda_graph(test, device):
    _, model, skinning = _fixture(device)
    nodes = wp.array(np.tile([0, 0, 0, 0, 0, 0, 1], (2, 2, 1)), dtype=wp.transform, device=device)
    state = model.state()
    skinning.update(nodes, state)
    with wp.ScopedCapture(device=device) as capture:
        skinning.update(nodes, state)
    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(skinning.transforms.numpy()[:, :, 0], [[0, 1], [0, 1]])


class TestGaussianSkinning(unittest.TestCase):
    def test_rejects_registration_after_camera_creation(self):
        _, model, _ = _fixture("cpu")
        newton.sensors.SensorTiledCamera(model)
        with self.assertRaisesRegex(RuntimeError, "before"):
            GaussianSkinning(
                model,
                shape_ids=np.array([0]),
                node_indices=np.array([[0], [0]]),
                node_weights=np.ones((2, 1)),
                node_count=1,
            )


for _device in get_test_devices():
    for _test in (
        test_skinning_isolation_bounds_and_reset,
        test_skinning_shared_appearance_and_lifetime,
        test_skinning_rotation_equivariance,
        test_skinning_deformed_bounds_remain_visible,
    ):
        add_function_test(TestGaussianSkinning, _test.__name__, _test, devices=[_device])
    if _device.is_cuda:
        add_function_test(
            TestGaussianSkinning, test_skinning_cuda_graph.__name__, test_skinning_cuda_graph, devices=[_device]
        )

if __name__ == "__main__":
    unittest.main()
