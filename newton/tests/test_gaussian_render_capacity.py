# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import newton
from newton._src.sensors.warp_raytrace.render_context import RenderContext


class TestGaussianRenderCapacity(unittest.TestCase):
    def _asset(self, offset=0.0):
        return newton.Gaussian(np.array([[offset, 0.0, 0.0]], dtype=np.float32))

    def _context(self, builder):
        model = builder.finalize(device="cpu")
        context = RenderContext(device="cpu")
        context.init_from_model(model, load_textures=False)
        return model, context

    def test_repeated_shapes_need_distinct_hit_slots(self):
        builder = newton.ModelBuilder()
        asset = self._asset()
        for _ in range(3):
            builder.add_shape_gaussian(-1, gaussian=asset)
        model, context = self._context(builder)
        self.assertEqual(model.gaussians_count, 1)
        self.assertEqual(context.state.num_gaussians, 3)

    def test_independent_worlds_do_not_expand_ray_scratch(self):
        builder = newton.ModelBuilder()
        for world in range(8):
            template = newton.ModelBuilder()
            template.add_shape_gaussian(-1, gaussian=self._asset(float(world)))
            builder.add_world(template)
        model, context = self._context(builder)
        self.assertEqual(model.gaussians_count, 8)
        self.assertEqual(context.state.num_gaussians, 1)

    def test_global_group_and_worlds_use_largest_group(self):
        builder = newton.ModelBuilder()
        asset = self._asset()
        for _ in range(5):
            builder.add_shape_gaussian(-1, gaussian=asset)
        template = newton.ModelBuilder()
        for _ in range(3):
            template.add_shape_gaussian(-1, gaussian=asset)
        builder.add_world(template)
        _, context = self._context(builder)
        self.assertEqual(context.state.num_gaussians, 5)


if __name__ == "__main__":
    unittest.main()
