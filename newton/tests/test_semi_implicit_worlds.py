# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_particle_contacts_isolate_coincident_worlds(test, device):
    template = newton.ModelBuilder(gravity=wp.vec3(0.0))
    for x in (0.0, 0.15):
        template.add_particle(pos=wp.vec3(x, 0.0, 1.0), vel=wp.vec3(0.0), mass=1.0, radius=0.1)
    results = []
    for worlds in (1, 4):
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.replicate(template, worlds)
        model = builder.finalize(device=device)
        state, output = model.state(), model.state()
        newton.solvers.SolverSemiImplicit(model).step(state, output, None, None, 0.001)
        velocity = output.particle_qd.numpy().reshape(worlds, 2, 3)
        test.assertTrue(np.isfinite(velocity).all())
        test.assertLess(velocity[0, 0, 0], 0.0)
        test.assertGreater(velocity[0, 1, 0], 0.0)
        results.append(velocity)
    np.testing.assert_allclose(results[1], np.repeat(results[0], 4, axis=0), atol=1.0e-6, rtol=1.0e-6)


def test_particle_contacts_include_global_particles(test, device):
    template = newton.ModelBuilder(gravity=wp.vec3(0.0))
    template.add_particle(pos=wp.vec3(0.15, 0.0, 1.0), vel=wp.vec3(0.0), mass=1.0, radius=0.1)
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.add_particle(pos=wp.vec3(0.0, 0.0, 1.0), vel=wp.vec3(0.0), mass=0.0, radius=0.1)
    builder.replicate(template, 2)
    model = builder.finalize(device=device)
    state, output = model.state(), model.state()
    newton.solvers.SolverSemiImplicit(model).step(state, output, None, None, 0.001)
    worlds = model.particle_world.numpy()
    velocity = output.particle_qd.numpy()
    test.assertTrue(np.isfinite(velocity).all())
    test.assertTrue(np.all(velocity[worlds >= 0, 0] > 0.0))
    np.testing.assert_array_equal(velocity[worlds == -1], 0.0)


class TestSemiImplicitWorlds(unittest.TestCase):
    pass


for _test in (test_particle_contacts_isolate_coincident_worlds, test_particle_contacts_include_global_particles):
    add_function_test(TestSemiImplicitWorlds, _test.__name__, _test, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main()
