# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.semi_implicit.kernels_particle import eval_spring_forces
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _world(extra=False):
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    for position in ((0, 0, 1), (1, 0, 1), (0, 1, 1)):
        builder.add_particle(pos=wp.vec3(*position), vel=wp.vec3(0.0), mass=1.0)
    builder.add_spring(0, 1, ke=10.0, kd=0.1, control=0.0)
    builder.add_spring(0, 2, ke=20.0, kd=0.2, control=0.0)
    if extra:
        builder.add_particle(pos=wp.vec3(1.0, 1.0, 1.0), vel=wp.vec3(0.0), mass=1.0)
        builder.add_spring(2, 3, ke=15.0, kd=0.1, control=0.0)
    return builder


def test_gather_matches_atomic_trajectory_and_live_materials(test, device):
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.replicate(_world(), 4)
    model = builder.finalize(device=device)
    atomic = newton.solvers.SolverSemiImplicit(model)
    gather = newton.solvers.SolverSemiImplicit(model, spring_force_mode="gather")
    test.assertEqual(gather._spring_gather.particles_per_world, 3)
    test.assertEqual(gather._spring_gather.indices.size, 4)
    test.assertEqual(gather._spring_gather.signed_springs.size, 4)
    stiffness = model.spring_stiffness.numpy()
    stiffness[2:4] *= 2.0
    model.spring_stiffness.assign(stiffness)
    position = model.particle_q.numpy()
    position[1::3, 0] += 0.1
    initial = []
    for _ in range(2):
        state = model.state()
        state.particle_q.assign(position)
        initial.append(state)
    outputs = [model.state(), model.state()]
    external = np.tile([0.25, -0.1, 0.0], (model.particle_count, 1)).astype(np.float32)
    for _ in range(32):
        for solver, state, output in zip((atomic, gather), initial, outputs, strict=True):
            state.clear_forces()
            state.particle_f.assign(external)
            solver.step(state, output, None, None, 0.001)
        initial, outputs = outputs, initial
    np.testing.assert_allclose(initial[0].particle_q.numpy(), initial[1].particle_q.numpy(), atol=1.0e-6, rtol=1.0e-6)
    np.testing.assert_allclose(initial[0].particle_qd.numpy(), initial[1].particle_qd.numpy(), atol=1.0e-6, rtol=1.0e-6)
    test.assertGreater(np.linalg.norm(initial[1].particle_qd.numpy()), 0.0)
    test.assertGreater(np.linalg.norm(initial[1].particle_qd.numpy()[0] - initial[1].particle_qd.numpy()[3]), 0.01)


def test_gather_heterogeneous_world_fallback(test, device):
    builder = newton.ModelBuilder()
    builder.add_world(_world())
    builder.add_world(_world(extra=True))
    model = builder.finalize(device=device)
    gather = newton.solvers.SolverSemiImplicit(model, spring_force_mode="gather")
    test.assertEqual(gather._spring_gather.particles_per_world, model.particle_count)
    state = model.state()
    state.particle_qd.fill_(wp.vec3(0.1, 0.2, 0.3))
    atomic_out, gather_out = model.state(), model.state()
    newton.solvers.SolverSemiImplicit(model).step(state, atomic_out, None, None, 0.001)
    state.clear_forces()
    gather.step(state, gather_out, None, None, 0.001)
    np.testing.assert_allclose(atomic_out.particle_q.numpy(), gather_out.particle_q.numpy(), atol=1.0e-6, rtol=1.0e-6)


@wp.kernel
def _force_loss(forces: wp.array[wp.vec3], loss: wp.array[float]):
    loss[0] = forces[0][0]


def test_atomic_spring_position_gradient(test, device):
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.add_particle(pos=wp.vec3(0.0), vel=wp.vec3(0.0), mass=1.0)
    builder.add_particle(pos=wp.vec3(1.0, 0.0, 0.0), vel=wp.vec3(0.0), mass=1.0)
    builder.add_spring(0, 1, ke=10.0, kd=0.0, control=0.0)
    model = builder.finalize(device=device, requires_grad=True)
    state = model.state()
    state.particle_q.assign(np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]))
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    with wp.Tape() as tape:
        eval_spring_forces(model, state, state.particle_f)
        wp.launch(_force_loss, dim=1, inputs=[state.particle_f, loss], device=device)
    tape.backward(loss)
    np.testing.assert_allclose(state.particle_q.grad.numpy()[:, 0], [-10.0, 10.0], atol=1.0e-5)


def test_gather_graph_replay_reads_live_materials(test, device):
    model = _world().finalize(device=device)
    gather = newton.solvers.SolverSemiImplicit(model, spring_force_mode="gather")._spring_gather
    state = model.state()
    position = state.particle_q.numpy()
    position[1, 0] = 1.1
    state.particle_q.assign(position)
    gather.evaluate(model, state, state.particle_f)
    with wp.ScopedCapture(device=device) as capture:
        state.clear_forces()
        gather.evaluate(model, state, state.particle_f)
    wp.capture_launch(capture.graph)
    original = state.particle_f.numpy()
    model.spring_stiffness.assign(model.spring_stiffness.numpy() * 2.0)
    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(state.particle_f.numpy(), original * 2.0, atol=1.0e-6)


class TestSpringGather(unittest.TestCase):
    def test_rejects_invalid_mode(self):
        with self.assertRaisesRegex(ValueError, "spring_force_mode"):
            newton.solvers.SolverSemiImplicit(_world().finalize(device="cpu"), spring_force_mode="unknown")

    def test_rejects_differentiable_model(self):
        with self.assertRaisesRegex(ValueError, "forward"):
            newton.solvers.SolverSemiImplicit(
                _world().finalize(device="cpu", requires_grad=True), spring_force_mode="gather"
            )


for _test in (
    test_gather_matches_atomic_trajectory_and_live_materials,
    test_gather_heterogeneous_world_fallback,
    test_atomic_spring_position_gradient,
):
    add_function_test(TestSpringGather, _test.__name__, _test, devices=get_test_devices())

for _device in get_test_devices():
    if _device.is_cuda:
        add_function_test(
            TestSpringGather,
            "test_gather_graph_replay_reads_live_materials",
            test_gather_graph_replay_reads_live_materials,
            devices=[_device],
        )

if __name__ == "__main__":
    unittest.main()
