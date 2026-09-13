# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Two-pass spring force assembly with reusable incidence for repeated worlds."""

import numpy as np
import warp as wp

from ...sim import Model, State
from .kernels_particle import spring_force


@wp.kernel(enable_backward=False)
def _compute_spring_forces(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    rest_lengths: wp.array[wp.float32],
    stiffness: wp.array[wp.float32],
    damping: wp.array[wp.float32],
    particles_per_world: int,
    springs_per_world: int,
    forces: wp.array[wp.vec3],
):
    spring = wp.tid()
    world = spring // springs_per_world
    local = spring % springs_per_world
    i = indices[2 * local]
    j = indices[2 * local + 1]
    force = wp.vec3(0.0)
    if i >= 0 and j >= 0:
        i += world * particles_per_world
        j += world * particles_per_world
        force = spring_force(
            positions[i] - positions[j],
            velocities[i] - velocities[j],
            rest_lengths[spring],
            stiffness[spring],
            damping[spring],
        )
    forces[spring] = force


@wp.kernel(enable_backward=False)
def _gather_particle_forces(
    offsets: wp.array[wp.int32],
    signed_springs: wp.array[wp.int32],
    spring_forces: wp.array[wp.vec3],
    particles_per_world: int,
    springs_per_world: int,
    forces: wp.array[wp.vec3],
):
    particle = wp.tid()
    world = particle // particles_per_world
    local = particle % particles_per_world
    total = wp.vec3(0.0)
    for entry in range(offsets[local], offsets[local + 1]):
        signed = signed_springs[entry]
        spring = wp.abs(signed) - 1 + world * springs_per_world
        if signed < 0:
            total -= spring_forces[spring]
        else:
            total += spring_forces[spring]
    forces[particle] += total


class SpringForceGather:
    """Cache structural incidence while reading live per-spring material arrays."""

    def __init__(self, model: Model):
        if model.requires_grad:
            raise ValueError("Spring gather currently supports forward simulation only")
        links = model.spring_indices.numpy().reshape(-1, 2)
        self.particles_per_world = model.particle_count
        self.springs_per_world = model.spring_count
        worlds = model.world_count
        if worlds > 1 and model.particle_count % worlds == 0 and model.spring_count % worlds == 0:
            particles = model.particle_count // worlds
            springs = model.spring_count // worlds
            particle_world = model.particle_world.numpy()
            if np.array_equal(particle_world, np.repeat(np.arange(worlds), particles)):
                blocks = links.reshape(worlds, springs, 2)
                normalized = np.where(blocks >= 0, blocks - np.arange(worlds)[:, None, None] * particles, blocks)
                if np.array_equal(normalized, np.broadcast_to(normalized[0], normalized.shape)):
                    links = normalized[0]
                    self.particles_per_world = particles
                    self.springs_per_world = springs
        valid = np.all(links >= 0, axis=1)
        ids = np.flatnonzero(valid).astype(np.int32) + 1
        nodes = np.concatenate((links[valid, 0], links[valid, 1]))
        signed = np.concatenate((-ids, ids))
        order = np.argsort(nodes, kind="stable")
        offsets = np.zeros(self.particles_per_world + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(np.bincount(nodes, minlength=self.particles_per_world))
        with wp.ScopedDevice(model.device):
            self.indices = wp.array(links.reshape(-1), dtype=wp.int32)
            self.offsets = wp.array(offsets, dtype=wp.int32)
            self.signed_springs = wp.array(signed[order], dtype=wp.int32)
            self.forces = wp.empty(model.spring_count, dtype=wp.vec3)

    def evaluate(self, model: Model, state: State, forces: wp.array[wp.vec3]) -> None:
        if state.particle_q.requires_grad or state.particle_qd.requires_grad:
            raise ValueError("Spring gather currently supports forward simulation only")
        wp.launch(
            _compute_spring_forces,
            dim=model.spring_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                self.indices,
                model.spring_rest_length,
                model.spring_stiffness,
                model.spring_damping,
                self.particles_per_world,
                self.springs_per_world,
                self.forces,
            ],
            device=model.device,
        )
        wp.launch(
            _gather_particle_forces,
            dim=model.particle_count,
            inputs=[
                self.offsets,
                self.signed_springs,
                self.forces,
                self.particles_per_world,
                self.springs_per_world,
                forces,
            ],
            device=model.device,
        )
