# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare spring force assembly with identical states and material parameters."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton._src.solvers.semi_implicit.kernels_particle import eval_spring_forces
from newton._src.solvers.semi_implicit.spring_gather import SpringForceGather


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", nargs="+", type=int, default=[1, 8, 32, 128, 256])
    parser.add_argument("--particles", type=int, default=1024)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument(
        "--fixture", type=Path, help="NPZ containing positions, springs, rest_lengths, stiffness, damping"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(7)
    template = newton.ModelBuilder()
    fixture = None
    fixture_sha256 = None
    if args.fixture is not None:
        fixture = np.load(args.fixture, allow_pickle=False)
        fixture_sha256 = hashlib.sha256(args.fixture.read_bytes()).hexdigest()
        for position in fixture["positions"]:
            template.add_particle(pos=wp.vec3(position), vel=wp.vec3(0.0), mass=1.0)
        for (i, j), stiffness, damping in zip(
            fixture["springs"], fixture["stiffness"], fixture["damping"], strict=True
        ):
            template.add_spring(int(i), int(j), ke=float(stiffness), kd=float(damping), control=0.0)
        args.particles = len(fixture["positions"])
    else:
        for position in rng.uniform(-0.5, 0.5, (args.particles, 3)):
            template.add_particle(pos=wp.vec3(position), vel=wp.vec3(0.0), mass=1.0)
        for node in range(args.particles):
            for offset in range(1, args.neighbors + 1):
                template.add_spring(node, (node + offset) % args.particles, ke=10.0, kd=0.1, control=0.0)
    records = []
    for worlds in args.worlds:
        builder = newton.ModelBuilder()
        builder.replicate(template, worlds)
        model = builder.finalize(device="cuda:0")
        if fixture is not None:
            model.spring_rest_length.assign(np.tile(fixture["rest_lengths"], worlds))
        state = model.state()
        positions = state.particle_q.numpy()
        positions += rng.uniform(-0.01, 0.01, positions.shape).astype(np.float32)
        state.particle_q.assign(positions)
        gather = SpringForceGather(model)
        reference = None
        for mode in ("atomic", "gather"):

            def evaluate(state=state, mode=mode, model=model, gather=gather):
                state.clear_forces()
                if mode == "atomic":
                    eval_spring_forces(model, state, state.particle_f)
                else:
                    gather.evaluate(model, state, state.particle_f)

            evaluate()
            forces = state.particle_f.numpy()
            if reference is None:
                reference = forces
            relative_l2_error = float(np.linalg.norm(forces - reference) / max(np.linalg.norm(reference), 1.0e-30))
            if fixture is None:
                np.testing.assert_allclose(forces, reference, atol=2.0e-5, rtol=2.0e-5)
            elif relative_l2_error > 1.0e-6:
                # Stiff networks have cancellation near zero in individual
                # components; report absolute errors and bound the vector norm.
                raise AssertionError(f"Spring force relative L2 error: {relative_l2_error}")
            with wp.ScopedCapture(device=model.device) as capture:
                evaluate()
            for _ in range(20):
                wp.capture_launch(capture.graph)
            wp.synchronize_device(model.device)
            samples = []
            for _ in range(5):
                start = time.perf_counter()
                for _ in range(200):
                    wp.capture_launch(capture.graph)
                wp.synchronize_device(model.device)
                samples.append((time.perf_counter() - start) * 1000.0 / 200)
            record = {
                "workload": "spring_force_assembly",
                "fixture_sha256": fixture_sha256,
                "gpu": model.device.name,
                "warp_version": wp.__version__,
                "mode": mode,
                "worlds": worlds,
                "particles_per_world": args.particles,
                "springs_per_world": template.spring_count,
                "update_ms_samples": samples,
                "max_force_error": float(np.max(np.abs(forces - reference))),
                "relative_l2_force_error": relative_l2_error,
                "max_reference_force": float(np.max(np.abs(reference))),
                "includes_integration": False,
                "includes_contacts": False,
                "includes_rendering": False,
                "includes_force_clear": True,
                "includes_host_submission": True,
            }
            records.append(record)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
