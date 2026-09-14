# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure Gaussian deformation, bounds and BVH updates independently of physics."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.geometry import GaussianSkinning


class GaussianDeformation:
    params = ([1, 8, 32, 128, 256], [1000, 10000])
    param_names = ["worlds", "points"]
    repeat = 3
    number = 1

    def setup(self, worlds, points):
        if not wp.is_cuda_available():
            raise NotImplementedError("CUDA device required")
        self.device = wp.get_device("cuda:0")
        rng = np.random.default_rng(7)
        positions = rng.uniform(-0.5, 0.5, (points, 3)).astype(np.float32)
        asset = newton.Gaussian(
            positions,
            scales=np.full((points, 3), 0.005, dtype=np.float32),
            sh_coeffs=rng.uniform(-0.25, 0.25, (points, 48)).astype(np.float32),
        )
        template = newton.ModelBuilder()
        template.add_shape_gaussian(-1, gaussian=asset)
        builder = newton.ModelBuilder()
        builder.replicate(template, worlds)
        self.model = builder.finalize(device=self.device)
        self.state = self.model.state()
        indices = rng.integers(0, 64, (points, 4), dtype=np.int32)
        weights = rng.uniform(0.1, 1.0, (points, 4)).astype(np.float32)
        weights /= weights.sum(axis=1, keepdims=True)
        self.skinning = GaussianSkinning(
            self.model,
            shape_ids=np.arange(worlds),
            node_indices=indices,
            node_weights=weights,
            node_count=64,
        )
        nodes = np.zeros((worlds, 64, 7), dtype=np.float32)
        nodes[:, :, :3] = rng.uniform(-0.02, 0.02, (worlds, 64, 3))
        nodes[:, :, 6] = 1.0
        self.nodes = wp.array(nodes, dtype=wp.transform, device=self.device)
        self.skinning.update(self.nodes, self.state)
        with wp.ScopedCapture(device=self.device) as capture:
            self.skinning.update(self.nodes, self.state)
        self.graph = capture.graph
        wp.synchronize_device(self.device)

    def time_update_graph(self, worlds, points):
        for _ in range(100):
            wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)

    def track_shared_appearance_bytes(self, worlds, points):
        view = self.skinning._views[0]
        return sum(
            array.size * wp.types.type_size_in_bytes(array.dtype)
            for array in (view.scales, view.opacities, view.sh_coeffs)
        )

    def track_pose_and_leaf_bounds_bytes(self, worlds, points):
        return worlds * points * (7 + 3 + 3) * 4

    def teardown(self, worlds, points):
        self.graph = None
        self.skinning = None
        self.model = None
        self.state = None
        self.nodes = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, nargs="+", default=[1, 8, 32, 128, 256])
    parser.add_argument("--points", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wp.init()
    records = []
    for worlds in args.worlds:
        benchmark = GaussianDeformation()
        start = time.perf_counter()
        benchmark.setup(worlds, args.points)
        setup_seconds = time.perf_counter() - start
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            benchmark.time_update_graph(worlds, args.points)
            samples.append((time.perf_counter() - start) * 1000.0 / 100)
        record = {
            "workload": "synthetic_gaussian_deformation_with_bvh",
            "gpu": benchmark.device.name,
            "warp_version": wp.__version__,
            "worlds": worlds,
            "points_per_world": args.points,
            "influences": 4,
            "bvh_layout": "grouped",
            "setup_seconds": setup_seconds,
            "update_ms_samples": samples,
            "instance_updates_per_second": worlds * 1000.0 / float(np.median(samples)),
            "shared_appearance_bytes": benchmark.track_shared_appearance_bytes(worlds, args.points),
            "pose_and_leaf_bounds_bytes": benchmark.track_pose_and_leaf_bounds_bytes(worlds, args.points),
            "includes_physics": False,
            "includes_rendering": False,
            "includes_host_submission": True,
            "includes_bvh_refits": True,
        }
        records.append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        benchmark.teardown(worlds, args.points)


if __name__ == "__main__":
    main()
