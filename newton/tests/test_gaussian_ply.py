# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

import newton


@unittest.skipUnless(importlib.util.find_spec("open3d") is not None, "Requires open3d")
class TestGaussianPly(unittest.TestCase):
    def load_scales(self, scales):
        properties = ["x", "y", "z", *[f"scale_{i}" for i in range(len(scales))]]
        header = ["ply", "format ascii 1.0", "element vertex 1"]
        header.extend(f"property float {name}" for name in properties)
        header.append("end_header")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.ply"
            path.write_text("\n".join(header) + "\n" + " ".join(map(str, [0, 0, 0, *scales])) + "\n")
            return newton.Gaussian.create_from_ply(str(path))

    def test_isotropic_scale(self):
        asset = self.load_scales([np.log(0.125)])
        np.testing.assert_allclose(asset.scales, [[0.125, 0.125, 0.125]], atol=1.0e-7)

    def test_anisotropic_scales(self):
        asset = self.load_scales(np.log([0.125, 0.25, 0.5]))
        np.testing.assert_allclose(asset.scales, [[0.125, 0.25, 0.5]], atol=1.0e-7)

    def test_rejects_incomplete_anisotropic_scales(self):
        with self.assertRaisesRegex(ValueError, "scale attributes"):
            self.load_scales([0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
