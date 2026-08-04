# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

from newton._src.palatial.load import _synchronize_newton_contact_capacity


class TestPalatialLoad(unittest.TestCase):
    def test_synchronizes_newton_and_mujoco_contact_capacity(self):
        model = SimpleNamespace(rigid_contact_max=1160)
        solver_params = {
            "nconmax": 4096,
            "use_mujoco_contacts": False,
        }

        _synchronize_newton_contact_capacity(model, "mujoco", solver_params)

        self.assertEqual(model.rigid_contact_max, 4096)

    def test_preserves_capacity_for_native_mujoco_contacts(self):
        model = SimpleNamespace(rigid_contact_max=1160)
        solver_params = {
            "nconmax": 4096,
            "use_mujoco_contacts": True,
        }

        _synchronize_newton_contact_capacity(model, "mujoco", solver_params)

        self.assertEqual(model.rigid_contact_max, 1160)


if __name__ == "__main__":
    unittest.main()
