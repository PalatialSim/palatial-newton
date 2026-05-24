# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compatibility re-export for Palatial VBD particle kernels."""

from ..vbd import particle_vbd_kernels as _particle_vbd_kernels

for _name in dir(_particle_vbd_kernels):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_particle_vbd_kernels, _name)

__all__ = [
    name
    for name in dir(_particle_vbd_kernels)
    if not name.startswith("__")
]
