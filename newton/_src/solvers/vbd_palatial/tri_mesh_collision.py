# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compatibility re-export for Palatial VBD triangle collision helpers."""

from ..vbd import tri_mesh_collision as _tri_mesh_collision

for _name in dir(_tri_mesh_collision):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_tri_mesh_collision, _name)

__all__ = [
    name
    for name in dir(_tri_mesh_collision)
    if not name.startswith("__")
]
