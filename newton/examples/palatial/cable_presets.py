# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Named anisotropic cable presets for Palatial example generation."""

from __future__ import annotations


_ANISOTROPIC_CABLE_PRESETS: dict[str, dict[str, object]] = {
    "flat_balanced_demo": {
        "cross_section_type": "flatRect",
        "length": 1.5,
        "segment_count": 16,
        "drop_height": 0.3,
        "twist_total": 0.0,
        "width": 0.012,
        "thickness": 0.004,
        "density": 1000.0,
        "stretch_stiffness": 1.0e5,
        "stretch_damping": 0.05,
        "compress_stiffness": 1.0e5,
        "compress_damping": 0.05,
        "bend_y_stiffness": 8.0e2,
        "bend_y_damping": 0.1,
        "bend_z_stiffness": 1.6e3,
        "bend_z_damping": 0.1,
        "torsion_stiffness": 4.0e2,
        "torsion_damping": 0.05,
        "fps": 120,
        "solver": "vbd_palatial",
        "solver_iterations": 2,
        "solver_substeps": 4,
    },
    "flat_bend_z_dominant": {
        "cross_section_type": "flatRect",
        "length": 1.8,
        "segment_count": 18,
        "drop_height": 0.35,
        "twist_total": 0.15,
        "width": 0.02,
        "thickness": 0.0025,
        "density": 1100.0,
        "stretch_stiffness": 1.5e5,
        "stretch_damping": 0.06,
        "compress_stiffness": 1.5e5,
        "compress_damping": 0.06,
        "bend_y_stiffness": 4.0e2,
        "bend_y_damping": 0.08,
        "bend_z_stiffness": 3.0e3,
        "bend_z_damping": 0.14,
        "torsion_stiffness": 1.5e2,
        "torsion_damping": 0.04,
        "fps": 120,
        "solver": "vbd_palatial",
        "solver_iterations": 2,
        "solver_substeps": 6,
    },
    "flat_bend_y_dominant": {
        "cross_section_type": "flatRect",
        "length": 1.2,
        "segment_count": 14,
        "drop_height": 0.25,
        "twist_total": 0.3,
        "width": 0.008,
        "thickness": 0.006,
        "density": 950.0,
        "stretch_stiffness": 1.2e5,
        "stretch_damping": 0.05,
        "compress_stiffness": 1.2e5,
        "compress_damping": 0.05,
        "bend_y_stiffness": 2.4e3,
        "bend_y_damping": 0.12,
        "bend_z_stiffness": 6.0e2,
        "bend_z_damping": 0.08,
        "torsion_stiffness": 3.0e2,
        "torsion_damping": 0.05,
        "fps": 120,
        "solver": "vbd_palatial",
        "solver_iterations": 2,
        "solver_substeps": 5,
    },
    "round_low_torsion_demo": {
        "cross_section_type": "roundSolid",
        "length": 2.0,
        "segment_count": 20,
        "drop_height": 0.4,
        "twist_total": 0.5,
        "radius": 0.006,
        "density": 1050.0,
        "stretch_stiffness": 8.0e4,
        "stretch_damping": 0.04,
        "compress_stiffness": 8.0e4,
        "compress_damping": 0.04,
        "bend_y_stiffness": 1.2e3,
        "bend_y_damping": 0.1,
        "bend_z_stiffness": 1.2e3,
        "bend_z_damping": 0.1,
        "torsion_stiffness": 8.0e1,
        "torsion_damping": 0.03,
        "fps": 120,
        "solver": "vbd_palatial",
        "solver_iterations": 2,
        "solver_substeps": 6,
    },
}


def list_anisotropic_cable_presets() -> tuple[str, ...]:
    """Return the available named anisotropic cable presets."""

    return tuple(_ANISOTROPIC_CABLE_PRESETS)


def get_anisotropic_cable_preset(name: str) -> dict[str, object]:
    """Return a copy of one named anisotropic cable preset."""

    preset = _ANISOTROPIC_CABLE_PRESETS.get(name)
    if preset is None:
        available = ", ".join(list_anisotropic_cable_presets())
        raise ValueError(f"Unknown anisotropic cable preset {name!r}. Available presets: {available}")
    return dict(preset)

