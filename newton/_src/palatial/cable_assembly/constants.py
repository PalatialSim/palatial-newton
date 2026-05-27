# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Constants for the v1 Power cable assembly asset shape."""

from __future__ import annotations

POWER_CABLE_REQUIRED_PRIM_NAMES: tuple[str, ...] = ("Power_Cable_Body",)
POWER_IEC_REQUIRED_PRIM_NAMES: tuple[str, ...] = (
    "Pow_IEC_Strain",
    "Pow_IEC_Body",
    "Pow_IEC_Recess",
    "Pow_IEC_Slot0",
    "Pow_IEC_Slot1",
    "Pow_IEC_ESlot",
)
POWER_NEMA_REQUIRED_PRIM_NAMES: tuple[str, ...] = (
    "Pow_NEMA_Strain",
    "Pow_NEMA_Body",
    "Pow_NEMA_Face",
    "Pow_NEMA_Hot",
    "Pow_NEMA_Neut",
    "Pow_NEMA_Gnd",
    "Pow_NEMA_FS0",
    "Pow_NEMA_FS1",
    "Pow_NEMA_GS",
)
POWER_REQUIRED_PRIM_NAMES: tuple[str, ...] = (
    *POWER_CABLE_REQUIRED_PRIM_NAMES,
    *POWER_IEC_REQUIRED_PRIM_NAMES,
    *POWER_NEMA_REQUIRED_PRIM_NAMES,
)

DEFAULT_ASSEMBLY_FPS = 60
DEFAULT_ASSEMBLY_SUBSTEPS = 10
DEFAULT_ASSEMBLY_ITERATIONS = 2
DEFAULT_ASSEMBLY_ROD_SEGMENT_COUNT = 60
DEFAULT_CONNECTOR_MASS = 1.0e-1
DEFAULT_ROD_DENSITY = 1000.0
DEFAULT_ROD_STRETCH_STIFFNESS = 1.0e6
DEFAULT_ROD_BEND_STIFFNESS = 1.0e2
DEFAULT_ROD_TORSION_STIFFNESS = 1.0e2
DEFAULT_ROD_DAMPING = 1.0e-1
DEFAULT_CONTACT_KE = 1.0e4
DEFAULT_CONTACT_KD = 1.0e-1
DEFAULT_CONTACT_KF = 1.0e3
DEFAULT_CONTACT_MU = 1.0

GEOMETRY_SCOPE_PATH = "/World/Geometry"
PCA_CABLE_EPS = 1.0e-9
