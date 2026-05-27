# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Power cable assembly loader helpers for Palatial USD assets."""

from .builder import build_power_cable_assembly
from .canonical import build_power_canonical_assembly
from .constants import (
    DEFAULT_ASSEMBLY_FPS,
    DEFAULT_ASSEMBLY_ITERATIONS,
    DEFAULT_ASSEMBLY_SUBSTEPS,
    POWER_REQUIRED_PRIM_NAMES,
)
from .detection import is_power_cable_assembly_stage
from .extraction import extract_power_cable_assembly
from .model import build_power_cable_assembly_model
from .types import (
    CableExtraction,
    CanonicalCable,
    CanonicalConnectorBody,
    CanonicalPowerCableAssembly,
    ConnectorAttachmentSite,
    ConnectorEndpointExtraction,
    ConnectorMergedMesh,
    EndpointId,
    ExtractedPrim,
    Point3,
    PowerCableAssemblyBuildResult,
    PowerCableAssemblyExtraction,
)

__all__ = [
    "CableExtraction",
    "CanonicalCable",
    "CanonicalConnectorBody",
    "CanonicalPowerCableAssembly",
    "ConnectorAttachmentSite",
    "ConnectorEndpointExtraction",
    "ConnectorMergedMesh",
    "DEFAULT_ASSEMBLY_FPS",
    "DEFAULT_ASSEMBLY_ITERATIONS",
    "DEFAULT_ASSEMBLY_SUBSTEPS",
    "EndpointId",
    "ExtractedPrim",
    "POWER_REQUIRED_PRIM_NAMES",
    "Point3",
    "PowerCableAssemblyBuildResult",
    "PowerCableAssemblyExtraction",
    "build_power_cable_assembly",
    "build_power_cable_assembly_model",
    "build_power_canonical_assembly",
    "extract_power_cable_assembly",
    "is_power_cable_assembly_stage",
]
