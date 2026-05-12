# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

try:
    # register the newton schema plugin before any other USD code is executed
    import newton_usd_schemas  # noqa: F401
except ImportError:
    pass

#defining our APIS
import pathlib as _pathlib

from pxr import Plug as _Plug

_Plug.Registry().RegisterPlugins([
    str((_pathlib.Path(__file__).parent / "schemas_ext").absolute())
])
_p = _Plug.Registry().GetPluginWithName("newton_shell")
if _p and not _p.isLoaded:
    _p.Load()

from .utils import (
    get_attribute,
    get_attributes_in_namespace,
    get_custom_attribute_declarations,
    get_custom_attribute_values,
    get_float,
    get_gaussian,
    get_gprim_axis,
    get_mesh,
    get_quat,
    get_scale,
    get_transform,
    has_attribute,
    type_to_warp,
    value_to_warp,
)

__all__ = [
    "get_attribute",
    "get_attributes_in_namespace",
    "get_custom_attribute_declarations",
    "get_custom_attribute_values",
    "get_float",
    "get_gaussian",
    "get_gprim_axis",
    "get_mesh",
    "get_quat",
    "get_scale",
    "get_transform",
    "has_attribute",
    "type_to_warp",
    "value_to_warp",
]
