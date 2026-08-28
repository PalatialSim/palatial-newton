# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

try:
    # register the newton schema plugin before any other USD code is executed
    import newton_usd_schemas  # noqa: F401
except ImportError as exc:
    _newton_usd_schemas_import_error = exc
else:
    _newton_usd_schemas_import_error = None

# Register the Palatial deformable-schema plugin (NewtonDeformableAPI,
# NewtonClothAPI, NewtonShellAPI, NewtonRodAPI + their material APIs).
# These are NOT part of upstream newton-usd-schemas; they ship with the
# Palatial fork under schemas_ext/ and must be registered before the USD
# SchemaRegistry initializes so the loader can read newton:rod:* / shell:* /
# cloth:* attributes off authored prims.
import pathlib as _pathlib

try:
    from pxr import Plug as _Plug
except ImportError:
    _Plug = None

if _Plug is not None:
    _Plug.Registry().RegisterPlugins([str((_pathlib.Path(__file__).parent / "schemas_ext").absolute())])
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


def require_newton_usd_schemas(Usd=None) -> None:
    """Raise if Newton USD schema support is unavailable."""
    if Usd is None:
        return

    if _newton_usd_schemas_import_error is not None:
        raise ImportError(
            "Newton USD support requires newton-usd-schemas. Install the USD importer dependencies with "
            "`pip install 'newton[importers]'`."
        ) from _newton_usd_schemas_import_error


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
    "require_newton_usd_schemas",
    "type_to_warp",
    "value_to_warp",
]
