# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

try:
    # register the newton schema plugin before any other USD code is executed
    import newton_usd_schemas  # noqa: F401
except ImportError as exc:
    _newton_usd_schemas_import_error = exc
else:
    _newton_usd_schemas_import_error = None

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


def _register_deformable_schemas() -> None:
    """Register the fork's USD schemas when the optional USD runtime is present."""
    from pxr import Plug

    registry = Plug.Registry()
    registry.RegisterPlugins([str((Path(__file__).parent / "schemas_ext").absolute())])
    plugin = registry.GetPluginWithName("newton_shell")
    if plugin and not plugin.isLoaded:
        plugin.Load()


if _newton_usd_schemas_import_error is None:
    _register_deformable_schemas()


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
