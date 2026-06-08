# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Locate Newton's schema resolver classes regardless of API location."""
from __future__ import annotations


def get_default_resolvers():
    """Return ``[SchemaResolverNewton, SchemaResolverPhysx, SchemaResolverMjc]`` or ``None``."""
    try:
        from newton.usd import (  # type: ignore
            SchemaResolverNewton, SchemaResolverPhysx, SchemaResolverMjc,
        )
    except ImportError:
        try:
            from newton._src.usd.schemas import (  # type: ignore
                SchemaResolverNewton, SchemaResolverPhysx, SchemaResolverMjc,
            )
        except ImportError:
            return None
    return [SchemaResolverNewton(), SchemaResolverPhysx(), SchemaResolverMjc()]
