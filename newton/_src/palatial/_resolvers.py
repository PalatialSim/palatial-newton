"""Locate Newton's schema resolver classes regardless of API location."""
from __future__ import annotations


def get_default_resolvers():
    """Return [SchemaResolverNewton, SchemaResolverPhysx, SchemaResolverMjc] instances.

    Tries public path first, falls back to the internal `_src` path used by
    newton 0.2.0. Returns None if not available; caller should pass that
    through unchanged so add_usd uses its own default.
    """
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
