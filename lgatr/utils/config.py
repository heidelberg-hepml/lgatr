"""Shared caster for the L-GATr configuration dataclasses."""

from collections.abc import Mapping
from typing import Any


def cast_config(cls: type, config: Any):
    """Cast an instance or a mapping to ``cls``; raise on anything else (including None)."""
    if isinstance(config, cls):
        return config
    if isinstance(config, Mapping):
        return cls(**config)
    raise ValueError(f"Cannot cast {config!r} to {cls.__name__}")
