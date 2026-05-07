"""Shared dependencies for Sigma controller variants."""
from __future__ import annotations

from .force_dimension_sdk import ForceDimensionSDK

try:
    from .config import DEFAULT_PROFILE, TELEOP_CONFIGS
except ImportError:
    DEFAULT_PROFILE = None
    TELEOP_CONFIGS = {}

__all__ = ["DEFAULT_PROFILE", "ForceDimensionSDK", "TELEOP_CONFIGS"]
