"""AndroidWorld integration for the IVI visual agent.

``bridge`` (pure translation helpers) imports with no extra dependencies. The
``IviVisualAgent`` adapter needs ``android_world`` installed and is imported
lazily so ``import ivi_agent.integrations.androidworld.bridge`` works without it.
"""

from __future__ import annotations

from . import bridge

__all__ = ["bridge", "IviVisualAgent"]


def __getattr__(name: str):  # PEP 562 lazy attribute
    if name == "IviVisualAgent":
        from .agent import IviVisualAgent

        return IviVisualAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
