"""The device interface the agent drives.

The agent is platform-agnostic: it observes (screenshot + UI tree), grounds, acts,
and verifies through this small interface. ``AdbDevice`` (Android) implements it;
``ivi_agent.windows.WindowsDevice`` (Windows desktop, via Appium/UIA) is a second
backend. Anything above this line — grounding ladder, scene graph, manual RAG,
verification, replay, learning — is shared across platforms.

This is a structural ``Protocol``: existing devices conform without subclassing.
It documents the contract and lets the backends be type-checked and swapped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from .types import Action


@runtime_checkable
class Device(Protocol):
    """What a platform backend must provide for the agent to run."""

    def ensure_ready(self) -> None:
        """Raise if the target isn't reachable/ready."""

    def wake_if_needed(self) -> None:
        """Best-effort: bring the target to an interactive state."""

    def screen_size(self) -> tuple[int, int]:
        """Current framebuffer size in pixels (width, height)."""

    def capture(self, destination: Path) -> bytes:
        """Screenshot the target, write it to ``destination``, return the bytes."""

    def ui_dump(self) -> str:
        """The accessibility tree as uiautomator-style XML (see windows.uia).

        Every node carries ``bounds='[x1,y1][x2,y2]'`` plus ``text`` /
        ``content-desc`` / ``resource-id`` / ``class`` / ``clickable`` so the
        shared perception layer parses it identically across platforms.
        """

    def execute(self, action: Action, size: tuple[int, int]) -> None:
        """Perform ``action`` (tap/swipe/text/...) against a screen of ``size``."""

    def wait_until_stable(self, directory: Path, timeout: float, poll: float) -> None:
        """Block until the screen stops changing (or ``timeout`` elapses)."""

    # Optional lifecycle / diagnostics (agent calls these behind config flags).
    def clear_logcat(self) -> None:
        """Clear the platform log buffer (no-op where there isn't one)."""

    def logcat_dump(self, tail_lines: int = 4000) -> str:
        """Return recent platform logs for crash scanning ("" where none)."""

    def relaunch(self, package: str) -> None:
        """Cold-restart the app under test for clean-start hygiene."""
