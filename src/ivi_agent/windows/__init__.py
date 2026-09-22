"""Windows desktop (.exe) backend for the agent, via Appium / UI Automation.

The whole agent brain is platform-agnostic; only the device layer is new here.
``WindowsDevice`` implements ``ivi_agent.device.Device`` over an Appium WebDriver
session (WinAppDriver under the hood), and ``uia_to_uiautomator`` normalises the
Windows UI-Automation tree into the same uiautomator-style XML the shared
perception layer already parses — so grounding, the scene graph, verification and
learning all work unchanged.

Install with the ``[windows]`` extra (Appium client); Appium is imported lazily
so importing this package never requires it.
"""

from __future__ import annotations

from .uia import uia_to_uiautomator
from .device import WindowsDevice

__all__ = ["uia_to_uiautomator", "WindowsDevice"]
