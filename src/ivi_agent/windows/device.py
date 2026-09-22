"""WindowsDevice: the agent's device backend for Windows desktop apps.

Implements ``ivi_agent.device.Device`` over an Appium WebDriver session
(WinAppDriver under the hood): screenshots via ``get_screenshot_as_png``, the UI
tree via ``page_source`` (converted by ``uia_to_uiautomator``), and actions via
Appium's ``windows:`` gesture extensions.

The Appium client is imported lazily in ``connect()`` so importing this module
never requires the ``[windows]`` extra; the class itself takes an already-created
driver, which makes it unit-testable with a fake driver (no Windows needed).

Phase-0 scope: capture / ui_dump / screen_size / tap+text execution are wired to
the driver; DPI/multi-monitor coordinate handling, swipe/gesture parity,
Event-Log/WER crash capture, and clean-start launch/kill are Phase 1–2 (marked
below). Everything above the device — grounding, scene graph, verification,
learning — already works once these return the shared schema.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..types import Action
from .uia import uia_to_uiautomator

_TAP_LIKE = {"tap", "double_tap", "long_press"}


class WindowsDevice:
    def __init__(self, driver: Any) -> None:
        # ``driver`` is an Appium/Selenium WebDriver session (or a compatible
        # fake in tests) exposing get_screenshot_as_png / page_source /
        # get_window_rect / execute_script.
        self.driver = driver

    @classmethod
    def connect(
        cls, app: str, server_url: str = "http://127.0.0.1:4723"
    ) -> "WindowsDevice":
        """Open a WinAppDriver session for ``app`` (an .exe path or AppUserModelId).

        Requires the ``[windows]`` extra and a running Appium server with the
        WinAppDriver dependency installed (``appium driver install --source=npm
        appium-windows-driver`` then its ``install-wad`` script).
        """
        try:
            from appium import webdriver  # noqa: PLC0415
            from appium.options.windows import WindowsOptions  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - import guard
            raise SystemExit(
                "The Windows backend needs the '[windows]' extra:\n"
                "  pip install -e '.[windows]'"
            ) from exc
        options = WindowsOptions()
        options.app = app
        return cls(webdriver.Remote(server_url, options=options))

    # -- observation -------------------------------------------------------
    def ensure_ready(self) -> None:
        if self.driver is None:
            raise RuntimeError("WindowsDevice has no WinAppDriver session")

    def wake_if_needed(self) -> None:
        return  # desktop is always interactive

    def screen_size(self) -> tuple[int, int]:
        rect = self.driver.get_window_rect()
        return int(rect["width"]), int(rect["height"])

    def capture(self, destination: Path) -> bytes:
        data = self.driver.get_screenshot_as_png()
        Path(destination).write_bytes(data)
        return data

    def ui_dump(self) -> str:
        return uia_to_uiautomator(self.driver.page_source)

    # -- action ------------------------------------------------------------
    def execute(self, action: Action, size: tuple[int, int]) -> None:
        width, height = size

        def point(nx: float | None, ny: float | None) -> tuple[int, int]:
            if nx is None or ny is None:
                raise ValueError("action is missing coordinates")
            return int(round(nx * (width - 1))), int(round(ny * (height - 1)))

        kind = action.type
        # NOTE (Phase 1): coordinates here are window-relative; DPI scaling and
        # window origin offset must be applied on a real high-DPI/multi-monitor
        # setup. Validated on-device before trusting these taps.
        if kind in _TAP_LIKE:
            x, y = point(action.x, action.y)
            params: dict[str, Any] = {"x": x, "y": y}
            if kind == "double_tap":
                params["times"] = 2
            elif kind == "long_press":
                params["durationMs"] = max(action.duration_ms, 600)
            self.driver.execute_script("windows: click", params)
        elif kind == "input_text":
            x, y = point(action.x, action.y)
            self.driver.execute_script("windows: click", {"x": x, "y": y})
            self.driver.execute_script("windows: keys", {"actions": [{"text": action.text}]})
        elif kind == "keyboard_enter":
            self.driver.execute_script("windows: keys", {"actions": [{"virtualKeyCode": 0x0D}]})
        else:
            raise NotImplementedError(
                f"WindowsDevice.execute: action {kind!r} is not supported yet "
                "(Phase 1: add swipe/gesture parity)."
            )

    def wait_until_stable(self, directory: Path, timeout: float, poll: float = 0.2) -> None:
        # Phase-0 settle: a fixed wait. Phase 1 will poll screenshots for
        # stability like AdbDevice does.
        time.sleep(max(0.0, min(timeout, 10.0)))

    # -- lifecycle / diagnostics (Phase 1-2) -------------------------------
    def clear_logcat(self) -> None:
        return  # Windows has no logcat; crash capture via Event Log is Phase 2

    def logcat_dump(self, tail_lines: int = 4000) -> str:
        return ""  # Phase 2: read Windows Event Log (Application) / WER dumps

    def relaunch(self, package: str) -> None:  # noqa: ARG002
        # Phase 2: close_app + launch_app, or taskkill + re-create session.
        raise NotImplementedError("WindowsDevice.relaunch is Phase 2 (clean-start)")
