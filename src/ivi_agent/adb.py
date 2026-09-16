from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

from .perception import hash_distance, perceptual_hash
from .types import Action


class AdbError(RuntimeError):
    pass


class AdbDevice:
    def __init__(self, serial: str | None = None, display_id: int | None = None) -> None:
        self.serial = serial
        self.display_id = display_id
        self._resolved_capture_display_id: int | None = display_id

    def _base(self) -> list[str]:
        command = ["adb"]
        if self.serial:
            command += ["-s", self.serial]
        return command

    def _run(self, *args: str, binary: bool = False, timeout: float = 20) -> bytes | str:
        try:
            result = subprocess.run(
                [*self._base(), *args],
                check=True,
                capture_output=True,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise AdbError("adb was not found. Install Android platform-tools and add adb to PATH.") from exc
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.decode(errors="replace").strip()
            raise AdbError(message or "ADB command failed") from exc
        except subprocess.TimeoutExpired as exc:
            raise AdbError("ADB command timed out") from exc
        return result.stdout if binary else result.stdout.decode(errors="replace")

    def ensure_ready(self) -> None:
        state = str(self._run("get-state")).strip()
        if state != "device":
            raise AdbError(f"Device is not ready: {state or 'unknown state'}")

    def wake_if_needed(self) -> None:
        """Wake an Android display without assuming any product UI or unlock route."""
        output = str(self._run("shell", "dumpsys", "power", timeout=10))
        awake = re.search(r"(?:mWakefulness=|Wakefulness:\s*)Awake\b", output)
        if awake:
            return
        self._run("shell", "input", "keyevent", "KEYCODE_WAKEUP", timeout=10)
        time.sleep(0.5)

    def go_home(self) -> None:
        """Return to Android Home between independent test cases."""
        self._run("shell", "input", "keyevent", "KEYCODE_HOME", timeout=10)
        time.sleep(0.5)

    def screen_size(self) -> tuple[int, int]:
        output = str(self._run("shell", "wm", "size"))
        matches = re.findall(r"(\d+)x(\d+)", output)
        if not matches:
            raise AdbError(f"Could not determine screen size from: {output.strip()}")
        width, height = matches[-1]
        return int(width), int(height)

    def capture_display_id(self) -> int | None:
        if self._resolved_capture_display_id is not None:
            return self._resolved_capture_display_id
        try:
            output = str(
                self._run("shell", "dumpsys", "SurfaceFlinger", "--display-id", timeout=10)
            )
        except AdbError:
            return None
        matches = re.findall(r"Display\s+(\d+)\s+\(HWC display\s+(\d+)\)", output)
        if not matches:
            return None
        primary = next((display for display, hwc in matches if hwc == "0"), matches[0][0])
        self._resolved_capture_display_id = int(primary)
        return self._resolved_capture_display_id

    def capture(self, destination: Path) -> bytes:
        args = ["exec-out", "screencap", "-p"]
        capture_display_id = self.capture_display_id()
        if capture_display_id is not None:
            args += ["-d", str(capture_display_id)]
        image = self._run(*args, binary=True, timeout=30)
        assert isinstance(image, bytes)
        if not image.startswith(b"\x89PNG"):
            raise AdbError("ADB returned invalid screenshot data")
        destination.write_bytes(image)
        return image

    def ui_dump(self) -> str:
        remote = "/sdcard/ivi-agent-window.xml"
        self._run("shell", "uiautomator", "dump", remote, timeout=15)
        return str(self._run("exec-out", "cat", remote, timeout=15))

    @staticmethod
    def _escape_input_text(text: str) -> str:
        """Make text safe for `adb shell input text`.

        `input text` treats %s as a space and is parsed by the on-device shell,
        so both the percent/space convention and shell metacharacters must be
        escaped. This handles far more than the old space-and-percent handling,
        which broke on ampersands, quotes, parentheses, and similar characters.
        """
        escaped = text.replace("%", "%25")
        for char in "()<>|;&*~\"'`$\\":
            escaped = escaped.replace(char, "\\" + char)
        return escaped.replace(" ", "%s")

    def type_text(self, text: str) -> None:
        self._run("shell", "input", "text", self._escape_input_text(text))

    def open_app(self, app_name: str) -> None:
        """Best-effort launch by package name for the standalone agent.

        The AndroidWorld adapter never calls this; it delegates open_app to the
        benchmark environment, which resolves human app names to packages.
        """
        package = app_name.strip()
        if not re.fullmatch(r"[A-Za-z][\w]*(?:\.[A-Za-z0-9_]+)+", package):
            raise AdbError(
                f"Cannot resolve app name to a package: {app_name!r}. "
                "Provide a package id (e.g. com.android.settings) or tap the launcher."
            )
        self._run("shell", "monkey", "-p", package, "-c", "android.intent.category.LAUNCHER", "1")
        time.sleep(1.0)

    def execute(self, action: Action, size: tuple[int, int]) -> None:
        width, height = size

        def point(x: float | None, y: float | None) -> tuple[str, str]:
            if x is None or y is None:
                raise AdbError("Action is missing coordinates")
            return str(round(x * (width - 1))), str(round(y * (height - 1)))

        if action.type == "tap":
            x, y = point(action.x, action.y)
            self._run("shell", "input", "tap", x, y)
        elif action.type == "double_tap":
            x, y = point(action.x, action.y)
            self._run("shell", "input", "tap", x, y)
            time.sleep(0.12)
            self._run("shell", "input", "tap", x, y)
        elif action.type == "long_press":
            x, y = point(action.x, action.y)
            # Android has no direct long-press; a same-point swipe with a long
            # duration is the standard equivalent.
            duration = str(max(action.duration_ms, 600))
            self._run("shell", "input", "swipe", x, y, x, y, duration)
        elif action.type == "keyboard_enter":
            self._run("shell", "input", "keyevent", "KEYCODE_ENTER")
        elif action.type == "open_app":
            self.open_app(action.app_name or action.target)
        elif action.type == "input_text":
            x, y = point(action.x, action.y)
            self._run("shell", "input", "tap", x, y)
            time.sleep(0.2)
            self.type_text(action.text)
        elif action.type == "swipe":
            x1, y1 = point(action.x, action.y)
            x2, y2 = point(action.x2, action.y2)
            self._run("shell", "input", "swipe", x1, y1, x2, y2, str(action.duration_ms))
        elif action.type == "gesture":
            starts = {
                "reveal_below": (0.5, 0.75, 0.5, 0.25),
                "reveal_above": (0.5, 0.25, 0.5, 0.75),
                "reveal_right": (0.75, 0.5, 0.25, 0.5),
                "reveal_left": (0.25, 0.5, 0.75, 0.5),
            }
            x1n, y1n, x2n, y2n = starts[action.direction]
            if action.direction in {"reveal_above", "reveal_below"} and action.region in {"left", "right"}:
                x1n = x2n = 0.25 if action.region == "left" else 0.75
            if action.direction in {"reveal_left", "reveal_right"} and action.region in {"top", "bottom"}:
                y1n = y2n = 0.25 if action.region == "top" else 0.75
            x1, y1 = point(x1n, y1n)
            x2, y2 = point(x2n, y2n)
            self._run("shell", "input", "swipe", x1, y1, x2, y2, str(action.duration_ms))
        elif action.type == "back":
            self._run("shell", "input", "keyevent", "KEYCODE_BACK")
        elif action.type == "home":
            self._run("shell", "input", "keyevent", "KEYCODE_HOME")
        elif action.type == "wait":
            time.sleep(max(0.0, min(action.seconds, 10.0)))
        elif action.type == "text":
            self.type_text(action.text)
        else:
            raise AdbError(f"Cannot execute action type: {action.type}")

    def wait_until_stable(self, directory: Path, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        previous: int | None = None
        stable_count = 0
        sample = directory / ".stability.png"
        while time.monotonic() < deadline:
            image = self.capture(sample)
            digest = perceptual_hash(image)
            if previous is not None and hash_distance(digest, previous) <= 2:
                stable_count += 1
                if stable_count >= 2:
                    sample.unlink(missing_ok=True)
                    return True
            else:
                stable_count = 0
            previous = digest
            time.sleep(0.35)
        sample.unlink(missing_ok=True)
        return False
