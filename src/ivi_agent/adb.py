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

    def screen_size(self) -> tuple[int, int]:
        output = str(self._run("shell", "wm", "size"))
        matches = re.findall(r"(\d+)x(\d+)", output)
        if not matches:
            raise AdbError(f"Could not determine screen size from: {output.strip()}")
        width, height = matches[-1]
        return int(width), int(height)

    def capture(self, destination: Path) -> bytes:
        args = ["exec-out", "screencap", "-p"]
        if self.display_id is not None:
            args += ["-d", str(self.display_id)]
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

    def execute(self, action: Action, size: tuple[int, int]) -> None:
        width, height = size

        def point(x: float | None, y: float | None) -> tuple[str, str]:
            if x is None or y is None:
                raise AdbError("Action is missing coordinates")
            return str(round(x * (width - 1))), str(round(y * (height - 1)))

        if action.type == "tap":
            x, y = point(action.x, action.y)
            self._run("shell", "input", "tap", x, y)
        elif action.type == "swipe":
            x1, y1 = point(action.x, action.y)
            x2, y2 = point(action.x2, action.y2)
            self._run("shell", "input", "swipe", x1, y1, x2, y2, str(action.duration_ms))
        elif action.type == "gesture":
            starts = {
                "up": (0.5, 0.75, 0.5, 0.25),
                "down": (0.5, 0.25, 0.5, 0.75),
                "left": (0.75, 0.5, 0.25, 0.5),
                "right": (0.25, 0.5, 0.75, 0.5),
            }
            x1n, y1n, x2n, y2n = starts[action.direction]
            if action.direction in {"up", "down"} and action.region in {"left", "right"}:
                x1n = x2n = 0.25 if action.region == "left" else 0.75
            if action.direction in {"left", "right"} and action.region in {"top", "bottom"}:
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
            safe_text = action.text.replace("%", "%25").replace(" ", "%s")
            self._run("shell", "input", "text", safe_text)
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
