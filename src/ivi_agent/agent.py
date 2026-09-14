from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

from .adb import AdbDevice
from .config import Config
from .model import OllamaVisionModel
from .policy import PolicyViolation, validate_action
from .report import write_report
from .types import RunResult, StepRecord


class GoalAgent:
    def __init__(self, device: AdbDevice, model: OllamaVisionModel, config: Config) -> None:
        self.device = device
        self.model = model
        self.config = config

    def run(self, goal: str, output_root: Path, dry_run: bool = False) -> RunResult:
        started = datetime.now(timezone.utc)
        run_id = started.strftime("%Y%m%dT%H%M%SZ")
        directory = output_root / run_id
        directory.mkdir(parents=True, exist_ok=False)
        result = RunResult(
            goal=goal,
            outcome="inconclusive",
            reason="Action budget exhausted",
            run_directory=str(directory.resolve()),
            started_at=started.isoformat(),
        )
        deadline = time.monotonic() + self.config.timeout_seconds
        history: list[dict[str, object]] = []
        previous_digest: str | None = None

        try:
            self.device.ensure_ready()
            size = self.device.screen_size()
            for number in range(1, self.config.max_actions + 1):
                if time.monotonic() >= deadline:
                    result.reason = "Run timeout reached"
                    break
                screenshot_name = f"step-{number:02d}.png"
                screenshot_path = directory / screenshot_name
                image = self.device.capture(screenshot_path)
                try:
                    ui_dump = self.device.ui_dump()
                except Exception:
                    ui_dump = ""
                action = self.model.plan(goal, image, ui_dump, history)
                validate_action(action, self.config)
                step = StepRecord(
                    number=number,
                    screenshot=screenshot_name,
                    action=action,
                    ui_dump_available=bool(ui_dump),
                )
                result.steps.append(step)

                if action.type == "finish":
                    verification = self.model.verify(goal, image)
                    outcome = str(verification.get("outcome", "inconclusive"))
                    confidence = float(verification.get("confidence", 0.0))
                    evidence = str(verification.get("evidence", "No verification evidence"))
                    if outcome == "pass" and confidence >= self.config.minimum_success_confidence:
                        result.outcome = "pass"
                        result.reason = evidence
                    elif outcome == "fail" and confidence >= self.config.minimum_success_confidence:
                        result.outcome = "fail"
                        result.reason = evidence
                    else:
                        result.outcome = "inconclusive"
                        result.reason = f"Independent verification was uncertain: {evidence}"
                    break

                if dry_run:
                    result.reason = "Dry run: proposed action was not executed"
                    break

                before = hashlib.sha256(image).hexdigest()
                self.device.execute(action, size)
                self.device.wait_until_stable(directory, self.config.settle_timeout_seconds)
                check_path = directory / f"step-{number:02d}-after.png"
                after_image = self.device.capture(check_path)
                after = hashlib.sha256(after_image).hexdigest()
                step.screen_changed = before != after
                history.append(
                    {
                        "step": number,
                        "action": action.type,
                        "target": action.target,
                        "reason": action.reason,
                        "screen_changed": step.screen_changed,
                    }
                )
                if not step.screen_changed and previous_digest == after:
                    history.append({"warning": "The last two actions did not change the screen."})
                previous_digest = after
        except (PolicyViolation, Exception) as exc:
            result.outcome = "inconclusive"
            result.reason = f"Stopped safely: {exc}"
        finally:
            result.finished_at = datetime.now(timezone.utc).isoformat()
            write_report(result)
        return result

