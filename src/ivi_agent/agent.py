from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .adb import AdbDevice
from .config import Config
from .model import OllamaVisionModel
from .perception import hash_distance, perceptual_hash
from .policy import PolicyViolation, validate_action
from .report import write_report
from .types import RunResult, StepRecord


def action_signature(action: object) -> tuple[object, ...]:
    action_type = getattr(action, "type", None)
    if action_type in {"tap", "swipe"}:
        return (
            action_type,
            round(getattr(action, "x", 0.0) or 0.0, 2),
            round(getattr(action, "y", 0.0) or 0.0, 2),
            round(getattr(action, "x2", 0.0) or 0.0, 2),
            round(getattr(action, "y2", 0.0) or 0.0, 2),
        )
    if action_type == "gesture":
        return (
            action_type,
            getattr(action, "direction", ""),
            getattr(action, "region", "center"),
        )
    return (action_type,)


class GoalAgent:
    def __init__(
        self,
        device: AdbDevice,
        model: OllamaVisionModel,
        config: Config,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.device = device
        self.model = model
        self.config = config
        self.progress = progress or (lambda _message: None)

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
        previous_action_signature: tuple[object, ...] | None = None

        try:
            self.device.ensure_ready()
            size = self.device.screen_size()
            for number in range(1, self.config.max_actions + 1):
                if time.monotonic() >= deadline:
                    result.reason = "Run timeout reached"
                    break
                screenshot_name = f"step-{number:02d}.png"
                screenshot_path = directory / screenshot_name
                self.progress(f"Step {number}: capturing device state")
                image = self.device.capture(screenshot_path)
                try:
                    ui_dump = self.device.ui_dump()
                except Exception:
                    ui_dump = ""
                self.progress(f"Step {number}: extracting grounded controls and planning")
                decision_started = time.monotonic()
                action = self.model.plan(goal, image, ui_dump, history)
                decision_seconds = time.monotonic() - decision_started
                validate_action(action, self.config)
                signature = action_signature(action)
                if signature == previous_action_signature and action.type != "wait":
                    history.append(
                        {
                            "blocked_repetition": action.type,
                            "instruction": (
                                f"Do not repeat {action.type}; it was just attempted. "
                                "Choose a different action that advances the goal."
                            ),
                        }
                    )
                    self.progress(f"Step {number}: replanning to prevent a repeated action")
                    decision_started = time.monotonic()
                    replacement = self.model.plan(goal, image, ui_dump, history)
                    decision_seconds += time.monotonic() - decision_started
                    validate_action(replacement, self.config)
                    if action_signature(replacement) == signature:
                        raise PolicyViolation(
                            f"Repeated-action loop detected: model chose {action.type} again"
                        )
                    action = replacement
                    signature = action_signature(action)
                step = StepRecord(
                    number=number,
                    screenshot=screenshot_name,
                    action=action,
                    ui_dump_available=bool(ui_dump),
                    decision_seconds=decision_seconds,
                )
                result.steps.append(step)
                self.progress(
                    f"Step {number}: {action.type} {action.target or action.direction} "
                    f"({decision_seconds:.1f}s, confidence {action.confidence:.2f})"
                )

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

                before = perceptual_hash(image)
                self.device.execute(action, size)
                previous_action_signature = signature
                self.device.wait_until_stable(directory, self.config.settle_timeout_seconds)
                check_path = directory / f"step-{number:02d}-after.png"
                after_image = self.device.capture(check_path)
                after = perceptual_hash(after_image)
                step.screen_changed = hash_distance(before, after) > 4
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
