from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .adb import AdbDevice
from .config import Config
from .model import OllamaVisionModel
from .perception import extract_screen_titles, hash_distance, perceptual_hash
from .policy import PolicyViolation, validate_action
from .report import write_report
from .types import RunResult, StepRecord, SubgoalRecord


def action_signature(action: object) -> tuple[object, ...]:
    action_type = getattr(action, "type", None)
    target = " ".join(
        re.findall(r"[a-z0-9]+", str(getattr(action, "target", "")).lower())
    )
    if action_type in {"tap", "input_text"} and target:
        signature: tuple[object, ...] = (action_type, target)
        if action_type == "input_text":
            signature += (getattr(action, "text", ""),)
        return signature
    if action_type in {"tap", "input_text", "swipe"}:
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


def blocked_actions_for_state(
    memory: list[tuple[int, set[tuple[object, ...]]]],
    screen_hash: int,
    maximum_distance: int = 4,
) -> set[tuple[object, ...]]:
    blocked: set[tuple[object, ...]] = set()
    for known_hash, signatures in memory:
        if hash_distance(screen_hash, known_hash) <= maximum_distance:
            blocked.update(signatures)
    return blocked


def remember_failed_action(
    memory: list[tuple[int, set[tuple[object, ...]]]],
    screen_hash: int,
    signature: tuple[object, ...],
    maximum_distance: int = 4,
) -> None:
    for known_hash, signatures in memory:
        if hash_distance(screen_hash, known_hash) <= maximum_distance:
            signatures.add(signature)
            return
    memory.append((screen_hash, {signature}))


def title_satisfies_navigation_goal(goal: str, titles: list[str]) -> str | None:
    """Return the exact visible title that proves a navigation milestone."""
    normalize = lambda value: " ".join(re.findall(r"[a-z0-9]+", value.lower()))
    destination = normalize(goal)
    for prefix in (
        "open the ",
        "open ",
        "show the ",
        "show ",
        "go to the ",
        "go to ",
        "navigate to the ",
        "navigate to ",
        "launch the ",
        "launch ",
    ):
        if destination.startswith(prefix):
            destination = destination[len(prefix) :]
            break
    for suffix in (
        " application",
        " app",
        " screen",
        " page",
        " menu",
    ):
        if destination.endswith(suffix):
            destination = destination[: -len(suffix)]
            break
    for title in titles:
        normalized_title = normalize(title)
        if destination and normalized_title == destination:
            return title
        goal_has_navigation_state = bool(
            re.search(
                r"\b(?:display|launch|navigate|open|reach|show)\b",
                normalize(goal),
            )
        )
        title_is_named_in_goal = bool(
            re.search(rf"(?:^| ){re.escape(normalized_title)}(?: |$)", normalize(goal))
        )
        if normalized_title and goal_has_navigation_state and title_is_named_in_goal:
            return title
    return None


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
        failed_action_memory: list[tuple[int, set[tuple[object, ...]]]] = []
        current_subgoal_index = 0

        try:
            self.device.ensure_ready()
            self.device.wake_if_needed()
            size = self.device.screen_size()
            self.progress("Planning observable subgoals")
            plan = self.model.create_plan(goal)
            result.subgoals = [
                SubgoalRecord(number=index + 1, description=description)
                for index, description in enumerate(plan)
            ]
            result.subgoals[0].status = "running"
            self.progress(
                "Plan: " + " -> ".join(item.description for item in result.subgoals)
            )
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
                screen_hash = perceptual_hash(image)
                blocked_signatures = blocked_actions_for_state(
                    failed_action_memory, screen_hash
                )
                current_subgoal = result.subgoals[current_subgoal_index]
                matched_title = title_satisfies_navigation_goal(
                    current_subgoal.description,
                    extract_screen_titles(ui_dump),
                )
                if (
                    matched_title is not None
                    and current_subgoal_index < len(result.subgoals) - 1
                ):
                    current_subgoal.status = "passed"
                    current_subgoal.evidence = (
                        f"Visible screen title exactly matches: {matched_title}"
                    )
                    current_subgoal_index += 1
                    result.subgoals[current_subgoal_index].status = "running"
                    history.append(
                        {
                            "completed_subgoal": current_subgoal.description,
                            "evidence": current_subgoal.evidence,
                            "next_subgoal": result.subgoals[
                                current_subgoal_index
                            ].description,
                        }
                    )
                    self.progress(
                        f"Subgoal {current_subgoal.number} observed; advancing"
                    )
                    current_subgoal = result.subgoals[current_subgoal_index]
                if current_subgoal_index == len(result.subgoals) - 1:
                    verification = self.model.verify(goal, image, ui_dump)
                    outcome = str(verification.get("outcome", "inconclusive"))
                    confidence = float(verification.get("confidence", 0.0))
                    evidence = str(
                        verification.get("evidence", "No verification evidence")
                    )
                    if (
                        outcome == "pass"
                        and confidence >= self.config.minimum_success_confidence
                    ):
                        current_subgoal.status = "passed"
                        current_subgoal.evidence = evidence
                        result.outcome = "pass"
                        result.reason = evidence
                        self.progress("Final goal independently verified")
                        break
                self.progress(
                    f"Step {number}: observing and pursuing subgoal "
                    f"{current_subgoal.number}/{len(result.subgoals)}"
                )
                decision_started = time.monotonic()
                action = self.model.plan(
                    goal,
                    image,
                    ui_dump,
                    history,
                    current_subgoal=current_subgoal.description,
                    blocked_actions=[repr(item) for item in sorted(blocked_signatures, key=repr)],
                )
                decision_seconds = time.monotonic() - decision_started
                validate_action(action, self.config, goal)
                signature = action_signature(action)
                if signature in blocked_signatures and action.type != "wait":
                    history.append(
                        {
                            "blocked_repetition": repr(signature),
                            "current_subgoal": current_subgoal.description,
                            "instruction": (
                                "This action already made no visible progress on this screen. "
                                "Choose a different strategy."
                            ),
                        }
                    )
                    self.progress(
                        f"Step {number}: changing strategy after a failed action"
                    )
                    decision_started = time.monotonic()
                    replacement = self.model.plan(
                        goal,
                        image,
                        ui_dump,
                        history,
                        current_subgoal=current_subgoal.description,
                        blocked_actions=[
                            repr(item) for item in sorted(blocked_signatures, key=repr)
                        ],
                    )
                    decision_seconds += time.monotonic() - decision_started
                    validate_action(replacement, self.config, goal)
                    if action_signature(replacement) in blocked_signatures:
                        raise PolicyViolation(
                            "Repeated-action loop detected: the replacement already failed "
                            "on this screen"
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
                    verification_goal = (
                        goal
                        if current_subgoal_index == len(result.subgoals) - 1
                        else current_subgoal.description
                    )
                    verification = self.model.verify(verification_goal, image, ui_dump)
                    outcome = str(verification.get("outcome", "inconclusive"))
                    confidence = float(verification.get("confidence", 0.0))
                    evidence = str(verification.get("evidence", "No verification evidence"))
                    if outcome == "pass" and confidence >= self.config.minimum_success_confidence:
                        current_subgoal.status = "passed"
                        current_subgoal.evidence = evidence
                        if current_subgoal_index == len(result.subgoals) - 1:
                            result.outcome = "pass"
                            result.reason = evidence
                            break
                        current_subgoal_index += 1
                        result.subgoals[current_subgoal_index].status = "running"
                        history.append(
                            {
                                "completed_subgoal": current_subgoal.description,
                                "evidence": evidence,
                                "next_subgoal": result.subgoals[
                                    current_subgoal_index
                                ].description,
                            }
                        )
                        self.progress(
                            f"Subgoal {current_subgoal.number} verified; advancing"
                        )
                        continue
                    elif outcome == "fail" and confidence >= self.config.minimum_success_confidence:
                        current_subgoal.status = "failed"
                        current_subgoal.evidence = evidence
                        result.outcome = "fail"
                        result.reason = evidence
                        break
                    else:
                        remember_failed_action(
                            failed_action_memory, screen_hash, signature
                        )
                        history.append(
                            {
                                "completion_rejected": current_subgoal.description,
                                "evidence": evidence,
                                "instruction": "Continue; the subgoal is not yet proven.",
                            }
                        )
                        continue

                if dry_run:
                    result.reason = "Dry run: proposed action was not executed"
                    break

                before = screen_hash
                self.device.execute(action, size)
                self.device.wait_until_stable(directory, self.config.settle_timeout_seconds)
                check_path = directory / f"step-{number:02d}-after.png"
                after_image = self.device.capture(check_path)
                after = perceptual_hash(after_image)
                step.screen_changed = hash_distance(before, after) > 4
                history.append(
                    {
                        "step": number,
                        "current_subgoal": current_subgoal.description,
                        "action": action.type,
                        "target": action.target,
                        "text": action.text if action.type == "input_text" else "",
                        "reason": action.reason,
                        "screen_changed": step.screen_changed,
                    }
                )
                if not step.screen_changed:
                    remember_failed_action(
                        failed_action_memory, screen_hash, signature
                    )
                    history.append(
                        {
                            "warning": "Action made no visible progress and is blocked on this screen.",
                            "blocked_action": repr(signature),
                        }
                    )
        except (PolicyViolation, Exception) as exc:
            result.outcome = "inconclusive"
            result.reason = f"Stopped safely: {exc}"
        finally:
            result.finished_at = datetime.now(timezone.utc).isoformat()
            write_report(result)
        return result
