from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .adb import AdbDevice
from .config import Config
from .model import OllamaVisionModel
from .knowledge import KnowledgeBase, prompt_context, reference_images
from .perception import (
    extract_ocr_screen_titles,
    extract_screen_titles,
    extract_ui_elements,
    extract_visible_text,
    hash_distance,
    perceptual_hash,
)
from .graph import SceneGraph
from .logs import crash_summary, scan_crashes
from .policy import PolicyViolation, validate_action
from .report import write_report
from .trace import NullTrace, RunTrace
from .types import Action, RunResult, StepRecord, SubgoalRecord
from .vision_match import cv_ground_from_knowledge


def screen_made_progress(
    before_hash: int,
    after_hash: int,
    before_ui: str,
    after_ui: str,
    maximum_distance: int = 4,
) -> bool:
    """True if the screen changed visibly OR in its UI text.

    The perceptual hash misses small-but-meaningful changes -- a toggle flipping,
    a list item becoming selected, a "running" status appearing -- so a change in
    the visible UI text also counts as progress. This prevents the agent from
    treating a successful state change as a no-op and looping on it.
    """
    if hash_distance(before_hash, after_hash) > maximum_distance:
        return True
    return set(extract_visible_text(before_ui)) != set(extract_visible_text(after_ui))


_TARGET_STOPWORDS = {
    "tile", "button", "the", "a", "an", "screen", "icon", "open", "tap", "to",
    "activate", "select", "press", "program", "card", "row", "menu", "option",
}


def target_keywords(name: str) -> set[str]:
    """Distinctive tokens of a control name for accessibility-tree matching."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(name).lower())
        if token not in _TARGET_STOPWORDS
    }


def map_subgoal_targets(
    subgoals: list, task_data: dict | None, control_names: dict[str, str]
) -> dict[int, set[str]]:
    """Map subgoal index -> target-control keywords (for the a11y fast-path).

    Aligns each subgoal to its manual step by milestone, resolves the step's
    ``target`` control id to its name, and reduces that to distinctive keywords.
    """
    targets: dict[int, set[str]] = {}
    if not isinstance(task_data, dict):
        return targets
    by_milestone: dict[str, str] = {}
    for step in task_data.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        milestone = step.get("milestone", step.get("expected", ""))
        control_id = step.get("target")
        if milestone and control_id:
            key = " ".join(re.findall(r"[a-z0-9]+", str(milestone).lower()))
            if key:
                by_milestone[key] = str(control_id)
    for index, subgoal in enumerate(subgoals):
        key = " ".join(re.findall(r"[a-z0-9]+", subgoal.description.lower()))
        control_id = by_milestone.get(key)
        if control_id:
            keywords = target_keywords(control_names.get(control_id, control_id))
            if keywords:
                targets[index] = keywords
    return targets


def resolve_target_in_tree(ui_dump: str, keywords: set[str]) -> tuple[float, float] | None:
    """Return the normalized center of the clickable element matching keywords.

    Matches only when a single actionable element's label contains all target
    keywords (preferring an exact token match on ties); returns None when absent
    or ambiguous, so the caller falls back to CV / the model.
    """
    if not keywords or not ui_dump:
        return None
    elements = extract_ui_elements(ui_dump)
    matches = [
        element
        for element in elements
        if keywords <= set(re.findall(r"[a-z0-9]+", element.label.lower()))
    ]
    if len(matches) == 1:
        return matches[0].center
    exact = [
        element
        for element in matches
        if set(re.findall(r"[a-z0-9]+", element.label.lower())) == keywords
    ]
    if len(exact) == 1:
        return exact[0].center
    return None


def map_subgoal_success_cues(
    subgoals: list, task_data: dict | None
) -> dict[int, str]:
    """Map subgoal index -> a `success_text` cue from the matching manual step.

    A manual task step may declare ``success_text``: a substring that, once
    visible on screen, means that step is complete. Subgoals are matched to steps
    by normalized milestone text (the planner derives subgoals from the same
    milestones). This lets an in-screen state change (e.g. a program becoming
    selected) advance a subgoal, not just a screen-title change.
    """
    cues: dict[int, str] = {}
    if not isinstance(task_data, dict):
        return cues
    by_milestone: dict[str, str] = {}
    for step in task_data.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        cue = step.get("success_text")
        milestone = step.get("milestone", step.get("expected", ""))
        if cue and milestone:
            key = " ".join(re.findall(r"[a-z0-9]+", str(milestone).lower()))
            if key:
                by_milestone[key] = str(cue)
    for index, subgoal in enumerate(subgoals):
        key = " ".join(re.findall(r"[a-z0-9]+", subgoal.description.lower()))
        if key in by_milestone:
            cues[index] = by_milestone[key]
    return cues


def subgoal_cue_satisfied(cue: str, visible_text: list[str]) -> bool:
    """True if the step's success cue is present in the visible screen text."""
    if not cue:
        return False
    return cue.lower() in " ".join(visible_text).lower()


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
        "reach the ",
        "reach ",
    ):
        if destination.startswith(prefix):
            destination = destination[len(prefix) :]
            break
    for suffix in (
        " application main screen",
        " app main screen",
        " main screen",
        " application",
        " app",
        " screen",
        " page",
        " menu",
        " panel",
        " overlay",
        " dialog",
    ):
        if destination.endswith(suffix):
            destination = destination[: -len(suffix)]
            break
    destination_terms = destination.split()
    if len(destination_terms) > 1 and destination_terms[-1] == "settings":
        destination = " ".join(destination_terms[:-1])
    for title in titles:
        normalized_title = normalize(title)
        if destination and normalized_title == destination:
            return title
    return None


class GoalAgent:
    def __init__(
        self,
        device: AdbDevice,
        model: OllamaVisionModel,
        config: Config,
        knowledge: KnowledgeBase | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.device = device
        self.model = model
        self.config = config
        self.knowledge = knowledge
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
        trace: RunTrace = (
            RunTrace(directory, run_id) if getattr(self.config, "trace", True) else NullTrace()
        )
        trace.write_json(
            "task.json",
            {
                "run_id": run_id,
                "goal": goal,
                "dry_run": dry_run,
                "started_at": started.isoformat(),
                "config": {
                    "model": self.config.model,
                    "grounding_mode": self.config.grounding_mode,
                    "knowledge_profile": self.config.knowledge_profile,
                    "cv_fast_path": self.config.cv_fast_path,
                    "max_actions": self.config.max_actions,
                    "timeout_seconds": self.config.timeout_seconds,
                },
            },
        )
        trace.event("run_start", goal=goal, run_id=run_id, dry_run=dry_run)

        # Living scene graph: seed the expected model from the manual (once),
        # then confirm/grow it as screens are reached. Persisted per profile.
        scene_graph: SceneGraph | None = None
        scene_graph_path: Path | None = None
        if getattr(self.config, "scene_graph", True) and self.knowledge is not None:
            try:
                manual = {
                    "screens": [
                        chunk["data"]
                        for chunk in self.knowledge.chunks
                        if chunk.get("kind") == "screen"
                        and isinstance(chunk.get("data"), dict)
                    ]
                }
                scene_graph_path = self.knowledge.directory / "scene_graph.json"
                scene_graph = SceneGraph.load_or_seed(scene_graph_path, manual)
                trace.event(
                    "graph_seeded",
                    screens=len(scene_graph.nodes),
                    edges=len(scene_graph.edges),
                    persisted=scene_graph_path.is_file(),
                )
            except Exception as exc:  # noqa: BLE001 - graph is advisory, never fatal
                self.progress(f"scene graph unavailable: {exc}")
                scene_graph = None
        graph_prev_node: str | None = None
        graph_last_action: tuple[str, str] = ("", "")  # (action_type, target)

        deadline = time.monotonic() + self.config.timeout_seconds
        history: list[dict[str, object]] = []
        failed_action_memory: list[tuple[int, set[tuple[object, ...]]]] = []
        current_subgoal_index = 0
        # Carry the post-action observation forward as the next step's start
        # observation, so we don't re-capture + re-`uiautomator dump` the same
        # screen twice per step. (image_bytes, ui_dump, perceptual_hash).
        carried_observation: tuple[bytes, str, int] | None = None

        try:
            self.device.ensure_ready()
            self.device.wake_if_needed()
            if getattr(self.config, "capture_logs", True):
                self.device.clear_logcat()
            size = self.device.screen_size()
            initial_knowledge = (
                self.knowledge.query(goal, self.config.knowledge_top_k)
                if self.knowledge
                else {}
            )
            initial_context = prompt_context(initial_knowledge) if initial_knowledge else {}
            if initial_knowledge:
                result.knowledge = {
                    "profile": initial_knowledge.get("profile"),
                    "manual_id": initial_knowledge.get("manual_id"),
                    "retrieved_chunk_ids": [
                        chunk.get("id") for chunk in initial_knowledge.get("chunks", [])
                    ],
                }
                self.progress(
                    "Retrieved local manual context: "
                    + ", ".join(
                        str(item) for item in result.knowledge["retrieved_chunk_ids"]
                    )
                )
                trace.event(
                    "retrieval",
                    scope="initial",
                    profile=result.knowledge.get("profile"),
                    chunk_ids=result.knowledge["retrieved_chunk_ids"],
                )
            self.progress("Planning observable subgoals")
            plan = self.model.create_plan(goal, initial_context)
            result.subgoals = [
                SubgoalRecord(number=index + 1, description=description)
                for index, description in enumerate(plan)
            ]
            result.subgoals[0].status = "running"
            self.progress(
                "Plan: " + " -> ".join(item.description for item in result.subgoals)
            )
            trace.write_json(
                "plan.json",
                {"goal": goal, "subgoals": [item.description for item in result.subgoals]},
            )
            trace.event(
                "plan", subgoals=[item.description for item in result.subgoals]
            )
            # Per-subgoal completion cues from the manual task steps, so an
            # in-screen state change (e.g. a program becoming selected) can
            # advance a subgoal instead of the model re-tapping it.
            task_chunk = next(
                (
                    chunk
                    for chunk in (initial_knowledge.get("chunks", []) if initial_knowledge else [])
                    if chunk.get("kind") == "task"
                ),
                None,
            )
            subgoal_success_cues = map_subgoal_success_cues(
                result.subgoals, task_chunk.get("data") if task_chunk else None
            )
            # Control id -> name across all known screens, for the a11y fast-path.
            control_names: dict[str, str] = {}
            if self.knowledge is not None:
                for chunk in self.knowledge.chunks:
                    if chunk.get("kind") == "screen" and isinstance(chunk.get("data"), dict):
                        for control in chunk["data"].get("controls", []) or []:
                            if isinstance(control, dict) and control.get("id"):
                                control_names[str(control["id"])] = str(control.get("name", ""))
            subgoal_targets = map_subgoal_targets(
                result.subgoals,
                task_chunk.get("data") if task_chunk else None,
                control_names,
            )
            for number in range(1, self.config.max_actions + 1):
                if time.monotonic() >= deadline:
                    result.reason = "Run timeout reached"
                    break
                screenshot_name = f"step-{number:02d}.png"
                screenshot_path = directory / screenshot_name
                if carried_observation is not None and getattr(
                    self.config, "reuse_after_state", True
                ):
                    # Reuse the previous step's post-action screen instead of a
                    # fresh capture + uiautomator dump (same screen, no round-trip).
                    image, ui_dump, screen_hash = carried_observation
                    try:
                        screenshot_path.write_bytes(image)
                    except Exception:  # noqa: BLE001 - screenshot is evidence only
                        pass
                else:
                    self.progress(f"Step {number}: capturing device state")
                    image = self.device.capture(screenshot_path)
                    try:
                        ui_dump = self.device.ui_dump()
                    except Exception:
                        ui_dump = ""
                    screen_hash = perceptual_hash(image)
                carried_observation = None
                blocked_signatures = blocked_actions_for_state(
                    failed_action_memory, screen_hash
                )
                current_subgoal = result.subgoals[current_subgoal_index]
                observed_titles = extract_screen_titles(ui_dump)
                if not observed_titles and self.model.enable_ocr:
                    observed_titles = extract_ocr_screen_titles(image)
                visible_text = extract_visible_text(ui_dump)
                retrieval_query = " ".join(
                    [goal, current_subgoal.description, *observed_titles, *visible_text[:12]]
                )
                step_knowledge = (
                    self.knowledge.query(retrieval_query, self.config.knowledge_top_k)
                    if self.knowledge
                    else {}
                )
                step_context = (
                    prompt_context(step_knowledge, current_subgoal.description)
                    if step_knowledge
                    else {}
                )
                step_references = reference_images(step_knowledge) if step_knowledge else []
                trace.event(
                    "step_begin",
                    step=number,
                    subgoal=current_subgoal.description,
                    subgoal_index=current_subgoal_index,
                    titles=observed_titles,
                    retrieved_chunk_ids=[
                        chunk.get("id") for chunk in step_knowledge.get("chunks", [])
                    ]
                    if step_knowledge
                    else [],
                )

                # Grow the living scene graph with the screen we just reached,
                # attributing the transition to the previous step's action.
                if scene_graph is not None:
                    observation = scene_graph.observe(
                        observed_titles,
                        phash=screen_hash,
                        screenshot=screenshot_name,
                        run_id=run_id,
                        came_from=graph_prev_node,
                        via_action=graph_last_action[0],
                        via_target=graph_last_action[1],
                    )
                    graph_prev_node = observation.node_id
                    trace.event(
                        "graph_observe",
                        step=number,
                        node=observation.node_id,
                        matched=observation.matched,
                        new_node=observation.is_new_node,
                        titles=observed_titles,
                    )
                    if observation.finding is not None:
                        self.progress(
                            f"⚠ HMI divergence ({observation.finding.kind}): "
                            f"{observation.finding.detail} — flagged for review "
                            f"[{observation.finding.id}]"
                        )
                        trace.event(
                            "graph_divergence",
                            step=number,
                            finding=observation.finding.id,
                            finding_kind=observation.finding.kind,
                            detail=observation.finding.detail,
                        )

                final_title = title_satisfies_navigation_goal(goal, observed_titles)
                if final_title is not None:
                    for prerequisite in result.subgoals[:-1]:
                        if prerequisite.status != "passed":
                            prerequisite.status = "passed"
                            prerequisite.evidence = (
                                "Final destination was already visible; this navigation "
                                "prerequisite did not need a separate stop"
                            )
                    result.subgoals[-1].status = "passed"
                    result.subgoals[-1].evidence = (
                        f"Visible screen title exactly matches: {final_title}"
                    )
                    result.outcome = "pass"
                    result.reason = result.subgoals[-1].evidence
                    self.progress("Final goal visibly reached")
                    break
                matched_title = title_satisfies_navigation_goal(
                    current_subgoal.description,
                    observed_titles,
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
                if (
                    getattr(self.config, "verify_each_step", True)
                    and current_subgoal_index == len(result.subgoals) - 1
                    and number > 1
                ):
                    verification = self.model.verify(
                        goal,
                        image,
                        ui_dump,
                        knowledge_context=step_context,
                        reference_images=step_references,
                    )
                    outcome = str(verification.get("outcome", "inconclusive"))
                    confidence = float(verification.get("confidence", 0.0))
                    evidence = str(
                        verification.get("evidence", "No verification evidence")
                    )
                    trace.event(
                        "verify",
                        scope="final_subgoal",
                        step=number,
                        outcome=outcome,
                        confidence=round(confidence, 3),
                        evidence=evidence,
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
                # Fast path: if the retriever matched a manual icon for this
                # subgoal and OpenCV can locate that icon on screen, tap it
                # directly and skip the slow model call. Falls through to the
                # model whenever the match is weak, blocked, or policy-rejected.
                action = None
                grounded_by = "model"
                # Fastest path: the current subgoal's target control is a single
                # clickable element in the live tree -> tap it directly.
                if getattr(self.config, "accessibility_fast_path", True) and ui_dump:
                    keywords = subgoal_targets.get(current_subgoal_index)
                    hit = resolve_target_in_tree(ui_dump, keywords) if keywords else None
                    if hit is not None:
                        cand = Action(
                            type="tap",
                            confidence=1.0,
                            reason=f"accessibility tree match {sorted(keywords)}",
                            target=" ".join(sorted(keywords)),
                            x=hit[0],
                            y=hit[1],
                        )
                        if action_signature(cand) not in blocked_signatures:
                            try:
                                validate_action(cand, self.config, goal)
                                action = cand
                                grounded_by = "a11y"
                                self.progress(
                                    f"Step {number}: accessibility fast-path tapped "
                                    f"{cand.target!r}; skipping model"
                                )
                            except PolicyViolation:
                                action = None
                if action is None and self.config.cv_fast_path and step_knowledge:
                    candidate = cv_ground_from_knowledge(
                        image, step_knowledge, self.config
                    )
                    if candidate is not None and action_signature(candidate) not in blocked_signatures:
                        try:
                            validate_action(candidate, self.config, goal)
                            action = candidate
                            grounded_by = "cv"
                            self.progress(
                                f"Step {number}: cv2 fast-path grounded "
                                f"'{candidate.target}' (score {candidate.confidence:.2f}); "
                                "skipping model"
                            )
                        except PolicyViolation:
                            action = None
                if action is None:
                    action = self.model.plan(
                        goal,
                        image,
                        ui_dump,
                        history,
                        current_subgoal=current_subgoal.description,
                        blocked_actions=[repr(item) for item in sorted(blocked_signatures, key=repr)],
                        knowledge_context=step_context,
                        reference_images=step_references,
                    )
                    validate_action(action, self.config, goal)
                decision_seconds = time.monotonic() - decision_started
                signature = action_signature(action)
                if signature in blocked_signatures and action.type != "wait":
                    # Cheap check first: if the final subgoal's success cue is
                    # already on screen (e.g. the goal was reached, or the run
                    # started mid-flow with "Massage running" already showing),
                    # finish without paying for a full model verify.
                    final_cue = subgoal_success_cues.get(len(result.subgoals) - 1)
                    if final_cue and subgoal_cue_satisfied(
                        final_cue, extract_visible_text(ui_dump)
                    ):
                        for prior in result.subgoals:
                            if prior.status != "passed":
                                prior.status = "passed"
                                prior.evidence = f"Observed on screen: {final_cue!r}"
                        result.outcome = "pass"
                        result.reason = f"Observed on screen: {final_cue!r}"
                        trace.event(
                            "verify", scope="success_cue", step=number,
                            cue=final_cue, outcome="pass",
                        )
                        self.progress(f"Goal already satisfied (cue {final_cue!r})")
                        break
                    # Otherwise fall back to a full model verify: fine-grained
                    # subgoals (e.g. "select a program") may lack a clean signal, so
                    # the agent can finish the goal while the tracker lags and loop.
                    overall = self.model.verify(
                        goal,
                        image,
                        ui_dump,
                        knowledge_context=step_context,
                        reference_images=step_references,
                    )
                    if (
                        str(overall.get("outcome")) == "pass"
                        and float(overall.get("confidence", 0.0))
                        >= self.config.minimum_success_confidence
                    ):
                        evidence = str(overall.get("evidence", "Goal already satisfied"))
                        for prior in result.subgoals:
                            if prior.status != "passed":
                                prior.status = "passed"
                                prior.evidence = evidence
                        result.outcome = "pass"
                        result.reason = evidence
                        self.progress("Overall goal already satisfied; finishing")
                        break
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
                        knowledge_context=step_context,
                        reference_images=step_references,
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
                    grounded_by=grounded_by,
                )
                result.steps.append(step)
                self.progress(
                    f"Step {number}: {action.type} {action.target or action.direction} "
                    f"({decision_seconds:.1f}s, confidence {action.confidence:.2f})"
                )
                trace.event(
                    "decision",
                    step=number,
                    grounded_by=grounded_by,
                    action=action.type,
                    target=action.target,
                    confidence=round(action.confidence, 3),
                    decision_seconds=round(decision_seconds, 3),
                    reason=action.reason,
                )

                if action.type == "finish":
                    verification_goal = (
                        goal
                        if current_subgoal_index == len(result.subgoals) - 1
                        else current_subgoal.description
                    )
                    verification = self.model.verify(
                        verification_goal,
                        image,
                        ui_dump,
                        knowledge_context=step_context,
                        reference_images=step_references,
                    )
                    outcome = str(verification.get("outcome", "inconclusive"))
                    confidence = float(verification.get("confidence", 0.0))
                    evidence = str(verification.get("evidence", "No verification evidence"))
                    trace.event(
                        "verify",
                        scope="finish_action",
                        step=number,
                        outcome=outcome,
                        confidence=round(confidence, 3),
                        evidence=evidence,
                    )
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
                self.device.wait_until_stable(
                    directory,
                    self.config.settle_timeout_seconds,
                    getattr(self.config, "settle_poll_seconds", 0.2),
                )
                check_path = directory / f"step-{number:02d}-after.png"
                after_image = self.device.capture(check_path)
                after = perceptual_hash(after_image)
                try:
                    after_ui = self.device.ui_dump()
                except Exception:
                    after_ui = ""
                step.screen_changed = screen_made_progress(before, after, ui_dump, after_ui)
                # This post-action screen is the next step's start observation;
                # carry it forward to skip a duplicate capture + uiautomator dump.
                carried_observation = (after_image, after_ui, after)
                trace.event(
                    "execute",
                    step=number,
                    action=action.type,
                    target=action.target,
                    screen_changed=step.screen_changed,
                    screenshot=screenshot_name,
                    after_screenshot=f"step-{number:02d}-after.png",
                )
                # Remember this action so the next reached screen's graph edge is
                # attributed to it (only meaningful when it changed the screen).
                graph_last_action = (
                    (action.type, action.target) if step.screen_changed else graph_last_action
                )
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
                # Advance an in-screen subgoal when its manual-declared success cue
                # is now visible (e.g. a program becoming "Selected"), so the agent
                # moves on instead of re-tapping the same control.
                advanced_by_cue = False
                cue = subgoal_success_cues.get(current_subgoal_index)
                cue_seen = bool(cue) and subgoal_cue_satisfied(
                    cue, extract_visible_text(after_ui)
                )
                if cue_seen and current_subgoal_index < len(result.subgoals) - 1:
                    current_subgoal.status = "passed"
                    current_subgoal.evidence = f"Observed on screen: {cue!r}"
                    history.append(
                        {
                            "completed_subgoal": current_subgoal.description,
                            "evidence": current_subgoal.evidence,
                            "next_subgoal": result.subgoals[current_subgoal_index + 1].description,
                        }
                    )
                    current_subgoal_index += 1
                    result.subgoals[current_subgoal_index].status = "running"
                    trace.event(
                        "subgoal_advanced",
                        step=number,
                        via="success_cue",
                        cue=cue,
                        next=result.subgoals[current_subgoal_index].description,
                    )
                    self.progress(
                        f"Subgoal advanced via observed cue {cue!r}; "
                        f"now pursuing {result.subgoals[current_subgoal_index].description!r}"
                    )
                    advanced_by_cue = True
                elif cue_seen:
                    # Final subgoal: its success cue completes the whole goal. This
                    # is the deterministic completion path when verify_each_step is
                    # off, and it stops the agent before it toggles the state back
                    # (e.g. tapping Stop after the massage is already running).
                    current_subgoal.status = "passed"
                    current_subgoal.evidence = f"Observed on screen: {cue!r}"
                    result.outcome = "pass"
                    result.reason = current_subgoal.evidence
                    trace.event("verify", scope="success_cue", step=number, cue=cue, outcome="pass")
                    self.progress(f"Goal reached via observed cue {cue!r}")
                    break
                if not step.screen_changed and not advanced_by_cue:
                    remember_failed_action(
                        failed_action_memory, screen_hash, signature
                    )
                    trace.event(
                        "incident",
                        kind_detail="no_progress",
                        step=number,
                        action=action.type,
                        target=action.target,
                        blocked_action=repr(signature),
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
            trace.event("incident", kind_detail="stopped_safely", error=str(exc))
        finally:
            finished = datetime.now(timezone.utc)
            result.finished_at = finished.isoformat()
            cv_steps = sum(1 for step in result.steps if step.grounded_by == "cv")
            a11y_steps = sum(1 for step in result.steps if step.grounded_by == "a11y")
            model_steps = sum(
                1 for step in result.steps if step.grounded_by not in ("cv", "a11y")
            )
            decision_total = sum(
                step.decision_seconds or 0.0 for step in result.steps
            )
            wall_total = max(0.0, (finished - started).total_seconds())
            steps_done = len(result.steps)
            result.grounding = {
                "cv_fast_path_steps": cv_steps,
                "accessibility_fast_path_steps": a11y_steps,
                "model_steps": model_steps,
                "total_steps": steps_done,
                "total_decision_seconds": round(decision_total, 3),
                # Wall-clock is what beats a human tester; decision time is only
                # the model/CV part. The gap is capture + settle + verify overhead.
                "total_wall_seconds": round(wall_total, 3),
                "wall_seconds_per_step": round(wall_total / steps_done, 3) if steps_done else 0.0,
            }
            if scene_graph is not None:
                pending = scene_graph.pending_findings()
                result.scene_graph = {
                    "coverage": scene_graph.coverage(),
                    "pending_findings": [
                        {"id": f.id, "kind": f.kind, "detail": f.detail}
                        for f in pending
                    ],
                }
                if scene_graph_path is not None:
                    try:
                        scene_graph.save(scene_graph_path)
                        scene_graph.save(directory / "scene_graph.json")
                    except Exception:  # noqa: BLE001
                        pass
                trace.event(
                    "graph_summary",
                    coverage=result.scene_graph["coverage"],
                    pending_findings=len(pending),
                )
                if pending:
                    self.progress(
                        f"{len(pending)} HMI divergence(s) need review "
                        "(candidate defects) — see result.json / scene_graph.json"
                    )
            # Crash / ANR capture: scan logcat once at the end (cleared at start),
            # write logcat.txt for debugging, and surface any fatal events.
            if getattr(self.config, "capture_logs", True):
                try:
                    logcat_text = self.device.logcat_dump(
                        getattr(self.config, "log_tail_lines", 4000)
                    )
                    if logcat_text:
                        try:
                            (directory / "logcat.txt").write_text(
                                logcat_text, encoding="utf-8"
                            )
                        except Exception:  # noqa: BLE001
                            pass
                        events = scan_crashes(
                            logcat_text, self.config.target_package or None
                        )
                        result.crashes = crash_summary(events)
                        for event in events:
                            trace.event(
                                "crash",
                                crash_kind=event.kind,
                                package=event.package,
                                summary=event.summary,
                                logcat_line=event.line,
                            )
                        if events:
                            self.progress(
                                f"⚠ {len(events)} fatal log event(s) detected "
                                f"(see logcat.txt): "
                                + "; ".join(f"{e.kind}:{e.package}" for e in events[:3])
                            )
                            if getattr(self.config, "fail_on_crash", True):
                                result.outcome = "fail"
                                first = events[0]
                                result.reason = (
                                    f"Crash detected during run: {first.kind} "
                                    f"{first.package} — {first.summary}"
                                )
                except Exception:  # noqa: BLE001 - log capture is advisory
                    pass
            trace.event(
                "done",
                outcome=result.outcome,
                reason=result.reason,
                grounding=result.grounding,
            )
            trace.close()
            write_report(result)
        return result
