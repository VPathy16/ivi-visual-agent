"""AndroidWorld agent that drives the local IVI visual agent.

This wraps :class:`ivi_agent.model.OllamaVisionModel` and the IVI decision loop
as an ``android_world`` ``EnvironmentInteractingAgent`` so the same local,
goal-driven planner/grounder/verifier can be scored on the AndroidWorld
benchmark. AndroidWorld computes task reward independently via
``task.is_successful``; this agent only decides actions and signals ``done``.

The heavy imports (``android_world``) happen here, not in :mod:`bridge`, so the
pure translation helpers remain testable without the benchmark installed.
"""

from __future__ import annotations

import sys
import traceback
from typing import Any

from android_world.agents import base_agent
from android_world.env import json_action

from ...agent import (
    action_signature,
    blocked_actions_for_state,
    remember_failed_action,
    screen_made_progress,
    title_satisfies_navigation_goal,
)
from ...config import Config
from ...model import OllamaVisionModel
from ...perception import extract_screen_titles, hash_distance, perceptual_hash
from ...policy import PolicyViolation, validate_action
from . import bridge


class IviVisualAgent(base_agent.EnvironmentInteractingAgent):
    """Runs the local IVI visual agent against an AndroidWorld environment."""

    def __init__(
        self,
        env: Any,
        config_path: str | None = None,
        name: str = "ivi_visual_agent",
        transition_pause: float | None = 0.5,
        verbose: bool = True,
    ) -> None:
        super().__init__(env, name, transition_pause=transition_pause)
        self._verbose = verbose
        self.config = Config.load(config_path)
        self.model = OllamaVisionModel(
            self.config.ollama_url,
            self.config.model,
            timeout=self.config.model_timeout_seconds,
            prefer_ui_tree=self.config.prefer_ui_tree,
            enable_ocr=self.config.enable_ocr,
            max_image_dimension=self.config.max_image_dimension,
            grounding_mode=self.config.grounding_mode,
            lenient=self.config.lenient_planning,
            num_ctx=self.config.model_context_tokens,
        )
        self._reset_episode()

    # -- episode state -----------------------------------------------------
    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home)
        self._reset_episode()

    def _reset_episode(self) -> None:
        self._plan: list[str] | None = None
        self._history: list[dict[str, Any]] = []
        self._failed_action_memory: list[tuple[int, set[tuple[object, ...]]]] = []
        self._subgoal_index = 0
        self._step_count = 0

    # -- helpers -----------------------------------------------------------
    def _log(self, message: str) -> None:
        if self._verbose:
            print(f"[ivi] {message}", file=sys.stderr, flush=True)

    def _done(self, status: str, data: dict[str, Any]) -> base_agent.AgentInteractionResult:
        self._log(f"DONE status={status} reason={data.get('reason', '')}")
        try:
            self.env.execute_action(
                json_action.JSONAction(action_type="status", goal_status=status)
            )
        except Exception:  # noqa: BLE001 - status is advisory; never fail on it
            pass
        return base_agent.AgentInteractionResult(True, data)

    def _observe(self) -> tuple[bytes, str, tuple[int, int]]:
        state = self.get_post_transition_state()
        screen_size = self.env.logical_screen_size
        image = bridge.pixels_to_png(state.pixels)
        ui_dump = bridge.ui_elements_to_uiautomator_xml(state.ui_elements, screen_size)
        return image, ui_dump, screen_size

    # -- main loop ---------------------------------------------------------
    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        self._step_count += 1
        image, ui_dump, screen_size = self._observe()
        data: dict[str, Any] = {"goal": goal, "step": self._step_count}

        if self._plan is None:
            self._plan = self.model.create_plan(goal)
            self._subgoal_index = 0
            self._log(f"plan: {' -> '.join(self._plan)}")
        current_subgoal = self._plan[self._subgoal_index]
        data["plan"] = list(self._plan)

        titles = extract_screen_titles(ui_dump)
        self._log(
            f"step {self._step_count} subgoal[{self._subgoal_index}]={current_subgoal!r} "
            f"titles={titles} ui_elements={ui_dump.count('<node')}"
        )

        # 1. Final destination visible? Independent title match ends the episode.
        if title_satisfies_navigation_goal(goal, titles) is not None:
            data["reason"] = "final destination title matched"
            return self._done("complete", data)

        # 2. Advance to the next subgoal when its milestone title is visible.
        if (
            title_satisfies_navigation_goal(current_subgoal, titles) is not None
            and self._subgoal_index < len(self._plan) - 1
        ):
            self._subgoal_index += 1
            current_subgoal = self._plan[self._subgoal_index]

        # 3. On the final subgoal, optionally verify independently before planning.
        #    This is a full extra inference per step, so it is off by default
        #    (verify_each_step): completion is still caught by a finish action and
        #    the free screen-title match above.
        if (
            getattr(self.config, "verify_each_step", True)
            and self._subgoal_index == len(self._plan) - 1
            and self._step_count > 1  # nothing can be complete before any action
        ):
            verification = self.model.verify(goal, image, ui_dump)
            if (
                str(verification.get("outcome")) == "pass"
                and float(verification.get("confidence", 0.0))
                >= self.config.minimum_success_confidence
            ):
                data["reason"] = str(verification.get("evidence", "verified"))
                return self._done("complete", data)

        # 4. Decide one action, with a single re-plan if it repeats a dead action.
        screen_hash = perceptual_hash(image)
        blocked = blocked_actions_for_state(self._failed_action_memory, screen_hash)
        blocked_repr = [repr(item) for item in sorted(blocked, key=repr)]
        try:
            action = self.model.plan(
                goal,
                image,
                ui_dump,
                self._history,
                current_subgoal=current_subgoal,
                blocked_actions=blocked_repr,
            )
            validate_action(action, self.config, goal)
            signature = action_signature(action)
            if signature in blocked and action.type != "wait":
                # The goal may already be satisfied while a fine-grained subgoal
                # tracker lags; verify before looping.
                done_check = self.model.verify(goal, image, ui_dump)
                if (
                    str(done_check.get("outcome")) == "pass"
                    and float(done_check.get("confidence", 0.0))
                    >= self.config.minimum_success_confidence
                ):
                    data["reason"] = str(done_check.get("evidence", "goal satisfied"))
                    return self._done("complete", data)
                self._history.append(
                    {
                        "blocked_repetition": repr(signature),
                        "instruction": "That action made no progress here; try another.",
                    }
                )
                action = self.model.plan(
                    goal,
                    image,
                    ui_dump,
                    self._history,
                    current_subgoal=current_subgoal,
                    blocked_actions=blocked_repr,
                )
                validate_action(action, self.config, goal)
                signature = action_signature(action)
                if signature in blocked:
                    data["reason"] = "repeated-action loop; no viable progress"
                    return self._done("infeasible", data)
        except (PolicyViolation, Exception) as exc:  # noqa: BLE001
            # A single planning failure should not end the episode: recover with a
            # safe Back and let the next step re-observe, until the step budget runs
            # out. This keeps one malformed model response from aborting the run.
            data["reason"] = f"planning failed, recovering with back: {exc}"
            self._log(data["reason"])
            if self._verbose:
                traceback.print_exc()
            try:
                self.env.execute_action(json_action.JSONAction(action_type="navigate_back"))
            except Exception:  # noqa: BLE001
                pass
            return base_agent.AgentInteractionResult(self._at_budget(), data)

        data["action"] = {"type": action.type, "target": action.target, "reason": action.reason}
        self._log(
            f"action: {action.type} target={action.target!r} "
            f"conf={action.confidence:.2f} x={action.x} y={action.y} reason={action.reason!r}"
        )

        # 5. A finish proposal is a request for verification, not success itself.
        if action.type == "finish":
            verification = self.model.verify(goal, image, ui_dump)
            passed = (
                str(verification.get("outcome")) == "pass"
                and float(verification.get("confidence", 0.0))
                >= self.config.minimum_success_confidence
            )
            self._log(f"finish -> verify outcome={verification.get('outcome')} "
                      f"conf={verification.get('confidence')} passed={passed}")
            data["reason"] = str(verification.get("evidence", "verification"))
            if passed:
                return self._done("complete", data)
            remember_failed_action(self._failed_action_memory, screen_hash, action_signature(action))
            return base_agent.AgentInteractionResult(self._at_budget(), data)

        # 6. Execute the translated action and detect visible progress.
        try:
            kwargs = bridge.ivi_action_to_json_action_kwargs(action, screen_size)
            self.env.execute_action(json_action.JSONAction(**kwargs))
        except Exception as exc:  # noqa: BLE001
            data["reason"] = f"action execution failed: {exc}"
            return base_agent.AgentInteractionResult(self._at_budget(), data)

        after_state = self.get_post_transition_state()
        after_hash = perceptual_hash(bridge.pixels_to_png(after_state.pixels))
        after_ui = bridge.ui_elements_to_uiautomator_xml(
            after_state.ui_elements, screen_size
        )
        changed = screen_made_progress(screen_hash, after_hash, ui_dump, after_ui)
        self._log(f"executed {action.type}; screen_changed={changed}")
        self._history.append(
            {
                "step": self._step_count,
                "current_subgoal": current_subgoal,
                "action": action.type,
                "target": action.target,
                "screen_changed": changed,
            }
        )
        if not changed:
            remember_failed_action(
                self._failed_action_memory, screen_hash, action_signature(action)
            )
        return base_agent.AgentInteractionResult(self._at_budget(), data)

    def _at_budget(self) -> bool:
        return self._max_steps is not None and self._step_count >= self._max_steps
