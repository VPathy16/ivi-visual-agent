from __future__ import annotations

import base64
import http.client
import json
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from .perception import (
    extract_ocr_elements,
    extract_screen_titles,
    extract_ui_elements,
    extract_visible_text,
    packages_in_ui,
    prepare_grid_grounding_image,
    prepare_grounded_model_image,
    prepare_model_image,
)
from .types import Action


ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": ["tap", "input_text", "gesture", "back", "home", "wait", "finish"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "target": {"type": "string"},
        "element_id": {"type": "integer", "minimum": 1},
        "direction": {
            "type": "string",
            "enum": ["reveal_above", "reveal_below", "reveal_left", "reveal_right"],
        },
        "region": {"type": "string", "enum": ["center", "top", "bottom", "left", "right"]},
        "x": {"type": "number", "minimum": 0, "maximum": 1},
        "y": {"type": "number", "minimum": 0, "maximum": 1},
        "x2": {"type": "number", "minimum": 0, "maximum": 1},
        "y2": {"type": "number", "minimum": 0, "maximum": 1},
        "duration_ms": {"type": "integer", "minimum": 100, "maximum": 3000},
        "seconds": {"type": "number", "minimum": 0, "maximum": 10},
        "text": {"type": "string"},
        "outcome": {
            "type": "string",
            "enum": ["pass", "fail", "inconclusive"],
        },
        "evidence": {"type": "string"},
    },
    "required": ["type", "confidence", "reason"],
    "additionalProperties": False,
}

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subgoals": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {"description": {"type": "string"}},
                "required": ["description"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["subgoals"],
    "additionalProperties": False,
}

VERIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "outcome": {
            "type": "string",
            "enum": ["pass", "fail", "inconclusive"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
    },
    "required": ["outcome", "confidence", "evidence"],
    "additionalProperties": False,
}

GROUNDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "cell": {"type": "integer", "minimum": 1, "maximum": 72},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
    },
    "required": ["found", "cell", "confidence", "evidence"],
    "additionalProperties": False,
}


SUBGOAL_PROMPT = """You are the milestone planner for an arbitrary Android IVI.
Turn the user's goal into 1 to 4 sequential, observable subgoals. Describe WHAT visible
screen or result must be reached after navigation, not the visibility, highlighting, or
selection of an icon, button, field, or option. Preserve exact names and text from the
goal. Do not assume a phone launcher, package name, menu hierarchy, coordinate, or
device-specific route. Use the fewest milestones needed. The final subgoal must visibly
satisfy the complete user goal.
Return only JSON: {"subgoals":[{"description":"..."}]}.
"""


PLANNER_PROMPT = """You are a goal-directed controller for an arbitrary Android IVI on
a stationary test bench. Choose exactly ONE safe action that visibly advances the USER
GOAL toward the CURRENT SUBGOAL. Do not assume a phone launcher, app drawer, menu name, screen layout, navigation
gesture, or coordinate convention beyond what the current screenshot and UI elements
show. Ignore unrelated content. Use Home or Back only when the observed state and
history justify it. Never repeat an action that history says made no progress.

Every tap must have current visible evidence that the selected control is either the
requested destination or an immediate semantic parent of it. Do not take indirect
shortcuts through notifications, status indicators, unrelated settings, or apps merely
because they might eventually provide another route.
For a goal that asks to open, show, or navigate to a screen, never tap a candidate marked
stateful; it changes a setting instead of navigating to the requested destination.

The screenshot is always supplied. When numbered UI candidates are supplied, tap only
by returning their element_id; never invent coordinates. When no accessibility-tree
candidates exist, a visual tap may name an unboxed target; a separate visual grounder
will locate it. Never claim a screen is absent when its title or content is visibly shown.
When a search or text field is visible and entering text advances the current subgoal,
return input_text with that field's element_id and the exact text to enter. input_text
focuses and types atomically. Do not repeatedly tap an already visible or focused field.
For scrolling or paging return type "gesture" with direction reveal_above,
reveal_below, reveal_left, or reveal_right and region center/top/bottom/left/right.
Direction names the content the gesture should reveal, not finger motion. Do not assume
what a gesture opens on this device. A finish action requires an outcome. Never guess an
invisible control, delete data, place calls, purchase, reset, update software, or accept
surprising permissions. Keep reason under 20 words. Return only action JSON.
"""


VERIFIER_PROMPT = """Independently verify whether the goal is visibly complete in this
Android automotive IVI screenshot. Return only JSON:
{"outcome":"pass|fail|inconclusive","confidence":0.0,"evidence":"specific visible evidence"}
Use pass only when the screenshot clearly proves completion. Use fail only for a clear
error or contradiction. Otherwise use inconclusive. Do not rely on the planner's claim.
A launcher, dock, search result, notification, icon, button, or menu item that could be
used to reach the goal is NOT completion. For "open" goals, the requested app or screen
must be the active visible content, proven by its title and/or distinctive page content.
Never infer that a future tap or navigation route would satisfy the goal.
"""


class ModelError(RuntimeError):
    pass


class OllamaVisionModel:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 120,
        prefer_ui_tree: bool = True,
        enable_ocr: bool = True,
        max_image_dimension: int = 1024,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.prefer_ui_tree = prefer_ui_tree
        self.enable_ocr = enable_ocr
        self.max_image_dimension = max_image_dimension

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise ModelError(f"Model did not return JSON: {text[:300]}")
            try:
                value = json.loads(text[start : end + 1])
            except json.JSONDecodeError as exc:
                raise ModelError(f"Model returned invalid JSON: {text[:300]}") from exc
        if not isinstance(value, dict):
            raise ModelError("Model response must be a JSON object")
        return value

    def _chat(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes | None,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        user_message: dict[str, Any] = {"role": "user", "content": user_prompt}
        if image is not None:
            user_message["images"] = [base64.b64encode(image).decode("ascii")]
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            # JSON mode is more compatible and faster than grammar compilation on the
            # older local Ollama versions used by bench machines. Validation is local.
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                user_message,
            ],
            "options": {"temperature": 0},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace").strip()
            raise ModelError(f"Ollama returned HTTP {exc.code}: {detail or exc.reason}") from exc
        except (urllib.error.URLError, http.client.HTTPException) as exc:
            raise ModelError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc
        try:
            return self._extract_json(result["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise ModelError(f"Unexpected Ollama response: {result}") from exc

    @staticmethod
    def _compact_ui_dump(ui_dump: str, limit: int = 20) -> list[dict[str, str]]:
        if not ui_dump.strip():
            return []
        try:
            root = ET.fromstring(ui_dump)
        except ET.ParseError:
            return []
        elements: list[dict[str, str]] = []
        for node in root.iter("node"):
            attributes = {
                key: node.attrib.get(key, "")
                for key in ("text", "content-desc", "resource-id", "class", "clickable", "bounds")
                if node.attrib.get(key)
            }
            meaningful = any(attributes.get(key) for key in ("text", "content-desc", "resource-id"))
            if meaningful or attributes.get("clickable") == "true":
                elements.append(attributes)
            if len(elements) >= limit:
                break
        return elements

    @staticmethod
    def _packages(ui_dump: str, limit: int = 8) -> list[str]:
        return packages_in_ui(ui_dump, limit)

    @staticmethod
    def _validate_action_shape(
        action: Action,
        candidate_ids: set[int] | None = None,
        allow_visual_tap: bool = False,
    ) -> None:
        candidate_ids = candidate_ids or set()
        if action.type == "tap":
            if action.element_id is not None and action.element_id not in candidate_ids:
                raise ValueError("tap must select a valid UI candidate element_id")
            if action.element_id is None and (action.x is None or action.y is None):
                raise ValueError("visual tap is missing x or y")
            if action.element_id is None and not allow_visual_tap:
                raise ValueError("tap must select a UI candidate element_id")
        if action.type == "input_text":
            if action.element_id not in candidate_ids:
                raise ValueError("input_text must select a valid UI candidate element_id")
            if not action.text:
                raise ValueError("input_text is missing text")
        if action.type == "gesture":
            if action.direction not in {
                "reveal_above",
                "reveal_below",
                "reveal_left",
                "reveal_right",
            }:
                raise ValueError("gesture is missing a valid direction")
            if action.region not in {"center", "top", "bottom", "left", "right"}:
                raise ValueError("gesture has an invalid region")
        if action.type == "swipe" and any(
            value is None for value in (action.x, action.y, action.x2, action.y2)
        ):
            raise ValueError("swipe is missing x, y, x2, or y2")
        if action.type == "finish" and action.outcome not in {
            "pass",
            "fail",
            "inconclusive",
        }:
            raise ValueError("finish is missing a valid outcome")

    @staticmethod
    def _direct_candidate(goal: str, elements: list[Any]) -> Any | None:
        def normalize(value: str) -> str:
            return " ".join(re.findall(r"[a-z0-9]+", value.lower()))

        destination = normalize(goal)
        prefixes = (
            "open the ",
            "open ",
            "go to the ",
            "go to ",
            "navigate to the ",
            "navigate to ",
            "launch the ",
            "launch ",
        )
        for prefix in prefixes:
            if destination.startswith(prefix):
                destination = destination[len(prefix) :]
                break
        for suffix in (" screen", " page", " menu"):
            if destination.endswith(suffix):
                destination = destination[: -len(suffix)]
                break

        # Accept only exact destination names. A shared word is not enough:
        # "Notification settings" and "Show Bluetooth devices without names" must
        # never match a goal for "Bluetooth settings".
        variants = {destination}
        if destination.endswith(" settings") and destination != "settings":
            variants.add(destination.removesuffix(" settings"))
        variants.discard("")
        for element in elements:
            if getattr(element, "stateful", False) or getattr(element, "checkable", False):
                continue
            if normalize(element.label) in variants:
                return element
        return None

    def create_plan(self, goal: str) -> list[str]:
        prompt = f"USER GOAL: {goal}\nCreate observable, device-independent milestones."
        last_error = ""
        for attempt in range(2):
            retry = (
                "\nThe previous response was invalid. Return only the required short JSON. "
                f"Error: {last_error}"
                if attempt
                else ""
            )
            try:
                response = self._chat(SUBGOAL_PROMPT, prompt + retry, None, PLAN_SCHEMA)
                raw_subgoals = response.get("subgoals")
                if not isinstance(raw_subgoals, list):
                    raise ValueError("subgoals must be a list")
                subgoals: list[str] = []
                for raw in raw_subgoals[:5]:
                    description = raw.get("description") if isinstance(raw, dict) else raw
                    if not isinstance(description, str) or not description.strip():
                        raise ValueError("each subgoal needs a description")
                    cleaned = description.strip()
                    if cleaned not in subgoals:
                        subgoals.append(cleaned)
                if not subgoals:
                    raise ValueError("plan was empty")
                return subgoals
            except (ModelError, TypeError, ValueError) as exc:
                last_error = str(exc)
        raise ModelError(f"Model failed to create a valid subgoal plan: {last_error}")

    def _ground_visual_target(self, image: bytes, target: str) -> tuple[float, float, float]:
        prompt = (
            f"Find the numbered grid cell containing the visible center of: {target!r}. "
            "The image has 72 cells: 12 columns by 6 rows, numbered left-to-right "
            "then top-to-bottom. Return the one cell containing the target's center. "
            "If it is not clearly visible, set found to false. Return only JSON."
        )
        prepared = prepare_grid_grounding_image(image, self.max_image_dimension)
        response = self._chat(
            "You are a visual grounding component. Locate only the requested target; "
            "do not choose another action or infer an invisible control.",
            prompt,
            prepared,
            GROUNDING_SCHEMA,
        )
        if isinstance(response.get("data"), dict):
            response = response["data"]
        found = response.get("found")
        if isinstance(found, str):
            found = found.strip().lower() == "true"
        if found is not True:
            raise ValueError(f"visual target was not found: {target}")
        try:
            raw_cell = response.get(
                "cell",
                response.get("cell_number", response.get("grid_cell")),
            )
            cell = int(raw_cell)
            confidence = float(response.get("confidence", 0.8))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("visual grounder returned an invalid grid cell") from exc
        if not 1 <= cell <= 72 or confidence < 0.6:
            raise ValueError("visual grounder was not confident in a valid grid cell")
        zero_based = cell - 1
        column = zero_based % 12
        row = zero_based // 12
        x = (column + 0.5) / 12
        y = (row + 0.5) / 6
        return x, y, confidence

    def plan(
        self,
        goal: str,
        image: bytes,
        ui_dump: str,
        history: list[dict[str, Any]],
        current_subgoal: str | None = None,
        blocked_actions: list[str] | None = None,
    ) -> Action:
        decision_goal = current_subgoal or goal
        elements = extract_ui_elements(ui_dump)
        direct = self._direct_candidate(decision_goal, elements)
        if direct is None and self.enable_ocr and not elements:
            ocr_elements = extract_ocr_elements(image, start_id=len(elements) + 1)
            existing = {(item.label.lower(), item.bounds) for item in elements}
            elements.extend(
                item
                for item in ocr_elements
                if (item.label.lower(), item.bounds) not in existing
            )
            direct = self._direct_candidate(decision_goal, elements)
        candidate_ids = {element.id for element in elements}
        if direct is not None:
            return Action(
                type="tap",
                element_id=direct.id,
                x=direct.center[0],
                y=direct.center[1],
                target=direct.label,
                confidence=0.99,
                reason=f"Visible UI candidate '{direct.label}' directly matches the goal",
            )
        context = {
            "USER_GOAL": goal,
            "CURRENT_SUBGOAL": decision_goal,
            "visible_packages": self._packages(ui_dump),
            "screen_titles": extract_screen_titles(ui_dump),
            "ui_candidates": [element.to_prompt_dict() for element in elements],
            "blocked_actions_on_this_screen": blocked_actions or [],
            "history": history[-10:],
        }
        has_ui_tree_candidates = any(item.source == "ui_tree" for item in elements)
        use_visual_fallback = not has_ui_tree_candidates or not self.prefer_ui_tree
        # Keep visual context even when the accessibility tree is useful. The tree
        # grounds safe actions, while the screenshot carries layout, titles, and
        # state that many IVI implementations do not expose as interactive nodes.
        model_image = prepare_grounded_model_image(
            image, elements, self.max_image_dimension
        )
        mode = (
            "Use the attached screenshot. Red boxes and numbers identify OCR candidates; "
            "for an unboxed visual control, tap with normalized x and y and no element_id."
            if use_visual_fallback
            else (
                "Use the attached screenshot to understand the current screen. Red boxes "
                "and numbers correspond to UI candidates. Tap only with element_id."
            )
        )
        prompt = (
            f"USER GOAL: {goal}\n"
            f"CURRENT SUBGOAL: {decision_goal}\n"
            f"{mode} Choose one action that advances this current subgoal. "
            "Do not tap unrelated content.\n"
            "Required JSON keys are type, confidence, and reason. For a candidate tap "
            "also return element_id and target exactly matching its candidate label. "
            "For input_text return element_id, exact candidate target, and text. "
            "For a gesture also return direction and region. "
            "Return finish with outcome pass only when the current subgoal is visibly complete. "
            "Use the key type, never action.\n"
            f"CONTEXT: {json.dumps(context)}"
        )
        invalid: dict[str, Any] | None = None
        validation_error = ""
        for attempt in range(2):
            retry_note = ""
            if attempt:
                retry_note = (
                    "\nThe previous response was invalid or incomplete. Return one short, "
                    "complete JSON object. "
                    f"Validation error: {validation_error}. Required: type, confidence, reason; "
                    "candidate tap needs element_id and its exact target label (or x/y only in "
                    "visual fallback); input_text needs element_id, exact target, and text; "
                    "gesture needs direction and region."
                )
                if invalid is not None:
                    retry_note += f" Previous response: {json.dumps(invalid)[:600]}"
            try:
                response = self._chat(
                    PLANNER_PROMPT,
                    prompt + retry_note,
                    model_image,
                    ACTION_SCHEMA,
                )
                if isinstance(response.get("element_id"), str) and response["element_id"].isdigit():
                    response["element_id"] = int(response["element_id"])
                # A finish proposal is only a request for independent visual
                # verification; it never directly marks the run successful. Let
                # compact local-model responses omit redundant result fields.
                if response.get("type") == "finish":
                    response.setdefault("outcome", "inconclusive")
                    response.setdefault("evidence", str(response.get("reason", "")))
                action = Action.from_dict(response)
                if action.type not in {
                    "tap",
                    "input_text",
                    "gesture",
                    "back",
                    "home",
                    "wait",
                    "finish",
                }:
                    raise ValueError("unknown action type")
                if (
                    action.type == "tap"
                    and use_visual_fallback
                    and action.element_id is None
                    and (action.x is None or action.y is None)
                    and action.target
                ):
                    action.x, action.y, grounding_confidence = self._ground_visual_target(
                        image, action.target
                    )
                    action.confidence = min(action.confidence, grounding_confidence)
                if (
                    action.type == "tap"
                    and use_visual_fallback
                    and action.element_id is not None
                    and action.element_id not in candidate_ids
                    and action.target
                ):
                    # Small vision models sometimes invent a box number for an
                    # unboxed icon. Never execute that ID; re-ground the semantic
                    # target against the independent numbered grid instead.
                    action.x, action.y, grounding_confidence = self._ground_visual_target(
                        image, action.target
                    )
                    action.element_id = None
                    action.confidence = min(action.confidence, grounding_confidence)
                self._validate_action_shape(
                    action, candidate_ids, allow_visual_tap=use_visual_fallback
                )
                if action.type in {"tap", "input_text"} and action.element_id is not None:
                    selected = next(item for item in elements if item.id == action.element_id)
                    declared_target = " ".join(
                        re.findall(r"[a-z0-9]+", action.target.lower())
                    )
                    selected_target = " ".join(
                        re.findall(r"[a-z0-9]+", selected.label.lower())
                    )
                    if not declared_target:
                        raise ValueError("candidate action must declare its exact target label")
                    if declared_target != selected_target:
                        if use_visual_fallback and action.type == "tap":
                            # OCR boxes are hints, not authority. If the model names a
                            # different visible control, discard its incorrect box ID
                            # and independently ground that semantic target.
                            action.x, action.y, grounding_confidence = (
                                self._ground_visual_target(image, action.target)
                            )
                            action.element_id = None
                            action.confidence = min(
                                action.confidence, grounding_confidence
                            )
                            return action
                        raise ValueError(
                            "declared target does not match the selected candidate label"
                        )
                    navigation_goal = bool(
                        re.match(
                            r"^\s*(?:open|show|go\s+to|navigate\s+to|launch)\b",
                            decision_goal,
                            re.IGNORECASE,
                        )
                    )
                    if action.type == "tap" and navigation_goal and selected.stateful:
                        raise ValueError(
                            "a navigation goal cannot tap a state-changing control"
                        )
                    if action.type == "input_text" and not selected.editable:
                        raise ValueError("input_text target is not an editable field")
                    action.x, action.y = selected.center
                    action.target = selected.label
                return action
            except (ModelError, TypeError, ValueError) as exc:
                if "response" in locals():
                    invalid = response
                validation_error = str(exc)
        raise ModelError(
            "Model failed to return a valid action after retry: "
            f"{invalid}; validation error: {validation_error}"
        )

    def verify(self, goal: str, image: bytes, ui_dump: str = "") -> dict[str, Any]:
        visible_text = extract_visible_text(ui_dump)
        if not visible_text and self.enable_ocr:
            visible_text = [item.label for item in extract_ocr_elements(image)]
        prompt = (
            f"GOAL: {goal}\n"
            f"ANDROID-REPORTED VISIBLE TEXT: {json.dumps(visible_text)}\n"
            "Return outcome, confidence, and specific visible evidence. Any text named "
            "as evidence must occur in the supplied visible-text list or visibly in the image."
        )
        prepared = prepare_model_image(image, self.max_image_dimension)
        result = self._chat(VERIFIER_PROMPT, prompt, prepared, VERIFICATION_SCHEMA)
        if result.get("outcome") == "pass":
            stop_words = {
                "a", "an", "and", "app", "application", "connect", "go", "launch",
                "menu", "navigate", "open", "page", "screen", "settings", "show",
                "the", "to",
            }
            goal_terms = {
                term
                for term in re.findall(r"[a-z0-9]+", goal.lower())
                if term not in stop_words and len(term) > 1
            }
            observed_terms = set(
                re.findall(r"[a-z0-9]+", " ".join(visible_text).lower())
            )
            if goal_terms and not goal_terms.issubset(observed_terms):
                missing = ", ".join(sorted(goal_terms - observed_terms))
                return {
                    "outcome": "inconclusive",
                    "confidence": 0.0,
                    "evidence": (
                        "Completion claim rejected because target text was not observed: "
                        f"{missing}"
                    ),
                }
        return result
