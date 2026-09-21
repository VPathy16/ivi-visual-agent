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
    extract_ocr_screen_titles,
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
            "enum": [
                "tap",
                "long_press",
                "double_tap",
                "input_text",
                "keyboard_enter",
                "gesture",
                "swipe",
                "open_app",
                "back",
                "home",
                "wait",
                "finish",
            ],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "target": {"type": "string"},
        "app_name": {"type": "string"},
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

POINT_GROUNDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "x": {"type": "number", "minimum": 0, "maximum": 1},
        "y": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
    },
    "required": ["found", "x", "y", "confidence", "evidence"],
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
what a gesture opens on this device.
Use long_press to open a context menu on a visible item, and double_tap only when a
single tap is clearly insufficient; both ground exactly like tap (element_id or visual
target). To drag a slider or seek bar (for example to set brightness or volume to its
maximum or minimum), return type swipe with x,y at the slider thumb and x2,y2 at the
target end of the track. Do not tap a scrollable list to move it; use a gesture. Use keyboard_enter to submit text already typed into a focused field. Use
open_app with app_name only to launch a named application directly instead of hunting
through a launcher; never invent a package name. When the goal concerns a system
setting (for example Wi-Fi, Bluetooth, brightness, sound, or notifications) and the
Settings app is NOT already open, prefer open_app with app_name "Settings" rather than
tapping status-bar icons, clocks, or home-screen widgets. Never open_app an app that is
already the active screen: once Settings is open, tap the relevant category (for
example Display for brightness), and if that category is not currently visible, SCROLL
with a gesture (reveal_below) to find it instead of re-opening Settings.
A finish action requires an outcome.
Never guess an invisible control, delete data, place calls, purchase, reset, update
software, or accept surprising permissions. Keep reason under 20 words. Return only
action JSON.
"""


VERIFIER_PROMPT = """Independently verify whether the goal is visibly complete in this
Android screenshot. Return only JSON:
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
        grounding_mode: str = "grid",
        lenient: bool = False,
        num_ctx: int = 8192,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.prefer_ui_tree = prefer_ui_tree
        self.enable_ocr = enable_ocr
        self.max_image_dimension = max_image_dimension
        if grounding_mode not in {"grid", "point"}:
            raise ValueError("grounding_mode must be 'grid' or 'point'")
        self.grounding_mode = grounding_mode
        # Lenient mode trusts a concretely-resolved element_id: it backfills a
        # missing target label and does not hard-fail on target-mismatch or
        # semantic-relatedness. Small models exploring a benchmark need this;
        # the strict default preserves the original goal-driven guardrails.
        self.lenient = lenient

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
        image: bytes | list[bytes] | None,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        user_message: dict[str, Any] = {"role": "user", "content": user_prompt}
        if image is not None:
            images = image if isinstance(image, list) else [image]
            user_message["images"] = [
                base64.b64encode(item).decode("ascii") for item in images
            ]
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            # Keep the model resident between calls so it is not reloaded (a multi-GB
            # reload dominates latency when the model is evicted between steps).
            "keep_alive": "10m",
            # JSON mode is more compatible and faster than grammar compilation on the
            # older local Ollama versions used by bench machines. Validation is local.
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                user_message,
            ],
            # num_ctx raises Ollama's default context window (often 4096), which a
            # screenshot + UI candidates + history easily exceeds.
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
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
        if action.type in {"tap", "long_press", "double_tap"}:
            if action.element_id is not None and action.element_id not in candidate_ids:
                raise ValueError(f"{action.type} must select a valid UI candidate element_id")
            if action.element_id is None and (action.x is None or action.y is None):
                raise ValueError(f"visual {action.type} is missing x or y")
            if action.element_id is None and not allow_visual_tap:
                raise ValueError(f"{action.type} must select a UI candidate element_id")
        if action.type == "open_app" and not action.app_name.strip():
            raise ValueError("open_app requires a non-empty app_name")
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

    @staticmethod
    def _destination_name(goal: str) -> str:
        destination = " ".join(re.findall(r"[a-z0-9]+", goal.lower()))
        destination = re.sub(
            r"^(?:(?:go|navigate) to|open|show|launch|reach|view|display)"
            r"(?: and (?:open|show|view|display))? (?:the )?",
            "",
            destination,
        )
        suffixes = (
            "main screen",
            "application",
            "app",
            "screen",
            "page",
            "menu",
            "panel",
            "overlay",
            "dialog",
        )
        changed = True
        while changed:
            changed = False
            for suffix in suffixes:
                if destination.endswith(f" {suffix}"):
                    destination = destination[: -(len(suffix) + 1)].strip()
                    changed = True
                    break
        return destination

    @staticmethod
    def _candidate_semantically_advances(
        action: Action,
        decision_goal: str,
    ) -> bool:
        generic = {
            "a", "an", "and", "button", "car", "container", "control", "icon",
            "item", "menu", "open", "screen", "show", "the", "to", "toolbar", "ui",
        }
        tokens = lambda value: set(re.findall(r"[a-z0-9]+", value.lower()))
        target_tokens = tokens(action.target) - generic
        goal_tokens = tokens(decision_goal) - generic - {
            "app", "application", "display", "go", "launch", "navigate", "page", "reach",
            "view",
        }
        if target_tokens & goal_tokens:
            return True
        reason_tokens = tokens(action.reason)
        # An icon/symbol tile (e.g. "~M~") yields no lexical word to overlap a
        # worded goal, so it could never pass the target/goal token test however
        # correct the tap is. When the target has no real word, judge it by
        # whether the model's stated reason ties the action to the goal
        # ("...the Seat Comfort tile to open the massage screen").
        if not {token for token in target_tokens if len(token) >= 2}:
            return bool(goal_tokens & reason_tokens)
        return bool(target_tokens & reason_tokens) and bool(goal_tokens & reason_tokens)

    @staticmethod
    def _documented_subgoals(
        goal: str, knowledge_context: dict[str, Any] | None
    ) -> list[str] | None:
        if not knowledge_context:
            return None
        generic = {"a", "an", "as", "the", "to", "and", "open", "select", "show"}
        goal_terms = set(re.findall(r"[a-z0-9]+", goal.lower())) - generic
        for chunk in knowledge_context.get("retrieved_chunks", []):
            if chunk.get("kind") != "task" or not isinstance(chunk.get("data"), dict):
                continue
            task = chunk["data"]
            task_goal = str(task.get("goal", task.get("name", "")))
            task_terms = set(re.findall(r"[a-z0-9]+", task_goal.lower())) - generic
            if not goal_terms.intersection(task_terms):
                continue
            steps = task.get("steps", [])
            if not isinstance(steps, list) or not steps:
                continue
            milestones = [
                str(step.get("milestone", step.get("expected", ""))).strip()
                for step in steps[:-1]
                if isinstance(step, dict)
                and str(step.get("milestone", step.get("expected", ""))).strip()
            ]
            milestones.append(goal.strip())
            return list(dict.fromkeys(milestones))[:5]
        return None

    def create_plan(
        self, goal: str, knowledge_context: dict[str, Any] | None = None
    ) -> list[str]:
        documented = self._documented_subgoals(goal, knowledge_context)
        if documented:
            return documented
        prompt = (
            f"USER GOAL: {goal}\n"
            "Create observable, device-independent milestones.\n"
            f"RETRIEVED DOCUMENTATION: {json.dumps(knowledge_context or {})}"
        )
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
                    control_visibility = re.search(
                        r"\b(?:icon|button|field|control|category)\b.{0,100}"
                        r"\b(?:visible|selected|highlighted|labeled)\b",
                        cleaned,
                        re.IGNORECASE,
                    )
                    if control_visibility:
                        raise ValueError(
                            "a control being visible or selected is not a completed "
                            "screen/result milestone"
                        )
                    if cleaned not in subgoals:
                        subgoals.append(cleaned)
                if not subgoals:
                    raise ValueError("plan was empty")
                exact_goal = goal.strip()
                if not exact_goal:
                    raise ValueError("goal was empty")
                # The model may suggest an entry screen, but it must not dictate
                # speculative intermediate routes. Those are discovered from each
                # observed screen. Always keep the user's exact goal as the final
                # milestone.
                first = subgoals[0]
                normalize = lambda value: " ".join(
                    re.findall(r"[a-z0-9]+", value.lower())
                )
                destination = self._destination_name(exact_goal)
                feature_settings_goal = (
                    destination.endswith(" settings") and destination != "settings"
                )
                if feature_settings_goal and (
                    len(subgoals) == 1 or "settings" not in normalize(first)
                ):
                    return ["Open the Settings application", exact_goal]
                if normalize(first) == normalize(exact_goal) or len(subgoals) == 1:
                    return [exact_goal]
                return [first, exact_goal]
            except (ModelError, TypeError, ValueError) as exc:
                last_error = str(exc)
        # The unsplit user goal remains a valid, device-independent plan. This
        # prevents weak local models from making navigation depend on a malformed
        # milestone while preserving goal-driven behavior.
        exact_goal = goal.strip()
        destination = self._destination_name(exact_goal)
        if destination.endswith(" settings") and destination != "settings":
            return ["Open the Settings application", exact_goal]
        return [exact_goal]

    def _ground_visual_target_point(
        self,
        image: bytes,
        target: str,
        reference_images: list[bytes] | None = None,
    ) -> tuple[float, float, float]:
        """Ask a grounding-capable VLM (e.g. qwen3-vl) for the target's point.

        Coordinates are returned NORMALIZED to [0, 1] relative to the image the
        model actually receives, so the caller never has to know the device
        resolution or the downscaled image size. This is far more precise than
        the 72-cell grid on models trained for GUI grounding.
        """
        prompt = (
            f"IMAGE 1 is a live Android screenshot. Locate the visible center of: "
            f"{target!r}. Return normalized coordinates where x is the fraction of the "
            "image width from the left edge (0.0 to 1.0) and y is the fraction of the "
            "image height from the top edge (0.0 to 1.0). Any later images are isolated "
            "reference icons that only help identify the target; never return a location "
            "from a reference image. If the target is not clearly visible in IMAGE 1, set "
            "found to false. Return only JSON."
        )
        prepared = prepare_model_image(image, self.max_image_dimension)
        response = self._chat(
            "You are a GUI visual grounding component. Return the normalized on-screen "
            "point of only the requested target in IMAGE 1; do not choose an action or "
            "infer an invisible control.",
            prompt,
            [prepared, *(reference_images or [])],
            POINT_GROUNDING_SCHEMA,
        )
        if isinstance(response.get("data"), dict):
            response = response["data"]
        found = response.get("found")
        if isinstance(found, str):
            found = found.strip().lower() == "true"
        if found is not True:
            raise ValueError(f"visual target was not found: {target}")
        try:
            x = float(response["x"])
            y = float(response["y"])
            confidence = float(response.get("confidence", 0.8))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("visual grounder returned invalid coordinates") from exc
        # Some GUI-grounding models emit 0-1000 or pixel coordinates instead of a
        # 0-1 fraction; rescale defensively so a valid location is not rejected.
        if x > 1.0 or y > 1.0:
            scale = 1000.0 if max(x, y) <= 1000.0 else float(max(x, y))
            x, y = x / scale, y / scale
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) or confidence < 0.6:
            raise ValueError("visual grounder was not confident in a valid point")
        return x, y, confidence

    def _ground_visual_target(
        self,
        image: bytes,
        target: str,
        reference_images: list[bytes] | None = None,
    ) -> tuple[float, float, float]:
        if self.grounding_mode == "point":
            return self._ground_visual_target_point(image, target, reference_images)
        prompt = (
            f"In IMAGE 1, find the numbered grid cell containing the visible center of: "
            f"{target!r}. IMAGE 1 has 72 cells: 12 columns by 6 rows, numbered left-to-right "
            "then top-to-bottom. Return the one cell containing the target's center. "
            "Any later images are isolated reference icons that may help identify the target; "
            "never return a location from a reference image. "
            "If it is not clearly visible, set found to false. Return only JSON."
        )
        prepared = prepare_grid_grounding_image(image, self.max_image_dimension)
        response = self._chat(
            "You are a visual grounding component. Locate only the requested target; "
            "do not choose another action or infer an invisible control. Ground only in "
            "the numbered live screenshot in IMAGE 1.",
            prompt,
            [prepared, *(reference_images or [])],
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
        knowledge_context: dict[str, Any] | None = None,
        reference_images: list[bytes] | None = None,
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
            "retrieved_documentation": knowledge_context or {},
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
            "Retrieved documentation describes semantic routes and proprietary symbols, but "
            "the live screen must confirm a target before any action. When reference images "
            "are attached, IMAGE 1 is the live device and later images are isolated icon "
            "examples only. Follow only the active documented step; never skip to a later "
            "task target. "
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
                    [model_image, *(reference_images or [])],
                    ACTION_SCHEMA,
                )
                # Ollama JSON mode does not enforce the enum, so small models
                # sometimes return a placeholder type ("action") or put the real
                # verb under an "action"/"action_type" key. Recover it before
                # validation instead of failing the whole step.
                valid_types = set(ACTION_SCHEMA["properties"]["type"]["enum"])
                if response.get("type") not in valid_types:
                    alternate = response.get("action") or response.get("action_type")
                    if isinstance(alternate, str) and alternate in valid_types:
                        response["type"] = alternate
                    elif response.get("element_id") is not None or (
                        response.get("x") is not None and response.get("y") is not None
                    ):
                        # A concrete tap target was named; the verb was just mislabeled.
                        response["type"] = "tap"
                # The schema requires type, confidence, and reason, but small models
                # routinely omit confidence and/or reason -- even for taps. Default
                # them for any recognized action type rather than failing
                # Action.from_dict on a missing required argument.
                if isinstance(response.get("type"), str):
                    response.setdefault("confidence", 0.8)
                    response.setdefault(
                        "reason",
                        f"{response['type']} action proposed by local model",
                    )
                if (
                    response.get("type") in {"tap", "input_text"}
                    and not response.get("target")
                    and isinstance(response.get("candidate_target"), str)
                ):
                    response["target"] = response["candidate_target"]
                if isinstance(response.get("element_id"), str) and response["element_id"].isdigit():
                    response["element_id"] = int(response["element_id"])
                elif isinstance(response.get("element_id"), str):
                    # Some models put a label (e.g. "3G") in element_id instead of a
                    # candidate number. Treat it as a visual target and drop the id.
                    if not response.get("target"):
                        response["target"] = response["element_id"]
                    response["element_id"] = None
                # A finish proposal is only a request for independent visual
                # verification; it never directly marks the run successful. Let
                # compact local-model responses omit redundant result fields.
                if response.get("type") == "finish":
                    response.setdefault("outcome", "inconclusive")
                    response.setdefault("evidence", str(response.get("reason", "")))
                action = Action.from_dict(response)
                tap_like = {"tap", "long_press", "double_tap"}
                if action.type not in {
                    "tap",
                    "long_press",
                    "double_tap",
                    "input_text",
                    "keyboard_enter",
                    "gesture",
                    "swipe",
                    "open_app",
                    "back",
                    "home",
                    "wait",
                    "finish",
                }:
                    raise ValueError("unknown action type")
                if (
                    action.type in ({"input_text"} | tap_like)
                    and action.element_id is None
                    and action.target
                    and elements
                ):
                    # Models often name the control ("Network & internet") instead
                    # of returning its candidate number. Resolve the named target to
                    # an accessibility candidate by label before falling back to
                    # visual grounding.
                    normalize = lambda value: " ".join(
                        re.findall(r"[a-z0-9]+", value.lower())
                    )
                    wanted = normalize(action.target)
                    match = next(
                        (el for el in elements if normalize(el.label) == wanted), None
                    )
                    if match is None and wanted:
                        match = next(
                            (el for el in elements if wanted in normalize(el.label)),
                            None,
                        )
                    if match is not None:
                        action.element_id = match.id
                        action.x, action.y = match.center
                        action.target = match.label
                if (
                    action.type in tap_like
                    and (use_visual_fallback or self.lenient)
                    and action.element_id is None
                    and (action.x is None or action.y is None)
                    and action.target
                ):
                    # In lenient mode, a named target that matched no accessibility
                    # candidate is still located visually from the screenshot rather
                    # than failing the step.
                    action.x, action.y, grounding_confidence = self._ground_visual_target(
                        image, action.target, reference_images
                    )
                    action.confidence = min(action.confidence, grounding_confidence)
                if (
                    action.type in tap_like
                    and use_visual_fallback
                    and action.element_id is not None
                    and action.element_id not in candidate_ids
                    and action.target
                ):
                    # Small vision models sometimes invent a box number for an
                    # unboxed icon. Never execute that ID; re-ground the semantic
                    # target against the independent numbered grid instead.
                    action.x, action.y, grounding_confidence = self._ground_visual_target(
                        image, action.target, reference_images
                    )
                    action.element_id = None
                    action.confidence = min(action.confidence, grounding_confidence)
                self._validate_action_shape(
                    action,
                    candidate_ids,
                    allow_visual_tap=use_visual_fallback or self.lenient,
                )
                if action.type in ({"input_text"} | tap_like) and action.element_id is not None:
                    selected = next(item for item in elements if item.id == action.element_id)
                    if action.type in tap_like and selected.scrollable:
                        # The model tapped a scroll container -- it means to scroll,
                        # not tap. Tapping the middle of a list opens whatever row is
                        # there (the brightness runs looped on this). Convert to a
                        # scroll gesture so list navigation actually advances.
                        upward = re.search(
                            r"\b(up|back|previous|top|above)\b", action.reason.lower()
                        )
                        return Action(
                            type="gesture",
                            direction="reveal_above" if upward else "reveal_below",
                            region="center",
                            confidence=max(action.confidence, 0.8),
                            reason=action.reason or "Scroll the list to reveal more",
                        )
                    declared_target = " ".join(
                        re.findall(r"[a-z0-9]+", action.target.lower())
                    )
                    selected_target = " ".join(
                        re.findall(r"[a-z0-9]+", selected.label.lower())
                    )
                    if not declared_target:
                        if self.lenient:
                            # Trust the resolved candidate; backfill its label.
                            action.target = selected.label
                            declared_target = selected_target
                        else:
                            raise ValueError(
                                "candidate action must declare its exact target label"
                            )
                    if declared_target != selected_target:
                        if self.lenient and action.element_id in candidate_ids:
                            # A concrete id was chosen; trust it over the mislabel.
                            action.target = selected.label
                            declared_target = selected_target
                        elif use_visual_fallback and action.type in tap_like:
                            # OCR boxes are hints, not authority. If the model names a
                            # different visible control, discard its incorrect box ID
                            # and independently ground that semantic target.
                            action.x, action.y, grounding_confidence = (
                                self._ground_visual_target(
                                    image, action.target, reference_images
                                )
                            )
                            action.element_id = None
                            action.confidence = min(
                                action.confidence, grounding_confidence
                            )
                            return action
                        raise ValueError(
                            "declared target does not match the selected candidate label"
                        )
                    if not self._candidate_semantically_advances(action, decision_goal):
                        if self.lenient:
                            # Let the agent explore; loop-detection and independent
                            # verification still guard against unproductive taps.
                            pass
                        elif use_visual_fallback and action.type in tap_like:
                            intended_target = self._destination_name(decision_goal)
                            goal_terms = set(re.findall(r"[a-z0-9]+", intended_target))
                            reason_terms = set(
                                re.findall(r"[a-z0-9]+", action.reason.lower())
                            )
                            if intended_target and goal_terms & reason_terms:
                                action.x, action.y, grounding_confidence = (
                                    self._ground_visual_target(
                                        image, intended_target, reference_images
                                    )
                                )
                                action.element_id = None
                                action.target = intended_target.title()
                                action.confidence = min(
                                    action.confidence, grounding_confidence
                                )
                                return action
                        if not self.lenient:
                            raise ValueError(
                                "selected candidate is not semantically related to the "
                                "current subgoal"
                            )
                    navigation_goal = bool(
                        re.match(
                            r"^\s*(?:open|show|go\s+to|navigate\s+to|launch)\b",
                            decision_goal,
                            re.IGNORECASE,
                        )
                    )
                    if action.type in tap_like and navigation_goal and selected.stateful:
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

    def verify(
        self,
        goal: str,
        image: bytes,
        ui_dump: str = "",
        knowledge_context: dict[str, Any] | None = None,
        reference_images: list[bytes] | None = None,
    ) -> dict[str, Any]:
        visible_text = extract_visible_text(ui_dump)
        if not visible_text and self.enable_ocr:
            visible_text = [item.label for item in extract_ocr_elements(image)]
        prompt = (
            f"GOAL: {goal}\n"
            f"ANDROID-REPORTED VISIBLE TEXT: {json.dumps(visible_text)}\n"
            f"RETRIEVED DOCUMENTATION: {json.dumps(knowledge_context or {})}\n"
            "Return outcome, confidence, and specific visible evidence. Any text named "
            "as evidence must occur in the supplied visible-text list or visibly in IMAGE 1. "
            "IMAGE 1 is the live device; any later images are documentation references and "
            "cannot prove completion."
        )
        prepared = prepare_model_image(image, self.max_image_dimension)
        result = self._chat(
            VERIFIER_PROMPT,
            prompt,
            [prepared, *(reference_images or [])],
            VERIFICATION_SCHEMA,
        )
        if result.get("outcome") == "pass" and self.lenient:
            # Trust the model's visual verdict on the benchmark. The text-subset
            # heuristics below cannot verify state-change goals ("Turn wifi on":
            # the verb never appears on screen and "wifi" tokenizes as wi/fi), so
            # they would loop the agent on an already-completed task.
            return result
        if result.get("outcome") == "pass":
            navigation_goal = bool(
                re.match(
                    r"^\s*(?:display|go\s+to|launch|navigate\s+to|open|reach|show|view)\b",
                    goal,
                    re.IGNORECASE,
                )
            )
            titles = extract_screen_titles(ui_dump)
            if not titles and self.enable_ocr:
                titles = extract_ocr_screen_titles(image)
            destination = self._destination_name(goal)
            destination_terms = set(re.findall(r"[a-z0-9]+", destination))
            if len(destination_terms) > 1 and "settings" in destination_terms:
                destination_terms.remove("settings")
            matching_titles = [
                title
                for title in titles
                if destination_terms
                and destination_terms.issubset(
                    set(re.findall(r"[a-z0-9]+", title.lower()))
                )
            ]
            matching_state_controls = [
                element.label
                for element in extract_ui_elements(ui_dump)
                if element.stateful
                and destination_terms
                and destination_terms.issubset(
                    set(re.findall(r"[a-z0-9]+", element.label.lower()))
                )
            ]
            if navigation_goal and ui_dump.strip() and not (
                matching_titles or matching_state_controls
            ):
                return {
                    "outcome": "inconclusive",
                    "confidence": 0.0,
                    "evidence": (
                        "Completion claim rejected because the destination was not the "
                        "active screen title or a matching state control"
                    ),
                }
            stop_words = {
                "a", "an", "and", "app", "application", "as", "choose", "connect",
                "go", "launch", "make", "menu", "navigate", "open", "page",
                "screen", "select", "set", "settings", "show", "the", "to", "use",
            }
            aliases = {"bt": "bluetooth", "media": "audio", "sources": "source"}
            goal_terms = {
                aliases.get(term, term)
                for term in re.findall(r"[a-z0-9]+", goal.lower())
                if term not in stop_words and len(term) > 1
            }
            observed_terms = {
                aliases.get(term, term)
                for term in re.findall(r"[a-z0-9]+", " ".join(visible_text).lower())
            }
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
