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
    extract_ui_elements,
    packages_in_ui,
    prepare_model_image,
)
from .types import Action


ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": ["tap", "gesture", "back", "home", "wait", "text", "finish"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "target": {"type": "string"},
        "element_id": {"type": "integer", "minimum": 1},
        "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
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


PLANNER_PROMPT = """You are a goal-directed controller for an arbitrary Android IVI on
a stationary test bench. Choose exactly ONE safe action that visibly advances the USER
GOAL. Do not assume a phone launcher, app drawer, menu name, screen layout, navigation
gesture, or coordinate convention beyond what the current screenshot and UI elements
show. Ignore unrelated content. Use Home or Back only when the observed state and
history justify it. Never repeat an action that history says made no progress.

When numbered UI candidates are supplied, tap only by returning their element_id; never
invent coordinates. When no candidates exist, a visual tap requires normalized x and y.
For scrolling or paging return type "gesture", direction up/down/left/right, and region
center/top/bottom/left/right. A finish action requires an outcome. Never guess an
invisible control, delete data, place calls, purchase, reset, update software, or accept
surprising permissions. Return only action JSON.
"""


VERIFIER_PROMPT = """Independently verify whether the goal is visibly complete in this
Android automotive IVI screenshot. Return only JSON:
{"outcome":"pass|fail|inconclusive","confidence":0.0,"evidence":"specific visible evidence"}
Use pass only when the screenshot clearly proves completion. Use fail only for a clear
error or contradiction. Otherwise use inconclusive. Do not rely on the planner's claim.
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
    def _validate_action_shape(action: Action, candidate_ids: set[int] | None = None) -> None:
        candidate_ids = candidate_ids or set()
        if action.type == "tap":
            if candidate_ids and action.element_id not in candidate_ids:
                raise ValueError("tap must select a valid UI candidate element_id")
            if not candidate_ids and (action.x is None or action.y is None):
                raise ValueError("visual tap is missing x or y")
        if action.type == "gesture":
            if action.direction not in {"up", "down", "left", "right"}:
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
        words = set(re.findall(r"[a-z0-9]+", goal.lower()))
        stop_words = {
            "a",
            "an",
            "and",
            "at",
            "in",
            "is",
            "of",
            "on",
            "open",
            "screen",
            "show",
            "the",
            "to",
        }
        goal_words = words - stop_words
        if not goal_words:
            return None

        def score(element: Any, ignored: set[str]) -> int:
            label_words = set(re.findall(r"[a-z0-9]+", element.label.lower()))
            return sum(len(word) for word in label_words & (goal_words - ignored))

        # Prefer the specific feature over a generic container such as Settings.
        ranked = sorted(elements, key=lambda item: score(item, {"settings"}), reverse=True)
        if ranked and score(ranked[0], {"settings"}) >= 4:
            return ranked[0]
        ranked = sorted(elements, key=lambda item: score(item, set()), reverse=True)
        if ranked and score(ranked[0], set()) >= 4:
            return ranked[0]
        return None

    def plan(self, goal: str, image: bytes, ui_dump: str, history: list[dict[str, Any]]) -> Action:
        elements = extract_ui_elements(ui_dump)
        direct = self._direct_candidate(goal, elements)
        if direct is None and self.enable_ocr:
            ocr_elements = extract_ocr_elements(image, start_id=len(elements) + 1)
            existing = {(item.label.lower(), item.bounds) for item in elements}
            elements.extend(
                item
                for item in ocr_elements
                if (item.label.lower(), item.bounds) not in existing
            )
            direct = self._direct_candidate(goal, elements)
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
            "visible_packages": self._packages(ui_dump),
            "ui_candidates": [element.to_prompt_dict() for element in elements],
            "history": history[-5:],
        }
        has_ui_tree_candidates = any(item.source == "ui_tree" for item in elements)
        use_visual_fallback = not has_ui_tree_candidates or not self.prefer_ui_tree
        model_image = (
            prepare_model_image(image, self.max_image_dimension)
            if use_visual_fallback
            else None
        )
        mode = (
            "No usable UI candidates were exposed; use the attached resized screenshot."
            if use_visual_fallback
            else "Use the numbered UI candidates. No screenshot is attached in this fast path."
        )
        prompt = (
            f"USER GOAL: {goal}\n"
            f"{mode} Choose one action that advances this exact goal. "
            "Do not tap unrelated content.\n"
            "Required JSON keys are type, confidence, and reason. For a candidate tap "
            "also return element_id. For a gesture also return direction and region. "
            "Use the key type, never action.\n"
            f"CONTEXT: {json.dumps(context)}"
        )
        invalid: dict[str, Any] | None = None
        validation_error = ""
        for attempt in range(2):
            retry_note = ""
            if attempt and invalid is not None:
                retry_note = (
                    "\nCorrect the previous JSON without reconsidering the UI. "
                    f"Validation error: {validation_error}. Required: type, confidence, reason; "
                    "tap also needs element_id (or x/y only in visual fallback); gesture also "
                    "needs direction and region. Previous response: "
                    f"{json.dumps(invalid)[:1200]}"
                )
            response = self._chat(
                PLANNER_PROMPT,
                prompt + retry_note,
                model_image if attempt == 0 else None,
                ACTION_SCHEMA,
            )
            try:
                action = Action.from_dict(response)
                if action.type not in {"tap", "gesture", "back", "home", "wait", "text", "finish"}:
                    raise ValueError("unknown action type")
                self._validate_action_shape(action, candidate_ids)
                if action.type == "tap" and action.element_id is not None:
                    selected = next(item for item in elements if item.id == action.element_id)
                    action.x, action.y = selected.center
                    action.target = selected.label
                return action
            except (TypeError, ValueError) as exc:
                invalid = response
                validation_error = str(exc)
        raise ModelError(f"Model failed to return a valid action after retry: {invalid}")

    def verify(self, goal: str, image: bytes) -> dict[str, Any]:
        prompt = (
            f"GOAL: {goal}\nReturn outcome, confidence, and specific visible evidence."
        )
        prepared = prepare_model_image(image, self.max_image_dimension)
        return self._chat(VERIFIER_PROMPT, prompt, prepared, VERIFICATION_SCHEMA)
