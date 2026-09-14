from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Any

from .types import Action


PLANNER_PROMPT = """You control an Android automotive IVI on a stationary test bench.
Act like a careful first-time user trying to accomplish the supplied goal.
Choose exactly ONE next action based only on the screenshot, UI hierarchy, and history.

Allowed action JSON shapes:
{"type":"tap","x":0.0,"y":0.0,"target":"visible control","confidence":0.0,"reason":"..."}
{"type":"swipe","x":0.0,"y":0.0,"x2":0.0,"y2":0.0,"duration_ms":500,"target":"...","confidence":0.0,"reason":"..."}
{"type":"back","confidence":0.0,"reason":"..."}
{"type":"home","confidence":0.0,"reason":"..."}
{"type":"wait","seconds":1.0,"confidence":0.0,"reason":"..."}
{"type":"text","text":"...","confidence":0.0,"reason":"..."}
{"type":"finish","outcome":"pass|fail|inconclusive","evidence":"...","confidence":0.0,"reason":"..."}

Coordinates are normalized from 0.0 to 1.0 across the screenshot. Never guess an
invisible control. Do not delete data, place calls, purchase, factory-reset, update
software, or accept surprising permissions. Return only one JSON object, no markdown.
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
    def __init__(self, base_url: str, model: str, timeout: float = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

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

    def _chat(self, prompt: str, image: bytes) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(image).decode("ascii")],
                }
            ],
            "options": {"temperature": 0.1},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read())
        except urllib.error.URLError as exc:
            raise ModelError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc
        try:
            return self._extract_json(result["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise ModelError(f"Unexpected Ollama response: {result}") from exc

    def plan(self, goal: str, image: bytes, ui_dump: str, history: list[dict[str, Any]]) -> Action:
        context = {
            "goal": goal,
            "ui_hierarchy": ui_dump[:24000],
            "history": history[-8:],
        }
        response = self._chat(f"{PLANNER_PROMPT}\nCURRENT CONTEXT:\n{json.dumps(context)}", image)
        try:
            return Action.from_dict(response)
        except (TypeError, ValueError) as exc:
            raise ModelError(f"Invalid action response: {response}") from exc

    def verify(self, goal: str, image: bytes) -> dict[str, Any]:
        return self._chat(f"{VERIFIER_PROMPT}\nGOAL: {goal}", image)

