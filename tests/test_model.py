import unittest
import io

from typing import Any

from PIL import Image

from ivi_agent.model import ACTION_SCHEMA, OllamaVisionModel
from ivi_agent.perception import UIElement
from ivi_agent.types import Action


class ModelResponseTests(unittest.TestCase):
    def test_extracts_json_from_plain_response(self) -> None:
        value = OllamaVisionModel._extract_json(
            '{"type":"back","confidence":0.9,"reason":"go back"}'
        )
        self.assertEqual(value["type"], "back")

    def test_extracts_json_from_code_fence(self) -> None:
        value = OllamaVisionModel._extract_json('```json\n{"outcome":"pass"}\n```')
        self.assertEqual(value["outcome"], "pass")

    def test_action_schema_requires_action_fields(self) -> None:
        self.assertEqual(ACTION_SCHEMA["required"], ["type", "confidence", "reason"])
        self.assertFalse(ACTION_SCHEMA["additionalProperties"])

    def test_compacts_ui_dump_to_meaningful_elements(self) -> None:
        source = """<?xml version='1.0'?><hierarchy>
          <node class='android.widget.FrameLayout' clickable='false'>
            <node text='Settings' resource-id='com.android:id/settings' clickable='true'
                  bounds='[10,20][110,120]' />
            <node class='android.view.View' clickable='false' />
          </node>
        </hierarchy>"""
        elements = OllamaVisionModel._compact_ui_dump(source)
        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0]["text"], "Settings")
        self.assertEqual(elements[0]["bounds"], "[10,20][110,120]")

    def test_extracts_unique_packages(self) -> None:
        source = """<hierarchy>
          <node package='com.android.systemui'><node package='com.example.launcher'/></node>
          <node package='com.example.launcher'/>
        </hierarchy>"""
        self.assertEqual(
            OllamaVisionModel._packages(source),
            ["com.android.systemui", "com.example.launcher"],
        )

    def test_rejects_swipe_without_coordinates(self) -> None:
        with self.assertRaisesRegex(ValueError, "swipe is missing"):
            OllamaVisionModel._validate_action_shape(
                Action(type="swipe", confidence=0.9, reason="open drawer")
            )

    def test_accepts_complete_swipe(self) -> None:
        OllamaVisionModel._validate_action_shape(
            Action(
                type="swipe",
                x=0.5,
                y=0.85,
                x2=0.5,
                y2=0.25,
                confidence=0.9,
                reason="open drawer",
            )
        )

    def test_direct_candidate_prefers_specific_goal_term(self) -> None:
        settings = UIElement(1, "Settings", "Button", "ivi", "", (0, 0, 10, 10), (0.1, 0.1), False)
        bluetooth = UIElement(2, "Bluetooth", "Button", "ivi", "", (10, 0, 20, 10), (0.2, 0.1), False)
        selected = OllamaVisionModel._direct_candidate(
            "Open the Bluetooth settings screen", [settings, bluetooth]
        )
        self.assertEqual(selected.id, 2)

    def test_direct_candidate_falls_back_to_settings(self) -> None:
        settings = UIElement(1, "Settings", "Button", "ivi", "", (0, 0, 10, 10), (0.1, 0.1), False)
        media = UIElement(2, "Media", "Button", "ivi", "", (10, 0, 20, 10), (0.2, 0.1), False)
        selected = OllamaVisionModel._direct_candidate(
            "Open the Bluetooth settings screen", [settings, media]
        )
        self.assertEqual(selected.id, 1)


class FakeModel(OllamaVisionModel):
    def __init__(self) -> None:
        super().__init__("http://unused", "fake")
        self.responses = [
            {"description": "only described the image"},
            {"type": "home", "confidence": 0.9, "reason": "wake the device"},
        ]

    def _chat(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        return self.responses.pop(0)


class PlannerRetryTests(unittest.TestCase):
    def test_retries_description_and_returns_action(self) -> None:
        image = io.BytesIO()
        Image.new("RGB", (4, 4), "black").save(image, format="PNG")
        action = FakeModel().plan("Open settings", image.getvalue(), "", [])
        self.assertEqual(action.type, "home")
        self.assertEqual(action.confidence, 0.9)


if __name__ == "__main__":
    unittest.main()
