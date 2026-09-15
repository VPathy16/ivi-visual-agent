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

    def test_direct_candidate_does_not_use_generic_settings_overlap(self) -> None:
        settings = UIElement(1, "Settings", "Button", "ivi", "", (0, 0, 10, 10), (0.1, 0.1), False)
        media = UIElement(2, "Media", "Button", "ivi", "", (10, 0, 20, 10), (0.2, 0.1), False)
        selected = OllamaVisionModel._direct_candidate(
            "Open the Bluetooth settings screen", [settings, media]
        )
        self.assertIsNone(selected)

    def test_direct_candidate_rejects_descriptive_toggle(self) -> None:
        toggle = UIElement(
            1,
            "Show Bluetooth devices without names",
            "Switch",
            "ivi",
            "",
            (0, 0, 10, 10),
            (0.1, 0.1),
            False,
            checkable=True,
        )
        self.assertIsNone(
            OllamaVisionModel._direct_candidate(
                "Open the Bluetooth settings screen", [toggle]
            )
        )

    def test_direct_candidate_rejects_other_settings_destination(self) -> None:
        notifications = UIElement(
            1,
            "Notification settings",
            "Button",
            "ivi",
            "",
            (0, 0, 10, 10),
            (0.1, 0.1),
            False,
        )
        self.assertIsNone(
            OllamaVisionModel._direct_candidate(
                "Open the Bluetooth settings screen", [notifications]
            )
        )


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


class StatefulRetryModel(OllamaVisionModel):
    def __init__(self) -> None:
        super().__init__("http://unused", "fake", enable_ocr=False)
        self.responses = [
            {
                "type": "tap",
                "element_id": "1",
                "confidence": 0.95,
                "reason": "Use Bluetooth control",
            },
            {"type": "home", "confidence": 0.9, "reason": "Leave unrelated control"},
        ]

    def _chat(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes | None,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        return self.responses.pop(0)


class CompactGestureModel(OllamaVisionModel):
    def __init__(self) -> None:
        super().__init__("http://unused", "fake", enable_ocr=False)

    def _chat(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes | None,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        return {"type": "gesture", "direction": "reveal_below", "region": "left"}


class CandidateTargetAliasModel(OllamaVisionModel):
    def __init__(self) -> None:
        super().__init__("http://unused", "fake", enable_ocr=False)

    def _chat(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes | None,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "type": "input_text",
            "element_id": 1,
            "candidate_target": "Search...",
            "text": "System",
            "confidence": 0.9,
            "reason": "Search for System",
        }


class GridGroundingModel(OllamaVisionModel):
    def __init__(self) -> None:
        super().__init__("http://unused", "fake")

    def _chat(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes | None,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        return {"found": True, "cell_number": 68}


class InvalidMilestoneModel(OllamaVisionModel):
    def __init__(self) -> None:
        super().__init__("http://unused", "fake")

    def _chat(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes | None,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "subgoals": [
                {"description": "The Settings icon is visible on the home screen"}
            ]
        }


class SpeculativeRouteModel(OllamaVisionModel):
    def __init__(self) -> None:
        super().__init__("http://unused", "fake")

    def _chat(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes | None,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "subgoals": [
                {"description": "Open the Settings application"},
                {"description": "Navigate to Network and Internet"},
                {"description": "Open the Notifications settings screen"},
            ]
        }


class AlwaysPassVerifierModel(OllamaVisionModel):
    def __init__(self, evidence: str) -> None:
        super().__init__("http://unused", "fake", enable_ocr=False)
        self.evidence = evidence

    def _chat(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes | None,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        return {"outcome": "pass", "confidence": 0.99, "evidence": self.evidence}


class PlannerRetryTests(unittest.TestCase):
    def test_retries_description_and_returns_action(self) -> None:
        image = io.BytesIO()
        Image.new("RGB", (4, 4), "black").save(image, format="PNG")
        action = FakeModel().plan("Open settings", image.getvalue(), "", [])
        self.assertEqual(action.type, "home")
        self.assertEqual(action.confidence, 0.9)

    def test_accepts_compact_constrained_gesture(self) -> None:
        image = io.BytesIO()
        Image.new("RGB", (100, 100), "black").save(image, format="PNG")
        action = CompactGestureModel().plan(
            "Open System settings", image.getvalue(), "", []
        )
        self.assertEqual(action.type, "gesture")
        self.assertEqual(action.direction, "reveal_below")
        self.assertEqual(action.confidence, 0.8)

    def test_accepts_candidate_target_alias_for_validated_input(self) -> None:
        image = io.BytesIO()
        Image.new("RGB", (100, 100), "black").save(image, format="PNG")
        ui_dump = """<hierarchy><node bounds='[0,0][100,100]'>
          <node text='Search…' class='android.widget.EditText' clickable='true'
                bounds='[10,10][90,30]' />
        </node></hierarchy>"""
        action = CandidateTargetAliasModel().plan(
            "Open System settings", image.getvalue(), ui_dump, []
        )
        self.assertEqual(action.type, "input_text")
        self.assertEqual(action.target, "Search…")
        self.assertEqual(action.text, "System")

    def test_maps_small_model_grid_alias_to_cell_center(self) -> None:
        image = io.BytesIO()
        Image.new("RGB", (120, 60), "black").save(image, format="PNG")
        x, y, confidence = GridGroundingModel()._ground_visual_target(
            image.getvalue(), "Settings"
        )
        self.assertAlmostEqual(x, 0.625)
        self.assertAlmostEqual(y, 11 / 12)
        self.assertEqual(confidence, 0.8)

    def test_invalid_control_visibility_plan_falls_back_to_user_goal(self) -> None:
        goal = "Open the Display settings screen"
        self.assertEqual(
            InvalidMilestoneModel().create_plan(goal),
            ["Open the Settings application", goal],
        )

    def test_speculative_middle_route_is_pruned(self) -> None:
        goal = "Open the Notifications settings screen"
        self.assertEqual(
            SpeculativeRouteModel().create_plan(goal),
            ["Open the Settings application", goal],
        )

    def test_rejects_candidate_when_reason_names_a_different_target(self) -> None:
        action = Action(
            type="tap",
            target="Local",
            confidence=0.95,
            reason="Settings icon is visible",
        )
        self.assertFalse(
            OllamaVisionModel._candidate_semantically_advances(
                action, "Open the Settings application"
            )
        )

    def test_accepts_semantic_parent_explained_by_reason(self) -> None:
        action = Action(
            type="tap",
            target="Connected devices",
            confidence=0.95,
            reason="Bluetooth settings are inside Connected devices",
        )
        self.assertTrue(
            OllamaVisionModel._candidate_semantically_advances(
                action, "Open the Bluetooth settings screen"
            )
        )

    def test_extracts_visual_destination_without_route_words(self) -> None:
        self.assertEqual(
            OllamaVisionModel._destination_name(
                "Reach the Settings application main screen"
            ),
            "settings",
        )

    def test_rejects_completion_not_anchored_to_screen_title(self) -> None:
        image = io.BytesIO()
        Image.new("RGB", (100, 100), "black").save(image, format="PNG")
        ui_dump = """<hierarchy><node bounds='[0,0][100,100]'>
          <node content-desc='Apps' clickable='true' bounds='[0,0][20,20]' />
        </node></hierarchy>"""
        result = AlwaysPassVerifierModel("The Settings gear icon is selected").verify(
            "Open the Apps settings screen", image.getvalue(), ui_dump
        )
        self.assertEqual(result["outcome"], "inconclusive")

    def test_accepts_completion_anchored_to_matching_visible_title(self) -> None:
        image = io.BytesIO()
        Image.new("RGB", (100, 100), "black").save(image, format="PNG")
        ui_dump = """<hierarchy><node bounds='[0,0][100,100]'>
          <node text='Apps' resource-id='ivi:id/toolbar_title'
                bounds='[0,0][50,20]' />
        </node></hierarchy>"""
        result = AlwaysPassVerifierModel("The Apps screen title is visible").verify(
            "Open the Apps settings screen", image.getvalue(), ui_dump
        )
        self.assertEqual(result["outcome"], "pass")

    def test_rejects_state_change_for_navigation_goal_and_replans(self) -> None:
        image = io.BytesIO()
        Image.new("RGB", (4, 4), "black").save(image, format="PNG")
        ui_dump = """<hierarchy><node bounds='[0,0][100,100]'>
          <node content-desc='Off, Bluetooth, Button' class='android.widget.LinearLayout'
                clickable='true' bounds='[10,10][90,40]' />
        </node></hierarchy>"""
        action = StatefulRetryModel().plan(
            "Open the Bluetooth settings screen", image.getvalue(), ui_dump, []
        )
        self.assertEqual(action.type, "home")


if __name__ == "__main__":
    unittest.main()
