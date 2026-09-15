import io
import unittest

from PIL import Image

from ivi_agent.adb import AdbDevice, AdbError
from ivi_agent.config import Config
from ivi_agent.model import ACTION_SCHEMA, OllamaVisionModel
from ivi_agent.policy import PolicyViolation, validate_action
from ivi_agent.types import Action


class RecordingDevice(AdbDevice):
    def __init__(self) -> None:
        super().__init__(serial="emulator-5554")
        self.calls: list[tuple[str, ...]] = []

    def _run(self, *args, binary=False, timeout=20):  # type: ignore[override]
        self.calls.append(args)
        return b"" if binary else ""


class ExpandedPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config()

    def test_accepts_long_press_and_double_tap(self) -> None:
        for kind in ("long_press", "double_tap"):
            validate_action(
                Action(type=kind, x=0.4, y=0.6, confidence=0.9, reason="visible"),
                self.config,
            )

    def test_accepts_keyboard_enter(self) -> None:
        validate_action(
            Action(type="keyboard_enter", confidence=0.9, reason="submit"), self.config
        )

    def test_open_app_requires_app_name(self) -> None:
        with self.assertRaisesRegex(PolicyViolation, "app_name"):
            validate_action(
                Action(type="open_app", confidence=0.9, reason="launch"), self.config
            )
        validate_action(
            Action(type="open_app", app_name="Chrome", confidence=0.9, reason="launch"),
            self.config,
        )

    def test_long_press_needs_normalized_coordinates(self) -> None:
        with self.assertRaisesRegex(PolicyViolation, "normalized"):
            validate_action(
                Action(type="long_press", x=1.5, y=0.5, confidence=0.9, reason="x"),
                self.config,
            )

    def test_destructive_guard_applies_to_long_press(self) -> None:
        with self.assertRaisesRegex(PolicyViolation, "blocked"):
            validate_action(
                Action(
                    type="long_press",
                    x=0.5,
                    y=0.5,
                    target="Uninstall app",
                    confidence=0.9,
                    reason="x",
                ),
                self.config,
            )


class AdbExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = RecordingDevice()
        self.size = (1000, 2000)

    def test_double_tap_issues_two_taps(self) -> None:
        self.device.execute(
            Action(type="double_tap", x=0.5, y=0.5, confidence=0.9, reason="x"), self.size
        )
        taps = [c for c in self.device.calls if c[:3] == ("shell", "input", "tap")]
        self.assertEqual(len(taps), 2)

    def test_long_press_uses_same_point_swipe(self) -> None:
        self.device.execute(
            Action(type="long_press", x=0.5, y=0.5, duration_ms=800, confidence=0.9, reason="x"),
            self.size,
        )
        swipe = next(c for c in self.device.calls if c[:3] == ("shell", "input", "swipe"))
        # start == end point, long duration
        self.assertEqual(swipe[3], swipe[5])
        self.assertEqual(swipe[4], swipe[6])
        self.assertEqual(swipe[7], "800")

    def test_keyboard_enter_sends_enter_keyevent(self) -> None:
        self.device.execute(
            Action(type="keyboard_enter", confidence=0.9, reason="x"), self.size
        )
        self.assertIn(("shell", "input", "keyevent", "KEYCODE_ENTER"), self.device.calls)

    def test_open_app_rejects_non_package_name(self) -> None:
        with self.assertRaisesRegex(AdbError, "package"):
            self.device.execute(
                Action(type="open_app", app_name="Chrome", confidence=0.9, reason="x"),
                self.size,
            )

    def test_open_app_launches_package(self) -> None:
        self.device.execute(
            Action(
                type="open_app",
                app_name="com.android.settings",
                confidence=0.9,
                reason="x",
            ),
            self.size,
        )
        self.assertTrue(
            any(c[:2] == ("shell", "monkey") for c in self.device.calls)
        )

    def test_input_text_escaping_handles_special_characters(self) -> None:
        escaped = AdbDevice._escape_input_text('a b&c(d)"e"')
        self.assertNotIn(" ", escaped)  # spaces become %s
        self.assertIn("%s", escaped)
        self.assertIn("\\&", escaped)
        self.assertIn("\\(", escaped)


class ModelSchemaTests(unittest.TestCase):
    def test_action_schema_includes_new_types(self) -> None:
        enum = set(ACTION_SCHEMA["properties"]["type"]["enum"])
        for kind in ("long_press", "double_tap", "keyboard_enter", "open_app"):
            self.assertIn(kind, enum)

    def test_grounding_mode_validation(self) -> None:
        with self.assertRaises(ValueError):
            OllamaVisionModel("http://x", "m", grounding_mode="bogus")


class PointGroundingModel(OllamaVisionModel):
    def __init__(self) -> None:
        super().__init__("http://localhost", "qwen3-vl:8b-instruct", grounding_mode="point")

    def _chat(self, system_prompt, user_prompt, image, schema):  # type: ignore[override]
        return {"found": True, "x": 0.42, "y": 0.66, "confidence": 0.9}


class ThousandScalePointModel(PointGroundingModel):
    def _chat(self, system_prompt, user_prompt, image, schema):  # type: ignore[override]
        return {"found": True, "x": 420, "y": 660, "confidence": 0.9}


def _blank_png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class ActionTypeAliasModel(OllamaVisionModel):
    def __init__(self) -> None:
        super().__init__("http://localhost", "m")

    def _chat(self, system_prompt, user_prompt, image, schema):  # type: ignore[override]
        # Mimics a small model that mislabels the action type as "action".
        return {
            "type": "action",
            "element_id": 1,
            "target": "Wifi",
            "confidence": 0.9,
            "reason": "tap wifi",
        }


class ActionTypeCoercionTests(unittest.TestCase):
    def test_placeholder_action_type_is_coerced_to_tap(self) -> None:
        ui = (
            '<hierarchy rotation="0">'
            '<node class="android.widget.FrameLayout" bounds="[0,0][100,200]">'
            '<node text="Wifi" class="android.widget.TextView" clickable="true" '
            'bounds="[10,10][90,40]"/>'
            "</node></hierarchy>"
        )
        action = ActionTypeAliasModel().plan("Turn wifi on", _blank_png(), ui, [])
        self.assertEqual(action.type, "tap")
        self.assertEqual(action.element_id, 1)


class LenientNoTargetModel(OllamaVisionModel):
    def __init__(self, lenient: bool) -> None:
        super().__init__("http://localhost", "m", lenient=lenient)

    def _chat(self, system_prompt, user_prompt, image, schema):  # type: ignore[override]
        # Valid element_id but NO target label and an unrelated reason.
        return {"type": "tap", "element_id": 1, "confidence": 0.9, "reason": "go"}


class LenientPlanningTests(unittest.TestCase):
    UI = (
        '<hierarchy rotation="0">'
        '<node class="android.widget.FrameLayout" bounds="[0,0][100,200]">'
        '<node text="Settings" class="android.widget.TextView" clickable="true" '
        'bounds="[10,10][90,40]"/>'
        "</node></hierarchy>"
    )

    def test_strict_mode_rejects_missing_target(self) -> None:
        from ivi_agent.model import ModelError

        with self.assertRaises(ModelError):
            LenientNoTargetModel(lenient=False).plan("Turn wifi on", _blank_png(), self.UI, [])

    def test_lenient_mode_backfills_target_and_taps(self) -> None:
        action = LenientNoTargetModel(lenient=True).plan(
            "Turn wifi on", _blank_png(), self.UI, []
        )
        self.assertEqual(action.type, "tap")
        self.assertEqual(action.element_id, 1)
        self.assertEqual(action.target, "Settings")


class PointGroundingTests(unittest.TestCase):
    def test_point_mode_returns_normalized_coordinates(self) -> None:
        x, y, confidence = PointGroundingModel()._ground_visual_target(_blank_png(), "OK")
        self.assertAlmostEqual(x, 0.42)
        self.assertAlmostEqual(y, 0.66)
        self.assertAlmostEqual(confidence, 0.9)

    def test_point_mode_rescales_thousand_based_coordinates(self) -> None:
        x, y, _ = ThousandScalePointModel()._ground_visual_target(_blank_png(), "OK")
        self.assertAlmostEqual(x, 0.42)
        self.assertAlmostEqual(y, 0.66)


if __name__ == "__main__":
    unittest.main()
