import io
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from PIL import Image

from ivi_agent.adb import AdbDevice, AdbError
from ivi_agent.agent import GoalAgent
from ivi_agent.config import Config
from ivi_agent.model import ACTION_SCHEMA, GROUNDING_SCHEMA, OllamaVisionModel
from ivi_agent.policy import PolicyViolation, validate_action
from ivi_agent.types import Action


def _small_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (48, 96), "black").save(buf, format="PNG")
    return buf.getvalue()


class _FinishingModel:
    """A model that plans one subgoal and never needs to plan/verify, because the
    run finishes via a title match on the first observation."""

    enable_ocr = False

    def create_plan(self, goal, knowledge_context=None):
        return [goal]

    def plan(self, *a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("model.plan should not be called")

    def verify(self, *a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("model.verify should not be called")


class _CleanStartDevice(AdbDevice):
    """Fake device whose Home screen satisfies the goal immediately."""

    def __init__(self) -> None:
        super().__init__(serial="emulator-5554")
        self.relaunched: list[str] = []

    def ensure_ready(self) -> None:
        pass

    def wake_if_needed(self) -> None:
        pass

    def clear_logcat(self) -> None:
        pass

    def logcat_dump(self, tail_lines: int = 4000) -> str:
        return ""

    def relaunch(self, package: str) -> None:
        self.relaunched.append(package)

    def screen_size(self):  # type: ignore[override]
        return (1080, 2400)

    def capture(self, destination: Path) -> bytes:
        data = _small_png()
        Path(destination).write_bytes(data)
        return data

    def ui_dump(self) -> str:
        return (
            "<hierarchy><node bounds='[0,0][1080,2400]'>"
            "<node text='Home' bounds='[40,240][300,320]'/>"
            "</node></hierarchy>"
        )


class _FinishActionModel:
    """Plans a finish action; verify() must never be called when level=off."""

    enable_ocr = False

    def create_plan(self, goal, knowledge_context=None):
        return [goal]

    def plan(self, *a, **k):
        return Action(
            type="finish", confidence=0.95, reason="believed complete",
            outcome="pass", evidence="all steps done",
        )

    def verify(self, *a, **k):  # pragma: no cover - asserts it is not reached
        raise AssertionError("model.verify must not be called at verification_level=off")


class _PlainDevice(_CleanStartDevice):
    def ui_dump(self) -> str:
        # A title that does NOT satisfy the goal, so the run proceeds to decide
        # an action (rather than completing via a title match).
        return (
            "<hierarchy><node bounds='[0,0][1080,2400]'>"
            "<node text='Some screen' bounds='[40,240][300,320]'/>"
            "</node></hierarchy>"
        )


class VerificationOffRunTests(unittest.TestCase):
    def test_off_completes_finish_without_model_verify(self) -> None:
        device = _PlainDevice()
        config = Config()
        config.verification_level = "off"
        config.scene_graph = False
        config.trace = False
        agent = GoalAgent(device, _FinishActionModel(), config, knowledge=None)
        with tempfile.TemporaryDirectory() as out:
            result = agent.run("Complete the flow", Path(out))
        self.assertEqual(result.outcome, "pass")  # finished, no verify raised


class _RecoveringModel:
    """First plans an un-executable open_app, then a valid finish."""

    enable_ocr = False

    def __init__(self) -> None:
        self.plan_calls = 0

    def create_plan(self, goal, knowledge_context=None):
        return [goal]

    def plan(self, *a, **k):
        self.plan_calls += 1
        if self.plan_calls == 1:
            return Action(type="open_app", app_name="Vehicle Settings",
                          confidence=0.9, reason="go to settings first")
        return Action(type="finish", confidence=0.95, reason="done",
                      outcome="pass", evidence="complete")

    def verify(self, *a, **k):  # pragma: no cover
        raise AssertionError("verify not expected at verification_level=off")


class _ExecFailDevice(_PlainDevice):
    def execute(self, action, size):  # type: ignore[override]
        if action.type == "open_app":
            raise AdbError("Cannot resolve app name to a package: 'Vehicle Settings'.")
        # other actions are no-ops for the fake


class ActionFailureRecoveryTests(unittest.TestCase):
    def test_unexecutable_action_is_recoverable_not_fatal(self) -> None:
        device = _ExecFailDevice()
        model = _RecoveringModel()
        config = Config()
        config.verification_level = "off"
        config.scene_graph = False
        config.trace = False
        agent = GoalAgent(device, model, config, knowledge=None)
        with tempfile.TemporaryDirectory() as out:
            result = agent.run("Start the seat massage", Path(out))
        # The failed open_app did not abort the run; it recovered and finished.
        self.assertEqual(result.outcome, "pass")
        self.assertEqual(result.steps[0].action.type, "open_app")
        self.assertIn("Cannot resolve app name", result.steps[0].error or "")
        self.assertGreaterEqual(model.plan_calls, 2)  # it replanned


class CleanStartRunTests(unittest.TestCase):
    def _run(self, relaunch: bool):
        device = _CleanStartDevice()
        config = Config()
        config.relaunch_before_run = relaunch
        config.target_package = "com.example.iviwv"
        config.scene_graph = False
        config.trace = False
        agent = GoalAgent(device, _FinishingModel(), config, knowledge=None)
        with tempfile.TemporaryDirectory() as out:
            result = agent.run("Reach the Home screen", Path(out))
        return device, result

    def test_relaunches_when_enabled(self) -> None:
        device, result = self._run(relaunch=True)
        self.assertEqual(device.relaunched, ["com.example.iviwv"])
        self.assertEqual(result.outcome, "pass")

    def test_no_relaunch_when_disabled(self) -> None:
        device, result = self._run(relaunch=False)
        self.assertEqual(device.relaunched, [])
        self.assertEqual(result.outcome, "pass")


class RecordingDevice(AdbDevice):
    def __init__(self) -> None:
        super().__init__(serial="emulator-5554")
        self.calls: list[tuple[str, ...]] = []

    def _run(self, *args, binary=False, timeout=20):  # type: ignore[override]
        self.calls.append(args)
        return b"" if binary else ""


class CleanStartTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = RecordingDevice()
        self._sleep = mock.patch("ivi_agent.adb.time.sleep", return_value=None)
        self._sleep.start()

    def tearDown(self) -> None:
        self._sleep.stop()

    def test_force_stop_issues_am_force_stop(self) -> None:
        self.device.force_stop("com.example.iviwv")
        self.assertIn(("shell", "am", "force-stop", "com.example.iviwv"), self.device.calls)

    def test_relaunch_stops_then_launches_in_order(self) -> None:
        self.device.relaunch("com.example.iviwv")
        stop_idx = self.device.calls.index(("shell", "am", "force-stop", "com.example.iviwv"))
        launch_idx = next(
            i for i, c in enumerate(self.device.calls)
            if c[:3] == ("shell", "monkey", "-p")
        )
        self.assertLess(stop_idx, launch_idx)  # cold: stop before launch

    def test_relaunch_rejects_bad_package(self) -> None:
        for bad in ("not a package", "single", ""):
            with self.assertRaises(AdbError):
                self.device.relaunch(bad)
        # A bad package must not have issued any force-stop.
        self.assertFalse(any(c[:3] == ("shell", "am", "force-stop") for c in self.device.calls))


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


class FakeScreencapDevice(AdbDevice):
    def __init__(self, png: bytes) -> None:
        super().__init__(serial="emulator-5554")
        self._png = png

    def _run(self, *args, binary=False, timeout=20):  # type: ignore[override]
        if args[:1] == ("exec-out",) and "screencap" in args:
            return self._png
        if args[:2] == ("shell", "wm"):
            return "Physical size: 1080x2400\n"
        return b"" if binary else ""


class ScreenSizeTests(unittest.TestCase):
    def test_screen_size_uses_current_framebuffer_not_wm_size(self) -> None:
        buf = io.BytesIO()
        Image.new("RGB", (2400, 1080), "black").save(buf, format="PNG")
        device = FakeScreencapDevice(buf.getvalue())
        # Landscape framebuffer, NOT the transposed 1080x2400 that `wm size` reports.
        self.assertEqual(device.screen_size(), (2400, 1080))


class _DumpsysDevice(AdbDevice):
    DUMP = (
        "Display 4619827259835644672 (HWC display 0): port=0 ...\n"
        "Display 4619827551948147201 (HWC display 1): port=1 ...\n"
    )

    def _run(self, *args, binary=False, timeout=20):  # type: ignore[override]
        if args[:2] == ("shell", "dumpsys"):
            return self.DUMP
        return b"" if binary else ""


class CaptureDisplayIdTests(unittest.TestCase):
    def test_hwc_index_maps_to_physical_id(self) -> None:
        # --display-id 0 is an HWC index; screencap needs the physical id.
        self.assertEqual(_DumpsysDevice(display_id=0).capture_display_id(), 4619827259835644672)
        self.assertEqual(_DumpsysDevice(display_id=1).capture_display_id(), 4619827551948147201)

    def test_physical_id_passed_through(self) -> None:
        self.assertEqual(
            _DumpsysDevice(display_id=4619827259835644672).capture_display_id(),
            4619827259835644672,
        )

    def test_default_picks_primary(self) -> None:
        self.assertEqual(_DumpsysDevice().capture_display_id(), 4619827259835644672)


class _MultiDisplayDevice(AdbDevice):
    """Mimics a multi-display AVD: plain `screencap` hangs (times out to an
    AdbError), only the targeted `-d <physical id>` capture returns a PNG."""

    DUMP = "Display 4619827259835644672 (HWC display 0): port=0 ...\n"

    def __init__(self, png: bytes) -> None:
        super().__init__(serial="emulator-5554")
        self._png = png
        self.plain_screencaps = 0

    def _run(self, *args, binary=False, timeout=20):  # type: ignore[override]
        if args[:2] == ("shell", "dumpsys"):
            return self.DUMP
        if args[:1] == ("exec-out",) and "screencap" in args:
            if "-d" in args:
                return self._png
            self.plain_screencaps += 1
            raise AdbError("ADB command timed out")
        if args[:2] == ("shell", "wm"):
            return "Physical size: 1080x2400\n"
        return b"" if binary else ""


class TargetedCaptureTests(unittest.TestCase):
    def _png(self) -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (2400, 1080), "black").save(buf, format="PNG")
        return buf.getvalue()

    def test_screen_size_uses_targeted_display_without_touching_plain(self) -> None:
        device = _MultiDisplayDevice(self._png())
        # Resolves via `-d`, so it must never fall to the hanging plain screencap.
        self.assertEqual(device.screen_size(), (2400, 1080))
        self.assertEqual(device.plain_screencaps, 0)

    def test_capture_uses_targeted_display_without_touching_plain(self) -> None:
        device = _MultiDisplayDevice(self._png())
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            device.capture(Path(tmp) / "shot.png")
        self.assertEqual(device.plain_screencaps, 0)


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


class NumCtxPayloadTests(unittest.TestCase):
    def test_num_ctx_is_sent_in_options(self) -> None:
        captured = {}

        @contextmanager
        def fake_urlopen(request, timeout=None):
            captured["data"] = json.loads(request.data.decode("utf-8"))

            class _Resp:
                def read(self_inner):
                    return json.dumps({"message": {"content": "{}"}}).encode("utf-8")

            yield _Resp()

        model = OllamaVisionModel("http://localhost", "m", num_ctx=16384)
        with mock.patch("ivi_agent.model.urllib.request.urlopen", fake_urlopen):
            model._chat("sys", "user", None, {})
        self.assertEqual(captured["data"]["options"]["num_ctx"], 16384)


class ModelSchemaTests(unittest.TestCase):
    def test_action_schema_includes_new_types(self) -> None:
        enum = set(ACTION_SCHEMA["properties"]["type"]["enum"])
        for kind in ("long_press", "double_tap", "keyboard_enter", "open_app", "swipe"):
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


class StringElementIdModel(OllamaVisionModel):
    def __init__(self) -> None:
        super().__init__("http://localhost", "m", lenient=True)  # grid grounding

    def _chat(self, system_prompt, user_prompt, image, schema):  # type: ignore[override]
        if schema is GROUNDING_SCHEMA:
            return {"found": True, "cell": 1, "confidence": 0.9, "evidence": "x"}
        return {
            "type": "tap",
            "element_id": "3G",  # a label, not a candidate number
            "target": "",
            "confidence": 0.9,
            "reason": "tap 3g",
        }


class StringElementIdTests(unittest.TestCase):
    def test_non_numeric_element_id_becomes_visual_target(self) -> None:
        action = StringElementIdModel().plan("Turn wifi on", _blank_png(), "", [])
        self.assertEqual(action.type, "tap")
        self.assertIsNone(action.element_id)
        self.assertIsNotNone(action.x)
        self.assertIsNotNone(action.y)


class NamedTargetModel(OllamaVisionModel):
    def __init__(self) -> None:
        super().__init__("http://localhost", "m", lenient=True)

    def _chat(self, system_prompt, user_prompt, image, schema):  # type: ignore[override]
        # Names the control, gives no element_id and no coordinates.
        return {
            "type": "tap",
            "target": "Network & internet",
            "confidence": 0.9,
            "reason": "open network settings",
        }


class NamedTargetResolutionTests(unittest.TestCase):
    def test_named_target_resolves_to_accessibility_candidate(self) -> None:
        ui = (
            '<hierarchy rotation="0">'
            '<node class="android.widget.FrameLayout" bounds="[0,0][100,200]">'
            '<node text="Network &amp; internet" class="android.widget.TextView" '
            'clickable="true" bounds="[10,10][90,40]"/>'
            "</node></hierarchy>"
        )
        action = NamedTargetModel().plan("Turn wifi on", _blank_png(), ui, [])
        self.assertEqual(action.type, "tap")
        self.assertEqual(action.element_id, 1)
        self.assertIsNotNone(action.x)
        self.assertIsNotNone(action.y)


class UngroundedNamedTargetModel(OllamaVisionModel):
    def __init__(self) -> None:
        super().__init__("http://localhost", "m", lenient=True)  # grid grounding

    def _chat(self, system_prompt, user_prompt, image, schema):  # type: ignore[override]
        if schema is GROUNDING_SCHEMA:
            return {"found": True, "cell": 5, "confidence": 0.9, "evidence": "x"}
        return {"type": "tap", "target": "Wi-Fi", "confidence": 0.9, "reason": "tap wifi"}


class NoConfidenceModel(OllamaVisionModel):
    def __init__(self) -> None:
        super().__init__("http://localhost", "m")

    def _chat(self, system_prompt, user_prompt, image, schema):  # type: ignore[override]
        # A tap that omits the required confidence and reason fields.
        return {"type": "tap", "element_id": 1, "target": "Wifi"}


class MissingFieldDefaultTests(unittest.TestCase):
    def test_tap_without_confidence_is_defaulted_not_crashed(self) -> None:
        ui = (
            '<hierarchy rotation="0">'
            '<node class="android.widget.FrameLayout" bounds="[0,0][100,200]">'
            '<node text="Wifi" class="android.widget.TextView" clickable="true" '
            'bounds="[10,10][90,40]"/>'
            "</node></hierarchy>"
        )
        action = NoConfidenceModel().plan("Turn wifi on", _blank_png(), ui, [])
        self.assertEqual(action.type, "tap")
        self.assertGreaterEqual(action.confidence, 0.75)


class LenientVisualGroundingTests(unittest.TestCase):
    def test_named_target_not_a_candidate_is_visually_grounded(self) -> None:
        # The only accessibility candidate is Bluetooth, so "Wi-Fi" cannot match a
        # candidate; lenient mode must fall back to visual grounding.
        ui = (
            '<hierarchy rotation="0">'
            '<node class="android.widget.FrameLayout" bounds="[0,0][100,200]">'
            '<node text="Bluetooth" class="android.widget.TextView" clickable="true" '
            'bounds="[10,10][90,40]"/>'
            "</node></hierarchy>"
        )
        action = UngroundedNamedTargetModel().plan("Turn wifi on", _blank_png(), ui, [])
        self.assertEqual(action.type, "tap")
        self.assertIsNone(action.element_id)
        self.assertIsNotNone(action.x)
        self.assertIsNotNone(action.y)


class _VerifyPassModel(OllamaVisionModel):
    def __init__(self, lenient: bool) -> None:
        super().__init__("http://localhost", "m", lenient=lenient, enable_ocr=False)

    def _chat(self, system_prompt, user_prompt, image, schema):  # type: ignore[override]
        return {"outcome": "pass", "confidence": 0.95, "evidence": "blue Wi-Fi toggle is on"}


class StateChangeVerifyTests(unittest.TestCase):
    def test_lenient_verify_trusts_model_pass_for_state_change_goal(self) -> None:
        result = _VerifyPassModel(lenient=True).verify("Turn wifi on", _blank_png(), "")
        self.assertEqual(result["outcome"], "pass")

    def test_strict_verify_downgrades_when_terms_not_on_screen(self) -> None:
        result = _VerifyPassModel(lenient=False).verify("Turn wifi on", _blank_png(), "")
        self.assertEqual(result["outcome"], "inconclusive")


class ScrollTapModel(OllamaVisionModel):
    def __init__(self) -> None:
        super().__init__("http://localhost", "m", lenient=True)

    def _chat(self, system_prompt, user_prompt, image, schema):  # type: ignore[override]
        return {
            "type": "tap",
            "target": "main content",
            "confidence": 0.8,
            "reason": "scroll down to find Display",
        }


class ScrollTapConversionTests(unittest.TestCase):
    def test_tap_on_scrollable_container_becomes_scroll_gesture(self) -> None:
        ui = (
            '<hierarchy rotation="0">'
            '<node class="android.widget.FrameLayout" bounds="[0,0][1000,2000]">'
            '<node content-desc="main content" '
            'class="androidx.recyclerview.widget.RecyclerView" scrollable="true" '
            'clickable="true" bounds="[0,100][1000,1900]"/>'
            "</node></hierarchy>"
        )
        action = ScrollTapModel().plan(
            "Turn brightness to the max value", _blank_png(), ui, []
        )
        self.assertEqual(action.type, "gesture")
        self.assertEqual(action.direction, "reveal_below")


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
