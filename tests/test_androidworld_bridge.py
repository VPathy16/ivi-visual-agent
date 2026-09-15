import unittest
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from ivi_agent.integrations.androidworld import bridge
from ivi_agent.perception import extract_ui_elements, extract_screen_titles
from ivi_agent.types import Action


@dataclass
class FakeBBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass
class FakeElement:
    text: str = ""
    content_description: str = ""
    class_name: str = "android.widget.TextView"
    resource_id: str = ""
    package_name: str = "com.example"
    is_clickable: bool = False
    is_checkable: bool = False
    is_checked: bool = False
    is_scrollable: bool = False
    is_editable: bool = False
    is_focused: bool = False
    is_long_clickable: bool = False
    bbox_pixels: FakeBBox | None = None


class BridgeXmlTest(unittest.TestCase):
    def test_xml_is_parseable_and_round_trips_through_perception(self) -> None:
        elements = [
            FakeElement(
                text="Bluetooth",
                is_clickable=True,
                bbox_pixels=FakeBBox(0, 200, 400, 300),
            ),
            FakeElement(
                content_description="Wi-Fi & internet",
                is_clickable=True,
                bbox_pixels=FakeBBox(0, 300, 400, 400),
            ),
        ]
        xml = bridge.ui_elements_to_uiautomator_xml(elements, (1000, 2000))
        ET.fromstring(xml)  # must be well-formed

        parsed = extract_ui_elements(xml)
        labels = {element.label for element in parsed}
        self.assertIn("Bluetooth", labels)
        self.assertIn("Wi-Fi & internet", labels)

    def test_center_normalization_uses_full_screen_frame(self) -> None:
        element = FakeElement(
            text="OK", is_clickable=True, bbox_pixels=FakeBBox(400, 900, 600, 1100)
        )
        xml = bridge.ui_elements_to_uiautomator_xml([element], (1000, 2000))
        parsed = extract_ui_elements(xml)
        self.assertEqual(len(parsed), 1)
        cx, cy = parsed[0].center
        self.assertAlmostEqual(cx, 0.5, places=2)  # (400+600)/2 / 1000
        self.assertAlmostEqual(cy, 0.5, places=2)  # (900+1100)/2 / 2000

    def test_special_characters_are_escaped(self) -> None:
        element = FakeElement(
            text='Sound & "vibration" <settings>',
            is_clickable=True,
            bbox_pixels=FakeBBox(0, 0, 100, 100),
        )
        xml = bridge.ui_elements_to_uiautomator_xml([element], (500, 500))
        ET.fromstring(xml)  # would raise if unescaped
        self.assertNotIn("& ", xml.replace("&amp;", ""))

    def test_titles_extractable_when_resource_id_marks_a_title(self) -> None:
        # AndroidWorld elements carry resource_name; the bridge must preserve it
        # so the repo's semantic title heuristic (keyed on the resource role)
        # can drive subgoal advancement.
        element = FakeElement(
            text="Display",
            resource_id="com.android.settings:id/action_bar_title",
            bbox_pixels=FakeBBox(0, 0, 800, 120),
        )
        xml = bridge.ui_elements_to_uiautomator_xml([element], (800, 1600))
        self.assertIn("Display", extract_screen_titles(xml))


class BridgeActionTest(unittest.TestCase):
    def test_tap_maps_to_pixel_click(self) -> None:
        action = Action(type="tap", x=0.5, y=0.25, confidence=0.9, reason="x")
        kwargs = bridge.ivi_action_to_json_action_kwargs(action, (1000, 2000))
        self.assertEqual(kwargs["action_type"], "click")
        self.assertEqual(kwargs["x"], round(0.5 * 999))
        self.assertEqual(kwargs["y"], round(0.25 * 1999))

    def test_long_press_and_double_tap(self) -> None:
        lp = bridge.ivi_action_to_json_action_kwargs(
            Action(type="long_press", x=0.1, y=0.1, confidence=0.9, reason="x"), (100, 100)
        )
        dt = bridge.ivi_action_to_json_action_kwargs(
            Action(type="double_tap", x=0.1, y=0.1, confidence=0.9, reason="x"), (100, 100)
        )
        self.assertEqual(lp["action_type"], "long_press")
        self.assertEqual(dt["action_type"], "double_tap")

    def test_input_text_clears_and_types(self) -> None:
        action = Action(type="input_text", x=0.5, y=0.5, text="hello", confidence=0.9, reason="x")
        kwargs = bridge.ivi_action_to_json_action_kwargs(action, (200, 200))
        self.assertEqual(kwargs["action_type"], "input_text")
        self.assertEqual(kwargs["text"], "hello")
        self.assertTrue(kwargs["clear_text"])

    def test_gesture_directions_map_to_scroll(self) -> None:
        cases = {
            "reveal_above": "up",
            "reveal_below": "down",
            "reveal_left": "left",
            "reveal_right": "right",
        }
        for direction, expected in cases.items():
            action = Action(
                type="gesture", direction=direction, confidence=0.9, reason="x"
            )
            kwargs = bridge.ivi_action_to_json_action_kwargs(action, (100, 100))
            self.assertEqual(kwargs, {"action_type": "scroll", "direction": expected})

    def test_open_app_uses_app_name(self) -> None:
        action = Action(type="open_app", app_name="Chrome", confidence=0.9, reason="x")
        kwargs = bridge.ivi_action_to_json_action_kwargs(action, (100, 100))
        self.assertEqual(kwargs, {"action_type": "open_app", "app_name": "Chrome"})

    def test_navigation_and_control_actions(self) -> None:
        self.assertEqual(
            bridge.ivi_action_to_json_action_kwargs(
                Action(type="back", confidence=0.9, reason="x"), (10, 10)
            )["action_type"],
            "navigate_back",
        )
        self.assertEqual(
            bridge.ivi_action_to_json_action_kwargs(
                Action(type="home", confidence=0.9, reason="x"), (10, 10)
            )["action_type"],
            "navigate_home",
        )
        self.assertEqual(
            bridge.ivi_action_to_json_action_kwargs(
                Action(type="keyboard_enter", confidence=0.9, reason="x"), (10, 10)
            )["action_type"],
            "keyboard_enter",
        )

    def test_finish_maps_outcome_to_status(self) -> None:
        done = bridge.ivi_action_to_json_action_kwargs(
            Action(type="finish", outcome="pass", confidence=0.9, reason="x"), (10, 10)
        )
        stuck = bridge.ivi_action_to_json_action_kwargs(
            Action(type="finish", outcome="fail", confidence=0.9, reason="x"), (10, 10)
        )
        self.assertEqual(done, {"action_type": "status", "goal_status": "complete"})
        self.assertEqual(stuck, {"action_type": "status", "goal_status": "infeasible"})


if __name__ == "__main__":
    unittest.main()
