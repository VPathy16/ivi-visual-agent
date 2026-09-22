import tempfile
import unittest
from pathlib import Path

from ivi_agent.agent import resolve_target_in_tree
from ivi_agent.adb import AdbDevice
from ivi_agent.device import Device
from ivi_agent.perception import extract_ui_elements, extract_visible_text
from ivi_agent.types import Action
from ivi_agent.windows import WindowsDevice, uia_to_uiautomator

# A representative WinAppDriver page_source: a wrapper <Root>, a top Window that
# supplies the screen bounds, a clickable Button, an Edit, and a static Text.
PAGE_SOURCE = """<Root>
  <Window Name="Calculator" ClassName="ApplicationFrameWindow" x="0" y="0" width="400" height="600">
    <Text Name="Display is 0" AutomationId="CalculatorResults" x="10" y="50" width="380" height="80"/>
    <Button Name="Seven" AutomationId="num7Button" x="40" y="300" width="80" height="60"/>
    <Edit Name="" AutomationId="searchBox" x="20" y="500" width="360" height="40"/>
  </Window>
</Root>"""


class UiaAdapterTests(unittest.TestCase):
    def test_maps_controls_to_uiautomator_schema(self) -> None:
        xml = uia_to_uiautomator(PAGE_SOURCE)
        self.assertIn('bounds="[40,300][120,360]"', xml)   # x,y,w,h -> x1,y1,x2,y2
        self.assertIn('text="Seven"', xml)
        self.assertIn('resource-id="num7Button"', xml)
        self.assertIn('clickable="true"', xml)             # Button is actionable
        self.assertIn('class="EditText"', xml)             # Edit -> EditText role

    def test_empty_and_bad_input(self) -> None:
        self.assertEqual(uia_to_uiautomator(""), "")
        self.assertEqual(uia_to_uiautomator("<not xml"), "")

    def test_perception_reads_converted_tree(self) -> None:
        xml = uia_to_uiautomator(PAGE_SOURCE)
        labels = [e.label for e in extract_ui_elements(xml)]
        self.assertIn("Seven", labels)
        self.assertIn("Display is 0", extract_visible_text(xml))

    def test_a11y_fast_path_grounds_on_converted_tree(self) -> None:
        # The whole point: the existing grounder resolves a Windows control.
        xml = uia_to_uiautomator(PAGE_SOURCE)
        point = resolve_target_in_tree(xml, {"seven"})
        self.assertIsNotNone(point)
        cx, cy = point
        self.assertAlmostEqual(cx, 80 / 400, places=2)     # (40+120)/2 / width
        self.assertAlmostEqual(cy, 330 / 600, places=2)    # (300+360)/2 / height


class _FakeDriver:
    def __init__(self, page_source: str) -> None:
        self._ps = page_source
        self.calls: list[tuple[str, dict]] = []

    def get_screenshot_as_png(self) -> bytes:
        return b"\x89PNG\r\n\x1a\nfake"

    @property
    def page_source(self) -> str:
        return self._ps

    def get_window_rect(self) -> dict:
        return {"x": 0, "y": 0, "width": 400, "height": 600}

    def execute_script(self, script: str, params: dict | None = None):
        self.calls.append((script, params or {}))
        return None


class WindowsDeviceTests(unittest.TestCase):
    def _device(self) -> WindowsDevice:
        return WindowsDevice(_FakeDriver(PAGE_SOURCE))

    def test_conforms_to_device_protocol(self) -> None:
        self.assertIsInstance(self._device(), Device)
        self.assertIsInstance(AdbDevice(serial="x"), Device)  # both backends conform

    def test_screen_size_and_capture(self) -> None:
        device = self._device()
        self.assertEqual(device.screen_size(), (400, 600))
        with tempfile.TemporaryDirectory() as tmp:
            shot = Path(tmp) / "s.png"
            data = device.capture(shot)
            self.assertTrue(shot.is_file())
            self.assertTrue(data.startswith(b"\x89PNG"))

    def test_ui_dump_is_converted(self) -> None:
        self.assertIn('text="Seven"', self._device().ui_dump())

    def test_tap_issues_windows_click_at_pixels(self) -> None:
        device = self._device()
        device.execute(Action(type="tap", x=0.2, y=0.55, confidence=1.0, reason="x"), (400, 600))
        script, params = device.driver.calls[-1]
        self.assertEqual(script, "windows: click")
        # 0.2*(400-1) ~ 80, 0.55*(600-1) ~ 329
        self.assertAlmostEqual(params["x"], 80, delta=1)
        self.assertAlmostEqual(params["y"], 329, delta=1)

    def test_input_text_clicks_then_types(self) -> None:
        device = self._device()
        device.execute(
            Action(type="input_text", x=0.5, y=0.83, text="hi", confidence=1.0, reason="x"),
            (400, 600),
        )
        scripts = [s for s, _ in device.driver.calls]
        self.assertEqual(scripts, ["windows: click", "windows: keys"])


if __name__ == "__main__":
    unittest.main()
