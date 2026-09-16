import unittest

from ivi_agent.agent import screen_made_progress

UI_A = '<hierarchy rotation="0"><node text="Start" bounds="[0,0][100,50]"/></hierarchy>'
UI_B = (
    '<hierarchy rotation="0">'
    '<node text="Stop" bounds="[0,0][100,50]"/>'
    '<node text="Massage running: shoulder" bounds="[0,60][300,90]"/>'
    "</hierarchy>"
)


class ScreenProgressTests(unittest.TestCase):
    def test_pixel_change_counts_as_progress(self) -> None:
        # 5 differing bits > threshold of 4
        self.assertTrue(screen_made_progress(0, 0b11111, UI_A, UI_A))

    def test_identical_screen_is_no_progress(self) -> None:
        self.assertFalse(screen_made_progress(0, 0, UI_A, UI_A))

    def test_ui_text_change_counts_even_without_pixel_change(self) -> None:
        # Start -> Stop + "Massage running" is a small pixel delta but a real
        # state change; the UI-text diff must catch it.
        self.assertTrue(screen_made_progress(0, 0, UI_A, UI_B))


if __name__ == "__main__":
    unittest.main()
