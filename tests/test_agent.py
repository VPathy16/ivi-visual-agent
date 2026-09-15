import unittest

from ivi_agent.agent import action_signature, title_satisfies_navigation_goal
from ivi_agent.types import Action


class ActionSignatureTests(unittest.TestCase):
    def test_home_actions_share_signature(self) -> None:
        first = Action(type="home", confidence=0.9, reason="first")
        second = Action(type="home", confidence=0.8, reason="second")
        self.assertEqual(action_signature(first), action_signature(second))

    def test_exact_title_completes_navigation_milestone(self) -> None:
        self.assertEqual(
            title_satisfies_navigation_goal(
                "Open the Settings application", ["Settings", "Connected devices"]
            ),
            "Settings",
        )

    def test_related_title_does_not_complete_navigation_milestone(self) -> None:
        self.assertIsNone(
            title_satisfies_navigation_goal(
                "Open the Bluetooth settings screen", ["Developer options"]
            )
        )

    def test_nearby_taps_share_signature(self) -> None:
        first = Action(type="tap", x=0.501, y=0.201, confidence=0.9, reason="first")
        second = Action(type="tap", x=0.502, y=0.202, confidence=0.9, reason="second")
        self.assertEqual(action_signature(first), action_signature(second))
