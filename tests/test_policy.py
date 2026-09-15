import unittest

from ivi_agent.config import Config
from ivi_agent.policy import PolicyViolation, validate_action
from ivi_agent.types import Action


class PolicyTests(unittest.TestCase):
    def test_accepts_high_confidence_normalized_tap(self) -> None:
        validate_action(
            Action(type="tap", x=0.5, y=0.25, confidence=0.9, reason="visible"),
            Config(),
        )

    def test_rejects_low_confidence_action(self) -> None:
        with self.assertRaisesRegex(PolicyViolation, "below"):
            validate_action(
                Action(type="tap", x=0.5, y=0.25, confidence=0.5, reason="guess"),
                Config(),
            )

    def test_rejects_protected_region(self) -> None:
        config = Config(protected_regions=[[0.0, 0.0, 0.2, 0.2]])
        with self.assertRaisesRegex(PolicyViolation, "protected"):
            validate_action(
                Action(type="tap", x=0.1, y=0.1, confidence=0.9, reason="visible"),
                config,
            )

    def test_rejects_unrequested_permission_change(self) -> None:
        with self.assertRaisesRegex(PolicyViolation, "explicit permission goal"):
            validate_action(
                Action(
                    type="tap",
                    target="Grant permission",
                    x=0.5,
                    y=0.5,
                    confidence=0.95,
                    reason="App requests access",
                ),
                Config(),
                "Open the Notifications settings screen",
            )

    def test_rejects_bare_allow_button_without_permission_goal(self) -> None:
        with self.assertRaisesRegex(PolicyViolation, "explicit permission goal"):
            validate_action(
                Action(
                    type="tap",
                    target="Allow",
                    x=0.5,
                    y=0.5,
                    confidence=0.95,
                    reason="Visible button",
                ),
                Config(),
                "Open sound settings",
            )

    def test_allows_explicit_permission_goal(self) -> None:
        validate_action(
            Action(
                type="tap",
                target="Grant permission",
                x=0.5,
                y=0.5,
                confidence=0.95,
                reason="User requested permission",
            ),
            Config(),
            "Grant the media permission",
        )

    def test_blocks_destructive_action_even_when_confident(self) -> None:
        with self.assertRaisesRegex(PolicyViolation, "blocked"):
            validate_action(
                Action(
                    type="tap",
                    target="Factory reset",
                    x=0.5,
                    y=0.5,
                    confidence=0.99,
                    reason="Visible control",
                ),
                Config(),
                "Open system settings",
            )


if __name__ == "__main__":
    unittest.main()
