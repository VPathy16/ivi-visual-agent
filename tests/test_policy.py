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


if __name__ == "__main__":
    unittest.main()
