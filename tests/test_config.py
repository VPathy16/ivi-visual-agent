import json
import tempfile
import unittest
from pathlib import Path

from ivi_agent.config import Config, PROFILES


class ProfileTests(unittest.TestCase):
    def test_balanced_is_library_defaults(self) -> None:
        base = Config()
        applied = Config().apply_profile("balanced")
        self.assertEqual(applied.verification_level, base.verification_level)
        self.assertEqual(applied.enable_ocr, base.enable_ocr)
        self.assertEqual(applied.settle_timeout_seconds, base.settle_timeout_seconds)

    def test_fast_minimizes_overhead(self) -> None:
        config = Config().apply_profile("fast")
        self.assertEqual(config.verification_level, "final")
        self.assertFalse(config.enable_ocr)
        self.assertEqual(config.settle_timeout_seconds, 1.0)
        # The grounding ladder is untouched — a11y stays on to carry the run.
        self.assertTrue(config.accessibility_fast_path)

    def test_strict_maximizes_evidence(self) -> None:
        config = Config().apply_profile("strict")
        self.assertEqual(config.verification_level, "strict")
        self.assertTrue(config.enable_ocr)
        self.assertEqual(config.settle_timeout_seconds, 3.0)
        self.assertEqual(config.minimum_success_confidence, 0.9)

    def test_none_profile_is_noop(self) -> None:
        config = Config().apply_profile(None)
        self.assertEqual(config.settle_timeout_seconds, Config().settle_timeout_seconds)

    def test_unknown_profile_raises(self) -> None:
        with self.assertRaises(ValueError):
            Config().apply_profile("turbo")

    def test_profile_overrides_config_json_for_its_keys(self) -> None:
        # --profile is a deliberate CLI intent, so it wins over config.json for
        # the keys it owns; keys the profile doesn't touch are preserved.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps({"settle_timeout_seconds": 5.0, "model": "custom:1b"}),
                encoding="utf-8",
            )
            config = Config.load(str(path)).apply_profile("fast")
            self.assertEqual(config.settle_timeout_seconds, 1.0)  # profile wins
            self.assertEqual(config.verification_level, "final")
            self.assertEqual(config.model, "custom:1b")  # untouched key preserved

    def test_relaunch_before_run_defaults_off(self) -> None:
        # Clean-start must be opt-in so it never surprises an existing setup.
        self.assertFalse(Config().relaunch_before_run)

    def test_profiles_only_touch_known_fields(self) -> None:
        known = {f for f in vars(Config()).keys()}
        for name, overrides in PROFILES.items():
            for key in overrides:
                self.assertIn(key, known, f"{name} sets unknown field {key}")


class VerificationLevelTests(unittest.TestCase):
    def test_default_is_checkpoints(self) -> None:
        self.assertEqual(Config().verification_level, "checkpoints")

    def test_unknown_level_raises(self) -> None:
        with self.assertRaises(ValueError):
            Config(verification_level="paranoid")

    def test_uses_model_verify(self) -> None:
        self.assertFalse(Config(verification_level="off").uses_model_verify())
        for level in ("final", "checkpoints", "strict"):
            self.assertTrue(Config(verification_level=level).uses_model_verify())

    def test_verify_step_matrix(self) -> None:
        # (level, is_final) -> should the step be model-verified?
        cases = {
            ("off", True): False, ("off", False): False,
            ("final", True): False, ("final", False): False,
            ("checkpoints", True): True, ("checkpoints", False): False,
            ("strict", True): True, ("strict", False): True,
        }
        for (level, is_final), expected in cases.items():
            self.assertEqual(
                Config(verification_level=level).verify_step(is_final), expected,
                f"{level} is_final={is_final}",
            )


if __name__ == "__main__":
    unittest.main()
