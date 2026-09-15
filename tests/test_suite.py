import json
import tempfile
import unittest
from pathlib import Path

from ivi_agent.suite import _write_suite_report, load_suite_cases


class SuiteCaseTests(unittest.TestCase):
    def test_loads_named_goals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.json"
            path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {"name": "Bluetooth", "goal": "Open Bluetooth settings"},
                            {"name": "Sound", "goal": "Open Sound settings"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                load_suite_cases(path),
                [
                    {"name": "Bluetooth", "goal": "Open Bluetooth settings"},
                    {"name": "Sound", "goal": "Open Sound settings"},
                ],
            )

    def test_rejects_duplicate_case_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.json"
            path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {"name": "Same", "goal": "First"},
                            {"name": "Same", "goal": "Second"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_suite_cases(path)

    def test_rejects_empty_suite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.json"
            path.write_text('{"cases": []}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty"):
                load_suite_cases(path)

    def test_writes_aggregate_json_and_html_with_case_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_directory = root / "bluetooth" / "run"
            summary = {
                "outcome": "pass",
                "passed": 1,
                "total": 1,
                "duration_seconds": 12.5,
                "cases": [
                    {
                        "name": "Bluetooth",
                        "goal": "Open Bluetooth settings",
                        "outcome": "pass",
                        "actions": 2,
                        "duration_seconds": 12.5,
                        "reason": "Visible Bluetooth screen",
                        "run_directory": str(run_directory),
                    }
                ],
            }
            _write_suite_report(summary, root)
            self.assertTrue((root / "suite-result.json").exists())
            report = (root / "suite-report.html").read_text(encoding="utf-8")
            self.assertIn("1/1 passed", report)
            self.assertIn("bluetooth/run/report.html", report)


if __name__ == "__main__":
    unittest.main()
