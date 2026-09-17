import json
import tempfile
import unittest
from pathlib import Path

from ivi_agent.report import write_report
from ivi_agent.types import Action, RunResult, StepRecord


class GroundingTelemetryTests(unittest.TestCase):
    def _result(self) -> RunResult:
        directory = Path(tempfile.mkdtemp())
        result = RunResult(
            goal="Start the seat massage",
            outcome="pass",
            reason="done",
            run_directory=str(directory),
        )
        result.steps = [
            StepRecord(
                number=1,
                screenshot="step-01.png",
                action=Action(type="tap", confidence=0.91, reason="cv", target="Seat", x=0.4, y=0.3),
                ui_dump_available=False,
                decision_seconds=0.02,
                grounded_by="cv",
            ),
            StepRecord(
                number=2,
                screenshot="step-02.png",
                action=Action(type="tap", confidence=0.8, reason="model", target="Start", x=0.5, y=0.5),
                ui_dump_available=True,
                decision_seconds=3.10,
                grounded_by="model",
            ),
        ]
        result.grounding = {
            "cv_fast_path_steps": 1,
            "model_steps": 1,
            "total_steps": 2,
            "total_decision_seconds": 3.12,
        }
        return result

    def test_result_json_carries_grounding(self) -> None:
        result = self._result()
        write_report(result)
        data = json.loads((Path(result.run_directory) / "result.json").read_text())
        self.assertEqual(data["grounding"]["cv_fast_path_steps"], 1)
        self.assertEqual(data["grounding"]["model_steps"], 1)
        self.assertEqual(data["steps"][0]["grounded_by"], "cv")
        self.assertEqual(data["steps"][1]["grounded_by"], "model")

    def test_report_html_shows_grounding(self) -> None:
        result = self._result()
        write_report(result)
        html = (Path(result.run_directory) / "report.html").read_text()
        self.assertIn("Grounded by", html)          # table column
        self.assertIn("Grounding:", html)           # summary line
        self.assertIn("1 CV / 1 model", html)


if __name__ == "__main__":
    unittest.main()
