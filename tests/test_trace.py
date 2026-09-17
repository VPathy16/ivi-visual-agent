import json
import tempfile
import unittest
from pathlib import Path

from ivi_agent.trace import NullTrace, RunTrace


class RunTraceTests(unittest.TestCase):
    def _trace(self) -> tuple[RunTrace, Path]:
        directory = Path(tempfile.mkdtemp())
        return RunTrace(directory, run_id="20260101T000000Z"), directory

    def test_events_are_appended_as_jsonl(self) -> None:
        trace, directory = self._trace()
        trace.event("run_start", goal="Start the seat massage")
        trace.event("decision", step=1, grounded_by="cv", confidence=0.9)
        trace.event("done", outcome="pass")
        trace.close()

        lines = (directory / "events.jsonl").read_text().strip().splitlines()
        self.assertEqual(len(lines), 3)
        records = [json.loads(line) for line in lines]
        self.assertEqual([r["kind"] for r in records], ["run_start", "decision", "done"])
        self.assertEqual([r["seq"] for r in records], [1, 2, 3])
        self.assertTrue(all("ts" in r for r in records))
        self.assertEqual(records[1]["grounded_by"], "cv")

    def test_agent_log_written(self) -> None:
        trace, directory = self._trace()
        trace.event("plan", subgoals=["open climate", "set temp"])
        trace.close()
        log = (directory / "agent.log").read_text()
        self.assertIn("plan", log)
        self.assertIn("subgoals", log)

    def test_write_json_artifacts(self) -> None:
        trace, directory = self._trace()
        trace.write_json("task.json", {"goal": "g", "run_id": "r"})
        trace.write_json("plan.json", {"subgoals": ["a", "b"]})
        trace.close()
        task = json.loads((directory / "task.json").read_text())
        plan = json.loads((directory / "plan.json").read_text())
        self.assertEqual(task["goal"], "g")
        self.assertEqual(plan["subgoals"], ["a", "b"])

    def test_close_is_idempotent(self) -> None:
        trace, _ = self._trace()
        trace.event("run_start")
        trace.close()
        trace.close()  # must not raise

    def test_non_serializable_field_does_not_crash(self) -> None:
        trace, directory = self._trace()
        trace.event("weird", value=object())  # default=str handles it
        trace.close()
        line = json.loads((directory / "events.jsonl").read_text().strip())
        self.assertIn("value", line)


class NullTraceTests(unittest.TestCase):
    def test_writes_nothing(self) -> None:
        directory = Path(tempfile.mkdtemp())
        trace = NullTrace()
        trace.event("run_start", goal="x")
        trace.write_json("task.json", {"a": 1})
        trace.close()
        self.assertFalse((directory / "events.jsonl").exists())
        self.assertFalse((directory / "task.json").exists())


if __name__ == "__main__":
    unittest.main()
