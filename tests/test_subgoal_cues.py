import unittest

from ivi_agent.agent import map_subgoal_success_cues, subgoal_cue_satisfied
from ivi_agent.types import SubgoalRecord


TASK = {
    "steps": [
        {"milestone": "Open the seat massage screen", "success_text": "Seat massage"},
        {"milestone": "Select a massage program", "success_text": "Selected"},
        {"milestone": "Start the seat massage", "success_text": "Massage running"},
    ]
}


class MapCuesTests(unittest.TestCase):
    def _subgoals(self):
        return [
            SubgoalRecord(number=1, description="Open the seat massage screen"),
            SubgoalRecord(number=2, description="Select a massage program"),
            SubgoalRecord(number=3, description="Start the seat massage"),
        ]

    def test_maps_by_milestone(self) -> None:
        cues = map_subgoal_success_cues(self._subgoals(), TASK)
        self.assertEqual(cues, {0: "Seat massage", 1: "Selected", 2: "Massage running"})

    def test_no_task_data(self) -> None:
        self.assertEqual(map_subgoal_success_cues(self._subgoals(), None), {})

    def test_unmatched_subgoal_has_no_cue(self) -> None:
        subs = [SubgoalRecord(number=1, description="Do something else")]
        self.assertEqual(map_subgoal_success_cues(subs, TASK), {})

    def test_steps_without_success_text_are_skipped(self) -> None:
        task = {"steps": [{"milestone": "Select a massage program"}]}
        subs = [SubgoalRecord(number=1, description="Select a massage program")]
        self.assertEqual(map_subgoal_success_cues(subs, task), {})


class CueSatisfiedTests(unittest.TestCase):
    def test_cue_present_case_insensitive(self) -> None:
        self.assertTrue(subgoal_cue_satisfied("Selected", ["Selected wave — press Start"]))
        self.assertTrue(subgoal_cue_satisfied("massage running", ["Massage running: lumbar"]))

    def test_cue_absent(self) -> None:
        self.assertFalse(subgoal_cue_satisfied("Selected", ["Idle — select a program"]))

    def test_empty_cue(self) -> None:
        self.assertFalse(subgoal_cue_satisfied("", ["anything"]))


if __name__ == "__main__":
    unittest.main()
