import unittest

from ivi_agent.agent import (
    map_subgoal_targets,
    resolve_target_in_tree,
    target_keywords,
)
from ivi_agent.types import SubgoalRecord


def _tree(*nodes: str) -> str:
    frame = '<node class="android.widget.FrameLayout" bounds="[0,0][1000,600]">'
    body = "".join(nodes)
    return f'<hierarchy rotation="0">{frame}{body}</node></hierarchy>'


def _node(text: str, bounds: str, clickable: str = "true") -> str:
    return f'<node text="{text}" class="android.widget.Button" clickable="{clickable}" bounds="{bounds}"/>'


TASK = {
    "steps": [
        {"milestone": "Open the seat massage screen", "target": "home.seat_tile"},
        {"milestone": "Select a massage program", "target": "seat.program_wave"},
        {"milestone": "Start the seat massage", "target": "seat.start"},
    ]
}
CONTROL_NAMES = {
    "home.seat_tile": "Seat Comfort tile",
    "seat.program_wave": "Wave program",
    "seat.start": "Start",
}


class TargetKeywordsTests(unittest.TestCase):
    def test_strips_generic_words(self) -> None:
        self.assertEqual(target_keywords("Wave program"), {"wave"})
        self.assertEqual(target_keywords("Seat Comfort tile"), {"seat", "comfort"})
        self.assertEqual(target_keywords("Start"), {"start"})


class MapTargetsTests(unittest.TestCase):
    def test_maps_by_milestone_to_control_keywords(self) -> None:
        subs = [
            SubgoalRecord(number=1, description="Open the seat massage screen"),
            SubgoalRecord(number=2, description="Select a massage program"),
            SubgoalRecord(number=3, description="Start the seat massage"),
        ]
        got = map_subgoal_targets(subs, TASK, CONTROL_NAMES)
        self.assertEqual(got[0], {"seat", "comfort"})
        self.assertEqual(got[1], {"wave"})
        self.assertEqual(got[2], {"start"})


class ResolveInTreeTests(unittest.TestCase):
    def test_unique_match_returns_center(self) -> None:
        tree = _tree(_node("Wave", "[100,200][300,260]"), _node("Lumbar", "[400,200][600,260]"))
        center = resolve_target_in_tree(tree, {"wave"})
        self.assertIsNotNone(center)
        cx, cy = center
        self.assertAlmostEqual(cx, 200 / 1000, places=2)   # (100+300)/2 / width
        self.assertAlmostEqual(cy, 230 / 600, places=2)

    def test_exact_wins_over_superset_on_ambiguity(self) -> None:
        tree = _tree(_node("Start", "[0,0][100,50]"), _node("Start massage now", "[0,100][300,150]"))
        center = resolve_target_in_tree(tree, {"start"})
        self.assertIsNotNone(center)      # exact "Start" chosen over the longer label
        self.assertAlmostEqual(center[1], 25 / 600, places=2)

    def test_no_match_returns_none(self) -> None:
        tree = _tree(_node("Climate", "[0,0][100,50]"))
        self.assertIsNone(resolve_target_in_tree(tree, {"wave"}))

    def test_ambiguous_returns_none(self) -> None:
        tree = _tree(_node("Wave", "[0,0][100,50]"), _node("Wave", "[0,100][100,150]"))
        self.assertIsNone(resolve_target_in_tree(tree, {"wave"}))

    def test_empty_keywords_returns_none(self) -> None:
        self.assertIsNone(resolve_target_in_tree(_tree(_node("Wave", "[0,0][100,50]")), set()))


if __name__ == "__main__":
    unittest.main()
