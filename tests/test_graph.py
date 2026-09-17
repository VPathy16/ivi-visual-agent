import json
import tempfile
import unittest
from pathlib import Path

from ivi_agent.graph import SceneGraph

MANUAL = {
    "screens": [
        {
            "id": "screen.home",
            "name": "Home - tile grid",
            "landmarks": ["Climate", "Seat Comfort", "Vehicle tiles"],
            "controls": [
                {"id": "home.climate_tile", "action": "tap", "result": "screen.climate"},
                {"id": "home.seat_tile", "action": "tap", "result": "screen.seat_massage"},
            ],
        },
        {"id": "screen.climate", "name": "Climate", "landmarks": ["Temperature slider"], "controls": []},
        {"id": "screen.seat_massage", "name": "Seat massage", "landmarks": ["Start button"], "controls": []},
    ],
}


class FromManualTests(unittest.TestCase):
    def test_seeds_nodes_and_edges(self) -> None:
        g = SceneGraph.from_manual(MANUAL)
        self.assertEqual(set(g.nodes), {"screen.home", "screen.climate", "screen.seat_massage"})
        self.assertTrue(all(n.status == "expected" for n in g.nodes.values()))
        self.assertIn(("screen.home", "screen.climate"), g.edges)
        self.assertIn(("screen.home", "screen.seat_massage"), g.edges)
        self.assertEqual(g.edges[("screen.home", "screen.climate")].source, "manual")


class ObserveTests(unittest.TestCase):
    def test_matches_and_confirms_documented_screen(self) -> None:
        g = SceneGraph.from_manual(MANUAL)
        res = g.observe(["Climate"], phash=1, run_id="r1")
        self.assertTrue(res.matched)
        self.assertFalse(res.is_new_node)
        self.assertEqual(res.node_id, "screen.climate")
        self.assertEqual(g.nodes["screen.climate"].status, "confirmed")
        self.assertIsNone(res.finding)

    def test_confirms_documented_transition(self) -> None:
        g = SceneGraph.from_manual(MANUAL)
        g.observe(["Home"], run_id="r1")  # home matches via 'home' token? landmarks include tiles
        res = g.observe(
            ["Climate"], run_id="r1", came_from="screen.home",
            via_action="tap", via_target="home.climate_tile",
        )
        edge = g.edges[("screen.home", "screen.climate")]
        self.assertEqual(edge.status, "confirmed")
        self.assertIsNone(res.finding)

    def test_undocumented_screen_becomes_defect_candidate(self) -> None:
        g = SceneGraph.from_manual(MANUAL)
        res = g.observe(["Ambient Lighting"], phash=999, run_id="r1", came_from="screen.home",
                        via_action="tap", via_target="mystery.tile")
        self.assertFalse(res.matched)
        self.assertTrue(res.is_new_node)
        self.assertEqual(g.nodes[res.node_id].status, "pending_review")
        self.assertIsNotNone(res.finding)
        self.assertEqual(res.finding.kind, "undocumented_screen")
        self.assertEqual(len(g.pending_findings()), 1)

    def test_undocumented_transition_between_known_screens(self) -> None:
        g = SceneGraph.from_manual(MANUAL)
        # climate -> seat_massage is not a documented edge
        g.observe(["Climate"], run_id="r1")
        res = g.observe(["Seat massage"], run_id="r1", came_from="screen.climate",
                        via_action="tap", via_target="mystery")
        self.assertTrue(res.matched)
        self.assertIsNotNone(res.finding)
        self.assertEqual(res.finding.kind, "undocumented_transition")

    def test_phash_matches_titleless_screen(self) -> None:
        g = SceneGraph.from_manual(MANUAL)
        g.observe(["Climate"], phash=0b1010, run_id="r1")
        res = g.observe([], phash=0b1011, run_id="r1")  # 1 bit off -> same node
        self.assertEqual(res.node_id, "screen.climate")
        self.assertFalse(res.is_new_node)


class ReviewTests(unittest.TestCase):
    def test_approve_and_defect(self) -> None:
        g = SceneGraph.from_manual(MANUAL)
        r1 = g.observe(["Ambient Lighting"], run_id="r1")
        r2 = g.observe(["Charging"], run_id="r1")
        g.review(r1.finding.id, "approve")
        g.review(r2.finding.id, "defect")
        self.assertEqual(g.nodes[r1.node_id].status, "approved")
        self.assertEqual(g.nodes[r2.node_id].status, "defect")
        self.assertEqual(len(g.pending_findings()), 0)
        self.assertEqual(g.coverage()["defects"], 1)

    def test_review_bad_decision_raises(self) -> None:
        g = SceneGraph.from_manual(MANUAL)
        r = g.observe(["Weird"], run_id="r1")
        with self.assertRaises(ValueError):
            g.review(r.finding.id, "maybe")


class CoverageAndPersistenceTests(unittest.TestCase):
    def test_coverage_counts(self) -> None:
        g = SceneGraph.from_manual(MANUAL)
        g.observe(["Climate"], run_id="r1")
        cov = g.coverage()
        self.assertEqual(cov["manual_screens"], 3)
        self.assertEqual(cov["confirmed_screens"], 1)
        self.assertIn("screen.home", cov["unreached_screens"])

    def test_round_trip_and_load_or_seed(self) -> None:
        g = SceneGraph.from_manual(MANUAL)
        g.observe(["Climate"], phash=5, run_id="r1")
        g.observe(["Undocumented"], run_id="r1")
        path = Path(tempfile.mkdtemp()) / "scene_graph.json"
        g.save(path)

        loaded = SceneGraph.load(path)
        self.assertEqual(set(loaded.nodes), set(g.nodes))
        self.assertEqual(loaded.nodes["screen.climate"].status, "confirmed")
        self.assertEqual(len(loaded.findings), len(g.findings))
        # counters survive so new observed ids don't collide
        self.assertEqual(loaded._obs_counter, g._obs_counter)

        # load_or_seed picks up the persisted living graph over a fresh seed
        again = SceneGraph.load_or_seed(path, MANUAL)
        self.assertEqual(again.nodes["screen.climate"].status, "confirmed")
        # and seeds from manual when no file exists
        fresh = SceneGraph.load_or_seed(Path(tempfile.mkdtemp()) / "none.json", MANUAL)
        self.assertEqual(fresh.nodes["screen.climate"].status, "expected")


if __name__ == "__main__":
    unittest.main()
