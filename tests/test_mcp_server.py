"""Tests for the MCP server's pure helpers (no `mcp` package required)."""

import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ivi_agent.adb import AdbDevice
from ivi_agent.graph import SceneGraph
from ivi_agent import mcp_server as m


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (60, 100), "black").save(buf, format="PNG")
    return buf.getvalue()


class SummarizeResultTests(unittest.TestCase):
    def test_compacts_result(self) -> None:
        result = {
            "goal": "Start the seat massage",
            "outcome": "pass",
            "reason": "done",
            "run_directory": "/nonexistent/run",
            "subgoals": [
                {"number": 1, "description": "a", "status": "passed", "evidence": "x"},
                {"number": 2, "description": "b", "status": "failed", "evidence": "y"},
            ],
            "grounding": {"total_steps": 2},
            "scene_graph": {"coverage": {"defects": 0}, "pending_findings": [{"id": "f1"}]},
            "crashes": [],
        }
        summary = m._summarize_result(result)
        self.assertEqual(summary["outcome"], "pass")
        self.assertEqual(summary["subgoals"]["total"], 2)
        self.assertEqual(summary["subgoals"]["passed"], 1)
        self.assertEqual(summary["scene_graph"], {"defects": 0})
        self.assertEqual(summary["pending_findings"], [{"id": "f1"}])
        # replay path only surfaced when the file actually exists
        self.assertIsNone(summary["replay_html"])


class InspectTraceTests(unittest.TestCase):
    def test_reads_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "run1"
            d.mkdir()
            (d / "result.json").write_text(json.dumps({
                "goal": "g", "outcome": "pass", "reason": "r",
                "grounding": {"total_steps": 1},
                "steps": [{"number": 1, "action": {"type": "tap", "target": "wave"},
                           "grounded_by": "a11y", "screen_changed": False, "decision_seconds": 0.1}],
                "crashes": [],
            }))
            (d / "events.jsonl").write_text(
                "\n".join(json.dumps(e) for e in [
                    {"kind": "run_start"}, {"kind": "decision", "step": 1}, {"kind": "decision", "step": 1},
                ])
            )
            out = m.inspect_trace(str(d))
            self.assertEqual(out["outcome"], "pass")
            self.assertEqual(out["event_counts"], {"run_start": 1, "decision": 2})
            self.assertEqual(out["total_events"], 3)
            self.assertEqual(out["steps"][0]["grounded_by"], "a11y")

    def test_missing_dir(self) -> None:
        self.assertIn("error", m.inspect_trace("/no/such/dir"))


class ListRunsTests(unittest.TestCase):
    def test_lists_newest_first_paginated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("20260101T000000Z", "20260102T000000Z", "20260103T000000Z"):
                d = root / name
                d.mkdir()
                (d / "result.json").write_text(json.dumps({"goal": name, "outcome": "pass"}))
            out = m.list_runs(str(root), limit=2, offset=0)
            self.assertEqual(out["total"], 3)
            self.assertEqual(out["count"], 2)
            self.assertTrue(out["has_more"])
            self.assertEqual(out["runs"][0]["goal"], "20260103T000000Z")  # newest first
            self.assertEqual(out["next_offset"], 2)

    def test_missing_root(self) -> None:
        out = m.list_runs("/no/such/root")
        self.assertEqual(out["total"], 0)
        self.assertFalse(out["has_more"])


class SceneGraphStateTests(unittest.TestCase):
    def test_reports_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile_dir = Path(tmp) / "benz"
            profile_dir.mkdir()
            graph = SceneGraph.from_manual({"screens": [
                {"id": "screen.home", "title": "Home", "controls": []},
                {"id": "screen.seat", "title": "Seat massage", "controls": []},
            ]})
            graph.save(profile_dir / "scene_graph.json")
            out = m.scene_graph_state("benz", str(tmp))
            self.assertEqual(out["profile"], "benz")
            self.assertIn("coverage", out)
            self.assertEqual(out["coverage"]["manual_screens"], 2)

    def test_missing_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("error", m.scene_graph_state("benz", str(tmp)))


class _FakeDevice(AdbDevice):
    UI = ("<hierarchy><node bounds='[0,0][1080,2400]'>"
          "<node text='Seat massage' bounds='[40,240][700,320]'/>"
          "<node text='Start' clickable='true' bounds='[40,900][300,980]'/>"
          "</node></hierarchy>")

    def __init__(self) -> None:
        super().__init__(serial="emulator-5554")

    def capture(self, destination: Path) -> bytes:
        data = _png()
        Path(destination).write_bytes(data)
        return data

    def ui_dump(self) -> str:
        return self.UI

    def screen_size(self):  # type: ignore[override]
        return (1080, 2400)


class DeviceStateTests(unittest.TestCase):
    def test_snapshot_reads_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = m.device_state(device=_FakeDevice(), output=tmp)
            self.assertEqual(out["screen_size"], [1080, 2400])
            self.assertIn("Seat massage", out["titles"])
            self.assertIn("Seat massage", out["visible_text"])
            labels = [e["label"] for e in out["clickable_elements"]]
            self.assertIn("Start", labels)
            self.assertTrue(Path(out["screenshot"]).is_file())


class PreflightTests(unittest.TestCase):
    def test_returns_check_structure(self) -> None:
        out = m.preflight(None)
        self.assertIn("ok", out)
        self.assertTrue(all({"name", "passed", "detail"} <= set(c) for c in out["checks"]))
        names = [c["name"] for c in out["checks"]]
        self.assertIn("adb", names)


if __name__ == "__main__":
    unittest.main()
