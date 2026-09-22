import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ivi_agent.replay import build_replay, build_replay_data


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 80), "black").save(buf, format="PNG")
    return buf.getvalue()


def _make_run(tmp: Path) -> Path:
    directory = tmp / "run1"
    directory.mkdir()
    (directory / "step-01.png").write_bytes(_png())
    (directory / "step-01-after.png").write_bytes(_png())
    result = {
        "goal": "Start the seat massage",
        "outcome": "pass",
        "reason": "Observed on screen: 'Massage running'",
        "run_directory": str(directory),
        "subgoals": [
            {"number": 1, "description": "Open seat massage", "status": "passed", "evidence": "title"},
        ],
        "steps": [
            {
                "number": 1,
                "screenshot": "step-01.png",
                "action": {"type": "tap", "target": "wave", "confidence": 1.0, "reason": "a11y match"},
                "screen_changed": False,
                "decision_seconds": 0.0004,
                "grounded_by": "a11y",
                "error": None,
            }
        ],
        "grounding": {
            "total_steps": 1,
            "accessibility_fast_path_steps": 1,
            "cv_fast_path_steps": 0,
            "model_steps": 0,
            "total_decision_seconds": 0.0,
            "total_wall_seconds": 7.6,
            "phase_seconds": {"startup": 0.9, "ui_dump": 4.7, "settle": 1.6},
        },
        "scene_graph": {"coverage": {"manual_screens": 4, "confirmed_screens": 3, "defects": 0},
                        "pending_findings": []},
        "crashes": [],
        "started_at": "2026-09-17T15:00:00Z",
        "finished_at": "2026-09-17T15:00:07Z",
    }
    (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
    events = [
        {"seq": 1, "ts": "t", "kind": "run_start", "goal": "Start the seat massage"},
        {"seq": 2, "ts": "t", "kind": "decision", "step": 1, "grounded_by": "a11y", "action": "tap"},
        {"seq": 3, "ts": "t", "kind": "verify", "step": 1, "scope": "success_cue", "outcome": "pass"},
    ]
    (directory / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events), encoding="utf-8"
    )
    return directory


class BuildReplayDataTests(unittest.TestCase):
    def test_embeds_images_and_groups_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = _make_run(Path(tmp))
            data = build_replay_data(directory)
            self.assertEqual(len(data["steps"]), 1)
            step = data["steps"][0]
            self.assertTrue(step["image"].startswith("data:image/png;base64,"))
            self.assertTrue(step["after_image"].startswith("data:image/png;base64,"))
            # Step-scoped events attach to the step; run-level events go to run_events.
            self.assertEqual([e["kind"] for e in step["events"]], ["decision", "verify"])
            self.assertEqual([e["kind"] for e in data["run_events"]], ["run_start"])

    def test_missing_events_degrades_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = _make_run(Path(tmp))
            (directory / "events.jsonl").unlink()
            data = build_replay_data(directory)
            self.assertEqual(data["steps"][0]["events"], [])
            self.assertEqual(data["run_events"], [])


class BuildReplayHtmlTests(unittest.TestCase):
    def test_writes_self_contained_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = _make_run(Path(tmp))
            out = build_replay(directory)
            self.assertTrue(out.is_file())
            html = out.read_text(encoding="utf-8")
            self.assertIn("<!doctype html>", html)
            self.assertIn("Start the seat massage", html)
            # Payload embedded, image inlined, and no premature </script> break.
            self.assertIn("data:image/png;base64,", html)
            self.assertNotIn("</script></script>", html)
            self.assertIn("const DATA =", html)

    def test_html_has_no_external_resources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = _make_run(Path(tmp))
            html = build_replay(directory).read_text(encoding="utf-8")
            # Offline: no CDN scripts/styles pulled in.
            self.assertNotIn("http://", html.replace("http-equiv", ""))
            self.assertNotIn("https://", html)


if __name__ == "__main__":
    unittest.main()
