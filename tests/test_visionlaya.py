import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visionlaya.dataset import export_run, export_dataset, read_jsonl, write_jsonl
from visionlaya.hf import map_androidcontrol, map_screenspot
from visionlaya.schema import Example, GROUND, VERIFY


def _png(path: Path) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (32, 48), "black").save(buf, format="PNG")
    path.write_bytes(buf.getvalue())


def _make_run(root: Path, name: str, outcome: str = "pass") -> Path:
    run = root / name
    run.mkdir()
    _png(run / "step-01.png")
    _png(run / "step-01-after.png")
    _png(run / "step-02.png")
    _png(run / "step-02-after.png")
    result = {
        "goal": "Start the seat massage",
        "outcome": outcome,
        "steps": [
            {"number": 1, "screenshot": "step-01.png", "grounded_by": "model",
             "action": {"type": "tap", "target": "~M~", "x": 0.5, "y": 0.34,
                        "reason": "Tap the Seat Comfort tile"}},
            {"number": 2, "screenshot": "step-02.png", "grounded_by": "a11y",
             "action": {"type": "tap", "target": "start", "x": 0.09, "y": 0.58,
                        "reason": "accessibility tree match ['start']"}},
        ],
    }
    (run / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return run


class SchemaTests(unittest.TestCase):
    def test_example_roundtrips(self) -> None:
        ex = Example(task=GROUND, image="a.png", goal="g", target="~M~", point=(0.5, 0.3))
        back = Example.from_dict(json.loads(json.dumps(ex.to_dict())))
        self.assertEqual(back.point, (0.5, 0.3))
        self.assertEqual(back.target, "~M~")


class ExportTests(unittest.TestCase):
    def test_passed_run_yields_verify_and_ground(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _make_run(Path(tmp), "run1")
            examples = export_run(run)
            verify = [e for e in examples if e.task == VERIFY]
            ground = [e for e in examples if e.task == GROUND]
            # one positive (final after-image) + one negative (initial before)
            self.assertEqual({e.satisfied for e in verify}, {True, False})
            self.assertTrue(any(e.image.endswith("step-02-after.png") and e.satisfied for e in verify))
            # a ground example per tap, with target + point + provenance
            self.assertEqual(len(ground), 2)
            m = next(e for e in ground if e.target == "~M~")
            self.assertEqual(m.point, (0.5, 0.34))
            self.assertEqual(m.grounded_by, "model")

    def test_failed_run_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = _make_run(Path(tmp), "bad", outcome="inconclusive")
            self.assertEqual(export_run(run), [])

    def test_export_dataset_and_jsonl_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run(root, "run1")
            _make_run(root, "run2")
            examples = export_dataset(root)
            self.assertTrue(len(examples) >= 4)
            out = root / "data.jsonl"
            self.assertEqual(write_jsonl(examples, out), len(examples))
            reloaded = read_jsonl(out)
            self.assertEqual(len(reloaded), len(examples))
            self.assertEqual(reloaded[0].to_dict(), examples[0].to_dict())

    def test_missing_root(self) -> None:
        self.assertEqual(export_dataset(Path("/no/such/runs")), [])


class HfMapperTests(unittest.TestCase):
    def test_androidcontrol_tap_maps_to_point(self) -> None:
        row = {"goal": "Open settings", "instruction": "tap the gear",
               "action": {"x": 0.5, "y": 0.25}}
        fields = map_androidcontrol(row)
        self.assertEqual(fields["point"], (0.5, 0.25))
        self.assertEqual(fields["instruction"], "tap the gear")

    def test_androidcontrol_pixel_coords_normalized(self) -> None:
        row = {"instruction": "tap x", "action": {"touch_x": 540, "touch_y": 600},
               "width": 1080, "height": 2400}
        self.assertEqual(map_androidcontrol(row)["point"], (0.5, 0.25))

    def test_androidcontrol_non_tap_skipped(self) -> None:
        self.assertIsNone(map_androidcontrol({"instruction": "swipe", "action": "SCROLL_DOWN"}))
        self.assertIsNone(map_androidcontrol({"instruction": "nothing"}))

    def test_screenspot_bbox_center(self) -> None:
        row = {"instruction": "the wifi icon", "bbox": [0.4, 0.2, 0.6, 0.4]}
        self.assertEqual(map_screenspot(row)["point"], (0.5, 0.3))

    def test_screenspot_requires_instruction(self) -> None:
        self.assertIsNone(map_screenspot({"bbox": [0, 0, 1, 1]}))


if __name__ == "__main__":
    unittest.main()
