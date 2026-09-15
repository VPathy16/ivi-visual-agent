import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ivi_agent.manual_pdf import (
    ManualValidationError,
    build_manual_pdf,
    load_manual_source,
)


class ManualSourceTests(unittest.TestCase):
    def make_source(self) -> tuple[tempfile.TemporaryDirectory[str], Path, dict]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        images = root / "images"
        images.mkdir()
        Image.new("RGB", (40, 30), "blue").save(images / "screen.png")
        Image.new("RGBA", (20, 20), "green").save(images / "icon.png")
        manifest = {
            "schema_version": 1,
            "manual": {
                "id": "test-manual",
                "title": "Test manual",
                "ui_profile": "Test UI",
                "version": "1.0",
                "description": "Test description",
            },
            "icons": [
                {
                    "id": "bluetooth",
                    "name": "Bluetooth",
                    "image": "images/icon.png",
                    "meaning": "Bluetooth source",
                    "opens": "action.select_bluetooth",
                    "synonyms": ["BT"],
                }
            ],
            "screens": [
                {
                    "id": "screen.home",
                    "name": "Home",
                    "image": "images/screen.png",
                    "description": "Home screen",
                    "landmarks": ["Tile grid"],
                    "controls": [
                        {
                            "id": "home.bluetooth",
                            "name": "Bluetooth tile",
                            "icon_id": "bluetooth",
                            "action": "Select it",
                            "result": "Bluetooth is selected",
                        }
                    ],
                }
            ],
            "tasks": [
                {
                    "id": "task.bluetooth",
                    "name": "Select Bluetooth",
                    "goal": "Select Bluetooth",
                    "steps": [
                        {
                            "screen": "screen.home",
                            "target": "home.bluetooth",
                            "instruction": "Select the Bluetooth tile",
                            "expected": "Bluetooth is selected",
                        }
                    ],
                    "success": ["Bluetooth is selected"],
                    "forbidden": ["Do not pair a device"],
                }
            ],
        }
        return temporary, root, manifest

    def write_manifest(self, root: Path, manifest: dict) -> None:
        (root / "manual.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_loads_valid_manual_folder(self) -> None:
        temporary, root, manifest = self.make_source()
        self.addCleanup(temporary.cleanup)
        self.write_manifest(root, manifest)
        loaded = load_manual_source(root)
        self.assertEqual(loaded["manual"]["id"], "test-manual")
        self.assertTrue(Path(loaded["screens"][0]["_image_path"]).is_file())

    def test_rejects_missing_image(self) -> None:
        temporary, root, manifest = self.make_source()
        self.addCleanup(temporary.cleanup)
        manifest["icons"][0]["image"] = "images/missing.png"
        self.write_manifest(root, manifest)
        with self.assertRaisesRegex(ManualValidationError, "was not found"):
            load_manual_source(root)

    def test_rejects_image_path_outside_source_folder(self) -> None:
        temporary, root, manifest = self.make_source()
        self.addCleanup(temporary.cleanup)
        manifest["icons"][0]["image"] = "../outside.png"
        self.write_manifest(root, manifest)
        with self.assertRaisesRegex(ManualValidationError, "remain inside"):
            load_manual_source(root)

    def test_rejects_duplicate_ids(self) -> None:
        temporary, root, manifest = self.make_source()
        self.addCleanup(temporary.cleanup)
        manifest["screens"][0]["id"] = "bluetooth"
        self.write_manifest(root, manifest)
        with self.assertRaisesRegex(ManualValidationError, "duplicate id"):
            load_manual_source(root)

    def test_rejects_unknown_task_target(self) -> None:
        temporary, root, manifest = self.make_source()
        self.addCleanup(temporary.cleanup)
        manifest["tasks"][0]["steps"][0]["target"] = "unknown"
        self.write_manifest(root, manifest)
        with self.assertRaisesRegex(ManualValidationError, "unknown control or icon"):
            load_manual_source(root)

    def test_generated_pdf_embeds_manifest_and_reference_images(self) -> None:
        try:
            from pypdf import PdfReader
            import reportlab  # noqa: F401
        except ImportError:
            self.skipTest("optional PDF dependencies are not installed")
        temporary, root, manifest = self.make_source()
        self.addCleanup(temporary.cleanup)
        self.write_manifest(root, manifest)
        output = root / "manual.pdf"
        build_manual_pdf(root, output)
        attachments = PdfReader(output).attachments
        self.assertIn("manual.json", attachments)
        self.assertIn("images/icon.png", attachments)
        self.assertIn("images/screen.png", attachments)


if __name__ == "__main__":
    unittest.main()
