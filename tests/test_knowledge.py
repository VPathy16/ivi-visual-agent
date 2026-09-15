import json
import tempfile
import unittest
from pathlib import Path

from ivi_agent.knowledge import KnowledgeBase, KnowledgeError, prompt_context
from ivi_agent.model import OllamaVisionModel


class KnowledgeTests(unittest.TestCase):
    def make_profile(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        profile = Path(temporary.name) / "sample"
        profile.mkdir()
        (profile / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "profile": "sample",
                    "manual_id": "manual-v1",
                }
            ),
            encoding="utf-8",
        )
        chunks = [
            {
                "id": "task.bluetooth",
                "kind": "task",
                "name": "Select Bluetooth source",
                "text": "Select Bluetooth as the media source",
                "data": {
                    "goal": "Select Bluetooth as the media source",
                    "steps": [
                        {"expected": "Audio Hub is visible"},
                        {"expected": "Bluetooth is selected"},
                    ],
                },
            },
            {
                "id": "screen.sources",
                "kind": "screen",
                "name": "Sources",
                "text": "Audio source screen containing BT Audio",
                "data": {},
            },
            {
                "id": "icon.linkwave",
                "kind": "icon",
                "name": "LinkWave",
                "text": "Proprietary LinkWave symbol means Bluetooth audio",
                "data": {},
            },
            {
                "id": "pdf.page.7",
                "kind": "pdf_page",
                "name": "Page 7",
                "text": "Bluetooth source selection procedure",
                "page": 7,
            },
            {
                "id": "screen.unrelated",
                "kind": "screen",
                "name": "Climate",
                "text": "Climate fan temperature",
                "data": {},
            },
        ]
        (profile / "chunks.jsonl").write_text(
            "".join(json.dumps(chunk) + "\n" for chunk in chunks), encoding="utf-8"
        )
        return temporary, profile

    def test_query_retrieves_diverse_relevant_chunks(self) -> None:
        temporary, profile = self.make_profile()
        self.addCleanup(temporary.cleanup)
        result = KnowledgeBase(profile).query("Select Bluetooth audio source", limit=4)
        kinds = {chunk["kind"] for chunk in result["chunks"]}
        self.assertEqual(kinds, {"task", "screen", "icon", "pdf_page"})
        self.assertNotIn(
            "screen.unrelated", {chunk["id"] for chunk in result["chunks"]}
        )

    def test_prompt_context_excludes_local_image_paths(self) -> None:
        result = {
            "profile": "sample",
            "manual_id": "manual-v1",
            "chunks": [
                {
                    "id": "icon.bt",
                    "kind": "icon",
                    "text": "Bluetooth",
                    "image": "/private/icon.png",
                    "score": 2.0,
                }
            ],
        }
        context = prompt_context(result)
        self.assertNotIn("image", context["retrieved_chunks"][0])

    def test_prompt_context_keeps_only_the_active_task_step(self) -> None:
        temporary, profile = self.make_profile()
        self.addCleanup(temporary.cleanup)
        result = KnowledgeBase(profile).query("Select Bluetooth source")
        context = prompt_context(result, "Audio Hub is visible")
        task = next(
            chunk for chunk in context["retrieved_chunks"] if chunk["kind"] == "task"
        )
        self.assertEqual(
            task["data"]["active_step"]["expected"], "Audio Hub is visible"
        )
        self.assertNotIn("Bluetooth is selected", task["text"])

    def test_documented_task_creates_observable_route(self) -> None:
        temporary, profile = self.make_profile()
        self.addCleanup(temporary.cleanup)
        context = prompt_context(
            KnowledgeBase(profile).query("Select Bluetooth as the media source")
        )
        self.assertEqual(
            OllamaVisionModel._documented_subgoals(
                "Select Bluetooth as the media source", context
            ),
            ["Audio Hub is visible", "Select Bluetooth as the media source"],
        )

    def test_rejects_invalid_profile_name(self) -> None:
        with self.assertRaisesRegex(KnowledgeError, "profile must use"):
            KnowledgeBase.open(Path("knowledge"), "../escape")


if __name__ == "__main__":
    unittest.main()
