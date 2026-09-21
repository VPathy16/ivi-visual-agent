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


class PromotionTests(unittest.TestCase):
    def _profile(self, tmp: Path) -> Path:
        profile = tmp / "benz"
        profile.mkdir()
        (profile / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "profile": "benz", "manual_id": "m1"}),
            encoding="utf-8",
        )
        chunks = [
            {"id": "screen.climate", "kind": "screen", "name": "Climate",
             "text": "Climate", "data": {"id": "screen.climate", "name": "Climate", "controls": []}},
            {"id": "screen.seat_massage", "kind": "screen", "name": "Seat massage",
             "text": "Seat massage", "data": {"id": "screen.seat_massage", "name": "Seat massage", "controls": []}},
        ]
        (profile / "chunks.jsonl").write_text(
            "\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8"
        )
        return profile

    def test_document_transition_persists_and_reloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._profile(root)
            kb = KnowledgeBase.open(root, "benz")
            self.assertTrue(
                kb.document_transition("screen.climate", "screen.seat_massage",
                                       "Seat massage", "~M~")
            )
            # A fresh load (next run) sees the learned control on the source screen.
            reloaded = KnowledgeBase.open(root, "benz")
            climate = reloaded._screen_chunk("screen.climate")
            controls = climate["data"]["controls"]
            self.assertEqual(len(controls), 1)
            self.assertEqual(controls[0]["result"], "screen.seat_massage")
            self.assertEqual(controls[0]["name"], "~M~")
            self.assertIn("~M~", climate["text"])  # retrieval can now surface it

    def test_document_transition_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._profile(root)
            kb = KnowledgeBase.open(root, "benz")
            kb.document_transition("screen.climate", "screen.seat_massage", "Seat massage", "~M~")
            # Second approval of the same route adds no duplicate control.
            kb.document_transition("screen.climate", "screen.seat_massage", "Seat massage", "~M~")
            climate = KnowledgeBase.open(root, "benz")._screen_chunk("screen.climate")
            self.assertEqual(len(climate["data"]["controls"]), 1)

    def test_document_screen_adds_new_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._profile(root)
            kb = KnowledgeBase.open(root, "benz")
            self.assertTrue(kb.document_screen("screen.vehicle", "Vehicle", ["Vehicle"]))
            self.assertFalse(kb.document_screen("screen.vehicle", "Vehicle"))  # idempotent
            reloaded = KnowledgeBase.open(root, "benz")
            self.assertIsNotNone(reloaded._screen_chunk("screen.vehicle"))


if __name__ == "__main__":
    unittest.main()
