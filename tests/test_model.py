import unittest

from ivi_agent.model import OllamaVisionModel


class ModelResponseTests(unittest.TestCase):
    def test_extracts_json_from_plain_response(self) -> None:
        value = OllamaVisionModel._extract_json(
            '{"type":"back","confidence":0.9,"reason":"go back"}'
        )
        self.assertEqual(value["type"], "back")

    def test_extracts_json_from_code_fence(self) -> None:
        value = OllamaVisionModel._extract_json('```json\n{"outcome":"pass"}\n```')
        self.assertEqual(value["outcome"], "pass")


if __name__ == "__main__":
    unittest.main()
