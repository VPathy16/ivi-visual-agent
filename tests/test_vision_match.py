import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ivi_agent.vision_match import (
    available,
    cv_ground_from_knowledge,
    locate_template,
)


def _needs_cv():
    return unittest.skipUnless(available(), "OpenCV ([cv] extra) not installed")


def _png(array) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array, "L").save(buffer, format="PNG")
    return buffer.getvalue()


def _patch():
    """A structured 40x40 icon: gradient + centered block + border.

    Structured (not random) so it survives resizing, like a real icon crop.
    """
    import numpy as np

    patch = np.tile(np.linspace(0, 255, 40, dtype=np.uint8), (40, 1))
    patch[8:32, 8:32] = 255
    patch[16:24, 16:24] = 0
    patch[0, :] = patch[-1, :] = patch[:, 0] = patch[:, -1] = 0
    return patch


def _scene():
    """A 400x800 screen with the structured patch at top=100, left=300.

    Expected normalized center: x = (300+20)/800 = 0.4, y = (100+20)/400 = 0.3.
    """
    import numpy as np

    screen = np.full((400, 800), 128, dtype=np.uint8)
    patch = _patch()
    screen[100:140, 300:340] = patch
    return screen, patch


@_needs_cv()
class LocateTemplateTests(unittest.TestCase):
    def test_finds_patch_at_expected_center(self) -> None:
        screen, patch = _scene()
        match = locate_template(_png(screen), _png(patch), threshold=0.75)
        self.assertIsNotNone(match)
        score, cx, cy, _box = match
        self.assertGreaterEqual(score, 0.9)
        self.assertAlmostEqual(cx, 0.4, delta=0.02)
        self.assertAlmostEqual(cy, 0.3, delta=0.02)

    def test_absent_template_returns_none(self) -> None:
        import numpy as np

        screen, _p = _scene()
        rng = np.random.default_rng(99)
        other = rng.integers(0, 255, (40, 40), dtype=np.uint8)
        self.assertIsNone(locate_template(_png(screen), _png(other), threshold=0.75))

    def test_scaled_patch_still_found(self) -> None:
        import cv2  # type: ignore

        screen, patch = _scene()
        bigger = cv2.resize(patch, (52, 52), interpolation=cv2.INTER_LINEAR)  # 1.3x
        match = locate_template(_png(screen), _png(bigger), threshold=0.6)
        self.assertIsNotNone(match)  # multi-scale sweep recovers it


@_needs_cv()
class CvGroundFromKnowledgeTests(unittest.TestCase):
    def _knowledge(self, path: Path, score: float) -> dict:
        return {"chunks": [{"kind": "icon", "name": "A/C", "image": str(path), "score": score}]}

    def _config(self):
        return SimpleNamespace(cv_min_retrieval_score=2.0, cv_match_threshold=0.75)

    def test_taps_matched_icon(self) -> None:
        screen, patch = _scene()
        tmp = Path(tempfile.mkdtemp()) / "ac.png"
        tmp.write_bytes(_png(patch))
        action = cv_ground_from_knowledge(_png(screen), self._knowledge(tmp, 5.0), self._config())
        self.assertIsNotNone(action)
        self.assertEqual(action.type, "tap")
        self.assertEqual(action.target, "A/C")
        self.assertAlmostEqual(action.x, 0.4, delta=0.03)
        self.assertAlmostEqual(action.y, 0.3, delta=0.03)
        self.assertGreaterEqual(action.confidence, 0.75)

    def test_low_retrieval_score_is_ignored(self) -> None:
        screen, patch = _scene()
        tmp = Path(tempfile.mkdtemp()) / "ac.png"
        tmp.write_bytes(_png(patch))
        # Icon crop IS on screen, but the retriever didn't rank it -> no fast path.
        self.assertIsNone(
            cv_ground_from_knowledge(_png(screen), self._knowledge(tmp, 1.0), self._config())
        )

    def test_no_icon_chunks_returns_none(self) -> None:
        screen, _patch = _scene()
        knowledge = {"chunks": [{"kind": "screen", "name": "Home", "score": 9.0}]}
        self.assertIsNone(cv_ground_from_knowledge(_png(screen), knowledge, self._config()))


if __name__ == "__main__":
    unittest.main()
