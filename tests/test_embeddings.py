import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ivi_agent import embeddings as emb
from ivi_agent.embeddings import (
    FASTEMBED_DEFAULT_MODEL,
    FastEmbedTextEmbedder,
    cosine,
    resolve_text_embedder,
)
from ivi_agent.knowledge import KnowledgeBase, _embed_chunks


class FakeTextEmbedder:
    """Deterministic 3-d embeddings keyed by topic words (bt / wifi / other)."""

    def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if any(w in lowered for w in ("bluetooth", "bt", "audio")):
                vectors.append([1.0, 0.0, 0.0])
            elif any(w in lowered for w in ("wifi", "wireless", "internet", "network")):
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors

    def available(self):
        return True


class FakeImageEmbedder:
    def __init__(self, vector):
        self.vector = vector

    def embed_images(self, images):
        return [list(self.vector) for _ in images]


def _write_profile(directory: Path, chunks: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "profile": directory.name, "manual_id": "m"}),
        encoding="utf-8",
    )
    (directory / "chunks.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8"
    )


def _png(color) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buf, format="PNG")
    return buf.getvalue()


class CosineTests(unittest.TestCase):
    def test_identical_and_orthogonal_and_empty(self) -> None:
        self.assertAlmostEqual(cosine([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)
        self.assertEqual(cosine([], [1, 2]), 0.0)
        self.assertEqual(cosine([0, 0], [1, 1]), 0.0)


class SemanticQueryTests(unittest.TestCase):
    def _base(self, embedder=None) -> KnowledgeBase:
        tmp = Path(tempfile.mkdtemp())
        _write_profile(
            tmp / "p",
            [
                {"id": "icon.bt", "kind": "icon", "name": "LinkWave",
                 "text": "linkwave bt audio bluetooth", "embedding": [1.0, 0.0, 0.0]},
                {"id": "icon.wifi", "kind": "icon", "name": "Wi-Fi",
                 "text": "wifi network settings", "embedding": [0.0, 1.0, 0.0]},
            ],
        )
        return KnowledgeBase.open(tmp, "p", embedder=embedder)

    def test_semantic_recall_beyond_keywords(self) -> None:
        # Query shares NO tokens with either chunk, so keyword score is 0 for both;
        # only the embedder can surface the Wi-Fi chunk.
        kb = self._base(embedder=FakeTextEmbedder())
        self.assertTrue(kb.has_text_embeddings)
        result = kb.query("wireless internet connection", limit=2)
        ids = [c["id"] for c in result["chunks"]]
        self.assertIn("icon.wifi", ids)
        self.assertEqual(ids[0], "icon.wifi")

    def test_keyword_only_without_embedder_returns_nothing_for_paraphrase(self) -> None:
        kb = self._base(embedder=None)
        result = kb.query("wireless internet connection", limit=2)
        self.assertEqual(result["chunks"], [])


class MatchIconTests(unittest.TestCase):
    def _base(self) -> KnowledgeBase:
        tmp = Path(tempfile.mkdtemp())
        _write_profile(
            tmp / "p",
            [
                {"id": "icon.bt", "kind": "icon", "name": "BT", "image": "bt.png",
                 "image_embedding": [1.0, 0.0, 0.0]},
                {"id": "icon.wifi", "kind": "icon", "name": "WiFi", "image": "wifi.png",
                 "image_embedding": [0.0, 1.0, 0.0]},
            ],
        )
        return KnowledgeBase.open(tmp, "p")

    def test_matches_closest_icon(self) -> None:
        kb = self._base()
        best = kb.match_icon(_png("blue"), FakeImageEmbedder([0.9, 0.1, 0.0]), top_k=1)
        self.assertEqual(len(best), 1)
        self.assertEqual(best[0]["id"], "icon.bt")
        self.assertNotIn("image_embedding", best[0])  # stripped from output

    def test_returns_empty_without_embedder(self) -> None:
        self.assertEqual(self._base().match_icon(_png("red"), None), [])


class EmbedChunksTests(unittest.TestCase):
    def test_text_embeddings_attached(self) -> None:
        chunks = [{"text": "bluetooth audio"}, {"text": "wifi network"}]
        written = _embed_chunks(chunks, Path("."), FakeTextEmbedder(), None)
        self.assertTrue(written["text"])
        self.assertEqual(chunks[0]["embedding"], [1.0, 0.0, 0.0])
        self.assertEqual(chunks[1]["embedding"], [0.0, 1.0, 0.0])

    def test_icon_image_embeddings_attached(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        (tmp / "assets").mkdir()
        (tmp / "assets" / "bt.png").write_bytes(_png("blue"))
        chunks = [{"kind": "icon", "text": "bt", "image": "assets/bt.png"}]
        written = _embed_chunks(chunks, tmp, None, FakeImageEmbedder([0.5, 0.5, 0.0]))
        self.assertTrue(written["image"])
        self.assertEqual(chunks[0]["image_embedding"], [0.5, 0.5, 0.0])


class ResolveTextEmbedderTests(unittest.TestCase):
    def test_disabled_returns_none(self) -> None:
        self.assertIsNone(
            resolve_text_embedder("http://x", "nomic-embed-text", False, "fastembed")
        )
        self.assertIsNone(
            resolve_text_embedder("http://x", "nomic-embed-text", False, "ollama")
        )

    def test_fastembed_unavailable_falls_back_to_none(self) -> None:
        # Simulate the [embeddings] extra not being installed.
        original = FastEmbedTextEmbedder.available
        FastEmbedTextEmbedder.available = staticmethod(lambda: False)  # type: ignore
        try:
            self.assertIsNone(
                resolve_text_embedder("http://x", "BAAI/bge-small-en-v1.5", True, "fastembed")
            )
        finally:
            FastEmbedTextEmbedder.available = original  # type: ignore

    def test_fastembed_selected_and_probed(self) -> None:
        # Available + probes cleanly -> the constructed embedder is returned.
        built = {}

        class FakeFast:
            def __init__(self, model_name):
                built["model"] = model_name

            def embed(self, texts):
                return [[0.1, 0.2, 0.3] for _ in texts]

            @staticmethod
            def available():
                return True

        original = emb.FastEmbedTextEmbedder
        emb.FastEmbedTextEmbedder = FakeFast  # type: ignore
        try:
            # An Ollama-style model id must NOT leak into fastembed; it uses the default.
            got = resolve_text_embedder("http://x", "nomic-embed-text", True, "fastembed")
            self.assertIsInstance(got, FakeFast)
            self.assertEqual(built["model"], FASTEMBED_DEFAULT_MODEL)
            # A fastembed-style id (contains "/") is passed through.
            resolve_text_embedder("http://x", "BAAI/bge-base-en-v1.5", True, "fastembed")
            self.assertEqual(built["model"], "BAAI/bge-base-en-v1.5")
        finally:
            emb.FastEmbedTextEmbedder = original  # type: ignore

    def test_fastembed_probe_failure_returns_none(self) -> None:
        class BadFast:
            def __init__(self, model_name):
                pass

            def embed(self, texts):
                raise emb.EmbeddingError("boom")

            @staticmethod
            def available():
                return True

        original = emb.FastEmbedTextEmbedder
        emb.FastEmbedTextEmbedder = BadFast  # type: ignore
        try:
            self.assertIsNone(
                resolve_text_embedder("http://x", "x/y", True, "fastembed")
            )
        finally:
            emb.FastEmbedTextEmbedder = original  # type: ignore


if __name__ == "__main__":
    unittest.main()
