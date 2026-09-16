"""Local embedding backends for semantic knowledge retrieval.

Pluggable, optional backends upgrade the manual/RAG retriever from keyword
matching to semantic matching, while staying local:

* :class:`OllamaTextEmbedder` — text embeddings via a local Ollama model
  (default ``nomic-embed-text``). Uses only ``urllib`` (no new dependency), the
  same transport as :mod:`ivi_agent.model`. Requires a running Ollama with the
  model pulled.
* :class:`FastEmbedTextEmbedder` — text embeddings via ``fastembed`` (ONNX
  runtime, no ``torch``, no separate server). Pip-installable with the
  ``[embeddings]`` extra; the small model is fetched automatically on first use
  and cached. Use this when you don't want to run Ollama or pull a model.
* :class:`ClipImageEmbedder` — CLIP image/text embeddings for matching a live
  icon crop against the manual's reference icons. Needs the optional ``[clip]``
  extra (``open-clip-torch`` + ``torch``); import is guarded so the core stays
  lightweight.

All are optional: when unavailable, the retriever falls back to the existing
keyword/TF-IDF ranking, so nothing breaks.
"""

from __future__ import annotations

import http.client
import importlib.util
import io
import json
import math
import urllib.error
import urllib.request
from typing import Any


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 if either is empty)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class EmbeddingError(RuntimeError):
    pass


class OllamaTextEmbedder:
    """Text embeddings from a local Ollama model via ``/api/embed``."""

    def __init__(
        self,
        base_url: str,
        model: str = "nomic-embed-text",
        timeout: float = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.model, "input": texts}
        request = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace").strip()
            raise EmbeddingError(
                f"Ollama embeddings returned HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except (urllib.error.URLError, http.client.HTTPException) as exc:
            raise EmbeddingError(
                f"Could not reach Ollama embeddings at {self.base_url}: {exc}"
            ) from exc
        vectors = result.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            # Some Ollama versions return {"embedding": [...]} for a single input.
            single = result.get("embedding")
            if isinstance(single, list) and len(texts) == 1:
                return [[float(v) for v in single]]
            raise EmbeddingError(f"Unexpected Ollama embeddings response: {result}")
        return [[float(v) for v in vector] for vector in vectors]

    def available(self) -> bool:
        try:
            return bool(self.embed(["ping"])[0])
        except Exception:  # noqa: BLE001 - availability probe
            return False


#: Default fastembed model — small (~130 MB), CPU-friendly, 384-dim, strong
#: retrieval quality. Overridable via ``embedding_model`` when it names a
#: fastembed model id (e.g. ``BAAI/bge-base-en-v1.5``).
FASTEMBED_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedTextEmbedder:
    """Text embeddings via ``fastembed`` — no Ollama, no torch, no server.

    ``fastembed`` runs the model on the ONNX runtime and downloads/caches the
    weights on first use, so semantic retrieval works from a plain
    ``pip install`` without pulling an Ollama model.
    """

    def __init__(self, model_name: str = FASTEMBED_DEFAULT_MODEL) -> None:
        try:
            from fastembed import TextEmbedding  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise EmbeddingError(
                "fastembed embeddings need the optional extra: "
                "python -m pip install -e '.[embeddings]'"
            ) from exc
        self.model_name = model_name
        try:
            self._model = TextEmbedding(model_name=model_name)
        except Exception as exc:  # noqa: BLE001 - bad name / download failure
            raise EmbeddingError(
                f"Could not load fastembed model {model_name!r}: {exc}"
            ) from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = list(self._model.embed(list(texts)))
        except Exception as exc:  # noqa: BLE001 - runtime embedding failure
            raise EmbeddingError(f"fastembed embedding failed: {exc}") from exc
        return [[float(value) for value in vector] for vector in vectors]

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("fastembed") is not None


class ClipImageEmbedder:
    """CLIP image/text embeddings (optional; needs the ``[clip]`` extra)."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
    ) -> None:
        try:
            import open_clip  # type: ignore
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise EmbeddingError(
                "CLIP icon matching needs the optional extra: "
                "python -m pip install -e '.[clip]'"
            ) from exc
        self._torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    @staticmethod
    def available() -> bool:
        return (
            importlib.util.find_spec("open_clip") is not None
            and importlib.util.find_spec("torch") is not None
        )

    def embed_images(self, images: list[bytes]) -> list[list[float]]:
        if not images:
            return []
        from PIL import Image  # local import; Pillow is a core dependency

        torch = self._torch
        tensors = torch.stack(
            [
                self.preprocess(Image.open(io.BytesIO(data)).convert("RGB"))
                for data in images
            ]
        )
        with torch.no_grad():
            features = self.model.encode_image(tensors)
            features = features / features.norm(dim=-1, keepdim=True)
        return [row.tolist() for row in features]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        torch = self._torch
        tokens = self.tokenizer(texts)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return [row.tolist() for row in features]


def resolve_text_embedder(
    base_url: str, model: str, enabled: bool, backend: str = "ollama"
) -> OllamaTextEmbedder | FastEmbedTextEmbedder | None:
    """Return a ready text embedder, or None when disabled/unavailable.

    ``backend`` selects the implementation:

    * ``"ollama"`` (default) — needs a running Ollama with ``model`` pulled.
    * ``"fastembed"`` — self-contained; needs the ``[embeddings]`` extra. Uses
      ``model`` when it names a fastembed model id (contains ``/``), else the
      built-in default, so an Ollama-style ``embedding_model`` doesn't leak in.

    Any failure (disabled, missing dependency, unreachable server, download
    error) returns ``None`` so the retriever falls back to keyword ranking.
    """
    if not enabled:
        return None
    if backend == "fastembed":
        if not FastEmbedTextEmbedder.available():
            return None
        fast_model = model if "/" in model else FASTEMBED_DEFAULT_MODEL
        try:
            fast = FastEmbedTextEmbedder(fast_model)
            return fast if fast.embed(["ping"])[0] else None
        except (EmbeddingError, IndexError):
            return None
    embedder = OllamaTextEmbedder(base_url, model)
    return embedder if embedder.available() else None


def resolve_image_embedder(model_name: str, enabled: bool) -> Any | None:
    """Return a CLIP image embedder, or None when disabled/unavailable."""
    if not enabled or not ClipImageEmbedder.available():
        return None
    try:
        return ClipImageEmbedder(model_name)
    except EmbeddingError:
        return None
