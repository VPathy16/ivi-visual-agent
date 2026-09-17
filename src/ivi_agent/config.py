from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class Config:
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3.5:4b"
    max_actions: int = 12
    timeout_seconds: int = 120
    model_timeout_seconds: int = 30
    minimum_action_confidence: float = 0.75
    minimum_success_confidence: float = 0.85
    # Max time to wait for the screen to stop changing after an action. Lower is
    # faster; raise it for animation-heavy IVIs. See settle_poll_seconds.
    settle_timeout_seconds: float = 2.0
    settle_poll_seconds: float = 0.2
    prefer_ui_tree: bool = True
    enable_ocr: bool = True
    max_image_dimension: int = 768
    # Visual grounding strategy when no accessibility candidate exists:
    #   "grid"  -> numbered 12x6 cell overlay (works with any small VLM)
    #   "point" -> ask the VLM for normalized coordinates directly (needs a
    #              grounding-capable model such as qwen3-vl)
    grounding_mode: str = "grid"
    # Trust a concretely-resolved element_id: backfill a missing target label and
    # do not hard-fail on target-mismatch/semantic-relatedness. Needed for small
    # models exploring a benchmark; keep False for the strict goal-driven default.
    lenient_planning: bool = False
    # Ollama context window (num_ctx). A screenshot + UI candidates + history can
    # exceed Ollama's 4096 default, causing HTTP 400 exceed_context_size errors.
    model_context_tokens: int = 8192
    # If True, the AndroidWorld adapter verifies completion before planning on each
    # step of the final subgoal. This ends state-change tasks as soon as the state
    # flips (fewer steps overall), and prevents the agent from re-toggling a control
    # it already set. Default True; set False only for pure navigation runs.
    verify_each_step: bool = True
    allow_text_input: bool = True
    protected_regions: list[list[float]] | None = None
    knowledge_root: str = "knowledge"
    knowledge_profile: str | None = None
    knowledge_top_k: int = 4
    # Semantic retrieval. When enabled, the manual/RAG retriever blends a local
    # text-embedding score with keyword matching (better paraphrase and
    # proprietary-icon recall); it falls back to keyword-only when the embedding
    # backend is unavailable.
    #   embedding_backend "ollama"    -> local Ollama model (needs Ollama running
    #                                    and the model pulled); embedding_model is
    #                                    an Ollama tag, default nomic-embed-text.
    #   embedding_backend "fastembed" -> self-contained ONNX model via the
    #                                    '[embeddings]' extra; no Ollama, no
    #                                    torch, weights auto-downloaded on first
    #                                    use. embedding_model may name a fastembed
    #                                    id (e.g. BAAI/bge-base-en-v1.5) or is
    #                                    ignored in favor of the built-in default.
    # icon_matching adds optional CLIP image matching of a live icon crop against
    # the manual icons (needs the '[clip]' extra).
    use_embeddings: bool = False
    embedding_backend: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    icon_matching: bool = False
    clip_model: str = "ViT-B-32"
    # OpenCV fast-path. When enabled, before calling the (slow) vision model on a
    # step, the agent tries to locate a keyword/semantic-matched manual icon on
    # the live screen with template matching and taps it directly, skipping the
    # model call. It only fires when the retriever ranked the icon at or above
    # cv_min_retrieval_score AND the template match clears cv_match_threshold;
    # otherwise the normal model path runs. Needs the '[cv]' extra and a knowledge
    # profile whose icons have crops. cv_match_threshold should stay >=
    # minimum_action_confidence so a matched tap also passes the safety policy.
    cv_fast_path: bool = False
    cv_match_threshold: float = 0.75
    cv_min_retrieval_score: float = 2.0
    # Structured run trace. When enabled, each run also writes events.jsonl (the
    # ordered event stream), agent.log (human-readable), task.json, and plan.json
    # into the run directory — the backbone for replay, diagnostics, and an
    # engineer report. Best-effort; never aborts a run.
    trace: bool = True
    # Living scene graph. When a knowledge profile is active, seed an expected
    # screen graph from the manual, then confirm nodes/edges as the agent reaches
    # screens and flag screens/transitions that diverge from the manual as
    # pending-review findings (candidate HMI defects). Persisted per profile and
    # grown across runs. No-op without a knowledge profile.
    scene_graph: bool = True

    def __post_init__(self) -> None:
        if self.protected_regions is None:
            self.protected_regions = []

    @classmethod
    def load(cls, path: str | None) -> "Config":
        if not path:
            return cls()
        data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(unknown)}")
        return cls(**data)
