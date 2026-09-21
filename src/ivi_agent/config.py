from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

# Named execution profiles, ARTEMIS-style: a small, coherent bundle of speed vs
# rigour knobs picked with one flag. Only the keys listed here are touched. A
# profile passed on the CLI is a deliberate "run it this way now" intent, so it
# overrides the persistent config.json for these keys (precedence: library
# defaults < config.json < --profile). The untouched grounding ladder
# (a11y -> cv -> model) stays on for every profile.
#   fast     -> minimum overhead: no per-step model verify, no OCR fallback,
#               short settle. The a11y/cv fast paths carry the run.
#   balanced -> library defaults (nothing overridden).
#   strict   -> maximum evidence: verify every step, OCR fallback on, longer
#               settle for animation-heavy screens, higher success bar.
# How hard the agent works to *prove* a goal was reached, trading speed for
# confidence (ARTEMIS-style off/final/checkpoints/strict):
#   off         -> never call the vision model to verify; trust cheap signals
#                  (screen title match, manual success_text cues). Fastest.
#   final       -> cheap signals drive the run; the model confirms only the
#                  OVERALL goal (finish / already-satisfied). One check.
#   checkpoints -> final + the model verifies each step while pursuing the final
#                  subgoal (catches the goal flipping mid-step). (Default.)
#   strict      -> the model verifies every step of every subgoal, so each
#                  milestone is independently proven. Slowest, most defensible.
VERIFICATION_LEVELS = ("off", "final", "checkpoints", "strict")

PROFILES: dict[str, dict[str, Any]] = {
    "fast": {
        "verification_level": "final",
        "enable_ocr": False,
        "settle_timeout_seconds": 1.0,
    },
    "balanced": {},
    "strict": {
        "verification_level": "strict",
        "enable_ocr": True,
        "settle_timeout_seconds": 3.0,
        "minimum_success_confidence": 0.9,
    },
}


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
    # Reuse the post-action screen as the next step's start observation, avoiding
    # a duplicate screenshot + uiautomator dump per step. Set false to force a
    # fresh capture each step.
    reuse_after_state: bool = True
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
    # How hard the agent proves success: off | final | checkpoints | strict
    # (see VERIFICATION_LEVELS). Default "checkpoints" keeps the historical
    # behaviour (model-verify each step of the final subgoal). Set via
    # config.json or --profile (fast->final, strict->strict).
    verification_level: str = "checkpoints"
    # Legacy alias, superseded by verification_level and no longer read by the
    # agent. Kept so older config.json files still load; the AndroidWorld adapter
    # still honours it for its own loop.
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
    # Accessibility fast-path: when the current subgoal's target control name
    # matches a single clickable element in the live UI tree, tap it directly
    # (no model, no CV). The fastest grounding when an a11y tree is present;
    # a no-op (falls through) when nothing matches uniquely.
    accessibility_fast_path: bool = True
    # Structured run trace. When enabled, each run also writes events.jsonl (the
    # ordered event stream), agent.log (human-readable), task.json, and plan.json
    # into the run directory — the backbone for replay, diagnostics, and an
    # engineer report. Best-effort; never aborts a run.
    trace: bool = True
    # Crash/ANR capture. When enabled, the run clears logcat at start, scans it
    # after each action and at the end for fatal events (Java crash, ANR, native
    # signal, process death), writes logcat.txt into the run directory, and lists
    # any crashes in result.json / the report. target_package (e.g. the app under
    # test) attributes and filters events; empty = report all fatal events.
    # fail_on_crash marks the run failed if a crash is detected.
    capture_logs: bool = True
    target_package: str = ""
    fail_on_crash: bool = True
    log_tail_lines: int = 4000
    # Clean-start test hygiene. When enabled (and target_package is set), the run
    # force-stops and relaunches the app before starting, so every validation
    # begins from a known cold state instead of whatever a previous run left
    # behind. Off by default so it never surprises an existing setup; turn it on
    # for trustworthy regression results. No-op without target_package.
    relaunch_before_run: bool = False
    # Living scene graph. When a knowledge profile is active, seed an expected
    # screen graph from the manual, then confirm nodes/edges as the agent reaches
    # screens and flag screens/transitions that diverge from the manual as
    # pending-review findings (candidate HMI defects). Persisted per profile and
    # grown across runs. No-op without a knowledge profile.
    scene_graph: bool = True

    def __post_init__(self) -> None:
        if self.protected_regions is None:
            self.protected_regions = []
        if self.verification_level not in VERIFICATION_LEVELS:
            raise ValueError(
                f"Unknown verification_level: {self.verification_level!r}. "
                f"Choose from {', '.join(VERIFICATION_LEVELS)}."
            )

    def uses_model_verify(self) -> bool:
        """Whether the vision model is used to verify success at all."""
        return self.verification_level != "off"

    def verify_step(self, is_final_subgoal: bool) -> bool:
        """Whether to model-verify the current step, given the subgoal it pursues.

        strict verifies every step; checkpoints verifies only while on the final
        subgoal; final/off never verify per step (they rely on cheap signals and,
        for final, a single overall/finish check).
        """
        if self.verification_level == "strict":
            return True
        if self.verification_level == "checkpoints":
            return is_final_subgoal
        return False

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

    def apply_profile(self, name: str | None) -> "Config":
        """Apply a named execution profile in place; return self for chaining.

        A profile is a deliberate CLI intent, so it overrides config.json for the
        keys it owns (see PROFILES). ``None`` is a no-op; an unknown name raises.
        """
        if not name:
            return self
        if name not in PROFILES:
            raise ValueError(
                f"Unknown profile: {name!r}. Choose from {', '.join(sorted(PROFILES))}."
            )
        for key, value in PROFILES[name].items():
            setattr(self, key, value)
        return self
