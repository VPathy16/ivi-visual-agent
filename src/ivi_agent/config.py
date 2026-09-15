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
    settle_timeout_seconds: float = 5.0
    prefer_ui_tree: bool = True
    enable_ocr: bool = True
    max_image_dimension: int = 768
    allow_text_input: bool = True
    protected_regions: list[list[float]] | None = None

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
