from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ActionKind = Literal[
    "tap",
    "long_press",
    "double_tap",
    "input_text",
    "keyboard_enter",
    "gesture",
    "swipe",
    "open_app",
    "back",
    "home",
    "wait",
    "text",
    "finish",
]
Outcome = Literal["pass", "fail", "inconclusive"]


@dataclass
class Action:
    type: ActionKind
    confidence: float
    reason: str
    target: str = ""
    element_id: int | None = None
    direction: str = ""
    region: str = "center"
    x: float | None = None
    y: float | None = None
    x2: float | None = None
    y2: float | None = None
    duration_ms: int = 500
    seconds: float = 1.0
    text: str = ""
    app_name: str = ""
    outcome: Outcome | None = None
    evidence: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Action":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items() if key in allowed})


@dataclass
class StepRecord:
    number: int
    screenshot: str
    action: Action
    ui_dump_available: bool
    screen_changed: bool | None = None
    decision_seconds: float | None = None
    # How the action was grounded: "cv" (OpenCV fast-path, no model call) or
    # "model" (vision-model planner/grounder).
    grounded_by: str = "model"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SubgoalRecord:
    number: int
    description: str
    status: Literal["pending", "running", "passed", "failed"] = "pending"
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunResult:
    goal: str
    outcome: Outcome
    reason: str
    run_directory: str
    knowledge: dict[str, Any] = field(default_factory=dict)
    subgoals: list[SubgoalRecord] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    # Populated at finish: counts of how steps were grounded and decision time.
    grounding: dict[str, Any] = field(default_factory=dict)
    # Populated at finish when a scene graph is active: coverage + pending
    # divergence findings (candidate HMI defects awaiting user review).
    scene_graph: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
