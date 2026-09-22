"""VisionLaya decision schema and the training Example.

Two decision tasks, both non-autoregressive (one forward pass, calibrated
probability), mirroring Laya's primitives:

* VERIFY  (bool)   — "is this screen the goal state?" -> pass/fail + confidence.
* GROUND  (choice) — "which control advances the goal?" -> a target + point.

An Example is one labeled training row exported from a real agent run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

VERIFY = "verify"
GROUND = "ground"
TASKS = (VERIFY, GROUND)


@dataclass
class Example:
    """One labeled training row derived from an agent run.

    Common:
        task       : "verify" | "ground"
        image      : path to the screenshot this decision is about
        goal       : the run goal (plain language)
        run_id     : originating run id
        source     : originating run directory
        grounded_by: how the agent made this decision ("a11y" | "cv" | "model"),
                     a provenance/quality signal for weighting examples.

    VERIFY:
        satisfied  : True if this screen is the goal state, else False.

    GROUND:
        instruction: what the agent was trying to do at this step.
        target     : the control label it tapped (the correct choice).
        point      : normalized (x, y) tap location, or None.
    """

    task: str
    image: str
    goal: str
    run_id: str = ""
    source: str = ""
    grounded_by: str = ""
    # verify
    satisfied: bool | None = None
    # ground
    instruction: str = ""
    target: str = ""
    point: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("point") is not None:
            data["point"] = list(data["point"])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Example":
        point = data.get("point")
        return cls(
            task=data["task"],
            image=data.get("image", ""),
            goal=data.get("goal", ""),
            run_id=data.get("run_id", ""),
            source=data.get("source", ""),
            grounded_by=data.get("grounded_by", ""),
            satisfied=data.get("satisfied"),
            instruction=data.get("instruction", ""),
            target=data.get("target", ""),
            point=tuple(point) if isinstance(point, (list, tuple)) else None,
        )
