"""Turn agent runs into a VisionLaya training set.

Every run directory already holds screenshots + ``result.json`` (goal, steps,
outcome). This module reads those and emits labeled ``Example`` rows — no ML,
no device, pure filesystem — so the data a run generates becomes training data.

Labeling heuristics (v0; deliberately conservative — the approve/defect review
loop and more runs refine them over time):

* VERIFY positives come only from PASSED runs: the final step's after-image (or
  final screenshot) *is* the goal state.
* VERIFY negatives: the first step's before-image of a passed run (the initial
  state, goal not yet met). Runs that started already-satisfied (a single step)
  are skipped for negatives to avoid mislabeling a dirty start.
* GROUND positives: each concrete tap in a passed run — (before-image,
  instruction, target, point) — since those taps demonstrably advanced the flow.

Only PASSED runs are exported by default: their labels are trustworthy. Failed /
inconclusive runs are skipped (their "correct" decision is unknown).
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import Example, GROUND, VERIFY

_TAP_LIKE = {"tap", "double_tap", "long_press", "input_text"}


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _after_image(run_dir: Path, step_number: int) -> str:
    candidate = run_dir / f"step-{step_number:02d}-after.png"
    return str(candidate) if candidate.is_file() else ""


def _before_image(run_dir: Path, step: dict) -> str:
    shot = step.get("screenshot", "")
    path = run_dir / shot if shot else None
    return str(path) if path and path.is_file() else ""


def export_run(run_dir: Path) -> list[Example]:
    """Export labeled examples from one run directory (empty list if unusable)."""
    run_dir = Path(run_dir)
    result = _load_json(run_dir / "result.json")
    if result.get("outcome") != "pass":
        return []  # only trust labels from passed runs
    goal = str(result.get("goal", ""))
    run_id = run_dir.name
    steps = [s for s in result.get("steps", []) if isinstance(s, dict)]
    examples: list[Example] = []

    # VERIFY positive: the goal state at the end of a passed run.
    positive_image = ""
    if steps:
        last = steps[-1]
        positive_image = _after_image(run_dir, int(last.get("number", len(steps)))) or _before_image(run_dir, last)
    if positive_image:
        examples.append(Example(
            task=VERIFY, image=positive_image, goal=goal, run_id=run_id,
            source=str(run_dir), satisfied=True,
            grounded_by=str(steps[-1].get("grounded_by", "")) if steps else "",
        ))
    # VERIFY negative: initial state, only when the run actually did work.
    if len(steps) > 1:
        first_before = _before_image(run_dir, steps[0])
        if first_before and first_before != positive_image:
            examples.append(Example(
                task=VERIFY, image=first_before, goal=goal, run_id=run_id,
                source=str(run_dir), satisfied=False,
            ))

    # GROUND positives: each concrete tap that advanced the flow.
    for step in steps:
        action = step.get("action") or {}
        if action.get("type") not in _TAP_LIKE:
            continue
        image = _before_image(run_dir, step)
        if not image:
            continue
        x, y = action.get("x"), action.get("y")
        point = (float(x), float(y)) if isinstance(x, (int, float)) and isinstance(y, (int, float)) else None
        examples.append(Example(
            task=GROUND, image=image, goal=goal, run_id=run_id, source=str(run_dir),
            grounded_by=str(step.get("grounded_by", "")),
            instruction=str(action.get("reason", "")) or goal,
            target=str(action.get("target", "")),
            point=point,
        ))
    return examples


def export_dataset(runs_root: Path) -> list[Example]:
    """Export examples from every run directory under ``runs_root``."""
    root = Path(runs_root)
    if not root.is_dir():
        return []
    examples: list[Example] = []
    for run_dir in sorted(root.iterdir()):
        if run_dir.is_dir() and (run_dir / "result.json").is_file():
            examples.extend(export_run(run_dir))
    return examples


def write_jsonl(examples: list[Example], out_path: Path) -> int:
    """Write examples to a JSONL file; return the count written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(e.to_dict(), ensure_ascii=False) for e in examples) + "\n"
        if examples else "",
        encoding="utf-8",
    )
    return len(examples)


def read_jsonl(path: Path) -> list[Example]:
    """Read examples back from a JSONL file."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Example.from_dict(json.loads(line)) for line in lines if line.strip()]
