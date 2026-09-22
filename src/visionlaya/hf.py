"""Import Hugging Face GUI-grounding datasets into VisionLaya's Example schema.

Bootstraps the GROUND head with public Android data so you don't have to collect
hundreds of IVI runs first. Each row becomes ``Example(task="ground", image,
instruction, target, point)``; images are saved next to the output JSONL so the
trainer can read them.

The row→fields mappers are tolerant to column-name variants across dataset
mirrors and are pure (unit-testable without downloading anything). ``import_hf``
does the actual download via the ``datasets`` library (optional dep) and image
saving. Adjust the mapper key lists if your chosen mirror uses other names.

NOTE on scope (this is a product): prefer permissively-licensed datasets. Many
GUI datasets are research-only — check each dataset's license before shipping a
model trained on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .schema import Example, GROUND

# ---- pure row mappers (testable) -----------------------------------------
# Each returns (instruction, target, point_or_None, goal) or None to skip.


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _norm_point(x: Any, y: Any, width: Any = None, height: Any = None) -> tuple[float, float] | None:
    try:
        fx, fy = float(x), float(y)
    except (TypeError, ValueError):
        return None
    if width and height:  # pixel coords -> normalize
        try:
            fx, fy = fx / float(width), fy / float(height)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    if 0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0:
        return (round(fx, 4), round(fy, 4))
    return None


def map_androidcontrol(row: dict[str, Any]) -> dict[str, Any] | None:
    """AndroidControl-style row: per-step instruction + a tap action."""
    instruction = _first(row, "instruction", "step_instruction", "low_instruction")
    goal = _first(row, "goal", "task", "high_instruction") or ""
    if not instruction and not goal:
        return None
    action = _first(row, "action", "normalized_action") or {}
    if isinstance(action, str):
        return None  # unparsed action string; skip rather than guess
    x = _first(action, "x", "touch_x") if isinstance(action, dict) else None
    y = _first(action, "y", "touch_y") if isinstance(action, dict) else None
    point = _norm_point(x, y, row.get("width"), row.get("height"))
    if point is None:
        return None  # only keep tap-localizable steps for GROUND
    return {"instruction": str(instruction or goal), "target": "", "point": point, "goal": str(goal)}


def map_screenspot(row: dict[str, Any]) -> dict[str, Any] | None:
    """ScreenSpot-style row: instruction + a target element bbox."""
    instruction = _first(row, "instruction", "prompt")
    if not instruction:
        return None
    bbox = _first(row, "bbox", "bounding_box", "box")
    point: tuple[float, float] | None = None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        point = _norm_point(cx, cy, row.get("width"), row.get("height")) or _norm_point(cx, cy)
    if point is None:
        return None
    return {"instruction": str(instruction), "target": str(_first(row, "data_type", "instruction") or ""),
            "point": point, "goal": str(instruction)}


MAPPERS: dict[str, Callable[[dict[str, Any]], dict[str, Any] | None]] = {
    "androidcontrol": map_androidcontrol,
    "screenspot": map_screenspot,
}


def _image_column(row: dict[str, Any]) -> Any:
    return _first(row, "image", "screenshot", "img")


def import_hf(
    dataset: str,
    hf_path: str,
    out_dir: Path,
    split: str = "train",
    limit: int | None = None,
) -> list[Example]:
    """Download ``hf_path`` and map rows into GROUND Examples, saving images.

    ``dataset`` selects the row mapper (see MAPPERS). Requires the ``[hf]`` extra.
    """
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "HF import needs the datasets library:\n  pip install -e '.[hf]'"
        ) from exc
    mapper = MAPPERS.get(dataset)
    if mapper is None:
        raise SystemExit(f"Unknown dataset {dataset!r}. Choose from {', '.join(MAPPERS)}.")
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(hf_path, split=split, streaming=True)
    examples: list[Example] = []
    for index, row in enumerate(data):
        if limit is not None and len(examples) >= limit:
            break
        fields = mapper(dict(row))
        if fields is None:
            continue
        image = _image_column(dict(row))
        if image is None:
            continue
        image_path = images_dir / f"{dataset}-{index:07d}.png"
        try:
            image.save(image_path)  # datasets yields a PIL image for image columns
        except Exception:  # noqa: BLE001
            continue
        examples.append(Example(
            task=GROUND, image=str(image_path), goal=fields["goal"],
            run_id=f"hf:{dataset}", source=hf_path,
            grounded_by="hf", instruction=fields["instruction"],
            target=fields["target"], point=fields["point"],
        ))
    return examples
