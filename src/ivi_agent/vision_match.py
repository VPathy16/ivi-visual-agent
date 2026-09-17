"""OpenCV fast-path: ground a keyword/semantic-matched manual icon directly.

The vision-model planner/grounder is the slow part of a step. When the RAG
retriever has already surfaced a manual **icon** for the current subgoal (a
keyword or semantic match), we often don't need the model to *find* that icon on
screen — a cheap template match against the icon crop can locate it in
milliseconds. When that match is confident, the agent taps it directly and skips
the model call for that step, falling back to the model whenever the match is
weak, ambiguous, or absent.

Everything here is optional and guarded: without the ``[cv]`` extra
(``opencv-python-headless``) the helpers report unavailable and the agent uses
its normal model path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .types import Action


def available() -> bool:
    """True when OpenCV (and numpy) can be imported."""
    import importlib.util

    return (
        importlib.util.find_spec("cv2") is not None
        and importlib.util.find_spec("numpy") is not None
    )


def _decode_gray(png: bytes) -> Any | None:
    """Decode PNG bytes to a grayscale ndarray, or None on failure."""
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    array = np.frombuffer(png, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    return image


# Template scales tried against the live screen. A manual icon crop rarely
# matches the on-screen rendering size, so we sweep a range and keep the best.
_DEFAULT_SCALES = tuple(round(0.5 + 0.1 * step, 2) for step in range(11))  # 0.5 .. 1.5


def locate_template(
    screen_png: bytes,
    template_png: bytes,
    threshold: float,
    scales: tuple[float, ...] = _DEFAULT_SCALES,
) -> tuple[float, float, float, tuple[int, int, int, int]] | None:
    """Multi-scale template match of an icon crop against the live screen.

    Returns ``(score, cx_norm, cy_norm, (left, top, right, bottom))`` for the
    best match, where the center is normalized to ``[0, 1]``; or ``None`` when no
    scale clears ``threshold`` (or OpenCV is unavailable). Uses grayscale
    normalized cross-correlation (``TM_CCOEFF_NORMED``).
    """
    if not available():
        return None
    import cv2  # type: ignore

    screen = _decode_gray(screen_png)
    template = _decode_gray(template_png)
    if screen is None or template is None:
        return None
    height, width = screen.shape[:2]
    base_h, base_w = template.shape[:2]
    if base_h < 4 or base_w < 4:
        return None

    best: tuple[float, float, float, tuple[int, int, int, int]] | None = None
    for scale in scales:
        tw, th = int(round(base_w * scale)), int(round(base_h * scale))
        if tw < 8 or th < 8 or th > height or tw > width:
            continue
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(template, (tw, th), interpolation=interpolation)
        result = cv2.matchTemplate(screen, resized, cv2.TM_CCOEFF_NORMED)
        _, max_value, _, max_location = cv2.minMaxLoc(result)
        if best is None or max_value > best[0]:
            left, top = int(max_location[0]), int(max_location[1])
            right, bottom = left + tw, top + th
            cx = (left + tw / 2) / width
            cy = (top + th / 2) / height
            best = (float(max_value), cx, cy, (left, top, right, bottom))

    if best is None or best[0] < threshold:
        return None
    return best


def screen_similarity(screen_png: bytes, reference_png: bytes) -> float:
    """Rough global similarity of two screens via ORB feature matching.

    Returns the fraction of reference keypoints with a good (Lowe-ratio) match on
    the live screen, in ``[0, 1]``; ``0.0`` when OpenCV is unavailable or either
    image has too few features. Useful as a corroborating "am I on this screen"
    signal, not a precise metric.
    """
    if not available():
        return 0.0
    import cv2  # type: ignore

    screen = _decode_gray(screen_png)
    reference = _decode_gray(reference_png)
    if screen is None or reference is None:
        return 0.0
    orb = cv2.ORB_create(nfeatures=600)
    kp_screen, des_screen = orb.detectAndCompute(screen, None)
    kp_ref, des_ref = orb.detectAndCompute(reference, None)
    if des_screen is None or des_ref is None or len(kp_ref) < 10 or len(kp_screen) < 10:
        return 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(des_ref, des_screen, k=2)
    good = 0
    for pair in pairs:
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            good += 1
    return good / max(len(kp_ref), 1)


def cv_ground_from_knowledge(
    screen_png: bytes,
    knowledge: dict[str, Any],
    config: Any,
) -> Action | None:
    """Return a direct tap Action for a retrieved icon located on screen, else None.

    Only icon chunks that the retriever ranked at or above
    ``config.cv_min_retrieval_score`` are considered — that is the "the keyword
    matched this icon" gate. Among those (best-ranked first), the first whose crop
    is found on the live screen at or above ``config.cv_match_threshold`` yields a
    tap at the matched center. The tap's confidence is the match score, so it must
    also clear the normal action-confidence policy.
    """
    if not available():
        return None
    icons = [
        chunk
        for chunk in knowledge.get("chunks", [])
        if chunk.get("kind") == "icon"
        and isinstance(chunk.get("image"), str)
        and float(chunk.get("score", 0.0)) >= config.cv_min_retrieval_score
    ]
    icons.sort(key=lambda chunk: -float(chunk.get("score", 0.0)))
    for icon in icons:
        path = Path(str(icon["image"]))
        if not path.is_file():
            continue
        match = locate_template(
            screen_png, path.read_bytes(), config.cv_match_threshold
        )
        if match is None:
            continue
        score, cx, cy, _box = match
        name = str(icon.get("name") or icon.get("id") or "icon")
        return Action(
            type="tap",
            confidence=min(score, 1.0),
            reason=f"cv2 template match for manual icon '{name}' (score {score:.2f})",
            target=name,
            x=cx,
            y=cy,
        )
    return None
